"""Policy/value network.

The engine sends four maps and a vector of exact numbers per agent per decision:

* local: the player view, 64x64 cells of 2 m around the cycle (40 ahead), rotated so the agent
  always drives "up"; 8 bit-planes packed in one byte per cell;
* close: the same view at 1 m, a 64 m window, for the close fight;
* global: the whole arena at 64x64, same rotation, the minimap;
* territory: arena-aligned; bit 0 "I reach this cell first", bit 1 "an enemy does", bits 2..7 my
  BFS distance, the partition a good player holds in their head when glancing at the minimap;
* scalars: ray distances, speed, rubber, nearest enemies.

v2 (`PolicyNet`): one residual tower per map; the previous frame's maps are stacked under the
current ones so motion is visible. v1 (`PolicyNetV1`) is kept so old checkpoints and the game's
in-engine brain stay loadable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .protocol import GRID, MAP_PLANES, N_ACTIONS, N_GLOBAL_PLANES, N_LOCAL_PLANES, N_MAPS, N_SCALARS

N_AUX = 2  # territory share in 2 s, dead within 2 s
AUX_HORIZON = 40  # decisions (2 s at 20 a second)

def _masked(dtype) -> float:
    """The logit of a forbidden move: very negative, but representable in half precision too."""
    return -1e8 if dtype in (torch.float32, torch.float64) else -3e4


_BITS = torch.tensor([[(v >> b) & 1 for b in range(8)] for v in range(256)], dtype=torch.float32)


class Unpack(nn.Module):
    """(B, H, W) uint8 bit-planes -> (B, n_planes, H, W) float32 in {0, 1} via a lookup table."""

    def __init__(self, n_planes: int):
        super().__init__()
        self.register_buffer("table", _BITS[:, :n_planes].clone(), persistent=False)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        # contiguous matters: convolution backward on a permuted view is ~3x slower on MPS
        return self.table[packed.long()].permute(0, 3, 1, 2).contiguous()


# ------------------------------------------------------------------------------------------ v1

class PatchTower(nn.Module):
    def __init__(self, in_ch: int, width: int, depth: int, out_dim: int):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 4, stride=4), nn.ReLU(inplace=True)]  # 16x16
        for _ in range(depth):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8x8
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 4x4
        ]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(2 * width * (GRID // 16) ** 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.net(x).flatten(1)))


class PolicyNetV1(nn.Module):
    """The first network: two maps, plain conv towers. Kept for old checkpoints."""

    version = 1
    n_maps = 2
    stack_prev = False

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.unpack_local = Unpack(N_LOCAL_PLANES)
        self.unpack_global = Unpack(N_GLOBAL_PLANES)
        self.local = PatchTower(N_LOCAL_PLANES, 64, 2, 256)
        self.globl = PatchTower(N_GLOBAL_PLANES, 32, 1, 256)
        self.scalars = nn.Sequential(nn.Linear(N_SCALARS, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(768, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def act(self, maps, feats, greedy: bool = False):
        logits, value = self(maps, feats[:, :N_SCALARS], feats[:, N_SCALARS:] > 0.5)
        return _pick(logits, value, greedy)

    def forward(self, maps, scalars, mask=None):
        local, globl = maps[:, 0], maps[:, 2] if maps.shape[1] >= 3 else maps[:, 1]
        h = torch.cat([self.local(self.unpack_local(local)), self.globl(self.unpack_global(globl)), self.scalars(scalars)], 1)
        h = self.trunk(h)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, _masked(logits.dtype))
        return logits, self.v(h).squeeze(1)


# ------------------------------------------------------------------------------------------ v2

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c0 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.c0(x))
        return torch.relu(x + self.c1(y))


class ResTower(nn.Module):
    """stem (4x4 patches) -> residual blocks at 16x16 -> two strided convs -> linear.

    Exported layer names: towers.M.convs.0 (stem), towers.M.res.B.0/.1, towers.M.convs.1/.2
    (strided), towers.M.fc. The C++ side (gNeuralNet.cpp) replays exactly this order."""

    def __init__(self, in_ch: int, width: int, blocks: int, out_dim: int):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 4, stride=4)  # 16x16
        self.res = nn.ModuleList([ResBlock(width) for _ in range(blocks)])
        self.down0 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)  # 8x8
        self.down1 = nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1)  # 4x4
        self.fc = nn.Linear(2 * width * (GRID // 16) ** 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        for b in self.res:
            x = b(x)
        x = torch.relu(self.down0(x))
        x = torch.relu(self.down1(x))
        return torch.relu(self.fc(x.flatten(1)))

    def export_layers(self, prefix: str):
        """(name, module) in forward order, the names the game's loader expects."""
        out = [(f"{prefix}.convs.0", self.stem)]
        for i, b in enumerate(self.res):
            out += [(f"{prefix}.res.{i}.0", b.c0), (f"{prefix}.res.{i}.1", b.c1)]
        out += [(f"{prefix}.convs.1", self.down0), (f"{prefix}.convs.2", self.down1), (f"{prefix}.fc", self.fc)]
        return out


# The four maps come in two frames of reference, and maps that share one are fed to one tower, so
# their planes meet in the very first filter ("a wall that borders my region", "a gap in the close
# view that the 2 m view shows leads somewhere"): the player tower sees the 2 m and 1 m views
# (both centred on the cycle, rotated with it), the arena tower sees the minimap and territory
# (both arena-aligned, cell for cell). Each gets the previous frame's planes stacked underneath.
TOWER_MAPS = ((0, 1), (2, 3))  # indices into the N_MAPS maps
TOWER_SPECS = (  # per tower: (width, residual blocks, output dim)
    (80, 3, 320),  # player: local + close
    (64, 3, 256),  # arena: global + territory
)
TOWER_NAMES = ("player", "arena")


class PolicyNet(nn.Module):
    """v2: two residual towers (player view, arena view), each fed its maps with the previous frame stacked."""

    version = 2
    n_maps = N_MAPS
    stack_prev = True

    def __init__(self, hidden: int = 768):
        super().__init__()
        self.unpack = nn.ModuleList([Unpack(p) for p in MAP_PLANES])
        self.towers = nn.ModuleList([
            ResTower(2 * sum(MAP_PLANES[m] for m in maps), w, b, d) for maps, (w, b, d) in zip(TOWER_MAPS, TOWER_SPECS)])
        fused = sum(d for _, _, d in TOWER_SPECS) + 256
        self.scalars = nn.Sequential(nn.Linear(N_SCALARS, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(fused, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        # auxiliary predictions, trained by the teacher and ignored at play time: my share of the
        # territory two seconds from now, and whether I am dead within two seconds. They cost
        # nothing in the game and sharpen the features the move and value heads read.
        self.aux = nn.Linear(hidden, N_AUX)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.orthogonal_(self.aux.weight, gain=0.1)

    def act(self, maps, feats, greedy: bool = False):
        """maps: (B, 2*N_MAPS, G, G) uint8, current maps then the previous frame's; feats: (B, N_SCALARS +
        N_ACTIONS) float32 whose last N_ACTIONS columns are the action mask. Returns (action, logp, value)."""
        logits, value = self(maps, feats[:, :N_SCALARS], feats[:, N_SCALARS:] > 0.5)
        return _pick(logits, value, greedy)

    def tower_input(self, maps, t: int) -> torch.Tensor:
        """The planes tower T sees: its maps' current planes, then the same maps' previous planes."""
        stacked = maps.shape[1] >= 2 * N_MAPS
        cur = [self.unpack[m](maps[:, m]) for m in TOWER_MAPS[t]]
        prev = [self.unpack[m](maps[:, N_MAPS + m]) for m in TOWER_MAPS[t]] if stacked else cur
        return torch.cat(cur + prev, 1)

    def trunk_out(self, maps, scalars) -> torch.Tensor:
        parts = [self.towers[t](self.tower_input(maps, t)) for t in range(len(TOWER_MAPS))]
        parts.append(self.scalars(scalars))
        return self.trunk(torch.cat(parts, 1))

    def forward(self, maps, scalars, mask=None):
        h = self.trunk_out(maps, scalars)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, _masked(logits.dtype))
        return logits, self.v(h).squeeze(1)

    def heads(self, maps, scalars, mask=None):
        """(logits, value, aux logits): the teacher trains all three."""
        h = self.trunk_out(maps, scalars)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, _masked(logits.dtype))
        return logits, self.v(h).squeeze(1), self.aux(h)


def _pick(logits, value, greedy):
    if greedy:
        a = logits.argmax(1)
    else:
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-9, 1.0)))
        a = (logits + gumbel).argmax(1)
    logp = torch.log_softmax(logits, 1).gather(1, a[:, None]).squeeze(1)
    return a, logp, value


def make_net(version: int = 2) -> nn.Module:
    return PolicyNet() if version == 2 else PolicyNetV1()


def net_version(state_dict) -> int:
    """Which network a checkpoint's weights belong to."""
    return 2 if any(k.startswith("towers.") for k in state_dict) else 1
