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

import math

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

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The 16x16 map after the residual blocks (what the ownership head reads)."""
        x = torch.relu(self.stem(x))
        for b in self.res:
            x = b(x)
        return x

    def head(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.down0(x))
        x = torch.relu(self.down1(x))
        return torch.relu(self.fc(x.flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

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
    specs = TOWER_SPECS

    def make_tower(self, t: int, in_ch: int, width: int, blocks: int, out_dim: int) -> nn.Module:
        return ResTower(in_ch, width, blocks, out_dim)

    def __init__(self, hidden: int = 768):
        super().__init__()
        self.unpack = nn.ModuleList([Unpack(p) for p in MAP_PLANES])
        self.towers = nn.ModuleList([
            self.make_tower(t, 2 * sum(MAP_PLANES[m] for m in maps), w, b, d)
            for t, (maps, (w, b, d)) in enumerate(zip(TOWER_MAPS, self.specs))])
        fused = sum(d for _, _, d in self.specs) + 256
        self.scalars = nn.Sequential(nn.Linear(N_SCALARS, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(fused, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        # auxiliary predictions, trained by the teacher and ignored at play time: my share of the
        # territory two seconds from now, and whether I am dead within two seconds. They cost
        # nothing in the game and sharpen the features the move and value heads read.
        self.aux = nn.Linear(hidden, N_AUX)
        # ownership: who reaches each minimap cell first two seconds from now (nobody, me, an enemy),
        # read off the arena tower's 16x16 map and trained by the teacher; the game never runs it.
        self.own = nn.Conv2d(self.specs[1][0], 3, 1)
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

    def trunk_and_arena(self, maps, scalars):
        """The trunk output and the arena tower's 16x16 features, for the training heads."""
        arena = self.towers[1].features(self.tower_input(maps, 1))
        parts = [self.towers[0](self.tower_input(maps, 0)), self.towers[1].head(arena), self.scalars(scalars)]
        return self.trunk(torch.cat(parts, 1)), arena

    def forward(self, maps, scalars, mask=None):
        h = self.trunk_out(maps, scalars)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, _masked(logits.dtype))
        return logits, self.v(h).squeeze(1)

    def heads(self, maps, scalars, mask=None):
        """(logits, value, aux logits, ownership logits (B, 3, G, G)): the teacher trains all four."""
        h, arena = self.trunk_and_arena(maps, scalars)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, _masked(logits.dtype))
        own = nn.functional.interpolate(self.own(arena), size=(GRID, GRID), mode="bilinear", align_corners=False)
        return logits, self.v(h).squeeze(1), self.aux(h), own


# ------------------------------------------------------------------------------------------ v3

class GPoolResBlock(nn.Module):
    """A residual block that also sees the whole map: the first conv's first G channels are pooled
    over the board (mean and max) and fed back as a per-channel bias before the second conv, so
    every layer knows the global counts while reading local shapes (KataGo's global pooling)."""

    def __init__(self, ch: int, g: int = 32):
        super().__init__()
        self.g = g
        self.c0 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gp = nn.Linear(2 * g, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.c0(x))
        pooled = torch.cat([y[:, :self.g].mean((2, 3)), y[:, :self.g].amax((2, 3))], 1)
        y = y + self.gp(pooled)[:, :, None, None]
        return torch.relu(x + self.c1(y))


class AttnLayer(nn.Module):
    """One pre-norm transformer layer over the map's cells, so any cell can look at any other."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.proj = nn.Linear(ch, ch)
        self.ln2 = nn.LayerNorm(ch)
        self.fc1 = nn.Linear(ch, 2 * ch)
        self.fc2 = nn.Linear(2 * ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        B, N, C = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(C // h), -1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, N, C))
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x))))


class ResTowerV3(nn.Module):
    """stem (4x4 patches) -> residual blocks at 16x16, every third with global pooling -> optional
    attention layers over the 256 cells -> two strided convs -> linear.

    Exported layer names: towers.M.stem, towers.M.res.B.c0/.c1 (and .gp for pooling blocks),
    towers.M.pos, towers.M.attn.L.ln1/.qkv/.proj/.ln2/.fc1/.fc2, towers.M.down0/.down1, towers.M.fc.
    The C++ side (gNeuralNet.cpp) replays exactly this order."""

    def __init__(self, in_ch: int, width: int, blocks: int, out_dim: int, attn_layers: int = 0, gpool_every: int = 3,
                 gpool_ch: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, 4, stride=4)  # 16x16
        self.res = nn.ModuleList([GPoolResBlock(width, gpool_ch) if (i + 1) % gpool_every == 0 else ResBlock(width)
                                  for i in range(blocks)])
        self.cells = (GRID // 4) ** 2
        if attn_layers:
            self.pos = nn.Parameter(torch.zeros(1, self.cells, width))
            self.attn = nn.ModuleList([AttnLayer(width) for _ in range(attn_layers)])
        else:
            self.attn = nn.ModuleList()
        self.down0 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)  # 8x8
        self.down1 = nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1)  # 4x4
        self.fc = nn.Linear(2 * width * (GRID // 16) ** 2, out_dim)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        for b in self.res:
            x = b(x)
        if len(self.attn):
            B, C, H, W = x.shape
            t = x.flatten(2).transpose(1, 2) + self.pos
            for layer in self.attn:
                t = layer(t)
            x = t.transpose(1, 2).reshape(B, C, H, W)
        return x

    def head(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.down0(x))
        x = torch.relu(self.down1(x))
        return torch.relu(self.fc(x.flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def export_layers(self, prefix: str):
        """(name, module or parameter) in forward order, the names the game's loader expects."""
        out = [(f"{prefix}.stem", self.stem)]
        for i, b in enumerate(self.res):
            out.append((f"{prefix}.res.{i}.c0", b.c0))
            if isinstance(b, GPoolResBlock):
                out.append((f"{prefix}.res.{i}.gp", b.gp))
            out.append((f"{prefix}.res.{i}.c1", b.c1))
        if len(self.attn):
            out.append((f"{prefix}.pos", self.pos))
            for i, a in enumerate(self.attn):
                out += [(f"{prefix}.attn.{i}.ln1", a.ln1), (f"{prefix}.attn.{i}.qkv", a.qkv), (f"{prefix}.attn.{i}.proj", a.proj),
                        (f"{prefix}.attn.{i}.ln2", a.ln2), (f"{prefix}.attn.{i}.fc1", a.fc1), (f"{prefix}.attn.{i}.fc2", a.fc2)]
        out += [(f"{prefix}.down0", self.down0), (f"{prefix}.down1", self.down1), (f"{prefix}.fc", self.fc)]
        return out


TOWER_SPECS_V3 = (  # per tower: (width, residual blocks, output dim)
    (96, 5, 320),  # player: local + close
    (128, 5, 256),  # arena: global + territory
)
TOWER_ATTN_V3 = (0, 2)  # attention layers per tower: the arena view reasons about connections


class PolicyNetV3(PolicyNet):
    """v3: the v2 layout with deeper towers, global pooling in every third block, and attention over the
    arena view's cells. Same inputs, same heads, so the teacher and the game treat it like v2."""

    version = 3
    specs = TOWER_SPECS_V3

    def make_tower(self, t: int, in_ch: int, width: int, blocks: int, out_dim: int) -> nn.Module:
        return ResTowerV3(in_ch, width, blocks, out_dim, attn_layers=TOWER_ATTN_V3[t])

    def __init__(self, hidden: int = 768):
        super().__init__(hidden)
        self.register_buffer("net_version", torch.tensor(3))


def _pick(logits, value, greedy):
    if greedy:
        a = logits.argmax(1)
    else:
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-9, 1.0)))
        a = (logits + gumbel).argmax(1)
    logp = torch.log_softmax(logits, 1).gather(1, a[:, None]).squeeze(1)
    return a, logp, value


def make_net(version: int = 2) -> nn.Module:
    if version == 3:
        return PolicyNetV3()
    return PolicyNet() if version == 2 else PolicyNetV1()


def net_version(state_dict) -> int:
    """Which network a checkpoint's weights belong to."""
    if "net_version" in state_dict:
        return int(state_dict["net_version"])
    return 2 if any(k.startswith("towers.") for k in state_dict) else 1
