"""Policy/value network.

The engine sends three things per agent per decision:

* a local egocentric map (64x64 cells of 2 m, packed as 8 bit-planes in one byte per cell),
  rotated so the agent always drives "up" the image,
* a global map of the whole arena (64x64, same packing, same rotation),
* a vector of exact scalar features (ray distances, speed, rubber, nearest enemies...).

Each map is cut into 4x4 patches (a strided conv, which keeps every bit of the patch) and then
processed by a small conv tower; the scalars go through an MLP; the three embeddings are fused
and fed to a policy head (one logit per action) and a value head. Convolutions at full 64x64
resolution are very slow on Apple GPUs, the patch stem makes training ~15x faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .protocol import GRID, N_ACTIONS, N_GLOBAL_PLANES, N_LOCAL_PLANES, N_SCALARS

_BITS = torch.tensor([[(v >> b) & 1 for b in range(8)] for v in range(256)], dtype=torch.float32)


class Unpack(nn.Module):
    """(B, H, W) uint8 bit-planes -> (B, n_planes, H, W) float32 in {0, 1} via a lookup table."""

    def __init__(self, n_planes: int):
        super().__init__()
        self.register_buffer("table", _BITS[:, :n_planes].clone(), persistent=False)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        # contiguous matters: convolution backward on a permuted view is ~3x slower on MPS
        return self.table[packed.long()].permute(0, 3, 1, 2).contiguous()


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


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.unpack_local = Unpack(N_LOCAL_PLANES)
        self.unpack_global = Unpack(N_GLOBAL_PLANES)
        self.local = PatchTower(N_LOCAL_PLANES, 64, 2, 256)
        self.globl = PatchTower(N_GLOBAL_PLANES, 32, 1, 256)
        self.scalars = nn.Sequential(
            nn.Linear(N_SCALARS, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(768, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def act(self, maps, feats, greedy: bool = False):
        """Sample actions. maps: (B, 2, G, G) uint8, feats: (B, N_SCALARS + N_ACTIONS) float32 whose
        last N_ACTIONS columns are the action mask. Returns (action, logp, value)."""
        logits, value = self(maps[:, 0], maps[:, 1], feats[:, :N_SCALARS], feats[:, N_SCALARS:] > 0.5)
        if greedy:
            a = logits.argmax(1)
        else:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-9, 1.0)))
            a = (logits + gumbel).argmax(1)
        logp = torch.log_softmax(logits, 1).gather(1, a[:, None]).squeeze(1)
        return a, logp, value

    def forward(self, local_packed, global_packed, scalars, mask=None):
        h = torch.cat([
            self.local(self.unpack_local(local_packed)),
            self.globl(self.unpack_global(global_packed)),
            self.scalars(scalars),
        ], dim=1)
        h = self.trunk(h)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e8)
        return logits, self.v(h).squeeze(1)
