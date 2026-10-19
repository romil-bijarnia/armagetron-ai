"""Policy/value network.

The engine sends four maps and a vector of exact numbers per agent per decision:

* local: the player view, 64x64 cells of 2 m around the cycle (40 ahead), rotated so the agent
  always drives "up"; 8 bit-planes packed in one byte per cell;
* close: the same view at 1 m, a 64 m window, for the close fight;
* global: the whole arena at 64x64, same rotation, the minimap;
* territory: arena-aligned; bit 0 "I reach this cell first", bit 1 "an enemy does", bits 2..7 my
  BFS distance, the partition a good player holds in their head when glancing at the minimap;
* scalars: ray distances, speed, rubber, nearest enemies.

The V4 fighter separates Nearby and Close, processes local geometry at 32x32, and conditions
local features on the arena. A shared masked encoder handles the three nearest enemies. The
heads predict six steering/brake actions, bounded outcome value, attack, territory and danger.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .protocol import (GRID, MAP_PLANES, N_MAPS, N_SCALARS, OBSERVATION_VERSION,
                       OWN_SCALARS, ENEMY_EXTRA_OFFSET, ENEMY_EXTRA_FEATURES, ENEMY_FEATURES)

MEMORY_VERSION = 1
MEMORY_HIDDEN = 256
MEMORY_CONTEXT = 7
DECISION_INTERVAL = 0.05


def recurrent_context(previous_actions, elapsed_seconds) -> torch.Tensor:
    """Previous *executed* action (or -1 at birth) and elapsed game seconds.

    Works with batch or batch/time tensors. The six action channels are one-hot;
    the last channel measures elapsed time in nominal 50 ms decisions.
    """
    actions = torch.as_tensor(previous_actions)
    elapsed = torch.as_tensor(elapsed_seconds, device=actions.device)
    if actions.shape != elapsed.shape:
        raise ValueError("previous actions and elapsed seconds must have identical shapes")
    if (actions.dtype == torch.bool or actions.is_floating_point()
            or actions.is_complex() or ((actions < -1) | (actions >= 6)).any()):
        raise ValueError("previous executed actions must be integers from -1 to 5")
    if elapsed.is_complex() or not torch.isfinite(elapsed).all() or (elapsed < 0).any():
        raise ValueError("elapsed game seconds must be finite and nonnegative")
    dtype = elapsed.dtype if elapsed.is_floating_point() else torch.float32
    elapsed = elapsed.to(dtype)
    one_hot = nn.functional.one_hot(actions.long().clamp_min(0), 6).to(dtype)
    one_hot = one_hot * (actions >= 0)[..., None]
    return torch.cat([one_hot, (elapsed / DECISION_INTERVAL)[..., None]], -1)


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


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c0 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.c0(x))
        return torch.relu(x + self.c1(y))


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


class ResidualTower(nn.Module):
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


class ProgressiveTower(ResidualTower):
    """Read local geometry at 32x32 before reducing it to the common 16x16 feature grid.

    Nearby and Close have separate instances: their cells represent different physical scales.
    The arena-conditioned features are passed to ``head`` after this tower's residual blocks.
    """

    def __init__(self, in_ch: int, width: int = 64, blocks: int = 3, out_dim: int = 192):
        super().__init__(in_ch, width, blocks, out_dim)
        self.stem = nn.Conv2d(in_ch, 48, 3, stride=2, padding=1)
        self.pre = ResBlock(48)
        self.reduce = nn.Conv2d(48, width, 3, stride=2, padding=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(torch.relu(self.stem(x)))
        x = torch.relu(self.reduce(x))
        for block in self.res:
            x = block(x)
        return x

    def export_layers(self, prefix: str):
        layers = super().export_layers(prefix)
        return layers[:1] + [(f"{prefix}.pre.c0", self.pre.c0), (f"{prefix}.pre.c1", self.pre.c1),
                             (f"{prefix}.reduce", self.reduce)] + layers[1:]


class PolicyNet(nn.Module):
    """Combat architecture: independent scales, arena-conditioned tactics and opponent selection.

    Six joint steering/brake actions cover straight, left and right with optional braking. The
    third auxiliary target is a credited kill within two seconds while surviving that horizon. It
    teaches offensive features without paying the policy for a suicidal trade. Value remains
    grounded in discounted round outcome and is bounded to [-1, 1].
    """

    version = 4
    observation_schema = OBSERVATION_VERSION
    n_maps = N_MAPS
    stack_prev = True
    n_actions = 6
    n_aux = 3
    specs = ((64, 3, 192), (64, 3, 192), (128, 5, 256))
    tower_maps = ((0,), (1,), (2, 3))
    tower_names = ("nearby", "close", "arena")
    memory_schema = MEMORY_VERSION
    state_size = MEMORY_HIDDEN

    def __init__(self, hidden: int = 768):
        super().__init__()
        if hidden != 768:
            raise ValueError("The V4 recurrent trunk has exactly 768 features")
        self.unpack = nn.ModuleList([Unpack(p) for p in MAP_PLANES])
        self.towers = nn.ModuleList([
            ProgressiveTower(16, 64, 3, 192),
            ProgressiveTower(16, 64, 3, 192),
            ResidualTower(32, 128, 5, 256, attn_layers=2),
        ])
        self.conditioning = nn.ModuleList([nn.Linear(256, 128), nn.Linear(256, 128)])
        self.scalars = nn.Sequential(nn.Linear(OWN_SCALARS, 256), nn.ReLU(inplace=True),
                                     nn.Linear(256, 256), nn.ReLU(inplace=True))
        # Same encoder for each of the three nearest enemies; presence gates absent tokens.
        self.opponents = nn.Sequential(nn.Linear(ENEMY_FEATURES, 64), nn.ReLU(inplace=True),
                                       nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.opponent_query = nn.Linear(896, 64)
        self.trunk = nn.Sequential(nn.Linear(960, hidden), nn.ReLU(inplace=True),
                                   nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.pi = nn.Linear(hidden, self.n_actions)
        self.v = nn.Linear(hidden, 1)
        self.aux = nn.Linear(hidden, self.n_aux)
        self.own = nn.Conv2d(128, 3, 1)
        self.register_buffer("net_version", torch.tensor(self.version))
        self.register_buffer("observation_version", torch.tensor(OBSERVATION_VERSION))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        for module in self.conditioning:
            nn.init.zeros_(module.weight)  # begin with identity conditioning
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.orthogonal_(self.aux.weight, gain=0.1)
        # New resource/timing connections start inactive; migration copies the learned
        # eight-field encoder exactly and uses the same zero initialization.
        with torch.no_grad():
            self.opponents[0].weight[:, 8:].zero_()
        # Append after every existing module so old optimizer parameter IDs remain stable.
        self.memory_gru = nn.GRU(hidden + MEMORY_CONTEXT, MEMORY_HIDDEN, batch_first=True)
        self.memory_projection = nn.Linear(MEMORY_HIDDEN, hidden)
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)
        self.register_buffer("memory_version", torch.tensor(MEMORY_VERSION))

    def act_step(self, maps, feats, state, greedy: bool = False, reset=None, context=None):
        """maps: (B, 2*N_MAPS, G, G) uint8, current maps then the previous frame's; feats: (B, N_SCALARS +
        N_ACTIONS) float32 whose last N_ACTIONS columns are the action mask. State is explicit;
        returns (action, logp, value, next_state), after consuming this observation."""
        logits, value, next_state = self.forward_step(maps, feats[:, :N_SCALARS],
                                                     feats[:, N_SCALARS:] > 0.5, state, reset, context)
        return (*_pick(logits, value, greedy), next_state)

    def act(self, maps, feats, greedy: bool = False):
        """Fresh-state frame diagnostic. Stateful controllers must use ``act_step``."""
        return self.act_step(maps, feats, self.initial_state(len(maps)), greedy=greedy)[:3]

    def initial_state(self, batch: int, device=None, dtype=None):
        """Explicit start-of-life state. Callers retain the returned state between decisions."""
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise ValueError("state batch size must be a positive integer")
        weight = self.memory_gru.weight_ih_l0
        return torch.zeros(batch, MEMORY_HIDDEN, device=weight.device if device is None else device,
                           dtype=weight.dtype if dtype is None else dtype)

    def _remember(self, features, state, reset=None, context=None):
        batch, steps, _ = features.shape
        dtype = self.memory_gru.weight_ih_l0.dtype
        if not isinstance(state, torch.Tensor) or state.shape != (batch, MEMORY_HIDDEN):
            raise ValueError("explicit recurrent state must have shape [batch, 256]")
        if state.device != features.device or not state.is_floating_point():
            raise ValueError("recurrent state must be floating and match the feature device")
        if reset is not None and (reset.shape != (batch, steps) or reset.dtype != torch.bool
                                  or reset.device != features.device):
            raise ValueError("reset must be a boolean [batch, time] tensor on the feature device")
        if context is None:
            context = features.new_zeros(batch, steps, MEMORY_CONTEXT, dtype=dtype)
        elif (context.shape != (batch, steps, MEMORY_CONTEXT) or not context.is_floating_point()
              or context.device != features.device):
            raise ValueError("memory context must be floating [batch, time, 7] on the feature device")
        else:
            context = context.to(dtype)
        if reset is not None:
            context = context.masked_fill(reset[..., None], 0)
        # MPS autocast does not enable nn.GRU's input/weight dtype exemption.
        # Keep recurrent gates in the module's native precision (FP32 learner,
        # FP16 actor), while the expensive visual trunk can still use BF16.
        inputs = torch.cat([features.to(dtype), context], -1)
        hidden = state.to(dtype).unsqueeze(0)
        with torch.autocast(features.device.type, enabled=False):
            if reset is None:
                output, hidden = self.memory_gru(inputs, hidden)
            else:
                # Preserve fused execution between life resets. Resets also cut
                # gradients from the preceding life without mutating caller state.
                boundaries = reset.any(0).nonzero().flatten().tolist()
                outputs, offset = [], 0
                for boundary in boundaries + [steps]:
                    if boundary > offset:
                        output, hidden = self.memory_gru(inputs[:, offset:boundary], hidden)
                        outputs.append(output)
                    if boundary < steps:
                        hidden = hidden * (~reset[:, boundary])[None, :, None]
                    offset = boundary
                output = torch.cat(outputs, 1)
            residual = self.memory_projection(output)
        return features + residual.to(features.dtype), hidden[0]

    def tower_input(self, maps, t: int) -> torch.Tensor:
        """The planes tower T sees: its maps' current planes, then the same maps' previous planes."""
        stacked = maps.shape[1] >= 2 * N_MAPS
        cur = [self.unpack[m](maps[:, m]) for m in self.tower_maps[t]]
        prev = [self.unpack[m](maps[:, N_MAPS + m]) for m in self.tower_maps[t]] if stacked else cur
        return torch.cat(cur + prev, 1)

    def condition(self, features: torch.Tensor, arena: torch.Tensor, tower: int) -> torch.Tensor:
        gain, bias = self.conditioning[tower](arena).chunk(2, dim=1)
        return torch.relu(features * (1 + 0.1 * torch.tanh(gain)[:, :, None, None])
                          + 0.1 * torch.tanh(bias)[:, :, None, None])

    @staticmethod
    def scalar_input(scalars: torch.Tensor) -> torch.Tensor:
        """Own/arena sensors only; enemy slots belong exclusively to the shared enemy encoder."""
        return torch.cat([scalars[:, :64], scalars[:, 88:96]], 1)

    @staticmethod
    def opponent_input(scalars: torch.Tensor) -> torch.Tensor:
        """Retain each enemy's original eight fields, then its four new measurements."""
        if scalars.shape[1] != N_SCALARS:
            raise ValueError(f"Observation schema {OBSERVATION_VERSION} requires {N_SCALARS} scalars")
        original = scalars[:, 64:88].reshape(-1, 3, 8)
        extra = scalars[:, ENEMY_EXTRA_OFFSET:].reshape(-1, 3, ENEMY_EXTRA_FEATURES)
        return torch.cat([original, extra], dim=2)

    def attend_opponents(self, scalars: torch.Tensor, context: torch.Tensor):
        enemies = self.opponent_input(scalars)
        present = enemies[:, :, 7] > 0.5
        tokens = self.opponents(enemies)
        query = torch.tanh(self.opponent_query(context))
        scores = (tokens * query[:, None, :]).sum(-1) / math.sqrt(64)
        # Normalise in float32: 1e-9 rounds to zero in float16, making an all-absent
        # opponent set produce 0/0 on the half-precision actor otherwise.
        weights = torch.softmax(scores.float().masked_fill(~present, -1e4), dim=1)
        weights = weights * present.to(weights.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-9)
        return (tokens * weights.to(tokens.dtype)[:, :, None]).sum(1), weights, tokens, query

    def trunk_and_arena(self, maps, scalars):
        arena = self.towers[2].features(self.tower_input(maps, 2))
        arena_summary = self.towers[2].head(arena)
        parts = [self.towers[t].head(self.condition(self.towers[t].features(self.tower_input(maps, t)),
                                                  arena_summary, t)) for t in (0, 1)]
        parts.extend([arena_summary, self.scalars(self.scalar_input(scalars))])
        context = torch.cat(parts, 1)
        enemies, _, _, _ = self.attend_opponents(scalars, context)
        return self.trunk(torch.cat([context, enemies], 1)), arena

    def trunk_out(self, maps, scalars):
        return self.trunk_and_arena(maps, scalars)[0]

    def _policy_value(self, h, mask):
        logits = self.pi(h)
        if mask is not None:
            if mask.shape != logits.shape or mask.dtype != torch.bool:
                raise ValueError("legal mask must be boolean with one entry per action")
            logits = logits.masked_fill(~mask, _masked(logits.dtype))
        return logits, torch.tanh(self.v(h).squeeze(-1))

    def forward_step(self, maps, scalars, mask, state, reset=None, context=None):
        h = self.trunk_out(maps, scalars)
        h, next_state = self._remember(h[:, None], state, None if reset is None else reset[:, None],
                                       None if context is None else context[:, None])
        logits, value = self._policy_value(h[:, 0], mask)
        return logits, value, next_state

    def heads_step(self, maps, scalars, mask, state, reset=None, context=None):
        h, arena = self.trunk_and_arena(maps, scalars)
        h, next_state = self._remember(h[:, None], state, None if reset is None else reset[:, None],
                                       None if context is None else context[:, None])
        h = h[:, 0]
        logits, value = self._policy_value(h, mask)
        own = nn.functional.interpolate(self.own(arena), size=(GRID, GRID), mode="bilinear", align_corners=False)
        return logits, value, self.aux(h), own, next_state

    def _sequence_features(self, maps, scalars, ownership=False):
        if maps.ndim != 5 or scalars.ndim != 3 or maps.shape[:2] != scalars.shape[:2]:
            raise ValueError("sequences require maps [batch,time,maps,height,width] and scalars [batch,time,features]")
        batch, steps = maps.shape[:2]
        if batch < 1 or steps < 1:
            raise ValueError("a sequence needs a nonempty batch and time axis")
        flat_maps, flat_scalars = maps.flatten(0, 1), scalars.flatten(0, 1)
        h, arena = self.trunk_and_arena(flat_maps, flat_scalars)
        return h.reshape(batch, steps, -1), arena if ownership else None

    def forward_sequence(self, maps, scalars, mask, state, reset=None, context=None):
        h, _ = self._sequence_features(maps, scalars)
        h, next_state = self._remember(h, state, reset, context)
        logits, value = self._policy_value(h, mask)
        return logits, value, next_state

    def heads_sequence(self, maps, scalars, mask, state, reset=None, context=None):
        h, arena = self._sequence_features(maps, scalars, ownership=True)
        h, next_state = self._remember(h, state, reset, context)
        logits, value = self._policy_value(h, mask)
        own = nn.functional.interpolate(self.own(arena), size=(GRID, GRID), mode="bilinear", align_corners=False)
        own = own.reshape(*h.shape[:2], 3, GRID, GRID)
        return logits, value, self.aux(h), own, next_state

    def forward(self, maps, scalars, mask=None):
        """Fresh-state diagnostic; real inference must retain state with ``forward_step``."""
        return self.forward_step(maps, scalars, mask, self.initial_state(len(maps)))[:2]

    def heads(self, maps, scalars, mask=None):
        """Fresh-state diagnostic; sequence learning must use ``heads_sequence``."""
        return self.heads_step(maps, scalars, mask, self.initial_state(len(maps)))[:4]


def _pick(logits, value, greedy):
    if greedy:
        a = logits.argmax(1)
    else:
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-9, 1.0)))
        a = (logits + gumbel).argmax(1)
    logp = torch.log_softmax(logits, 1).gather(1, a[:, None]).squeeze(1)
    return a, logp, value


def make_net(version: int = 4) -> PolicyNet:
    """Construct the supported fighter architecture."""
    if version != 4:
        raise ValueError(f"Only V4 networks are supported; received version {version}.")
    return PolicyNet()


def net_version(state_dict) -> int:
    """Require an explicit V4 checkpoint marker; architecture is never guessed from keys."""
    marker = state_dict.get("net_version")
    if marker is None:
        raise ValueError("Only V4 checkpoints are supported; the net_version=4 marker is missing.")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if not isinstance(marker, torch.Tensor) or marker.ndim != 0 or marker.dtype not in integer_types:
        raise ValueError("Invalid checkpoint version marker: expected the scalar integer net_version=4.")
    version = int(marker.item())
    if version != 4:
        raise ValueError(f"Only V4 checkpoints are supported; received version {version}.")
    observation = state_dict.get("observation_version")
    if (not isinstance(observation, torch.Tensor) or observation.ndim != 0
            or observation.dtype not in integer_types or int(observation.item()) != OBSERVATION_VERSION):
        raise ValueError(f"V4 observation schema {OBSERVATION_VERSION} is required; migrate the checkpoint before loading")
    memory = state_dict.get("memory_version")
    if (not isinstance(memory, torch.Tensor) or memory.ndim != 0 or memory.dtype not in integer_types
            or int(memory.item()) != MEMORY_VERSION):
        raise ValueError(f"V4 memory schema {MEMORY_VERSION} is required; migrate the checkpoint before loading")
    return version
