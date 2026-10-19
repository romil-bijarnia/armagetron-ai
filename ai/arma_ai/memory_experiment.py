"""Compare a small recurrent policy adapter with a matched stateless adapter.

The current policy and its trunk stay frozen. Features, adapters and optimizer states stay in
memory; the only optional output is a JSON report. This module never replaces a checkpoint.
Both adapters start with an exactly zero residual and receive the same contiguous training
windows. Validation carries recurrent state through each complete life and resets between lives.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import protocol as P
from .model import _masked


class GRUResidual(nn.Module):
    def __init__(self, feature_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.feature_dim, self.hidden = feature_dim, hidden
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.projection = nn.Linear(hidden, feature_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features, state=None, reset=None):
        single = features.ndim == 2
        x = features[:, None] if single else features
        if x.ndim != 3 or x.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape [batch, time, feature_dim]")
        batch, steps, _ = x.shape
        if batch < 1 or steps < 1:
            raise ValueError("a recurrent sequence must contain a batch and a time step")
        hidden = x.new_zeros(1, batch, self.hidden) if state is None else state.unsqueeze(0)
        if hidden.shape != (1, batch, self.hidden):
            raise ValueError("recurrent state has the wrong batch or hidden size")
        if reset is None:
            output, hidden = self.gru(x, hidden)
        else:
            reset = reset[:, None] if single and reset.ndim == 1 else reset
            if reset.shape != (batch, steps) or reset.dtype != torch.bool:
                raise ValueError("reset must be a boolean [batch, time] mask")
            # Usually there is only a reset at the start. Split only where some life resets,
            # keeping ordinary unrolls inside the fused GRU implementation.
            boundaries = reset.any(0).nonzero().flatten().tolist()
            output, offset = [], 0
            for boundary in boundaries + [steps]:
                if boundary > offset:
                    piece, hidden = self.gru(x[:, offset:boundary], hidden)
                    output.append(piece)
                if boundary < steps:
                    hidden = hidden * (~reset[:, boundary])[None, :, None]
                offset = boundary
            output = torch.cat(output, 1)
        adapted = x + self.projection(output)
        return (adapted[:, 0] if single else adapted), hidden[0]


class StatelessResidual(nn.Module):
    def __init__(self, feature_dim: int = 768, hidden: int = 640):
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden)
        self.projection = nn.Linear(hidden, feature_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features, state=None, reset=None):
        return features + self.projection(F.relu(self.input(features))), None


class IdentityResidual(nn.Module):
    def forward(self, features, state=None, reset=None):
        return features, None


class AdapterPolicy(nn.Module):
    """A frozen copy of the base policy head; gradients still reach the residual's inputs."""

    def __init__(self, policy_head: nn.Linear, adapter: nn.Module):
        super().__init__()
        self.policy_head = copy.deepcopy(policy_head).cpu().float().requires_grad_(False)
        self.adapter = adapter

    @property
    def extra_parameters(self) -> int:
        return sum(p.numel() for p in self.adapter.parameters())

    def base_logits(self, features, legal):
        logits = self.policy_head(features)
        return logits.masked_fill(~legal, _masked(logits.dtype))

    def forward(self, features, legal, state=None, reset=None):
        adapted, state = self.adapter(features, state, reset)
        return self.base_logits(adapted, legal), state


@dataclass(frozen=True)
class FeatureLife:
    life_id: str
    source_life: str
    features: torch.Tensor
    legal: torch.Tensor
    actions: torch.Tensor
    times: torch.Tensor

    def __post_init__(self):
        n = len(self.features)
        if n < 1 or self.features.ndim != 2 or self.features.device.type != "cpu":
            raise ValueError("cached life features must be a nonempty CPU [time, features] tensor")
        if self.features.dtype != torch.float32 or not torch.isfinite(self.features).all():
            raise ValueError("cached features must be finite float32 values")
        if self.legal.shape != (n, P.N_ACTIONS) or self.legal.dtype != torch.bool:
            raise ValueError("cached legal masks must have shape [time, six actions]")
        if any(value.device.type != "cpu" for value in (self.legal, self.actions, self.times)):
            raise ValueError("all cached life tensors must stay on the CPU")
        if self.actions.shape != (n,) or self.actions.dtype != torch.long:
            raise ValueError("cached labels must be a long tensor with one action per time")
        if ((self.actions < 0) | (self.actions >= P.N_ACTIONS)).any():
            raise ValueError("cached action is out of range")
        if not self.legal.gather(1, self.actions[:, None]).all():
            raise ValueError("cached demonstration action is illegal")
        if self.times.shape != (n,) or not torch.isfinite(self.times).all():
            raise ValueError("cached life times are invalid")
        if n > 1 and not torch.allclose(self.times[1:] - self.times[:-1], self.times.new_full((n - 1,), .05), atol=.001, rtol=0):
            raise ValueError("a cached life cannot cross a missing decision or reset")


@dataclass
class FeatureCache:
    lives: tuple[FeatureLife, ...]
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    class_weights: torch.Tensor
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        by_id = {life.life_id: life for life in self.lives}
        if len(by_id) != len(self.lives) or not self.training_ids or not self.validation_ids:
            raise ValueError("feature lives and both split lists must be nonempty and unique")
        train, valid = set(self.training_ids), set(self.validation_ids)
        if len(train) != len(self.training_ids) or len(valid) != len(self.validation_ids) or train & valid or train | valid != set(by_id):
            raise ValueError("every feature life must belong to exactly one split")
        if {by_id[k].source_life for k in train} & {by_id[k].source_life for k in valid}:
            raise ValueError("all segments of a source life must stay in the same split")
        if len({life.features.shape[1] for life in self.lives}) != 1:
            raise ValueError("all cached lives must use the same feature width")
        if self.class_weights.shape != (P.N_ACTIONS,) or not torch.isfinite(self.class_weights).all() or (self.class_weights <= 0).any():
            raise ValueError("class weights must be six finite positive values")

    def split(self, name: str) -> list[FeatureLife]:
        ids = self.training_ids if name == "training" else self.validation_ids if name == "validation" else None
        if ids is None:
            raise ValueError("split must be training or validation")
        lookup = {life.life_id: life for life in self.lives}
        return [lookup[key] for key in ids]


def _weight_digest(model) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def cache_features(base, demonstrations, batch: int = 64, *, progress=None) -> FeatureCache:
    """Cache the fresh observation-schema-2 trunk once, preserving the dataset's whole-life split."""
    if getattr(base, "observation_schema", None) != 2 or P.N_SCALARS != 108:
        raise ValueError("memory comparisons require the upgraded 108-scalar observation schema")
    if batch < 1 or base.pi.weight.dtype != torch.float32:
        raise ValueError("use a positive batch and a float32 frozen base")
    rows = np.asarray(demonstrations.rows)
    if rows.ndim != 2 or rows.shape[1] != 4 or not len(rows):
        raise ValueError("demonstrations must provide indexed observations")
    membership = np.full(len(rows), -1, np.int8)
    membership[demonstrations.training] = 0
    if np.any(membership[demonstrations.validation] != -1):
        raise ValueError("training and validation observations overlap")
    membership[demonstrations.validation] = 1
    if np.any(membership < 0):
        raise ValueError("every observation must belong to a split")
    for episode in np.unique(rows[:, 0]):
        if len(np.unique(membership[rows[:, 0] == episode])) != 1:
            raise ValueError("training and validation cannot split a source life")
    device = base.pi.weight.device
    was_training = base.training
    base.eval()
    features = torch.empty((len(rows), base.pi.in_features), dtype=torch.float32)
    try:
        for start in range(0, len(rows), batch):
            indices = np.arange(start, min(start + batch, len(rows)))
            maps, scalars, _, _ = demonstrations.batch(indices, device)
            if scalars.shape[-1] != P.N_SCALARS:
                raise ValueError("cached observations do not have 108 scalars")
            features[indices] = base.trunk_out(maps, scalars).float().cpu()
            if progress:
                progress(min(start + batch, len(rows)), len(rows))
    finally:
        base.train(was_training)
    lives, train_ids, valid_ids = [], [], []
    for episode, slot in sorted(set(map(tuple, rows[:, [0, 2]]))):
        indices = np.flatnonzero((rows[:, 0] == episode) & (rows[:, 2] == slot))
        indices = indices[np.argsort(rows[indices, 1])]
        r = demonstrations.episodes[episode]
        segments, current = [], []
        for index in indices:
            tick, previous = rows[index, 1], rows[index, 3]
            if current:
                last_tick = rows[current[-1], 1]
                if previous != last_tick or tick != last_tick + 1 or abs(r["times"][tick] - r["times"][last_tick] - .05) >= .001:
                    segments.append(current)
                    current = []
            current.append(int(index))
        if current:
            segments.append(current)
        identity = getattr(demonstrations, "episode_ids", None)
        source = identity[episode]["sha256"] if identity else f"episode:{episode}"
        for segment in segments:
            ticks = rows[segment, 1]
            key = f"{episode}:{slot}:{ticks[0]}"
            bits = r["masks"][ticks, slot]
            life = FeatureLife(key, source, features[segment].clone(),
                               torch.from_numpy(((bits[:, None] >> np.arange(P.N_ACTIONS)) & 1).astype(bool)),
                               torch.as_tensor(demonstrations.labels[segment], dtype=torch.long).clone(),
                               torch.as_tensor(r["times"][ticks], dtype=torch.float64).clone())
            lives.append(life)
            (train_ids if membership[segment[0]] == 0 else valid_ids).append(key)
    metadata = {"observation_version": 2, "feature_width": base.pi.in_features,
                "base_weights_sha256": _weight_digest(base), "samples": len(rows),
                "raw_resource_inputs_appended": False,
                "new_resource_columns_zero": (bool(torch.count_nonzero(base.opponents[0].weight[:, 8:]) == 0)
                                              if hasattr(base, "opponents") else None),
                "sources": [{k: source[k] for k in ("path", "sha256") if k in source}
                            for source in getattr(demonstrations, "sources", [])]}
    return FeatureCache(tuple(lives), tuple(train_ids), tuple(valid_ids),
                        torch.as_tensor(demonstrations.weights, dtype=torch.float32).clone(), metadata)


@dataclass(frozen=True)
class Window:
    life_id: str
    start: int
    end: int


def training_windows(cache: FeatureCache, unroll: int = 32) -> tuple[Window, ...]:
    if unroll < 1:
        raise ValueError("unroll must be positive")
    return tuple(Window(life.life_id, start, min(start + unroll, len(life.features)))
                 for life in cache.split("training") for start in range(0, len(life.features), unroll))


@dataclass
class WindowBatch:
    features: torch.Tensor
    legal: torch.Tensor
    actions: torch.Tensor
    learn: torch.Tensor
    reset: torch.Tensor

    def to(self, device):
        return WindowBatch(*(getattr(self, key).to(device) for key in ("features", "legal", "actions", "learn", "reset")))


def window_batch(cache: FeatureCache, windows: list[Window] | tuple[Window, ...], burn_in: int = 16,
                 unroll: int = 32) -> WindowBatch:
    if not windows or burn_in < 0 or unroll < 1:
        raise ValueError("windows and unroll must be nonempty; burn-in cannot be negative")
    lives = {life.life_id: life for life in cache.split("training")}
    width = cache.lives[0].features.shape[1]
    size = (len(windows), burn_in + unroll)
    out = WindowBatch(torch.zeros(*size, width), torch.ones(*size, P.N_ACTIONS, dtype=torch.bool),
                      torch.zeros(size, dtype=torch.long), torch.zeros(size, dtype=torch.bool),
                      torch.zeros(size, dtype=torch.bool))
    for i, window in enumerate(windows):
        if window.life_id not in lives:
            raise ValueError("training windows cannot contain a held-out life")
        life = lives[window.life_id]
        if not 0 <= window.start < window.end <= len(life.features) or window.end - window.start > unroll:
            raise ValueError("window is outside its source life")
        first = max(0, window.start - burn_in)
        offset = burn_in - (window.start - first)
        end = offset + window.end - first
        out.features[i, offset:end] = life.features[first:window.end]
        out.legal[i, offset:end] = life.legal[first:window.end]
        out.actions[i, offset:end] = life.actions[first:window.end]
        out.reset[i, offset] = True  # discard padded prefix state before the first real observation
        out.learn[i, burn_in:burn_in + window.end - window.start] = True
    return out


def training_loss(policy: AdapterPolicy, batch: WindowBatch, burn_in: int, class_weights, anchor: float):
    state = None
    if burn_in:
        with torch.no_grad():
            _, state = policy.adapter(batch.features[:, :burn_in], reset=batch.reset[:, :burn_in])
        if state is not None:
            state = state.detach()
    x, legal = batch.features[:, burn_in:], batch.legal[:, burn_in:]
    logits, _ = policy(x, legal, state, batch.reset[:, burn_in:])
    labels, valid = batch.actions[:, burn_in:], batch.learn[:, burn_in:]
    logp = F.log_softmax(logits, -1)[valid]
    if not len(logp):
        raise ValueError("a training batch needs at least one learned decision")
    labels = labels[valid]
    with torch.no_grad():
        reference = F.log_softmax(policy.base_logits(x, legal), -1)[valid]
    imitation = F.nll_loss(logp, labels, weight=class_weights)
    return imitation + anchor * F.kl_div(logp, reference, reduction="batchmean", log_target=True), len(labels)


@torch.inference_mode()
def sequence_predictions(policy: AdapterPolicy, lives: list[FeatureLife] | tuple[FeatureLife, ...],
                         chunk_size: int = 128, *, reset_each_decision: bool = False) -> dict[str, torch.Tensor]:
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    device = policy.policy_head.weight.device
    policy.eval()
    result = {}
    for life in lives:
        state, outputs = None, []  # never inherit another life, even when its length happens to match
        for start in range(0, len(life.features), chunk_size):
            features = life.features[start:start + chunk_size].to(device)
            legal = life.legal[start:start + chunk_size].to(device)
            if reset_each_decision:
                # Treat time as the batch dimension: every decision gets zero hidden state.
                logits, _ = policy(features, legal)
            else:
                logits, state = policy(features[None], legal[None], state)
                logits = logits[0]
            outputs.append(F.log_softmax(logits, -1).cpu())
        result[life.life_id] = torch.cat(outputs)
    return result


def evaluate_cached(policy: AdapterPolicy, lives, class_weights, chunk_size: int = 128, *,
                    reset_each_decision: bool = False) -> dict:
    predictions = sequence_predictions(policy, lives, chunk_size, reset_each_decision=reset_each_decision)
    logp = torch.cat([predictions[life.life_id] for life in lives])
    labels, legal = torch.cat([life.actions for life in lives]), torch.cat([life.legal for life in lives])
    predicted = logp.argmax(1)
    turns = torch.isin(labels, torch.tensor([1, 2, 4, 5]))
    predicted_turns = torch.isin(predicted, torch.tensor([1, 2, 4, 5]))
    eligible = legal[:, [1, 2, 4, 5]].any(1) & ~turns
    confusion = torch.bincount(labels * P.N_ACTIONS + predicted, minlength=P.N_ACTIONS ** 2).reshape(P.N_ACTIONS, P.N_ACTIONS)
    nll = -logp[torch.arange(len(labels)), labels]
    weight = class_weights.cpu()[labels]
    return {"samples": len(labels), "lives": len(lives), "nll": float(nll.mean()),
            "history": "reset_each_decision" if reset_each_decision else "whole_life",
            "weighted_nll": float((nll * weight).sum() / weight.sum()),
            "accuracy": float((predicted == labels).float().mean()),
            "straight_only_accuracy": float((labels == 0).float().mean()),
            "turn_accuracy": float((predicted[turns] == labels[turns]).float().mean()) if turns.any() else None,
            "false_turn_rate": float(predicted_turns[~turns].float().mean()) if (~turns).any() else None,
            "eligible_false_turn_rate": float(predicted_turns[eligible].float().mean()) if eligible.any() else None,
            "action_support": confusion.sum(1).tolist(), "predicted_support": confusion.sum(0).tolist(),
            "confusion": confusion.tolist(), "mean_braking_probability": float(logp.exp()[:, [3, 4, 5]].sum(1).mean())}


@dataclass
class ComparisonConfig:
    updates: int = 120
    batch_windows: int = 2  # up to 64 learned decisions, matching the existing 64-decision update
    burn_in: int = 16
    unroll: int = 32
    gru_hidden: int = 256
    stateless_hidden: int = 640
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    anchor: float = .1
    class_balancing: bool = True
    seed: int = 113
    device: str = "cpu"
    evaluation_chunk: int = 128

    def validate(self):
        if min(self.updates, self.batch_windows, self.unroll, self.gru_hidden, self.stateless_hidden, self.evaluation_chunk) < 1 or self.burn_in < 0:
            raise ValueError("comparison sizes must be positive; burn-in cannot be negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if any(not math.isfinite(x) or x < 0 for x in (self.weight_decay, self.anchor)):
            raise ValueError("weight decay and anchor must be finite and nonnegative")


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def compare_adapters(base, cache: FeatureCache, cfg: ComparisonConfig, *, progress=None,
                     report_path: Path | None = None) -> tuple[dict, dict[str, AdapterPolicy]]:
    """Train equal update/token schedules and return in-memory policies plus a JSON-safe report."""
    cfg.validate()
    if base.pi.in_features != cache.lives[0].features.shape[1]:
        raise ValueError("base head and feature cache have different widths")
    if cache.metadata.get("base_weights_sha256") and cache.metadata["base_weights_sha256"] != _weight_digest(base):
        raise ValueError("feature cache belongs to different base weights")
    if report_path is not None and report_path.suffix.lower() != ".json":
        raise ValueError("only a JSON report may be persisted")
    device = torch.device(cfg.device)
    with torch.random.fork_rng():
        torch.manual_seed(cfg.seed)
        recurrent = AdapterPolicy(base.pi, GRUResidual(base.pi.in_features, cfg.gru_hidden)).to(device)
        torch.manual_seed(cfg.seed)
        stateless = AdapterPolicy(base.pi, StatelessResidual(base.pi.in_features, cfg.stateless_hidden)).to(device)
    policies = {"gru": recurrent, "stateless": stateless,
                "base": AdapterPolicy(base.pi, IdentityResidual()).to(device)}
    weights = cache.class_weights if cfg.class_balancing else torch.ones(P.N_ACTIONS)
    validation = cache.split("validation")
    baseline = evaluate_cached(policies["base"], validation, weights, cfg.evaluation_chunk)
    windows = training_windows(cache, cfg.unroll)
    rng = np.random.default_rng(cfg.seed)
    optimizers = {name: torch.optim.AdamW(policy.adapter.parameters(), lr=cfg.learning_rate,
                                         weight_decay=cfg.weight_decay) for name, policy in policies.items() if name != "base"}
    seconds, learned = {name: 0. for name in optimizers}, {name: 0 for name in optimizers}
    for step in range(cfg.updates):
        chosen = rng.choice(len(windows), size=min(cfg.batch_windows, len(windows)), replace=False)
        batch = window_batch(cache, [windows[i] for i in chosen], cfg.burn_in, cfg.unroll).to(device)
        for name, optimizer in optimizers.items():
            policy = policies[name]
            policy.train()
            _sync(device)
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss, tokens = training_loss(policy, batch, cfg.burn_in, weights.to(device), cfg.anchor)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {name} adapter loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.adapter.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            _sync(device)
            seconds[name] += time.monotonic() - started
            learned[name] += tokens
            if progress:
                progress(name, step + 1, cfg.updates, float(loss.detach()))
    report = {"config": asdict(cfg), "cache": cache.metadata, "base_parameters": sum(p.numel() for p in base.parameters()),
              "baseline": baseline, "training_lives": list(cache.training_ids), "validation_lives": list(cache.validation_ids),
              "class_weights": weights.tolist(), "models": {},
              "training_state": "Zero at each burn-in window start; detached after burn-in. Validation carries state through whole lives.",
              "paired_training_windows": True,
              "interpretation": "Offline imitation metrics; playing strength requires closed-loop evaluation."}
    for name in optimizers:
        policy = policies[name]
        report["models"][name] = {"extra_parameters": policy.extra_parameters, "updates": cfg.updates,
                                  "learned_decisions": learned[name], "training_seconds": seconds[name],
                                  "validation": evaluate_cached(policy, validation, weights, cfg.evaluation_chunk)}
    report["models"]["gru"]["reset_each_decision_ablation"] = evaluate_cached(
        policies["gru"], validation, weights, cfg.evaluation_chunk, reset_each_decision=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report, policies


class SimulatorActor:
    """State belongs to one controlled agent per game. Search-node inference is rejected."""

    def __init__(self, base, policy: AdapterPolicy):
        self.base, self.policy = base.eval(), policy.eval()
        if base.pi.weight.device != policy.policy_head.weight.device:
            raise ValueError("base and policy must use the same device")
        self.states = {}
        self.last_turn, self.game_decisions = {}, {}
        self.action_counts = np.zeros(P.N_ACTIONS, np.int64)
        self.inference_times, self.inference_batch_sizes = [], []
        self.entropy_total, self.hidden_norm_total = 0., 0.
        self.hidden_count, self.hidden_norm_max, self.rapid_reversals = 0, 0., 0

    def reset(self, game: int):
        self.states.pop(game, None)
        self.last_turn.pop(game, None)
        self.game_decisions.pop(game, None)

    @torch.inference_mode()
    def feed(self, simulator, n: int):
        if not n:
            return
        games = simulator.game_ids[:n].tolist()
        if len(set(games)) != n or not np.all(simulator.roots[:n] == 1):
            raise ValueError("memory rollout requires one real-decision agent per game and no search")
        device = self.policy.policy_head.weight.device
        _sync(device)
        started = time.monotonic()
        features = self.base.trunk_out(torch.from_numpy(simulator.maps[:n]).to(device),
                                       torch.from_numpy(simulator.scalars[:n]).to(device))
        legal = torch.from_numpy(((simulator.masks[:n, None] >> np.arange(P.N_ACTIONS)) & 1).astype(bool)).to(device)
        state = None
        if isinstance(self.policy.adapter, GRUResidual):
            state = torch.stack([self.states.get(game, features.new_zeros(self.policy.adapter.hidden)) for game in games])
        logits, updated = self.policy(features, legal, state)
        values = torch.tanh(self.base.v(features)).squeeze(-1)
        _sync(device)
        self.inference_times.append(time.monotonic() - started)
        self.inference_batch_sizes.append(n)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all() or (updated is not None and not torch.isfinite(updated).all()):
            raise FloatingPointError("adapter rollout produced non-finite state or outputs")
        if updated is not None:
            self.states.update({game: updated[i].detach() for i, game in enumerate(games)})
            norms = updated.norm(dim=1)
            self.hidden_norm_total += float(norms.sum())
            self.hidden_count += n
            self.hidden_norm_max = max(self.hidden_norm_max, float(norms.max()))
        logp = F.log_softmax(logits, 1)
        self.entropy_total += float(-(logp.exp() * logp).sum())
        for game, action in zip(games, logits.argmax(1).cpu().tolist()):
            self.action_counts[action] += 1
            decision = self.game_decisions.get(game, 0)
            steering = -1 if action in (1, 4) else 1 if action in (2, 5) else 0
            if steering:
                previous, when = self.last_turn.get(game, (0, -1000))
                self.rapid_reversals += previous == -steering and decision - when <= 6
                self.last_turn[game] = steering, decision
            self.game_decisions[game] = decision + 1
        simulator.feed(logits.float().cpu().numpy(), values.float().cpu().numpy())

    def statistics(self):
        n = int(self.action_counts.sum())
        turns = int(self.action_counts[[1, 2, 4, 5]].sum())
        milliseconds = np.asarray(self.inference_times) * 1000
        return {"scope": "Issued controls, including decisions in unfinished games", "decisions": n,
                "action_counts": self.action_counts.tolist(), "turn_fraction": turns / n if n else None,
                "brake_fraction": float(self.action_counts[[3, 4, 5]].sum()) / n if n else None,
                "rapid_opposite_turns_within_300ms": int(self.rapid_reversals),
                "mean_policy_entropy": self.entropy_total / n if n else None,
                "mean_hidden_l2": self.hidden_norm_total / self.hidden_count if self.hidden_count else None,
                "max_hidden_l2": self.hidden_norm_max if self.hidden_count else None,
                "inference_mean_ms_per_decision": float(milliseconds.sum()) / n if n else None,
                "inference_batch_p50_ms": float(np.median(milliseconds)) if len(milliseconds) else None,
                "inference_batch_p95_ms": float(np.percentile(milliseconds, 95)) if len(milliseconds) else None,
                "inference_batch_max_ms": float(milliseconds.max()) if len(milliseconds) else None,
                "inference_batches": len(self.inference_batch_sizes),
                "inference_batch_size_min": min(self.inference_batch_sizes, default=None),
                "inference_batch_size_max": max(self.inference_batch_sizes, default=None),
                "inference_batch_size_mean": float(np.mean(self.inference_batch_sizes)) if self.inference_batch_sizes else None,
                "inference_scope": "PyTorch shared trunk plus adapter, synchronized; excludes simulator and statistics",
                "numerically_finite": True}


def evaluate_hunter(base, policy: AdapterPolicy, *, rounds: int = 24, sizes=(-3., -1.5, 0.),
                    games: int = 4, max_decisions: int = 1200, seed: int = 113,
                    time_budget_s: float | None = None, progress=None) -> dict:
    """Optional closed-loop check. Fixed starts repeat; seeds do not make independent trials.

    With the three default sizes, six rounds cover each size/rotation pair once. The optional
    progress callback receives completed rounds, requested rounds, active games and decisions.
    """
    from .sim import AGENT_HUNTER, SelfPlay

    if min(rounds, games, max_decisions) < 1 or not sizes:
        raise ValueError("rounds, games, sizes and decision limit must be positive")
    if any(not math.isfinite(float(size)) for size in sizes) or (time_budget_s is not None and (not math.isfinite(time_budget_s) or time_budget_s <= 0)):
        raise ValueError("arena sizes must be finite and an evaluation budget must be positive")
    actor = SimulatorActor(base, policy)
    simulator = SelfPlay(min(games, rounds), sims=0, search_prob=0, temperature=0, threads=1,
                         max_decisions=max_decisions, hunter_depth=2)
    active, next_round, results, trials = set(), 0, [], {}
    started = time.monotonic()

    def start(game):
        nonlocal next_round
        index = next_round
        next_round += 1
        trials[game] = index
        actor.reset(game)
        simulator.start(game, float(sizes[index % len(sizes)]), [0, AGENT_HUNTER], [False, False],
                        rotate=index % 2, seed=seed + index)
        active.add(game)

    for game in range(simulator.games):
        start(game)
    while active and (time_budget_s is None or time.monotonic() - started < time_budget_s):
        n = simulator.collect()
        actor.feed(simulator, n)
        for game in tuple(active):
            info = simulator.info(game)
            if info is None:
                continue
            active.remove(game)
            results.append({**info, "trial": trials[game], "spawn_rotation": trials[game] % 2})
            if next_round < rounds:
                start(game)
        if progress:
            progress(len(results), rounds, len(active), int(actor.action_counts.sum()))
    outcomes = [r["outcome"][0] for r in results]
    wins, losses = sum(x >= .999 for x in outcomes), sum(x < -1e-6 for x in outcomes)
    return {"environment": "simulator", "opponent": "hunter_depth_2", "search": False, "policy": "greedy",
            "rounds": len(results), "requested": rounds, "wins": wins, "draws": len(results) - wins - losses,
            "losses": losses, "credited_kills": sum(r["kills"][0] for r in results),
            "surviving_kill_rounds": sum(not r["died"][0] and r["kills"][0] > 0 for r in results),
            "deaths_within_5s": sum(r["died"][0] and r["decisions"] <= 100 for r in results),
            "limit_draws": sum(r["decisions"] >= max_decisions and not any(r["died"]) for r in results),
            "mean_decisions": float(np.mean([r["decisions"] for r in results])) if results else None,
            "max_decisions": max_decisions, "sizes": list(sizes), "seed": seed,
            "games": simulator.games, "time_budget_s": time_budget_s,
            "planned_starts": [{"trial": i, "size": float(sizes[i % len(sizes)]),
                                "spawn_rotation": i % 2, "seed": seed + i} for i in range(rounds)],
            "wall_seconds": time.monotonic() - started, "partial": len(results) < rounds,
            "behavior": actor.statistics(), "round_results": results}
