"""Continue training the same Blacklight checkpoint by imitating maximum-difficulty built-in bots.

    imitate                         # continue until stop or Ctrl-C; save to runs/blacklight/latest.pt
    imitate --epochs 3              # three additional learning cycles, then save and stop

Every session resumes the current weights. Imitation and RL share one persistent checkpoint
and an exclusive learner lock. Parallel games supply experience; they do not create new AIs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import fcntl
import os
import signal
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from . import protocol as P
from .model import make_net, net_version
from .inline_ui import add_task, progress_bar, update_bar
from .record import load, load_metadata, record, save

PROJECT = Path(__file__).resolve().parent.parent
ACTION_NAMES = list(P.ACTION_NAMES)


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_provenance(metadata: dict) -> None:
    expected = {"schema_version": 1, "observation_version": P.OBSERVATION_VERSION, "n_scalars": P.N_SCALARS,
                "teacher_type": "engine_builtin", "iq": 100,
                "action_encoding": "steer_brake_6", "observation_timing": "before_all_teacher_actions",
                "label_source": "native_executed_control", "teacher_schedule": "policy_decision_boundaries",
                "emergency_help": False, "teacher_slots": 1}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Demonstrations require {key}={value!r}; got {metadata.get(key)!r}")
    interval = metadata.get("decision_interval")
    if not isinstance(interval, (int, float)) or not math.isclose(interval, .05, abs_tol=1e-6):
        raise ValueError("V4 demonstrations must use a 0.05-second decision interval")
    if not isinstance(metadata.get("engine_sha256"), str) or len(metadata["engine_sha256"]) != 64:
        raise ValueError("Demonstrations must identify the engine binary")
    if not metadata.get("source_fingerprint"):
        raise ValueError("Demonstrations must identify their recording sources")
    if not isinstance(metadata.get("builtin_opponents"), int) or metadata["builtin_opponents"] < 1:
        raise ValueError("The teacher must play against normal built-in opponents")


def episode_samples(r: dict, episode: int) -> np.ndarray:
    """Index observations without copying gigabytes of maps or leaking between lives."""
    required = ("times", "maps", "scalars", "masks", "actions", "flags")
    if any(key not in r for key in required):
        raise ValueError("Demonstration round lacks observations or native actions")
    actions, flags, masks = (r[key] for key in ("actions", "flags", "masks"))
    if actions.ndim != 2:
        raise ValueError("Actions must have shape [ticks, players]")
    ticks, slots = actions.shape
    if slots != 1:
        raise ValueError("Native imitation requires exactly one recorded teacher slot")
    expected = {"maps": ((ticks, slots, P.N_MAPS, P.GRID, P.GRID), np.uint8),
                "scalars": ((ticks, slots, P.N_SCALARS), np.float32),
                "masks": ((ticks, slots), np.uint8), "flags": ((ticks, slots), np.uint8),
                "actions": ((ticks, slots), np.int8), "times": ((ticks,), np.float64)}
    for key, (shape, dtype) in expected.items():
        if r[key].shape != shape or r[key].dtype != dtype:
            raise ValueError(f"Invalid {key} shape or dtype in demonstration round")
    if not np.isfinite(r["times"]).all() or np.any(np.diff(r["times"]) <= 0):
        raise ValueError("Demonstration times must be finite and increasing")
    needs = flags & P.FLAG_NEEDS_ACTION != 0
    if np.any(needs & (flags & P.FLAG_ALIVE == 0)):
        raise ValueError("A dead player cannot supply a demonstration action")
    if np.any((actions < 0) & needs) or np.any((actions >= 0) & ~needs):
        raise ValueError("Every decision needs exactly one recorded native action")
    if np.any(actions[needs] >= P.N_ACTIONS) or np.any(masks[needs] >> P.N_ACTIONS):
        raise ValueError("Demonstrations use an unknown action or mask bit")
    if np.any(((masks[needs] >> actions[needs]) & 1) == 0):
        raise ValueError("A recorded native action is illegal for its pre-action mask")
    if not np.isfinite(r["scalars"][needs]).all():
        raise ValueError("Demonstration scalars must be finite")
    rows = []
    for tick, slot in np.argwhere(needs):
        reset = flags[tick, slot] & (P.FLAG_SPAWNED | P.FLAG_DIED)
        previous = tick
        if (tick > 0 and not reset and needs[tick - 1, slot]
                and not flags[tick - 1, slot] & P.FLAG_DIED
                and abs(r["times"][tick] - r["times"][tick - 1] - .05) < .001):
            previous = tick - 1
        rows.append((episode, tick, slot, previous))
    return np.asarray(rows, np.int32).reshape(-1, 4)


class Demonstrations:
    """Deduplicate completed teacher lives, then split by round, never by frame."""

    def __init__(self, paths: list[Path], seed: int = 17, validation_fraction: float = .2):
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        self.episodes: list[dict] = []
        self.sources, self.episode_ids = [], []
        samples, seen = [], set()
        duplicates = 0
        for path in paths:
            metadata = load_metadata(path)
            validate_provenance(metadata)
            rounds = load(path)
            if len(metadata.get("rounds", [])) != len(rounds):
                raise ValueError("Demonstrations lack per-round provenance")
            self.sources.append({"path": str(path.resolve()), "sha256": fingerprint(path), "metadata": metadata})
            for index, r in enumerate(rounds):
                provenance = metadata["rounds"][index]
                rows = episode_samples(r, len(self.episodes))
                # Four-player arenas reset when the selected teacher dies, even if its
                # opponents are still fighting. That is a completed life, not a cut-off
                # demonstration. We learn no round-value targets from these recordings.
                died = bool(len(r["flags"]) and np.all(r["flags"][-1] & P.FLAG_DIED)
                            and not np.any(r["flags"][-1] & P.FLAG_ALIVE))
                if provenance.get("truncated") is not False or not (provenance.get("complete") is True or died):
                    raise ValueError("Imitation requires complete teacher lives without truncated recordings")
                if not len(rows):
                    continue
                h = hashlib.sha256()
                for key in ("times", "maps", "scalars", "masks", "actions", "flags"):
                    h.update(str(r[key].shape).encode())
                    h.update(r[key].tobytes())
                digest = h.hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                self.episode_ids.append({"source": len(self.sources) - 1, "round": index, "sha256": digest})
                self.episodes.append(r)
                samples.append(rows)
        if len(self.episodes) < 3:
            raise ValueError("Need at least three distinct teacher rounds for a held-out evaluation")
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(self.episodes))
        n_valid = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
        self.validation_rounds = sorted(shuffled[:n_valid].tolist())
        self.training_rounds = sorted(shuffled[n_valid:].tolist())
        self.rows = np.concatenate(samples)
        self.validation = np.flatnonzero(np.isin(self.rows[:, 0], self.validation_rounds))
        self.training = np.flatnonzero(np.isin(self.rows[:, 0], self.training_rounds))
        self.labels = np.asarray([self.episodes[e]["actions"][t, s] for e, t, s, _ in self.rows], np.int64)
        counts = np.bincount(self.labels[self.training], minlength=P.N_ACTIONS)
        # Temper the straight-action majority without letting a single rare move dominate.
        self.weights = np.minimum(8., np.sqrt(len(self.training) / np.maximum(counts, 1))).astype(np.float32)
        self.weights /= np.dot(counts, self.weights) / counts.sum()
        self.summary = {"sources": self.sources, "unique_rounds": len(self.episodes),
                        "duplicate_rounds_removed": duplicates, "episode_ids": self.episode_ids,
                        "training_rounds": self.training_rounds, "validation_rounds": self.validation_rounds,
                        "training_samples": len(self.training), "validation_samples": len(self.validation),
                        "actions": ACTION_NAMES, "training_action_counts": counts.tolist(),
                        "validation_action_counts": np.bincount(self.labels[self.validation], minlength=6).tolist(),
                        "class_weights": self.weights.tolist(), "seed": seed}
        self._index_sequences()

    def _index_sequences(self):
        """Keep windows inside a contiguous life, including across recording gaps."""
        self.segments = []
        current = []
        for index, (episode, tick, slot, previous) in enumerate(self.rows):
            if current:
                old = self.rows[current[-1]]
                if episode != old[0] or slot != old[2] or previous != old[1] or tick == previous:
                    self.segments.append(np.asarray(current, np.int64))
                    current = []
            current.append(index)
        if current:
            self.segments.append(np.asarray(current, np.int64))
        self.segment_of = np.empty(len(self.rows), np.int64)
        self.position_in_segment = np.empty(len(self.rows), np.int64)
        for number, segment in enumerate(self.segments):
            self.segment_of[segment] = number
            self.position_in_segment[segment] = np.arange(len(segment))

    def contexts(self, indices: np.ndarray) -> np.ndarray:
        """The teacher's prior executed control, and elapsed time in decision intervals."""
        context = np.zeros((len(indices), 7), np.float32)
        for out, (episode, tick, slot, previous) in enumerate(self.rows[indices]):
            if previous == tick:
                continue
            recording = self.episodes[episode]
            context[out, 6] = (recording["times"][tick] - recording["times"][previous]) / .05
            # Engine/simulator ignore controls at their initial time-zero observation.
            action = int(recording["actions"][previous, slot])
            if recording["times"][previous] > 1e-8 and 0 <= action < 6:
                context[out, action] = 1
        return context

    def sequence_starts(self, unroll: int) -> np.ndarray:
        if not hasattr(self, "segments"):
            self._index_sequences()
        allowed = np.zeros(len(self.rows), bool)
        allowed[self.training] = True
        starts = []
        for segment in self.segments:
            if allowed[segment].all():
                starts.extend(segment[:max(1, len(segment) - unroll + 1)])
            elif allowed[segment].any():
                raise ValueError("A recurrent life cannot cross training and validation splits")
        return np.asarray(starts, np.int64)

    def sequence_batch(self, starts, burn_in: int, unroll: int, device):
        if not hasattr(self, "segments"):
            self._index_sequences()
        if burn_in < 0 or unroll < 1 or not len(starts):
            raise ValueError("A sequence needs nonnegative burn-in and a positive unroll and batch")
        if not np.isin(starts, self.training).all():
            raise ValueError("Recurrent training windows cannot use held-out lives")
        indices = np.full((len(starts), burn_in + unroll), -1, np.int64)
        reset = np.zeros(indices.shape, bool)
        learn = np.zeros(indices.shape, bool)
        for row, start in enumerate(starts):
            segment = self.segments[self.segment_of[start]]
            offset = int(self.position_in_segment[start])
            history = segment[max(0, offset - burn_in):offset]
            future = segment[offset:offset + unroll]
            left = burn_in - len(history)
            indices[row, left:burn_in] = history
            indices[row, burn_in:burn_in + len(future)] = future
            reset[row, left] = True
            learn[row, burn_in:burn_in + len(future)] = True
        safe = np.maximum(indices, 0).reshape(-1)
        maps, scalars, mask, labels = self.batch(safe, device)
        shape = indices.shape
        contexts = torch.from_numpy(self.contexts(safe)).to(device).reshape(*shape, 7)
        return {"maps": maps.reshape(*shape, *maps.shape[1:]),
                "scalars": scalars.reshape(*shape, scalars.shape[-1]),
                "mask": mask.reshape(*shape, mask.shape[-1]),
                "labels": labels.reshape(shape), "context": contexts,
                "reset": torch.from_numpy(reset).to(device),
                "learn": torch.from_numpy(learn).to(device), "indices": indices}

    def batch(self, indices: np.ndarray, device: torch.device):
        maps, scalars, masks = [], [], []
        for episode, tick, slot, previous in self.rows[indices]:
            r = self.episodes[episode]
            maps.append(np.concatenate([r["maps"][tick, slot], r["maps"][previous, slot]]))
            scalars.append(r["scalars"][tick, slot])
            masks.append(r["masks"][tick, slot])
        legal = (np.asarray(masks)[:, None] >> np.arange(P.N_ACTIONS)) & 1
        return (torch.from_numpy(np.stack(maps)).to(device),
                torch.from_numpy(np.stack(scalars)).to(device),
                torch.from_numpy(legal.astype(bool)).to(device),
                torch.from_numpy(self.labels[indices]).to(device))


def metrics(log_probs: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict:
    predicted = log_probs.argmax(1)
    nll = -log_probs[np.arange(len(labels)), labels]
    support = np.bincount(labels, minlength=6)
    correct = np.bincount(labels[predicted == labels], minlength=6)
    turns = np.isin(labels, [1, 2, 4, 5])
    predicted_turns = np.isin(predicted, [1, 2, 4, 5])
    recall = [float(c / n) if n else None for c, n in zip(correct, support)]
    confusion = np.zeros((6, 6), np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    return {"samples": len(labels), "nll": float(nll.mean()),
            "weighted_nll": float(np.average(nll, weights=weights[labels])),
            "accuracy": float((predicted == labels).mean()),
            "straight_only_accuracy": float((labels == 0).mean()),
            "turn_accuracy": float((predicted[turns] == labels[turns]).mean()) if turns.any() else None,
            "false_turn_rate": float(predicted_turns[~turns].mean()) if (~turns).any() else None,
            "balanced_accuracy": float(np.mean([r for r in recall if r is not None])),
            "action_support": support.tolist(), "action_recall": recall, "confusion": confusion.tolist()}


@torch.inference_mode()
def predict(net, data: Demonstrations, indices: np.ndarray, batch: int, device: torch.device,
            progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    net.eval()
    if not hasattr(data, "segments"):
        data._index_sequences()
    wanted = np.zeros(len(data.rows), bool)
    wanted[indices] = True
    result = np.empty((len(data.rows), P.N_ACTIONS), np.float32)
    done = 0
    for segment in data.segments:
        positions = np.flatnonzero(wanted[segment])
        if not len(positions):
            continue
        # Reconstruct the entire prefix, not just the requested/held-out frames.
        state = net.initial_state(1, device=device)
        for start in range(0, int(positions[-1]) + 1, batch):
            rows = segment[start:min(start + batch, int(positions[-1]) + 1)]
            x, s, legal, _ = data.batch(rows, device)
            context = torch.from_numpy(data.contexts(rows)).to(device)[None]
            logits, _, state = net.forward_sequence(x[None], s[None], legal[None], state, context=context)
            result[rows] = F.log_softmax(logits[0], -1).float().cpu().numpy()
            done += int(wanted[rows].sum())
            if progress:
                progress(done, len(indices))
    result = result[indices]
    if not np.isfinite(result).all():
        raise FloatingPointError("Policy produced non-finite probabilities")
    return result


def sequence_outputs(net, data, burn_in: int):
    """Reconstruct recent state without gradients, then learn only the real unroll."""
    maps, scalars, mask = (data[key] for key in ("maps", "scalars", "mask"))
    state = net.initial_state(maps.shape[0], device=maps.device)
    if burn_in:
        with torch.no_grad():
            _, _, state = net.forward_sequence(maps[:, :burn_in], scalars[:, :burn_in],
                mask[:, :burn_in], state, reset=data["reset"][:, :burn_in],
                context=data["context"][:, :burn_in])
        state = state.detach()
    return net.heads_sequence(maps[:, burn_in:], scalars[:, burn_in:], mask[:, burn_in:], state,
        reset=data["reset"][:, burn_in:], context=data["context"][:, burn_in:])


@dataclass
class Config:
    epochs: int = 0  # additional collection/learning cycles; zero continues until stopped
    steps_per_epoch: int = 120
    batch: int = 64
    learning_rate: float = 1e-5
    anchor: float = .1
    seed: int = 17
    validation_fraction: float = .2
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    save_seconds: float = 60
    sequence_burn_in: int = 16
    sequence_unroll: int = 32
    prediction_anchor: float = 1.
    bot_pool: str = "stock"  # diverse is a candidate curriculum; qualify before a long run

    def validate(self):
        if self.bot_pool not in ("stock", "diverse"):
            raise ValueError("bot_pool must be stock or diverse")
        if self.epochs < 0 or min(self.steps_per_epoch, self.batch) < 1:
            raise ValueError("epochs must be nonnegative; steps_per_epoch and batch must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.anchor) or self.anchor < 0:
            raise ValueError("anchor must be finite and nonnegative")
        if self.sequence_burn_in < 0 or self.sequence_unroll < 1:
            raise ValueError("sequence_burn_in must be nonnegative and sequence_unroll positive")
        if not math.isfinite(self.prediction_anchor) or self.prediction_anchor < 0:
            raise ValueError("prediction_anchor must be finite and nonnegative")
        if not math.isfinite(self.save_seconds) or self.save_seconds <= 0:
            raise ValueError("save_seconds must be positive and finite")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")


class StopTraining(Exception):
    """A stop request observed outside an optimizer update."""


class RunLock:
    """The same exclusive writer lock used by the RL learner."""

    def __init__(self, run: Path):
        run.mkdir(parents=True, exist_ok=True)
        self.file = (run / ".lock").open("a")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError(f"Blacklight is already being trained in {run}; use stop before changing methods") from None

    def close(self):
        self.file.close()


class ImitationTrainer:
    """Continue the one V4 checkpoint, retaining optimizer state between imitation sessions."""

    def __init__(self, run: Path, cfg: Config, progress=None, stop_requested=None):
        cfg.validate()
        self.run, self.cfg = run.resolve(), cfg
        self.lock = RunLock(self.run)
        self.callback, self.stop_requested = progress, stop_requested or (lambda: False)
        self.started_at = time.time()
        self.pid_path = self.run / "imitation.pid"
        self.closed = False
        try:
            path = self.run / "latest.pt"
            if not path.exists():
                raise FileNotFoundError(f"No saved Blacklight at {path}; imitation will not silently start a new AI")
            self.ck = torch.load(path, map_location="cpu", weights_only=True)
            if self.ck.get("trainer") != "teach":
                raise ValueError("Expected the shared Blacklight training checkpoint, not an old isolated experiment")
            torch.manual_seed(cfg.seed)
            self.device = torch.device(cfg.device)
            self.net = make_net(net_version(self.ck["model"])).to(self.device)
            self.net.load_state_dict(self.ck["model"])
            for name, parameter in self.net.named_parameters():
                if name.startswith(("v.", "aux.", "own.")):
                    parameter.requires_grad_(False)
            self.optimizer = torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad],
                                              lr=cfg.learning_rate, weight_decay=1e-4)
            state = self.ck.get("imitation_state", {})
            self.last_method = self.ck.get("last_training_method", "rl")
            self.optimizer_resumed = self.last_method == "imitation" and bool(state.get("optimizer"))
            if self.optimizer_resumed:
                self.optimizer.load_state_dict(state["optimizer"])
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg.learning_rate
            self.rng = np.random.default_rng(cfg.seed)
            if state.get("rng_state"):
                self.rng.bit_generator.state = json.loads(state["rng_state"])
            if isinstance(state.get("torch_rng_state"), torch.Tensor):
                torch.set_rng_state(state["torch_rng_state"].cpu())
            self.update = self.start_update = int(self.ck.get("update", 0))
            self.steps = int(state.get("cumulative_steps", 0))
            self.samples = int(state.get("cumulative_samples", 0))
            self.cycles = int(state.get("cumulative_epochs", 0))
            self.trained = list(self.ck.get("trained", [0, 0]))
            self.validation = state.get("validation")
            self.validation_update = state.get("validation_update")
            self.last_saved_update = self.update
            self.saved_at = path.stat().st_mtime
            self.last_save = time.monotonic()
            self.pid_path.write_text(str(os.getpid()) + "\n")
            self.report("collecting", 0, 0, {"stage": "Resuming Blacklight"})
        except BaseException:
            self.pid_path.unlink(missing_ok=True)
            self.lock.close()
            raise

    def report(self, phase, done=0, total=0, info=None):
        value = {"trainer": "imitate", "run": str(self.run), "pid": os.getpid(),
                 "started_at": self.started_at, "time": time.time(), "phase": phase,
                 "stage": phase, "completed": done, "total": total, "update": self.update,
                 "imitation_steps": self.steps, "imitation_samples": self.samples,
                 "session_updates": self.update - self.start_update, "trained": self.trained[0],
                 "saved_update": self.last_saved_update, "saved_at": self.saved_at,
                 "config": asdict(self.cfg), "mode": "imitation", "actions": 6, "net": 4, **(info or {})}
        atomic_json(self.run / "progress.json", value)
        if self.callback:
            self.callback(phase, done, total, value)

    def check_stop(self):
        if self.stop_requested():
            raise StopTraining

    def save(self):
        if self.update == self.last_saved_update:
            return
        state = {**self.ck.get("imitation_state", {}), "optimizer": self.optimizer.state_dict(),
                 "cumulative_steps": self.steps, "cumulative_samples": self.samples,
                 "cumulative_epochs": self.cycles, "config": asdict(self.cfg),
                 "rng_state": json.dumps(self.rng.bit_generator.state), "torch_rng_state": torch.get_rng_state(),
                 "validation": self.validation, "validation_update": self.validation_update,
                 "head_supervision": "policy_with_prediction_preservation",
                 "memory_version": int(self.net.memory_version)}
        checkpoint = {**self.ck, "model": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                      "trainer": "teach", "last_training_method": "imitation", "update": self.update,
                      "trained": self.trained, "imitation_state": state}
        # Main RL moments refer to a different objective and an older set of weights.
        checkpoint["opt"] = {**checkpoint.get("opt", {}), "state": {}}
        atomic_checkpoint(self.run / "latest.pt", checkpoint)
        self.ck = checkpoint
        self.last_saved_update = self.update
        self.saved_at = (self.run / "latest.pt").stat().st_mtime
        self.last_save = time.monotonic()

    def learn(self, data: Demonstrations):
        cfg = self.cfg
        atomic_json(self.run / "imitation-dataset.json", data.summary)
        self.check_stop()

        def inspecting(done, total):
            self.report("validating", done, total, {"stage": "Reading demonstrations"})
            self.check_stop()

        reference = predict(self.net, data, np.arange(len(data.rows)), cfg.batch, self.device, inspecting)
        before = metrics(reference[data.validation], data.labels[data.validation], data.weights)
        reference_net = copy.deepcopy(self.net).eval().requires_grad_(False)
        weights = torch.from_numpy(data.weights).to(self.device)
        self.net.train()
        loss_total = 0.
        unroll = min(cfg.sequence_unroll, cfg.batch, max(len(s) for s in data.segments
                     if np.isin(s, data.training).all()))
        starts = data.sequence_starts(unroll)
        sequences = math.ceil(cfg.batch / unroll)
        for step in range(cfg.steps_per_epoch):
            self.check_stop()
            selected = self.rng.choice(starts, size=sequences, replace=True)
            window = data.sequence_batch(selected, cfg.sequence_burn_in, unroll, self.device)
            learn = window["learn"][:, cfg.sequence_burn_in:].clone()
            # Keep --batch as the maximum number of learned decisions, not sequences.
            valid = learn.flatten().nonzero().flatten()
            if len(valid) > cfg.batch:
                learn.flatten()[valid[cfg.batch:]] = False
            logits, value, auxiliary, ownership, _ = sequence_outputs(self.net, window, cfg.sequence_burn_in)
            with torch.no_grad():
                _, old_value, old_auxiliary, old_ownership, _ = sequence_outputs(
                    reference_net, window, cfg.sequence_burn_in)
            logp = F.log_softmax(logits[learn], -1)
            labels = window["labels"][:, cfg.sequence_burn_in:][learn]
            indices = window["indices"][:, cfg.sequence_burn_in:][learn.cpu().numpy()]
            target = torch.from_numpy(reference[indices]).to(self.device)
            prediction_loss = (F.mse_loss(value[learn], old_value[learn])
                               + F.mse_loss(auxiliary[learn], old_auxiliary[learn])
                               + F.mse_loss(ownership[learn], old_ownership[learn]))
            loss = (F.nll_loss(logp, labels, weight=weights)
                    + cfg.anchor * F.kl_div(logp, target, reduction="batchmean", log_target=True)
                    + cfg.prediction_anchor * prediction_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("Imitation loss became non-finite; the saved checkpoint is retained")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1., error_if_nonfinite=True)
            # Signals request a stop; they do not interrupt this update halfway through.
            self.optimizer.step()
            self.update += 1
            self.steps += 1
            self.samples += len(indices)
            self.trained[0] += len(indices)
            self.last_method = "imitation"
            loss_total += float(loss.detach())
            if time.monotonic() - self.last_save >= cfg.save_seconds:
                self.save()
            self.report("learning", step + 1, cfg.steps_per_epoch,
                        {"stage": "Learning from bots", "loss": float(loss.detach()),
                         "prediction_loss": float(prediction_loss.detach()),
                         "sequence_burn_in": cfg.sequence_burn_in, "sequence_unroll": unroll})
        self.cycles += 1
        # Save completed learning before any potentially slow or interrupted validation.
        self.save()

        def validating(done, total):
            self.report("validating", done, total, {"stage": "Checking learned moves"})
            self.check_stop()

        logp = predict(self.net, data, data.validation, cfg.batch, self.device, validating)
        self.validation = metrics(logp, data.labels[data.validation], data.weights)
        self.validation_update = self.update
        result = {"update": self.update, "imitation_steps": self.steps, "cycle": self.cycles,
                  "loss": loss_total / cfg.steps_per_epoch, "before": before, "validation": self.validation,
                  "time": time.time()}
        with (self.run / "imitation-metrics.jsonl").open("a") as f:
            f.write(json.dumps(result, allow_nan=False) + "\n")
        # Validation describes these exact saved weights, so persist its association too.
        self.last_saved_update = -1
        self.save()
        self.report("cycle_complete", 1, 1, {"stage": "Cycle saved"})
        return result

    def close(self, phase="stopped"):
        if self.closed:
            return
        try:
            self.save()
            self.report(phase, info={"stage": "Saved Blacklight" if phase == "stopped" else "Training failed"})
        finally:
            if self.pid_path.exists() and self.pid_path.read_text().strip() == str(os.getpid()):
                self.pid_path.unlink(missing_ok=True)
            self.lock.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        # Catching cancellation here still saves every fully completed optimizer update.
        self.close("stopped" if exc_type in (None, StopTraining, KeyboardInterrupt) else "failed")
        return exc_type in (StopTraining, KeyboardInterrupt)


def collect(run: Path, rounds: int, size: float, seed: int, progress=None,
            bot_pool: str = "stock") -> list[Path]:
    """Replace one bounded set of demonstrations; do not create a new model or experiment."""
    if bot_pool not in ("stock", "diverse"):
        raise ValueError("bot_pool must be stock or diverse")
    if bot_pool == "diverse":
        from .bot_profiles import schedule
        matches = schedule(rounds, seed)
    else:
        matches = [None] * rounds
    rng = np.random.default_rng(seed)
    directory = run / "imitation-data"
    directory.mkdir(parents=True, exist_ok=True)
    paths, configurations = [], []
    decisions = 0
    for i in range(rounds):
        arena_size = float(size + rng.uniform(-.6, .6))
        match = matches[i]
        opponents = len(match.opponents) if match is not None else 3 if i % 3 == 2 else 1
        profiles = ({"teacher_profile": match.teacher, "opponent_profiles": match.opponents}
                    if match is not None else {})
        stage = f"Recording {match.teacher}" if match is not None else "Recording strong bots"

        def collecting(info):
            if progress:
                progress("collecting", i, rounds, {"decisions": decisions + info["labels"],
                                                   "stage": stage, **profiles})

        recording = record(rounds=1, slots=1, builtin_ais=opponents, size=arena_size, seed=seed,
                           max_ticks=12002, workdir=run / "imitation-engine", teacher_iq=100,
                           base_port=48110, progress=collecting, **profiles)
        path = directory / f"round-{i:04}.npz"
        save(recording, path)
        decisions += sum(r.metadata["round"]["teacher_labels"] for r in recording)
        paths.append(path)
        configurations.append({"file": str(path.resolve()), "size": arena_size, "builtin_opponents": opponents})
        if match is not None:
            configurations[-1].update(teacher_profile=match.teacher, opponent_profiles=list(match.opponents))
        if progress:
            progress("collecting", i + 1, rounds, {"decisions": decisions, "stage": stage, **profiles})
    # Drop surplus examples if the user reduced --rounds; the cache remains bounded.
    for old in directory.glob("round-*.npz"):
        if old not in paths:
            old.unlink()
    atomic_json(run / "imitation-collection.json", {"rounds": configurations, "decisions": decisions,
                                                   "seed": seed, "bot_pool": bot_pool,
                                                   "native_rng_seeded": False})
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=PROJECT / "runs/blacklight", help="shared Blacklight run directory")
    ap.add_argument("--data", type=Path, nargs="+", help="use existing maximum-IQ demonstrations")
    ap.add_argument("--rounds", type=int, default=32, help="new teacher lives per learning cycle")
    ap.add_argument("--size", type=float, default=-3)
    ap.add_argument("--epochs", type=int, default=0, help="additional learning cycles; 0 trains until stopped")
    ap.add_argument("--steps-per-epoch", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--anchor", type=float, default=.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default=Config.device)
    ap.add_argument("--save-seconds", type=float, default=60)
    ap.add_argument("--burn-in", type=int, default=16)
    ap.add_argument("--unroll", type=int, default=32)
    ap.add_argument("--prediction-anchor", type=float, default=1.)
    ap.add_argument("--bot-pool", choices=("stock", "diverse"), default=Config.bot_pool,
                    help="stock teachers or candidate varied close-combat/native profiles")
    args = ap.parse_args()
    cfg = Config(args.epochs, args.steps_per_epoch, args.batch, args.lr, args.anchor, args.seed,
                 device=args.device, save_seconds=args.save_seconds, sequence_burn_in=args.burn_in,
                 sequence_unroll=args.unroll, prediction_anchor=args.prediction_anchor, bot_pool=args.bot_pool)
    try:
        cfg.validate()
        if args.rounds < 3 or not math.isfinite(args.size):
            raise ValueError("need at least three teacher lives and a finite arena size")
    except ValueError as exc:
        ap.error(str(exc))
    torch.set_num_threads(1)
    # Resolve relative paths the same way as RL and the bare control commands.
    run = (PROJECT / args.run).resolve()
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    console = Console(highlight=False)
    trainer = None
    try:
        with progress_bar(console) as bars:
            task = add_task(bars)

            def progress(phase, done, total, info):
                update_bar(bars, task, stage=info.get("stage", phase), update=info["update"],
                           completed=done, total=total, elapsed=time.time() - info["started_at"],
                           saved_update=info.get("saved_update"))

            with ImitationTrainer(run, cfg, progress, lambda: stopping) as trainer:
                console.print(f"Resumed Blacklight at update {trainer.update:,} · {trainer.steps:,} imitation updates saved")
                console.print("Training continues until stop or Ctrl-C." if not cfg.epochs
                              else f"Training {cfg.epochs} additional learning cycles.")
                console.print("Checkpoint: runs/blacklight/latest.pt" if run == (PROJECT / "runs/blacklight").resolve()
                              else f"Checkpoint: {run / 'latest.pt'}", style="#8b949e")
                cycle = 0
                while not cfg.epochs or cycle < cfg.epochs:
                    trainer.check_stop()

                    def collecting(phase, done, total, info):
                        trainer.report(phase, done, total, info)
                        trainer.check_stop()

                    paths = args.data or collect(run, args.rounds, args.size,
                                                 cfg.seed + trainer.cycles, collecting, bot_pool=cfg.bot_pool)
                    data = Demonstrations(paths, seed=cfg.seed, validation_fraction=cfg.validation_fraction)
                    trainer.learn(data)
                    del data
                    cycle += 1
            console.print(f"Stopped. Blacklight saved at update {trainer.update:,}.")
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console.print(str(exc), style="red", markup=False)
        raise SystemExit(1) from None
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
