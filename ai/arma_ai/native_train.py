"""Continue Blacklight with reinforcement learning against native scripted bots.

One neural player per real-engine match. Opponents are the six configured custom
profiles, never neural copies or native teacher labels. Complete lives provide
bounded outcome returns. Collected policies stay fixed until all matches finish.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import signal
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import protocol as P
from .bot_profiles import TRAINING_PROFILES, make_roster
from .engine import ArenaConfig, EnginePool
from .imitate import RunLock, atomic_checkpoint, atomic_json
from .model import make_net, net_version
from .play import PolicyHistory
from .teach import DutyPacer, _synchronize_device

PROJECT = Path(__file__).resolve().parent.parent
OBJECTIVE = "native_bot_ppo_v1"
OPPONENTS = tuple(name for name in TRAINING_PROFILES if not name.startswith("stock_"))


@dataclass
class Config:
    run: str = "runs/blacklight"
    mode: str = "background"
    engines: int = 2
    lr: float = 1e-5
    gamma: float = .997
    policy_clip: float = .2
    entropy_coef: float = .01
    value_coef: float = .5
    sequence_unroll: int = 16
    sequence_burn_in: int = 16
    epochs: int = 1
    max_updates: int = 0
    round_time_limit: float = 120
    duty_cycle: float = .35
    save_seconds: float = 60
    snapshot_every: int = 1000
    torch_threads: int = 2
    seed: int = 17
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    def validate(self):
        if self.mode not in ("background", "full"):
            raise ValueError("mode must be background or full")
        if not 1 <= self.engines <= 16:
            raise ValueError("engines must be between 1 and 16")
        if min(self.epochs, self.sequence_unroll, self.torch_threads) < 1:
            raise ValueError("epochs, sequence_unroll and torch_threads must be positive")
        if self.sequence_burn_in < 0 or self.max_updates < 0 or self.snapshot_every < 0:
            raise ValueError("burn-in and update/snapshot limits must be nonnegative")
        for name in ("lr", "round_time_limit", "save_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not math.isfinite(self.policy_clip) or not 0 < self.policy_clip < 1:
            raise ValueError("policy_clip must be in (0, 1)")
        if any(not math.isfinite(v) or v < 0 for v in (self.entropy_coef, self.value_coef)):
            raise ValueError("loss coefficients must be finite and nonnegative")
        DutyPacer(self.duty_cycle)


@dataclass
class Experience:
    maps: np.ndarray
    scalars: np.ndarray
    mask: np.ndarray
    state: np.ndarray
    context: np.ndarray
    action: int
    logp: float
    value: float


@dataclass
class Episode:
    opponents: tuple[str, ...]
    size: float
    steps: list[Experience] = field(default_factory=list)
    round_id: int | None = None
    result: str | None = None
    seconds: float = 0
    kills: int = 0


def outcome(step) -> str | None:
    slot = step.slots[0]
    if slot.flags & P.FLAG_WON:
        return "win"
    if slot.flags & P.FLAG_DIED:
        # The native round timer kills surviving cycles simultaneously. A timed-out
        # life is distinguished in collect(); ordinary mutual deaths count as losses.
        return "loss"
    if step.round_over:
        return "draw" if slot.alive else "loss"
    return None


def returns(result: str, length: int, gamma: float) -> np.ndarray:
    if result not in ("win", "loss", "draw") or length < 0 or not 0 < gamma <= 1:
        raise ValueError("Expected a completed result, nonnegative length and valid discount")
    reward = {"win": 1., "loss": -1., "draw": 0.}[result]
    return (reward * np.power(gamma, np.arange(length - 1, -1, -1))).astype(np.float32)


class NativeTrainer:
    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.run = (PROJECT / cfg.run).resolve()
        self.lock = RunLock(self.run)
        self.closed = False
        self.stop = threading.Event()
        self.started = time.time()
        self.last_heartbeat = 0.
        self.last_save = time.monotonic()
        self.pacer = DutyPacer(cfg.duty_cycle)
        self.results = deque(maxlen=200)
        self.profile_counts = Counter()
        self.session_decisions = 0
        self.completed = self.total = 0
        self.phase = "starting"
        self.stage = "Loading Blacklight for native bot RL"
        self.losses = {}
        self.batch_number = 0
        self.profile_cursor = 0
        self.aborted_games = 0
        self.dirty = False
        try:
            path = self.run / "latest.pt"
            if not path.exists():
                raise FileNotFoundError(f"No saved Blacklight at {path}; native RL requires its existing brain")
            self.ck = torch.load(path, map_location="cpu", weights_only=True)
            if self.ck.get("trainer") != "teach":
                raise ValueError("Expected the canonical Blacklight training checkpoint")
            torch.set_num_threads(cfg.torch_threads)
            torch.manual_seed(cfg.seed)
            self.device = torch.device(cfg.device)
            self.net = make_net(net_version(self.ck["model"])).to(self.device)
            self.net.load_state_dict(self.ck["model"])
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr,
                                              weight_decay=1e-4, eps=1e-6)
            resumed = (self.ck.get("rl_objective") == OBJECTIVE
                       and self.ck.get("last_training_method") == "native_rl")
            if resumed:
                self.optimizer.load_state_dict(self.ck["opt"])
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg.lr
            state = self.ck.get("native_rl_state", {})
            self.rng = np.random.default_rng(cfg.seed)
            if resumed and state.get("rng_state"):
                self.rng.bit_generator.state = json.loads(state["rng_state"])
            if resumed and isinstance(state.get("torch_rng_state"), torch.Tensor):
                torch.set_rng_state(state["torch_rng_state"].cpu())
            self.profile_cursor = int(state.get("profile_cursor", 0))
            self.native_games = int(state.get("games", 0))
            self.native_decisions = int(state.get("decisions", 0))
            self.update = self.start_update = int(self.ck.get("update", 0))
            self.decisions = int(self.ck.get("decisions", 0))
            self.games = int(self.ck.get("games", 0))
            self.trained = list(self.ck.get("trained", [0, 0]))
            self.generated = list(self.ck.get("generated", [0, 0]))
            self.saved_update = self.update
            self.saved_at = path.stat().st_mtime
            self.snapshot_mark = self.update // cfg.snapshot_every if cfg.snapshot_every else 0
            self.session_games = 0
            if not resumed:
                folder = self.run / "snapshots"
                folder.mkdir(exist_ok=True)
                backup = folder / f"before-native-bots-u{self.update:08d}.pt"
                if not backup.exists():
                    shutil.copy2(path, backup)
            self.report(force=True)
        except BaseException:
            self.lock.close()
            raise

    def report(self, *, force=False):
        if not force and time.time() - self.last_heartbeat < 1:
            return
        self.last_heartbeat = time.time()
        counts = Counter(self.results)
        self.progress = {
            "trainer": "native_rl", "run": str(self.run), "pid": os.getpid(),
            "started_at": self.started, "time": self.last_heartbeat,
            "phase": self.phase, "stage": self.stage,
            "completed": self.completed, "total": self.total,
            "update": self.update, "session_updates": self.update - self.start_update,
            "saved_update": self.saved_update, "saved_at": self.saved_at,
            "decisions": self.decisions, "steps": self.decisions, "games": self.games,
            "session_games": self.session_games, "native_games": self.native_games,
            "native_decisions": self.native_decisions, "trained": self.trained[0],
            "sps": self.session_decisions / max(time.time() - self.started, 1),
            "wins": counts["win"], "draws": counts["draw"], "losses": counts["loss"],
            "opponents": list(OPPONENTS), "profile_games": dict(self.profile_counts),
            "opponent_mode": "native_scripted_bots", "rl_objective": OBJECTIVE,
            "aborted_games": self.aborted_games, "config": asdict(self.cfg),
            "mode": self.cfg.mode, "actions": 6, "net": 4, **self.losses,
        }
        atomic_json(self.run / "progress.json", self.progress)

    def arenas(self):
        configs, episodes = [], []
        for _ in range(self.cfg.engines):
            primary = OPPONENTS[self.profile_cursor % len(OPPONENTS)]
            self.profile_cursor += 1
            opponents = [primary]
            if self.rng.random() < .25:
                opponents += self.rng.choice([n for n in OPPONENTS if n != primary], 2,
                                             replace=False).tolist()
            size = float(self.rng.choice([-3., -1.5, 0.]))
            roster = make_roster("fighter_rush", tuple(opponents))
            configs.append(ArenaConfig(
                slots=1, builtin_ais=len(opponents), ai_iq=100, size_factor=size,
                round_time_limit=self.cfg.round_time_limit, ai_characters=roster.text,
                controlled_profile=roster.controlled_name, builtin_profiles=roster.builtin_names))
            episodes.append(Episode(tuple(opponents), size))
        return configs, episodes

    def stopping(self):
        if (self.run / "trainctl.stop").exists():
            self.stop.set()
        return self.stop.is_set()

    @torch.inference_mode()
    def collect(self) -> list[Episode]:
        self.net.eval()
        history = PolicyHistory(self.net)
        configs, episodes = self.arenas()
        finished = set()
        self.phase, self.stage = "collecting", "Playing native bots"
        self.completed, self.total = 0, len(configs)
        self.report(force=True)
        # Close the engines at each batch boundary. No life crosses a policy update.
        with EnginePool(configs, workdir=self.run / "native-engines", base_port=49700) as pool:
            while len(finished) < len(configs) and not self.stopping():
                work_started = time.monotonic()
                ready = pool.poll(min_batch=len(finished) + 1, max_wait=.003, timeout=30)
                snapshots, requests = [], []
                for step in ready:
                    i, slot = step.engine, step.slots[0]
                    episode = episodes[i]
                    if episode.round_id is not None and episode.round_id != step.round_id:
                        self.aborted_games += 1
                        episode.steps.clear()
                        history.clear(i)
                    episode.round_id = step.round_id
                    if slot.teacher_action is not None:
                        raise RuntimeError("Native RL received teacher labels instead of its own actions")
                    result = outcome(step)
                    if result is not None:
                        if (result == "loss" and step.n_alive == 0
                                and step.round_time >= self.cfg.round_time_limit - .1):
                            result = "draw"
                        episode.result, episode.seconds = result, float(step.round_time)
                        episode.kills += slot.kills
                        self.results.append(result)
                        self.profile_counts.update(episode.opponents)
                        self.games += 1
                        self.native_games += 1
                        self.session_games += 1
                        finished.add(i)
                        self.completed = len(finished)
                        # Withhold acknowledgement: this engine stays blocked until the
                        # whole collection batch completes and is closed below.
                        continue
                    episode.kills += slot.kills
                    history.begin(step)
                    if slot.needs_action:
                        snapshots.append(history.snapshot(step, 0))
                        requests.append(step)
                    else:
                        history.finish(step, [0])
                        pool.act(i, [0])
                if requests:
                    maps = torch.from_numpy(np.stack([s.maps for s in snapshots])).to(self.device)
                    features = torch.from_numpy(np.stack([
                        np.concatenate([s.scalars, s.mask]) for s in snapshots])).to(self.device)
                    state = torch.cat([s.state for s in snapshots])
                    context = torch.cat([s.context for s in snapshots])
                    actions, logp, values, next_state = self.net.act_step(
                        maps, features, state, greedy=False, context=context)
                    actions, logp, values = actions.cpu().tolist(), logp.cpu().tolist(), values.cpu().tolist()
                    for j, (step, snapshot, action) in enumerate(zip(requests, snapshots, actions)):
                        if step.round_time > 1e-6:  # the engine ignores controls at time zero
                            episodes[step.engine].steps.append(Experience(
                                snapshot.maps, snapshot.scalars, snapshot.mask,
                                snapshot.state[0].cpu().numpy().copy(),
                                snapshot.context[0].cpu().numpy().copy(),
                                int(action), float(logp[j]), float(values[j])))
                            self.decisions += 1
                            self.native_decisions += 1
                            self.session_decisions += 1
                            self.generated[0] += 1
                        history.commit(step, 0, int(action), next_state[j:j + 1])
                        history.finish(step, [int(action)])
                        pool.act(step.engine, [int(action)])
                self.report()
                _synchronize_device(self.device)
                self.pacer.pause(work_started, self.stop)
        return [episode for episode in episodes if episode.result is not None and episode.steps]

    def step(self, episode: Episode, start: int, end: int, targets: np.ndarray):
        begin = max(0, start - self.cfg.sequence_burn_in)
        rows = episode.steps[begin:end]
        maps = torch.from_numpy(np.stack([x.maps for x in rows]))[None].to(self.device)
        scalars = torch.from_numpy(np.stack([x.scalars for x in rows]))[None].to(self.device)
        masks = torch.from_numpy(np.stack([x.mask for x in rows]))[None].to(self.device)
        context = torch.from_numpy(np.stack([x.context for x in rows]))[None].to(self.device)
        state = torch.from_numpy(rows[0].state.copy())[None].to(self.device)
        burn = start - begin
        if burn:
            with torch.no_grad():
                _, _, state = self.net.forward_sequence(
                    maps[:, :burn], scalars[:, :burn], masks[:, :burn], state,
                    context=context[:, :burn])
        logits, value, _ = self.net.forward_sequence(
            maps[:, burn:], scalars[:, burn:], masks[:, burn:], state.detach(),
            context=context[:, burn:])
        rows = episode.steps[start:end]
        actions = torch.tensor([x.action for x in rows], device=self.device)
        if not masks[0, burn:].gather(1, actions[:, None]).all():
            raise ValueError("Native experience contains an illegal action")
        logp = torch.log_softmax(logits[0].float(), -1)
        selected = logp.gather(1, actions[:, None]).squeeze(1)
        old = torch.tensor([x.logp for x in rows], device=self.device)
        target = torch.from_numpy(targets[start:end]).to(self.device)
        baseline = torch.tensor([x.value for x in rows], device=self.device)
        advantage = (target - baseline).detach()
        ratio = (selected - old).clamp(-20, 20).exp()
        pg = -torch.minimum(ratio * advantage,
                            ratio.clamp(1 - self.cfg.policy_clip, 1 + self.cfg.policy_clip) * advantage).mean()
        vf = F.mse_loss(value[0].float(), target)
        entropy = -(logp.exp() * logp).sum(-1).mean()
        loss = pg + self.cfg.value_coef * vf - self.cfg.entropy_coef * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("Native RL loss became nonfinite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.update += 1
        self.trained[0] += len(rows)
        self.dirty = True
        return {"pg": pg.item(), "vf": vf.item(), "ent": entropy.item(),
                "grad_norm": norm.item(), "policy_kl": ((ratio - 1) - (selected - old)).mean().item()}

    def learn(self, episodes):
        self.net.train()
        chunks = [(i, start, min(start + self.cfg.sequence_unroll, len(ep.steps)))
                  for i, ep in enumerate(episodes)
                  for start in range(0, len(ep.steps), self.cfg.sequence_unroll)]
        targets = [returns(ep.result, len(ep.steps), self.cfg.gamma) for ep in episodes]
        self.phase, self.stage = "learning", "Learning from bot-match outcomes"
        self.completed, self.total = 0, len(chunks) * self.cfg.epochs
        self.report(force=True)
        for _ in range(self.cfg.epochs):
            for index in self.rng.permutation(len(chunks)):
                if self.stopping() or self.at_limit():
                    return
                work_started = time.monotonic()
                i, start, end = chunks[index]
                self.losses = self.step(episodes[i], start, end, targets[i])
                self.completed += 1
                self.report()
                if time.monotonic() - self.last_save >= self.cfg.save_seconds:
                    self.save()
                _synchronize_device(self.device)
                self.pacer.pause(work_started, self.stop)

    def at_limit(self):
        return bool(self.cfg.max_updates and self.update >= self.cfg.max_updates)

    def save(self):
        if not self.dirty:
            return
        state = {"games": self.native_games, "decisions": self.native_decisions,
                 "profile_cursor": self.profile_cursor,
                 "rng_state": json.dumps(self.rng.bit_generator.state),
                 "torch_rng_state": torch.get_rng_state(),
                 "opponents": list(OPPONENTS), "head_supervision": "policy_and_outcome_value"}
        imitation = dict(self.ck.get("imitation_state", {}))
        imitation.pop("optimizer", None)
        checkpoint = {**self.ck, "trainer": "teach", "last_training_method": "native_rl",
                      "rl_objective": OBJECTIVE, "native_rl_state": state,
                      "model": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                      "opt": self.optimizer.state_dict(), "update": self.update,
                      "trained": self.trained, "generated": self.generated,
                      "decisions": self.decisions, "games": self.games,
                      "imitation_state": imitation, "config": asdict(self.cfg)}
        atomic_checkpoint(self.run / "latest.pt", checkpoint)
        self.ck = checkpoint
        self.saved_update = self.update
        self.saved_at = (self.run / "latest.pt").stat().st_mtime
        self.last_save = time.monotonic()
        self.dirty = False
        every = self.cfg.snapshot_every
        if every and self.update // every > self.snapshot_mark:
            self.snapshot_mark = self.update // every
            folder = self.run / "snapshots"
            folder.mkdir(exist_ok=True)
            shutil.copy2(self.run / "latest.pt", folder / f"update-{self.update:08d}.pt")

    def train(self):
        print(f"Native bot RL: resumed Blacklight at update {self.update:,}; "
              f"{self.cfg.engines} real engines, {self.device}, {self.cfg.mode}", flush=True)
        print("Opponents: " + ", ".join(OPPONENTS), flush=True)
        while not self.stopping() and not self.at_limit():
            episodes = self.collect()
            if self.stopping():
                break
            if not episodes:
                raise RuntimeError("Native games produced no complete learning experiences")
            self.learn(episodes)
            self.batch_number += 1
            counts = Counter(self.results)
            print(f"u{self.update} native bots {counts['win']}W/{counts['draw']}D/{counts['loss']}L "
                  f"games={self.session_games} policy={self.losses.get('pg', 0):.4f} "
                  f"value={self.losses.get('vf', 0):.4f}", flush=True)
            self.report(force=True)
            with (self.run / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(self.progress) + "\n")
            if time.monotonic() - self.last_save >= self.cfg.save_seconds:
                self.save()
        self.save()

    def close(self, *, failed=False):
        if self.closed:
            return
        try:
            if not failed:
                self.save()
            self.phase = "failed" if failed else "stopped"
            self.stage = "Training failed; see train.log" if failed else "Blacklight saved"
            self.report(force=True)
        finally:
            self.closed = True
            self.lock.close()


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("background", "full"), default="background")
    parser.add_argument("--run", default="runs/blacklight")
    parser.add_argument("--engines", type=int)
    parser.add_argument("--duty-cycle", type=float)
    for name, kind, default in (
        ("lr", float, 1e-5), ("gamma", float, .997), ("policy-clip", float, .2),
        ("entropy-coef", float, .01), ("value-coef", float, .5),
        ("sequence-unroll", int, 16), ("sequence-burn-in", int, 16),
        ("epochs", int, 1), ("max-updates", int, 0), ("round-time-limit", float, 120),
        ("save-seconds", float, 60), ("snapshot-every", int, 1000),
        ("torch-threads", int, 2), ("seed", int, 17),
        ("device", str, "mps" if torch.backends.mps.is_available() else "cpu"),
    ):
        parser.add_argument("--" + name, type=kind, default=default)
    args = vars(parser.parse_args(argv))
    if args["engines"] is None:
        args["engines"] = 2 if args["mode"] == "background" else 8
    if args["duty_cycle"] is None:
        args["duty_cycle"] = .35 if args["mode"] == "background" else 1.
    cfg = Config(**args)
    cfg.validate()
    return cfg


def main():
    trainer = NativeTrainer(parse_config())
    handlers = {sig: signal.signal(sig, lambda *_: trainer.stop.set())
                for sig in (signal.SIGINT, signal.SIGTERM)}
    failed = True
    try:
        trainer.train()
        failed = False
    finally:
        trainer.close(failed=failed)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
