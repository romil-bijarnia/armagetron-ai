"""Search-guided self-play and RL updates to the canonical V4 checkpoint.

    arma-teach                         # train runs/blacklight, resuming where it stopped
    arma-teach --games 128 --sims 24   # lighter

How it works:

* Hundreds of rounds run at once in the C++ simulator (ai/sim), which reproduces the engine's
  physics and the network's view of the arena, checked against recorded engine rounds. On a quarter of the decisions every player
  searches ahead (Gumbel tree search over 0.1 s steps, guided by the network), and the search's
  improved move probabilities become the training target. Every decision also gets a discounted
  n-step value target (or the outcome if the round ends first), and targets for territory, danger, and a
  credited kill while staying alive over the next two seconds.
* A quarter of duels begin in varied close encounters, alongside ordinary full-arena games.
  The space fighter's search replies follow its known control until its next replan, where the
  network value bootstraps the continuation. Network opponents retain full tree search.
* Most rounds are the network against itself: duels on three arena sizes and four-player
  free-for-alls. An exploiter trains against a frozen current-policy target and is refreshed
  after it solves that target or exhausts its update budget.
* V4 guides its own search from the start, exploring all six steering/brake actions.

Two processes share the GPU: the actor (self-play) and the learner (this one), connected by
shared-memory replay rings. latest.pt is the single persistent checkpoint. Actor, exploiter and
target weight files are temporary exchanges for the running workers; metrics.jsonl records reports.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import multiprocessing as mp
import os
import queue
import random
import signal
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import protocol as P
from .curriculum import combat_start
from .model import make_net, net_version
from .progress_state import replay_requirements
from .sim import AGENT_HUNTER
from .training_watch import TrainingPublisher, match_results

PROJECT = Path(__file__).resolve().parent.parent

# (SIZE_FACTOR, players, share of the self-play rounds)
ARENAS = ((-3.0, 2, 0.30), (-1.5, 2, 0.25), (0.0, 2, 0.15), (-1.0, 4, 0.20), (0.0, 4, 0.10))
DUEL_SIZES = (-3.0, -1.5, 0.0)
AGENT_MAIN, AGENT_EXPLOITER, AGENT_TARGET = 0, 1, 2
FULL_SIM_THREADS = max(1, os.cpu_count() or 8)


@dataclass
class Config:
    run: str = "runs/blacklight"
    mode: str = "full"  # CLI presets: full or background; explicit flags always take precedence
    games: int = 256  # rounds played at once
    sims: int = 24  # simulations per searched decision
    macro: int = 2  # decisions per search step (0.1 s, the turn delay)
    search_prob: float = 0.25
    gumbel: float = 1.0
    temperature: float = 1.0
    threads: int = FULL_SIM_THREADS
    torch_threads: int = 2  # CPU tensor workers per actor/learner; simulator workers are separate
    duty_cycle: float = 1.0  # work / (work + deliberate rest), independently for actor and learner
    ring: int = 200_000  # the main network's replay window, in decisions
    exploiter_ring: int = 60_000
    batch: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-4
    value_coef: float = 1.0
    aux_coef: float = 0.25
    own_coef: float = 0.5  # weight of the ownership map loss (who holds each cell two seconds on)
    hunter_prob: float = 0.20  # rounds against the simulator's space fighter (a minimax over territory)
    hunter_depth: int = 2  # how many plies it looks ahead
    combat_prob: float = 0.25  # fraction of duels starting in a varied close encounter
    max_grad_norm: float = 1.0
    bf16: bool = True  # train in bfloat16 (about 20% faster on Apple GPUs)
    replay_ratio: float = 1.5  # training samples per generated one (the rest of the GPU goes to self-play)
    sequence_burn_in: int = 16
    sequence_unroll: int = 32
    warmup: int = 20_000  # decisions in the ring before learning starts
    publish_seconds: float = 20  # the actor (and the brain page) get new weights this often
    save_seconds: float = 300  # full checkpoint (both networks and optimisers)
    log_every: int = 25
    max_updates: int = 0  # total main-network update cap (0 runs until stopped), including resumed updates
    exploiter_prob: float = 0.20  # rounds played by the exploiter
    exploiter_win: float = 0.60  # refresh the exploiter once it wins this often...
    exploiter_games: int = 400  # ... over this many of its recent rounds
    exploiter_patience: int = 30_000  # or after this many of its updates, whichever is first
    value_steps: int = 60  # a sample's value target: the value estimate 3 s later (or the outcome, if sooner)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    seed: int = 0
    resume: bool = True


MODE_DEFAULTS = {
    "full": {"games": 256, "threads": FULL_SIM_THREADS, "batch": 512, "torch_threads": 2, "duty_cycle": 1.0},
    "background": {"games": 96, "threads": 4, "batch": 256, "torch_threads": 2, "duty_cycle": 0.35},
}


class DutyPacer:
    """Rest after measured work, without accumulating credit for a later burst.

    Each process has its own budget. This is deliberately a work/rest target, not a claim
    about GPU utilisation or temperature. The caller synchronises pending accelerator work
    before measuring its duration; stop.wait makes even a long rest immediately cancellable.
    """

    def __init__(self, duty_cycle: float, clock=time.monotonic):
        if not math.isfinite(duty_cycle) or not 0 < duty_cycle <= 1:
            raise ValueError("duty_cycle must be greater than zero and at most one")
        self.duty_cycle = duty_cycle
        self.clock = clock
        self.rest_seconds = 0.0

    def pause(self, started: float, stop) -> None:
        if self.duty_cycle == 1:
            return
        now = self.clock()
        work = max(0.0, now - started)
        delay = work * (1 / self.duty_cycle - 1)
        if delay:
            stop.wait(delay)
            self.rest_seconds += max(0.0, self.clock() - now)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _half(sd: dict) -> dict:
    """Weights in half precision: half the disk, and what the actor runs anyway."""
    return {k: v.detach().to("cpu", torch.float16) if v.is_floating_point() else v.detach().cpu() for k, v in sd.items()}


def _load_net(path: Path, device: torch.device, half: bool = False):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"]
    net = make_net(net_version(sd))
    net.load_state_dict(sd)
    net = net.to(device).eval()
    if half:
        net = net.half()
    return net, ck


def _has_training_history(run_dir: Path) -> bool:
    """A missing canonical checkpoint must not silently reset an established run."""
    def trained(record) -> bool:
        update = record.get("update") if isinstance(record, dict) else None
        return isinstance(update, (int, float)) and not isinstance(update, bool) and update > 0

    try:
        if trained(json.loads((run_dir / "progress.json").read_text())):
            return True
    except (OSError, ValueError):
        pass
    try:
        with (run_dir / "metrics.jsonl").open() as metrics:
            for line in metrics:
                try:
                    if trained(json.loads(line)):
                        return True
                except ValueError:
                    continue
    except OSError:
        pass
    return False


# ====================================================================== actor
class Actor:
    """Plays self-play, space-fighter and exploiter rounds with the current networks."""

    def __init__(self, cfg: Config, run_dir: Path, rings, out_q):
        from .sim import SelfPlay
        self.cfg = cfg
        self.run_dir = run_dir
        self.out_q = out_q
        self.device = torch.device(cfg.device)
        self.half = self.device.type == "mps"
        self.rng = random.Random(cfg.seed + 17)
        self.sp = SelfPlay(cfg.games, sims=cfg.sims, macro=cfg.macro, search_prob=cfg.search_prob,
                           temperature=cfg.temperature, gumbel=cfg.gumbel, threads=cfg.threads,
                           value_steps=cfg.value_steps, hunter_depth=cfg.hunter_depth, recurrent=True)
        self.sp.set_ring(AGENT_MAIN, rings[0])
        self.sp.set_ring(AGENT_EXPLOITER, rings[1])
        self.nets: dict[int, torch.nn.Module] = {}
        self.model_updates: dict[int, int] = {}
        self.model_generations: dict[int, int] = {}
        self.game_models: dict[int, dict] = {}
        self.mtimes: dict[tuple, float] = {}
        self.exploiter_gen = -1
        self.exploiter_results: deque = deque(maxlen=cfg.exploiter_games)
        self.promotion_sent = False
        self.kind: dict[int, tuple] = {}
        self.start_kinds: dict[int, str] = {}
        self.round_ids = defaultdict(int)
        self.watch = TrainingPublisher(run_dir, cfg.games)
        self.seed = cfg.seed * 1_000_003
        self.results = defaultdict(list)
        self.lengths: list = []
        self.pacer = DutyPacer(cfg.duty_cycle)

    # ---------------------------------------------------------------- networks
    def _file(self, agent: int) -> Path:
        return {AGENT_MAIN: self.run_dir / "actor.pt", AGENT_EXPLOITER: self.run_dir / "exploiter.pt",
                AGENT_TARGET: self.run_dir / "target.pt"}[agent]

    def reload(self) -> None:
        for agent in (AGENT_MAIN, AGENT_EXPLOITER, AGENT_TARGET):
            f = self._file(agent)
            try:
                m = f.stat().st_mtime
            except FileNotFoundError:
                continue
            key = (agent, str(f))
            if self.mtimes.get(key) == m:
                continue
            try:
                net, ck = _load_net(f, self.device, self.half)
            except Exception:
                continue  # being replaced; next time
            self.nets[agent] = net
            self.model_updates[agent] = int(ck.get("update", 0))
            self.model_generations[agent] = int(ck.get("generation", 0))
            self.mtimes[key] = m
            if agent == AGENT_EXPLOITER:
                gen = int(ck.get("generation", 0))
                if gen != self.exploiter_gen:
                    self.exploiter_gen = gen
                    self.exploiter_results.clear()
                    self.promotion_sent = False

    @torch.no_grad()
    def evaluate(self, n: int) -> None:
        sp = self.sp
        logits = np.full((n, P.N_ACTIONS), -1e8, np.float32)
        values = np.zeros(n, np.float32)
        updates = np.full(n, -1, np.int64)
        states = np.empty((n, 256), np.float32)
        groups = defaultdict(list)
        for index in range(n):
            game, agent, length = int(sp.game_ids[index]), int(sp.agents[index]), int(sp.lengths[index])
            net, update, _ = self.game_models[game][agent]
            groups[(net, update, length)].append(index)
        dtype = torch.float16 if self.half else torch.float32
        for (net, update, length), indices in groups.items():
            idx = np.asarray(indices)
            maps = torch.from_numpy(sp.sequence_maps[idx, :length]).to(self.device)
            scalars = torch.from_numpy(sp.sequence_scalars[idx, :length]).to(self.device, dtype)
            bits = (sp.sequence_masks[idx, :length, None] >> np.arange(P.N_ACTIONS) & 1).astype(bool)
            masks = torch.from_numpy(bits).to(self.device)
            initial = torch.from_numpy(sp.states[idx]).to(self.device, dtype)
            context = torch.from_numpy(sp.sequence_contexts[idx, :length]).to(self.device, dtype)
            lg, value, hidden = net.forward_sequence(maps, scalars, masks, initial, context=context)
            logits[idx] = lg[:, -1].float().cpu().numpy()
            values[idx] = value[:, -1].float().cpu().numpy()
            states[idx] = hidden.float().cpu().numpy()
            updates[idx] = update
        sp.feed(logits, values, policy_updates=updates, next_states=states)

    # ---------------------------------------------------------------- matchups
    def start(self, g: int) -> None:
        cfg, r = self.cfg, self.rng
        self.seed += 1
        x = r.random()
        if (x < cfg.exploiter_prob and AGENT_EXPLOITER in self.nets and AGENT_TARGET in self.nets
                and self.model_generations[AGENT_EXPLOITER] == self.model_generations[AGENT_TARGET]):
            size = r.choice(DUEL_SIZES)
            agents, learn, kind = [AGENT_EXPLOITER, AGENT_TARGET], [1, 0], ("exploiter", None)
        elif cfg.exploiter_prob <= x < cfg.exploiter_prob + cfg.hunter_prob:
            # the space fighter: it plays for territory and punishes hesitation, from the first round on
            size = r.choice(DUEL_SIZES)
            agents, learn, kind = [AGENT_MAIN, AGENT_HUNTER], [1, 0], ("hunter", None)
        else:
            size, n, _ = r.choices(ARENAS, [w for _, _, w in ARENAS])[0]
            agents, learn, kind = [AGENT_MAIN] * n, [1] * n, ("self", f"{'duel' if n == 2 else f'ffa{n}'}_{size:g}")
        # who gets which spawn point is random; so is the order in the engine's object list
        order = list(range(len(agents)))
        r.shuffle(order)
        agents = [agents[i] for i in order]
        learn = [learn[i] for i in order]
        hero = order.index(0)
        self.kind[g] = (kind, hero, agents)
        self.round_ids[g] += 1
        rotate = r.randrange(len(agents))
        placements = None
        self.start_kinds[g] = "standard"
        if len(agents) == 2 and cfg.combat_prob > 0 and r.random() < cfg.combat_prob:
            self.start_kinds[g], placements = combat_start(size, r)
        # A life and all of its search branches keep one immutable weight generation.
        # Old in-memory modules disappear when their last game finishes; no disk snapshots.
        self.game_models[g] = {agent: (self.nets[agent], self.model_updates[agent], self.model_generations[agent])
                               for agent in set(agents) if agent >= 0}
        generations = [self.game_models[g][agent][2] if agent >= 0 else 0 for agent in agents]
        self.sp.start(g, size, agents, learn, rotate=rotate, seed=self.seed,
                      placements=placements, generations=generations)

    def finished(self, g: int, info: dict) -> None:
        (kind, name), hero, _ = self.kind[g]
        u = info["outcome"][hero]
        score = 1.0 if u >= 0.999 else (0.5 if abs(u) < 1e-6 else 0.0)
        self.lengths.append(info["decisions"])
        if kind == "self":
            self.results[f"len_{name}"].append(info["decisions"])
        elif kind == "hunter":
            self.results["win_vs_hunter"].append(score)
        elif kind == "exploiter":
            if self.game_models[g][AGENT_EXPLOITER][2] != self.exploiter_gen:
                return  # an older pinned helper cannot promote the current generation
            self.exploiter_results.append(score)
            self.results["win_exploiter"].append(score)
            if (not self.promotion_sent and len(self.exploiter_results) == self.exploiter_results.maxlen
                    and np.mean(self.exploiter_results) >= self.cfg.exploiter_win):
                self.out_q.put(("promote", {"win_rate": float(np.mean(self.exploiter_results)),
                                            "generation": self.exploiter_gen}))
                self.promotion_sent = True

    def run(self, stop) -> None:
        parent = os.getppid()
        while AGENT_MAIN not in self.nets:
            self.reload()
            if AGENT_MAIN not in self.nets:
                time.sleep(0.5)
            if stop.is_set() or os.getppid() != parent:
                return
        for g in range(self.cfg.games):
            self.start(g)
        self.watch.publish(self, force=True)
        last_report = last_reload = time.time()
        evals = 0
        while not stop.is_set() and os.getppid() == parent:
            work_started = time.monotonic()
            n = self.sp.collect()
            if n:
                self.evaluate(n)
                evals += int(self.sp.lengths[:n].sum())
            self.watch.publish(self)
            for g in range(self.cfg.games):
                info = self.sp.info(g)
                if info is not None:
                    if g == self.watch.arena:
                        self.watch.publish(self, force=True)
                    self.finished(g, info)
                    self.start(g)
            now = time.time()
            if now - last_reload > 2.0:
                self.reload()
                last_reload = now
            if now - last_report > 5.0:
                st = self.sp.stats()
                st.update(evals=evals, seconds=now - last_report, results=dict(self.results), lengths=self.lengths,
                          throttle_seconds=self.pacer.rest_seconds)
                self.out_q.put(("stats", st))
                self.results = defaultdict(list)
                self.lengths = []
                evals = 0
                last_report = now
            if self.cfg.duty_cycle < 1:
                _synchronize_device(self.device)
                self.pacer.pause(work_started, stop)
        # A bounded architecture check can stop before the periodic report; preserve those
        # real decisions and completed games in the final checkpoint too.
        st = self.sp.stats()
        st.update(evals=evals, seconds=max(time.time() - last_report, 1e-6), results=dict(self.results),
                  lengths=self.lengths,
                  throttle_seconds=self.pacer.rest_seconds)
        self.out_q.put(("stats", st))
        self.watch.publish(self, status="stopped", force=True)


def actor_main(cfg_dict: dict, run_dir: str, ring_names: list, ring_caps: list, out_q, stop) -> None:
    from .sim import Ring
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cfg = Config(**cfg_dict)
    torch.set_num_threads(cfg.torch_threads)
    rings = [Ring(c, shm_name=n) for n, c in zip(ring_names, ring_caps)]
    try:
        Actor(cfg, Path(run_dir), rings, out_q).run(stop)
    finally:
        for r in rings:
            r.close()


# ====================================================================== learner
class Learner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.started_at = time.time()
        self.pacer = DutyPacer(cfg.duty_cycle)
        if cfg.mode not in MODE_DEFAULTS:
            raise ValueError("mode must be background or full")
        if cfg.torch_threads < 1 or cfg.threads < 1:
            raise ValueError("torch_threads and threads must be positive")
        torch.set_num_threads(cfg.torch_threads)
        self.actor_throttle_seconds = 0.0
        if cfg.sequence_burn_in < 0 or cfg.sequence_unroll < 2 or cfg.batch < 2:
            raise ValueError("sequence_burn_in must be nonnegative; sequence_unroll and batch must be at least two")
        if cfg.max_updates < 0:
            raise ValueError("max_updates must be zero (unlimited) or a positive total update cap")
        if not math.isfinite(cfg.combat_prob) or not 0 <= cfg.combat_prob <= 1:
            raise ValueError("combat_prob must be between zero and one")
        # Ring.window reserves the oldest 4096 slots while its writer is active. A smaller
        # window can never produce the ready samples batch() requires, regardless of runtime.
        read_margin = 4096
        _, ready_needed, _ = replay_requirements(cfg.warmup, cfg.batch)
        ready_needed = max(1, ready_needed)
        for name in ("ring", "exploiter_ring"):
            cap = getattr(cfg, name)
            if cap <= read_margin:
                raise ValueError(f"{name} must exceed the {read_margin} replay read safety margin; got {cap}")
            if cap - read_margin < ready_needed:
                raise ValueError(f"{name} must be at least {read_margin + ready_needed} for this warmup/batch "
                                 f"({read_margin} safety slots plus {ready_needed} ready samples); got {cap}")
        from .sim import Ring, build
        build()  # compile the simulator before the actor process needs it
        self.run_dir = (PROJECT / cfg.run).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.run_dir / ".lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise SystemExit(f"another training process is already using {cfg.run}") from None
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.device = torch.device(cfg.device)
        saved = self.run_dir / "latest.pt"
        saved = saved if saved.exists() else None
        try:
            if saved is None and _has_training_history(self.run_dir):
                raise FileNotFoundError(f"{self.run_dir / 'latest.pt'} is missing, but this run has trained updates; "
                                        "restore the canonical checkpoint before continuing")
            if saved and not cfg.resume:
                raise SystemExit(f"{cfg.run} already holds a trained AI; pick another --run for a fresh start")
            ck = torch.load(saved, map_location=self.device, weights_only=True) if saved else None
            if ck:
                if ck.get("trainer") != "teach":
                    raise SystemExit(f"{saved} was written by another trainer; use a new --run folder")
                net_version(ck["model"])
                net_version(ck["exploiter"])
        except BaseException:
            self._lock.close()
            raise
        self.main = make_net().to(self.device)
        self.exploiter = make_net().to(self.device)
        self.opt = self._optimizer(self.main)
        self.xopt = self._optimizer(self.exploiter)
        self.update = 0
        self.xupdates = 0  # updates of the current exploiter
        self.generation = 0  # exploiters so far
        self.trained = [0, 0]  # samples trained per agent, over the whole run
        self.generated = [0, 0]  # samples generated per agent, over the whole run
        self.base = [0, 0]  # ... before this session (the rings start empty every session)
        self.session_trained = [0, 0]  # what the replay-ratio gate compares with the rings
        self.boundary = [0, 0]  # ring samples older than this belong to a previous exploiter
        self.decisions = 0
        self.games = 0
        self.imitation_state = dict((ck or {}).get("imitation_state", {}))
        self.saved_update = int(ck.get("update", 0)) if ck else None
        self.saved_at = saved.stat().st_mtime if saved else None
        self.last_training_method = (ck or {}).get("last_training_method", "rl")
        after_imitation = self.last_training_method == "imitation"
        if saved:
            # Configuration from an older V4 checkpoint is metadata only. Current CLI settings
            # govern this session; model weights, optimiser moments and counters resume exactly.
            self.main.load_state_dict(ck["model"])
            if after_imitation:
                # Imitation changed the shared weights under a different objective. Fresh
                # moments and a current-policy opponent avoid carrying stale RL momentum.
                self.exploiter.load_state_dict(self.main.state_dict())
            else:
                self.opt.load_state_dict(ck["opt"])
                self.exploiter.load_state_dict(ck["exploiter"])
                self.xopt.load_state_dict(ck["exploiter_opt"])
            for o in (self.opt, self.xopt):
                for group in o.param_groups:
                    group["lr"] = cfg.lr
                    group["weight_decay"] = cfg.weight_decay
            self.update = ck["update"]
            self.xupdates = 0 if after_imitation else ck.get("xupdates", 0)
            self.generation = ck.get("generation", 0)
            self.trained = ck.get("trained", [0, 0])
            self.generated = ck.get("generated", [0, 0])
            self.base = list(self.generated)
            self.decisions = ck.get("decisions", 0)
            self.games = ck.get("games", 0)
            print(f"resumed {self.run_dir.name} (V4) at update {self.update} ({self.decisions:,} decisions)", flush=True)
        else:
            self.exploiter.load_state_dict(self.main.state_dict())
            print(f"new run in {self.run_dir.name} (V4, {sum(p.numel() for p in self.main.parameters()) / 1e6:.1f} M parameters)",
                  flush=True)
        # a sample needs value_steps more decisions of its round before it can be learned from; the
        # ring must hold it that long even when every round has four learning players
        wrap = cfg.ring // (cfg.games * 4)
        if cfg.value_steps > wrap // 2:
            print(f"warning: --ring {cfg.ring:,} turns over every ~{wrap} decisions with {cfg.games} rounds; "
                  f"--value-steps {cfg.value_steps} needs a bigger ring or fewer --games", flush=True)
        self.rings = [Ring(cfg.ring, create=True), Ring(cfg.exploiter_ring, create=True)]
        self.log = open(self.run_dir / "metrics.jsonl", "a")
        self.publish(main=True, exploiter=True, target=True)
        self.window = defaultdict(lambda: deque(maxlen=400))
        self.lengths = deque(maxlen=400)
        self.sps = 0.0
        self.evals_s = 0.0
        self.acc = defaultdict(float)
        self.acc_n = 0
        self.write_progress("starting")

    def _optimizer(self, net):
        return torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, eps=1e-6)

    # ---------------------------------------------------------------- files
    def write_progress(self, phase: str | None = None) -> None:
        """Small atomic heartbeat for the terminal monitor, including replay warmup."""
        ring = self.rings[0]
        written = ring.written()
        ready = ring.window()
        _, ready_target, searched_target = replay_requirements(self.cfg.warmup, self.cfg.batch)
        if phase is None:
            phase = "learning" if self.session_trained[0] else "warming_up"
        rec = {"trainer": "teach", "run": str(self.run_dir), "pid": os.getpid(), "time": time.time(),
               "mode": self.cfg.mode, "config": asdict(self.cfg), "started_at": self.started_at,
               "hyperparameters": self.hyperparameters(),
               "resources": {"duty_cycle": self.cfg.duty_cycle, "scope": "per_process",
                             "torch_threads": self.cfg.torch_threads,
                             "actor_throttle_seconds": round(self.actor_throttle_seconds, 3),
                             "learner_throttle_seconds": round(self.pacer.rest_seconds, 3)},
               "phase": phase, "update": self.update, "max_updates": self.cfg.max_updates,
               "saved_update": self.saved_update, "saved_at": self.saved_at,
               "steps": self.decisions, "games": self.games,
               "samples": self.base[0] + written, "trained": self.trained[0],
               "warmup_samples": written, "warmup_target": self.cfg.warmup,
               "warmup_ready_target": ready_target, "warmup_searched_target": searched_target,
               "ready_samples": len(ready), "searched_samples": int(np.count_nonzero(ring.has_policy[ready])),
               "batch": self.cfg.batch, "actions": self.main.n_actions, "net": self.main.version,
               "sps": round(self.sps), "evals_s": round(self.evals_s)}
        tmp = self.run_dir / "progress.json.tmp"
        try:
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.replace(self.run_dir / "progress.json")
        except OSError as exc:
            # Monitoring must not interrupt an otherwise healthy learner.
            if not getattr(self, "_progress_warning", False):
                print(f"could not write terminal progress: {exc}", flush=True)
                self._progress_warning = True

    def hyperparameters(self) -> dict:
        """Resolved settings for the live viewer, including the actual optimiser values."""
        names = ("batch", "weight_decay", "value_coef", "aux_coef", "own_coef", "max_grad_norm", "bf16",
                 "replay_ratio", "warmup", "sequence_burn_in", "sequence_unroll", "ring", "exploiter_ring", "sims", "macro", "search_prob", "gumbel",
                 "temperature", "value_steps", "hunter_prob", "hunter_depth", "combat_prob", "exploiter_prob",
                 "exploiter_win", "exploiter_games",
                 "exploiter_patience", "publish_seconds", "save_seconds", "seed")
        return {**{name: getattr(self.cfg, name) for name in names},
                "lr": self.opt.param_groups[0]["lr"],
                "weight_decay": self.opt.param_groups[0]["weight_decay"],
                "optimizer": "AdamW", "optimizer_eps": self.opt.param_groups[0]["eps"],
                "optimizer_betas": list(self.opt.param_groups[0]["betas"])}

    def publish(self, main=True, exploiter=False, target=False) -> None:
        if main:
            _atomic_save({"model": _half(self.main.state_dict()), "update": self.update}, self.run_dir / "actor.pt")
        if exploiter:
            _atomic_save({"model": _half(self.exploiter.state_dict()), "update": self.update,
                          "generation": self.generation}, self.run_dir / "exploiter.pt")
        if target:
            _atomic_save({"model": _half(self.main.state_dict()), "update": self.update,
                          "generation": self.generation}, self.run_dir / "target.pt")

    def save(self) -> None:
        ck = {"trainer": "teach", "model": self.main.state_dict(), "opt": self.opt.state_dict(),
              "last_training_method": self.last_training_method, "imitation_state": self.imitation_state,
              "exploiter": self.exploiter.state_dict(), "exploiter_opt": self.xopt.state_dict(),
              "update": self.update, "xupdates": self.xupdates, "generation": self.generation,
              "trained": self.trained, "generated": self.generated, "decisions": self.decisions,
              "games": self.games, "config": asdict(self.cfg)}
        latest = self.run_dir / "latest.pt"
        tmp = self.run_dir / "latest.pt.tmp"
        torch.save(ck, tmp)
        tmp.replace(latest)
        self.saved_update, self.saved_at = self.update, latest.stat().st_mtime

    def cleanup_runtime_models(self) -> None:
        """Worker exchanges are temporary; latest.pt is the saved training state."""
        for name in ("actor.pt", "exploiter.pt", "target.pt"):
            (self.run_dir / name).unlink(missing_ok=True)
            (self.run_dir / (name + ".tmp")).unlink(missing_ok=True)

    def promote(self, why: str) -> None:
        """Refresh the live exploiter without retaining a historical checkpoint."""
        self.generation += 1
        self.fresh_exploiter()
        print(f"exploiter {self.generation} starts from update {self.update} ({why})", flush=True)

    def fresh_exploiter(self) -> None:
        """A new exploiter, a copy of the current network, against a frozen copy of it (the target).
        It learns only from its own rounds: the ring's older samples are its predecessor's."""
        self.exploiter.load_state_dict(self.main.state_dict())
        self.xopt = self._optimizer(self.exploiter)
        self.xupdates = 0
        self.boundary[1] = self.rings[1].written()
        self.session_trained[1] = 0
        self.publish(main=False, exploiter=True, target=True)

    # ---------------------------------------------------------------- learning
    def batch(self, ring, n: int, boundary: int = 0, generation: int | None = None):
        ready = ring.window()
        if boundary:
            ready = ready[ring.gen[ready] >= boundary]
        if generation is not None:
            ready = ready[ring.generation[ready] == generation]
        _, ready_target, searched_target = replay_requirements(self.cfg.warmup, n)
        if len(ready) < ready_target or np.count_nonzero(ring.has_policy[ready]) < searched_target:
            return None
        data = ring.sequences(n, self.rng, self.cfg.sequence_burn_in, self.cfg.sequence_unroll,
                              boundary=boundary, generation=generation)
        if data is None:
            return None
        return {key: torch.from_numpy(np.ascontiguousarray(value)).to(self.device) for key, value in data.items()}

    def step(self, net, opt, data) -> dict:
        cfg = self.cfg
        burn = cfg.sequence_burn_in
        maps, scal, mask, reset, context = (data[key] for key in ("maps", "scalars", "mask", "reset", "context"))
        state = net.initial_state(maps.shape[0], device=self.device)

        def unroll():
            nonlocal state
            if burn:
                with torch.no_grad():
                    _, _, state = net.forward_sequence(maps[:, :burn], scal[:, :burn], mask[:, :burn],
                                                        state, reset[:, :burn], context=context[:, :burn])
                state = state.detach()
            return net.heads_sequence(maps[:, burn:], scal[:, burn:], mask[:, burn:], state, reset[:, burn:],
                                      context=context[:, burn:])

        if cfg.bf16 and self.device.type in ("mps", "cuda"):
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                logits, value, aux, own, _ = unroll()
            logits, value, aux, own = logits.float(), value.float(), aux.float(), own.float()
        else:
            logits, value, aux, own, _ = unroll()
        mask, learn = mask[:, burn:], data["learn"][:, burn:]
        policy = data["policy"][:, burn:].masked_fill(~mask, 0)
        mass = policy.sum(-1, keepdim=True)
        has_policy = data["has_policy"][:, burn:] & learn & (mass[..., 0] > 1e-8)
        policy = policy / mass.clamp_min(1e-8)
        logp = torch.log_softmax(logits.masked_fill(~mask, -1e8), -1)
        pl, target = logp[has_policy], policy[has_policy]
        policy_loss = -(target * pl.clamp_min(-30)).sum(-1).mean() if len(target) else logits.sum() * 0
        z, v = data["value"][:, burn:][learn], value[learn]
        value_loss = F.mse_loss(v, z)
        aux_target, aux = data["aux"][:, burn:][learn], aux[learn]
        territory_loss = F.binary_cross_entropy_with_logits(aux[:, 0], aux_target[:, 0])
        danger_loss = F.binary_cross_entropy_with_logits(aux[:, 1], aux_target[:, 1])
        safe_kill_loss = F.binary_cross_entropy_with_logits(aux[:, 2], aux_target[:, 2])
        aux_loss = territory_loss + danger_loss + safe_kill_loss
        own_ok = data["own_ok"][:, burn:] & learn
        own_loss = (F.cross_entropy(own[own_ok], data["own"][:, burn:][own_ok])
                    if own_ok.any() else own.sum() * 0)
        loss = policy_loss + cfg.value_coef * value_loss + cfg.aux_coef * aux_loss + cfg.own_coef * own_loss
        if not torch.isfinite(loss).item():
            raise FloatingPointError("non-finite teacher loss; checkpoint weights were not updated")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
        opt.step()
        if net is self.main:
            self.imitation_state.pop("optimizer", None)
            self.last_training_method = "rl"
        with torch.no_grad():
            entropy = -(pl.exp() * pl).sum(-1).mean() if len(target) else v.new_zeros(())
            kl = ((target * (torch.log(target.clamp_min(1e-8)) - pl.clamp_min(-30))).sum(-1).mean()
                  if len(target) else v.new_zeros(()))
            explained = 1 - torch.var(z - v, correction=0) / (torch.var(z, correction=0) + 1e-8)
        return {"pg": policy_loss.detach(), "vf": value_loss.detach(), "aux": aux_loss.detach(),
                "territory": territory_loss.detach(), "danger": danger_loss.detach(), "safe_kill": safe_kill_loss.detach(),
                "own": own_loss.detach(), "ent": entropy, "search_kl": kl, "grad_norm": gn.detach(),
                "value_explained": explained, "n": int(learn.sum())}

    def drain(self, q) -> None:
        while True:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                return
            if kind == "stats":
                self.actor_throttle_seconds = payload.get("throttle_seconds", self.actor_throttle_seconds)
                self.decisions += payload["decisions"]
                self.games += len(payload["lengths"])
                self.sps = payload["decisions"] / max(payload["seconds"], 1e-6)
                self.evals_s = payload["evals"] / max(payload["seconds"], 1e-6)
                for k, v in payload["results"].items():
                    self.window[k].extend(v)
                self.lengths.extend(payload["lengths"])
            elif kind == "promote":
                if payload.get("generation") == self.generation:
                    self.promote(f"won {payload['win_rate']:.0%} of its last rounds")

    def run(self) -> None:
        cfg = self.cfg
        ctx = mp.get_context("spawn")
        q, stop = ctx.Queue(), ctx.Event()
        actor = ctx.Process(target=actor_main, daemon=True,
                            args=(asdict(cfg), str(self.run_dir), [r.name for r in self.rings],
                                  [r.cap for r in self.rings], q, stop))
        actor.start()
        print(f"actor started: {cfg.games} rounds at once, {cfg.sims} simulations per searched decision; "
              f"learning on {self.device}", flush=True)
        print(f"mode {cfg.mode}: {cfg.threads} simulator workers, batch {cfg.batch}, "
              f"{cfg.torch_threads} CPU tensor workers per process; "
              + (f"{cfg.duty_cycle:.0%} work/rest target per actor and learner" if cfg.duty_cycle < 1
                 else "no work/rest throttling"), flush=True)
        last_log = last_publish = last_save = time.time()
        last_progress = time.monotonic()
        self.write_progress()
        end_phase = "stopped"
        try:
            while not cfg.max_updates or self.update < cfg.max_updates:
                work_started = time.monotonic()
                self.drain(q)
                if not actor.is_alive():
                    raise RuntimeError("actor process died")
                self.generated = [b + r.written() for b, r in zip(self.base, self.rings)]
                trained_any = False
                for agent, (net, opt) in enumerate(((self.main, self.opt), (self.exploiter, self.xopt))):
                    ring = self.rings[agent]
                    fresh = ring.written() - self.boundary[agent]
                    if fresh < cfg.warmup:
                        continue
                    if self.session_trained[agent] > cfg.replay_ratio * max(fresh - cfg.warmup // 2, 1):
                        continue
                    data = self.batch(ring, cfg.batch, self.boundary[agent], self.generation if agent == 1 else None)
                    if data is None:
                        continue
                    out = self.step(net, opt, data)
                    self.trained[agent] += out["n"]
                    self.session_trained[agent] += out["n"]
                    trained_any = True
                    if agent == 0:
                        self.update += 1
                        for k, v in out.items():
                            if k != "n":
                                self.acc[k] += float(v)
                        self.acc_n += 1
                        now = time.time()
                        if now - last_publish > cfg.publish_seconds:
                            self.publish(main=True, exploiter=True)
                            last_publish = now
                        if now - last_save > cfg.save_seconds:
                            self.save()
                            last_save = now
                        if self.update % cfg.log_every == 0:
                            self.write_metrics(time.time() - last_log)
                            last_log = time.time()
                    else:
                        self.xupdates += 1
                        if self.xupdates >= cfg.exploiter_patience:
                            self.promote(f"{self.xupdates} updates without winning often enough")
                    if cfg.max_updates and self.update >= cfg.max_updates:
                        break
                if time.monotonic() - last_progress >= 2:
                    self.write_progress()
                    last_progress = time.monotonic()
                if not trained_any:
                    time.sleep(0.05)
                elif cfg.duty_cycle < 1:
                    _synchronize_device(self.device)
                    self.pacer.pause(work_started, stop)
        except KeyboardInterrupt:
            raise
        except BaseException:
            end_phase = "failed"
            raise
        finally:
            # a second stop signal (a double Ctrl-C, or uv passing a TERM on) must not cut the save short
            handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                self.write_progress("stopping")
                stop.set()
                deadline = time.monotonic() + 30
                while actor.is_alive() and time.monotonic() < deadline:
                    self.drain(q)  # let the child flush its last statistics before joining
                    actor.join(timeout=0.1)
                if actor.is_alive():
                    actor.terminate()
                    actor.join(timeout=5)
                self.drain(q)
                self.generated = [b + r.written() for b, r in zip(self.base, self.rings)]
                if self.acc_n:
                    self.write_metrics(time.time() - last_log)
                self.save()
                self.publish(main=True, exploiter=True)
            except BaseException:
                end_phase = "failed"
                raise
            finally:
                self.write_progress(end_phase)
                self.cleanup_runtime_models()
                for r in self.rings:
                    r.close(unlink=True)
                self.log.close()
                self._lock.close()
                q.close()
                q.join_thread()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)

    def write_metrics(self, dt: float) -> None:
        n = max(self.acc_n, 1)
        rec = {"trainer": "teach", "net": self.main.version, "actions": self.main.n_actions,
               "mode": self.cfg.mode,
               "update": self.update, "steps": self.decisions, "sps": round(self.sps), "evals_s": round(self.evals_s),
               "lr": self.opt.param_groups[0]["lr"], "games": self.games,
               "samples": self.generated[0], "trained": self.trained[0],
               "replay": round(self.trained[0] / max(self.generated[0], 1), 2),
               "exploiter_gen": self.generation,
               "ep_len": float(np.mean(self.lengths)) if self.lengths else 0.0,
               "match_results": match_results(self.window), "match_window": 400,
               **{k: round(v / n, 5) for k, v in self.acc.items()},
               **{k: round(float(np.mean(v)), 3) for k, v in self.window.items() if v and k.startswith("win_")},
               **{k: round(float(np.mean(v)), 1) for k, v in self.window.items() if v and k.startswith("len_")},
               "time": time.time()}
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        self.acc = defaultdict(float)
        self.acc_n = 0
        wins = " ".join(f"{k[4:]}={v:.2f}" for k, v in rec.items() if k.startswith("win_"))
        print(f"u{self.update} {self.decisions:,} decisions {rec['sps']:,}/s {rec['evals_s']:,} evals/s "
              f"len={rec['ep_len']:.0f} pg={rec.get('pg', 0):.3f} vf={rec.get('vf', 0):.3f} gain={rec.get('search_kl', 0):.3f} "
              f"replay={rec['replay']} {wins}", flush=True)


def parse_config(argv=None) -> Config:
    # Resolve the last --mode first, then apply its defaults. Explicit flags override the
    # preset regardless of their position (including flags before --mode).
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=tuple(MODE_DEFAULTS), default="full")
    mode, _ = mode_parser.parse_known_args(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    help_text = {
        "mode": "background: fewer parallel arenas, smaller batches and a work/rest budget; full: uncapped",
        "duty_cycle": "per-process work/rest target in (0, 1]; 1 disables pacing; not a GPU or thermal limit",
        "torch_threads": "CPU tensor workers per actor/learner, separate from simulator --threads",
        "max_updates": "stop and save at this total main-network update count (0: unlimited)",
        "combat_prob": "fraction of duels with varied close encounter starts; 0 keeps only standard spawns",
    }
    defaults = {**asdict(Config()), **MODE_DEFAULTS[mode.mode], "mode": mode.mode}
    for f, v in defaults.items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else
                        (lambda x: x.lower() in ("1", "true", "yes")), default=v, help=help_text.get(f))
    cfg = Config(**vars(ap.parse_args(argv)))
    if not math.isfinite(cfg.duty_cycle) or not 0 < cfg.duty_cycle <= 1:
        ap.error("--duty-cycle must be greater than zero and at most one")
    if cfg.torch_threads < 1 or cfg.threads < 1:
        ap.error("--torch-threads and --threads must be positive")
    if not math.isfinite(cfg.combat_prob) or not 0 <= cfg.combat_prob <= 1:
        ap.error("--combat-prob must be between zero and one")
    return cfg


def main() -> None:
    cfg = parse_config()
    learner = Learner(cfg)

    def on_term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        learner.run()
    except KeyboardInterrupt:
        print("stopping", flush=True)


if __name__ == "__main__":
    main()
