"""The teacher: search-guided self-play in the simulator, a league, and a learner.

    arma-teach                         # train runs/v2, resuming where it stopped
    arma-teach --games 128 --sims 24   # lighter

How it works:

* Hundreds of rounds run at once in the C++ simulator (ai/sim), which reproduces the engine's
  physics and the network's view of the arena exactly. On a quarter of the decisions every player
  searches ahead (Gumbel tree search over 0.1 s steps, guided by the network), and the search's
  improved move probabilities become the training target. Every decision also gets the round's
  actual outcome as the value target, and two auxiliary targets (territory share and danger two
  seconds ahead).
* The league: most rounds are the network against itself (duels on three arena sizes, and
  four-player free-for-alls); some are against frozen past versions of itself, picked more often
  the harder they are for it (prioritised fictitious self-play); and some are played by an
  exploiter, a copy that trains only to beat the current network. Once the exploiter wins often
  enough it is frozen into the pool, and a new one starts from the current network.
* Warm start: until the first BOOTSTRAP_SAMPLES decisions are in, the old brain (runs/main,
  the v1 network) does the thinking inside the search, so the new network learns from a competent
  teacher instead of from noise. After that it teaches itself.

Two processes share the GPU: the actor (self-play) and the learner (this one), connected by
shared-memory replay rings. Checkpoints: latest.pt (everything), actor.pt (the current network,
which the brain page follows), pool/ (frozen opponents), metrics.jsonl.
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
from .model import PolicyNet, make_net, net_version

PROJECT = Path(__file__).resolve().parent.parent

# (SIZE_FACTOR, players, share of the self-play rounds)
ARENAS = ((-3.0, 2, 0.30), (-1.5, 2, 0.25), (0.0, 2, 0.15), (-1.0, 4, 0.20), (0.0, 4, 0.10))
LEAGUE_SIZES = (-3.0, -1.5, 0.0)  # league rounds are duels on these arenas
AGENT_MAIN, AGENT_EXPLOITER, AGENT_TARGET, AGENT_POOL = 0, 1, 2, 3
MAX_POOL_LOADED = 4


@dataclass
class Config:
    run: str = "runs/v2"
    games: int = 256  # rounds played at once
    sims: int = 24  # simulations per searched decision
    macro: int = 2  # decisions per search step (0.1 s, the turn delay)
    search_prob: float = 0.25
    gumbel: float = 1.0
    temperature: float = 1.0
    threads: int = 8
    ring: int = 200_000  # the main network's replay window, in decisions
    exploiter_ring: int = 60_000
    batch: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-4
    value_coef: float = 1.0
    aux_coef: float = 0.25
    max_grad_norm: float = 1.0
    bf16: bool = True  # train in bfloat16 (about 20% faster on Apple GPUs)
    replay_ratio: float = 1.5  # training samples per generated one (the rest of the GPU goes to self-play)
    warmup: int = 20_000  # decisions in the ring before learning starts
    publish_seconds: float = 20  # the actor (and the brain page) get new weights this often
    save_seconds: float = 300  # full checkpoint (both networks and optimisers)
    snapshot_seconds: float = 1800  # a frozen copy of the network joins the league pool this often
    pool_keep: int = 40  # pool snapshots kept: the newest half, the rest spread over the older ones
    log_every: int = 25
    pool_prob: float = 0.15  # rounds against a frozen past self
    exploiter_prob: float = 0.20  # rounds played by the exploiter
    exploiter_win: float = 0.60  # the exploiter joins the pool once it wins this often...
    exploiter_games: int = 400  # ... over this many of its recent rounds
    exploiter_patience: int = 30_000  # or after this many of its updates, whichever is first
    bootstrap: str = "runs/main/latest.pt"  # the old brain that guides the search at first ("" for none)
    bootstrap_samples: int = 300_000  # samples (one per player and decision)
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    seed: int = 0
    resume: bool = True


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


# ====================================================================== actor
class Actor:
    """Plays the rounds: the self-play engine, the networks it asks, and the league bookkeeping."""

    def __init__(self, cfg: Config, run_dir: Path, rings, out_q):
        from .sim import SelfPlay
        self.cfg = cfg
        self.run_dir = run_dir
        self.out_q = out_q
        self.device = torch.device(cfg.device)
        self.half = self.device.type == "mps"
        self.rng = random.Random(cfg.seed + 17)
        self.sp = SelfPlay(cfg.games, sims=cfg.sims, macro=cfg.macro, search_prob=cfg.search_prob,
                           temperature=cfg.temperature, gumbel=cfg.gumbel, threads=cfg.threads)
        self.sp.set_ring(AGENT_MAIN, rings[0])
        self.sp.set_ring(AGENT_EXPLOITER, rings[1])
        self.nets: dict[int, torch.nn.Module] = {}
        self.mtimes: dict[tuple, float] = {}
        self.pool_files: dict[int, Path] = {}  # agent id -> pool file, for the opponents new rounds may use
        self.next_pool_agent = AGENT_POOL
        self.pool_listing: tuple = ()
        self.pool_results: dict[str, list] = {}  # pool file name -> [score, games] of the main network against it
        self.exploiter_gen = -1
        self.exploiter_results: deque = deque(maxlen=cfg.exploiter_games)
        self.promotion_sent = False
        self.bootstrapping = False
        self.kind: dict[int, tuple] = {}
        self.seed = cfg.seed * 1_000_003
        self.results = defaultdict(list)
        self.lengths: list = []

    # ---------------------------------------------------------------- networks
    def _file(self, agent: int) -> Path:
        return {AGENT_MAIN: self.run_dir / "actor.pt", AGENT_EXPLOITER: self.run_dir / "exploiter.pt",
                AGENT_TARGET: self.run_dir / "target.pt"}.get(agent) or self.pool_files[agent]

    def reload(self) -> None:
        state = self._league_state()
        self.bootstrapping = bool(state.get("bootstrapping"))
        for agent in (AGENT_MAIN, AGENT_EXPLOITER, AGENT_TARGET):
            f = self._file(agent)
            if self.bootstrapping and agent == AGENT_MAIN and self.cfg.bootstrap:
                f = PROJECT / self.cfg.bootstrap
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
            self.mtimes[key] = m
            if agent == AGENT_EXPLOITER:
                gen = int(ck.get("generation", 0))
                if gen != self.exploiter_gen:
                    self.exploiter_gen = gen
                    self.exploiter_results.clear()
                    self.promotion_sent = False
        self._reload_pool()

    def _league_state(self) -> dict:
        try:
            return json.loads((self.run_dir / "league.json").read_text())
        except (FileNotFoundError, ValueError):
            return {}

    def _reload_pool(self, force: bool = False) -> None:
        files = sorted((self.run_dir / "pool").glob("*.pt"))
        listing = tuple(f.name for f in files)
        if listing == self.pool_listing and not force:
            return
        self.pool_listing = listing
        self.pool_files = {}
        # networks of rounds still being played stay loaded until those rounds are over
        in_use = {a for (_, _, agents) in self.kind.values() for a in agents}
        for agent in [a for a in self.nets if a >= AGENT_POOL and a not in in_use]:
            del self.nets[agent]
        if not files:
            return
        # prioritised fictitious self-play: opponents the network struggles against come up more
        weights = []
        for f in files:
            score, games = self.pool_results.get(f.name, [0.0, 0])
            wr = (score + 1) / (games + 2)
            weights.append((1 - wr) ** 2 + 0.02)
        picks = set()
        k = min(MAX_POOL_LOADED, len(files))
        while len(picks) < k:
            picks.add(self.rng.choices(range(len(files)), weights)[0])
        for i in sorted(picks):
            agent = self.next_pool_agent
            self.next_pool_agent += 1
            try:
                net, _ = _load_net(files[i], self.device, self.half)
            except Exception:
                continue
            self.nets[agent] = net
            self.pool_files[agent] = files[i]

    @torch.no_grad()
    def evaluate(self, n: int) -> None:
        sp = self.sp
        logits = np.zeros((n, P.N_ACTIONS), np.float32)
        values = np.zeros(n, np.float32)
        agents = sp.agents[:n]
        mask_bits = ((sp.masks[:n, None] >> np.arange(P.N_ACTIONS)) & 1).astype(bool)
        for agent in np.unique(agents):
            idx = np.flatnonzero(agents == agent)
            net = self.nets.get(int(agent))
            if net is None:
                net = self.nets[AGENT_MAIN]
            dtype = torch.float16 if self.half else torch.float32
            maps = torch.from_numpy(sp.maps[idx]).to(self.device)
            sc = torch.from_numpy(sp.scalars[idx]).to(self.device, dtype)
            mk = torch.from_numpy(mask_bits[idx]).to(self.device)
            lg, v = net(maps, sc, mk)
            logits[idx] = lg.float().cpu().numpy()
            values[idx] = v.float().cpu().numpy()
        sp.feed(logits, values)

    # ---------------------------------------------------------------- matchups
    def start(self, g: int) -> None:
        cfg, r = self.cfg, self.rng
        self.seed += 1
        x = r.random()
        if not self.bootstrapping and x < cfg.exploiter_prob and AGENT_EXPLOITER in self.nets and AGENT_TARGET in self.nets:
            size = r.choice(LEAGUE_SIZES)
            agents, learn, kind = [AGENT_EXPLOITER, AGENT_TARGET], [1, 0], ("exploiter", None)
        elif not self.bootstrapping and x < cfg.exploiter_prob + cfg.pool_prob and self.pool_files:
            agent = r.choice(sorted(self.pool_files))
            size = r.choice(LEAGUE_SIZES)
            agents, learn, kind = [AGENT_MAIN, agent], [1, 0], ("pool", self.pool_files[agent].name)
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
        self.sp.start(g, size, agents, learn, rotate=r.randrange(len(agents)), seed=self.seed)

    def finished(self, g: int, info: dict) -> None:
        (kind, name), hero, _ = self.kind[g]
        u = info["outcome"][hero]
        score = 1.0 if u >= 0.999 else (0.5 if abs(u) < 1e-6 else 0.0)
        self.lengths.append(info["decisions"])
        if kind == "self":
            self.results[f"len_{name}"].append(info["decisions"])
        elif kind == "pool":
            s = self.pool_results.setdefault(name, [0.0, 0])
            s[0] += score
            s[1] += 1
            self.results["win_vs_past"].append(score)
        elif kind == "exploiter":
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
        last_report = last_reload = time.time()
        evals = 0
        pool_games_since = 0
        while not stop.is_set() and os.getppid() == parent:
            n = self.sp.collect()
            if n:
                self.evaluate(n)
                evals += n
            for g in range(self.cfg.games):
                info = self.sp.info(g)
                if info is not None:
                    if self.kind[g][0][0] == "pool":
                        pool_games_since += 1
                    self.finished(g, info)
                    self.start(g)
            now = time.time()
            if now - last_reload > 2.0:
                self.reload()
                if pool_games_since > 200:
                    self._reload_pool(force=True)
                    pool_games_since = 0
                last_reload = now
            if now - last_report > 5.0:
                st = self.sp.stats()
                st.update(evals=evals, seconds=now - last_report, results=dict(self.results), lengths=self.lengths,
                          pool=dict(self.pool_results), bootstrapping=self.bootstrapping)
                self.out_q.put(("stats", st))
                self.results = defaultdict(list)
                self.lengths = []
                evals = 0
                last_report = now


def actor_main(cfg_dict: dict, run_dir: str, ring_names: list, ring_caps: list, out_q, stop) -> None:
    from .sim import Ring
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cfg = Config(**cfg_dict)
    torch.set_num_threads(2)
    rings = [Ring(c, shm_name=n) for n, c in zip(ring_names, ring_caps)]
    try:
        Actor(cfg, Path(run_dir), rings, out_q).run(stop)
    finally:
        for r in rings:
            r.close()


# ====================================================================== learner
class Learner:
    def __init__(self, cfg: Config):
        from .sim import Ring, build
        build()  # compile the simulator before the actor process needs it
        self.cfg = cfg
        self.run_dir = (PROJECT / cfg.run).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "pool").mkdir(exist_ok=True)
        self._lock = open(self.run_dir / ".lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another training process is already using {cfg.run}") from None
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.device = torch.device(cfg.device)
        self.main = PolicyNet().to(self.device)
        self.exploiter = PolicyNet().to(self.device)
        self.opt = self._optimizer(self.main)
        self.xopt = self._optimizer(self.exploiter)
        self.update = 0
        self.xupdates = 0  # updates of the current exploiter
        self.generation = 0  # exploiters so far
        self.trained = [0, 0]  # samples trained per agent, over the whole run
        self.generated = [0, 0]  # samples generated per agent, over the whole run
        self.base = [0, 0]  # ... before this session (the rings start empty every session)
        self.session_trained = [0, 0]  # what the replay-ratio gate compares with the rings
        self.decisions = 0
        self.games = 0
        self.bootstrapping = bool(cfg.bootstrap) and (PROJECT / cfg.bootstrap).exists()

        saved = next((f for f in (self.run_dir / "latest.pt", self.run_dir / "previous.pt") if f.exists()), None)
        if saved and not cfg.resume:
            raise SystemExit(f"{cfg.run} already holds a trained AI; pick another --run for a fresh start")
        if saved:
            ck = torch.load(saved, map_location=self.device, weights_only=True)
            if ck.get("trainer") != "teach":
                raise SystemExit(f"{saved} was written by another trainer; use a new --run folder")
            self.main.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.exploiter.load_state_dict(ck["exploiter"])
            self.xopt.load_state_dict(ck["exploiter_opt"])
            for o in (self.opt, self.xopt):
                for group in o.param_groups:
                    group["lr"] = cfg.lr
            self.update = ck["update"]
            self.xupdates = ck.get("xupdates", 0)
            self.generation = ck.get("generation", 0)
            self.trained = ck.get("trained", [0, 0])
            self.generated = ck.get("generated", [0, 0])
            self.base = list(self.generated)
            self.decisions = ck.get("decisions", 0)
            self.games = ck.get("games", 0)
            self.bootstrapping = self.bootstrapping and self.generated[0] < cfg.bootstrap_samples
            print(f"resumed {self.run_dir.name} at update {self.update} ({self.decisions:,} decisions)", flush=True)
        else:
            self.exploiter.load_state_dict(self.main.state_dict())
            print(f"new run in {self.run_dir.name}" + (f", guided by {cfg.bootstrap} for the first "
                                                         f"{cfg.bootstrap_samples:,} samples" if self.bootstrapping else ""),
                  flush=True)
        self.rings = [Ring(cfg.ring, create=True), Ring(cfg.exploiter_ring, create=True)]
        self.log = open(self.run_dir / "metrics.jsonl", "a")
        self.publish(main=True, exploiter=True, target=not (self.run_dir / "target.pt").exists())
        self._write_league()
        self.window = defaultdict(lambda: deque(maxlen=400))
        self.lengths = deque(maxlen=400)
        self.sps = 0.0
        self.evals_s = 0.0
        self.acc = defaultdict(float)
        self.acc_n = 0

    def _optimizer(self, net):
        return torch.optim.AdamW(net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, eps=1e-6)

    # ---------------------------------------------------------------- files
    def _write_league(self) -> None:
        tmp = self.run_dir / "league.json.tmp"
        tmp.write_text(json.dumps({"bootstrapping": self.bootstrapping, "generation": self.generation,
                                   "update": self.update}))
        tmp.replace(self.run_dir / "league.json")

    def publish(self, main=True, exploiter=False, target=False) -> None:
        if main:
            _atomic_save({"model": _half(self.main.state_dict()), "update": self.update}, self.run_dir / "actor.pt")
        if exploiter:
            _atomic_save({"model": _half(self.exploiter.state_dict()), "update": self.update,
                          "generation": self.generation}, self.run_dir / "exploiter.pt")
        if target:
            _atomic_save({"model": _half(self.main.state_dict()), "update": self.update}, self.run_dir / "target.pt")

    def save(self) -> None:
        ck = {"trainer": "teach", "model": self.main.state_dict(), "opt": self.opt.state_dict(),
              "exploiter": self.exploiter.state_dict(), "exploiter_opt": self.xopt.state_dict(),
              "update": self.update, "xupdates": self.xupdates, "generation": self.generation,
              "trained": self.trained, "generated": self.generated, "decisions": self.decisions,
              "games": self.games, "config": asdict(self.cfg)}
        latest = self.run_dir / "latest.pt"
        tmp = self.run_dir / "latest.pt.tmp"
        torch.save(ck, tmp)
        if latest.exists():
            latest.replace(self.run_dir / "previous.pt")
        tmp.replace(latest)

    def snapshot(self) -> None:
        _atomic_save({"model": _half(self.main.state_dict()), "update": self.update},
                     self.run_dir / "pool" / f"u{self.update:07d}.pt")
        self.prune_pool()

    def prune_pool(self) -> None:
        """Keep the pool small: the newest snapshots, and a spread of older ones (the same for exploiters)."""
        keep = self.cfg.pool_keep
        for pattern in ("u*.pt", "x*.pt"):
            files = sorted((self.run_dir / "pool").glob(pattern))
            if len(files) <= keep:
                continue
            recent = files[-(keep // 2):]
            older = files[:-(keep // 2)]
            n = keep - len(recent)
            picks = {older[round(i * (len(older) - 1) / max(n - 1, 1))] for i in range(n)}
            for f in older:
                if f not in picks:
                    f.unlink(missing_ok=True)

    def promote(self, why: str) -> None:
        """The exploiter joins the pool; a new one starts from the current network."""
        self.generation += 1
        _atomic_save({"model": _half(self.exploiter.state_dict()), "update": self.update, "kind": "exploiter",
                      "why": why}, self.run_dir / "pool" / f"x{self.generation:04d}-u{self.update:07d}.pt")
        self.prune_pool()
        self.exploiter.load_state_dict(self.main.state_dict())
        self.xopt = self._optimizer(self.exploiter)
        self.xupdates = 0
        self.publish(main=False, exploiter=True, target=True)
        self._write_league()
        print(f"exploiter {self.generation} joins the pool ({why}); a new one starts from update {self.update}",
              flush=True)

    # ---------------------------------------------------------------- learning
    def batch(self, ring, n: int):
        """Half the batch from searched decisions (policy targets), half from any decision; None
        until enough rounds are over for a full batch of each."""
        ready = ring.window()
        a = ring.sample(n // 2, self.rng, policy_only=True, ready=ready)
        b = ring.sample(n - n // 2, self.rng, ready=ready)
        if len(a) < n // 2 or len(b) < n - n // 2 or len(ready) < self.cfg.warmup // 2:
            return None
        idx = np.concatenate([a, b])
        has_pol = np.zeros(len(idx), bool)
        has_pol[:len(a)] = True
        maps = np.concatenate([ring.maps[idx], ring.prev_maps(idx)], 1)
        d = self.device
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(d)  # noqa: E731
        mask = ((ring.mask[idx, None] >> np.arange(P.N_ACTIONS)) & 1).astype(bool)
        return (T(maps), T(ring.scalars[idx]), T(mask), T(ring.policy[idx]), T(has_pol), T(ring.value[idx]),
                T(ring.aux[idx]), len(idx))

    def step(self, net, opt, data) -> dict:
        cfg = self.cfg
        maps, scal, mask, pol, has_pol, z, aux_t, n = data
        if cfg.bf16 and self.device.type in ("mps", "cuda"):
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                logits, v, aux = net.heads(maps, scal, mask)
            logits, v, aux = logits.float(), v.float(), aux.float()
        else:
            logits, v, aux = net.heads(maps, scal, mask)
        logits = logits.masked_fill(~mask, -1e8)
        logp = torch.log_softmax(logits, 1)
        # policy: cross-entropy against the search's improved policy (searched decisions only)
        pl = logp[has_pol]
        tg = pol[has_pol]
        policy_loss = -(tg * pl.clamp_min(-30)).sum(1).mean() if len(tg) else logits.sum() * 0
        value_loss = F.mse_loss(v, z)
        aux_loss = (F.binary_cross_entropy_with_logits(aux[:, 0], aux_t[:, 0]) +
                    F.binary_cross_entropy_with_logits(aux[:, 1], aux_t[:, 1]))
        loss = policy_loss + cfg.value_coef * value_loss + cfg.aux_coef * aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        opt.step()
        with torch.no_grad():
            p = pl.exp()
            ent = -(p * pl).sum(1).mean() if len(tg) else torch.zeros((), device=v.device)
            tlog = torch.log(tg.clamp_min(1e-8))
            kl = (tg * (tlog - pl.clamp_min(-30))).sum(1).mean() if len(tg) else torch.zeros((), device=v.device)
            var = torch.var(z)
            expl = 1 - torch.var(z - v) / (var + 1e-8)
        return {"pg": policy_loss.detach(), "vf": value_loss.detach(), "aux": aux_loss.detach(), "ent": ent,
                "search_kl": kl, "grad_norm": gn.detach(), "value_explained": expl, "n": n}

    def drain(self, q) -> None:
        while True:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                return
            if kind == "stats":
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
        last_log = last_publish = last_save = last_snapshot = time.time()
        try:
            while True:
                self.drain(q)
                if not actor.is_alive():
                    raise RuntimeError("actor process died")
                self.generated = [b + r.written() for b, r in zip(self.base, self.rings)]
                if self.bootstrapping and self.generated[0] >= cfg.bootstrap_samples:
                    self.bootstrapping = False
                    self._write_league()
                    print(f"warm start done after {self.generated[0]:,} samples: the network guides its own search now",
                          flush=True)
                trained_any = False
                for agent, (net, opt) in enumerate(((self.main, self.opt), (self.exploiter, self.xopt))):
                    ring = self.rings[agent]
                    if ring.written() < cfg.warmup:
                        continue
                    if self.session_trained[agent] > cfg.replay_ratio * max(ring.written() - cfg.warmup // 2, 1):
                        continue
                    data = self.batch(ring, cfg.batch)
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
                        if now - last_snapshot > cfg.snapshot_seconds:
                            self.snapshot()
                            last_snapshot = now
                        if self.update % cfg.log_every == 0:
                            self.write_metrics(time.time() - last_log)
                            last_log = time.time()
                    else:
                        self.xupdates += 1
                        if self.xupdates >= cfg.exploiter_patience:
                            self.promote(f"{self.xupdates} updates without winning often enough")
                if not trained_any:
                    time.sleep(0.05)
        finally:
            stop.set()
            actor.join(timeout=30)
            if actor.is_alive():
                actor.terminate()
            self.save()
            self.publish(main=True, exploiter=True)
            for r in self.rings:
                r.close(unlink=True)

    def write_metrics(self, dt: float) -> None:
        n = max(self.acc_n, 1)
        rec = {"update": self.update, "steps": self.decisions, "sps": round(self.sps), "evals_s": round(self.evals_s),
               "lr": self.opt.param_groups[0]["lr"], "games": self.games,
               "samples": self.generated[0], "trained": self.trained[0],
               "replay": round(self.trained[0] / max(self.generated[0], 1), 2),
               "exploiter_gen": self.generation, "bootstrapping": self.bootstrapping,
               "ep_len": float(np.mean(self.lengths)) if self.lengths else 0.0,
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
              f"replay={rec['replay']} {wins}{' (warm start)' if self.bootstrapping else ''}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, v in asdict(Config()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else
                        (lambda x: x.lower() in ("1", "true", "yes")), default=v)
    cfg = Config(**vars(ap.parse_args()))
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
