"""Self-play PPO training against the real Armagetron engine.

Many headless engines run in parallel. Every neural-controlled cycle is one "stream" of
experience. Most streams are driven by the current learner; some are driven by frozen past
versions of it (so it can't forget how to beat old strategies) and some arenas also contain the
game's built-in AI. Only learner decisions are trained on.

Reward: dying costs up to -1 (scaled by how many opponents were still alive, so finishing second
in a free-for-all is better than dying first), surviving to win the round gives +1.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import protocol as P
from .engine import ArenaConfig, EnginePool, Step
from .model import PolicyNet

PROJECT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    run: str = "runs/main"
    engines: int = 12
    rollout: int = 32768  # learner transitions per PPO update
    epochs: int = 3
    minibatch: int = 4096
    lr: float = 2.5e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    target_kl: float = 0.03
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    kill_bonus: float = 0.1
    updates: int = 1_000_000
    save_every: int = 5
    snapshot_every: int = 20  # add the learner to the opponent pool this often
    past_prob: float = 0.35  # chance that a non-first self-play slot is a frozen past self
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    seed: int = 0
    resume: bool = True


def arena_mix(n: int) -> list[tuple[str, ArenaConfig]]:
    """The arenas each engine runs. Mixed sizes and player counts so the policy generalises."""
    templates = [
        ("duel_small", ArenaConfig(slots=2, size_factor=-3)),
        ("duel_mid", ArenaConfig(slots=2, size_factor=-1.5)),
        ("vs_ai_small", ArenaConfig(slots=1, builtin_ais=1, size_factor=-3)),
        ("ffa4", ArenaConfig(slots=4, size_factor=-1)),
        ("duel_std", ArenaConfig(slots=2, size_factor=0)),
        ("vs_ai_std", ArenaConfig(slots=1, builtin_ais=1, size_factor=0)),
        ("mixed_ffa", ArenaConfig(slots=2, builtin_ais=2, size_factor=-1)),
        ("duel_small", ArenaConfig(slots=2, size_factor=-3)),
        ("vs_ai_ffa", ArenaConfig(slots=1, builtin_ais=3, size_factor=-1)),
        ("duel_mid", ArenaConfig(slots=2, size_factor=-1.5)),
        ("ffa4", ArenaConfig(slots=4, size_factor=-1)),
        ("vs_ai_small", ArenaConfig(slots=1, builtin_ais=1, size_factor=-3)),
    ]
    return [templates[i % len(templates)] for i in range(n)]


class Store:
    """Flat transition storage with per-stream linkage (next index) for GAE.

    maps holds the local and global map of each observation, feats the scalars followed by the
    action mask (as 0/1), so one observation moves to the GPU in two transfers."""

    def __init__(self, cap: int):
        g = P.GRID
        self.cap = cap
        self.maps = np.empty((cap, 2, g, g), np.uint8)
        self.feats = np.empty((cap, P.N_SCALARS + P.N_ACTIONS), np.float32)
        self.action = np.empty(cap, np.int64)
        self.logp = np.empty(cap, np.float32)
        self.value = np.empty(cap, np.float32)
        self.reward = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, bool)
        self.next = np.full(cap, -1, np.int64)
        self.n = 0
        self.resolved = 0

    def add(self, local, globl, scal, mask: int) -> int:
        if self.n >= self.cap:
            raise RuntimeError("transition store overflow")
        i = self.n
        self.maps[i, 0] = local
        self.maps[i, 1] = globl
        self.feats[i, :P.N_SCALARS] = scal
        self.feats[i, P.N_SCALARS:] = MASK_TABLE[mask]
        self.reward[i] = 0.0
        self.done[i] = False
        self.next[i] = -1
        self.n += 1
        return i

    def compact(self, keep: list[int]) -> dict[int, int]:
        """Move the still-open transitions to the front; return old->new index map."""
        remap = {}
        for new, old in enumerate(keep):
            remap[old] = new
            for arr in (self.maps, self.feats, self.action, self.logp, self.value, self.reward, self.done):
                arr[new] = arr[old]
            self.next[new] = -1
        self.n = len(keep)
        self.resolved = 0
        return remap


MASK_TABLE = np.array([[(m >> a) & 1 for a in range(P.N_ACTIONS)] for m in range(1 << P.N_ACTIONS)], np.float32)


class Trainer:
    engine_command = None  # tests swap in a fake engine

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run_dir = (PROJECT / cfg.run).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "pool").mkdir(exist_ok=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.net = PolicyNet().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.update = 0
        self.total_steps = 0
        if cfg.resume and (self.run_dir / "latest.pt").exists():
            ck = torch.load(self.run_dir / "latest.pt", map_location=self.device, weights_only=True)
            self.net.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.update = ck["update"]
            self.total_steps = ck["total_steps"]
            print(f"resumed {self.run_dir.name} at update {self.update} ({self.total_steps:,} steps)")
        self.past_nets: list[PolicyNet] = []
        self._load_past()

        self.mix = arena_mix(cfg.engines)
        self.store = Store(cfg.rollout + 8192)
        self.open: dict[tuple[int, int], int] = {}  # stream -> index of its unresolved transition
        self.controller: dict[tuple[int, int], int] = {}  # stream -> -1 learner, k past net
        self.round_of: dict[int, int] = {}
        self.ep_len: dict[tuple[int, int], int] = defaultdict(int)
        self.results: dict[str, deque] = defaultdict(lambda: deque(maxlen=400))
        self.lengths: deque = deque(maxlen=400)
        self.log = open(self.run_dir / "metrics.jsonl", "a")

    # ------------------------------------------------------------------ opponents
    def _load_past(self) -> None:
        files = sorted((self.run_dir / "pool").glob("*.pt"))
        if not files:
            self.past_nets = []
            return
        # Recent snapshots are more relevant, but keep some old ones in the mix.
        picks = set(files[-2:])
        picks.update(random.sample(files, min(2, len(files))))
        nets = []
        for f in sorted(picks):
            n = PolicyNet().to(self.device)
            n.load_state_dict(torch.load(f, map_location=self.device, weights_only=True)["model"])
            n.eval()
            nets.append(n)
        self.past_nets = nets

    def _assign_controllers(self, engine: int, n_slots: int) -> None:
        name, arena = self.mix[engine]
        for k in range(n_slots):
            use_past = (k > 0 and arena.builtin_ais == 0 and self.past_nets
                        and random.random() < self.cfg.past_prob)
            self.controller[(engine, k)] = random.randrange(len(self.past_nets)) if use_past else -1

    # ------------------------------------------------------------------ reward bookkeeping
    def _resolve(self, key, reward: float, done: bool) -> None:
        i = self.open.get(key)
        if i is None:
            return
        self.store.reward[i] += reward
        if done:
            self.store.done[i] = True
            self.store.resolved += 1
            del self.open[key]

    def _process(self, step: Step) -> list[tuple[tuple[int, int], int]]:
        """Apply rewards from a STEP; return (stream, store_index or -1) rows needing actions."""
        e = step.engine
        if self.round_of.get(e) != step.round_id:
            # New round: anything still open from the old round ends without further reward.
            for k in range(len(step.slots)):
                self._resolve((e, k), 0.0, True)
            self.round_of[e] = step.round_id
            self._assign_controllers(e, len(step.slots))
        name = self.mix[e][0]
        rows = []
        for k, s in enumerate(step.slots):
            key = (e, k)
            learner = self.controller.get(key, -1) == -1
            r = self.cfg.kill_bonus * s.kills
            terminal = False
            if s.flags & P.FLAG_DIED:
                others = max(step.n_total - 1, 1)
                r -= step.n_alive / others  # n_alive excludes us now that we're dead
                terminal = True
                if learner:
                    self.results[name].append(0.0 if step.n_alive else 0.5)
            elif step.round_over and s.alive:
                won = bool(s.flags & P.FLAG_WON)
                r += 1.0 if won else 0.0
                terminal = True
                if learner:
                    self.results[name].append(1.0 if won else 0.5)
            if learner:
                self._resolve(key, r, terminal)
                if terminal:
                    self.lengths.append(self.ep_len.pop(key, 0))
            if s.needs_action:
                if learner:
                    i = self.store.add(s.local, s.globl, s.scalars, s.mask)
                    prev = self.open.get(key)
                    if prev is not None:
                        self.store.next[prev] = i
                        self.store.resolved += 1
                    self.open[key] = i
                    self.ep_len[key] += 1
                    rows.append((key, i))
                else:
                    rows.append((key, -1))
        return rows

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def _act(self, pool: EnginePool, steps: list[Step]) -> None:
        per_engine_actions = {st.engine: [0] * len(st.slots) for st in steps}
        learner_rows, past_rows = [], defaultdict(list)
        for st in steps:
            for key, idx in self._process(st):
                if idx >= 0:
                    learner_rows.append((key, idx))
                else:
                    past_rows[self.controller[key]].append((key, st.slots[key[1]]))
        d = self.device
        if learner_rows:
            ids = np.array([i for _, i in learner_rows])
            a, logp, v = self.net.act(torch.from_numpy(self.store.maps[ids]).to(d),
                                      torch.from_numpy(self.store.feats[ids]).to(d))
            out = torch.stack([a.float(), logp, v], 1).cpu().numpy()
            self.store.action[ids] = out[:, 0].astype(np.int64)
            self.store.logp[ids] = out[:, 1]
            self.store.value[ids] = out[:, 2]
            for (key, _), act in zip(learner_rows, self.store.action[ids]):
                per_engine_actions[key[0]][key[1]] = int(act)
        for k, rows in past_rows.items():
            maps = np.stack([np.stack([s.local, s.globl]) for _, s in rows])
            feats = np.stack([np.concatenate([s.scalars, MASK_TABLE[s.mask]]) for _, s in rows])
            a, _, _ = self.past_nets[k].act(torch.from_numpy(maps).to(d), torch.from_numpy(feats).to(d))
            for (key, _), act in zip(rows, a.cpu().numpy()):
                per_engine_actions[key[0]][key[1]] = int(act)
        for e, acts in per_engine_actions.items():
            pool.act(e, acts)

    # ------------------------------------------------------------------ learning
    def _gae(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, g, lam = self.store, self.cfg.gamma, self.cfg.lam
        train = np.array([i for i in range(n) if s.done[i] or s.next[i] >= 0], np.int64)
        adv = np.zeros(n, np.float32)
        # Every resolved transition's successor has a larger index, so a reverse sweep works.
        for i in reversed(range(n)):
            if s.done[i]:
                adv[i] = s.reward[i] - s.value[i]
            elif s.next[i] >= 0:
                j = s.next[i]
                delta = s.reward[i] + g * s.value[j] - s.value[i]
                adv[i] = delta + g * lam * adv[j]
        ret = adv + s.value[:n]
        return train, adv[train], ret[train]

    def _learn(self) -> dict:
        s, cfg, d = self.store, self.cfg, self.device
        n = s.n
        train, adv, ret = self._gae(n)
        explained = float(1 - np.var(ret - s.value[train]) / (np.var(ret) + 1e-8)) if len(train) else 0.0
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(d)  # noqa: E731
        maps, feats, act = T(s.maps[train]), T(s.feats[train]), T(s.action[train])
        old_logp, old_v = T(s.logp[train]), T(s.value[train])
        adv_t, ret_t = T(adv), T(ret)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        m = len(train)
        sums = torch.zeros(5, device=d)
        count = 0
        for epoch in range(cfg.epochs):
            perm = torch.randperm(m, device=d)
            epoch_kl = torch.zeros((), device=d)
            batches = 0
            for b in range(0, m, cfg.minibatch):
                mb = perm[b:b + cfg.minibatch]
                f = feats[mb]
                logits, v = self.net(maps[mb, 0], maps[mb, 1], f[:, :P.N_SCALARS], f[:, P.N_SCALARS:] > 0.5)
                logp_all = torch.log_softmax(logits, 1)
                logp = logp_all.gather(1, act[mb, None]).squeeze(1)
                ent = -(logp_all.exp() * logp_all).sum(1).mean()
                ratio = torch.exp(logp - old_logp[mb])
                pg = -torch.min(ratio * adv_t[mb], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_t[mb]).mean()
                v_clip = old_v[mb] + (v - old_v[mb]).clamp(-cfg.clip, cfg.clip)
                vl = 0.5 * torch.max((v - ret_t[mb]) ** 2, (v_clip - ret_t[mb]) ** 2).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean()
                    sums += torch.stack([pg.detach(), vl.detach(), ent.detach(), kl, clipfrac])
                    epoch_kl += kl
                count += 1
                batches += 1
            # one GPU sync per epoch: stop early if the policy moved too far
            if (epoch_kl / max(batches, 1)).item() > cfg.target_kl:
                break
        vals = (sums / max(count, 1)).tolist()
        # Keep unresolved transitions for the next rollout.
        keep = sorted(self.open.values())
        remap = s.compact(keep)
        self.open = {k: remap[i] for k, i in self.open.items()}
        out = dict(zip(("pg", "vf", "ent", "kl", "clipfrac"), vals))
        out["samples"] = m
        out["value_explained"] = explained
        return out

    def save(self) -> None:
        ck = {"model": self.net.state_dict(), "opt": self.opt.state_dict(), "update": self.update,
              "total_steps": self.total_steps, "config": asdict(self.cfg)}
        tmp = self.run_dir / "latest.pt.tmp"
        torch.save(ck, tmp)
        tmp.replace(self.run_dir / "latest.pt")
        if self.update % self.cfg.snapshot_every == 0:
            torch.save({"model": self.net.state_dict(), "update": self.update},
                       self.run_dir / "pool" / f"u{self.update:06d}.pt")
            self._load_past()

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        arenas = [a for _, a in self.mix]
        workdir = PROJECT / "runtime" / self.run_dir.name
        with EnginePool(arenas, workdir=workdir, command=self.engine_command) as pool:
            print(f"{len(arenas)} engines up; training on {self.device}")
            t_last, steps_last = time.time(), self.total_steps
            while self.update < self.cfg.updates:
                t0 = time.time()
                while self.store.resolved < self.cfg.rollout:
                    steps = pool.poll(max_wait=0.003)
                    before = self.store.n
                    self._act(pool, steps)
                    self.total_steps += self.store.n - before
                t1 = time.time()
                stats = self._learn()
                self.update += 1
                t2 = time.time()
                sps = (self.total_steps - steps_last) / max(t2 - t_last, 1e-6)
                t_last, steps_last = t2, self.total_steps
                rec = {"update": self.update, "steps": self.total_steps, "sps": round(sps),
                       "collect_s": round(t1 - t0, 2), "learn_s": round(t2 - t1, 2),
                       "ep_len": float(np.mean(self.lengths)) if self.lengths else 0.0,
                       **{k: round(v, 5) for k, v in stats.items()},
                       **{f"win_{k}": round(float(np.mean(v)), 3) for k, v in self.results.items() if v},
                       "time": time.time()}
                self.log.write(json.dumps(rec) + "\n")
                self.log.flush()
                wins = " ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(self.results.items()) if v)
                print(f"u{self.update} {self.total_steps:,} steps {sps:,.0f}/s len={rec['ep_len']:.0f} "
                      f"ent={stats.get('ent', 0):.3f} kl={stats.get('kl', 0):.4f} | {wins}", flush=True)
                if self.update % self.cfg.save_every == 0:
                    self.save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for f, v in asdict(Config()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else
                        (lambda x: x.lower() in ("1", "true", "yes")), default=v)
    cfg = Config(**vars(ap.parse_args()))
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
