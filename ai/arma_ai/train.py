"""Self-play PPO training against the real Armagetron engine.

Many headless engines run in parallel. Every neural-controlled cycle is one "stream" of
experience. Most streams are driven by the current learner; some are driven by frozen past
versions of it (so it can't forget how to beat old strategies) and some arenas also contain the
game's built-in AI. Only learner decisions are trained on.

Two processes share the work so neither the engines nor the GPU sit idle: an actor process runs
the engines and picks moves with a copy of the network, a learner process runs PPO updates. They
hand rollouts over through two shared-memory buffers and the actor reloads the learner's weights
after every rollout (so its policy lags the learner by at most two updates; PPO's clipped
importance ratio is computed against the policy that actually picked the moves).

Reward: dying costs up to -1 (scaled by how many opponents were still alive, so finishing second
in a free-for-all is better than dying first), surviving to win the round gives +1, killing an
enemy gives a small bonus.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import random
import signal
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

from . import protocol as P
from .engine import ArenaConfig, EnginePool, Step
from .model import PolicyNet

PROJECT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    run: str = "runs/main"
    engines: int = 24
    rollout: int = 32768  # learner transitions per PPO update
    epochs: int = 2  # data is cheap with async collection; fresher data beats more reuse
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
    # mostly self-play now that the built-in AI is beaten; a few built-in arenas stay as a sanity anchor
    templates = [
        ("duel_small", ArenaConfig(slots=2, size_factor=-3)),
        ("duel_mid", ArenaConfig(slots=2, size_factor=-1.5)),
        ("ffa4", ArenaConfig(slots=4, size_factor=-1)),
        ("duel_std", ArenaConfig(slots=2, size_factor=0)),
        ("vs_ai_small", ArenaConfig(slots=1, builtin_ais=1, size_factor=-3)),
        ("duel_small", ArenaConfig(slots=2, size_factor=-3)),
        ("duel_mid", ArenaConfig(slots=2, size_factor=-1.5)),
        ("mixed_ffa", ArenaConfig(slots=2, builtin_ais=2, size_factor=-1)),
        ("duel_small", ArenaConfig(slots=2, size_factor=-3)),
        ("ffa4", ArenaConfig(slots=4, size_factor=-1)),
        ("duel_std", ArenaConfig(slots=2, size_factor=0)),
        ("vs_ai_std", ArenaConfig(slots=1, builtin_ais=1, size_factor=0)),
    ]
    return [templates[i % len(templates)] for i in range(n)]


MASK_TABLE = np.array([[(m >> a) & 1 for a in range(P.N_ACTIONS)] for m in range(1 << P.N_ACTIONS)], np.float32)

FIELDS = (
    ("maps", np.uint8, (2, P.GRID, P.GRID)),  # local and global map
    ("feats", np.float32, (P.N_SCALARS + P.N_ACTIONS,)),  # scalars, then the action mask as 0/1
    ("action", np.int64, ()),
    ("logp", np.float32, ()),
    ("value", np.float32, ()),
    ("reward", np.float32, ()),
    ("done", np.bool_, ()),
    ("next", np.int64, ()),  # index of the same stream's next transition, -1 if not known yet
)


class Store:
    """Transition arrays with per-stream linkage (next index) for GAE.

    Private numpy memory by default; with ``shm`` the arrays live in a named shared-memory block
    so the actor can hand a finished rollout to the learner without pickling."""

    def __init__(self, cap: int, shm_name: str | None = None, create: bool = False):
        self.cap = cap
        self.shm = None
        if shm_name is not None or create:
            total = sum(-(-cap * int(np.prod(s, dtype=np.int64)) * np.dtype(t).itemsize // 64) * 64
                        for _, t, s in FIELDS)
            self.shm = shared_memory.SharedMemory(name=shm_name, create=create, size=total)
        off = 0
        for name, dtype, shape in FIELDS:
            if self.shm is None:
                arr = np.zeros((cap, *shape), dtype)
            else:
                nbytes = cap * int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
                arr = np.ndarray((cap, *shape), dtype, buffer=self.shm.buf, offset=off)
                off += -(-nbytes // 64) * 64
            setattr(self, name, arr)
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

    def copy_to(self, other: Store) -> None:
        n = self.n
        for name, _, _ in FIELDS:
            getattr(other, name)[:n] = getattr(self, name)[:n]
        other.n = n

    def compact(self, keep: list[int]) -> dict[int, int]:
        """Move the still-open transitions to the front; return old->new index map."""
        remap = {}
        for new, old in enumerate(keep):
            remap[old] = new
            for name, _, _ in FIELDS:
                arr = getattr(self, name)
                arr[new] = arr[old]
            self.next[new] = -1
        self.n = len(keep)
        self.resolved = 0
        return remap

    def close(self, unlink: bool = False) -> None:
        if self.shm is not None:
            for name, _, _ in FIELDS:
                setattr(self, name, None)
            self.shm.close()
            if unlink:
                self.shm.unlink()


def gae(store: Store, n: int, gamma: float, lam: float):
    """Advantages for the resolved transitions among the first n. Successors always have larger
    indices than their predecessors, so a single reverse sweep works."""
    done, nxt, rew, val = store.done[:n], store.next[:n], store.reward[:n], store.value[:n]
    adv = np.zeros(n, np.float32)
    for i in range(n - 1, -1, -1):
        if done[i]:
            adv[i] = rew[i] - val[i]
        elif nxt[i] >= 0:
            j = nxt[i]
            adv[i] = rew[i] + gamma * val[j] - val[i] + gamma * lam * adv[j]
    train = np.flatnonzero(done | (nxt >= 0))
    return train, adv[train], adv[train] + val[train]


# ====================================================================== actor
class Actor:
    """Runs the engines, picks moves, keeps the reward bookkeeping."""

    def __init__(self, cfg: Config, run_dir: Path, engine_command=None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.engine_command = engine_command
        self.device = torch.device(cfg.device)
        random.seed(cfg.seed + 1)
        torch.manual_seed(cfg.seed + 1)
        self.net = PolicyNet().to(self.device).eval()
        self.weights_mtime = 0.0
        self.weights_update = 0
        self.reload_weights()
        self.past_nets: list[PolicyNet] = []
        self.pool_listing: tuple = ()
        self.reload_pool()
        self.mix = arena_mix(cfg.engines)
        self.store = Store(cfg.rollout + 8192)
        self.open: dict[tuple[int, int], int] = {}  # stream -> index of its unresolved transition
        self.controller: dict[tuple[int, int], int] = {}  # stream -> -1 learner, k past net
        self.round_of: dict[int, int] = {}
        self.ep_len: dict[tuple[int, int], int] = defaultdict(int)
        self.results: dict[str, list] = defaultdict(list)
        self.lengths: list = []

    # ---------------------------------------------------------------- weights
    def reload_weights(self) -> None:
        f = self.run_dir / "actor.pt"
        if not f.exists():
            return
        m = f.stat().st_mtime
        if m == self.weights_mtime:
            return
        try:
            ck = torch.load(f, map_location=self.device, weights_only=True)
        except Exception:
            return  # being replaced right now; next time
        self.net.load_state_dict(ck["model"])
        self.weights_mtime = m
        self.weights_update = ck.get("update", 0)

    def reload_pool(self) -> None:
        files = sorted((self.run_dir / "pool").glob("*.pt"))
        listing = tuple(f.name for f in files)
        if listing == self.pool_listing:
            return
        self.pool_listing = listing
        if not files:
            self.past_nets = []
            return
        # recent snapshots matter most, but keep some old ones in the mix
        picks = set(files[-2:])
        picks.update(random.sample(files, min(2, len(files))))
        nets = []
        for f in sorted(picks):
            n = PolicyNet().to(self.device).eval()
            n.load_state_dict(torch.load(f, map_location=self.device, weights_only=True)["model"])
            nets.append(n)
        self.past_nets = nets

    # ---------------------------------------------------------------- bookkeeping
    def _assign_controllers(self, engine: int, n_slots: int) -> None:
        _, arena = self.mix[engine]
        for k in range(n_slots):
            use_past = (k > 0 and arena.builtin_ais == 0 and self.past_nets
                        and random.random() < self.cfg.past_prob)
            self.controller[(engine, k)] = random.randrange(len(self.past_nets)) if use_past else -1

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
            # new round: anything still open from the old round ends without further reward
            for k in range(len(step.slots)):
                self._resolve((e, k), 0.0, True)
                self.ep_len.pop((e, k), None)
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
                r -= step.n_alive / max(step.n_total - 1, 1)  # n_alive no longer counts us
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

    @torch.no_grad()
    def act(self, pool: EnginePool, steps: list[Step]) -> int:
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
        return len(learner_rows)

    def hand_over(self, block: Store) -> dict:
        """Copy the rollout into a shared block, keep the open transitions, return stats."""
        self.store.copy_to(block)
        keep = sorted(self.open.values())
        remap = self.store.compact(keep)
        self.open = {k: remap[i] for k, i in self.open.items()}
        stats = {"results": dict(self.results), "lengths": self.lengths, "policy_update": self.weights_update}
        self.results = defaultdict(list)
        self.lengths = []
        return stats

    def run(self, blocks: list[Store], full_q, free_q, stop) -> None:
        arenas = [a for _, a in self.mix]
        workdir = PROJECT / "runtime" / self.run_dir.name
        with EnginePool(arenas, workdir=workdir, command=self.engine_command) as pool:
            while not stop.is_set():
                t0 = time.time()
                steps_taken = 0
                while self.store.resolved < self.cfg.rollout and not stop.is_set():
                    steps_taken += self.act(pool, pool.poll(max_wait=0.003))
                if stop.is_set():
                    break
                while True:  # wait for a free buffer; engines simply pause meanwhile
                    try:
                        b = free_q.get(timeout=1.0)
                        break
                    except queue.Empty:
                        if stop.is_set():
                            return
                stats = self.hand_over(blocks[b])
                stats.update(steps=steps_taken, collect_s=time.time() - t0)
                full_q.put((b, blocks[b].n, stats))
                self.reload_weights()
                self.reload_pool()


def actor_main(cfg_dict: dict, run_dir: str, shm_names: list[str], cap: int, full_q, free_q, stop,
               engine_command=None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the learner coordinates shutdown
    cfg = Config(**cfg_dict)
    blocks = [Store(cap, shm_name=n) for n in shm_names]
    try:
        Actor(cfg, Path(run_dir), engine_command).run(blocks, full_q, free_q, stop)
    finally:
        for b in blocks:
            b.close()


# ====================================================================== learner
class Learner:
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
            for group in self.opt.param_groups:  # a --lr flag wins over the learning rate saved with the optimiser
                group["lr"] = cfg.lr
            self.update = ck["update"]
            self.total_steps = ck["total_steps"]
            print(f"resumed {self.run_dir.name} at update {self.update} ({self.total_steps:,} steps)")
        self.results: dict[str, deque] = defaultdict(lambda: deque(maxlen=400))
        self.lengths: deque = deque(maxlen=400)
        self.log = open(self.run_dir / "metrics.jsonl", "a")
        self.publish()

    def publish(self) -> None:
        """Weights for the actor (written atomically; the actor polls the file)."""
        tmp = self.run_dir / "actor.pt.tmp"
        torch.save({"model": self.net.state_dict(), "update": self.update}, tmp)
        tmp.replace(self.run_dir / "actor.pt")

    def save(self) -> None:
        ck = {"model": self.net.state_dict(), "opt": self.opt.state_dict(), "update": self.update,
              "total_steps": self.total_steps, "config": asdict(self.cfg)}
        tmp = self.run_dir / "latest.pt.tmp"
        torch.save(ck, tmp)
        tmp.replace(self.run_dir / "latest.pt")
        if self.update % self.cfg.snapshot_every == 0:
            torch.save({"model": self.net.state_dict(), "update": self.update},
                       self.run_dir / "pool" / f"u{self.update:06d}.pt")

    def learn(self, s: Store, n: int) -> dict:
        cfg, d = self.cfg, self.device
        train, adv, ret = gae(s, n, cfg.gamma, cfg.lam)
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
        out = dict(zip(("pg", "vf", "ent", "kl", "clipfrac"), (sums / max(count, 1)).tolist()))
        out["samples"] = m
        out["value_explained"] = explained
        return out

    def run(self, engine_command=None) -> None:
        cfg = self.cfg
        cap = cfg.rollout + 8192
        blocks = [Store(cap, create=True) for _ in range(2)]
        ctx = mp.get_context("spawn")
        full_q, free_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        for b in range(len(blocks)):
            free_q.put(b)
        actor = ctx.Process(target=actor_main, daemon=True,
                            args=(asdict(cfg), str(self.run_dir), [b.shm.name for b in blocks], cap,
                                  full_q, free_q, stop, engine_command))
        actor.start()
        print(f"actor started with {cfg.engines} engines; learning on {self.device}", flush=True)
        t_last, steps_last = time.time(), self.total_steps
        try:
            while self.update < cfg.updates:
                try:
                    b, n, stats = full_q.get(timeout=5.0)
                except queue.Empty:
                    if not actor.is_alive():
                        raise RuntimeError("actor process died") from None
                    continue
                t1 = time.time()
                out = self.learn(blocks[b], n)
                free_q.put(b)
                self.update += 1
                self.total_steps += stats["steps"]
                self.publish()
                for k, v in stats["results"].items():
                    self.results[k].extend(v)
                self.lengths.extend(stats["lengths"])
                t2 = time.time()
                sps = (self.total_steps - steps_last) / max(t2 - t_last, 1e-6)
                t_last, steps_last = t2, self.total_steps
                rec = {"update": self.update, "steps": self.total_steps, "sps": round(sps),
                       "collect_s": round(stats["collect_s"], 2), "learn_s": round(t2 - t1, 2),
                       "policy_lag": self.update - 1 - stats["policy_update"],
                       "ep_len": float(np.mean(self.lengths)) if self.lengths else 0.0,
                       **{k: round(v, 5) for k, v in out.items()},
                       **{f"win_{k}": round(float(np.mean(v)), 3) for k, v in self.results.items() if v},
                       "time": time.time()}
                self.log.write(json.dumps(rec) + "\n")
                self.log.flush()
                wins = " ".join(f"{k}={np.mean(v):.2f}" for k, v in sorted(self.results.items()) if v)
                print(f"u{self.update} {self.total_steps:,} steps {sps:,.0f}/s len={rec['ep_len']:.0f} "
                      f"ent={out.get('ent', 0):.3f} kl={out.get('kl', 0):.4f} | {wins}", flush=True)
                if self.update % cfg.save_every == 0:
                    self.save()
        finally:
            stop.set()
            actor.join(timeout=30)
            if actor.is_alive():
                actor.terminate()
            for blk in blocks:
                blk.close(unlink=True)
            self.save()


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
