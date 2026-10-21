"""The teacher's arena: a C++ port of the engine's physics and of the network's view (ai/sim).

The library is compiled on first use (and again whenever its sources change) with the system's
clang++; nothing else to install.

    from arma_ai.sim import Sim
    s = Sim(); s.reset(size=-3, n=2)
    maps, scalars = s.observe(0)
    s.step([0, 0])
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import protocol as P

SIM_DIR = Path(__file__).resolve().parent.parent / "sim"
BUILD_DIR = SIM_DIR / "build"
STATE_FIELDS = ("x", "y", "dx", "dy", "speed", "alive", "rubber", "brake_reservoir", "braking", "distance",
                "kills", "team")


def _sources() -> list[Path]:
    return sorted(p for p in SIM_DIR.iterdir() if p.suffix in (".cpp", ".h"))


def build(force: bool = False) -> Path:
    """Compile ai/sim into a shared library (cached by a hash of the sources)."""
    h = hashlib.sha1()
    for p in _sources():
        h.update(p.name.encode())
        h.update(p.read_bytes())
    tag = h.hexdigest()[:12]
    lib = BUILD_DIR / f"libtron-{tag}.dylib"
    if lib.exists() and not force:
        return lib
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        raise RuntimeError("no C++ compiler found (install the Xcode command line tools)")
    tmp = lib.with_suffix(".tmp")
    cmd = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-o", str(tmp)] + \
          [str(p) for p in _sources() if p.suffix == ".cpp"]
    subprocess.run(cmd, check=True)
    tmp.replace(lib)
    for old in BUILD_DIR.glob("libtron-*.dylib"):
        if old != lib:
            old.unlink(missing_ok=True)
    return lib


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        L = ctypes.CDLL(str(build()))
        vp, i32, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        ip = ctypes.POINTER(ctypes.c_int)
        fp = ctypes.POINTER(ctypes.c_float)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        sig = {
            "tr_new": ([], vp), "tr_free": ([vp], None), "tr_copy": ([vp, vp], None),
            "tr_reset": ([vp, f32, i32, i32, ip], None), "tr_set_bounds": ([vp, f32, f32, f32, f32], None),
            "tr_place": ([vp, i32, f32, f32, f32, f32], None), "tr_step": ([vp, ip], None),
            "tr_act": ([vp, i32, i32], None), "tr_frame": ([vp], None), "tr_mask": ([vp, i32], ctypes.c_uint),
            "tr_observe": ([vp, i32, u8p, fp], None), "tr_state": ([vp, fp], i32), "tr_time": ([vp], f32),
            "tr_over": ([vp], i32), "tr_bounds": ([vp, fp], None),
            "tr_fighter_move": ([vp, i32, i32, i32, i32], i32),
        }
        i64p = ctypes.POINTER(ctypes.c_int64)
        sig.update({
            "sp_new": ([i32], vp), "sp_free": ([vp], None),
            "sp_config": ([vp, i32, i32, f32, f32, f32, f32, f32, f32, i32, i32, i32, i32, i32, i32, i32, i32], None),
            "sp_set_ring": ([vp, i32, u8p, fp, u8p, fp, u8p, fp, fp, i64p, i64p, u8p, i64p, ctypes.c_int64, i64p, u8p, i64p, fp], None),
            "sp_set_generations": ([vp, i32, i64p], None),
            "sp_collect_recurrent": ([vp, u8p, fp, u8p, ip, ip, i32, u8p, fp, ip, u8p, fp, u8p, fp], i32),
            "sp_feed_recurrent": ([vp, fp, fp, i32, i64p, fp], None),
            "sp_start": ([vp, i32, f32, i32, ip, ip, i32, ctypes.c_ulonglong], None),
            "sp_place_start": ([vp, i32, i32, fp], i32),
            "sp_collect": ([vp, u8p, fp, u8p, ip, ip, i32, u8p], i32),
            "sp_feed": ([vp, fp, fp, i32, i64p], None),
            "sp_watch": ([vp, i32], ctypes.c_char_p),
            "sp_game_info": ([vp, i32, fp], i32),
            "sp_stats": ([vp, ctypes.POINTER(ctypes.c_longlong)], None),
        })
        for name, (args, res) in sig.items():
            fn = getattr(L, name)
            fn.argtypes = args
            fn.restype = res
        _lib = L
    return _lib


def _ptr(a: np.ndarray, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


class Sim:
    """One arena. Cheap to copy (``clone``), so the search can branch from any state."""

    def __init__(self, _handle=None):
        self.L = lib()
        self.h = _handle or self.L.tr_new()
        self.n = 0

    def __del__(self):
        h, self.h = getattr(self, "h", None), None
        if h and _lib is not None:
            _lib.tr_free(h)

    def clone(self) -> Sim:
        other = Sim(self.L.tr_new())
        self.L.tr_copy(other.h, self.h)
        other.n = self.n
        return other

    def reset(self, size: float = -3, n: int = 2, rotate: int = 0, teams=None) -> None:
        self.n = n
        t = None if teams is None else _ptr(np.ascontiguousarray(teams, np.int32), ctypes.c_int)
        self.L.tr_reset(self.h, size, n, rotate, t)

    def set_bounds(self, lx, ly, hx, hy) -> None:
        self.L.tr_set_bounds(self.h, lx, ly, hx, hy)

    def place(self, i: int, x: float, y: float, dx: float, dy: float) -> None:
        self.L.tr_place(self.h, i, x, y, dx, dy)

    def step(self, actions) -> None:
        a = np.ascontiguousarray(actions, np.int32)
        self.L.tr_step(self.h, _ptr(a, ctypes.c_int))

    def mask(self, i: int) -> int:
        return int(self.L.tr_mask(self.h, i))

    def observe(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        maps = np.zeros((P.N_MAPS, P.GRID, P.GRID), np.uint8)
        scal = np.zeros(P.N_SCALARS, np.float32)
        self.L.tr_observe(self.h, i, _ptr(maps, ctypes.c_uint8), _ptr(scal, ctypes.c_float))
        return maps, scal

    def state(self) -> dict[str, np.ndarray]:
        out = np.zeros((max(self.n, 1), len(STATE_FIELDS)), np.float32)
        n = self.L.tr_state(self.h, _ptr(out, ctypes.c_float))
        return {k: out[:n, j] for j, k in enumerate(STATE_FIELDS)}

    @property
    def time(self) -> float:
        return float(self.L.tr_time(self.h))

    @property
    def over(self) -> bool:
        return bool(self.L.tr_over(self.h))

    def bounds(self) -> tuple[float, ...]:
        out = np.zeros(4, np.float32)
        self.L.tr_bounds(self.h, _ptr(out, ctypes.c_float))
        return tuple(float(v) for v in out)


# ------------------------------------------------------------------------------------- self-play

RING_FIELDS = (
    ("maps", np.uint8, (P.N_MAPS, P.GRID, P.GRID)),
    ("scalars", np.float32, (P.N_SCALARS,)),
    ("mask", np.uint8, ()),
    ("policy", np.float32, (P.N_ACTIONS,)),
    ("has_policy", np.uint8, ()),
    ("value", np.float32, ()),
    ("aux", np.float32, (3,)),  # future territory, death within horizon, credited kill and survive horizon
    ("prev", np.int64, ()),
    ("gen", np.int64, ()),
    ("ready", np.uint8, ()),
    ("future", np.int64, ()),  # ring index of the sample whose territory map is the ownership target
    ("generation", np.int64, ()),
    ("context", np.float32, (P.N_ACTIONS + 1,)),  # previous executed action and elapsed seconds / .05
    ("heading", np.uint8, ()),  # heading quadrant the sample's minimap and territory are drawn in
)

AGENT_HUNTER = -1  # a seat played by the simulator's built-in space fighter (no network)


class Ring:
    """A learning agent's replay ring: the self-play engine writes samples (and later their
    targets) into it, the trainer samples from it. Private memory, or a named shared-memory block
    so another process can read it."""

    def __init__(self, cap: int, shm_name: str | None = None, create: bool = False):
        from multiprocessing import shared_memory
        self.cap = cap
        sizes = [cap * int(np.prod(shape, dtype=np.int64)) * np.dtype(t).itemsize for _, t, shape in RING_FIELDS]
        padded = [-(-n // 64) * 64 for n in sizes]
        total = sum(padded) + 64
        self.shm = None
        if shm_name is not None or create:
            self.shm = shared_memory.SharedMemory(name=shm_name, create=create, size=total)
            buf = self.shm.buf
        else:
            self._mem = bytearray(total)
            buf = memoryview(self._mem)
        off = 0
        for (name, t, shape), n in zip(RING_FIELDS, padded):
            setattr(self, name, np.ndarray((cap, *shape), t, buffer=buf, offset=off))
            off += n
        self.counter = np.ndarray((1,), np.int64, buffer=buf, offset=off)
        if create or self.shm is None:
            self.gen[:] = -1
            self.ready[:] = 0
            self.generation[:] = 0
            self.context[:] = 0
            self.counter[0] = 0

    @property
    def name(self) -> str | None:
        return self.shm.name if self.shm is not None else None

    def written(self) -> int:
        return int(self.counter[0])

    def window(self, margin: int = 4096) -> np.ndarray:
        """Slots holding samples whose targets are written (the oldest MARGIN are skipped: the
        writer may overwrite them while a batch is being copied out)."""
        total = self.written()
        lo = max(0, total - self.cap + margin)
        if total <= lo:
            return np.zeros(0, np.int64)
        slots = np.arange(lo, total) % self.cap
        return slots[self.ready[slots] == 1]

    def sample(self, n: int, rng: np.random.Generator, policy_only: bool = False, ready: np.ndarray | None = None):
        """Indices of N samples whose round is over (optionally only searched decisions), drawn
        uniformly from READY (default: the current window); fewer if there are not enough."""
        pool = self.window() if ready is None else ready
        if policy_only and len(pool):
            pool = pool[self.has_policy[pool] == 1]
        if len(pool) == 0:
            return np.zeros(0, np.int64)
        return pool[rng.integers(0, len(pool), size=n)] if len(pool) >= n else pool.copy()

    def ownership(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The ownership target of each sample: who reaches each minimap cell first, auxHorizon
        decisions later (0 nobody, 1 me, 2 an enemy), drawn in the sample's own frame; and whether
        the target is still in the ring. The later map was drawn in the player's heading then, so
        it is turned back by the quadrants the heading changed by."""
        fut = self.future[idx]
        slot = np.where(fut >= 0, fut % self.cap, idx)
        valid = (fut >= 0) & (self.gen[slot] == fut)
        terr = self.maps[slot, P.N_MAPS - 1]
        cls = np.where(terr & 1, 1, np.where((terr >> 1) & 1, 2, 0)).astype(np.int64)
        turn = (self.heading[idx].astype(int) - self.heading[slot].astype(int)) % 4
        out = cls.copy()
        for k in (1, 2, 3):
            sel = turn == k
            if sel.any():
                out[sel] = np.rot90(cls[sel], k, axes=(1, 2))
        return out, valid

    def prev_maps(self, idx: np.ndarray) -> np.ndarray:
        """The previous decision's maps of each sample (its own maps at a spawn or once overwritten)."""
        prev = self.prev[idx]
        slot = np.where(prev >= 0, prev % self.cap, idx)
        valid = (prev >= 0) & (self.gen[slot] == prev)
        slot = np.where(valid, slot, idx)
        return self.maps[slot]

    def sequences(self, n: int, rng, burn_in: int, unroll: int, *, boundary: int = 0,
                  generation: int | None = None):
        """Copy current raw-observation chains; never substitute stale actor hidden states.

        Missing overwritten predecessors reject a window. Only a real life boundary permits
        left padding. Burn-in may truncate older history, while every learned step remains
        contiguous with the prefix and has completed targets.
        """
        if n < 1 or burn_in < 0 or unroll < 1:
            raise ValueError("sequence batch and unroll must be positive; burn-in cannot be negative")
        ready = self.window()
        ready = ready[self.gen[ready] >= boundary]
        if generation is not None:
            ready = ready[self.generation[ready] == generation]
        searched = ready[self.has_policy[ready] == 1]
        if not len(ready) or not len(searched):
            return None
        length = min(unroll, n)
        windows = max(1, n // length)
        width = burn_in + length
        selected = []
        for w in range(windows):
            pool = searched if w < max(1, windows // 2) else ready
            found = None
            for _ in range(64):
                end = int(self.gen[int(rng.choice(pool))])
                chain_generation = int(self.generation[end % self.cap])
                chain, current, born = [], end, False
                low = max(0, self.written() - self.cap + 4096)
                for _ in range(width + 1):
                    slot = current % self.cap
                    if current < low or self.gen[slot] != current:
                        break
                    if self.generation[slot] != chain_generation:
                        break
                    chain.append(current)
                    previous = int(self.prev[slot])
                    if previous == -1:
                        born = True
                        break
                    if not 0 <= previous < current:
                        break
                    current = previous
                if not born and len(chain) < width + 1:
                    continue
                actual = list(reversed(chain[:width]))
                slots = np.asarray(actual, np.int64) % self.cap
                offset = width - len(slots)
                learn = np.arange(offset, width) >= burn_in
                if not self.ready[slots[learn]].all() or np.any(self.gen[slots[learn]] < boundary):
                    continue
                if not self.has_policy[slots[learn]].any() and w < max(1, windows // 2):
                    continue
                predecessors = self.prev[slots].copy()
                previous_slots = np.where(predecessors >= 0, predecessors % self.cap, slots)
                if np.any((predecessors >= 0) & (self.gen[previous_slots] != predecessors)):
                    continue
                expected = self.gen[slots].copy()
                old_expected = self.gen[previous_slots].copy()
                maps = np.concatenate([self.maps[slots].copy(), self.maps[previous_slots].copy()], 1)
                scalars = self.scalars[slots].copy()
                context = self.context[slots].copy()
                mask = self.mask[slots].copy()
                policy, has_policy = self.policy[slots].copy(), self.has_policy[slots].copy()
                value, aux = self.value[slots].copy(), self.aux[slots].copy()
                own, own_ok = self.ownership(slots)
                if not (np.array_equal(self.gen[slots], expected) and
                        np.array_equal(self.gen[previous_slots], old_expected) and self.ready[slots[learn]].all()):
                    continue
                found = (offset, maps, scalars, context, mask, policy, has_policy, value, aux, own, own_ok,
                         bool(predecessors[0] == -1))
                break
            if found is None:
                return None
            selected.append(found)
        shape = (windows, width)
        out = dict(maps=np.zeros((*shape, 2 * P.N_MAPS, P.GRID, P.GRID), np.uint8),
                   scalars=np.zeros((*shape, P.N_SCALARS), np.float32), mask=np.ones((*shape, P.N_ACTIONS), bool),
                   context=np.zeros((*shape, P.N_ACTIONS + 1), np.float32),
                   policy=np.zeros((*shape, P.N_ACTIONS), np.float32), has_policy=np.zeros(shape, bool),
                   value=np.zeros(shape, np.float32), aux=np.zeros((*shape, 3), np.float32),
                   own=np.zeros((*shape, P.GRID, P.GRID), np.int64), own_ok=np.zeros(shape, bool),
                   learn=np.zeros(shape, bool), reset=np.zeros(shape, bool))
        for i, (offset, maps, scalars, context, masks, policy, has_policy, value, aux, own, own_ok, born) in enumerate(selected):
            where = (i, slice(offset, None))
            for key, data in (("maps", maps), ("scalars", scalars), ("context", context), ("policy", policy),
                              ("has_policy", has_policy), ("value", value), ("aux", aux), ("own", own), ("own_ok", own_ok)):
                out[key][where] = data
            out["mask"][where] = (masks[:, None] >> np.arange(P.N_ACTIONS) & 1).astype(bool)
            out["learn"][i, max(offset, burn_in):] = True
            out["reset"][i, offset] = born or offset > 0
        return out

    def close(self, unlink: bool = False) -> None:
        if self.shm is not None:
            for name, _, _ in RING_FIELDS:
                setattr(self, name, None)
            self.counter = None
            self.shm.close()
            if unlink:
                self.shm.unlink()


class SelfPlay:
    """Many self-play rounds with search, in the C++ engine (ai/sim/selfplay.cpp)."""

    INFO = 3 + 4 * 8

    def __init__(self, games: int, *, sims: int = 32, macro: int = 2, gamma: float = 0.999, c_visit: float = 50,
                 c_scale: float = 0.1, search_prob: float = 0.25, temperature: float = 1.0, gumbel: float = 1.0,
                 max_decisions: int = 6000, aux_horizon: int = 40, threads: int = 8, max_players: int = 4,
                 value_steps: int = 60, hunter_depth: int = 2, hunter_replan: int = 2, hunter_ply: int = 4,
                 hunter_probe: int = 6, recurrent: bool = False):
        self.L = lib()
        self.recurrent = recurrent
        self.macro = max(1, macro)
        self.games = games
        self.h = self.L.sp_new(games)
        self.L.sp_config(self.h, sims, macro, gamma, c_visit, c_scale, search_prob, temperature, gumbel,
                         max_decisions, aux_horizon, threads, value_steps, hunter_depth, hunter_replan, hunter_ply,
                         hunter_probe)
        cap = games * max_players
        self.maps = np.zeros((cap, 2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
        self.scalars = np.zeros((cap, P.N_SCALARS), np.float32)
        self.masks = np.zeros(cap, np.uint8)
        self.agents = np.zeros(cap, np.int32)
        self.game_ids = np.zeros(cap, np.int32)
        self.roots = np.zeros(cap, np.uint8)  # 1: a real decision, 0: a position inside a search
        self.states = np.zeros((cap, 256), np.float32)
        self.lengths = np.ones(cap, np.int32)
        self.sequence_maps = np.zeros((cap, self.macro, 2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
        self.sequence_scalars = np.zeros((cap, self.macro, P.N_SCALARS), np.float32)
        self.sequence_masks = np.zeros((cap, self.macro), np.uint8)
        self.sequence_contexts = np.zeros((cap, self.macro, P.N_ACTIONS + 1), np.float32)
        self._info = np.zeros(self.INFO, np.float32)
        self._rings = {}
        self._pending = 0

    def __del__(self):
        h, self.h = getattr(self, "h", None), None
        if h and _lib is not None:
            _lib.sp_free(h)

    def set_ring(self, agent: int, ring: Ring) -> None:
        self._rings[agent] = ring  # keep the memory alive
        u8, f, i64 = ctypes.c_uint8, ctypes.c_float, ctypes.c_int64
        self.L.sp_set_ring(self.h, agent, _ptr(ring.maps, u8), _ptr(ring.scalars, f), _ptr(ring.mask, u8),
                           _ptr(ring.policy, f), _ptr(ring.has_policy, u8), _ptr(ring.value, f), _ptr(ring.aux, f),
                           _ptr(ring.prev, i64), _ptr(ring.gen, i64), _ptr(ring.ready, u8), _ptr(ring.counter, i64),
                           ring.cap, _ptr(ring.future, i64), _ptr(ring.heading, u8), _ptr(ring.generation, i64),
                           _ptr(ring.context, f))

    def start(self, g: int, size: float, agents, learn, rotate: int = 0, seed: int = 0,
              placements=None, generations=None) -> None:
        poses = None
        if placements is not None:
            poses = np.ascontiguousarray(placements, np.float32)
            if poses.shape != (len(agents), 4) or not np.isfinite(poses).all():
                raise ValueError("placements must contain a finite (x, y, dx, dy) for every player")
        a = np.ascontiguousarray(agents, np.int32)
        lr = np.ascontiguousarray([1 if x else 0 for x in learn], np.int32)
        self.L.sp_start(self.h, g, size, len(a), _ptr(a, ctypes.c_int), _ptr(lr, ctypes.c_int), rotate, seed)
        versions = np.asarray(generations if generations is not None else [0] * len(a), dtype=np.int64)
        if versions.shape != (len(a),):
            raise ValueError("one generation is required per player")
        self.L.sp_set_generations(self.h, g, _ptr(np.ascontiguousarray(versions), ctypes.c_int64))
        if poses is not None and not self.L.sp_place_start(self.h, g, len(a), _ptr(poses, ctypes.c_float)):
            raise ValueError("spawn positions must be inside the arena, separated, and face a cardinal direction")

    def collect(self) -> int:
        self._pending = int(self.L.sp_collect_recurrent(
            self.h, _ptr(self.maps, ctypes.c_uint8), _ptr(self.scalars, ctypes.c_float),
            _ptr(self.masks, ctypes.c_uint8), _ptr(self.agents, ctypes.c_int), _ptr(self.game_ids, ctypes.c_int),
            len(self.masks), _ptr(self.roots, ctypes.c_uint8), _ptr(self.states, ctypes.c_float),
            _ptr(self.lengths, ctypes.c_int), _ptr(self.sequence_maps, ctypes.c_uint8),
            _ptr(self.sequence_scalars, ctypes.c_float), _ptr(self.sequence_masks, ctypes.c_uint8),
            _ptr(self.sequence_contexts, ctypes.c_float)))
        return self._pending

    def feed(self, logits: np.ndarray, values: np.ndarray, *, policy_updates: np.ndarray | None = None,
             next_states: np.ndarray | None = None) -> None:
        """Supply six-action policies and values for the last collect().

        Values guide search and form the ordinary n-step replay targets. Legal moves come from
        the simulator's action mask; logits do not change the policy's action capabilities.
        """
        lg = np.ascontiguousarray(logits, np.float32)
        v = np.ascontiguousarray(values, np.float32)
        updates = None if policy_updates is None else np.ascontiguousarray(policy_updates, np.int64)
        if v.shape != (self._pending,):
            raise ValueError(f"values must have shape ({self._pending},), got {v.shape}")
        if lg.shape != (self._pending, P.N_ACTIONS):
            raise ValueError(f"logits must have shape ({self._pending}, {P.N_ACTIONS}), got {lg.shape}")
        if not np.isfinite(lg).all() or not np.isfinite(v).all():
            raise ValueError("logits and values must be finite")
        if updates is not None and updates.shape != v.shape:
            raise ValueError(f"policy_updates must have shape {v.shape}, got {updates.shape}")
        if self.recurrent and next_states is None:
            raise ValueError("recurrent self-play requires returned hidden states")
        states = None if next_states is None else np.ascontiguousarray(next_states, np.float32)
        if states is not None and (states.shape != (self._pending, 256) or not np.isfinite(states).all()):
            raise ValueError("next_states must be finite [requests, 256] values")
        self.L.sp_feed_recurrent(self.h, _ptr(lg, ctypes.c_float), _ptr(v, ctypes.c_float), len(v),
                                None if updates is None else _ptr(updates, ctypes.c_int64),
                                None if states is None else _ptr(states, ctypes.c_float))
        self._pending = 0

    def watch(self, g: int) -> dict:
        """A read-only snapshot of arena G, including the last moves actually executed.

        Call from the actor's thread between collect/feed calls. No physics, search, replay
        or RNG is advanced. In-progress lookahead never replaces the actual world shown.
        """
        if not 0 <= g < self.games:
            raise ValueError(f"arena must be between 0 and {self.games - 1}")
        return json.loads(self.L.sp_watch(self.h, g))

    def info(self, g: int) -> dict | None:
        """The result of round G if it is over, else None."""
        if not self.L.sp_game_info(self.h, g, _ptr(self._info, ctypes.c_float)):
            return None
        x = self._info
        n = int(x[0])
        return {"n": n, "decisions": int(x[1]), "size": float(x[2]),
                "agents": [int(x[3 + 4 * p]) for p in range(n)], "outcome": [float(x[4 + 4 * p]) for p in range(n)],
                "died": [bool(x[5 + 4 * p]) for p in range(n)], "kills": [int(x[6 + 4 * p]) for p in range(n)]}

    def stats(self) -> dict:
        out = np.zeros(4, np.int64)
        self.L.sp_stats(self.h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)))
        return dict(zip(("decisions", "searches", "sims", "samples"), out.tolist()))
