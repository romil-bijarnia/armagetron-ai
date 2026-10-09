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
        }
        i64p = ctypes.POINTER(ctypes.c_int64)
        sig.update({
            "sp_new": ([i32], vp), "sp_free": ([vp], None),
            "sp_config": ([vp, i32, i32, f32, f32, f32, f32, f32, f32, i32, i32, i32], None),
            "sp_set_ring": ([vp, i32, u8p, fp, u8p, fp, u8p, fp, fp, i64p, i64p, u8p, i64p, ctypes.c_int64], None),
            "sp_start": ([vp, i32, f32, i32, ip, ip, i32, ctypes.c_ulonglong], None),
            "sp_collect": ([vp, u8p, fp, u8p, ip, ip, i32], i32),
            "sp_feed": ([vp, fp, fp, i32], None),
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
    ("aux", np.float32, (2,)),
    ("prev", np.int64, ()),
    ("gen", np.int64, ()),
    ("ready", np.uint8, ()),
)


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
            self.counter[0] = 0

    @property
    def name(self) -> str | None:
        return self.shm.name if self.shm is not None else None

    def written(self) -> int:
        return int(self.counter[0])

    def window(self, margin: int = 64) -> np.ndarray:
        """Slots holding samples whose round is over (the oldest MARGIN are skipped: the writer is
        about to overwrite them)."""
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

    def prev_maps(self, idx: np.ndarray) -> np.ndarray:
        """The previous decision's maps of each sample (its own maps at a spawn or once overwritten)."""
        prev = self.prev[idx]
        slot = np.where(prev >= 0, prev % self.cap, idx)
        valid = (prev >= 0) & (self.gen[slot] == prev)
        slot = np.where(valid, slot, idx)
        return self.maps[slot]

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
                 max_decisions: int = 6000, aux_horizon: int = 40, threads: int = 8, max_players: int = 4):
        self.L = lib()
        self.games = games
        self.h = self.L.sp_new(games)
        self.L.sp_config(self.h, sims, macro, gamma, c_visit, c_scale, search_prob, temperature, gumbel,
                         max_decisions, aux_horizon, threads)
        cap = games * max_players
        self.maps = np.zeros((cap, 2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
        self.scalars = np.zeros((cap, P.N_SCALARS), np.float32)
        self.masks = np.zeros(cap, np.uint8)
        self.agents = np.zeros(cap, np.int32)
        self.game_ids = np.zeros(cap, np.int32)
        self._info = np.zeros(self.INFO, np.float32)
        self._rings = {}

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
                           ring.cap)

    def start(self, g: int, size: float, agents, learn, rotate: int = 0, seed: int = 0) -> None:
        a = np.ascontiguousarray(agents, np.int32)
        lr = np.ascontiguousarray([1 if x else 0 for x in learn], np.int32)
        self.L.sp_start(self.h, g, size, len(a), _ptr(a, ctypes.c_int), _ptr(lr, ctypes.c_int), rotate, seed)

    def collect(self) -> int:
        return int(self.L.sp_collect(self.h, _ptr(self.maps, ctypes.c_uint8), _ptr(self.scalars, ctypes.c_float),
                                     _ptr(self.masks, ctypes.c_uint8), _ptr(self.agents, ctypes.c_int),
                                     _ptr(self.game_ids, ctypes.c_int), len(self.masks)))

    def feed(self, logits: np.ndarray, values: np.ndarray) -> None:
        lg = np.ascontiguousarray(logits, np.float32)
        v = np.ascontiguousarray(values, np.float32)
        self.L.sp_feed(self.h, _ptr(lg, ctypes.c_float), _ptr(v, ctypes.c_float), len(v))

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
