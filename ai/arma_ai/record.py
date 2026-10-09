"""Record what the real engine does, tick by tick, so the teacher's simulator can be checked against it.

A headless engine runs rounds driven by a scripted, seeded random policy (turns, braking, the odd
drive straight into a wall so rubber and deaths get exercised). Every decision tick it records each
cycle's position, heading and speed (the engine's WORLD message), and for each driven cycle the
observation the network would see and the move that was sent back.

    arma-record --rounds 6 --slots 2 --size -3 --out runs/record.npz
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import protocol as P
from .engine import ArenaConfig, EnginePool, _parse_step

PROJECT = Path(__file__).resolve().parent.parent


@dataclass
class Round:
    bounds: tuple = (0.0, 0.0, 0.0, 0.0)
    times: list = field(default_factory=list)
    pos: list = field(default_factory=list)  # per tick: (n_cycles, 2)
    dirs: list = field(default_factory=list)  # (n_cycles, 2)
    speed: list = field(default_factory=list)  # (n_cycles,)
    alive: list = field(default_factory=list)  # (n_cycles,)
    actions: list = field(default_factory=list)  # (n_slots,) action sent this tick, -1 if none asked
    masks: list = field(default_factory=list)  # (n_slots,)
    maps: list = field(default_factory=list)  # (n_slots, N_MAPS, G, G), zeros if not asked
    scalars: list = field(default_factory=list)  # (n_slots, N_SCALARS)
    flags: list = field(default_factory=list)  # (n_slots,)
    slot_of: list = field(default_factory=list)  # cycle index -> slot

    def arrays(self) -> dict:
        return {"bounds": np.array(self.bounds, np.float32), "times": np.array(self.times, np.float64),
                "pos": np.array(self.pos, np.float64), "dirs": np.array(self.dirs, np.float64),
                "speed": np.array(self.speed, np.float64), "alive": np.array(self.alive, bool),
                "actions": np.array(self.actions, np.int8), "masks": np.array(self.masks, np.uint8),
                "maps": np.array(self.maps, np.uint8), "scalars": np.array(self.scalars, np.float32),
                "flags": np.array(self.flags, np.uint8), "slot_of": np.array(self.slot_of, np.int16)}


class Scripted:
    """A seeded random driver with a few habits that exercise the physics: mostly straight, turns
    in bursts, braking now and then, and sometimes a long run at a wall."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.mode = "cruise"
        self.left = 0

    def __call__(self, mask: int) -> int:
        r = self.rng
        if self.left <= 0:
            self.mode = r.choices(["cruise", "zigzag", "brake", "charge"], [0.5, 0.2, 0.15, 0.15])[0]
            self.left = r.randint(5, 40)
        self.left -= 1
        can = [a for a in (P.ACTION_LEFT, P.ACTION_RIGHT) if mask >> a & 1]
        if self.mode == "zigzag" and can and r.random() < 0.5:
            return r.choice(can)
        if self.mode == "brake":
            return P.ACTION_BRAKE if r.random() < 0.8 else P.ACTION_STRAIGHT
        if self.mode == "charge":
            return P.ACTION_STRAIGHT
        if can and r.random() < 0.06:
            return r.choice(can)
        return P.ACTION_STRAIGHT


class NetDriver:
    """A trained network at the wheel (for realistic, long rounds), with a little randomness so
    no two rounds are the same."""

    def __init__(self, checkpoint: Path, seed: int, noise: float = 0.03):
        import torch

        from .model import make_net, net_version
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        self.net = make_net(net_version(sd)).eval()
        self.net.load_state_dict(sd)
        self.torch = torch
        self.rng = random.Random(seed)
        self.noise = noise
        self.prev = None

    def __call__(self, mask: int, maps=None, scalars=None) -> int:
        torch = self.torch
        if self.rng.random() < self.noise:
            return self.rng.choice([a for a in range(P.N_ACTIONS) if mask >> a & 1])
        x = np.concatenate([maps, maps if self.prev is None else self.prev])[None]
        self.prev = np.array(maps)
        feats = np.concatenate([scalars, [(mask >> a) & 1 for a in range(P.N_ACTIONS)]]).astype(np.float32)[None]
        with torch.no_grad():
            a, _, _ = self.net.act(torch.from_numpy(x), torch.from_numpy(feats))
        return int(a[0])


def record(rounds: int = 4, slots: int = 2, size: float = -3, walls: float = 600, seed: int = 0,
           max_ticks: int = 4000, workdir: Path | None = None, driver: Path | None = None,
           base_port: int = 47900) -> list[Round]:
    arena = ArenaConfig(slots=slots, size_factor=size, walls_length=walls, extra={
        "NEURAL_SPECTATE": "1", "NEURAL_END_ROUND_WITHOUT_NEURAL": "1"})
    drivers = [NetDriver(driver, seed * 101 + k) if driver else Scripted(seed * 101 + k) for k in range(slots)]
    out: list[Round] = []
    with EnginePool([arena], workdir=workdir or PROJECT / "runtime" / "record", base_port=base_port) as pool:
        conn = pool.conns[0]
        conn.setblocking(True)
        cur: Round | None = None
        cur_id = None
        ids: list[int] = []
        while len(out) < rounds:
            mtype, payload = pool._read_msg(conn)
            if mtype == P.MSG_WORLD:
                rid, t, over, step_follows, lx, ly, hx, hy, n = P.WORLD_HEAD.unpack_from(payload, 0)
                off = P.WORLD_HEAD.size
                cyc = []
                for _ in range(n):
                    cyc.append(P.WORLD_CYCLE.unpack_from(payload, off))
                    off += P.WORLD_CYCLE.size
                if rid != cur_id:
                    if cur is not None and cur.times:
                        out.append(cur)
                        if len(out) >= rounds:
                            break
                    cur, cur_id = Round(bounds=(lx, ly, hx, hy)), rid
                    ids = [c[0] for c in cyc]
                    cur.slot_of = [c[1] if c[1] != 255 else -1 for c in cyc]
                by_id = {c[0]: c for c in cyc}
                if len(cur.times) < max_ticks:
                    cur.times.append(t)
                    cur.pos.append([(by_id[i][3], by_id[i][4]) if i in by_id else (np.nan, np.nan) for i in ids])
                    cur.dirs.append([(by_id[i][5], by_id[i][6]) if i in by_id else (np.nan, np.nan) for i in ids])
                    cur.speed.append([by_id[i][7] if i in by_id else np.nan for i in ids])
                    cur.alive.append([bool(by_id[i][2]) if i in by_id else False for i in ids])
                    cur.actions.append([-1] * slots)
                    cur.masks.append([0] * slots)
                    cur.maps.append(np.zeros((slots, P.N_MAPS, P.GRID, P.GRID), np.uint8))
                    cur.scalars.append(np.zeros((slots, P.N_SCALARS), np.float32))
                    cur.flags.append([0] * slots)
                if not step_follows:
                    conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, 0))
            elif mtype == P.MSG_STEP:
                st = _parse_step(0, payload)
                acts = [0] * slots
                for k, s in enumerate(st.slots):
                    if s.flags & (P.FLAG_SPAWNED | P.FLAG_DIED) and isinstance(drivers[k], NetDriver):
                        drivers[k].prev = None
                    if s.needs_action:
                        acts[k] = (drivers[k](s.mask, s.maps, s.scalars) if isinstance(drivers[k], NetDriver)
                                   else drivers[k](s.mask))
                    if cur is not None and len(cur.times) <= max_ticks and cur.times:
                        cur.flags[-1][k] = s.flags
                        if s.needs_action:
                            cur.actions[-1][k] = acts[k]
                            cur.masks[-1][k] = s.mask
                            cur.maps[-1][k] = s.maps
                            cur.scalars[-1][k] = s.scalars
                body = bytes([slots]) + bytes(acts)
                conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, len(body)) + body)
        if cur is not None and cur.times and len(out) < rounds:
            out.append(cur)
    return out


def save(rounds: list[Round], path: Path) -> None:
    data = {}
    for i, r in enumerate(rounds):
        for k, v in r.arrays().items():
            data[f"r{i}_{k}"] = v
    data["n_rounds"] = np.array(len(rounds))
    np.savez_compressed(path, **data)


def load(path: Path) -> list[dict]:
    z = np.load(path)
    return [{k[len(f"r{i}_"):]: z[k] for k in z.files if k.startswith(f"r{i}_")} for i in range(int(z["n_rounds"]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--size", type=float, default=-3)
    ap.add_argument("--walls", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--driver", type=Path, default=None, help="a checkpoint to drive with instead of the scripted driver")
    ap.add_argument("--out", type=Path, default=PROJECT / "runs" / "record.npz")
    args = ap.parse_args()
    rounds = record(args.rounds, args.slots, args.size, args.walls, args.seed, driver=args.driver)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(rounds, args.out)
    for i, r in enumerate(rounds):
        print(f"round {i}: {len(r.times)} ticks, t={r.times[0]:.3f}..{r.times[-1]:.3f}, bounds {r.bounds}")


if __name__ == "__main__":
    main()
