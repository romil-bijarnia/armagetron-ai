"""Record engine trajectories or demonstrations from its strongest built-in AI players.

A headless engine runs rounds driven by a scripted, seeded random policy (turns, braking, the odd
drive straight into a wall so rubber and deaths get exercised). Every decision tick it records each
cycle's position, heading and speed (the engine's WORLD message), and for each driven cycle the
observation the network would see and the move that was sent back.

    arma-record --rounds 6 --slots 2 --size -3 --out runs/record.npz
    arma-record --teacher-iq 100 --rounds 20 --slots 1 --builtin 1 --out runs/demonstrations.npz

Teacher mode records the native controller's executed legal action with observations captured
before any teacher acts. It runs that controller at the policy's 0.05-second decision boundaries;
normal between-boundary AI decisions and emergency help stay disabled for exact replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from . import protocol as P
from .engine import ArenaConfig, EnginePool, ENGINE_BIN, ENGINE_CONFIG, REPO_ROOT, SlotStep, Step, _parse_step

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
    metadata: dict = field(default_factory=dict)  # provenance and round completeness; not an observation

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
        from .play import PolicyHistory
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        self.net = make_net(net_version(sd)).eval()
        self.net.load_state_dict(sd)
        self.torch = torch
        self.rng = random.Random(seed)
        self.noise = noise
        self.history = PolicyHistory(self.net)

    def reset(self) -> None:
        self.history.clear()

    def __call__(self, mask: int, maps=None, scalars=None, *, step: Step,
                 slot: int = 0, player_id: int | None = None) -> int:
        """Consume one real observation, even when exploration overrides the policy's move."""
        torch = self.torch
        source = step.slots[slot]
        observation = SlotStep(source.flags, source.kills, mask, maps, scalars, source.teacher_action)
        own_step = Step(step.engine, step.round_id, step.tick, step.round_time, step.round_over,
                        step.n_alive, step.n_total, [observation])
        self.history.net = self.net
        identity = {0: player_id} if player_id is not None else None
        previous = self.history.begin(own_step, identities=identity)
        if previous is not None:
            return int(previous[0])
        if not observation.needs_action:
            self.history.finish(own_step, [0])
            return 0
        snapshot = self.history.snapshot(own_step, 0)
        feats = np.concatenate([snapshot.scalars, snapshot.mask]).astype(np.float32)[None]
        override = self.rng.random() < self.noise
        with torch.no_grad():
            action, _, _, state = self.net.act_step(torch.from_numpy(snapshot.maps)[None],
                torch.from_numpy(feats), snapshot.state, greedy=override, context=snapshot.context)
        chosen = (self.rng.choice(np.flatnonzero(snapshot.mask).tolist()) if override else int(action[0]))
        self.history.commit(own_step, 0, chosen, state)
        self.history.finish(own_step, [chosen])
        return chosen


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(arena: ArenaConfig, teacher_iq: int | None, driver: Path | None,
              seed: int, max_ticks: int) -> dict:
    metadata = {
        "schema_version": 1,
        "observation_version": P.OBSERVATION_VERSION,
        "n_scalars": P.N_SCALARS,
        "teacher_type": "engine_builtin" if teacher_iq is not None else "network" if driver else "scripted",
        "decision_interval": arena.decision_interval,
        "action_encoding": "steer_brake_6",
        "settings": arena.settings(),
        "max_ticks": max_ticks,
        "seed": seed,
    }
    if teacher_iq is not None:
        sources = ["src/tron/gNeural.cpp", "src/tron/gAIBase.cpp", "src/tron/gAICharacter.cpp",
                   "src/tron/gGame.cpp", "ai/arma_ai/record.py"]
        if arena.ai_characters is not None:
            sources.extend(["ai/arma_ai/bot_profiles.py", "ai/arma_ai/engine.py"])
        metadata.update({
            "iq": teacher_iq,
            "profile_selection": ("explicit_named_roles" if arena.ai_characters is not None
                                  else "strongest_available_builtin_per_team"),
            "observation_timing": "before_all_teacher_actions",
            "label_source": "native_executed_control",
            "teacher_schedule": "policy_decision_boundaries",
            "single_teacher_pre_action": True,
            "teacher_slots": arena.slots,
            "builtin_opponents": arena.builtin_ais,
            "emergency_help": False,
            "engine_sha256": _fingerprint(ENGINE_BIN),
            "teacher_profiles_sha256": (hashlib.sha256(arena.ai_characters.encode("ascii")).hexdigest()
                                        if arena.ai_characters is not None
                                        else _fingerprint(ENGINE_CONFIG / "aiplayers.cfg")),
            "source_fingerprint": {name: _fingerprint(REPO_ROOT / name) for name in sources},
            "engine_seed_controlled": False,
        })
    return metadata


def record(rounds: int = 4, slots: int = 2, size: float = -3, walls: float = 600, seed: int = 0,
           max_ticks: int = 4000, workdir: Path | None = None, driver: Path | None = None,
           base_port: int = 47900, teacher_iq: int | None = None,
           round_time_limit: float = 300, progress: Callable[[dict], None] | None = None,
           builtin_ais: int = 0, teacher_profile: str | None = None,
           opponent_profiles: tuple[str, ...] | None = None) -> list[Round]:
    if teacher_iq is not None and driver is not None:
        raise ValueError("teacher_iq and driver are mutually exclusive")
    if teacher_iq is not None and (type(teacher_iq) is not int or not 0 <= teacher_iq <= 100):
        raise ValueError("teacher_iq must be an integer from 0 to 100; 100 is the engine maximum")
    if teacher_iq is not None and (slots != 1 or builtin_ais < 1):
        raise ValueError("Teacher recording requires exactly one teacher slot and at least one built-in opponent")
    if type(builtin_ais) is not int or builtin_ais < 0 or builtin_ais + slots > 32:
        raise ValueError("builtin_ais must be nonnegative with at most 32 total players")
    if rounds < 1 or not 1 <= slots <= 32 or max_ticks < 1 or round_time_limit <= 0:
        raise ValueError("rounds, max_ticks and round_time_limit must be positive, with 1..32 slots")
    roster = None
    if teacher_profile is not None or opponent_profiles is not None:
        if slots != 1:
            raise ValueError("Named bot profiles require exactly one controlled slot")
        if teacher_profile is not None and teacher_iq is None:
            raise ValueError("A teacher profile requires native teacher mode")
        if opponent_profiles is not None and len(opponent_profiles) != builtin_ais:
            raise ValueError("The opponent profile count must match builtin_ais")
        from .bot_profiles import make_roster
        roster = make_roster(teacher_profile or "stock_wall",
                             tuple(opponent_profiles) if opponent_profiles is not None
                             else ("stock_wall",) * builtin_ais)
    arena = ArenaConfig(slots=slots, builtin_ais=builtin_ais, size_factor=size, walls_length=walls,
                        ai_iq=100 if teacher_iq is None else teacher_iq, round_time_limit=round_time_limit, extra={
        "NEURAL_SPECTATE": "1", "NEURAL_END_ROUND_WITHOUT_NEURAL": "1",
        "NEURAL_TEACHER": "1" if teacher_iq is not None else "0", "AI_CHARACTER_FILE": "aiplayers.cfg"})
    if roster is not None:
        arena.extra.pop("AI_CHARACTER_FILE")
        arena = replace(arena, ai_characters=roster.text, controlled_profile=roster.controlled_name,
                        builtin_profiles=roster.builtin_names)
    metadata = _metadata(arena, teacher_iq, driver, seed, max_ticks)
    if roster is not None:
        metadata["bot_pool"] = roster.manifest
    drivers = [] if teacher_iq is not None else [
        NetDriver(driver, seed * 101 + k) if driver else Scripted(seed * 101 + k) for k in range(slots)]
    out: list[Round] = []
    decisions_collected = labels_collected = ticks_collected = 0

    def report_progress():
        if progress is not None:
            progress({"rounds": len(out), "rounds_total": rounds, "decisions": decisions_collected,
                      "labels": labels_collected, "ticks": ticks_collected})

    with EnginePool([arena], workdir=workdir or PROJECT / "runtime" / "record", base_port=base_port,
                    log_engines=True) as pool:
        conn = pool.conns[0]
        conn.setblocking(True)
        cur: Round | None = None
        cur_id = None
        ids: list[int] = []
        recorded_world = False
        world_time = None
        slot_ids: dict[int, int] = {}
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
                        report_progress()
                        if len(out) >= rounds:
                            break
                    cur, cur_id = Round(bounds=(lx, ly, hx, hy), metadata={**metadata, "round": {
                        "round_id": rid, "complete": False, "truncated": False,
                        "teacher_life_complete": False, "ticks_seen": 0, "teacher_labels": 0}}), rid
                    ids = [c[0] for c in cyc]
                    cur.slot_of = [c[1] if c[1] != 255 else -1 for c in cyc]
                    for d in drivers:
                        if isinstance(d, NetDriver):
                            d.reset()
                by_id = {c[0]: c for c in cyc}
                slot_ids = {c[1]: c[0] for c in cyc if c[1] != 255 and c[2]}
                cur.metadata["round"]["ticks_seen"] += 1
                cur.metadata["round"]["complete"] |= bool(over)
                recorded_world = len(cur.times) < max_ticks
                world_time = t
                if recorded_world:
                    ticks_collected += 1
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
                else:
                    cur.metadata["round"]["truncated"] = True
                if not step_follows:
                    conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, 0))
            elif mtype == P.MSG_STEP:
                st = _parse_step(0, payload)
                if st.round_id != cur_id or st.round_time != world_time:
                    raise ValueError("STEP does not match the preceding WORLD observation")
                if len(st.slots) != slots:
                    raise ValueError("STEP slot count differs from the configured recording")
                acts = [0] * slots
                for k, s in enumerate(st.slots):
                    if teacher_iq is not None and cur is not None:
                        # Recording may stop after this teacher dies while several ordinary
                        # opponents remain alive. That finishes its life, not the full match.
                        terminal = (st.round_over or bool(s.flags & P.FLAG_WON)
                                    or bool(s.flags & P.FLAG_DIED and not s.alive))
                        cur.metadata["round"]["teacher_life_complete"] |= terminal
                    if teacher_iq is None and s.needs_action and s.teacher_action is not None:
                        raise ValueError("Unexpected native teacher label in a driver recording")
                    if drivers and isinstance(drivers[k], NetDriver):
                        acts[k] = drivers[k](s.mask, s.maps, s.scalars, step=st, slot=k,
                                             player_id=slot_ids.get(k))
                    elif s.needs_action:
                        if teacher_iq is not None:
                            if s.teacher_action is None:
                                raise ValueError("Teacher mode requested but engine sent an observation without a native label")
                            acts[k] = s.teacher_action
                        else:
                            acts[k] = drivers[k](s.mask)
                    if cur is not None and recorded_world:
                        cur.flags[-1][k] = s.flags
                        if s.needs_action:
                            decisions_collected += 1
                            cur.actions[-1][k] = acts[k]
                            cur.masks[-1][k] = s.mask
                            cur.maps[-1][k] = s.maps
                            cur.scalars[-1][k] = s.scalars
                            if teacher_iq is not None:
                                cur.metadata["round"]["teacher_labels"] += 1
                                labels_collected += 1
                if recorded_world and ticks_collected % 100 == 0:
                    report_progress()
                recorded_world = False
                body = bytes([slots]) + bytes(acts)
                conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, len(body)) + body)
        if cur is not None and cur.times and len(out) < rounds:
            out.append(cur)
            report_progress()
    return out


def save(rounds: list[Round], path: Path, metadata: dict | None = None) -> None:
    data = {}
    for i, r in enumerate(rounds):
        for k, v in r.arrays().items():
            data[f"r{i}_{k}"] = v
    data["n_rounds"] = np.array(len(rounds))
    shared = {k: v for k, v in rounds[0].metadata.items() if k != "round"} if rounds else {}
    if any({k: v for k, v in r.metadata.items() if k != "round"} != shared for r in rounds):
        raise ValueError("Cannot save rounds with different recording provenance together")
    if metadata:
        if any(key in shared and shared[key] != value for key, value in metadata.items()):
            raise ValueError("Supplied metadata conflicts with recording provenance")
        shared.update(metadata)
    shared["rounds"] = [r.metadata.get("round", {}) for r in rounds]
    data["metadata"] = np.array(json.dumps(shared, sort_keys=True))
    np.savez_compressed(path, **data)


def load(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as z:
        return [{k[len(f"r{i}_"):]: z[k] for k in z.files if k.startswith(f"r{i}_")} for i in range(int(z["n_rounds"]))]


def load_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if "metadata" not in z:
            return {}
        raw = z["metadata"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("Recording metadata must be a scalar Unicode JSON string")
        metadata = json.loads(str(raw.item()))
        if not isinstance(metadata, dict):
            raise ValueError("Recording metadata must contain a JSON object")
        return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--slots", type=int, default=None, help="controlled slots (default: one teacher or two drivers)")
    ap.add_argument("--builtin", type=int, default=None, help="ordinary built-in opponents (default: one for teacher mode)")
    ap.add_argument("--size", type=float, default=-3)
    ap.add_argument("--walls", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--driver", type=Path, default=None, help="a checkpoint to drive with instead of the scripted driver")
    source.add_argument("--teacher-iq", type=int, default=None, help="record native built-in AI actions; 100 is the maximum")
    ap.add_argument("--teacher-profile", help="named native teacher from the candidate bot pool; implies --teacher-iq 100")
    ap.add_argument("--opponent-profile", action="append", default=None,
                    help="named built-in opponent; repeat for a four-player roster")
    ap.add_argument("--max-ticks", type=int, default=4000)
    ap.add_argument("--round-time-limit", type=float, default=300)
    ap.add_argument("--out", type=Path, default=PROJECT / "runs" / "record.npz")
    args = ap.parse_args()
    teacher_iq = args.teacher_iq if args.teacher_iq is not None else 100 if args.teacher_profile else None
    if args.teacher_profile and args.driver is not None:
        ap.error("--teacher-profile and --driver are mutually exclusive")
    slots = args.slots if args.slots is not None else 1 if teacher_iq is not None else 2
    builtin = (args.builtin if args.builtin is not None else len(args.opponent_profile)
               if args.opponent_profile is not None else int(teacher_iq is not None))
    rounds = record(args.rounds, slots, args.size, args.walls, args.seed, driver=args.driver,
                    teacher_iq=teacher_iq, max_ticks=args.max_ticks, round_time_limit=args.round_time_limit,
                    builtin_ais=builtin, teacher_profile=args.teacher_profile,
                    opponent_profiles=None if args.opponent_profile is None else tuple(args.opponent_profile))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(rounds, args.out)
    for i, r in enumerate(rounds):
        print(f"round {i}: {len(r.times)} ticks, t={r.times[0]:.3f}..{r.times[-1]:.3f}, bounds {r.bounds}")


if __name__ == "__main__":
    main()
