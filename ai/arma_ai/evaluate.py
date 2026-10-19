"""Measure a checkpoint: duels against the game's strongest built-in AI or against another checkpoint.

    arma-eval                                    # latest vs built-in AI, small arena
    arma-eval --size 0 --rounds 400              # standard 500 m arena
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from . import protocol as P
from .engine import ArenaConfig, EnginePool
from .play import Brain

PROJECT = Path(__file__).resolve().parent.parent


class EvaluationCancelled(Exception):
    """A benchmark worker was asked to stop; keep the games already scored."""


class _EvaluationPool(EnginePool):
    """Keep engine startup and socket reads inside an evaluation's wall-clock budget."""

    def __init__(self, *args, deadline: float | None = None, **kwargs):
        self.deadline = deadline
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            # EnginePool can fail while connecting, before its context manager is entered.
            if hasattr(self, "procs"):
                self.close()
            raise

    def _remaining(self, fallback: float) -> float:
        if self.deadline is None:
            return fallback
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("evaluation time budget exhausted")
        return min(fallback, remaining)

    def _accept_all(self, timeout: float = 60.0) -> None:
        super()._accept_all(timeout=self._remaining(timeout))

    def _read_msg(self, conn):
        if self.deadline is not None:
            conn.settimeout(self._remaining(120.0))
        return super()._read_msg(conn)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.sel.close()


def evaluate(checkpoint: Path, opponent: Path | None, rounds: int, engines: int, size: float,
             sample: bool = False, workdir: Path | None = None, *, time_budget_s: float | None = None,
             progress: Callable[[dict], None] | None = None, device: str | None = None,
             base_port: int = 47000, builtin_profiles: tuple[str, ...] | None = None) -> dict:
    """Evaluate fixed weights, preserving completed games when a budget or cancellation stops it.

    Existing callers can omit the keyword-only options. ``win_rate`` retains its historic
    meaning (wins plus half of draws divided by games); ``outright_win_rate`` counts wins only.
    Progress and final records identify the deployment policy: real engine, no search, and
    either greedy or sampled moves. The inspector freezes its checkpoint before calling this.
    Combat counts use only completed rounds. Kill credit comes from the engine's enemy-influence
    rule; it does not by itself establish aggressive intent.
    """
    if rounds <= 0 or engines <= 0:
        raise ValueError("rounds and engines must be positive")
    if not math.isfinite(size):
        raise ValueError("arena size must be finite")
    if time_budget_s is not None and (not math.isfinite(time_budget_s) or time_budget_s < 0):
        raise ValueError("time budget must be finite and nonnegative")
    arena = ArenaConfig(slots=2 if opponent else 1, builtin_ais=0 if opponent else 1, size_factor=size)
    roster = None
    if builtin_profiles is not None:
        if opponent is not None:
            raise ValueError("Named built-in profiles cannot be combined with a checkpoint opponent")
        if not 1 <= len(builtin_profiles) <= 3:
            raise ValueError("Choose one to three named built-in opponents")
        from .bot_profiles import make_roster
        roster = make_roster("stock_wall", tuple(builtin_profiles))
        arena = replace(arena, builtin_ais=len(builtin_profiles), ai_characters=roster.text,
                        controlled_profile=roster.controlled_name, builtin_profiles=roster.builtin_names)
    wins = draws = losses = 0
    lengths = []
    win_lengths = []
    credited_kills = deaths = surviving_kill_rounds = wins_with_kill = 0
    round_kills: dict[tuple[int, int], int] = {}
    last_tick: dict[tuple[int, int], int] = {}
    scored: set[tuple[int, int]] = set()
    started_at, started = time.time(), time.monotonic()
    deadline = None if time_budget_s is None else started + time_budget_s
    hero = villain = None

    def record(status: str = "running", reason: str | None = None) -> dict:
        n = wins + draws + losses
        now = time.time()
        return {
            "kind": "fixed_opponent_benchmark", "schema_version": 1,
            "checkpoint": str(checkpoint), "update": hero.update if hero is not None else None,
            "opponent": str(opponent) if opponent else "builtin",
            "bot_pool": roster.manifest if roster is not None else None,
            "opponent_update": villain.update if villain is not None else None,
            "environment": "real_engine", "search": False,
            "sample": sample, "policy_mode": "sampled" if sample else "greedy",
            "arena": asdict(arena), "size_factor": size, "engines": engines,
            "rounds_requested": rounds, "rounds_completed": n, "rounds": n,
            "win_rate": round((wins + 0.5 * draws) / n, 4) if n else None,
            "score": round((wins + 0.5 * draws) / n, 4) if n else None,
            "outright_win_rate": round(wins / n, 4) if n else None,
            "wins": wins, "draws": draws, "losses": losses,
            "mean_round_s": round(float(np.mean(lengths)), 1) if lengths else 0.0,
            "combat": {
                "completed_rounds": n, "credited_kills": credited_kills, "deaths": deaths,
                "surviving_kill_rounds": surviving_kill_rounds,
                "surviving_kill_rate": round(surviving_kill_rounds / n, 4) if n else None,
                "wins_with_credited_kill": wins_with_kill,
                "wins_without_credited_kill": wins - wins_with_kill,
                "kills_per_round": round(credited_kills / n, 4) if n else None,
                "mean_win_s": round(float(np.mean(win_lengths)), 1) if win_lengths else None,
            },
            "started_at": started_at, "finished_at": now if status != "running" else None,
            "time": now, "wall_s": round(time.monotonic() - started, 2),
            "time_budget_s": time_budget_s, "status": status,
            "partial": n < rounds, "stopped_reason": reason,
        }

    status, reason = "complete", None
    try:
        hero = Brain(checkpoint, device=device, sample=sample)
        villain = Brain(opponent, device=device, sample=sample) if opponent else None
        if progress:
            progress(record())
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("evaluation time budget exhausted")
        with _EvaluationPool([arena] * engines, workdir=workdir or PROJECT / "runtime" / "eval",
                             deadline=deadline, base_port=base_port, log_engines=roster is not None) as pool:
            while wins + draws + losses < rounds:
                remaining = 120.0 if deadline is None else deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("evaluation time budget exhausted")
                for st in pool.poll(max_wait=0.003, timeout=remaining):
                    # Alternate checkpoint-vs-checkpoint spawns; built-in duels have one neural slot.
                    hero_slot = st.round_id % 2 if villain else 0
                    acts = hero.actions(st)
                    if villain:
                        other = villain.actions(st)
                        acts = [acts[k] if k == hero_slot else other[k] for k in range(len(acts))]
                    pool.act(st.engine, acts)
                    key = (st.engine, st.round_id)
                    if key in scored:
                        continue
                    if st.tick <= last_tick.get(key, -1):
                        continue
                    last_tick[key] = st.tick
                    h = st.slots[hero_slot]
                    round_kills[key] = round_kills.get(key, 0) + h.kills
                    result = None
                    if h.flags & P.FLAG_DIED:
                        result = "draw" if st.n_alive == 0 else "loss"
                    elif st.round_over:
                        result = ("win" if h.flags & P.FLAG_WON else
                                  "draw" if h.alive or st.n_alive == 0 else "loss")
                    if result:
                        scored.add(key)
                        wins += result == "win"
                        draws += result == "draw"
                        losses += result == "loss"
                        lengths.append(max(0.0, st.round_time))
                        kills = round_kills.pop(key)
                        last_tick.pop(key)
                        credited_kills += kills
                        deaths += not h.alive
                        surviving_kill_rounds += h.alive and kills > 0
                        wins_with_kill += result == "win" and kills > 0
                        if result == "win":
                            win_lengths.append(max(0.0, st.round_time))
                        if progress:
                            progress(record())
                        if wins + draws + losses >= rounds:
                            break
    except EvaluationCancelled:
        status, reason = "cancelled", "cancelled"
    except (TimeoutError, socket.timeout):
        status, reason = "time_limit", "time_budget" if deadline is not None else "engine_timeout"
    except RuntimeError as exc:
        if isinstance(exc.__cause__, socket.timeout) and deadline is not None:
            status, reason = "time_limit", "time_budget"
        else:
            raise
    result = record(status, reason)
    if progress:
        progress(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/blacklight/latest.pt")
    ap.add_argument("--opponent", type=Path, default=None, help="checkpoint to play against (default: built-in AI)")
    ap.add_argument("--builtin-profile", action="append", default=None,
                    help="named stock-bot candidate opponent; repeat for up to three opponents")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--engines", type=int, default=1,
                    help="default 1; parallel deterministic engines can repeat the same trajectories")
    ap.add_argument("--size", type=float, default=-3)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--time-budget", type=float, default=None, help="wall-clock seconds; save any completed games")
    ap.add_argument("--device", default=None, help="inference device, e.g. cpu")
    ap.add_argument("--log", type=Path, default=PROJECT / "runs/blacklight/eval.jsonl")
    args = ap.parse_args()
    if args.opponent is not None and args.builtin_profile is not None:
        ap.error("--opponent and --builtin-profile are mutually exclusive")
    res = evaluate(args.checkpoint, args.opponent, args.rounds, args.engines, args.size, args.sample,
                   time_budget_s=args.time_budget, device=args.device,
                   builtin_profiles=None if args.builtin_profile is None else tuple(args.builtin_profile))
    res["time"] = time.time()
    print(json.dumps(res))
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
