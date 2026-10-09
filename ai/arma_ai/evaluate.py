"""Measure a checkpoint: duels against the game's strongest built-in AI or against another checkpoint.

    arma-eval                                    # latest vs built-in AI, small arena
    arma-eval --size 0 --rounds 400              # standard 500 m arena
    arma-eval --opponent runs/main/pool/u000100.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import protocol as P
from .engine import ArenaConfig, EnginePool
from .play import Brain

PROJECT = Path(__file__).resolve().parent.parent


def evaluate(checkpoint: Path, opponent: Path | None, rounds: int, engines: int, size: float,
             sample: bool = False, workdir: Path | None = None) -> dict:
    hero = Brain(checkpoint, sample=sample)
    villain = Brain(opponent, sample=sample) if opponent else None
    arena = ArenaConfig(slots=2 if villain else 1, builtin_ais=0 if villain else 1, size_factor=size)
    wins = draws = losses = 0
    lengths = []
    seen_round = {}
    ticks = {}
    t0 = time.time()
    with EnginePool([arena] * engines, workdir=workdir or PROJECT / "runtime" / "eval") as pool:
        while wins + draws + losses < rounds:
            for st in pool.poll(max_wait=0.003):
                if seen_round.get(st.engine) != st.round_id:
                    seen_round[st.engine] = st.round_id
                    ticks[st.engine] = 0
                    # alternate which slot the hero drives so spawn positions don't bias the result
                    hero_slot = st.round_id % 2 if villain else 0
                    seen_round[(st.engine, "hero")] = hero_slot
                hero_slot = seen_round[(st.engine, "hero")]
                ticks[st.engine] += 1
                acts = hero.actions(st)
                if villain:
                    other = villain.actions(st)
                    acts = [acts[k] if k == hero_slot else other[k] for k in range(len(acts))]
                pool.act(st.engine, acts)
                h = st.slots[hero_slot]
                result = None
                if h.flags & P.FLAG_DIED:
                    result = "draw" if st.n_alive == 0 else "loss"
                elif st.round_over and h.alive:
                    result = "win" if h.flags & P.FLAG_WON else "draw"
                if result:
                    wins += result == "win"
                    draws += result == "draw"
                    losses += result == "loss"
                    lengths.append(ticks[st.engine] * 0.05)
    n = wins + draws + losses
    return {
        "checkpoint": str(checkpoint), "update": hero.update,
        "opponent": str(opponent) if opponent else "builtin", "size_factor": size,
        "rounds": n, "win_rate": round((wins + 0.5 * draws) / max(n, 1), 4),
        "wins": wins, "draws": draws, "losses": losses,
        "mean_round_s": round(float(np.mean(lengths)), 1) if lengths else 0.0,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/v2/latest.pt")
    ap.add_argument("--opponent", type=Path, default=None, help="checkpoint to play against (default: built-in AI)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--engines", type=int, default=4)
    ap.add_argument("--size", type=float, default=-3)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--log", type=Path, default=PROJECT / "runs/v2/eval.jsonl")
    args = ap.parse_args()
    res = evaluate(args.checkpoint, args.opponent, args.rounds, args.engines, args.size, args.sample)
    res["time"] = time.time()
    print(json.dumps(res))
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
