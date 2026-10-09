"""Head-to-head league: the latest checkpoint against a spread of earlier snapshots.

Prints each matchup's score and an Elo estimate relative to the oldest snapshot played, and
appends the results to runs/<run>/league.jsonl.

    arma-league                      # latest vs up to 5 snapshots, 100 duels each, small arena
    arma-league --games 200 --size -1.5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .evaluate import evaluate

PROJECT = Path(__file__).resolve().parent.parent


def elo_gap(score: float) -> float:
    score = min(max(score, 0.005), 0.995)
    return 400 * math.log10(score / (1 - score))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/v2")
    ap.add_argument("--checkpoint", type=Path, default=None, help="defaults to <run>/latest.pt")
    ap.add_argument("--opponents", type=int, default=5)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--engines", type=int, default=4)
    ap.add_argument("--size", type=float, default=-3)
    args = ap.parse_args()

    run = PROJECT / args.run
    hero = args.checkpoint or run / "latest.pt"
    pool = sorted((run / "pool").glob("*.pt"))
    if not pool:
        raise SystemExit("no snapshots yet")
    step = max(1, len(pool) // args.opponents)
    picks = pool[::step][-args.opponents:]
    if pool[0] not in picks:
        picks = [pool[0]] + picks[1:]
    rows = []
    for opp in picks:
        res = evaluate(hero, opp, args.games, args.engines, args.size, workdir=PROJECT / "runtime" / "league")
        res["elo_gap"] = round(elo_gap(res["win_rate"]), 1)
        rows.append(res)
        print(f"latest (u{res['update']}) vs {opp.stem}: {res['win_rate']:.3f} over {res['rounds']} duels "
              f"-> {res['elo_gap']:+.0f} Elo", flush=True)
    with open(run / "league.jsonl", "a") as f:
        f.write(json.dumps({"time": time.time(), "size_factor": args.size, "results": rows}) + "\n")


if __name__ == "__main__":
    main()
