"""The Elo ladder: how strong the network really is, measured in the real game.

The teacher trains in the simulator; this plays real engine duels (the network alone, no search,
exactly as it plays inside the game) between the current network and a roster of fixed
opponents: the game's strongest built-in AI, the old brain (runs/main), and frozen snapshots and
exploiters from the league pool. Every result is kept in <run>/ladder.jsonl and all of them are
fitted together into Elo ratings (Bradley-Terry), anchored at 1000 for the built-in AI.

    arma-ladder                       # the current network against up to 5 opponents, 40 duels each
    arma-ladder --games 100 --size 0  # more duels, on the full-size arena
    arma-ladder --table               # just print the ratings from the results so far
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from .evaluate import evaluate

PROJECT = Path(__file__).resolve().parent.parent
BUILTIN = "builtin"
V1 = "v1"
ANCHOR = 1000.0


def player_name(path: Path | None) -> str:
    """A stable name for a checkpoint: pool files keep their own name, others are named by update."""
    if path is None:
        return BUILTIN
    if path.resolve() == (PROJECT / "runs/main/latest.pt").resolve():
        return V1
    if path.parent.name == "pool":
        return path.stem
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        return f"u{int(ck.get('update', 0)):07d}"
    except Exception:
        return path.stem


def fit(rows: list[dict]) -> dict[str, float]:
    """Bradley-Terry strengths by minorisation-maximisation, as Elo. Each pair gets one virtual
    drawn game so that a perfect score still has a finite rating."""
    score = defaultdict(float)
    games = defaultdict(float)
    players = set()
    for r in rows:
        a, b, n, s = r["a"], r["b"], r["games"], r["score"]
        if n <= 0:
            continue
        players.update((a, b))
        score[(a, b)] += s * n
        score[(b, a)] += (1 - s) * n
        games[(a, b)] += n
        games[(b, a)] += n
    for (a, b) in list(games):
        score[(a, b)] += 0.5
        games[(a, b)] += 1
    if not players:
        return {}
    gamma = {p: 1.0 for p in players}
    for _ in range(500):
        new = {}
        for i in players:
            wins = sum(score[(i, j)] for j in players if (i, j) in games)
            denom = sum(games[(i, j)] / (gamma[i] + gamma[j]) for j in players if (i, j) in games)
            new[i] = wins / denom if denom > 0 else gamma[i]
        # keep the scale fixed (geometric mean 1)
        g = math.exp(sum(math.log(max(v, 1e-12)) for v in new.values()) / len(new))
        gamma = {p: v / g for p, v in new.items()}
    elo = {p: 400 * math.log10(max(v, 1e-12)) for p, v in gamma.items()}
    anchor = elo.get(BUILTIN, elo.get(V1, 0.0) - 500 if V1 in elo else 0.0)
    return {p: round(v - anchor + ANCHOR, 1) for p, v in sorted(elo.items(), key=lambda kv: -kv[1])}


def load_rows(run: Path) -> list[dict]:
    f = run / "ladder.jsonl"
    if not f.exists():
        return []
    rows = []
    for line in f.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def roster(run: Path, n: int) -> list[Path | None]:
    """Who the current network plays: the built-in AI, the old brain, and a spread of the pool
    (newest snapshot, newest exploiter, and older snapshots)."""
    out: list[Path | None] = [None]
    v1 = PROJECT / "runs/main/latest.pt"
    if v1.exists():
        out.append(v1)
    pool = run / "pool"
    snaps = sorted(pool.glob("u*.pt"))
    exploiters = sorted(pool.glob("x*.pt"))
    if exploiters:
        out.append(exploiters[-1])
    if snaps:
        out.append(snaps[-1])
        older = snaps[:-1]
        k = max(0, n - len(out))
        if older and k:
            step = max(1, len(older) // k)
            out += older[::-1][::step][:k]
    return out[:max(n, 2)]


def table(ratings: dict[str, float], rows: list[dict]) -> str:
    games = defaultdict(int)
    for r in rows:
        games[r["a"]] += r["games"]
        games[r["b"]] += r["games"]
    lines = [f"{'player':<22}{'Elo':>8}{'duels':>8}"]
    for p, e in ratings.items():
        lines.append(f"{p:<22}{e:>8.0f}{games[p]:>8}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/v2")
    ap.add_argument("--checkpoint", type=Path, default=None, help="the network to rate (default: <run>/latest.pt)")
    ap.add_argument("--opponents", type=int, default=5)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--engines", type=int, default=4)
    ap.add_argument("--size", type=float, default=-3)
    ap.add_argument("--table", action="store_true", help="only print the ratings so far")
    args = ap.parse_args()

    run = PROJECT / args.run
    if not args.table:
        hero = args.checkpoint or run / "latest.pt"
        if not hero.exists():
            raise SystemExit(f"no checkpoint at {hero}")
        a = player_name(hero)
        with open(run / "ladder.jsonl", "a") as log:
            for opp in roster(run, args.opponents):
                b = player_name(opp)
                if b == a:
                    continue
                res = evaluate(hero, opp, args.games, args.engines, args.size, workdir=PROJECT / "runtime" / "ladder")
                row = {"time": time.time(), "a": a, "b": b, "games": res["rounds"], "score": res["win_rate"],
                       "wins": res["wins"], "draws": res["draws"], "losses": res["losses"], "size": args.size}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"{a} vs {b}: {res['win_rate']:.0%} over {res['rounds']} duels", flush=True)
    rows = load_rows(run)
    ratings = fit(rows)
    if ratings:
        (run / "ratings.json").write_text(json.dumps({"time": time.time(), "ratings": ratings}, indent=1))
        print(table(ratings, rows))
    else:
        print("no ladder results yet")


if __name__ == "__main__":
    main()
