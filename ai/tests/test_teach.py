"""The teacher's bookkeeping: Elo fitting for the ladder and the league pool's pruning."""

from pathlib import Path

from arma_ai.ladder import BUILTIN, fit
from arma_ai.teach import Config, Learner


def test_elo_fit_orders_players_and_anchors_the_builtin_ai():
    rows = [
        {"a": "u0000100", "b": BUILTIN, "games": 100, "score": 0.76},  # ~ +200
        {"a": "u0000200", "b": "u0000100", "games": 100, "score": 0.76},
        {"a": "u0000200", "b": BUILTIN, "games": 100, "score": 0.91},
    ]
    r = fit(rows)
    assert r[BUILTIN] == 1000.0
    assert r["u0000200"] > r["u0000100"] > r[BUILTIN]
    assert 120 < r["u0000100"] - r[BUILTIN] < 260
    # a perfect score still gets a finite rating
    r2 = fit([{"a": "x", "b": BUILTIN, "games": 20, "score": 1.0}])
    assert 1000 < r2["x"] < 2000


def test_pool_pruning_keeps_new_and_spread_out_old(tmp_path: Path):
    pool = tmp_path / "pool"
    pool.mkdir()
    for u in range(100):
        (pool / f"u{u:07d}.pt").write_bytes(b"")
    learner = Learner.__new__(Learner)  # only the pruning, no networks or processes
    learner.cfg = Config(pool_keep=10)
    learner.run_dir = tmp_path
    learner.prune_pool()
    left = sorted(p.name for p in pool.glob("u*.pt"))
    assert len(left) == 10
    assert left[-5:] == [f"u{u:07d}.pt" for u in range(95, 100)]  # the newest half
    assert left[0] == "u0000000.pt"  # the oldest survives as part of the spread
