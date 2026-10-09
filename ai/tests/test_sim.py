"""The teacher's simulator moves cycles, kills them and draws the network's view the way the engine does."""

import numpy as np
import pytest
import torch

from arma_ai import protocol as P
from arma_ai.engine import ENGINE_BIN
from arma_ai.sim import Ring, SelfPlay, Sim


def test_arena_spawns_and_a_left_turn():
    s = Sim()
    s.reset(size=-3, n=2)
    lx, ly, hx, hy = s.bounds()
    assert abs(hx - 176.7767) < 1e-3 and abs(hy - 176.7767) < 1e-3
    st = s.state()
    assert abs(st["x"][0] - 90.156) < 1e-2 and abs(st["y"][0] - 17.678) < 1e-2 and st["dy"][0] == 1
    s.step([0, 0])  # moves at time 0 are ignored, as in the game
    s.step([P.ACTION_LEFT, 0])
    st = s.state()
    assert (st["dx"][0], st["dy"][0]) == (-1, 0)  # left turns counter-clockwise
    assert abs(st["speed"][0] - 19.221) < 1e-2  # 5% slower after a turn, recovering towards 20 m/s
    assert s.mask(0) == 0b1001  # the turn delay: no turn at the very next decision


def test_clones_are_independent_and_deterministic():
    a = Sim()
    a.reset(size=-1.5, n=4)
    rng = np.random.default_rng(0)
    for _ in range(40):
        a.step(rng.integers(0, 4, 4))
    b = a.clone()
    acts = rng.integers(0, 4, (60, 4))
    for x in acts:
        a.step(x)
        b.step(x)
    sa, sb = a.state(), b.state()
    for k in sa:
        assert np.array_equal(sa[k], sb[k]), k
    ma, _ = a.observe(0)
    mb, _ = b.observe(0)
    assert np.array_equal(ma, mb)


@pytest.mark.skipif(not ENGINE_BIN.exists(), reason="the engine is not built (build-dedicated)")
def test_simulator_matches_the_engine(tmp_path):
    """Replay rounds recorded from the real engine: every death at the same decision, positions
    within centimetres, maps nearly identical."""
    from arma_ai.record import record

    rounds = record(rounds=3, slots=2, size=-3, seed=5, workdir=tmp_path / "rec", base_port=47960)
    for r in rounds:
        a = r.arrays()
        n = len(a["slot_of"])
        s = Sim()
        s.reset(-3, n)
        s.set_bounds(*a["bounds"])
        for i in range(n):
            s.place(i, *a["pos"][0][i], *a["dirs"][0][i])
        cells = []
        for t in range(len(a["times"])):
            st = s.state()
            for i in range(n):
                assert bool(st["alive"][i]) == bool(a["alive"][t][i]), (t, i)
                if a["alive"][t][i]:
                    assert np.hypot(st["x"][i] - a["pos"][t][i][0], st["y"][i] - a["pos"][t][i][1]) < 0.05, (t, i)
                k = a["slot_of"][i]
                if k >= 0 and a["actions"][t][k] >= 0:
                    maps, _ = s.observe(i)
                    cells.append((maps != a["maps"][t][k]).sum(axis=(1, 2)))
            s.step([max(int(a["actions"][t][a["slot_of"][i]]), 0) for i in range(n)])
        assert np.mean(cells, axis=0).max() < 2.0  # cells differing per map and observation, out of 4096


def test_self_play_writes_complete_samples():
    torch.manual_seed(0)
    from arma_ai.model import PolicyNet

    net = PolicyNet().eval()
    sp = SelfPlay(6, sims=8, search_prob=0.5, threads=2)
    ring = Ring(20_000)
    sp.set_ring(0, ring)
    for g in range(6):
        sp.start(g, -3.0, [0, 0], [True, True], rotate=g % 2, seed=g + 1)
    finished = 0
    for _ in range(3000):
        n = sp.collect()
        if n:
            mask = torch.from_numpy(((sp.masks[:n, None] >> np.arange(4)) & 1).astype(bool))
            with torch.no_grad():
                lg, v = net(torch.from_numpy(sp.maps[:n]), torch.from_numpy(sp.scalars[:n]), mask)
            sp.feed(lg.numpy(), v.numpy())
        for g in range(6):
            info = sp.info(g)
            if info is not None:
                finished += 1
                assert sorted(info["outcome"]) in ([-1.0, 1.0], [0.0, 0.0])
                sp.start(g, -3.0, [0, 0], [True, True], seed=100 + finished)
        if finished >= 6:
            break
    assert finished >= 6
    ready = ring.window()
    assert len(ready) > 0
    assert np.all(np.abs(ring.value[ready]) <= 1.0)
    searched = ready[ring.has_policy[ready] == 1]
    assert len(searched) > 0
    assert np.allclose(ring.policy[searched].sum(1), 1, atol=1e-4)
    # the search's policy never puts weight on a forbidden move
    allowed = (ring.mask[searched, None] >> np.arange(4)) & 1
    assert np.all(ring.policy[searched][allowed == 0] < 1e-6)
    assert np.all((ring.aux[ready] >= 0) & (ring.aux[ready] <= 1))
    prev = ring.prev_maps(ready[:50])
    assert prev.shape == (min(50, len(ready)), P.N_MAPS, P.GRID, P.GRID)
