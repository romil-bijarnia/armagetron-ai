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


def test_samples_get_targets_long_before_their_round_ends():
    """n-step targets: a sample is ready value_steps decisions later, valued at the network's estimate
    then; only a life's last value_steps decisions wait for the outcome. Nothing is left unscored."""
    steps, gamma, v = 60, 0.999, 0.5  # (at least aux_horizon + 1: the aux targets need that much)
    sp = SelfPlay(4, sims=4, search_prob=0.25, threads=2, value_steps=steps, gamma=gamma)
    ring = Ring(50_000)
    sp.set_ring(0, ring)
    for g in range(4):
        sp.start(g, -1.0, [0, 0], [True, True], seed=g + 1)  # a big arena: driving straight takes a while
    logits = np.array([4.0, 0.0, 0.0, 0.0], np.float32)  # mostly straight on
    ready_before_any_end = 0
    finished = set()
    for _ in range(20_000):
        n = sp.collect()
        if n:
            sp.feed(np.tile(logits, (n, 1)), np.full(n, v, np.float32))
        for g in range(4):
            if g not in finished and sp.info(g) is not None:
                finished.add(g)
        if not finished:
            ready_before_any_end = len(ring.window())
        if len(finished) == 4:
            break
    assert len(finished) == 4
    assert ready_before_any_end > 0  # targets arrive while every round is still going
    total = ring.written()
    assert total < ring.cap and np.all(ring.ready[:total] == 1)  # every sample got its targets
    vals = ring.value[:total]
    boot = np.isclose(vals, v * gamma**steps, atol=1e-5)
    assert boot.mean() > 0.3  # most samples are valued 60 decisions on ...
    assert np.all(boot | (np.abs(vals) <= 1.0))  # ... the rest by the outcome
    assert np.all((ring.aux[:total] >= 0) & (ring.aux[:total] <= 1))


def test_requests_say_which_are_real_decisions_and_boots_feed_the_targets():
    """During the warm start the old network steers and the new one's values build the targets:
    collect() marks real decisions, and feed()'s BOOTS (not VALUES) end up in the n-step targets."""
    steps, gamma = 60, 0.999
    sp = SelfPlay(4, sims=4, search_prob=0.5, threads=2, value_steps=steps, gamma=gamma)
    ring = Ring(50_000)
    sp.set_ring(0, ring)
    for g in range(4):
        sp.start(g, -1.0, [0, 0], [True, True], seed=g + 1)
    logits = np.array([4.0, 0.0, 0.0, 0.0], np.float32)
    seen_root = seen_search = False
    for _ in range(3000):
        n = sp.collect()
        if n:
            roots = sp.roots[:n].copy()
            seen_root |= bool(roots.any())
            seen_search |= bool((roots == 0).any())
            sp.feed(np.tile(logits, (n, 1)), np.full(n, 0.9, np.float32), np.full(n, 0.25, np.float32))
        if len(ring.window()) > 200:
            break
    assert seen_root and seen_search
    vals = ring.value[ring.window()]
    assert np.any(np.isclose(vals, 0.25 * gamma**steps, atol=1e-5))
    assert not np.any(np.isclose(vals, 0.9 * gamma**steps, atol=1e-5))


def test_decisions_without_a_value_make_targets_reach_to_the_next_one():
    """Warm start: only some decisions get the new network's value (NaN for the rest). A sample's
    target then comes from the next decision that has one, a few decisions further on, and every
    sample still gets a target."""
    steps, gamma, v = 60, 0.999, 0.5
    sp = SelfPlay(4, sims=4, search_prob=0.25, threads=2, value_steps=steps, gamma=gamma)
    ring = Ring(50_000)
    sp.set_ring(0, ring)
    for g in range(4):
        sp.start(g, -1.0, [0, 0], [True, True], seed=g + 11)
    logits = np.array([4.0, 0.0, 0.0, 0.0], np.float32)
    rng = np.random.default_rng(0)
    finished = set()
    for _ in range(20_000):
        n = sp.collect()
        if n:
            boots = np.where(rng.random(n) < 1 / 8, v, np.nan).astype(np.float32)
            sp.feed(np.tile(logits, (n, 1)), np.full(n, 0.9, np.float32), boots)
        finished |= {g for g in range(4) if sp.info(g) is not None}
        if len(finished) == 4:
            break
    total = ring.written()
    assert len(finished) == 4 and np.all(ring.ready[:total] == 1)
    vals = ring.value[:total]
    from_value = vals[(vals > 0.1) & (vals <= v + 1e-6)]  # (outcomes score +1, -1 or 0, discounted a little)
    gaps = np.log(from_value / v) / np.log(gamma)  # targets built from a value: v * gamma^gap
    assert len(gaps) > 0 and gaps.min() > steps - 0.5 and np.median(gaps) < steps + 20
    assert not np.any(np.isclose(vals, 0.9 * gamma**steps, atol=1e-4))


def test_the_space_fighter_plays_without_a_network_and_does_not_crash_into_walls():
    from arma_ai.sim import AGENT_HUNTER

    sp = SelfPlay(2, sims=0, search_prob=0.0, threads=1, hunter_depth=2)
    for g in range(2):
        sp.start(g, -3.0, [AGENT_HUNTER, AGENT_HUNTER], [False, False], rotate=g, seed=g + 1)
    for _ in range(50):
        assert sp.collect() == 0  # nobody asks the network
        if all(sp.info(g) is not None for g in range(2)):
            break
    for g in range(2):
        info = sp.info(g)
        assert info is not None
        # two fighters playing for territory do not drive into the rim early: a round lasts
        assert info["decisions"] > 150
        assert sorted(info["outcome"]) in ([-1.0, 1.0], [0.0, 0.0])


def test_rounds_against_the_space_fighter_record_the_learner_only():
    torch.manual_seed(0)
    from arma_ai.model import PolicyNet
    from arma_ai.sim import AGENT_HUNTER

    net = PolicyNet().eval()
    sp = SelfPlay(4, sims=8, search_prob=0.5, threads=2, hunter_depth=1)
    ring = Ring(20_000)
    sp.set_ring(0, ring)
    for g in range(4):
        sp.start(g, -3.0, [0, AGENT_HUNTER], [True, False], rotate=g % 2, seed=g + 1)
    finished = 0
    for _ in range(4000):
        n = sp.collect()
        if n:
            assert np.all(sp.agents[:n] == 0)
            mask = torch.from_numpy(((sp.masks[:n, None] >> np.arange(4)) & 1).astype(bool))
            with torch.no_grad():
                lg, v = net(torch.from_numpy(sp.maps[:n]), torch.from_numpy(sp.scalars[:n]), mask)
            sp.feed(lg.numpy(), v.numpy())
        for g in range(4):
            info = sp.info(g)
            if info is not None:
                finished += 1
                assert info["agents"][1] == AGENT_HUNTER
                if finished < 6:
                    sp.start(g, -3.0, [0, AGENT_HUNTER], [True, False], seed=100 + finished)
        if finished >= 6:
            break
    assert finished >= 6
    ready = ring.window()
    assert len(ready) > 0
    assert np.all(ring.future[ready] >= 0)
    assert np.all(ring.heading[ready] <= 3)
    own, ok = ring.ownership(ready[:64])
    assert own.shape == (min(64, len(ready)), P.GRID, P.GRID) and own.max() <= 2
    assert ok.all()


def test_ownership_targets_turn_with_the_heading():
    sim = Sim()
    sim.reset(-3.0, 2)
    st = sim.state()
    x, y = float(st["x"][0]), float(st["y"][0])
    sim.place(0, x, y, 0.0, 1.0)
    up = sim.observe(0)[0][P.N_MAPS - 1].copy()
    sim.place(0, x, y, 1.0, 0.0)
    right = sim.observe(0)[0][P.N_MAPS - 1].copy()
    assert up.any() and not np.array_equal(up, right)
    # a quarter turn of the heading turns the map a quarter turn
    assert np.array_equal(np.rot90(up, 1), right)
    # and the ring turns a later map back into the sample's frame by the quadrants between them
    ring = Ring(4)
    ring.maps[0, P.N_MAPS - 1] = up
    ring.maps[1, P.N_MAPS - 1] = right
    ring.heading[0], ring.heading[1] = 0, 1
    ring.gen[0], ring.gen[1] = 0, 1
    ring.future[0], ring.future[1] = 1, -1
    own, ok = ring.ownership(np.array([0, 1]))
    assert ok.tolist() == [True, False]
    cls_up = np.where(up & 1, 1, np.where((up >> 1) & 1, 2, 0))
    assert np.array_equal(own[0], cls_up)


def test_the_network_predicts_an_ownership_map_the_game_never_loads():
    from arma_ai.export import layers_of
    from arma_ai.model import PolicyNet

    net = PolicyNet().eval()
    maps = torch.zeros((2, 2 * P.N_MAPS, P.GRID, P.GRID), dtype=torch.uint8)
    scalars = torch.zeros((2, P.N_SCALARS))
    mask = torch.ones((2, P.N_ACTIONS), dtype=torch.bool)
    with torch.no_grad():
        logits, v, aux, own = net.heads(maps, scalars, mask)
        l2, v2 = net(maps, scalars, mask)
    assert own.shape == (2, 3, P.GRID, P.GRID)
    assert torch.allclose(logits, l2) and torch.allclose(v, v2)
    assert not any(name.startswith("own") for name, _ in layers_of(net))
