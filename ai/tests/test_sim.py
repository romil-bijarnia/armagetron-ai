"""The teacher's simulator moves cycles, kills them and draws the network's view the way the engine does."""

import shutil
import subprocess
from pathlib import Path

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
        a.step(rng.integers(0, P.N_ACTIONS, 4))
    b = a.clone()
    acts = rng.integers(0, P.N_ACTIONS, (60, 4))
    for x in acts:
        a.step(x)
        b.step(x)
    sa, sb = a.state(), b.state()
    for k in sa:
        assert np.array_equal(sa[k], sb[k]), k
    ma, _ = a.observe(0)
    mb, _ = b.observe(0)
    assert np.array_equal(ma, mb)


@pytest.mark.parametrize("action, heading", [(P.ACTION_LEFT_BRAKE, (-1, 0)), (P.ACTION_RIGHT_BRAKE, (1, 0))])
def test_combined_turns_brake_and_obey_turn_delay(action, heading):
    s = Sim()
    s.reset(size=-3, n=2)
    s.step([P.ACTION_STRAIGHT, P.ACTION_STRAIGHT])
    assert s.mask(0) == (1 << P.N_ACTIONS) - 1
    unbraked = s.clone()
    unbraked.step([P.ACTION_LEFT if action == P.ACTION_LEFT_BRAKE else P.ACTION_RIGHT, P.ACTION_STRAIGHT])
    s.step([action, P.ACTION_STRAIGHT])
    st = s.state()
    assert (st["dx"][0], st["dy"][0]) == heading
    assert st["braking"][0] == 1 and st["brake_reservoir"][0] < 1
    assert st["speed"][0] < unbraked.state()["speed"][0]
    assert s.mask(0) == (1 << P.ACTION_STRAIGHT) | (1 << P.ACTION_BRAKE)
    s.step([P.ACTION_BRAKE, P.ACTION_STRAIGHT])
    assert s.mask(0) == (1 << P.N_ACTIONS) - 1
    assert (s.state()["dx"][0], s.state()["dy"][0]) == heading
    s.step([P.ACTION_STRAIGHT, P.ACTION_STRAIGHT])
    assert s.state()["braking"][0] == 0


def test_self_play_requires_six_action_finite_batches():
    sp = SelfPlay(1, sims=0, search_prob=0, temperature=0, threads=1)
    sp.start(0, -3, [0, 0], [False, False])
    n = sp.collect()
    assert n == 2
    with pytest.raises(ValueError, match=r"logits must have shape \(2, 6\)"):
        sp.feed(np.zeros((n, 4)), np.zeros(n))
    for logits, values, updates in (
        (np.zeros((n, 5)), np.zeros(n), None),
        (np.zeros((n - 1, P.N_ACTIONS)), np.zeros(n - 1), None),
        (np.zeros((n, P.N_ACTIONS)), np.zeros((n, 1)), None),
        (np.zeros((n, P.N_ACTIONS)), np.zeros(n), np.zeros(n - 1)),
        (np.zeros((n, P.N_ACTIONS)), np.full(n, np.nan), None),
        (np.full((n, P.N_ACTIONS), np.inf), np.zeros(n), None),
    ):
        with pytest.raises(ValueError):
            sp.feed(logits, values, policy_updates=updates)
    # Rejected inputs leave the pending batch intact for a valid six-action response.
    sp.feed(np.zeros((n, P.N_ACTIONS)), np.zeros(n))
    assert sp.collect() == n
    sp.feed(np.zeros((n, P.N_ACTIONS)), np.zeros(n))

@pytest.mark.skipif(shutil.which("clang++") is None and shutil.which("g++") is None, reason="no C++ compiler")
def test_self_play_macro_brakes_and_safe_kill_targets(tmp_path):
    here = Path(__file__).resolve().parent
    sim = here.parent / "sim"
    exe = tmp_path / "selfplay_check"
    cxx = shutil.which("clang++") or shutil.which("g++")
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(exe), str(here / "cpp/selfplay_check.cpp"),
                    str(sim / "tron.cpp"), str(sim / "observe.cpp"), str(sim / "fighter.cpp"),
                    str(sim / "selfplay.cpp")], check=True)
    subprocess.run([str(exe)], check=True)


@pytest.mark.skipif(not ENGINE_BIN.exists(), reason="the engine is not built (build-dedicated)")
@pytest.mark.parametrize("compound", [False, True])
def test_simulator_matches_the_engine(tmp_path, monkeypatch, compound, record_property):
    """Replay rounds recorded from the real engine: every death at the same decision, positions
    within centimetres, maps nearly identical."""
    from arma_ai import record as recording

    if compound:
        class CompoundDriver(recording.Scripted):
            def __init__(self, seed):
                super().__init__(seed)
                # The first decision at time zero ignores actions; then exercise both combinations.
                self.first = [P.ACTION_STRAIGHT, P.ACTION_LEFT_BRAKE, P.ACTION_BRAKE,
                              P.ACTION_RIGHT_BRAKE, P.ACTION_BRAKE]

            def __call__(self, mask):
                if self.first and (mask >> self.first[0]) & 1:
                    return self.first.pop(0)
                return super().__call__(mask)

        monkeypatch.setattr(recording, "Scripted", CompoundDriver)

    rounds = recording.record(rounds=3, slots=2, size=-3, seed=5, workdir=tmp_path / "rec", base_port=47960)
    seen_actions = set()
    max_opponent_state_error = np.zeros(4)
    for r in rounds:
        a = r.arrays()
        seen_actions.update(int(x) for x in a["actions"].ravel() if x >= 0)
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
                    assert np.allclose([st["dx"][i], st["dy"][i]], a["dirs"][t][i], atol=1e-6), (t, i)
                k = a["slot_of"][i]
                if k >= 0 and a["actions"][t][k] >= 0:
                    assert s.mask(i) == a["masks"][t][k], (t, i)
                    assert bool(st["braking"][i]) == bool(a["scalars"][t][k][3]), (t, i)
                    maps, scalars = s.observe(i)
                    cells.append((maps != a["maps"][t][k]).sum(axis=(1, 2)))
                    # Resources and turn waits refer to each enemy's actual cycle time, in
                    # the same sorted slots as its original position/heading token.
                    simulated = scalars[96:108].reshape(3, 4)
                    recorded = a["scalars"][t][k][96:108].reshape(3, 4)
                    opponent_error = np.abs(simulated - recorded)
                    max_opponent_state_error = np.maximum(max_opponent_state_error, opponent_error.max(axis=0))
                    np.testing.assert_allclose(simulated[:, [0, 2, 3]], recorded[:, [0, 2, 3]], atol=3e-4)
                    # The existing contact-physics approximation already differs by up to
                    # 0.0079 in original own-rubber scalar 1 on these replays (~9 mm position
                    # drift). Keep a specific 1%-of-budget bound for that physical quantity.
                    np.testing.assert_allclose(simulated[:, 1], recorded[:, 1], atol=0.01)
                    enemies = sorted((j for j in range(n) if j != i and a["alive"][t][j]),
                                     key=lambda j: np.linalg.norm(a["pos"][t][j] - a["pos"][t][i]))
                    for enemy_slot, j in enumerate(enemies[:3]):
                        other = a["slot_of"][j]
                        if other >= 0 and a["actions"][t][other] >= 0:
                            # Extraction itself must be exact: these are the same live cycle's
                            # already-existing own brake/rubber values, before any control.
                            np.testing.assert_array_equal(recorded[enemy_slot, :2],
                                                          a["scalars"][t][other][[2, 1]])
            s.step([max(int(a["actions"][t][a["slot_of"][i]]), 0) for i in range(n)])
        assert np.mean(cells, axis=0).max() < 2.0  # cells differing per map and observation, out of 4096
    if compound:
        assert {P.ACTION_LEFT_BRAKE, P.ACTION_RIGHT_BRAKE} <= seen_actions
    else:
        assert not ({P.ACTION_LEFT_BRAKE, P.ACTION_RIGHT_BRAKE} & seen_actions)
    record_property("max_opponent_state_error", " ".join(f"{value:.8g}" for value in max_opponent_state_error))


@pytest.mark.skipif(shutil.which("clang++") is None and shutil.which("g++") is None, reason="no C++ compiler")
def test_opponent_resources_break_aliasing_and_preserve_original_inputs(tmp_path):
    here = Path(__file__).resolve().parent
    sim = here.parent / "sim"
    executable = tmp_path / "opponent_observation_check"
    cxx = shutil.which("clang++") or shutil.which("g++")
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(executable),
                    str(here / "cpp/opponent_observation_check.cpp"),
                    str(sim / "tron.cpp"), str(sim / "observe.cpp")], check=True)
    subprocess.run([str(executable)], check=True)


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
            mask = torch.from_numpy(((sp.masks[:n, None] >> np.arange(net.pi.out_features)) & 1).astype(bool))
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
    allowed = (ring.mask[searched, None] >> np.arange(P.N_ACTIONS)) & 1
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
    logits = np.array([8.0] + [0.0] * (P.N_ACTIONS - 1), np.float32)  # long straight runs for n-step bootstrapping
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


def test_real_decision_values_feed_targets_without_search_node_values():
    """Search leaves have different values; n-step replay must use only real-decision values."""
    steps, gamma = 60, 0.999
    sp = SelfPlay(4, sims=4, search_prob=0.5, threads=2, value_steps=steps, gamma=gamma)
    ring = Ring(50_000)
    sp.set_ring(0, ring)
    for g in range(4):
        sp.start(g, -1.0, [0, 0], [True, True], seed=g + 1)
    logits = np.array([4.0] + [0.0] * (P.N_ACTIONS - 1), np.float32)
    seen_root = seen_search = False
    for _ in range(3000):
        n = sp.collect()
        if n:
            roots = sp.roots[:n].copy()
            seen_root |= bool(roots.any())
            seen_search |= bool((roots == 0).any())
            values = np.where(roots, 0.25, 0.9).astype(np.float32)
            sp.feed(np.tile(logits, (n, 1)), values)
        if len(ring.window()) > 200:
            break
    assert seen_root and seen_search
    vals = ring.value[ring.window()]
    assert np.any(np.isclose(vals, 0.25 * gamma**steps, atol=1e-5))
    assert not np.any(np.isclose(vals, 0.9 * gamma**steps, atol=1e-5))


def test_the_space_fighter_plays_without_a_network_and_does_not_crash_into_walls():
    from arma_ai.sim import AGENT_HUNTER

    sp = SelfPlay(2, sims=0, search_prob=0.0, threads=1, hunter_depth=2, max_decisions=220)
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
            mask = torch.from_numpy(((sp.masks[:n, None] >> np.arange(net.pi.out_features)) & 1).astype(bool))
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
    mask = torch.ones((2, net.pi.out_features), dtype=torch.bool)
    with torch.no_grad():
        logits, v, aux, own = net.heads(maps, scalars, mask)
        l2, v2 = net(maps, scalars, mask)
    assert own.shape == (2, 3, P.GRID, P.GRID)
    assert torch.allclose(logits, l2) and torch.allclose(v, v2)
    assert not any(name.startswith("own") for name, _ in layers_of(net))
