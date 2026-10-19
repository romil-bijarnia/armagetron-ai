"""Combat starts are varied, physically valid and use the ordinary observation/training path."""

import ctypes
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from arma_ai.curriculum import combat_start
from arma_ai.sim import SelfPlay, Sim
from arma_ai.teach import Actor, Config, parse_config


@pytest.mark.parametrize("size", [-5, -3, -1.5, 0])
def test_combat_starts_are_reproducible_varied_and_inside_arena(size):
    rng, again = random.Random(37), random.Random(37)
    kinds, headings, positions = set(), set(), set()
    side = 500 * 2 ** (size / 2)
    for _ in range(100):
        kind, poses = combat_start(size, rng)
        assert (kind, poses) == combat_start(size, again)
        kinds.add(kind)
        for x, y, dx, dy in poses:
            assert 0 < x < side and 0 < y < side
            assert abs(dx) + abs(dy) == 1 and dx * dy == 0
            headings.add((dx, dy))
            positions.add((x, y))
        assert 5 < np.linalg.norm(np.array(poses[0][:2]) - poses[1][:2]) < 55
    assert kinds == {"parallel", "crossing", "head_on"}
    assert len(headings) == 4 and len(positions) == 200


def test_custom_starts_match_physics_observations_and_reset_history():
    sp = SelfPlay(1, sims=0, search_prob=0, temperature=0, threads=1)
    rng = random.Random(2)
    for _ in range(4):
        _, poses = combat_start(-3, rng)
        sp.start(0, -3, [0, 0], [False, False], placements=poses)
        reference = Sim()
        reference.reset(-3, 2)
        for p, pose in enumerate(poses):
            reference.place(p, *pose)
        n = sp.collect()
        assert n == 2
        for p in range(2):
            maps, scalars = reference.observe(p)
            np.testing.assert_array_equal(sp.maps[p, :4], maps)
            np.testing.assert_array_equal(sp.maps[p, 4:], maps)
            np.testing.assert_array_equal(sp.scalars[p], scalars)
            assert sp.masks[p] == reference.mask(p)
        for tick in range(5):
            logits = np.full((n, 6), -100, np.float32)
            logits[:, 0] = 100
            sp.feed(logits, np.zeros(n, np.float32))
            n = sp.collect()
            reference.step([0, 0])
            state = reference.state()
            for p, player in enumerate(sp.watch(0)["players"]):
                assert player["x"] == pytest.approx(float(state["x"][p]), abs=1e-4)
                assert player["y"] == pytest.approx(float(state["y"][p]), abs=1e-4)
                assert player["alive"] == bool(state["alive"][p])


@pytest.mark.parametrize("poses", [
    [(50, 50, 1, 0)],
    [(50, 50, float("nan"), 0), (60, 60, 0, 1)],
    [(50, 50, 1, 1), (60, 60, 0, 1)],
    [(0, 50, 1, 0), (60, 60, 0, 1)],
    [(50, 50, 1, 0), (50, 50, 0, 1)],
])
def test_invalid_combat_spawns_are_rejected(poses):
    sp = SelfPlay(1, sims=0, threads=1)
    with pytest.raises(ValueError):
        sp.start(0, -3, [0, 0], [False, False], placements=poses)


def test_spawn_placement_cannot_change_a_round_after_observation():
    sp = SelfPlay(1, sims=0, threads=1)
    sp.start(0, -3, [0, 0], [False, False])
    sp.collect()
    before = sp.watch(0)
    poses = np.array([(50, 50, 1, 0), (60, 60, 0, 1)], np.float32)
    accepted = sp.L.sp_place_start(sp.h, 0, 2, poses.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    assert accepted == 0 and sp.watch(0) == before


@pytest.mark.parametrize("probability", [0, 1])
def test_actor_applies_curriculum_to_duels_without_changing_opponents(probability):
    actor = Actor.__new__(Actor)
    actor.cfg = Config(combat_prob=probability, hunter_prob=1, exploiter_prob=0)
    actor.rng = random.Random(7)
    actor.seed = 0
    actor.nets, actor.kind, actor.start_kinds = {0: object()}, {}, {}
    actor.model_updates, actor.model_generations, actor.game_models = {0: 0}, {0: 0}, {}
    actor.round_ids = defaultdict(int)
    calls = []
    actor.sp = SimpleNamespace(start=lambda *args, **kwargs: calls.append((args, kwargs)))
    for g in range(20):
        actor.start(g)
        args, kwargs = calls[-1]
        assert sorted(args[2]) == [-1, 0]  # still main vs the space fighter
        assert sorted(args[3]) == [0, 1]
        assert (kwargs["placements"] is not None) == bool(probability)
        assert (actor.start_kinds[g] != "standard") == bool(probability)


@pytest.mark.parametrize("value", ["-1", "1.1", "nan", "inf"])
def test_invalid_curriculum_probability_fails_before_training(value):
    with pytest.raises(SystemExit):
        parse_config(["--combat-prob", value])
