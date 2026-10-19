"""Teacher defaults, batching, checkpoint protection and reported ratings."""

import pytest

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


def test_new_run_defaults_use_the_current_fighter():
    cfg = Config()
    assert cfg.run == "runs/blacklight"
    # The fighter trains for the close fight: half of its duels start in one, three rounds in
    # ten are against the cutting space fighter, and a win ten seconds away is worth 55% of one now.
    assert cfg.combat_prob == .5 and cfg.hunter_prob == .3
    assert cfg.gamma == .997 and cfg.gamma ** 200 == pytest.approx(.549, abs=.001)
    assert cfg.max_updates == 0


def test_actor_batches_agents_and_preserves_values_and_policy_updates():
    import numpy as np
    import torch
    from arma_ai import protocol as P
    from arma_ai.teach import Actor

    class FixedNet:
        def __init__(self, value):
            self.value = value

        def forward_sequence(self, maps, scal, mask, state, context=None):
            assert context is not None
            self.context = context.clone()
            return (torch.arange(P.N_ACTIONS, dtype=torch.float32).expand(*maps.shape[:2], -1),
                    torch.full(maps.shape[:2], self.value), state + self.value)

    class Requests:
        agents = np.array([0, 1, 0])
        game_ids = np.array([0, 1, 2])
        lengths = np.array([1, 2, 1])
        states = np.zeros((3, 256), np.float32)
        sequence_maps = np.zeros((3, 2, 2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
        sequence_scalars = np.zeros((3, 2, P.N_SCALARS), np.float32)
        sequence_masks = np.full((3, 2), 63, np.uint8)
        sequence_contexts = np.arange(3 * 2 * 7, dtype=np.float32).reshape(3, 2, 7)
        masks = np.full(3, 63, np.uint8)
        maps = np.zeros((3, 2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
        scalars = np.zeros((3, P.N_SCALARS), np.float32)

        def feed(self, logits, values, policy_updates=None, next_states=None):
            self.result = logits, values
            self.policy_updates = policy_updates
            self.next_states = next_states

    actor = Actor.__new__(Actor)
    actor.cfg = Config(device="cpu")
    actor.sp = Requests()
    actor.device = torch.device("cpu")
    actor.half = False
    actor.nets = {0: FixedNet(2.5), 1: FixedNet(-.75)}
    actor.model_updates = {0: 12, 1: 15}
    actor.game_models = {i: {agent: (actor.nets[agent], actor.model_updates[agent], 0)}
                         for i, agent in enumerate(actor.sp.agents)}
    actor.evaluate(3)
    assert actor.sp.policy_updates.tolist() == [12, 15, 12]
    logits, values = actor.sp.result
    assert logits.shape == (3, 6)
    np.testing.assert_array_equal(logits, np.tile(np.arange(6), (3, 1)))
    np.testing.assert_array_equal(values, [2.5, -.75, 2.5])
    np.testing.assert_array_equal(actor.sp.next_states[:, 0], values)
    np.testing.assert_array_equal(actor.nets[1].context, actor.sp.sequence_contexts[[1]])


def test_actor_retains_one_weight_version_until_the_life_ends():
    import random
    from collections import defaultdict
    from types import SimpleNamespace
    from arma_ai.teach import Actor

    starts = []
    actor = Actor.__new__(Actor)
    actor.cfg = Config(hunter_prob=0, exploiter_prob=0, combat_prob=0)
    actor.rng = random.Random(7)
    actor.sp = SimpleNamespace(start=lambda *args, **kwargs: starts.append((args, kwargs)))
    old, new = object(), object()
    actor.nets, actor.model_updates, actor.model_generations = {0: old}, {0: 12}, {0: 0}
    actor.seed = 0
    actor.kind, actor.game_models, actor.start_kinds = {}, {}, {}
    actor.round_ids = defaultdict(int)
    actor.start(0)
    actor.nets[0], actor.model_updates[0] = new, 15
    actor.start(1)
    assert actor.game_models[0][0] == (old, 12, 0)
    assert actor.game_models[1][0] == (new, 15, 0)
    assert starts[0][1]["generations"] == [0] * len(starts[0][0][2])


@pytest.mark.parametrize("field,capacity", [
    ("ring", 0), ("ring", 512), ("ring", 4096), ("exploiter_ring", 4096),
    ("ring", 4097), ("exploiter_ring", 4097),
])
def test_impossible_replay_capacity_fails_before_allocation(tmp_path, monkeypatch, field, capacity):
    import arma_ai.sim as sim
    import arma_ai.teach as teach

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("invalid replay settings reached simulator/model allocation")

    monkeypatch.setattr(sim, "build", unexpected_allocation)
    monkeypatch.setattr(teach, "make_net", unexpected_allocation)
    run = tmp_path / "invalid"
    cfg = Config(run=str(run), warmup=16, batch=8, **{field: capacity})
    with pytest.raises(ValueError, match=f"{field} must"):
        Learner(cfg)
    assert not run.exists()
