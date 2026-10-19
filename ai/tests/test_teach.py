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
    # two rounds in ten involve the exploiter: half train it against the frozen target, half
    # train the main network against it
    assert cfg.exploiter_prob == .2 and cfg.exploiter_train_share == .5


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


def _matchup_actor(cfg, seed=7):
    """An actor with stub networks whose rounds are recorded instead of simulated."""
    import random
    from collections import defaultdict
    from types import SimpleNamespace
    from arma_ai.teach import Actor

    starts = []
    actor = Actor.__new__(Actor)
    actor.cfg = cfg
    actor.rng = random.Random(seed)
    actor.sp = SimpleNamespace(start=lambda g, size, agents, learn, **kwargs: starts.append((g, agents, learn, kwargs)))
    actor.nets = {0: object(), 1: object(), 2: object()}
    actor.model_updates = {0: 40, 1: 38, 2: 36}
    actor.model_generations = {0: 0, 1: 3, 2: 3}
    actor.seed = 0
    actor.kind, actor.game_models, actor.start_kinds = {}, {}, {}
    actor.round_ids = defaultdict(int)
    return actor, starts


def test_exploiter_rounds_split_between_its_own_training_and_the_main_network():
    from arma_ai.teach import AGENT_EXPLOITER, AGENT_MAIN, AGENT_TARGET

    actor, starts = _matchup_actor(Config(hunter_prob=0, exploiter_prob=1, combat_prob=0))
    for g in range(400):
        actor.start(g)
    own = {g for g in range(400) if actor.kind[g][0] == ("exploiter", None)}
    main = {g for g in range(400) if actor.kind[g][0] == ("exploiter", "main")}
    assert len(own) + len(main) == 400 and 150 <= len(own) <= 250
    for g, agents, learn, kwargs in starts:
        hero = actor.kind[g][1]
        if g in own:
            # the exploiter learns against the frozen target, which does not
            assert sorted(agents) == [AGENT_EXPLOITER, AGENT_TARGET]
            assert learn == [int(a == AGENT_EXPLOITER) for a in agents]
            assert agents[hero] == AGENT_EXPLOITER and kwargs["generations"] == [3, 3]
        else:
            # the main network learns against the current exploiter, which does not
            assert sorted(agents) == [AGENT_MAIN, AGENT_EXPLOITER]
            assert learn == [int(a == AGENT_MAIN) for a in agents]
            assert agents[hero] == AGENT_MAIN
            assert kwargs["generations"] == [3 if a == AGENT_EXPLOITER else 0 for a in agents]


def test_the_exploiter_share_stays_two_in_ten_and_the_split_is_a_setting():
    actor, _ = _matchup_actor(Config(combat_prob=0))  # three in ten space fighter, two in ten exploiter
    rounds = 4000
    for g in range(rounds):
        actor.start(g)
    kinds = [actor.kind[g][0][0] for g in range(rounds)]
    assert .17 * rounds <= kinds.count("exploiter") <= .23 * rounds
    assert .27 * rounds <= kinds.count("hunter") <= .33 * rounds
    for share, expected in ((1.0, {None}), (0.0, {"main"})):
        actor, _ = _matchup_actor(Config(hunter_prob=0, exploiter_prob=1, combat_prob=0, exploiter_train_share=share))
        for g in range(50):
            actor.start(g)
        assert {actor.kind[g][0][1] for g in range(50)} == expected


def _finished_actor(cfg):
    """An actor with three rounds in play: the main network against the exploiter, the exploiter
    against the frozen target, and an older exploiter still against the target."""
    from collections import defaultdict, deque
    from arma_ai.teach import Actor

    class Queue:
        def __init__(self):
            self.messages = []

        def put(self, message):
            self.messages.append(message)

    actor = Actor.__new__(Actor)
    actor.cfg = cfg
    actor.exploiter_gen = 3
    actor.exploiter_results = deque(maxlen=cfg.exploiter_games)
    actor.promotion_sent = False
    actor.results = defaultdict(list)
    actor.lengths = []
    actor.out_q = Queue()
    actor.kind = {0: (("exploiter", "main"), 0, [0, 1]), 1: (("exploiter", None), 1, [2, 1]),
                  2: (("exploiter", None), 0, [1, 2])}
    actor.game_models = {0: {0: (None, 40, 0), 1: (None, 38, 3)}, 1: {1: (None, 38, 3), 2: (None, 36, 3)},
                         2: {1: (None, 30, 2), 2: (None, 36, 3)}}
    return actor


def test_scores_against_the_exploiter_are_recorded_apart_from_its_own():
    actor = _finished_actor(Config(exploiter_games=4))
    actor.finished(0, {"outcome": [1.0, -1.0], "decisions": 120})  # the main network beat the exploiter
    actor.finished(0, {"outcome": [-1.0, 1.0], "decisions": 90})  # ... lost to it
    actor.finished(0, {"outcome": [0.0, 0.0], "decisions": 300})  # ... and drew
    assert actor.results["win_vs_exploiter"] == [1.0, 0.0, 0.5]
    assert "win_exploiter" not in actor.results and not actor.exploiter_results
    actor.finished(1, {"outcome": [-1.0, 1.0], "decisions": 80})  # the exploiter beat the frozen target
    assert actor.results["win_exploiter"] == [1.0] and list(actor.exploiter_results) == [1.0]
    assert actor.results["win_vs_exploiter"] == [1.0, 0.0, 0.5]
    assert actor.lengths == [120, 90, 300, 80] and actor.out_q.messages == []


def test_promotion_comes_only_from_the_exploiter_beating_the_frozen_target():
    actor = _finished_actor(Config(exploiter_games=4, exploiter_win=.6))
    for _ in range(4):
        actor.finished(0, {"outcome": [-1.0, 1.0], "decisions": 50})  # the exploiter beat the main network
    assert actor.results["win_vs_exploiter"] == [0.0] * 4
    assert not actor.exploiter_results and not actor.promotion_sent and actor.out_q.messages == []
    for _ in range(4):
        actor.finished(2, {"outcome": [1.0, -1.0], "decisions": 50})  # an older pinned exploiter's wins
    assert not actor.exploiter_results and actor.out_q.messages == []
    for outcome in ([-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]):
        actor.finished(1, {"outcome": outcome, "decisions": 50})  # the exploiter against the target
    assert list(actor.exploiter_results) == [1.0, 1.0, 0.0, 1.0]
    assert actor.out_q.messages == [("promote", {"win_rate": .75, "generation": 3})] and actor.promotion_sent
    actor.finished(1, {"outcome": [-1.0, 1.0], "decisions": 50})
    assert len(actor.out_q.messages) == 1  # one promotion per generation


def test_exploiter_readiness_counts_its_own_generation_against_what_a_batch_needs():
    import numpy as np

    class FakeRing:
        def __init__(self):
            self.gen = np.arange(80)  # write index per slot
            self.generation = np.where(self.gen >= 30, 3, 2)  # exploiter generation per slot
            self.has_policy = self.gen % 2 == 0

        def written(self):
            return 80

        def window(self):
            return np.arange(10, 70)  # the slots whose targets are resolved

    learner = Learner.__new__(Learner)
    learner.cfg = Config(warmup=16, batch=8, exploiter_patience=30)
    learner.rings = [None, FakeRing()]
    learner.boundary = [0, 20]  # slots written before 20 belong to the previous exploiter
    learner.generation, learner.xupdates, learner.session_xupdates = 3, 5, 2
    learner.exploiter_phase = "caught_up"
    learner.trained, learner.session_trained = [900, 400], [100, 64]
    assert learner.exploiter_readiness() == {
        "generation": 3, "phase": "caught_up", "updates": 5, "patience": 30, "session_updates": 2,
        "fresh_samples": 60, "warmup_target": 16,
        "ready_samples": 40, "ready_target": 8, "searched_samples": 20, "searched_target": 4,
        "trained": 400, "session_trained": 64}


@pytest.mark.parametrize("share", [-.1, 1.5, float("nan")])
def test_an_impossible_exploiter_split_fails_before_allocation(tmp_path, monkeypatch, share):
    import arma_ai.sim as sim
    import arma_ai.teach as teach
    from arma_ai.teach import parse_config

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("an invalid exploiter split reached simulator/model allocation")

    monkeypatch.setattr(sim, "build", unexpected_allocation)
    monkeypatch.setattr(teach, "make_net", unexpected_allocation)
    run = tmp_path / "invalid"
    with pytest.raises(ValueError, match="exploiter_train_share must"):
        Learner(Config(run=str(run), exploiter_train_share=share))
    assert not run.exists()
    with pytest.raises(SystemExit):
        parse_config(["--exploiter-train-share", str(share)])
