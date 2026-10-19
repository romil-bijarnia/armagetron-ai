"""The watch observes real actor decisions without changing them or controlling training."""

import base64
import json
import os
import queue
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from arma_ai.brain import Hub
from arma_ai.sim import SelfPlay
from arma_ai.training_watch import TrainingPublisher, match_results
from arma_ai.watch_server import TrainingFeed


@pytest.mark.parametrize("search,temperature,expected", [(0, 1, "policy_sampled"), (0, 0, "greedy"), (1, 1, "search")])
def test_observation_preserves_game_and_rng_and_records_actual_choices(search, temperature, expected):
    settings = dict(games=1, sims=3, search_prob=search, temperature=temperature,
                    threads=1, max_decisions=80)
    observed, plain = SelfPlay(**settings), SelfPlay(**settings)
    for sp in (observed, plain):
        sp.start(0, -3, [0, 0], [False, False], seed=1234)
    for _ in range(60):
        before = observed.watch(0)
        assert observed.watch(0) == before
        a, b = observed.collect(), plain.collect()
        assert a == b
        assert np.array_equal(observed.maps[:a], plain.maps[:b])
        assert np.array_equal(observed.scalars[:a], plain.scalars[:b])
        assert observed.info(0) == plain.info(0)
        if a:
            logits = np.tile(np.array([1, .5, -.5, .25, -.2, -.4], np.float32), (a, 1))
            values = np.zeros(a, np.float32)
            # A search's subsequent network calls must not overwrite its root checkpoint label.
            updates = np.where(observed.roots[:a], 17, 99).astype(np.int64)
            observed.feed(logits, values, policy_updates=updates)
            observed.watch(0)
            plain.feed(logits, values, policy_updates=updates)
        assert observed.watch(0) == plain.watch(0)
    frame = observed.watch(0)
    assert frame["source_counts"][expected] > 0
    for player in frame["players"]:
        assert player["selection"] == expected
        assert player["policy_update"] == 17
        assert player["mask"][player["action"]]
        assert sum(player["policy_probs"]) == pytest.approx(1)
        if expected == "search":
            assert sum(player["search_probs"]) == pytest.approx(1)
        else:
            assert player["search_probs"] is None


def test_actor_payload_names_exact_policy_and_separate_learner_update(tmp_path):
    sp = SelfPlay(1, sims=0, search_prob=0, threads=1)
    sp.start(0, -3, [0, -1], [False, False], seed=15)
    for _ in range(3):
        n = sp.collect()
        if n:
            sp.feed(np.zeros((n, 6), np.float32), np.zeros(n, np.float32),
                    policy_updates=np.full(n, 17, np.int64))
    (tmp_path / "progress.json").write_text(json.dumps({"update": 25, "time": 100}))
    actor = SimpleNamespace(sp=sp, kind={0: (("hunter", None), 0, [0, -1])},
                            round_ids={0: 3}, pool_files={}, exploiter_gen=0,
                            nets={0: SimpleNamespace(version=4)}, model_updates={0: 19})
    pub = TrainingPublisher(tmp_path, 1)
    pub.publish(actor, force=True)
    frame = json.loads((tmp_path / "watch.json").read_text())
    assert frame["source"] == "training_actor" and frame["round_id"] == 3
    assert frame["players"][0]["policy_update"] == 17
    assert frame["loaded_models"]["0"]["update"] == 19 and frame["trainer_update"] == 25
    assert frame["players"][1]["selection"] == "hunter"
    assert frame["players"][1]["policy_probs"] is None
    assert frame["opponent_names"] == ["Current policy", "Space fighter"]
    assert frame["trails"][0]["segments"] and frame["trails"][0]["color"].startswith("#")
    pub.publish(actor, status="stopped", force=True)
    assert json.loads((tmp_path / "watch.json").read_text())["status"] == "stopped"


def test_freeze_step_and_arena_selection_only_control_view(tmp_path):
    feed = TrainingFeed(tmp_path)
    hub = Hub(1, source="training", feed=feed)
    progress = {"pid": os.getpid(), "time": 100, "phase": "learning", "update": 20}
    frame = {"timestamp": 100, "arena": 0, "n_arenas": 2, "actor_pid": os.getpid()}
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    path = tmp_path / "watch.json"
    path.write_text(json.dumps(frame))
    feed.poll(hub, now=100)
    first = hub.sticky["training_frame"]
    hub.control({"paused": True, "speed": 16})
    frame["timestamp"] = 101
    path.write_text(json.dumps(frame))
    feed.poll(hub, now=101)
    assert hub.sticky["training_frame"] == first
    assert not (tmp_path / "watch_request.json").exists()
    assert json.loads((tmp_path / "progress.json").read_text()) == progress
    hub.control({"step": 1})
    feed.poll(hub, now=101)
    assert hub.sticky["training_frame"] != first and hub.steps == 0 and hub.paused
    hub.control({"arena": 1})
    assert json.loads((tmp_path / "watch_request.json").read_text()) == {"arena": 1}
    assert "training_frame" not in hub.sticky
    assert not feed.poll(hub, now=102)["available"]
    with pytest.raises(ValueError, match="available"):
        hub.control({"arena": 2})
    with pytest.raises(ValueError, match="separate"):
        hub.request_probe({"decision": 1})


def test_stale_or_stopped_actor_is_never_reported_live(tmp_path):
    feed = TrainingFeed(tmp_path)
    hub = Hub(1, source="training", feed=feed)
    (tmp_path / "progress.json").write_text(json.dumps({"pid": os.getpid(), "time": 100, "phase": "learning"}))
    (tmp_path / "watch.json").write_text(json.dumps({"timestamp": 100, "arena": 0, "actor_pid": os.getpid(), "status": "stopped"}))
    assert not feed.poll(hub, now=101)["available"]
    (tmp_path / "watch.json").write_text(json.dumps({"timestamp": 100, "arena": 0, "actor_pid": os.getpid()}))
    assert feed.poll(hub, now=101)["available"]
    assert feed.poll(hub, now=116)["stale"]
    assert not feed.poll(hub, now=131)["available"]


def test_outcome_counts_preserve_draws_and_actual_denominator():
    counts = match_results({"win_vs_hunter": deque([1, 0, .5, 0]), "len_duel": [123], "win_exploiter": []})
    assert counts == {"win_vs_hunter": {"wins": 1, "draws": 1, "losses": 2, "games": 4, "score": .375}}


def test_slow_viewer_drops_training_samples_without_losing_queued_controls():
    hub = Hub(1, source="training")
    client = queue.Queue(maxsize=1)
    hub.clients.add(client)
    hub.control({"paused": True})
    expected = hub.state_msg
    hub.publish("training_frame", "{}")
    assert client.get_nowait() == expected
    assert client.empty()


@pytest.mark.parametrize("search", [0, 1])
def test_live_observations_are_exact_executed_root_inputs(search):
    sp = SelfPlay(1, sims=3, search_prob=search, threads=1, max_decisions=100)
    sp.start(0, -3, [0, 0], [False, False], seed=27)
    roots = {}
    checked = 0
    for _ in range(45):
        n = sp.collect()
        for i in range(n):
            if sp.roots[i]:
                current = sp.maps[i, :4].tobytes()
                previous = sp.maps[i, 4:].tobytes()
                roots.setdefault((current, previous), []).append(sp.scalars[i].copy())
        if n:
            sp.feed(np.zeros((n, 6), np.float32), np.zeros(n, np.float32),
                    policy_updates=np.full(n, 42, np.int64))
        for player in sp.watch(0)["players"]:
            obs = player["observation"]
            if obs is None:
                continue
            assert obs["decision"] == player["decision"]
            key = base64.b64decode(obs["current"]), base64.b64decode(obs["previous"])
            assert len(key[0]) == len(key[1]) == 4 * 64 * 64
            assert key in roots, "viewer reconstructed an input or captured a search branch"
            # Raster maps can be identical for different sub-cell positions/players.
            assert any(np.array_equal(np.asarray(obs["scalars"], np.float32), candidate)
                       for candidate in roots[key]), "scalar inputs differ from every matching root request"
            assert player["policy_update"] == 42
            checked += 1
    assert checked > 10
    sp.start(0, -3, [0, -1], [False, False], seed=28)
    assert all(p["observation"] is None for p in sp.watch(0)["players"])


def test_slow_viewer_gets_latest_frame_after_queued_state_change():
    hub = Hub(1, source="training")
    client = queue.Queue(maxsize=6)
    hub.clients.add(client)
    hub.publish("training_frame", '{"sample":1}')
    hub.control({"arena": 1})
    state = hub.state_msg
    hub.publish("training_frame", '{"sample":2}')
    hub.publish("training_frame", '{"sample":3}')
    assert client.qsize() == 2
    assert client.get_nowait() == state
    assert b'"sample":3' in client.get_nowait()


def test_smaller_new_run_reselects_a_valid_arena_and_drops_old_frames(tmp_path):
    feed = TrainingFeed(tmp_path)
    hub = Hub(1, source="training", feed=feed)
    hub.n_arenas = 256
    hub.control({"arena": 200})
    hub.publish("training_frame", '{"arena":200}')
    (tmp_path / "progress.json").write_text(json.dumps({
        "pid": os.getpid(), "time": 100, "phase": "learning", "config": {"games": 96}}))
    (tmp_path / "watch.json").write_text(json.dumps({
        "timestamp": 100, "arena": 0, "n_arenas": 96, "actor_pid": os.getpid()}))
    status = feed.poll(hub, now=100)
    assert hub.arena == 0 and hub.n_arenas == 96 and status["available"]
    assert json.loads((tmp_path / "watch_request.json").read_text()) == {"arena": 0}
    stopped = {"pid": os.getpid(), "time": 101, "phase": "failed"}
    (tmp_path / "progress.json").write_text(json.dumps(stopped))
    assert not feed.poll(hub, now=101)["available"]
    assert "training_frame" not in hub.sticky


def test_actor_from_previous_trainer_session_is_not_live(tmp_path):
    hub = Hub(1, source="training")
    (tmp_path / "progress.json").write_text(json.dumps({"pid": os.getpid(), "time": 100, "phase": "learning"}))
    (tmp_path / "watch.json").write_text(json.dumps({
        "timestamp": 100, "arena": 0, "actor_pid": os.getpid(), "trainer_pid": os.getpid() + 1}))
    assert not TrainingFeed(tmp_path).poll(hub, now=100)["available"]
