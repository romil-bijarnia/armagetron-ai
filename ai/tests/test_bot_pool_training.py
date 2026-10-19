"""Named native opponents reach recordings, curriculum collection and honest evaluation."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from arma_ai import evaluate, imitate, record
from arma_ai import protocol as P
from arma_ai.bot_profiles import HELD_OUT_PROFILES, make_roster, schedule
from test_benchmarks import FakeBrain, FakePool, state
from test_demonstrations import _fake_pool, _step, _world


def test_named_teacher_recording_keeps_exact_profile_provenance(tmp_path, monkeypatch):
    messages = [(P.MSG_WORLD, _world(1, 0)), (P.MSG_STEP, _step(label=1)),
                (P.MSG_WORLD, _world(1, .05, over=True)),
                (P.MSG_STEP, _step(flags=P.FLAG_ALIVE | P.FLAG_WON, t=.05, over=True)),
                (P.MSG_WORLD, _world(2, 0))]
    _fake_pool(monkeypatch, messages)
    rounds = record.record(rounds=1, slots=1, builtin_ais=1, teacher_iq=100,
                           teacher_profile="fighter_rush", opponent_profiles=("stock_wall",),
                           workdir=tmp_path)
    path = tmp_path / "native-profile.npz"
    record.save(rounds, path)
    metadata = record.load_metadata(path)
    roster = make_roster("fighter_rush", ("stock_wall",))
    assert metadata["bot_pool"] == roster.manifest
    assert metadata["profile_selection"] == "explicit_named_roles"
    assert metadata["teacher_profiles_sha256"] == hashlib.sha256(roster.text.encode("ascii")).hexdigest()
    assert metadata["settings"]["NEURAL_CHARACTER_PROFILE"] == roster.controlled_name
    assert metadata["settings"]["BUILTIN_CHARACTER_PROFILES"] == roster.builtin_names[0]
    assert not metadata["emergency_help"] and metadata["teacher_schedule"] == "policy_decision_boundaries"
    imitate.validate_provenance(metadata)


@pytest.mark.parametrize("kwargs,match", [
    ({"teacher_profile": "fighter_rush"}, "native teacher mode"),
    ({"teacher_iq": 100, "teacher_profile": "missing"}, "Unknown bot profile"),
    ({"teacher_iq": 100, "opponent_profiles": ("stock_wall", "stock_quake")}, "profile count"),
])
def test_invalid_profile_recordings_fail_before_an_engine_starts(tmp_path, monkeypatch, kwargs, match):
    def forbidden(*args, **kw):
        raise AssertionError("Invalid profile request launched an engine")
    monkeypatch.setattr(record, "EnginePool", forbidden)
    with pytest.raises(ValueError, match=match):
        record.record(slots=1, builtin_ais=1, workdir=tmp_path, **kwargs)


def test_diverse_collection_reaches_each_requested_teacher_and_opponent(tmp_path, monkeypatch):
    calls = []
    def recording(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(metadata={"round": {"teacher_labels": 7}})]
    monkeypatch.setattr(imitate, "record", recording)
    monkeypatch.setattr(imitate, "save", lambda rounds, path: path.write_bytes(b"test recording"))
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"existing fighter")
    paths = imitate.collect(tmp_path, 32, -3, 17, bot_pool="diverse")
    planned = schedule(32, 17)
    assert len(paths) == len(calls) == 32 and all(path.is_file() for path in paths)
    for call, match in zip(calls, planned):
        assert call["teacher_profile"] == match.teacher
        assert call["opponent_profiles"] == match.opponents
        assert call["builtin_ais"] == len(match.opponents)
        assert call["teacher_iq"] == 100 and call["slots"] == 1
        assert not set((match.teacher, *match.opponents)) & set(HELD_OUT_PROFILES)
    assert len({call["size"] for call in calls}) == 32
    saved = json.loads((tmp_path / "imitation-collection.json").read_text())
    assert saved["bot_pool"] == "diverse" and saved["decisions"] == 224
    assert saved["native_rng_seeded"] is False
    assert checkpoint.read_bytes() == b"existing fighter"


def test_stock_collection_remains_available_without_profile_overrides(tmp_path, monkeypatch):
    calls = []
    def recording(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(metadata={"round": {"teacher_labels": 1}})]
    monkeypatch.setattr(imitate, "record", recording)
    monkeypatch.setattr(imitate, "save", lambda rounds, path: path.write_bytes(b"test recording"))
    imitate.collect(tmp_path, 3, -3, 17)
    assert [call["builtin_ais"] for call in calls] == [1, 1, 3]
    assert all("teacher_profile" not in call and "opponent_profiles" not in call for call in calls)


def test_named_held_out_opponents_are_evaluated_with_the_requested_roster(tmp_path, monkeypatch):
    arenas = []
    class Pool(FakePool):
        def __init__(self, configs, **kwargs):
            arenas.extend(configs)
            super().__init__(configs, **kwargs)
    monkeypatch.setattr(evaluate, "Brain", FakeBrain)
    monkeypatch.setattr(evaluate, "_EvaluationPool", Pool)
    Pool.batches = [[state(1, P.FLAG_ALIVE | P.FLAG_WON, 1)]]
    opponents = ("holdout_fighter", "holdout_path", "holdout_defender")
    result = evaluate.evaluate(tmp_path / "model.pt", None, 1, 1, -3, builtin_profiles=opponents)
    assert arenas[0].builtin_ais == 3
    roster = make_roster("stock_wall", opponents)
    assert arenas[0].ai_characters == roster.text
    assert arenas[0].builtin_profiles == roster.builtin_names
    assert result["wins"] == result["rounds_completed"] == 1 and result["status"] == "complete"
    assert [entry["name"] for entry in result["bot_pool"]["opponents"]] == list(opponents)


def test_unknown_pool_fails_before_collecting_or_learning(tmp_path):
    with pytest.raises(ValueError, match="bot_pool"):
        imitate.collect(tmp_path, 3, -3, 17, bot_pool="typo")
    with pytest.raises(ValueError, match="bot_pool"):
        imitate.Config(bot_pool="typo").validate()
    assert not list(tmp_path.iterdir())
