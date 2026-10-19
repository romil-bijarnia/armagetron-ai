"""Native built-in AI demonstrations have legal labels and pre-action observations."""

import json

import numpy as np
import pytest

from arma_ai import protocol as P
from arma_ai import record as recording
from arma_ai.engine import ArenaConfig, ENGINE_BIN, EnginePool, _parse_step


def _step(label=0, mask=63, flags=P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION, t=0.0, value=0, over=False):
    payload = P.STEP_HEAD.pack(1, int(t * 20) + 1, t, over, int(bool(flags & P.FLAG_ALIVE)), 1, 1)
    payload += P.SLOT_HEAD.pack(flags, 0, mask, label)
    if flags & P.FLAG_NEEDS_ACTION:
        payload += np.full((P.N_MAPS, P.GRID, P.GRID), value, np.uint8).tobytes()
        payload += np.full(P.N_SCALARS, value, np.float32).tobytes()
    return payload


@pytest.mark.parametrize("action", range(P.N_ACTIONS))
def test_teacher_labels_decode_all_six_controls(action):
    slot = _parse_step(0, _step(label=action + 1, mask=1 << action)).slots[0]
    assert slot.teacher_action == action
    assert slot.maps.shape == (P.N_MAPS, P.GRID, P.GRID)


def test_normal_step_has_no_teacher_label():
    assert _parse_step(0, _step()).slots[0].teacher_action is None


@pytest.mark.parametrize("label,mask,flags", [
    (7, 63, 3), (255, 63, 3), (2, 1, 3), (1, 63, P.FLAG_ALIVE), (1, 63, P.FLAG_NEEDS_ACTION),
])
def test_rejects_malformed_or_illegal_teacher_label(label, mask, flags):
    with pytest.raises(ValueError, match="Malformed teacher action"):
        _parse_step(0, _step(label, mask, flags))


def _world(round_id, t, over=False, step=True, alive=True):
    return (P.WORLD_HEAD.pack(round_id, t, over, step, 0, 0, 100, 100, 1)
            + P.WORLD_CYCLE.pack(1, 0, alive, 5, 5, 0, 1, 20, b"teacher"))


def _fake_pool(monkeypatch, messages):
    class Connection:
        def setblocking(self, enabled):
            pass

        def sendall(self, data):
            pass

    class Pool:
        def __init__(self, *args, **kwargs):
            self.conns = [Connection()]
            self.messages = iter(messages)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def _read_msg(self, conn):
            return next(self.messages)

    monkeypatch.setattr(recording, "EnginePool", Pool)
    monkeypatch.setattr(recording, "_fingerprint", lambda path: "1" * 64)


def test_truncated_teacher_recording_never_overwrites_last_saved_tick(tmp_path, monkeypatch):
    messages = []
    for i in range(3):
        t = i * 0.05
        messages.extend([(P.MSG_WORLD, _world(1, t)),
                         (P.MSG_STEP, _step(label=i + 2, t=t, value=i + 1))])
    messages.extend([(P.MSG_WORLD, _world(1, 0.15, over=True, step=False)),
                     (P.MSG_WORLD, _world(2, 0))])
    _fake_pool(monkeypatch, messages)
    progress = []
    rounds = recording.record(rounds=1, slots=1, builtin_ais=1, max_ticks=2, teacher_iq=100,
                              workdir=tmp_path, progress=progress.append)
    arrays = rounds[0].arrays()
    assert arrays["actions"].tolist() == [[P.ACTION_LEFT], [P.ACTION_RIGHT]]
    assert np.all(arrays["maps"][-1] == 2)
    assert np.all(arrays["scalars"][-1] == 2)
    assert rounds[0].metadata["round"] == {
        "round_id": 1, "complete": True, "truncated": True, "ticks_seen": 4, "teacher_labels": 2,
        "teacher_life_complete": False,
    }
    path = tmp_path / "teacher.npz"
    recording.save(rounds, path)
    metadata = recording.load_metadata(path)
    assert metadata["teacher_type"] == "engine_builtin"
    assert metadata["iq"] == 100
    assert metadata["decision_interval"] == 0.05
    assert metadata["observation_timing"] == "before_all_teacher_actions"
    assert metadata["label_source"] == "native_executed_control"
    assert metadata["single_teacher_pre_action"]
    assert metadata["teacher_slots"] == metadata["builtin_opponents"] == 1
    assert metadata["rounds"][0]["truncated"]
    assert len(recording.load(path)) == 1
    with np.load(path, allow_pickle=False) as z:
        assert z["metadata"].dtype.kind == "U"
        assert json.loads(z["metadata"].item()) == metadata
    assert progress[-1] == {"rounds": 1, "rounds_total": 1, "decisions": 2, "labels": 2, "ticks": 2}


@pytest.mark.parametrize("flags,over,life_complete", [
    (P.FLAG_DIED, False, True),
    (P.FLAG_ALIVE | P.FLAG_WON, False, True),
    (P.FLAG_ALIVE, True, True),
    (P.FLAG_ALIVE, False, False),
    (P.FLAG_ALIVE | P.FLAG_DIED, False, False),
])
def test_teacher_life_completion_is_distinct_from_full_match_completion(
        tmp_path, monkeypatch, flags, over, life_complete):
    messages = [
        (P.MSG_WORLD, _world(1, 0)), (P.MSG_STEP, _step(label=1)),
        (P.MSG_WORLD, _world(1, 0.05, over=over, alive=bool(flags & P.FLAG_ALIVE))),
        (P.MSG_STEP, _step(flags=flags, t=0.05, over=over)),
        (P.MSG_WORLD, _world(2, 0)),
    ]
    _fake_pool(monkeypatch, messages)
    rounds = recording.record(rounds=1, slots=1, builtin_ais=3, teacher_iq=100, workdir=tmp_path)
    details = rounds[0].metadata["round"]
    assert details["teacher_life_complete"] is life_complete
    assert details["complete"] is over
    assert not details["truncated"]
    assert rounds[0].arrays()["flags"][-1, 0] == flags
    path = tmp_path / "teacher.npz"
    recording.save(rounds, path)
    assert recording.load_metadata(path)["rounds"][0] == details


@pytest.mark.parametrize("teacher,label", [(100, 0), (None, 1)])
def test_recording_rejects_wrong_driver_provenance(tmp_path, monkeypatch, teacher, label):
    _fake_pool(monkeypatch, [(P.MSG_WORLD, _world(1, 0)), (P.MSG_STEP, _step(label=label))])
    with pytest.raises(ValueError, match="native label|Unexpected native teacher"):
        recording.record(rounds=1, slots=1, builtin_ais=1, teacher_iq=teacher, workdir=tmp_path)


def test_teacher_and_network_driver_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        recording.record(teacher_iq=100, driver=tmp_path / "policy.pt")


@pytest.mark.parametrize("iq", [-1, 101, 100.0, True])
def test_teacher_iq_validation(iq):
    with pytest.raises(ValueError, match="engine maximum"):
        recording.record(teacher_iq=iq)


@pytest.mark.parametrize("slots,builtin", [(2, 1), (1, 0)])
def test_teacher_recording_requires_one_teacher_with_opponents(slots, builtin):
    with pytest.raises(ValueError, match="exactly one teacher slot"):
        recording.record(teacher_iq=100, slots=slots, builtin_ais=builtin)


@pytest.mark.skipif(not ENGINE_BIN.exists(), reason="the dedicated engine is not built")
def test_native_teacher_labels_match_executed_control_and_pre_action_inputs(tmp_path, monkeypatch):
    class OppositeAcknowledgement:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def sendall(self, data):
            magic, kind, size = P.HEADER.unpack_from(data)
            if kind == P.MSG_ACTIONS and size:
                # A deliberately wrong brake-and-turn ACK would corrupt both heading and speed
                # if the engine applied Python's reply after the teacher had already acted.
                data = data[:P.HEADER.size + 1] + bytes([P.ACTION_RIGHT_BRAKE]) * data[P.HEADER.size]
            self.conn.sendall(data)

    class Pool(EnginePool):
        def __enter__(self):
            result = super().__enter__()
            self.conns[0] = OppositeAcknowledgement(self.conns[0])
            return result

    monkeypatch.setattr(recording, "EnginePool", Pool)
    rounds = recording.record(rounds=2, slots=1, builtin_ais=1, size=-4, max_ticks=300, teacher_iq=100,
                              round_time_limit=12, workdir=tmp_path / "teacher", base_port=47972)
    actions_seen = set()
    compared_turns = 0
    for recorded in rounds:
        a = recorded.arrays()
        assert recorded.metadata["round"]["complete"]
        assert not recorded.metadata["round"]["truncated"]
        np.testing.assert_allclose(np.diff(a["times"]), 0.05, atol=2e-5)
        own = int(np.flatnonzero(a["slot_of"] == 0)[0])
        enemy = int(np.flatnonzero(a["slot_of"] == -1)[0])
        low, high = a["bounds"][:2], a["bounds"][2:]
        center = (low + high) / 2
        extent = float(np.max(high - low)) / 2
        for tick in range(len(a["times"])):
            action = int(a["actions"][tick, 0])
            if action < 0:
                continue
            actions_seen.add(action)
            assert int(a["masks"][tick, 0]) & (1 << action)
            h = a["dirs"][tick, own]
            right = np.array([h[1], -h[0]])
            scalars = a["scalars"][tick, 0]
            relative = a["pos"][tick, own] - center
            # Check the ego frame against WORLD sent before the teacher was called. A turn's
            # post-action observation has a different frame and fails these assertions.
            np.testing.assert_allclose(scalars[[10, 11]],
                                       [relative @ right / extent, relative @ h / extent], atol=1e-5)
            cell = 2 * extent / P.GRID
            col = int(np.clip(np.floor((relative @ right + extent) / cell), 0, P.GRID - 1))
            row = int(np.clip(np.floor((extent - relative @ h) / cell), 0, P.GRID - 1))
            assert a["maps"][tick, 0, 2, row, col] & (1 << 3)  # own head in the pre-action global frame
            if a["alive"][tick, enemy]:
                delta = a["pos"][tick, enemy] - a["pos"][tick, own]
                direction = a["dirs"][tick, enemy]
                np.testing.assert_allclose(scalars[[64, 65, 67, 68]],
                                           [np.clip(delta @ h / 100, -3, 3),
                                            np.clip(delta @ right / 100, -3, 3),
                                            direction @ h, direction @ right], atol=1e-5)
            if tick + 1 < len(a["times"]) and a["alive"][tick + 1, own]:
                expected = h
                if action in (P.ACTION_LEFT, P.ACTION_LEFT_BRAKE):
                    expected = -right
                    compared_turns += 1
                elif action in (P.ACTION_RIGHT, P.ACTION_RIGHT_BRAKE):
                    expected = right
                    compared_turns += 1
                # No teacher control occurs between policy boundaries; the current label must
                # be the exact next heading despite our deliberately incorrect Python ACK.
                np.testing.assert_allclose(a["dirs"][tick + 1, own], expected, atol=1e-6)
                if a["actions"][tick + 1, 0] >= 0:
                    assert bool(a["scalars"][tick + 1, 0, 3]) == (action >= P.ACTION_BRAKE)
    assert compared_turns > 0
    assert P.ACTION_STRAIGHT in actions_seen
