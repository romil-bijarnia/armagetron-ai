"""Network recordings must carry the same recurrent policy used in live play."""

import numpy as np
import pytest
import torch

from arma_ai import protocol as P, record as recording
from arma_ai.engine import SlotStep, Step
from arma_ai.model import make_net, recurrent_context
from test_demonstrations import _fake_pool


@pytest.fixture
def checkpoint(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(741)
        net = make_net().eval()
        with torch.no_grad():
            # A learned memory path exposes accidental fresh-state diagnostics.
            net.memory_projection.weight.normal_(0, .04)
            net.memory_projection.bias.normal_(0, .01)
    path = tmp_path / "recurrent.pt"
    torch.save({"model": net.state_dict()}, path)
    return path


def step(tick, *, t=None, round_id=1, flags=P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION, value=1, mask=63):
    maps = np.full((P.N_MAPS, P.GRID, P.GRID), value, np.uint8)
    scalars = np.full(P.N_SCALARS, value / 20, np.float32)
    if not flags & P.FLAG_NEEDS_ACTION:
        maps = scalars = None
    slot = SlotStep(flags, 0, mask, maps, scalars)
    return Step(0, round_id, tick, tick * .05 if t is None else t, False,
                int(bool(flags & P.FLAG_ALIVE)), 1, [slot])


def drive(driver, observation, player_id=11):
    slot = observation.slots[0]
    return driver(slot.mask, slot.maps, slot.scalars, step=observation, player_id=player_id)


def capture(driver, monkeypatch):
    calls = []
    original = driver.net.act_step

    def record(maps, feats, state, **kwargs):
        calls.append({"maps": maps.clone(), "state": state.clone(),
                      "context": kwargs["context"].clone(), "greedy": kwargs["greedy"]})
        return original(maps, feats, state, **kwargs)

    monkeypatch.setattr(driver.net, "act_step", record)

    def diagnostic(*args, **kwargs):
        pytest.fail("A production recording used the fresh-state diagnostic wrapper")

    monkeypatch.setattr(driver.net, "act", diagnostic)
    return calls


def test_recording_driver_matches_nonzero_memory_sequence_and_actual_context(checkpoint, monkeypatch):
    driver = recording.NetDriver(checkpoint, seed=3, noise=0)
    calls = capture(driver, monkeypatch)
    state = driver.net.initial_state(1)
    previous_maps, previous_action = None, -1
    recurrent_logits = None
    for tick in range(3):
        observation = step(tick, value=tick + 1, mask=1 << P.ACTION_RIGHT_BRAKE)
        current = observation.slots[0]
        maps = np.concatenate([current.maps, current.maps if previous_maps is None else previous_maps])
        context = recurrent_context(torch.tensor([previous_action]), torch.tensor([0. if tick == 0 else .05]))
        mask = torch.arange(P.N_ACTIONS)[None] == P.ACTION_RIGHT_BRAKE
        with torch.no_grad():
            recurrent_logits, _, state = driver.net.forward_step(torch.from_numpy(maps)[None],
                torch.from_numpy(current.scalars)[None], mask, state, context=context)
        chosen = drive(driver, observation)
        assert chosen == P.ACTION_RIGHT_BRAKE
        assert torch.allclose(driver.history.memory[(0, 0)], state, atol=1e-6)
        assert torch.equal(calls[-1]["context"], context)
        assert np.array_equal(calls[-1]["maps"][0].numpy(), maps)
        previous_maps = current.maps.copy()
        previous_action = -1 if tick == 0 else chosen
    assert len(calls) == 3 and calls[-1]["state"].abs().sum() > 0
    # With all controls enabled, persistent history changes the policy's actual logits.
    with torch.no_grad():
        x, scalars = calls[-1]["maps"], torch.from_numpy(current.scalars)[None]
        legal = torch.ones(1, P.N_ACTIONS, dtype=torch.bool)
        carried, _, _ = driver.net.forward_step(x, scalars, legal, calls[-1]["state"],
                                               context=calls[-1]["context"])
        fresh, _, _ = driver.net.forward_step(x, scalars, legal, driver.net.initial_state(1),
                                             context=calls[-1]["context"])
    assert not torch.allclose(carried, fresh, atol=1e-6)


def test_random_overrides_consume_memory_and_become_the_next_control_context(checkpoint, monkeypatch):
    driver = recording.NetDriver(checkpoint, seed=7, noise=1)
    calls = capture(driver, monkeypatch)

    class Noise:
        draws = 0

        def random(self):
            self.draws += 1
            return 0.

        def choice(self, allowed):
            return allowed[-1]

    driver.rng = Noise()
    with torch.no_grad():
        driver.net.pi.weight.zero_()
        driver.net.pi.bias.zero_()
        driver.net.pi.bias[P.ACTION_STRAIGHT] = 20
    first = step(1)
    assert drive(driver, first) == P.ACTION_RIGHT_BRAKE  # overrides a straight policy
    before_duplicate = driver.history.memory[(0, 0)].clone()
    assert before_duplicate.abs().sum() > 0
    assert drive(driver, first) == P.ACTION_RIGHT_BRAKE
    assert len(calls) == driver.rng.draws == 1
    assert torch.equal(driver.history.memory[(0, 0)], before_duplicate)
    assert drive(driver, step(2, value=2)) == P.ACTION_RIGHT_BRAKE
    assert len(calls) == driver.rng.draws == 2
    assert calls[-1]["greedy"]  # no unused policy sampling draw during an override
    assert torch.equal(calls[-1]["state"], before_duplicate)
    assert calls[-1]["context"].tolist() == [[0, 0, 0, 0, 0, 1, 1]]
    assert not torch.equal(driver.history.memory[(0, 0)], before_duplicate)


def test_separate_recording_players_cannot_inherit_each_others_memory(checkpoint, monkeypatch):
    first = recording.NetDriver(checkpoint, seed=1, noise=0)
    second = recording.NetDriver(checkpoint, seed=2, noise=0)
    second_calls = capture(second, monkeypatch)
    drive(first, step(1, value=10), player_id=1)
    assert not second.history.memory
    drive(second, step(1, value=20), player_id=2)
    assert not second_calls[0]["state"].any() and not second_calls[0]["context"].any()
    second_memory = second.history.memory[(0, 0)].clone()
    drive(first, step(2, value=30), player_id=1)
    first.reset()
    assert not first.history.memory and not first.history.previous_actions
    assert torch.equal(second.history.memory[(0, 0)], second_memory)


def world_message(observation, player_id):
    slot = observation.slots[0]
    return (P.WORLD_HEAD.pack(observation.round_id, observation.round_time, False, True, 0, 0, 100, 100, 1)
            + P.WORLD_CYCLE.pack(player_id, 0, slot.alive, 5, 5, 0, 1, 20, b"Blacklight"))


def step_message(observation):
    slot = observation.slots[0]
    payload = P.STEP_HEAD.pack(observation.round_id, observation.tick, observation.round_time,
                               observation.round_over, observation.n_alive, observation.n_total, 1)
    payload += P.SLOT_HEAD.pack(slot.flags, slot.kills, slot.mask, 0)
    if slot.needs_action:
        payload += slot.maps.tobytes() + slot.scalars.tobytes()
    return payload


def test_record_loop_passes_real_life_identity_time_and_reset_events(checkpoint, tmp_path, monkeypatch):
    instances = []
    real_driver = recording.NetDriver

    class Driver(real_driver):
        def __init__(self, *args):
            super().__init__(*args)
            self.noise = 0
            self.calls = capture(self, monkeypatch)
            instances.append(self)

    monkeypatch.setattr(recording, "NetDriver", Driver)
    spawn = P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION | P.FLAG_SPAWNED
    observations = [(step(0, flags=spawn), 11), (step(1), 11),
                    (step(2, flags=P.FLAG_DIED, mask=0), 11), (step(3, flags=spawn), 11),
                    (step(4), 22), (step(6), 22),
                    (step(0, round_id=2, flags=spawn), 33), (step(1, round_id=2), 33)]
    messages = []
    for observation, identity in observations:
        messages.extend([(P.MSG_WORLD, world_message(observation, identity)),
                         (P.MSG_STEP, step_message(observation))])
    messages.append((P.MSG_WORLD, world_message(step(0, round_id=3), 44)))
    _fake_pool(monkeypatch, messages)
    rounds = recording.record(rounds=2, slots=1, driver=checkpoint, workdir=tmp_path / "record")
    assert len(rounds) == 2
    driver = instances[0]
    assert len(driver.calls) == 7  # death observations reset without running the network
    assert not driver.calls[0]["state"].any()
    assert driver.calls[1]["state"].abs().sum() > 0
    for index in (2, 3, 4, 5):  # respawn, reused player slot, missing decision and new round
        assert not driver.calls[index]["state"].any()
        assert not driver.calls[index]["context"].any()
    assert driver.calls[6]["state"].abs().sum() > 0
    assert driver.calls[6]["context"][0, 6] == pytest.approx(1., abs=1e-6)
    assert not driver.calls[6]["context"][0, :6].any()  # time-zero control was dropped
    assert driver.history.memory_gap_resets == 1
    assert driver.history.identities[(0, 0)] == 33
    assert rounds[0].arrays()["actions"][2, 0] == -1
