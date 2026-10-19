"""Startup inference must leave the player's policy, random draws and history untouched."""

import numpy as np
import pytest
import torch
from torch import nn

from arma_ai import play, protocol as P
from arma_ai.engine import SlotStep, Step
from arma_ai.model import _pick


class WarmupNet(nn.Module):
    n_actions = P.N_ACTIONS
    state_size = 256

    def __init__(self):
        super().__init__()
        self.probe = nn.Parameter(torch.tensor([0.0]))
        self.calls = []
        self.states = []
        self.contexts = []

    def initial_state(self, batch):
        return self.probe.new_zeros(batch, self.state_size)

    def act_step(self, maps, feats, state, greedy, context=None):
        self.calls.append((maps.clone(), feats.clone(), greedy, torch.is_grad_enabled(), self.training))
        self.states.append(state.clone())
        self.contexts.append(None if context is None else context.clone())
        logits = self.probe.expand(len(maps), P.N_ACTIONS).masked_fill(feats[:, P.N_SCALARS:] < .5, -1e8)
        return (*_pick(logits, self.probe.expand(len(maps)), greedy), state + 1 + feats[:, :1])


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU is unavailable"))])
def test_startup_prewarms_real_singleton_shapes_without_policy_rng_or_history_changes(monkeypatch, device):
    monkeypatch.setattr(play.torch, "load", lambda *args, **kwargs: {
        "model": {"probe": torch.tensor([3.0])}, "update": 11280})
    monkeypatch.setattr(play, "net_version", lambda state: 4)
    monkeypatch.setattr(play, "make_net", lambda version: WarmupNet())
    cpu_rng = torch.get_rng_state().clone()
    device_rng = torch.mps.get_rng_state().clone() if device == "mps" else None

    brain = play.Brain("unused.pt", device=device, sample=True)
    assert brain.update == 11280 and brain.sample
    assert not brain.prev and not brain.rounds
    assert len(brain.net.calls) == 1
    maps, feats, greedy, gradients, training = brain.net.calls[0]
    assert maps.shape == (1, 2 * P.N_MAPS, P.GRID, P.GRID)
    assert maps.dtype == torch.uint8 and maps.device.type == device
    assert feats.shape == (1, P.N_SCALARS + P.N_ACTIONS)
    assert feats.dtype == torch.float32 and feats.device.type == device
    assert not maps.any() and not feats[:, :P.N_SCALARS].any()
    assert (feats[:, P.N_SCALARS:] == 1).all()
    assert greedy and not gradients and not training

    history = np.ones((P.N_MAPS, P.GRID, P.GRID), np.uint8)
    brain.prev[(2, 0)], brain.rounds[2] = history, 7
    brain._prewarm()
    assert brain.prev[(2, 0)] is history and history.all()
    assert brain.rounds == {2: 7}
    assert brain.net.probe.item() == 3.0 and brain.net.probe.grad is None
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if device_rng is not None:
        assert torch.equal(torch.mps.get_rng_state(), device_rng)


def _brain(monkeypatch):
    monkeypatch.setattr(play.torch, "load", lambda *args, **kwargs: {
        "model": {"probe": torch.tensor([3.0])}, "update": 11280})
    monkeypatch.setattr(play, "net_version", lambda state: 4)
    monkeypatch.setattr(play, "make_net", lambda version: WarmupNet())
    return play.Brain("unused.pt", device="cpu")


def _step(tick, *, engine=0, round_id=7, time=None, flags=None, values=(0., 2.), action=5):
    slots = []
    for i, value in enumerate(values):
        flag = (P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION) if flags is None else flags[i]
        scalars = np.zeros(P.N_SCALARS, np.float32)
        scalars[0] = value
        slots.append(SlotStep(flag, 0, 1 << action,
                              np.full((P.N_MAPS, P.GRID, P.GRID), i + tick, np.uint8), scalars))
    return Step(engine, round_id, tick, tick * .05 if time is None else time,
                False, len(slots), len(slots), slots)


def test_live_memory_keeps_independent_slots_and_reuses_duplicate_controls(monkeypatch):
    brain = _brain(monkeypatch)
    spawn = P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION | P.FLAG_SPAWNED
    first = _step(0, flags=[spawn, spawn])
    assert brain.actions(first) == [5, 5]
    assert torch.equal(brain.memory[(0, 0)], torch.ones(1, 256))
    assert torch.equal(brain.memory[(0, 1)], torch.full((1, 256), 3.))
    draws = torch.get_rng_state().clone()
    brain.sample = True
    assert brain.actions(first) == [5, 5]
    assert len(brain.net.calls) == 2  # one warmup and one real batched decision
    assert torch.equal(torch.get_rng_state(), draws)
    brain.sample = False
    brain.actions(_step(1))
    assert torch.equal(brain.net.states[-1][0], torch.ones(256))
    assert torch.equal(brain.net.states[-1][1], torch.full((256,), 3.))
    # The first requested control was dropped by the game timer; only its elapsed time remains.
    assert torch.equal(brain.net.contexts[-1][:, :6], torch.zeros(2, 6))
    assert torch.equal(brain.net.contexts[-1][:, 6], torch.ones(2))
    brain.actions(_step(2))
    assert torch.equal(brain.net.contexts[-1][:, 5], torch.ones(2))
    assert torch.equal(brain.net.contexts[-1][:, :5], torch.zeros(2, 5))
    assert np.array_equal(brain.net.calls[-1][0][0, P.N_MAPS:].numpy(), first.slots[0].maps + 1)


def test_history_resets_only_the_replaced_player_dead_slot_or_gapped_life(monkeypatch):
    brain = _brain(monkeypatch)
    brain.actions(_step(1), identities={0: 10, 1: 20})
    brain.actions(_step(2), identities={0: 11, 1: 20})
    assert not brain.net.states[-1][0].any()
    assert brain.net.states[-1][1, 0] == 3
    assert not brain.net.contexts[-1][0].any()
    assert brain.net.contexts[-1][1, 5] == 1
    alive = P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION
    brain.actions(_step(3, flags=[P.FLAG_DIED, alive]), identities={0: 11, 1: 20})
    assert (0, 0) not in brain.memory and (0, 0) not in brain.prev
    assert (0, 0) not in brain.previous_actions and (0, 0) not in brain.observed
    brain.actions(_step(5), identities={0: 11, 1: 20})
    assert not brain.net.states[-1].any() and not brain.net.contexts[-1].any()
    assert brain.memory_gap_resets == 1
    for i in range(2):
        assert np.array_equal(brain.net.calls[-1][0][i, :P.N_MAPS].numpy(),
                              brain.net.calls[-1][0][i, P.N_MAPS:].numpy())


def test_round_engine_restart_slot_removal_and_weights_cannot_leak_memory(monkeypatch):
    brain = _brain(monkeypatch)
    brain.actions(_step(5, engine=1))
    other = brain.memory[(1, 0)].clone()
    brain.actions(_step(5))
    brain.actions(_step(6, round_id=8))
    assert not brain.net.states[-1].any()
    assert torch.equal(brain.memory[(1, 0)], other)
    brain.actions(_step(7, round_id=8, values=(0.,)))
    assert (0, 1) not in brain.memory and (0, 1) not in brain.identities
    with pytest.raises(ValueError, match="older game observation"):
        brain.actions(_step(6, round_id=8, values=(0.,)))
    spawn = P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION | P.FLAG_SPAWNED
    brain.actions(_step(0, round_id=8, flags=[spawn], values=(0.,)))
    assert not brain.net.states[-1].any()
    brain.net = WarmupNet().eval()
    brain.actions(_step(1, round_id=8, values=(0.,)))
    assert not brain.net.states[-1].any() and not brain.net.contexts[-1].any()
    assert (1, 0) not in brain.memory


def test_inspection_snapshots_are_private_and_checkpoint_generation_resets_history(monkeypatch):
    brain = _brain(monkeypatch)
    step = _step(1)
    brain.actions(step)
    next_step = _step(2)
    brain.begin(next_step)
    snapshot = brain.snapshot(next_step, 0)
    snapshot.state.fill_(999)
    snapshot.context.fill_(999)
    snapshot.maps.fill(255)
    snapshot.scalars.fill(255)
    assert brain.memory[(0, 0)][0, 0] == 1
    assert brain.prev[(0, 0)][0, 0, 0] == 1
    assert next_step.slots[0].scalars[0] == 0
    brain.begin(next_step, generation=2)
    fresh = brain.snapshot(next_step, 0)
    assert not fresh.state.any() and not fresh.context.any()
    assert np.array_equal(fresh.maps[:P.N_MAPS], fresh.maps[P.N_MAPS:])
