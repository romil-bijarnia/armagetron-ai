"""Rebuild recurrent inputs from raw, per-life observations and actual applied controls."""

import numpy as np
import pytest

from arma_ai import protocol as P
from arma_ai.sim import Ring, SelfPlay


class EndAt:
    def __init__(self, index):
        self.index = index

    def choice(self, pool):
        assert self.index in pool
        return self.index


def interleaved_lives():
    ring = Ring(4200)
    for index in range(10):
        ring.gen[index] = index
        ring.prev[index] = index - 2 if index >= 2 else -1
        ring.maps[index] = index
        ring.scalars[index, 0] = index
        ring.mask[index] = 63
        ring.context[index, index % P.N_ACTIONS] = index > 1
        ring.context[index, -1] = index > 1
        ring.generation[index] = index % 2
        ring.future[index] = -1
    ring.counter[0] = 10
    # Burn-in observations exist even when they do not have learning targets.
    ring.ready[[4, 6, 8]] = 1
    ring.has_policy[8] = 1
    ring.policy[8, 4] = 1
    return ring


def test_replay_keeps_whole_life_history_and_unready_previous_action_context():
    ring = interleaved_lives()
    data = ring.sequences(3, EndAt(8), burn_in=2, unroll=3, generation=0)
    np.testing.assert_array_equal(data["scalars"][0, :, 0], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(data["maps"][0, :, 0, 0, 0], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(data["maps"][0, :, P.N_MAPS, 0, 0], [0, 0, 2, 4, 6])
    np.testing.assert_array_equal(data["context"][0], ring.context[[0, 2, 4, 6, 8]])
    np.testing.assert_array_equal(data["learn"], [[False, False, True, True, True]])
    np.testing.assert_array_equal(data["reset"], [[True, False, False, False, False]])
    assert ring.ready[2] == 0 and data["context"][0, 1, 2] == 1


@pytest.mark.parametrize("damage", ["overwritten", "generation", "unready_target"])
def test_replay_rejects_broken_history_instead_of_inventing_a_spawn(damage):
    ring = interleaved_lives()
    if damage == "overwritten":
        ring.gen[2] = 4202
    elif damage == "generation":
        ring.generation[2] = 9
    else:
        ring.ready[6] = 0
    assert ring.sequences(3, EndAt(8), burn_in=2, unroll=3) is None


def test_simulator_capi_passes_applied_context_and_state_between_real_decisions():
    sp = SelfPlay(1, sims=0, search_prob=0, temperature=0, threads=1, recurrent=True)
    sp.start(0, -3, [0, 0], [False, False])
    assert sp.collect() == 2
    assert not sp.states.any() and not sp.sequence_contexts.any()
    logits = np.full((2, P.N_ACTIONS), -100, np.float32)
    logits[:, P.ACTION_LEFT_BRAKE] = 100
    for expected_state, expected_action in [(3., None), (5., P.ACTION_LEFT_BRAKE)]:
        sp.feed(logits, np.zeros(2), next_states=np.full((2, 256), expected_state, np.float32))
        assert sp.collect() == 2
        np.testing.assert_array_equal(sp.states[:2], expected_state)
        context = sp.sequence_contexts[:2, 0]
        if expected_action is None:
            assert not context[:, :P.N_ACTIONS].any()  # the time-zero command was dropped
        else:
            assert np.all(context[:, expected_action] == 1)
            assert np.all(context[:, :P.N_ACTIONS].sum(1) == 1)
        np.testing.assert_allclose(context[:, -1], 1, atol=1e-5)
    frame = sp.watch(0)
    assert frame["players"][0]["observation"]["context"] == [0., 0., 0., 0., 0., 0., 1.]
    sp.start(0, -3, [0, 0], [False, False])
    assert sp.collect() == 2
    assert not sp.states[:2].any() and not sp.sequence_contexts[:2, 0].any()
