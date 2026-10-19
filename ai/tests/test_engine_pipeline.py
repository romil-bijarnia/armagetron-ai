"""Engine recovery and frame-history checks for the current policy playback path."""

import sys
from pathlib import Path

import numpy as np
import torch

from arma_ai import protocol as P
from arma_ai.engine import SlotStep, Step

FAKE = Path(__file__).with_name("fake_engine.py")


def fake_command(i, root, sock_path, arena):
    return [sys.executable, str(FAKE), sock_path, str(i), str(arena.slots)]


def test_engine_that_dies_after_asking_is_relaunched_with_new_round_ids(tmp_path):
    from arma_ai.engine import ArenaConfig, EnginePool

    with EnginePool([ArenaConfig(slots=2)], workdir=tmp_path / "engines", command=fake_command) as pool:
        first = pool.poll()[0]
        pool.procs[0].kill()
        pool.procs[0].wait()
        pool.act(0, [0, 0])  # the engine is gone: relaunch instead of raising
        again = pool.poll()[0]
        assert pool.restarts[0] == 1
        assert again.round_id != first.round_id  # both engines started at round 1
        pool.act(0, [0, 0])


def test_playback_frame_history_is_per_engine_and_resets_each_round():
    from arma_ai.play import Brain, PolicyHistory

    class CapturingNet(torch.nn.Module):
        n_actions = P.N_ACTIONS

        def __init__(self):
            super().__init__()
            self.maps = []
            self.parameter = torch.nn.Parameter(torch.zeros(1))

        def initial_state(self, batch):
            return torch.zeros(batch, 256)

        def act_step(self, maps, feats, state, greedy, context=None):
            self.maps.append(maps.clone())
            return (torch.zeros(len(maps), dtype=torch.long), torch.zeros(len(maps)),
                    torch.zeros(len(maps)), state + 1)

    brain = Brain.__new__(Brain)
    PolicyHistory.__init__(brain, CapturingNet())
    brain.device = torch.device("cpu")
    brain.sample = False
    ticks = {}

    def step(engine, round_id, value, flags=P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION):
        slot = SlotStep(flags, 0, (1 << P.N_ACTIONS) - 1,
                        np.full((P.N_MAPS, P.GRID, P.GRID), value, np.uint8), np.zeros(P.N_SCALARS, np.float32))
        tick = ticks.get((engine, round_id), 0)
        ticks[engine, round_id] = tick + 1
        return Step(engine, round_id, tick, tick * .05, False, 2, 2, [slot])

    brain.actions(step(0, 1, 1))
    brain.actions(step(1, 1, 2))
    brain.actions(step(0, 1, 3))
    brain.actions(step(1, 1, 4))
    assert (brain.net.maps[2][:, P.N_MAPS:] == 1).all()
    assert (brain.net.maps[3][:, P.N_MAPS:] == 2).all()
    # A round change must work even when the first observed decision lacks a spawn flag.
    brain.actions(step(0, 2, 5))
    assert (brain.net.maps[-1][:, P.N_MAPS:] == 5).all()
    brain.actions(step(1, 1, 0, P.FLAG_DIED))
    assert (1, 0) not in brain.prev
    assert (0, 0) in brain.prev
