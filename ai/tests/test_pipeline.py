"""End-to-end check of the Python side (actor + learner processes) against fake engines."""

import sys
from pathlib import Path

import numpy as np
import torch

from arma_ai import protocol as P
from arma_ai.engine import SlotStep, Step
from arma_ai.model import PolicyNet
from arma_ai.train import Actor, Config, Learner, Store, gae

FAKE = Path(__file__).with_name("fake_engine.py")


def fake_command(i, root, sock_path, arena):
    return [sys.executable, str(FAKE), sock_path, str(i), str(arena.slots)]


def test_two_updates(tmp_path):
    cfg = Config(run=str(tmp_path / "run"), engines=3, rollout=2048, minibatch=512, epochs=2,
                 updates=2, save_every=1, snapshot_every=1, device="cpu", resume=False)
    learner = Learner(cfg)
    learner.run(engine_command=fake_command)
    assert learner.update == 2
    assert (tmp_path / "run" / "latest.pt").exists()
    assert (tmp_path / "run" / "actor.pt").exists()
    assert len(list((tmp_path / "run" / "pool").glob("*.pt"))) >= 2


def test_gae_links_streams():
    s = Store(8)
    # stream A: 0 -> 2 -> 4(done); stream B: 1 -> 3 (open, bootstrap only)
    for i in range(5):
        s.add(*([0] * 3), 0)
    s.value[:5] = [0.5, 0.2, 0.4, 0.1, 0.3]
    s.next[0], s.next[2], s.next[1] = 2, 4, 3
    s.done[4] = True
    s.reward[4] = 1.0
    train, adv, ret = gae(s, 5, gamma=1.0, lam=1.0)
    assert list(train) == [0, 1, 2, 4]
    # with gamma = lam = 1, return of every step in A is the terminal reward
    assert abs(ret[0] - 1.0) < 1e-6 and abs(ret[2] - 1.0) < 1e-6 and abs(ret[3] - 1.0) < 1e-6
    # B bootstraps from the open transition's value
    assert abs(ret[1] - 0.1) < 1e-6


class _RecordingPool:
    def __init__(self):
        self.sent = {}

    def act(self, engine, actions):
        self.sent[engine] = list(actions)


def test_pool_reload_mid_round_keeps_opponents(tmp_path):
    # Reloading the snapshot pool can leave fewer past nets than before while rounds are
    # still running; streams already driven by a past net must keep playing the round
    # with the same opponent (this crashed the actor with an IndexError).
    pool_dir = tmp_path / "run" / "pool"
    pool_dir.mkdir(parents=True)
    for u in range(6):
        torch.save({"model": PolicyNet().state_dict()}, pool_dir / f"u{u:06d}.pt")
    cfg = Config(run=str(tmp_path / "run"), engines=3, rollout=256, device="cpu", past_prob=1.0)
    actor = Actor(cfg, tmp_path / "run")
    engine = 2  # ffa4: four network slots, three of them past selves at past_prob 1

    def step():
        slot = lambda: SlotStep(P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION, 0, (1 << P.N_ACTIONS) - 1,
                                np.zeros((P.GRID, P.GRID), np.uint8),
                                np.zeros((P.GRID, P.GRID), np.uint8),
                                np.zeros(P.N_SCALARS, np.float32))
        return Step(engine, 1, 0, 0.0, False, 4, 4, [slot() for _ in range(4)])

    pool = _RecordingPool()
    actor.act(pool, [step()])
    opponents = {k: actor.controller[(engine, k)] for k in range(1, 4)}

    for f in sorted(pool_dir.glob("*.pt"))[:-1]:
        f.unlink()
    actor.reload_pool()
    assert len(actor.past_nets) == 1

    actor.act(pool, [step()])
    assert len(pool.sent[engine]) == 4
    assert {k: actor.controller[(engine, k)] for k in range(1, 4)} == opponents


def test_kill_after_death_reaches_final_transition(tmp_path):
    # A slot's wall stays in the arena after it dies; a kill it makes then is reported by the
    # engine and must be credited to the slot's last transition instead of being dropped.
    cfg = Config(run=str(tmp_path / "run"), engines=1, rollout=256, device="cpu", past_prob=0.0)
    actor = Actor(cfg, tmp_path / "run")
    pool = _RecordingPool()
    obs = lambda: (np.zeros((P.GRID, P.GRID), np.uint8), np.zeros((P.GRID, P.GRID), np.uint8),
                   np.zeros(P.N_SCALARS, np.float32))
    playing = lambda: SlotStep(P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION, 0, (1 << P.N_ACTIONS) - 1, *obs())

    actor.act(pool, [Step(0, 1, 0, 0.0, False, 2, 2, [playing(), playing()])])
    last = actor.open[(0, 0)]
    # slot 0 dies; with one opponent left alive that costs the full -1
    actor.act(pool, [Step(0, 1, 1, 0.1, False, 1, 2, [SlotStep(P.FLAG_DIED, 0, 0), playing()])])
    # slot 1 then crashes into slot 0's wall, which ends the round
    actor.act(pool, [Step(0, 1, 2, 0.2, True, 0, 2, [SlotStep(0, 1, 0), SlotStep(P.FLAG_DIED, 0, 0)])])
    assert abs(actor.store.reward[last] - (-1.0 + cfg.kill_bonus)) < 1e-6
