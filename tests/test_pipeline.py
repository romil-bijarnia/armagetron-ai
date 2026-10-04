"""End-to-end check of the Python side (actor + learner processes) against fake engines."""

import sys
from pathlib import Path

from arma_ai.train import Config, Learner, Store, gae

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
