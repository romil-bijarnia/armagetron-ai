"""End-to-end check of the Python side against fake engines."""

import sys
from pathlib import Path

import numpy as np

from arma_ai import protocol as P
from arma_ai.train import Config, Trainer

FAKE = Path(__file__).with_name("fake_engine.py")


def fake_command(trainer):
    def command(i, root):
        return [sys.executable, str(FAKE), trainer_pool_socket[0], str(i), str(trainer.mix[i][1].slots)]
    return command


trainer_pool_socket = [None]


def test_two_updates(tmp_path, monkeypatch):
    import arma_ai.engine as E

    orig_spawn = E.EnginePool._spawn

    def spawn(self, i):
        trainer_pool_socket[0] = self.sock_path
        orig_spawn(self, i)

    monkeypatch.setattr(E.EnginePool, "_spawn", spawn)
    cfg = Config(run=str(tmp_path / "run"), engines=3, rollout=2048, minibatch=512, epochs=2,
                 updates=2, save_every=1, snapshot_every=1, device="cpu", resume=False)
    t = Trainer(cfg)
    t.engine_command = fake_command(t)
    t.run()
    assert t.update == 2
    assert (tmp_path / "run" / "latest.pt").exists()
    assert len(list((tmp_path / "run" / "pool").glob("*.pt"))) == 2
    # open transitions carried over must still be unresolved
    s = t.store
    for i in t.open.values():
        assert not s.done[i] and s.next[i] == -1
