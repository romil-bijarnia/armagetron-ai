"""Artifact consistency at interruption boundaries and honest turn metrics."""

import numpy as np
import pytest
import torch
from torch import nn

from arma_ai import imitate
from arma_ai import protocol as P


class ConstantPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("net_version", torch.tensor(4))
        self.register_buffer("observation_version", torch.tensor(P.OBSERVATION_VERSION))
        self.register_buffer("memory_version", torch.tensor(1))
        self.scores = nn.Parameter(torch.zeros(6))

    def forward(self, maps, scalars, mask):
        logits = self.scores.expand(len(maps), -1).masked_fill(~mask, -1e8)
        return logits, torch.zeros(len(maps), device=maps.device)

    def initial_state(self, batch, device=None):
        return torch.zeros(batch, 1, device=device)

    def heads_sequence(self, maps, scalars, mask, state, reset=None, context=None):
        batch, steps = maps.shape[:2]
        logits = self.scores.expand(batch, steps, -1).masked_fill(~mask, -1e8)
        return (logits, maps.new_zeros(batch, steps, dtype=torch.float32),
                maps.new_zeros(batch, steps, 3, dtype=torch.float32),
                maps.new_zeros(batch, steps, 3, 1, 1, dtype=torch.float32), state)

    def forward_sequence(self, maps, scalars, mask, state, reset=None, context=None):
        logits, value, _, _, state = self.heads_sequence(maps, scalars, mask, state, reset, context)
        return logits, value, state


class SmallDemonstrations(imitate.Demonstrations):
    """A fixed held-out classification task whose first update improves validation."""

    rows = np.array([[0, tick, 0, max(0, tick - 1)] for tick in range(6)]
                    + [[1, tick, 0, max(0, tick - 1)] for tick in range(3)], np.int32)
    training = np.arange(6)
    validation = np.arange(6, 9)
    labels = np.ones(9, np.int64)
    weights = np.ones(6, np.float32)
    summary = {"training_samples": 6, "validation_samples": 3}

    def __init__(self):
        self.episodes = [{"times": np.arange(n) * .05, "actions": np.ones((n, 1), np.int8)}
                         for n in (6, 3)]
        self._index_sequences()

    def batch(self, indices, device):
        n = len(indices)
        return (torch.zeros(n, 8, 1, 1, device=device), torch.zeros(n, P.N_SCALARS, device=device),
                torch.ones(n, 6, dtype=torch.bool, device=device),
                torch.from_numpy(self.labels[indices]).to(device))


def setup_training(tmp_path, monkeypatch):
    monkeypatch.setattr(imitate, "make_net", lambda _: ConstantPolicy())
    path = tmp_path / "latest.pt"
    torch.save({"trainer": "teach", "model": ConstantPolicy().state_dict(), "update": 7514}, path)
    cfg = imitate.Config(epochs=1, steps_per_epoch=1, batch=3, learning_rate=.1, device="cpu")
    return SmallDemonstrations(), tmp_path, cfg


def test_interrupted_validation_keeps_the_actual_latest_learned_weights(tmp_path, monkeypatch):
    data, run, cfg = setup_training(tmp_path, monkeypatch)

    def interrupt(phase, _, __, info):
        if phase == "validating" and info["stage"] == "Checking learned moves":
            raise KeyboardInterrupt

    with imitate.ImitationTrainer(run, cfg, interrupt) as trainer:
        trainer.learn(data)
    latest = torch.load(run / "latest.pt", weights_only=True)
    assert latest["update"] == 7515
    assert latest["imitation_state"]["cumulative_steps"] == 1
    assert latest["imitation_state"]["validation_update"] is None
    assert not (run / "best.pt").exists()


def test_failed_atomic_save_keeps_last_good_weights_and_can_retry(tmp_path, monkeypatch):
    data, run, cfg = setup_training(tmp_path, monkeypatch)
    original = (run / "latest.pt").read_bytes()
    save = imitate.atomic_checkpoint
    calls = 0

    def fail_once(path, value):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated write failure")
        save(path, value)

    with imitate.ImitationTrainer(run, cfg) as trainer:
        monkeypatch.setattr(imitate, "atomic_checkpoint", fail_once)
        with pytest.raises(OSError, match="simulated write failure"):
            trainer.learn(data)
        assert (run / "latest.pt").read_bytes() == original
        assert trainer.update == 7515 and trainer.last_saved_update == 7514
    assert torch.load(run / "latest.pt", weights_only=True)["update"] == 7515


def test_no_turn_examples_cannot_produce_a_turn_accuracy_claim():
    logp = torch.log_softmax(torch.tensor([[10., 0., 0., 0., 0., 0.]] * 5), 1).numpy()
    result = imitate.metrics(logp, np.zeros(5, np.int64), np.ones(6))
    assert result["accuracy"] == result["straight_only_accuracy"] == 1
    assert result["turn_accuracy"] is None
    assert result["action_support"] == [5, 0, 0, 0, 0, 0]
