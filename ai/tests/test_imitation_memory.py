"""Sequence imitation must learn history without crossing lives or advancing validation twice."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from arma_ai import imitate
from arma_ai.imitate import Config, Demonstrations, ImitationTrainer, predict, sequence_outputs
from test_imitation import TinyPolicy, recording, shared_checkpoint


def demonstrations(tmp_path):
    return Demonstrations([recording(tmp_path / "expert.npz")])


def test_windows_use_prior_executed_controls_and_keep_validation_separate(tmp_path):
    data = demonstrations(tmp_path)
    segment = next(s for s in data.segments if s[0] in data.training)
    window = data.sequence_batch([segment[2]], 2, 3, torch.device("cpu"))
    assert window["indices"].tolist() == [segment[:5].tolist()]
    assert window["reset"].tolist() == [[True, False, False, False, False]]
    assert window["learn"].tolist() == [[False, False, True, True, True]]
    context = window["context"][0].numpy()
    assert not context[0].any()
    assert context[1].tolist() == [0, 0, 0, 0, 0, 0, 1]
    assert context[2, 1] == 1 and context[2, 6] == 1
    assert context[3, 2] == 1 and context[3, 6] == pytest.approx(1)
    with pytest.raises(ValueError, match="held-out"):
        data.sequence_batch([data.validation[0]], 2, 3, torch.device("cpu"))


def test_short_windows_mask_padding_and_reset_before_first_real_observation(tmp_path):
    data = demonstrations(tmp_path)
    segment = next(s for s in data.segments if s[0] in data.training)
    window = data.sequence_batch([segment[0], segment[-1]], 3, 4, torch.device("cpu"))
    assert window["indices"][0].tolist() == [-1, -1, -1, *segment[:4]]
    assert window["reset"][0].tolist() == [False, False, False, True, False, False, False]
    assert window["learn"].sum(1).tolist() == [4, 1]
    assert window["indices"][1].tolist() == [*segment[-4:], -1, -1, -1]


def test_sequence_prediction_keeps_full_life_state_for_requested_subset(tmp_path):
    torch.manual_seed(27)
    data = demonstrations(tmp_path)
    net = TinyPolicy().eval()
    with torch.no_grad():
        net.memory_projection.weight.normal_(0, .2)
    indices = np.arange(len(data.rows))
    full = predict(net, data, indices, 4, torch.device("cpu"))
    one_at_a_time = predict(net, data, indices, 1, torch.device("cpu"))
    subset = data.validation[2::2]
    requested = predict(net, data, subset, 2, torch.device("cpu"))
    assert np.allclose(full, one_at_a_time, atol=1e-6)
    assert np.allclose(full[subset], requested, atol=1e-6)
    # A frame-only evaluation genuinely differs once memory has nonzero learned weights.
    maps, scalars, mask, _ = data.batch(indices, torch.device("cpu"))
    independent = net.forward_sequence(maps[:, None], scalars[:, None], mask[:, None],
        net.initial_state(len(indices)), context=torch.from_numpy(data.contexts(indices))[:, None])[0]
    assert not np.allclose(full, independent[:, 0].log_softmax(-1).detach().numpy())


def test_burn_in_has_no_input_gradients_but_memory_receives_unroll_gradients(tmp_path):
    torch.manual_seed(29)
    data = demonstrations(tmp_path)
    segment = next(s for s in data.segments if s[0] in data.training)
    window = data.sequence_batch([segment[2]], 2, 3, torch.device("cpu"))
    window["scalars"].requires_grad_(True)
    net = TinyPolicy()
    with torch.no_grad():
        net.memory_projection.weight.fill_(.1)
    logits, _, _, _, _ = sequence_outputs(net, window, 2)
    loss = F.cross_entropy(logits.flatten(0, 1), window["labels"][:, 2:].flatten())
    loss.backward()
    assert not window["scalars"].grad[:, :2].any()
    assert window["scalars"].grad[:, 2:].abs().sum() > 0
    assert net.memory_gru.weight_ih_l0.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable"))])
def test_recurrent_imitation_learns_and_resumes_memory_optimizer(tmp_path, monkeypatch, device):
    torch.manual_seed(31)
    monkeypatch.setattr(imitate, "make_net", lambda _: TinyPolicy())
    data = demonstrations(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    initial = shared_checkpoint(run / "latest.pt")
    cfg = Config(epochs=1, steps_per_epoch=4, batch=6, sequence_burn_in=2,
                 sequence_unroll=3, learning_rate=.01, device=device)
    with ImitationTrainer(run, cfg) as learner:
        learner.learn(data)
    saved = torch.load(run / "latest.pt", weights_only=True)
    assert not torch.equal(saved["model"]["memory_gru.weight_ih_l0"], initial["memory_gru.weight_ih_l0"])
    assert torch.count_nonzero(saved["model"]["memory_projection.weight"]) > 0
    assert saved["imitation_state"]["head_supervision"] == "policy_with_prediction_preservation"
    assert saved["imitation_state"]["cumulative_samples"] == 24
    with ImitationTrainer(run, cfg) as resumed:
        assert resumed.optimizer_resumed and resumed.steps == 4
        assert all(float(s["step"]) == 4 for s in resumed.optimizer.state.values())
        resumed.learn(data)
        assert resumed.steps == 8
    assert [p.name for p in run.glob("*.pt")] == ["latest.pt"]


def test_prediction_preservation_reduces_drift_despite_frozen_output_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(imitate, "make_net", lambda _: TinyPolicy())
    data = demonstrations(tmp_path)
    drift = []
    for anchor in (0., 100.):
        torch.manual_seed(123)
        run = tmp_path / str(anchor)
        run.mkdir()
        shared_checkpoint(run / "latest.pt")
        cfg = Config(steps_per_epoch=16, batch=6, sequence_burn_in=2, sequence_unroll=3,
                     learning_rate=.01, prediction_anchor=anchor, device="cpu")
        with ImitationTrainer(run, cfg) as learner:
            def predictions():
                outputs = []
                with torch.no_grad():
                    for segment in data.segments:
                        x, s, mask, _ = data.batch(segment, torch.device("cpu"))
                        _, value, auxiliary, _, _ = learner.net.heads_sequence(
                            x[None], s[None], mask[None], learner.net.initial_state(1),
                            context=torch.from_numpy(data.contexts(segment))[None])
                        outputs.append(torch.cat([value[..., None], auxiliary], -1).flatten(0, 1))
                return torch.cat(outputs)
            before = predictions()
            old_policy = learner.net.pi.weight.detach().clone()
            learner.learn(data)
            drift.append(float((predictions() - before).square().mean()))
            assert not torch.equal(old_policy, learner.net.pi.weight)
    assert drift[0] > .01
    assert drift[1] < drift[0] * .2
