"""Synthetic contracts for the temporary memory/control comparison; no real game runs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from arma_ai import protocol as P
from arma_ai.memory_experiment import (AdapterPolicy, ComparisonConfig, FeatureCache, FeatureLife,
    GRUResidual, SimulatorActor, StatelessResidual, Window, cache_features, compare_adapters,
    sequence_predictions, training_loss, training_windows, window_batch)


def life(name, length=6, offset=0, source=None, width=4):
    features = torch.arange(length, dtype=torch.float32)[:, None].expand(-1, width).clone() + offset
    return FeatureLife(name, source or name, features, torch.ones(length, 6, dtype=torch.bool),
                       torch.arange(length) % 6, torch.arange(length, dtype=torch.float64) * .05)


def cache(length=70):
    return FeatureCache((life("train", length), life("validation", 6, offset=100)),
                        ("train",), ("validation",), torch.ones(6))


def policy(recurrent=True):
    torch.manual_seed(13)
    return AdapterPolicy(nn.Linear(4, 6), GRUResidual(4, 3) if recurrent else StatelessResidual(4, 6))


def activate_residual(net):
    with torch.no_grad():
        net.adapter.projection.weight.fill_(.1)


def test_adapters_have_the_promised_matched_parameter_budget():
    gru = GRUResidual()
    control = StatelessResidual()
    recurrent = sum(p.numel() for p in gru.parameters())
    stateless = sum(p.numel() for p in control.parameters())
    assert recurrent == 985_344 and stateless == 984_448
    assert abs(recurrent - stateless) / recurrent < .001


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU is unavailable"))])
def test_zero_residual_preserves_exact_base_logits(recurrent, device):
    net = policy(recurrent).to(device)
    features = torch.randn(2, 5, 4, device=device)
    legal = torch.ones(2, 5, 6, dtype=torch.bool, device=device)
    legal[:, :, 2] = False
    reset = torch.zeros(2, 5, dtype=torch.bool, device=device)
    reset[:, 0], reset[1, 3] = True, True
    actual, _ = net(features, legal, reset=reset)
    assert torch.equal(actual, net.base_logits(features, legal))
    assert all(not p.requires_grad for p in net.policy_head.parameters())


def test_recurrent_resets_prevent_another_life_from_changing_predictions():
    net = policy()
    activate_residual(net)
    a, b = life("a", 7, offset=1), life("b", 5, offset=-2)
    first = sequence_predictions(net, [a, b], chunk_size=2)
    reversed_order = sequence_predictions(net, [b, a], chunk_size=2)
    whole = sequence_predictions(net, [b], chunk_size=100)
    assert torch.equal(first["b"], reversed_order["b"])
    assert torch.allclose(first["b"], whole["b"], atol=1e-6)
    x = torch.cat([a.features, b.features])[None]
    mask = torch.cat([a.legal, b.legal])[None]
    reset = torch.zeros(1, len(a.features) + len(b.features), dtype=torch.bool)
    reset[0, len(a.features)] = True
    with_reset, _ = net(x, mask, reset=reset)
    alone, _ = net(b.features[None], b.legal[None])
    without_reset, _ = net(x, mask)
    assert torch.allclose(with_reset[:, len(a.features):], alone, atol=1e-6)
    assert not torch.allclose(without_reset[:, len(a.features):], alone, atol=1e-6)
    ablated = sequence_predictions(net, [b], chunk_size=2, reset_each_decision=True)["b"]
    independent = torch.cat([net(b.features[i:i + 1], b.legal[i:i + 1])[0].log_softmax(-1)
                             for i in range(len(b.features))])
    assert torch.allclose(ablated, independent, atol=1e-6)
    assert not torch.allclose(first["b"], ablated, atol=1e-6)


def test_windows_keep_real_burnin_and_mask_short_life_padding():
    data = cache()
    windows = training_windows(data)
    assert [(w.start, w.end) for w in windows] == [(0, 32), (32, 64), (64, 70)]
    batch = window_batch(data, windows)
    assert batch.learn.sum(1).tolist() == [32, 32, 6]
    assert not batch.learn[:, :16].any()
    assert batch.reset[0].nonzero().flatten().tolist() == [16]
    assert batch.reset[1].nonzero().flatten().tolist() == [0]
    assert batch.features[0, 16:, 0].tolist() == list(range(32))
    assert batch.features[1, :16, 0].tolist() == list(range(16, 32))
    assert batch.features[2, 16:22, 0].tolist() == list(range(64, 70))
    assert not batch.features[2, 22:].any()
    with pytest.raises(ValueError, match="held-out"):
        window_batch(data, [Window("validation", 0, 3)])


def test_burnin_state_is_detached_and_only_unroll_positions_receive_gradients():
    data = cache()
    batch = window_batch(data, [training_windows(data)[1]])
    batch.features.requires_grad_(True)
    net = policy()
    activate_residual(net)
    loss, tokens = training_loss(net, batch, 16, torch.ones(6), 0)
    loss.backward()
    assert tokens == 32 and torch.isfinite(loss)
    assert not batch.features.grad[:, :16].any()
    assert batch.features.grad[:, 16:].abs().sum() > 0
    assert net.adapter.gru.weight_ih_l0.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.policy_head.parameters())


def test_source_life_segments_cannot_leak_across_splits():
    with pytest.raises(ValueError, match="same split"):
        FeatureCache((life("a", source="shared"), life("b", source="shared")),
                     ("a",), ("b",), torch.ones(6))
    bad = life("bad")
    times = bad.times.clone()
    times[3:] += .05
    with pytest.raises(ValueError, match="missing decision"):
        FeatureLife("bad", "bad", bad.features, bad.legal, bad.actions, times)


class TinyBase(nn.Module):
    observation_schema = 2

    def __init__(self):
        super().__init__()
        self.pi = nn.Linear(4, 6)
        self.v = nn.Linear(4, 1)

    def trunk_out(self, maps, scalars):
        return torch.stack([maps[:, 0, 0, 0].float(), scalars[:, 0], scalars[:, 1], scalars[:, 2]], 1)


def small_demonstrations():
    rows = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 3, 0, 3], [1, 0, 0, 0], [1, 1, 0, 0]], np.int32)
    episodes = [{"times": np.arange(4) * .05, "masks": np.full((4, 1), 63, np.uint8)} for _ in range(2)]

    def batch(indices, device):
        maps = torch.zeros(len(indices), 8, 64, 64, dtype=torch.uint8, device=device)
        maps[:, 0, 0, 0] = torch.from_numpy(rows[indices, 1].astype(np.uint8)).to(device)
        return maps, torch.zeros(len(indices), P.N_SCALARS, device=device), None, None

    return SimpleNamespace(rows=rows, episodes=episodes, training=np.array([0, 1, 2]), validation=np.array([3, 4]),
                           labels=np.array([0, 1, 2, 0, 1]), weights=np.ones(6), batch=batch)


def test_cache_splits_gaps_and_rejects_frame_level_validation_leak():
    base, data = TinyBase(), small_demonstrations()
    result = cache_features(base, data, batch=2)
    assert [len(x.features) for x in result.split("training")] == [2, 1]
    assert len(result.validation_ids) == 1 and result.metadata["observation_version"] == 2
    assert base.training  # caching restores the caller's module mode
    data.validation = np.array([2, 3, 4])
    data.training = np.array([0, 1])
    with pytest.raises(ValueError, match="source life"):
        cache_features(base, data)


def test_simulator_adapter_keeps_game_states_separate_and_rejects_search_nodes():
    base, net = TinyBase(), policy()
    activate_residual(net)
    actor = SimulatorActor(base, net)
    simulator = SimpleNamespace(game_ids=np.array([7, 8]), roots=np.ones(2, np.uint8), masks=np.full(2, 63, np.uint8),
                                maps=np.ones((2, 8, 64, 64), np.uint8), scalars=np.zeros((2, P.N_SCALARS), np.float32))
    answers = []
    simulator.feed = lambda logits, values: answers.append(logits.copy())
    actor.feed(simulator, 2)
    first = answers[-1]
    actor.reset(7)
    actor.feed(simulator, 2)
    assert np.allclose(answers[-1][0], first[0])
    assert not np.allclose(answers[-1][1], first[1])
    statistics = actor.statistics()
    assert statistics["decisions"] == 4 and statistics["inference_batches"] == 2
    assert statistics["inference_batch_size_min"] == statistics["inference_batch_size_max"] == 2
    assert statistics["inference_batch_size_mean"] == 2
    simulator.roots[0] = 0
    with pytest.raises(ValueError, match="no search"):
        actor.feed(simulator, 2)


def test_synthetic_comparison_uses_identical_tokens_and_only_writes_json(tmp_path):
    base, data = TinyBase(), cache(8)
    initial = {key: value.clone() for key, value in base.state_dict().items()}
    cfg = ComparisonConfig(updates=2, batch_windows=1, burn_in=1, unroll=3, gru_hidden=3, stateless_hidden=6)
    report, models = compare_adapters(base, data, cfg, report_path=tmp_path / "report.json")
    assert report["models"]["gru"]["learned_decisions"] == report["models"]["stateless"]["learned_decisions"]
    assert report["models"]["gru"]["updates"] == report["models"]["stateless"]["updates"] == 2
    assert json.loads((tmp_path / "report.json").read_text())["config"]["burn_in"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]
    assert all(torch.equal(initial[key], value) for key, value in base.state_dict().items())
    assert set(models) == {"gru", "stateless", "base"}
    with pytest.raises(ValueError, match="only a JSON"):
        compare_adapters(base, data, cfg, report_path=tmp_path / "forbidden.pt")
