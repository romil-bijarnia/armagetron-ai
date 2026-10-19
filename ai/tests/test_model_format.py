"""V4 is the sole model format; unsupported checkpoints fail before loading weights."""

import pytest
import torch

from arma_ai.model import PolicyNet, make_net, net_version


def test_default_model_is_the_current_checkpoint_architecture():
    net = make_net()
    assert type(net) is PolicyNet and PolicyNet.__bases__ == (torch.nn.Module,)
    assert (net.version, net.n_maps, net.stack_prev, net.n_actions, net.n_aux) == (4, 4, True, 6, 3)
    assert sum(parameter.numel() for parameter in net.parameters()) == 8_155_373
    assert net_version(net.state_dict()) == 4
    assert int(net.observation_version) == 2
    assert int(net.memory_version) == 1 and net.memory_gru.input_size == 775


@pytest.mark.parametrize("version", [0, 1, 2, 3, 5])
def test_unsupported_architectures_are_rejected(version):
    with pytest.raises(ValueError, match="Only V4 networks are supported"):
        make_net(version)
    with pytest.raises(ValueError, match="Only V4 checkpoints are supported"):
        net_version({"net_version": torch.tensor(version)})


@pytest.mark.parametrize("state", [{}, {"towers.0.stem.weight": torch.zeros(1)},
                                 {"local.net.0.weight": torch.zeros(1)}])
def test_missing_version_is_never_guessed_from_weight_names(state):
    with pytest.raises(ValueError, match="net_version=4 marker is missing"):
        net_version(state)


@pytest.mark.parametrize("marker", [4, "4", torch.tensor([4]), torch.tensor([]),
                                  torch.tensor(4.0), torch.tensor(True), torch.tensor(float("nan"))])
def test_malformed_version_markers_are_rejected(marker):
    with pytest.raises(ValueError, match="expected the scalar integer net_version=4"):
        net_version({"net_version": marker})


def test_observation_revision_is_required_and_never_guessed():
    with pytest.raises(ValueError, match="observation schema 2"):
        net_version({"net_version": torch.tensor(4)})
    for marker in (1, torch.tensor(1), torch.tensor([2]), torch.tensor(2.0)):
        with pytest.raises(ValueError, match="observation schema 2"):
            net_version({"net_version": torch.tensor(4), "observation_version": marker})


def test_memory_revision_is_required_and_never_guessed():
    for marker in (None, 1, torch.tensor(0), torch.tensor([1]), torch.tensor(1.0)):
        state = {"net_version": torch.tensor(4), "observation_version": torch.tensor(2)}
        if marker is not None:
            state["memory_version"] = marker
        with pytest.raises(ValueError, match="memory schema 1"):
            net_version(state)
