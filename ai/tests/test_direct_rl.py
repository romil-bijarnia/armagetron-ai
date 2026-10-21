"""Direct policy learning must use action outcomes, never a search distribution."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arma_ai.teach import Config, Learner, parse_config


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(6))

    def initial_state(self, n, device):
        return torch.zeros(n, 1, device=device)

    def heads_sequence(self, maps, scal, mask, state, reset, context):
        shape = maps.shape[:2]
        zero = self.logits.sum() * 0
        return (self.logits.expand(*shape, 6), zero.expand(*shape), zero.expand(*shape, 3),
                zero.expand(*shape, 3, 1, 1), state)


def learning_case(outcome):
    net = TinyPolicy()
    learner = Learner.__new__(Learner)
    learner.cfg = Config(sequence_burn_in=0, entropy_coef=0, value_coef=0, aux_coef=0, own_coef=0)
    learner.device, learner.main, learner.imitation_state = torch.device("cpu"), net, {}
    data = dict(maps=torch.zeros(1, 2, 1), scalars=torch.zeros(1, 2, 1),
                mask=torch.ones(1, 2, 6, dtype=torch.bool), reset=torch.zeros(1, 2, dtype=torch.bool),
                context=torch.zeros(1, 2, 7), learn=torch.ones(1, 2, dtype=torch.bool),
                action=torch.full((1, 2), 4, dtype=torch.long),
                behavior_logp=torch.full((1, 2), -math.log(6)), behavior_value=torch.zeros(1, 2),
                value=torch.full((1, 2), float(outcome)), has_policy=torch.zeros(1, 2, dtype=torch.bool),
                aux=torch.zeros(1, 2, 3), own=torch.zeros(1, 2, 1, 1, dtype=torch.long),
                own_ok=torch.zeros(1, 2, dtype=torch.bool))
    return learner, net, data


@pytest.mark.parametrize("outcome", [1, -1])
def test_executed_action_becomes_more_likely_after_a_win_and_less_after_a_loss(outcome):
    learner, net, data = learning_case(outcome)
    opt = torch.optim.SGD(net.parameters(), lr=.1)
    before = net.logits.softmax(-1)[4].item()
    result = learner.step(net, opt, data)
    after = net.logits.softmax(-1)[4].item()
    assert (after - before) * outcome > 0
    assert all(torch.isfinite(v) for k, v in result.items() if k != "n")
    assert "search_kl" not in result


def test_clipping_stops_reinforcing_an_action_already_made_too_likely():
    learner, net, data = learning_case(1)
    with torch.no_grad():
        net.logits[4] = 2
    before = net.logits.detach().clone()
    result = learner.step(net, torch.optim.SGD(net.parameters(), lr=.1), data)
    torch.testing.assert_close(before, net.logits, rtol=0, atol=0)
    assert result["clip_fraction"] == 1


def test_search_targets_are_rejected_even_if_their_policy_would_agree():
    learner, net, data = learning_case(1)
    data["has_policy"][:] = True
    with pytest.raises(ValueError, match="searched decisions"):
        learner.step(net, torch.optim.SGD(net.parameters(), lr=.1), data)


def test_readiness_rejects_searched_and_stale_policy_samples():
    learner = Learner.__new__(Learner)
    learner.cfg = Config(max_policy_lag=4)
    ring = SimpleNamespace(window=lambda: np.arange(5), gen=np.arange(5),
                           has_policy=np.array([0, 0, 0, 1, 0]),
                           policy_update=np.array([0, 5, 6, 10, 10]), generation=np.zeros(5))
    assert learner.ready_samples(ring, update=10).tolist() == [2, 4]


@pytest.mark.parametrize("flag", ["--sims", "--macro", "--search-prob", "--gumbel"])
def test_removed_teacher_flags_cannot_reenable_search(flag):
    with pytest.raises(SystemExit):
        parse_config([flag, "1"])
