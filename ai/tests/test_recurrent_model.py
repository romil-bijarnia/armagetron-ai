"""Persistent control-aware memory, including reset, gradient and dtype contracts."""

import pytest
import torch

from arma_ai.model import MEMORY_CONTEXT, PolicyNet, recurrent_context


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def learned_net():
    torch.manual_seed(31)
    net = PolicyNet().eval()
    with torch.no_grad():
        net.memory_projection.weight.normal_(0, .03)
        net.memory_projection.bias.normal_(0, .01)
        net.memory_gru.weight_ih_l0[:, 768:].normal_(0, .1)
    return net


def test_context_encodes_executed_controls_and_actual_elapsed_time():
    context = recurrent_context(torch.tensor([-1, 0, 4, 5]), torch.tensor([0., .05, .125, .1]))
    assert context.shape == (4, MEMORY_CONTEXT)
    torch.testing.assert_close(context[:, :6].sum(1), torch.tensor([0., 1., 1., 1.]))
    torch.testing.assert_close(context[:, -1], torch.tensor([0., 1., 2.5, 2.]))
    assert context[2, 4] == context[3, 5] == 1
    for actions, elapsed in (([6], [0.]), ([-2], [0.]), ([1.], [.05]), ([1], [-1.]), ([1], [float("nan")])):
        with pytest.raises(ValueError):
            recurrent_context(actions, elapsed)


def test_long_unroll_matches_steps_and_isolated_life_resets():
    net = learned_net()
    features = torch.randn(2, 96, 768)
    actions = torch.arange(192).reshape(2, 96) % 6
    context = recurrent_context(actions, torch.full((2, 96), .05))
    reset = torch.zeros(2, 96, dtype=torch.bool)
    reset[0, 31], reset[1, 67] = True, True
    initial = torch.randn(2, 256)
    before = initial.clone()
    with torch.no_grad():
        expected, final = net._remember(features, initial, reset, context)
        state, pieces = initial, []
        for tick in range(96):
            piece, state = net._remember(features[:, tick:tick + 1], state,
                                         reset[:, tick:tick + 1], context[:, tick:tick + 1])
            pieces.append(piece)
        torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(state, final, atol=2e-6, rtol=2e-5)
        for batch, born in ((0, 31), (1, 67)):
            isolated, isolated_state = net._remember(features[batch:batch + 1, born:], net.initial_state(1),
                                                     reset[batch:batch + 1, born:], context[batch:batch + 1, born:])
            torch.testing.assert_close(isolated[0], expected[batch, born:], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(isolated_state[0], final[batch], atol=2e-6, rtol=2e-5)
    assert torch.equal(initial, before)


def test_nonzero_memory_and_context_affect_decisions_and_receive_gradients():
    net = learned_net()
    features = torch.randn(2, 3, 768, requires_grad=True)
    context = recurrent_context(torch.tensor([[0, 4, 1], [2, 5, 3]]), torch.full((2, 3), .05))
    a, state = net._remember(features, net.initial_state(2), context=context)
    b, _ = net._remember(features, net.initial_state(2), context=torch.zeros_like(context))
    assert not torch.allclose(a, b) and state.abs().sum() > 0
    net.pi(a).square().mean().backward()
    assert features.grad.abs().sum() > 0
    assert (net.memory_gru.weight_ih_l0.grad[:, -7:].abs().sum(0) > 0).all()
    assert net.memory_gru.weight_hh_l0.grad.abs().sum() > 0
    assert net.memory_projection.weight.grad.abs().sum() > 0


def test_step_sequence_and_diagnostic_interfaces_keep_their_contracts():
    net = learned_net()
    maps = torch.randint(0, 256, (1, 4, 8, 64, 64), dtype=torch.uint8)
    scalars = torch.zeros(1, 4, 108)
    mask = torch.ones(1, 4, 6, dtype=torch.bool)
    context = recurrent_context(torch.tensor([[-1, 4, 0, 5]]), torch.tensor([[0., .05, .05, .1]]))
    with torch.no_grad():
        sequence = net.heads_sequence(maps, scalars, mask, net.initial_state(1), context=context)
        state, rows = net.initial_state(1), []
        for tick in range(4):
            row = net.heads_step(maps[:, tick], scalars[:, tick], mask[:, tick], state, context=context[:, tick])
            rows.append(row)
            state = row[-1]
        for head in range(4):
            # Batched vs singleton convolutions use different accumulation order;
            # the full-resolution ownership interpolation exposes that rounding.
            tolerance = 3e-5 if head == 3 else 3e-6
            torch.testing.assert_close(torch.stack([row[head] for row in rows], 1), sequence[head], atol=tolerance, rtol=2e-5)
        torch.testing.assert_close(state, sequence[-1], atol=3e-5, rtol=2e-5)
        frame_maps, frame_scalars = maps[:, 0], scalars[:, 0]
        feats = torch.cat([frame_scalars, mask[:, 0].float()], 1)
        assert len(net(frame_maps, frame_scalars)) == 2
        assert len(net.heads(frame_maps, frame_scalars)) == 4
        assert len(net.act(frame_maps, feats, greedy=True)) == 3
        assert len(net.act_step(frame_maps, feats, net.initial_state(1), greedy=True)) == 4


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_recurrent_gates_accept_autocast_features_and_float32_replay_state(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    net = learned_net().to(device)
    features = torch.randn(2, 8, 768, device=device).bfloat16()
    state = net.initial_state(2)
    with torch.autocast(device, dtype=torch.bfloat16):
        out, next_state = net._remember(features, state, context=torch.zeros(2, 8, 7, device=device))
    assert out.dtype == torch.bfloat16 and next_state.dtype == torch.float32
    out.float().square().mean().backward()
    assert torch.isfinite(net.memory_gru.weight_ih_l0.grad).all()
