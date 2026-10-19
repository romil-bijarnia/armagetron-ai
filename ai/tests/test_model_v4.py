"""Combat model contracts: actual controls, spatial resolution and enemy masking."""

import numpy as np
import torch

from arma_ai import protocol as P
from arma_ai.model import PolicyNet


def test_observation_branches_keep_each_enemys_resources_with_its_token():
    net = PolicyNet()
    scalars = torch.arange(P.N_SCALARS)[None]
    own = net.scalar_input(scalars)
    enemies = net.opponent_input(scalars)
    assert own.tolist() == [list(range(64)) + list(range(88, 96))]
    assert enemies.shape == (1, 3, 12)
    for enemy in range(3):
        assert enemies[0, enemy].tolist() == (list(range(64 + 8 * enemy, 72 + 8 * enemy))
                                             + list(range(96 + 4 * enemy, 100 + 4 * enemy)))
    assert net.scalars[0].in_features == 72 and net.opponents[0].in_features == 12


def test_enemy_order_and_absent_enemy_garbage_cannot_change_the_policy():
    torch.manual_seed(12)
    net = PolicyNet().eval()
    with torch.no_grad():
        net.opponents[0].weight[:, 8:].normal_(0, .1)  # exercise the resource path after it has learned
    maps = torch.randint(0, 256, (1, 8, 64, 64), dtype=torch.uint8)
    scalars = torch.randn(1, P.N_SCALARS)
    enemies = scalars[:, 64:88].reshape(1, 3, 8)
    enemies[:, :, 7] = torch.tensor([[1.0, 1.0, 0.0]])
    reordered = scalars.clone()
    reordered[:, 64:88] = enemies[:, [2, 0, 1]].flatten(1)
    resources = scalars[:, 96:108].reshape(1, 3, 4)
    reordered[:, 96:108] = resources[:, [2, 0, 1]].flatten(1)
    absent_changed = scalars.clone()
    absent_changed[:, 80:87] = 1000 * torch.randn(1, 7)
    absent_changed[:, 104:108] = 1000 * torch.randn(1, 4)
    mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        policy, value = net(maps, scalars, mask)
        for other in (reordered, absent_changed):
            p, v = net(maps, other, mask)
            assert torch.allclose(p, policy, atol=1e-5)
            assert torch.allclose(v, value, atol=1e-5)
    assert net.scalars[0].in_features == 72


def test_empty_opponent_set_is_finite_and_zero_in_half_precision():
    torch.manual_seed(14)
    net = PolicyNet().half().eval()
    with torch.no_grad():
        net.opponents[0].weight[:, 8:].normal_(0, .1)
    scalars = torch.randn(2, P.N_SCALARS, dtype=torch.float16)
    scalars[:, [71, 79, 87]] = 0
    context = torch.randn(2, 896, dtype=torch.float16)
    with torch.no_grad():
        pooled, weights, _, _ = net.attend_opponents(scalars, context)
    assert torch.isfinite(pooled).all() and torch.isfinite(weights).all()
    assert torch.count_nonzero(pooled) == 0 and torch.count_nonzero(weights) == 0


def test_close_geometry_is_processed_at_32_cells_and_compound_controls_are_executable():
    torch.manual_seed(15)
    net = PolicyNet().eval()
    shapes = {}
    hooks = []
    for t in (0, 1):
        for name in ("stem", "pre", "reduce"):
            key = (t, name)
            hooks.append(getattr(net.towers[t], name).register_forward_hook(
                lambda _, __, out, key=key: shapes.__setitem__(key, tuple(out.shape))))
    maps = torch.randint(0, 256, (2, 8, 64, 64), dtype=torch.uint8)
    scalars = torch.zeros(2, P.N_SCALARS)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 4] = True
    mask[1, 5] = True
    with torch.no_grad():
        logits, value, aux, own = net.heads(maps, scalars, mask)
    for hook in hooks:
        hook.remove()
    for t in (0, 1):
        assert shapes[t, "stem"] == shapes[t, "pre"] == (2, 48, 32, 32)
        assert shapes[t, "reduce"] == (2, 64, 16, 16)
    assert logits.argmax(1).tolist() == [4, 5]
    assert np.allclose(torch.softmax(logits, 1).numpy().sum(1), 1)
    assert torch.all(value.abs() <= 1)
    assert aux.shape == (2, 3) and own.shape == (2, 3, 64, 64)


def test_policy_can_learn_from_each_map_frame_and_each_scalar_branch():
    """Policy-only supervision must reach every observation path, without auxiliary losses.

    A model can produce the right output shapes while silently ignoring a map, history,
    or opponent branch. Inspect gradients at the unpacked inputs to catch that failure.
    """
    torch.manual_seed(21)
    net = PolicyNet()
    observed = [[] for _ in net.unpack]

    def remember(index):
        def hook(_, __, out):
            out.requires_grad_(True)
            out.retain_grad()
            observed[index].append(out)
        return hook

    hooks = [unpack.register_forward_hook(remember(i)) for i, unpack in enumerate(net.unpack)]
    maps = torch.randint(0, 256, (2, 8, 64, 64), dtype=torch.uint8)
    scalars = torch.randn(2, P.N_SCALARS)
    scalars[:, [71, 79, 87]] = 1  # distinct, present opponents make attention learnable
    scalars.requires_grad_(True)
    try:
        logits, _ = net(maps, scalars, torch.ones(2, 6, dtype=torch.bool))
        torch.nn.functional.cross_entropy(logits, torch.tensor([4, 5])).backward()
    finally:
        for hook in hooks:
            hook.remove()

    for i, frames in enumerate(observed):
        assert len(frames) == 2, f"map {i} is missing its current or previous frame"
        for frame, value in enumerate(frames):
            assert value.grad is not None, f"map {i}, frame {frame} does not receive policy gradients"
            assert torch.isfinite(value.grad).all()
            assert value.grad.abs().sum() > 0, f"map {i}, frame {frame} is disconnected"
    assert torch.isfinite(scalars.grad).all()
    assert scalars.grad[:, :64].abs().sum() > 0
    assert scalars.grad[:, 64:88].abs().sum() > 0
    # New columns begin neutral to preserve the existing fighter. The first update must
    # nevertheless teach every resource input, and then gradients must reach those inputs.
    assert torch.count_nonzero(net.opponents[0].weight[:, 8:]) == 0
    assert (net.opponents[0].weight.grad[:, 8:].abs().sum(0) > 0).all()
    for name, module in (
        ("arena attention", net.towers[2].attn[0].qkv),
        ("nearby conditioning", net.conditioning[0]),
        ("close conditioning", net.conditioning[1]),
        ("opponent selection", net.opponent_query),
    ):
        assert module.weight.grad is not None, name
        assert torch.isfinite(module.weight.grad).all(), name
        assert module.weight.grad.abs().sum() > 0, name
    torch.optim.SGD(net.parameters(), lr=.01).step()
    net.zero_grad(set_to_none=True)
    learned_scalars = scalars.detach().clone().requires_grad_(True)
    learned_logits, _ = net(maps, learned_scalars, torch.ones(2, 6, dtype=torch.bool))
    torch.nn.functional.cross_entropy(learned_logits, torch.tensor([4, 5])).backward()
    assert (learned_scalars.grad[:, 96:108].reshape(2, 3, 4).abs().sum((0, 1)) > 0).all()
