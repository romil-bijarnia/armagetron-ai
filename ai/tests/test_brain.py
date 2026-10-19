"""The brain page's view of the network matches the network itself."""

import base64

import numpy as np
import pytest
import torch

from arma_ai import protocol as P
from arma_ai.brain import GLOBAL_PLANES, MOVES, attention_payload, default_checkpoint, layer_of, layer_table, learned, scalar_names, stacked, think, trace
from arma_ai.model import PolicyNet, make_net, recurrent_context


def _units(L):
    return int(np.prod(L["shape"]))


def test_scalar_names_preserve_old_indices_and_name_appended_enemy_resources():
    names = scalar_names()
    assert len(names) == P.N_SCALARS == 108
    assert names[64] == "enemy 1 ahead" and names[87] == "enemy 3 present"
    assert names[88:96] == ["unused"] * 8
    for enemy in range(3):
        assert names[96 + 4 * enemy:100 + 4 * enemy] == [
            f"enemy {enemy + 1} {field}" for field in
            ("brake reserve", "rubber used / budget", "left turn wait", "right turn wait")]


def test_enemy_resource_changes_reach_opponent_trace_without_leaking_to_own_sensors():
    torch.manual_seed(31)
    net = PolicyNet().eval()
    with torch.no_grad():
        net.opponents[0].weight[:, 8:].normal_(0, .1)  # a learned, non-neutral resource path
    rng = np.random.default_rng(32)
    maps = rng.integers(0, 256, (2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    scalars[[71, 79, 87]] = 1
    changed = scalars.copy()
    changed[96:108] += 10
    original_trace = dict(think(net, maps, scalars, np.ones(P.N_ACTIONS, bool)).layers)
    changed_trace = dict(think(net, maps, changed, np.ones(P.N_ACTIONS, bool)).layers)
    for stage in ("scalars_h1", "scalars_h2", "opponents_query"):
        np.testing.assert_array_equal(original_trace[stage], changed_trace[stage])
    assert not np.allclose(original_trace["opponents_h1"], changed_trace["opponents_h1"])


def test_trace_layers_match_the_table_and_the_real_forward_pass():
    torch.manual_seed(0)
    net = PolicyNet().eval()
    rng = np.random.default_rng(0)
    maps = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    prev = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.array([True, True, False, True, True, False])
    outs, probs, value = trace(net, stacked(maps, prev), scalars, mask)
    table = [L for L in layer_table(net) if L["kind"] in ("conv", "dense")]
    assert [k for k, _ in outs] == [L["id"] for L in table]
    assert all(list(v.shape) == L["shape"] for (_, v), L in zip(outs, table))
    with torch.no_grad():
        logits, v = net(torch.from_numpy(stacked(maps, prev))[None], torch.from_numpy(scalars)[None],
                        torch.from_numpy(mask)[None])
    assert np.allclose(probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(value - float(v[0])) < 1e-4
    assert probs[2] < 1e-6
    names = [L["name"] for L in table]
    assert names[:3] == ["arena view · stem", "arena view · block 1", "arena view · block 2"]
    assert "nearby view · down to 8×8" in names and "arena view · summary" in names


def test_the_heads_the_page_paints_are_the_networks_own():
    """Ownership, the aux predictions and the attention rows come from the same pass as the move."""
    torch.manual_seed(0)
    net = make_net().eval()
    arena_index = net.tower_names.index("arena")
    rng = np.random.default_rng(3)
    maps = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    prev = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.array([True, False, True, True, False, True])[:net.n_actions]
    t = think(net, stacked(maps, prev), scalars, mask)
    with torch.no_grad():
        logits, v, aux, own = net.heads(torch.from_numpy(stacked(maps, prev))[None], torch.from_numpy(scalars)[None],
                                        torch.from_numpy(mask)[None])
    assert np.allclose(t.probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(t.value - float(v[0])) < 1e-4
    assert np.allclose(t.aux, torch.sigmoid(aux[0]).numpy(), atol=1e-5)
    # the teacher upsamples the 16x16 ownership logits; the page gets the 16x16 probabilities themselves
    side = P.GRID // 4
    assert t.own.shape == (3, side, side)
    assert np.allclose(t.own.sum(0), 1, atol=1e-5)
    with torch.no_grad():  # the teacher upsamples these very logits to 64x64 for its loss
        x = torch.from_numpy(stacked(maps, prev))[None]
        direct = torch.softmax(net.own(net.towers[arena_index].features(net.tower_input(x, arena_index)))[0], 0).numpy()
    assert np.allclose(t.own, direct, atol=1e-5)
    assert own.shape == (1, 3, P.GRID, P.GRID)
    # one attention tensor per arena attention layer; every row is a distribution over the cells
    assert len(t.attention) == len(net.towers[arena_index].attn) == 2
    for a in t.attention:
        assert a.shape == (4, side * side, side * side)
        assert np.allclose(a.sum(-1), 1, atol=1e-4)
    # the payload names the cell my head is in when none is hovered
    global_map = np.zeros((P.GRID, P.GRID), np.uint8)
    global_map[9, 50] |= 1 << GLOBAL_PLANES.index("own head")
    pay = attention_payload(t.attention, None, global_map)
    assert pay["cell"] == (9 // 4) * side + 50 // 4 and pay["side"] == side and pay["heads"] == 4
    rows = np.frombuffer(base64.b64decode(pay["rows"]), np.uint8).reshape(2, 4, side * side)
    assert rows.max(-1).min() == 255  # every row scaled so its strongest cell is full
    got = np.frombuffer(base64.b64decode(pay["got"]), np.uint8).reshape(2, side * side)
    assert got.max(-1).min() == 255
    assert attention_payload(t.attention, 7, global_map)["cell"] == 7


def test_every_weight_belongs_to_a_stage_the_page_draws():
    net = PolicyNet().eval()
    ids = {L["id"] for L in layer_table(net)} | {"aux", "own"}
    for name, w in net.state_dict().items():
        if w.ndim >= 2:
            assert layer_of(name, net) in ids, name
    old = {k: v.clone() for k, v in net.state_dict().items()}
    new = {k: v.clone() for k, v in old.items()}
    new["towers.0.res.1.c1.weight"] += 0.1  # a big move relative to the weights' size
    new["pi.weight"] *= 1.001  # a small one
    moved = learned(net, new, old)
    assert moved[f"{net.tower_names[0]}_c3"] == 1.0 and 0 < moved["policy"] < 1 and "arena_c1" not in moved
    assert learned(net, old, old) == {}


def test_input_layout_extension_is_not_reported_as_a_learning_update():
    net = PolicyNet().eval()
    current = net.state_dict()
    old = dict(current)
    old["opponents.0.weight"] = current["opponents.0.weight"][:, :8].clone()
    assert learned(net, current, old) == {}


@pytest.mark.parametrize("present", [(1, 0, 1), (0, 0, 0)])
def test_v4_trace_measures_actual_forward_stages_and_opponent_routing(present):
    torch.manual_seed(11)
    net = PolicyNet().eval()
    with torch.no_grad():
        for layer in net.conditioning:
            layer.weight.normal_(0, 0.04)
            layer.bias.normal_(0, 0.1)
    rng = np.random.default_rng(13)
    maps = rng.integers(0, 256, (2 * P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    scalars[[71, 79, 87]] = present
    mask = np.array([True, False, True, True, True, False])
    thought = think(net, maps, scalars, mask)
    traced = dict(thought.layers)
    actual = {}
    handles = []

    def capture(module, name, transform=lambda x: x):
        def save(_module, _args, result):
            actual[name] = transform(result)[0].detach().clone().numpy()
        handles.append(module.register_forward_hook(save))

    for index, group in enumerate(net.tower_names):
        tower = net.towers[index]
        capture(tower.stem, f"{group}_c1", torch.relu)
        if index < 2:
            capture(tower.pre, f"{group}_pre")
            capture(tower.reduce, f"{group}_reduce", torch.relu)
            capture(net.conditioning[index], f"{group}_modulation")
            def save_condition(_module, args, key=f"{group}_condition"):
                actual[key] = args[0][0].detach().clone().numpy()
            handles.append(tower.down0.register_forward_pre_hook(save_condition))
        for block_index, block in enumerate(tower.res):
            capture(block, f"{group}_c{block_index + 2}")
        for attention_index, layer in enumerate(tower.attn):
            capture(layer, f"{group}_a{attention_index + 1}",
                    lambda cells: cells.transpose(1, 2).reshape(1, 128, 16, 16))
        capture(tower.down0, f"{group}_c{len(tower.res) + 2}", torch.relu)
        capture(tower.down1, f"{group}_c{len(tower.res) + 3}", torch.relu)
        capture(tower.fc, f"{group}_fc", torch.relu)
    capture(net.scalars[1], "scalars_h1")
    capture(net.scalars[3], "scalars_h2")
    capture(net.opponents[1], "opponents_h1", lambda tokens: tokens.flatten(1))
    capture(net.opponents[3], "opponents_h2", lambda tokens: tokens.flatten(1))
    capture(net.opponent_query, "opponents_query", torch.tanh)
    capture(net.trunk[1], "trunk1")
    capture(net.trunk[3], "trunk2")
    capture(net.memory_gru, "memory_state", lambda result: result[1][0])
    capture(net.memory_projection, "memory_residual", lambda result: result[:, 0])
    def memory_inputs(_module, args):
        actual["memory_context"] = args[0][0, 0, -7:].detach().clone().numpy()
        actual["memory_in"] = args[1][0, 0].detach().clone().numpy()
    def policy_inputs(_module, args):
        actual["memory_output"] = args[0][0].detach().clone().numpy()
    handles.extend([net.memory_gru.register_forward_pre_hook(memory_inputs),
                    net.pi.register_forward_pre_hook(policy_inputs)])
    try:
        with torch.no_grad():
            logits, value, aux, _ = net.heads(torch.from_numpy(maps)[None], torch.from_numpy(scalars)[None],
                                            torch.from_numpy(mask)[None])
    finally:
        for handle in handles:
            handle.remove()
    for name, output in actual.items():
        assert np.allclose(traced[name], output, atol=1e-5), name
    assert set(traced) == set(actual) | {"opponents_weights", "opponents_pool"}
    assert np.allclose(thought.probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(thought.value - value.item()) < 1e-5 and -1 <= thought.value <= 1
    assert np.allclose(thought.aux, torch.sigmoid(aux[0]).numpy(), atol=1e-5)
    assert thought.aux.shape == (3,)
    weights = thought.opponents
    assert np.array_equal(weights, traced["opponents_weights"])
    assert np.array_equal(weights[np.array(present) == 0], np.zeros(3 - sum(present)))
    assert abs(weights.sum() - bool(sum(present))) < 1e-6
    assert np.allclose(traced["opponents_pool"],
                       (traced["opponents_h2"].reshape(3, 64) * weights[:, None]).sum(0), atol=1e-6)
    if not any(present):
        assert np.array_equal(traced["opponents_pool"], np.zeros(64))
    table = [layer for layer in layer_table(net) if layer["kind"] in ("conv", "dense")]
    assert [name for name, _ in thought.layers] == [layer["id"] for layer in table]
    assert all(list(output.shape) == layer["shape"] for (_, output), layer in zip(thought.layers, table))
    order = list(traced)
    assert order.index("arena_fc") < order.index("nearby_c1") < order.index("nearby_modulation")
    assert order.index("nearby_condition") < order.index("nearby_c5") < order.index("close_c1")


def test_v4_six_control_masks_and_action_names():
    net = make_net().eval()
    policy = next(layer for layer in layer_table(net) if layer["id"] == "policy")
    assert policy["names"] == MOVES[:net.n_actions]
    assert policy["shape"] == [net.n_actions]
    maps = np.zeros((2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
    scalars = np.zeros(P.N_SCALARS, np.float32)
    for action in range(6):
        mask = np.arange(6) == action
        thought = think(net, maps, scalars, mask)
        assert np.array_equal(thought.probs, mask.astype(np.float32))
    assert policy["names"][-2:] == ["Left + brake", "Right + brake"]


def test_default_checkpoint_uses_only_current_run(tmp_path):
    assert default_checkpoint(tmp_path) == tmp_path / "runs/blacklight/latest.pt"
    directory = tmp_path / "runs/blacklight"
    directory.mkdir(parents=True)
    (directory / "latest.pt").touch()
    assert default_checkpoint(tmp_path) == directory / "latest.pt"
    (directory / "actor.pt").touch()
    assert default_checkpoint(tmp_path) == directory / "latest.pt"


def test_inspector_uses_the_exact_predecision_memory_for_every_decision_head():
    torch.manual_seed(117)
    net = make_net().eval()
    with torch.no_grad():
        net.memory_projection.weight.normal_(0, .03)
        net.memory_projection.bias.normal_(0, .01)
    rng = np.random.default_rng(117)
    maps = rng.integers(0, 256, (2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
    scalars = rng.normal(size=P.N_SCALARS).astype(np.float32)
    mask = np.array([True, False, True, True, False, True])
    state = torch.randn(1, 256)
    context = recurrent_context(torch.tensor([4]), torch.tensor([.073]))
    before, previous_controls = state.clone(), context.clone()
    thought = think(net, maps, scalars, mask, state=state, context=context)
    with torch.no_grad():
        logits, value, aux, _, next_state = net.heads_step(
            torch.from_numpy(maps)[None], torch.from_numpy(scalars)[None], torch.from_numpy(mask)[None],
            state, context=context)
    assert np.allclose(thought.probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(thought.value - value.item()) < 1e-5
    assert np.allclose(thought.aux, torch.sigmoid(aux[0]).numpy(), atol=1e-5)
    assert torch.allclose(thought.state, next_state, atol=1e-5)
    traced = dict(thought.layers)
    assert np.array_equal(traced["memory_in"], before[0].numpy())
    assert np.array_equal(traced["memory_context"], context[0].numpy())
    assert np.allclose(traced["memory_state"], next_state[0].numpy(), atol=1e-5)
    assert np.allclose(traced["memory_output"], traced["trunk2"] + traced["memory_residual"], atol=1e-6)
    assert torch.equal(state, before) and torch.equal(context, previous_controls)
    fresh = think(net, maps, scalars, mask)
    assert not np.allclose(thought.probs, fresh.probs, atol=1e-6)
