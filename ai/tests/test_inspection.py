"""The inspector must describe and intervene on the model it actually displays."""

import numpy as np
import pytest
import torch

from arma_ai import protocol as P
from arma_ai.brain import Hub, layer_of, layer_table, stacked
from arma_ai.inspection import activation_stats, describe, input_sensitivity
from arma_ai.model import make_net, recurrent_context


def test_structure_accounts_for_all_parameters_and_true_head_connections():
    net = make_net().eval()
    layers = layer_table(net)
    structure = describe(net, layers, layer_of)
    ids = {layer["id"] for layer in layers}
    assert sum(layer["parameters"] for layer in layers) == sum(p.numel() for p in net.parameters())
    assert structure["parameters"] == sum(p.numel() for p in net.parameters()) == 8_155_373
    assert structure["observation_version"] == 2 and structure["scalar_count"] == 108
    assert all(a in ids and b in ids for a, b in structure["edges"])
    assert structure["history_frames"] == ((2))
    assert [("arena_a2"), "own"] in structure["edges"]
    assert ["memory_output", "aux"] in structure["edges"]
    assert ["trunk2", "memory_output"] in structure["edges"]
    assert structure["memory"] == {"kind": "GRU", "hidden": 256, "persistent": True,
                                   "previous_action": True, "elapsed_time": True,
                                   "reset": "new life, round, missing decision, player or checkpoint change"}
    by_id = {layer["id"]: layer for layer in layers}
    assert by_id["memory_state"]["parameters"] == sum(p.numel() for p in net.memory_gru.parameters())
    assert by_id["memory_residual"]["parameters"] == sum(p.numel() for p in net.memory_projection.parameters())
    assert "775 inputs" in by_id["memory_state"]["operation"]
    assert all(["memory_output", head] in structure["edges"] for head in ("policy", "value", "aux"))
    assert ["trunk2", "policy"] not in structure["edges"]
    arena_index = net.tower_names.index("arena")
    first_attention = next(layer for layer in layers if layer["id"] == "arena_a1")
    assert first_attention["parameters"] == sum(p.numel() for p in net.towers[arena_index].attn[0].parameters()) + net.towers[arena_index].pos.numel()
    assert next(layer for layer in layers if layer["id"] == f"{net.tower_names[0]}_c4")["short"].startswith("Global pool")


def test_input_test_reproduces_the_original_and_an_independent_substitution():
    torch.manual_seed(7)
    net = make_net().eval()
    with torch.no_grad():
        net.opponents[0].weight[:, 8:].normal_(0, .1)
    rng = np.random.default_rng(17)
    maps = stacked(rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), np.uint8),
                   rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), np.uint8))
    original = maps.copy()
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    scalars[[71, 79, 87]] = 1
    mask = np.array([True, False, True, True, False, True])
    result = input_sensitivity(net, maps, scalars, mask)
    with torch.no_grad():
        logits, value = net(torch.from_numpy(maps)[None], torch.from_numpy(scalars)[None], torch.from_numpy(mask)[None])
    assert np.allclose(result["baseline"], torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    # Batched substitutions can change float32 reduction order relative to a single decision.
    assert abs(result["value"] - float(value[0])) < 3e-5
    assert np.array_equal(maps, original)  # interventions cannot change the match's observation
    assert all(row["probs"][1] == 0 and abs(sum(row["probs"]) - 1) < 1e-5 for row in result["rows"])
    assert len(result["rows"]) == 7
    assert len(result["baseline"]) == net.n_actions
    changed = maps.copy()
    changed[P.N_MAPS:] = changed[:P.N_MAPS]
    with torch.no_grad():
        lg, v = net(torch.from_numpy(changed)[None], torch.from_numpy(scalars)[None], torch.from_numpy(mask)[None])
    history = next(row for row in result["rows"] if row["id"] == "history")
    assert np.allclose(history["probs"], torch.softmax(lg[0], 0).numpy(), atol=1e-5)
    assert abs(history["value_delta"] - (float(v[0]) - result["value"])) < 3e-5
    changed_scalars = scalars.copy()
    changed_scalars[64:88] = 0
    changed_scalars[96:108] = 0
    with torch.no_grad():
        lg, v = net(torch.from_numpy(maps)[None], torch.from_numpy(changed_scalars)[None], torch.from_numpy(mask)[None])
    enemies = next(row for row in result["rows"] if row["id"] == "opponents")
    assert np.allclose(enemies["probs"], torch.softmax(lg[0], 0).numpy(), atol=1e-5)
    assert abs(enemies["value_delta"] - (float(v[0]) - result["value"])) < 3e-5
    assert all(row["probs"][4] == 0 for row in result["rows"])
    own_numbers_removed = np.zeros_like(scalars)
    own_numbers_removed[64:88] = scalars[64:88]
    own_numbers_removed[96:108] = scalars[96:108]
    with torch.no_grad():
        lg, v = net(torch.from_numpy(maps)[None], torch.from_numpy(own_numbers_removed)[None], torch.from_numpy(mask)[None])
    numbers = next(row for row in result["rows"] if row["id"] == "numbers")
    assert np.allclose(numbers["probs"], torch.softmax(lg[0], 0).numpy(), atol=1e-5)
    assert abs(numbers["value_delta"] - (float(v[0]) - result["value"])) < 3e-5


def test_v4_structure_preserves_cross_branch_dependencies_and_safe_kill_target():
    net = make_net(4).eval()
    layers = layer_table(net)
    structure = describe(net, layers, layer_of)
    by_id = {layer["id"]: layer for layer in layers}
    branches = {branch["id"]: branch for branch in structure["branches"]}
    edges = set(map(tuple, structure["edges"]))
    assert list(branches) == ["nearby", "close", "arena", "scalars", "opponents"]
    assert structure["context_stages"] == ["nearby_modulation", "close_modulation", "opponents_query"]
    for group, map_id, index in (("nearby", "local_in", 0), ("close", "close_in", 1)):
        assert branches[group]["inputs"] == [map_id]
        assert branches[group]["stages"] == [f"{group}_{suffix}" for suffix in (
            "c1", "pre", "reduce", "c2", "c3", "c4", "condition", "c5", "c6", "fc")]
        assert ("arena_fc", f"{group}_modulation") in edges
        assert (f"{group}_modulation", f"{group}_condition") in edges
        assert (f"{group}_c4", f"{group}_condition") in edges
        assert (f"{group}_condition", f"{group}_c5") in edges
        assert by_id[f"{group}_c1"]["shape"] == [48, 32, 32]
        assert by_id[f"{group}_pre"]["shape"] == [48, 32, 32]
        assert by_id[f"{group}_reduce"]["shape"] == [64, 16, 16]
        assert by_id[f"{group}_condition"]["shape"] == [64, 16, 16]
        assert by_id[f"{group}_modulation"]["shape"] == [128]
        assert by_id[f"{group}_modulation"]["parameters"] == sum(p.numel() for p in net.conditioning[index].parameters())
    assert branches["arena"]["inputs"] == ["global_in", "territory_in"]
    assert branches["opponents"]["stages"] == ["opponents_h1", "opponents_h2", "opponents_weights", "opponents_pool"]
    assert not set(structure["context_stages"]) & {stage for branch in branches.values() for stage in branch["stages"]}
    for source in ("nearby_fc", "close_fc", "arena_fc", "scalars_h2"):
        assert (source, "opponents_query") in edges
        assert (source, "trunk1") in edges
    assert ("opponents_h2", "opponents_weights") in edges
    assert ("opponents_query", "opponents_weights") in edges
    assert ("opponents_h2", "opponents_pool") in edges
    assert ("opponents_weights", "opponents_pool") in edges
    assert ("opponents_pool", "trunk1") in edges
    assert ("arena_a2", "own") in edges
    assert by_id["opponents_in"]["shape"] == [36]
    expected_indices = [index for enemy in range(3)
                        for index in (*range(64 + 8 * enemy, 72 + 8 * enemy),
                                      *range(96 + 4 * enemy, 100 + 4 * enemy))]
    assert by_id["opponents_in"]["scalar_indices"] == expected_indices
    assert len(by_id["opponents_in"]["names"]) == 36
    assert by_id["opponents_in"]["names"][0] == "enemy 1 ahead"
    assert by_id["opponents_in"]["names"][8] == "enemy 1 brake reserve"
    assert by_id["opponents_in"]["names"][-1] == "enemy 3 right turn wait"
    assert "3 × 12" in by_id["opponents_in"]["operation"]
    assert by_id["scalars_in"]["shape"] == [72]
    assert by_id["scalars_in"]["scalar_indices"] == list(range(64)) + list(range(88, 96))
    assert not any(name.startswith("enemy ") for name in by_id["scalars_in"]["names"])
    assert by_id["opponents_h1"]["shape"] == by_id["opponents_h2"]["shape"] == [192]
    assert by_id["aux"]["shape"] == [3]
    assert "credited kill" in by_id["aux"]["explanation"] and "AND survival" in by_id["aux"]["explanation"]
    assert "tanh" in by_id["value"]["operation"] and "[-1, 1]" in by_id["value"]["explanation"]
    assert by_id["policy"]["shape"] == [6]
    assert by_id["policy"]["names"][-2:] == ["Left + brake", "Right + brake"]


def test_absolute_activation_statistics_keep_negative_values():
    stats = activation_stats([("attention", np.array([-4.0, 0.0, 2.0], np.float32))])["attention"]
    assert stats["signed"] and stats["min"] == -4 and stats["max"] == 2
    assert abs(stats["zero"] - 1 / 3) < 1e-4


def test_input_substitutions_replay_a_saved_memory_without_advancing_it():
    torch.manual_seed(43)
    net = make_net().eval()
    with torch.no_grad():
        net.memory_projection.weight.normal_(0, .08)
        net.memory_projection.bias.normal_(0, .04)
    maps = np.full((2 * P.N_MAPS, P.GRID, P.GRID), 37, np.uint8)
    scalars = np.zeros(P.N_SCALARS, np.float32)
    mask = np.array([True, False, True, True, True, False])
    state = torch.randn(1, 256)
    context = recurrent_context(torch.tensor([4]), torch.tensor([.061]))
    before, controls_before = state.clone(), context.clone()
    result = input_sensitivity(net, maps, scalars, mask, state=state, context=context)
    rows = {row["id"]: row for row in result["rows"]}
    assert len(rows) == 9 and {"memory", "controls"} <= rows.keys()
    with torch.no_grad():
        x, s, m = torch.from_numpy(maps)[None], torch.from_numpy(scalars)[None], torch.from_numpy(mask)[None]
        logits, values, _ = net.forward_step(x, s, m, state, context=context)
        cleared_logits, cleared_value, _ = net.forward_step(x, s, m, net.initial_state(1), context=context)
        no_context_logits, _, _ = net.forward_step(x, s, m, state, context=torch.zeros_like(context))
    assert np.allclose(result["baseline"], torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(result["value"] - values.item()) < 3e-5
    assert np.allclose(rows["memory"]["probs"], torch.softmax(cleared_logits[0], 0).numpy(), atol=1e-5)
    assert abs(rows["memory"]["value_delta"] - (cleared_value.item() - values.item())) < 3e-5
    assert np.allclose(rows["controls"]["probs"], torch.softmax(no_context_logits[0], 0).numpy(), atol=1e-5)
    assert rows["memory"]["policy_shift"] > 1e-6
    assert rows["controls"]["policy_shift"] > 1e-7
    assert torch.equal(state, before) and torch.equal(context, controls_before)
    repeated = input_sensitivity(net, maps, scalars, mask, state=state, context=context)
    assert repeated == result


def test_input_test_rejects_a_running_historical_or_different_weight_decision():
    hub = Hub(1)
    hub.inspection = {"decision": 12, "update": 7}
    with pytest.raises(ValueError, match="Pause"):
        hub.request_probe({"decision": 12, "update": 7})
    hub.control({"paused": True})
    with pytest.raises(ValueError, match="latest"):
        hub.request_probe({"decision": 11, "update": 7})
    with pytest.raises(ValueError, match="checkpoint changed"):
        hub.request_probe({"decision": 12, "update": 6})
    hub.request_probe({"decision": 12, "update": 7})
    assert hub.probe_requested is hub.inspection
