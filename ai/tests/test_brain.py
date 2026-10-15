"""The brain visualizer's view of the network matches the network itself."""

import base64

import numpy as np
import torch

from arma_ai import protocol as P
from arma_ai.brain import edges, layer_table, stacked, trace, weigh
from arma_ai.model import PolicyNet, PolicyNetV1, PolicyNetV3


def _units(L):
    return int(np.prod(L["shape"]))


def test_trace_layers_match_the_table_and_the_real_forward_pass():
    torch.manual_seed(0)
    net = PolicyNet().eval()
    rng = np.random.default_rng(0)
    maps = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    prev = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.array([True, True, False, True])
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


def test_edges_point_at_real_units_and_weights():
    net = PolicyNet().eval()
    layers = layer_table(net)
    by_id = {L["id"]: L for L in layers}
    groups = edges(net, layers, np.random.default_rng(1))
    state = net.state_dict()
    assert {g["dst"] for g in groups} >= {"player_c1", "arena_c1", "player_fc", "arena_fc", "trunk1",
                                          "trunk2", "policy", "value"}
    # every input map feeds its tower's first layer
    assert {(g["src"], g["dst"]) for g in groups} >= {("local_in", "player_c1"), ("close_in", "player_c1"),
                                                      ("global_in", "arena_c1"), ("territory_in", "arena_c1")}
    for g in groups:
        assert g["si"].min() >= 0 and g["si"].max() < _units(by_id[g["src"]])
        assert g["di"].min() >= 0 and g["di"].max() < _units(by_id[g["dst"]])
        assert g["pidx"].max() < state[g["param"]].numel()
    for item in weigh(groups, state, state):
        n = len(base64.b64decode(item["si"])) // 4
        assert len(base64.b64decode(item["w"])) == n == len(base64.b64decode(item["dw"]))


def test_the_v1_brain_is_drawn_too():
    torch.manual_seed(0)
    net = PolicyNetV1().eval()
    rng = np.random.default_rng(2)
    maps = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.ones(P.N_ACTIONS, bool)
    outs, probs, value = trace(net, stacked(maps, None), scalars, mask)
    layers = layer_table(net)
    assert [L["id"] for L in layers if L["kind"] == "input"] == ["local_in", "global_in", "scalars_in"]
    table = [L for L in layers if L["kind"] in ("conv", "dense")]
    assert [k for k, _ in outs] == [L["id"] for L in table]
    with torch.no_grad():
        logits, v = net(torch.from_numpy(maps)[None], torch.from_numpy(scalars)[None], torch.from_numpy(mask)[None])
    assert np.allclose(probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(value - float(v[0])) < 1e-4
    by_id = {L["id"]: L for L in layers}
    state = net.state_dict()
    groups = edges(net, layers, np.random.default_rng(1))
    assert {(g["src"], g["dst"]) for g in groups} >= {("local_in", "local_c1"), ("global_in", "global_c1")}
    for g in groups:
        assert g["si"].max() < _units(by_id[g["src"]]) and g["di"].max() < _units(by_id[g["dst"]])
        assert g["pidx"].max() < state[g["param"]].numel()


def test_the_v3_brain_is_drawn_with_its_attention_layers():
    torch.manual_seed(0)
    net = PolicyNetV3().eval()
    rng = np.random.default_rng(3)
    maps = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    prev = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.array([True, False, True, True])
    outs, probs, value = trace(net, stacked(maps, prev), scalars, mask)
    layers = layer_table(net)
    table = [L for L in layers if L["kind"] in ("conv", "dense")]
    assert [k for k, _ in outs] == [L["id"] for L in table]
    assert all(list(v.shape) == L["shape"] for (_, v), L in zip(outs, table))
    # the arena tower's two attention layers are slabs of their own, the player tower has none
    ids = [L["id"] for L in layers]
    assert "arena_a1" in ids and "arena_a2" in ids and not any(i.startswith("player_a") for i in ids)
    assert next(L for L in layers if L["id"] == "arena_a2")["name"] == "arena view · attention 2"
    with torch.no_grad():
        logits, v = net(torch.from_numpy(stacked(maps, prev))[None], torch.from_numpy(scalars)[None],
                        torch.from_numpy(mask)[None])
    assert np.allclose(probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(value - float(v[0])) < 1e-4
    assert probs[1] < 1e-6
    by_id = {L["id"]: L for L in layers}
    groups = edges(net, layers, np.random.default_rng(1))
    state = net.state_dict()
    pairs = {(g["src"], g["dst"]) for g in groups}
    n_blocks = len(net.towers[1].res)
    assert {(f"arena_c{n_blocks + 1}", "arena_a1"), ("arena_a1", "arena_a2"), ("arena_a2", f"arena_c{n_blocks + 2}")} <= pairs
    for g in groups:
        assert g["si"].min() >= 0 and g["si"].max() < _units(by_id[g["src"]])
        assert g["di"].min() >= 0 and g["di"].max() < _units(by_id[g["dst"]])
        assert g["pidx"].max() < state[g["param"]].numel()
    for item in weigh(groups, state, state):
        n = len(base64.b64decode(item["si"])) // 4
        assert len(base64.b64decode(item["w"])) == n == len(base64.b64decode(item["dw"]))
