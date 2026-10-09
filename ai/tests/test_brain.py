"""The brain visualizer's view of the network matches the network itself."""

import base64

import numpy as np
import torch

from arma_ai import protocol as P
from arma_ai.brain import edges, layer_table, stacked, trace, weigh
from arma_ai.model import PolicyNet


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
