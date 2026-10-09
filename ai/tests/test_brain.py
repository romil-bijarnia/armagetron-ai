"""The brain visualizer's view of the network matches the network itself."""

import base64

import numpy as np
import torch

from arma_ai import protocol as P
from arma_ai.brain import edges, layer_table, trace, weigh
from arma_ai.model import PolicyNet


def _units(L):
    return int(np.prod(L["shape"]))


def test_trace_layers_match_the_table_and_the_real_forward_pass():
    torch.manual_seed(0)
    net = PolicyNet().eval()
    rng = np.random.default_rng(0)
    local = rng.integers(0, 256, (P.GRID, P.GRID), dtype=np.uint8)
    globl = rng.integers(0, 256, (P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.array([True, True, False, True])
    outs, probs, value = trace(net, local, globl, scalars, mask)
    table = [L for L in layer_table(net) if L["kind"] in ("conv", "dense")]
    assert [k for k, _ in outs] == [L["id"] for L in table]
    assert all(list(v.shape) == L["shape"] for (_, v), L in zip(outs, table))
    with torch.no_grad():
        logits, v = net(torch.from_numpy(local)[None], torch.from_numpy(globl)[None],
                        torch.from_numpy(scalars)[None], torch.from_numpy(mask)[None])
    assert np.allclose(probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(value - float(v[0])) < 1e-4
    assert probs[2] < 1e-6


def test_edges_point_at_real_units_and_weights():
    net = PolicyNet().eval()
    layers = layer_table(net)
    by_id = {L["id"]: L for L in layers}
    groups = edges(net, layers, np.random.default_rng(1))
    state = net.state_dict()
    assert {g["dst"] for g in groups} >= {"local_c1", "global_c1", "local_fc", "trunk1", "trunk2", "policy", "value"}
    for g in groups:
        assert g["si"].min() >= 0 and g["si"].max() < _units(by_id[g["src"]])
        assert g["di"].min() >= 0 and g["di"].max() < _units(by_id[g["dst"]])
        assert g["pidx"].max() < state[g["param"]].numel()
    for item in weigh(groups, state, state):
        n = len(base64.b64decode(item["si"])) // 4
        assert len(base64.b64decode(item["w"])) == n == len(base64.b64decode(item["dw"]))
