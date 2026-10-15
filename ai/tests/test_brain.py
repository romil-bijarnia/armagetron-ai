"""The brain page's view of the network matches the network itself."""

import base64

import numpy as np
import torch

from arma_ai import protocol as P
from arma_ai.brain import GLOBAL_PLANES, attention_payload, layer_of, layer_table, learned, stacked, think, trace
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
    names = [L["name"] for L in table]
    assert names[:3] == ["player view · stem", "player view · block 1", "player view · block 2"]
    assert "player view · down to 8×8" in names and "arena view · summary" in names


def test_the_heads_the_page_paints_are_the_networks_own():
    """Ownership, the aux predictions and the attention rows come from the same pass as the move."""
    torch.manual_seed(0)
    net = PolicyNetV3().eval()
    rng = np.random.default_rng(3)
    maps = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    prev = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.array([True, False, True, True])
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
        direct = torch.softmax(net.own(net.towers[1].features(net.tower_input(x, 1)))[0], 0).numpy()
    assert np.allclose(t.own, direct, atol=1e-5)
    assert own.shape == (1, 3, P.GRID, P.GRID)
    # one attention tensor per arena attention layer; every row is a distribution over the cells
    assert len(t.attention) == len(net.towers[1].attn) == 2
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
    for net in (PolicyNet().eval(), PolicyNetV3().eval()):
        ids = {L["id"] for L in layer_table(net)} | {"aux", "own"}
        for name, w in net.state_dict().items():
            if w.ndim >= 2:
                assert layer_of(name, net) in ids, name
        old = {k: v.clone() for k, v in net.state_dict().items()}
        new = {k: v.clone() for k, v in old.items()}
        new["towers.0.res.1.c1.weight"] += 0.1  # a big move relative to the weights' size
        new["pi.weight"] *= 1.001  # a small one
        moved = learned(net, new, old)
        assert moved["player_c3"] == 1.0 and 0 < moved["policy"] < 1 and "arena_c1" not in moved
        assert learned(net, old, old) == {}


def test_the_v1_brain_is_drawn_too():
    torch.manual_seed(0)
    net = PolicyNetV1().eval()
    rng = np.random.default_rng(2)
    maps = rng.integers(0, 256, (P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal(P.N_SCALARS).astype(np.float32)
    mask = np.ones(P.N_ACTIONS, bool)
    t = think(net, stacked(maps, None), scalars, mask)
    layers = layer_table(net)
    assert [L["id"] for L in layers if L["kind"] == "input"] == ["local_in", "global_in", "scalars_in"]
    table = [L for L in layers if L["kind"] in ("conv", "dense")]
    assert [k for k, _ in t.layers] == [L["id"] for L in table]
    with torch.no_grad():
        logits, v = net(torch.from_numpy(maps)[None], torch.from_numpy(scalars)[None], torch.from_numpy(mask)[None])
    assert np.allclose(t.probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(t.value - float(v[0])) < 1e-4
    assert t.own is None and t.aux is None and t.attention == []  # v1 has none of the newer heads
    assert layer_of("local.net.0.weight", net) == "local_c1" and layer_of("globl.fc.weight", net) == "global_fc"


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
    # the arena tower's two attention layers are stages of their own, the player tower has none
    ids = [L["id"] for L in layers]
    assert "arena_a1" in ids and "arena_a2" in ids and not any(i.startswith("player_a") for i in ids)
    assert next(L for L in layers if L["id"] == "arena_a2")["name"] == "arena view · attention 2"
    with torch.no_grad():
        logits, v = net(torch.from_numpy(stacked(maps, prev))[None], torch.from_numpy(scalars)[None],
                        torch.from_numpy(mask)[None])
    assert np.allclose(probs, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert abs(value - float(v[0])) < 1e-4
    assert probs[1] < 1e-6
    n_blocks = len(net.towers[1].res)
    assert layer_of("towers.1.attn.1.qkv.weight", net) == "arena_a2"
    assert layer_of("towers.1.down0.weight", net) == f"arena_c{n_blocks + 2}"
    assert layer_of("towers.1.down1.weight", net) == f"arena_c{n_blocks + 3}"
    assert f"arena_c{n_blocks + 3}" in ids
