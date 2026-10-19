"""The game's C++ policy network agrees with PyTorch on the exported weights."""

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from arma_ai import protocol as P
from arma_ai.export import export
from arma_ai.model import make_net

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


@pytest.fixture(scope="module")
def cpp_brain(tmp_path_factory):
    cxx = shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("cpp-brain") / "brain_check"
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(exe), str(HERE / "cpp/brain_check.cpp"),
                    str(REPO / "src/tron/gNeuralNet.cpp")], check=True)
    return exe


@pytest.fixture(scope="module")
def cpp_neural_matmul(tmp_path_factory):
    cxx = shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("cpp-neural-matmul") / "matmul_check"
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(exe),
                    str(HERE / "cpp/neural_matmul_check.cpp")], check=True)
    return exe


@pytest.mark.parametrize("shape,transpose_a,transpose_b", [
    ((3, 5, 7), False, False), ((3, 5, 7), True, False),
    ((3, 5, 7), False, True), ((3, 5, 7), True, True),
    ((256, 256, 32), True, False),  # native attention scores: Q^T K
    ((32, 256, 256), False, True),  # native attention values: V scores^T
])
def test_native_matmul_transposes_and_portable_fallback(cpp_neural_matmul, shape, transpose_a, transpose_b):
    m, n, k = shape
    rng = np.random.default_rng(21)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    # A double-precision NumPy product supplies the oracle; storage transposes must not
    # change that product. Nonsquare operands catch incorrect leading dimensions.
    expected = a.astype(np.float64) @ b.astype(np.float64)
    stored_a = np.ascontiguousarray(a.T if transpose_a else a)
    stored_b = np.ascontiguousarray(b.T if transpose_b else b)
    payload = struct.pack("<5i", m, n, k, transpose_a, transpose_b) + stored_a.tobytes() + stored_b.tobytes()
    run = subprocess.run([str(cpp_neural_matmul)], input=payload, capture_output=True, check=True)
    portable, selected = np.frombuffer(run.stdout, np.float32).reshape(2, m, n)
    np.testing.assert_allclose(portable, expected, rtol=2e-5, atol=8e-5)
    np.testing.assert_allclose(selected, expected, rtol=2e-5, atol=8e-5)


@pytest.mark.parametrize("half", [False, True], ids=["float32", "float16"])
def test_cpp_matches_pytorch(tmp_path, cpp_brain, half):
    torch.manual_seed(1)
    net = make_net().eval()
    # Initial conditioning is identity. Nonzero weights ensure parity exercises actual FiLM.
    with torch.no_grad():
        for layer in net.conditioning:
            layer.weight.normal_(0, 0.05)
            layer.bias.normal_(0, 0.1)
        # Migrated production weights start these columns at zero; exercise the full new
        # routing with nonzero columns so omitted/misordered opponent state fails parity.
        net.opponents[0].weight[:, 8:].normal_(0, 0.1)
        net.pi.weight.mul_(25)
        net.memory_projection.weight.normal_(0, .025)
        net.memory_projection.bias.normal_(0, .01)
        net.memory_gru.weight_ih_l0[:, 768:].normal_(0, .1)
    ck = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "update": 7}, ck)
    policy = tmp_path / "policy.bin"
    info = export(ck, policy, half=half)
    # the game's file carries every layer but the ownership head, which only the teacher uses
    assert info["update"] == 7
    assert info["parameters"] == sum(p.numel() for name, p in net.named_parameters() if not name.startswith("own"))

    assert policy.read_bytes()[:8] == b"ARMABRN4"
    assert struct.unpack_from("<I", policy.read_bytes(), 24)[0] == net.n_actions == 6
    assert struct.unpack_from("<I", policy.read_bytes(), 20)[0] == P.N_SCALARS == 108

    rng = np.random.default_rng(3)
    n = 12
    maps = rng.integers(0, 256, (n, P.N_MAPS, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal((n, P.N_SCALARS)).astype(np.float32)
    scalars[:, 96:108] = rng.random((n, 12), dtype=np.float32)
    # Empty, singleton, partial and full enemy sets, including the strict presence threshold.
    scalars[:, [71, 79, 87]] = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1],
        [0, 0, 0], [0, 1, 1], [1, 1, 0], [0.5, 0.5, 0.5], [0.5001, 0, 0], [1, 1, 1],
    ], dtype=np.float32)
    masks = np.array([63, 16, 32, 48, 47, 55, 1, 2, 4, 8, 17, 34], dtype=np.uint8)
    blob = [struct.pack("<I", n)]
    for i in range(n):
        blob += [maps[i].tobytes(), scalars[i].tobytes(), bytes([masks[i]])]
    inputs = tmp_path / "inputs.bin"
    inputs.write_bytes(b"".join(blob))

    reset_at = 6
    out = subprocess.run([str(cpp_brain), str(policy), str(inputs), str(reset_at)],
                         check=True, capture_output=True, text=True)
    rows = [list(map(float, line.split())) for line in out.stdout.strip().splitlines()]
    assert len(rows) == n

    with torch.no_grad():
        mask_t = torch.tensor([[(m >> a) & 1 for a in range(net.n_actions)] for m in masks], dtype=torch.bool)
        # The C++ net stacks each decision's maps with the previous decision's (the first with itself).
        prev = np.concatenate([maps[:1], maps[:-1]])
        prev[reset_at] = maps[reset_at]  # A new life discards the previous life's observation.
        x = np.concatenate([maps, prev], 1)
        state = net.initial_state(1)
        probabilities, values = [], []
        for i in range(n):
            if i == reset_at:
                state = net.initial_state(1)
            logits, v, state = net.forward_step(torch.from_numpy(x[i:i + 1]),
                                                 torch.from_numpy(scalars[i:i + 1]), mask_t[i:i + 1], state)
            probabilities.append(torch.softmax(logits, 1)[0])
            values.append(v[0])
        probs, value = torch.stack(probabilities).numpy(), torch.stack(values)
    for i, row in enumerate(rows):
        assert len(row) == net.n_actions + 2
        action, p, v = int(row[0]), np.array(row[1:1 + net.n_actions]), row[-1]
        prob_tol, value_tol = (2e-3, 2e-2) if half else (2e-5, 2e-4)
        assert np.allclose(p, probs[i], atol=prob_tol), (i, p, probs[i])
        assert abs(v - float(value[i])) < value_tol, (i, v, float(value[i]))
        assert action == int(probs[i].argmax()) or probs[i][action] >= probs[i].max() - prob_tol
        assert (masks[i] >> action) & 1
        assert abs(p.sum() - 1) < 1e-6
        assert all(p[a] == 0 for a in range(net.n_actions) if not (masks[i] >> a) & 1)
        assert -1 <= v <= 1


@pytest.mark.parametrize("half", [False, True], ids=["float32", "float16"])
def test_cpp_long_memory_context_and_hidden_state_match_python(tmp_path, cpp_brain, half):
    """Nonzero context/gates/projection must agree over more than a training unroll."""
    torch.manual_seed(39)
    net = make_net().eval()
    with torch.no_grad():
        net.memory_projection.weight.normal_(0, .025)
        net.memory_projection.bias.normal_(0, .01)
        net.memory_gru.weight_ih_l0[:, 768:].normal_(0, .12)
        net.memory_gru.weight_hh_l0.mul_(1.5)
        net.pi.weight.mul_(8)
    ck = tmp_path / "memory.pt"
    torch.save({"model": net.state_dict(), "update": 19}, ck)
    policy = tmp_path / "memory.bin"
    export(ck, policy, half=half)
    if half:
        # Native half files store quantized weights but compute in float32. Match
        # that oracle so long recurrent trajectories test arithmetic, not the
        # separate effect of quantizing every visual and recurrent tensor.
        net.load_state_dict({name: value.half().float() if value.is_floating_point() else value
                             for name, value in net.state_dict().items()})
    rng = np.random.default_rng(41)
    n, reset_at = 64, 37
    maps = rng.integers(0, 256, (n, 4, 64, 64), dtype=np.uint8)
    scalars = rng.standard_normal((n, 108)).astype(np.float32)
    scalars[:, [71, 79, 87]] = 1
    masks = np.full(n, 63, np.uint8)
    from arma_ai.model import recurrent_context
    actions = torch.as_tensor(rng.integers(0, 6, n))
    elapsed = torch.as_tensor(rng.uniform(.025, .15, n).astype(np.float32))
    actions[[0, reset_at]], elapsed[[0, reset_at]] = -1, 0
    contexts = recurrent_context(actions, elapsed).numpy()
    chunks = [struct.pack("<I", n)]
    for i in range(n):
        chunks += [maps[i].tobytes(), scalars[i].tobytes(), bytes([masks[i]]), contexts[i].tobytes()]
    inputs = tmp_path / "memory-input.bin"
    inputs.write_bytes(b"".join(chunks))
    result = subprocess.run([str(cpp_brain), str(policy), str(inputs), str(reset_at), "memory"],
                            check=True, capture_output=True, text=True)
    native = np.asarray([list(map(float, row.split())) for row in result.stdout.splitlines()])
    assert native.shape == (n, 264)
    history, state, expected = None, net.initial_state(1), []
    with torch.no_grad():
        for i in range(n):
            if i == reset_at:
                history, state = None, net.initial_state(1)
            pair = np.concatenate([maps[i], maps[i] if history is None else history])[None]
            logits, value, state = net.forward_step(torch.from_numpy(pair), torch.from_numpy(scalars[i:i + 1]),
                                                      torch.ones(1, 6, dtype=torch.bool), state,
                                                      context=torch.from_numpy(contexts[i:i + 1]))
            expected.append(np.concatenate([torch.softmax(logits, 1)[0].numpy(), value.numpy(), state[0].numpy()]))
            history = maps[i]
    expected = np.stack(expected)
    tolerance = 2e-4
    np.testing.assert_allclose(native[:, 1:], expected, atol=tolerance, rtol=0)
    assert np.max(np.abs(native[:, 8:])) > .1  # parity really exercised persistent hidden state
    np.testing.assert_allclose(native[:, 1:7].sum(1), 1, atol=2e-6)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_cpp_rejects_legacy_policy_header(tmp_path, cpp_brain, version):
    policy = tmp_path / "legacy.bin"
    # The version alone is enough to reject an obsolete file, before any legacy parsing.
    policy.write_bytes(f"ARMABRN{version}".encode())
    result = subprocess.run([str(cpp_brain), str(policy), str(tmp_path / "unused.bin")],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "ARMABRN4 required" in result.stderr


@pytest.mark.parametrize("version", [None, 1, 2, 3])
def test_export_rejects_legacy_checkpoint(tmp_path, version):
    state = {} if version is None else {"net_version": torch.tensor(version)}
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": state}, checkpoint)
    policy = tmp_path / "policy.bin"
    policy.write_bytes(b"existing policy")
    with pytest.raises(ValueError, match="Only V4 checkpoints are supported"):
        export(checkpoint, policy)
    assert policy.read_bytes() == b"existing policy"


def _layer_offsets(blob):
    """Locate records so loader tests can damage a valid export without changing unrelated bytes."""
    _, _, _, _, _, _, dtype, n_layers = struct.unpack_from("<8I", blob, 8)
    n_maps, = struct.unpack_from("<I", blob, 40)
    offset = 44 + 4 * n_maps + 4
    n_towers, = struct.unpack_from("<I", blob, offset)
    offset += 4
    for _ in range(n_towers):
        count, = struct.unpack_from("<I", blob, offset)
        offset += 4 + 4 * count
    layers = {}
    for _ in range(n_layers):
        start = offset
        kind, name_len = struct.unpack_from("<BB", blob, offset)
        offset += 2
        name = blob[offset:offset + name_len].decode()
        offset += name_len
        metadata = offset
        if kind == 1:
            cout, cin, kernel, _, _ = struct.unpack_from("<5I", blob, offset)
            offset += 20 + (cout * cin * kernel * kernel + cout) * (dtype // 8)
        elif kind == 2:
            out, inp = struct.unpack_from("<2I", blob, offset)
            offset += 8 + (out * inp + out) * (dtype // 8)
        elif kind == 3:
            width, = struct.unpack_from("<I", blob, offset)
            offset += 4 + 2 * width * (dtype // 8)
        elif kind == 5:
            inputs, hidden = struct.unpack_from("<2I", blob, offset)
            offset += 8 + (3 * hidden * inputs + 3 * hidden * hidden + 6 * hidden) * (dtype // 8)
        else:
            assert kind == 4
            ndim, = struct.unpack_from("<I", blob, offset)
            dims = struct.unpack_from(f"<{ndim}I", blob, offset + 4)
            offset += 4 + 4 * ndim + int(np.prod(dims)) * (dtype // 8)
        layers[name] = start, offset, metadata
    assert offset == len(blob)
    return layers


@pytest.fixture(scope="module")
def v4_export(tmp_path_factory):
    root = tmp_path_factory.mktemp("v4-export")
    ck = root / "ck.pt"
    torch.save({"model": make_net().state_dict()}, ck)
    out = root / "policy.bin"
    export(ck, out)
    blob = out.read_bytes()
    return blob, _layer_offsets(blob)


@pytest.mark.parametrize("layer", [
    "towers.0.pre.c0", "towers.1.reduce", "towers.2.attn.1.ln2", "conditioning.0",
    "opponents.2", "opponent_query", "aux", "memory_version", "memory_gru", "memory_projection",
])
def test_cpp_rejects_missing_v4_stage(tmp_path, cpp_brain, v4_export, layer):
    blob, layers = v4_export
    start, end, _ = layers[layer]
    broken = bytearray(blob[:start] + blob[end:])
    count, = struct.unpack_from("<I", broken, 36)
    struct.pack_into("<I", broken, 36, count - 1)
    policy = tmp_path / "missing.bin"
    policy.write_bytes(broken)
    inputs = tmp_path / "empty.bin"
    inputs.write_bytes(struct.pack("<I", 0))
    result = subprocess.run([str(cpp_brain), str(policy), str(inputs)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "load failed:" in result.stderr


@pytest.mark.parametrize("damage", ["version", "actions", "old_observation", "stem_stride", "position_shape"])
def test_cpp_rejects_invalid_v4_contract(tmp_path, cpp_brain, v4_export, damage):
    blob, layers = v4_export
    broken = bytearray(blob)
    if damage == "version":
        broken[7] = ord("5")
    elif damage == "actions":
        struct.pack_into("<I", broken, 24, 4)
    elif damage == "old_observation":
        struct.pack_into("<I", broken, 20, 96)
    elif damage == "stem_stride":
        struct.pack_into("<I", broken, layers["towers.0.stem"][2] + 12, 1)
    else:
        # Same number of values, wrong tensor shape: a size-only check would miss this.
        struct.pack_into("<3I", broken, layers["towers.2.pos"][2] + 4, 1, 128, 256)
    policy = tmp_path / "invalid.bin"
    policy.write_bytes(broken)
    inputs = tmp_path / "empty.bin"
    inputs.write_bytes(struct.pack("<I", 0))
    result = subprocess.run([str(cpp_brain), str(policy), str(inputs)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "load failed:" in result.stderr
    if damage == "old_observation":
        assert "108 scalars" in result.stderr


def test_cpp_rejects_old_eight_value_opponent_encoder(tmp_path, cpp_brain, v4_export):
    blob, layers = v4_export
    start, end, metadata = layers["opponents.0"]
    width, inputs = struct.unpack_from("<2I", blob, metadata)
    assert inputs == 12
    dtype = np.dtype("<f2")  # The module fixture exports float16.
    weights = np.frombuffer(blob, dtype, width * inputs, metadata + 8).reshape(width, inputs)
    bias = blob[metadata + 8 + width * inputs * dtype.itemsize:end]
    old_layer = blob[start:metadata] + struct.pack("<2I", width, 8) + weights[:, :8].tobytes() + bias
    policy = tmp_path / "old-opponents.bin"
    policy.write_bytes(blob[:start] + old_layer + blob[end:])
    result = subprocess.run([str(cpp_brain), str(policy), str(tmp_path / "unused.bin")],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "layers are incomplete or have incompatible dimensions" in result.stderr
