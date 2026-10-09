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
from arma_ai.model import PolicyNet, PolicyNetV1

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


@pytest.mark.skipif(shutil.which("clang++") is None and shutil.which("g++") is None, reason="no C++ compiler")
@pytest.mark.parametrize("version", [1, 2])
def test_cpp_matches_pytorch(tmp_path, version):
    torch.manual_seed(1)
    net = (PolicyNet() if version == 2 else PolicyNetV1()).eval()
    n_maps, stack = P.N_MAPS, net.stack_prev  # the game always hands over all four maps
    ck = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "update": 7}, ck)
    policy = tmp_path / "policy.bin"
    info = export(ck, policy)
    assert info["update"] == 7 and info["parameters"] == sum(p.numel() for p in net.parameters())

    cxx = shutil.which("clang++") or shutil.which("g++")
    exe = tmp_path / "brain_check"
    subprocess.run([cxx, "-O2", "-std=c++17", "-o", str(exe), str(HERE / "cpp/brain_check.cpp"),
                    str(REPO / "src/tron/gNeuralNet.cpp")], check=True)

    rng = np.random.default_rng(3)
    n = 12
    maps = rng.integers(0, 256, (n, n_maps, P.GRID, P.GRID), dtype=np.uint8)
    scalars = rng.standard_normal((n, P.N_SCALARS)).astype(np.float32)
    masks = rng.integers(1, 1 << P.N_ACTIONS, n).astype(np.uint8)
    masks[0] = 0b1111
    blob = [struct.pack("<I", n)]
    for i in range(n):
        blob += [maps[i].tobytes(), scalars[i].tobytes(), bytes([masks[i]])]
    inputs = tmp_path / "inputs.bin"
    inputs.write_bytes(b"".join(blob))

    out = subprocess.run([str(exe), str(policy), str(inputs)], check=True, capture_output=True, text=True)
    rows = [list(map(float, line.split())) for line in out.stdout.strip().splitlines()]
    assert len(rows) == n

    with torch.no_grad():
        mask_t = torch.tensor([[(m >> a) & 1 for a in range(P.N_ACTIONS)] for m in masks], dtype=torch.bool)
        if stack:
            # the C++ net stacks each decision's maps with the previous decision's (the first with itself)
            prev = np.concatenate([maps[:1], maps[:-1]])
            x = np.concatenate([maps, prev], 1)
        else:
            x = maps
        logits, value = net(torch.from_numpy(x), torch.from_numpy(scalars), mask_t)
        probs = torch.softmax(logits, 1).numpy()
    for i, row in enumerate(rows):
        action, p, v = int(row[0]), np.array(row[1:1 + P.N_ACTIONS]), row[-1]
        # float16 weights: agree to about three decimals
        assert np.allclose(p, probs[i], atol=2e-3), (i, p, probs[i])
        assert abs(v - float(value[i])) < 2e-2, (i, v, float(value[i]))
        assert action == int(probs[i].argmax()) or probs[i][action] >= probs[i].max() - 2e-3
        assert (masks[i] >> action) & 1
