"""Export a checkpoint as a policy file the game loads itself (src/tron/gNeuralNet.cpp).

    uv run arma-export                              # runs/main/latest.pt -> <repo>/brain/policy.bin
    uv run arma-export --checkpoint runs/main/pool/u000200.pt --out /tmp/old.bin

The game looks for brain/policy.bin in its data folders (the app bundle ships the one in the
repository; a copy in ~/Library/Application Support/Armagetron Advanced/brain/ wins over it), so
exporting here and rebuilding the app, or dropping the file there, is all it takes to put new
weights into the game.

File format (little-endian):
    char magic[8] = "ARMABRN1"
    u32 grid, local_planes, global_planes, n_scalars, n_actions, update, dtype (16 or 32), n_layers
    per layer: u8 kind (1 conv, 2 linear), u8 name_len, name,
               conv:   u32 cout, cin, k, stride, pad, then weights [cout*cin*k*k] and bias [cout]
               linear: u32 out, in, then weights [out*in] and bias [out]
    weights and biases are float16 (dtype 16) or float32 (dtype 32)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import protocol as P
from .model import PolicyNet

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT.parent


def layers_of(net: PolicyNet) -> list[tuple[str, nn.Module]]:
    """Every weight-bearing layer in forward order, with the names the game expects."""
    out: list[tuple[str, nn.Module]] = []
    for prefix, tower in (("local", net.local), ("globl", net.globl)):
        for i, m in enumerate(tower.net):
            if isinstance(m, nn.Conv2d):
                out.append((f"{prefix}.net.{i}", m))
        out.append((f"{prefix}.fc", tower.fc))
    for prefix, seq in (("scalars", net.scalars), ("trunk", net.trunk)):
        for i, m in enumerate(seq):
            if isinstance(m, nn.Linear):
                out.append((f"{prefix}.{i}", m))
    out.append(("pi", net.pi))
    out.append(("v", net.v))
    return out


def export(checkpoint: Path, out: Path, half: bool = True) -> dict:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = PolicyNet().eval()
    net.load_state_dict(ck["model"])
    update = int(ck.get("update", 0))
    dtype = np.float16 if half else np.float32
    layers = layers_of(net)
    chunks = [b"ARMABRN1", struct.pack("<8I", P.GRID, P.N_LOCAL_PLANES, P.N_GLOBAL_PLANES, P.N_SCALARS, P.N_ACTIONS,
                                       update, 16 if half else 32, len(layers))]
    n_params = 0
    for name, m in layers:
        nm = name.encode()
        w = m.weight.detach().numpy()
        b = m.bias.detach().numpy()
        n_params += w.size + b.size
        if isinstance(m, nn.Conv2d):
            assert m.kernel_size[0] == m.kernel_size[1] and m.stride[0] == m.stride[1] and m.padding[0] == m.padding[1]
            chunks.append(struct.pack("<BB", 1, len(nm)) + nm)
            chunks.append(struct.pack("<5I", w.shape[0], w.shape[1], m.kernel_size[0], m.stride[0], m.padding[0]))
        else:
            chunks.append(struct.pack("<BB", 2, len(nm)) + nm)
            chunks.append(struct.pack("<2I", w.shape[0], w.shape[1]))
        chunks.append(np.ascontiguousarray(w, dtype=dtype).tobytes())
        chunks.append(np.ascontiguousarray(b, dtype=dtype).tobytes())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(b"".join(chunks))
    tmp.replace(out)
    return {"update": update, "layers": len(layers), "parameters": n_params, "bytes": out.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/main/latest.pt")
    ap.add_argument("--out", type=Path, default=REPO / "brain/policy.bin")
    ap.add_argument("--float32", action="store_true", help="store full precision (twice the size)")
    args = ap.parse_args()
    info = export(args.checkpoint, args.out, half=not args.float32)
    print(f"wrote {args.out} (update {info['update']}, {info['parameters'] / 1e6:.2f} M parameters, "
          f"{info['bytes'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
