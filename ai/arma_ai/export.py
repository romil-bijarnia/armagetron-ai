"""Export a checkpoint as a policy file the game loads itself (src/tron/gNeuralNet.cpp).

    uv run arma-export                              # runs/blacklight/latest.pt -> <repo>/brain/policy.bin

The game looks for brain/policy.bin in its data folders (the app bundle ships the one in the
repository; a copy in ~/Library/Application Support/Armagetron Advanced/brain/ wins over it), so
exporting here and rebuilding the app, or dropping the file there, is all it takes to put new
weights into the game.

File format (little-endian):
    char magic[8] = "ARMABRN4"
    u32 grid, local_planes, global_planes, n_scalars, n_actions, update, dtype (16 or 32), n_layers
    u32 n_maps, u32 planes[n_maps], u32 stack_prev, u32 n_towers, per tower u32 k, u32 map_idx[k]
    per layer: u8 kind (1 conv, 2 linear, 3 layer norm, 4 parameter, 5 GRU), u8 name_len, name,
               conv:      u32 cout, cin, k, stride, pad, then weights [cout*cin*k*k] and bias [cout]
               linear:    u32 out, in, then weights [out*in] and bias [out]
               layernorm: u32 n, then weight [n] and bias [n]
               parameter: u32 ndim, u32 dims[ndim], then the values
               GRU:       u32 input, hidden, then weight_ih, weight_hh, bias_ih, bias_hh (r,z,n)
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
from .model import MEMORY_CONTEXT, MEMORY_HIDDEN, MEMORY_VERSION, make_net, net_version

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT.parent


def layers_of(net) -> list[tuple[str, nn.Module]]:
    """Every weight-bearing layer in forward order, with the names the game expects."""
    out: list[tuple[str, nn.Module]] = []
    for i, tower in enumerate(net.towers):
        out += tower.export_layers(f"towers.{i}")
    out += [(f"conditioning.{i}", layer) for i, layer in enumerate(net.conditioning)]
    out += [(f"opponents.{i}", layer) for i, layer in enumerate(net.opponents) if isinstance(layer, nn.Linear)]
    out.append(("opponent_query", net.opponent_query))
    for prefix, seq in (("scalars", net.scalars), ("trunk", net.trunk)):
        for i, m in enumerate(seq):
            if isinstance(m, nn.Linear):
                out.append((f"{prefix}.{i}", m))
    out.append(("pi", net.pi))
    out.append(("v", net.v))
    out.append(("aux", net.aux))
    out += [("memory_version", net.memory_version.reshape(1)),
            ("memory_gru", net.memory_gru), ("memory_projection", net.memory_projection)]
    return out


def export(checkpoint: Path, out: Path, half: bool = True) -> dict:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = make_net(net_version(ck["model"])).eval()
    net.load_state_dict(ck["model"])
    update = int(ck.get("update", 0))
    dtype = np.float16 if half else np.float32
    layers = layers_of(net)
    chunks = [b"ARMABRN4", struct.pack("<8I", P.GRID, P.N_LOCAL_PLANES, P.N_GLOBAL_PLANES, P.N_SCALARS, net.n_actions,
                                 update, 16 if half else 32, len(layers))]
    chunks.append(struct.pack("<I", net.n_maps) + struct.pack(f"<{net.n_maps}I", *P.MAP_PLANES[:net.n_maps])
                  + struct.pack("<I", int(net.stack_prev)))
    # Which maps feed which tower, in the order their planes are concatenated.
    chunks.append(struct.pack("<I", len(net.tower_maps)))
    for maps in net.tower_maps:
        chunks.append(struct.pack("<I", len(maps)) + struct.pack(f"<{len(maps)}I", *maps))
    n_params = 0
    for name, m in layers:
        nm = name.encode()
        if isinstance(m, nn.GRU):
            if m.num_layers != 1 or m.bidirectional or not m.bias:
                raise ValueError("Native memory supports one biased, forward GRU layer")
            chunks.append(struct.pack("<BB", 5, len(nm)) + nm)
            chunks.append(struct.pack("<2I", m.input_size, m.hidden_size))
            for field in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                value = getattr(m, field).detach().numpy()
                n_params += value.size
                chunks.append(np.ascontiguousarray(value, dtype=dtype).tobytes())
            continue
        if isinstance(m, torch.Tensor):  # a bare parameter, such as the attention's position embedding
            w = m.detach().numpy()
            n_params += w.size if name != "memory_version" else 0
            chunks.append(struct.pack("<BB", 4, len(nm)) + nm)
            chunks.append(struct.pack("<I", w.ndim) + struct.pack(f"<{w.ndim}I", *w.shape))
            chunks.append(np.ascontiguousarray(w, dtype=dtype).tobytes())
            continue
        w = m.weight.detach().numpy()
        b = m.bias.detach().numpy()
        n_params += w.size + b.size
        if isinstance(m, nn.Conv2d):
            assert m.kernel_size[0] == m.kernel_size[1] and m.stride[0] == m.stride[1] and m.padding[0] == m.padding[1]
            chunks.append(struct.pack("<BB", 1, len(nm)) + nm)
            chunks.append(struct.pack("<5I", w.shape[0], w.shape[1], m.kernel_size[0], m.stride[0], m.padding[0]))
        elif isinstance(m, nn.LayerNorm):
            chunks.append(struct.pack("<BB", 3, len(nm)) + nm)
            chunks.append(struct.pack("<I", w.shape[0]))
        else:
            chunks.append(struct.pack("<BB", 2, len(nm)) + nm)
            chunks.append(struct.pack("<2I", w.shape[0], w.shape[1]))
        chunks.append(np.ascontiguousarray(w, dtype=dtype).tobytes())
        chunks.append(np.ascontiguousarray(b, dtype=dtype).tobytes())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(b"".join(chunks))
    tmp.replace(out)
    return {"update": update, "layers": len(layers), "parameters": n_params, "bytes": out.stat().st_size,
            "memory_version": MEMORY_VERSION, "memory_hidden": MEMORY_HIDDEN, "memory_context": MEMORY_CONTEXT}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/blacklight/latest.pt")
    ap.add_argument("--out", type=Path, default=REPO / "brain/policy.bin")
    ap.add_argument("--float32", action="store_true", help="store full precision (twice the size)")
    args = ap.parse_args()
    info = export(args.checkpoint, args.out, half=not args.float32)
    print(f"wrote {args.out} (update {info['update']}, {info['parameters'] / 1e6:.2f} M parameters, "
          f"{info['bytes'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
