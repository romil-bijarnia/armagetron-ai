"""Record one isolated search for inspection, without writing experience or model weights.

    python -m arma_ai.search_trace --output /path/to/search-trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import protocol as P
from .sim import SelfPlay
from .teach import PROJECT, _load_net


@torch.no_grad()
def record(checkpoint: Path, *, simulations: int = 24, seed: int = 73) -> dict:
    if simulations < 1:
        raise ValueError("simulations must be positive")
    torch.set_num_threads(2)
    net, ck = _load_net(checkpoint, torch.device("cpu"))
    sp = SelfPlay(1, sims=simulations, macro=2, search_prob=1, temperature=1,
                  threads=1, gamma=.997, recurrent=True)
    # A staged opening leaves all six controls available at the inspected decision.
    # Both players then use the checkpoint for every hypothetical continuation.
    sp.start(0, -3, [0, 0], [0, 0], seed=seed,
             placements=[(25, 35, 1, 0), (40, 40, -1, 0)])
    frames = {}
    opening = True
    for _ in range(simulations * 8 + 16):
        n = sp.collect()
        tree = sp.search_tree()
        if tree is not None:
            frames[tree["simulations"]] = tree
            if not tree["searching"]:
                break
        if not n:
            raise RuntimeError("isolated search ended before a complete tree was captured")
        logits = np.zeros((n, P.N_ACTIONS), np.float32)
        values, states = np.zeros(n, np.float32), np.zeros((n, 256), np.float32)
        for length in np.unique(sp.lengths[:n]):
            idx = np.flatnonzero(sp.lengths[:n] == length)
            masks = (sp.sequence_masks[idx, :length, None] >> np.arange(P.N_ACTIONS) & 1).astype(bool)
            lg, val, hidden = net.forward_sequence(
                torch.from_numpy(sp.sequence_maps[idx, :length]),
                torch.from_numpy(sp.sequence_scalars[idx, :length]), torch.from_numpy(masks),
                torch.from_numpy(sp.states[idx]), context=torch.from_numpy(sp.sequence_contexts[idx, :length]))
            logits[idx], values[idx], states[idx] = lg[:, -1].numpy(), val[:, -1].numpy(), hidden.numpy()
        if opening:
            logits[:] = -1e8
            logits[:, 0] = 0  # one straight opening decision, outside the recorded tree
            opening = False
        sp.feed(logits, values, policy_updates=np.full(n, ck.get("update", 0), np.int64), next_states=states)
    else:
        raise RuntimeError("search trace exceeded its bounded evaluation budget")
    if not frames or max(frames) != simulations:
        raise RuntimeError("search trace did not complete its simulation budget")
    return {"checkpoint": str(checkpoint.resolve()), "update": ck.get("update", 0), "seed": seed,
            "source": "C++ SelfPlay search; isolated staged opening; no replay ring or optimizer",
            "actions": ["Straight", "Left", "Right", "Brake", "Left + brake", "Right + brake"],
            "frames": [frames[k] for k in sorted(frames)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/blacklight/latest.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=24)
    args = parser.parse_args()
    result = record(args.checkpoint, simulations=args.simulations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Recorded {args.simulations} simulations at checkpoint u{result['update']}")


if __name__ == "__main__":
    main()
