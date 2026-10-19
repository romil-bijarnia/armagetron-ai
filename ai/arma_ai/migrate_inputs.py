"""Explicit, behavior-preserving migration from V4's 96 inputs to observation schema 2.

The original layers, counters and optimizer moments are retained. Only the first
opponent layer expands: its four added columns and matching moments start at zero.
This module never initializes a replacement fighter or selects another checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from . import protocol as P
from .model import make_net, net_version

EXPANDED = "opponents.0.weight"
OLD_SHAPE = (64, 8)
NEW_SHAPE = (64, P.ENEMY_FEATURES)


def _integer_marker(value, expected):
    return (isinstance(value, torch.Tensor) and value.ndim == 0
            and value.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            and int(value.item()) == expected)


def expand_model(state: dict) -> tuple[dict, bool]:
    if not _integer_marker(state.get("net_version"), 4):
        raise ValueError("Input migration requires an explicitly marked V4 model")
    marker = state.get("observation_version")
    if _integer_marker(marker, P.OBSERVATION_VERSION):
        net_version(state)
        if tuple(state[EXPANDED].shape) != NEW_SHAPE:
            raise ValueError("Observation marker and opponent weight shape disagree")
        return dict(state), False
    if marker is not None and not _integer_marker(marker, 1):
        raise ValueError("Unsupported source observation version")
    if EXPANDED not in state or tuple(state[EXPANDED].shape) != OLD_SHAPE:
        raise ValueError("Legacy V4 opponent weights must have shape [64, 8]")
    out = dict(state)
    weight = state[EXPANDED]
    out[EXPANDED] = torch.cat([weight, weight.new_zeros(64, P.ENEMY_EXTRA_FEATURES)], dim=1)
    out["observation_version"] = state["net_version"].new_tensor(P.OBSERVATION_VERSION)
    return out, True


def expand_optimizer(optimizer: dict | None, parameter_names: list[str], old_state: dict,
                     changed: bool) -> dict | None:
    if optimizer is None:
        return None
    out = copy.deepcopy(optimizer)
    if not out.get("state"):
        return out
    ids = [i for group in out.get("param_groups", []) for i in group["params"]]
    if len(ids) != len(parameter_names) or len(set(ids)) != len(ids):
        raise ValueError("Cannot safely map optimizer parameters to the V4 layers")
    id_to_name = dict(zip(ids, parameter_names))
    for pid, values in out["state"].items():
        if pid not in id_to_name:
            raise ValueError("Optimizer state contains an unknown parameter")
        name = id_to_name[pid]
        shape = tuple(old_state[name].shape)
        for key, value in list(values.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step" and value.numel() == 1:
                continue
            if tuple(value.shape) != shape:
                raise ValueError(f"Optimizer moment {name}/{key} has an incompatible shape")
            if changed and name == EXPANDED:
                values[key] = torch.cat([value, value.new_zeros(64, P.ENEMY_EXTRA_FEATURES)], dim=1)
    return out


def convert_checkpoint(checkpoint: dict) -> tuple[dict, dict]:
    """Pure conversion; no source tensors or files are mutated."""
    if checkpoint.get("trainer") != "teach":
        raise ValueError("Migrate the complete shared Blacklight training checkpoint")
    model, main_changed = expand_model(checkpoint["model"])
    exploiter, exploiter_changed = expand_model(checkpoint["exploiter"])
    if main_changed != exploiter_changed:
        raise ValueError("Main and helper models have different observation revisions")
    # Strictly validate every layer before constructing a file that a trainer may use.
    net = make_net()
    net.load_state_dict(model, strict=True)
    net.load_state_dict(exploiter, strict=True)
    all_names = [name for name, _ in net.named_parameters()]
    policy_names = [name for name in all_names if not name.startswith(("v.", "aux.", "own."))]
    out = dict(checkpoint)
    out["model"], out["exploiter"] = model, exploiter
    out["opt"] = expand_optimizer(checkpoint.get("opt"), all_names, checkpoint["model"], main_changed)
    out["exploiter_opt"] = expand_optimizer(checkpoint.get("exploiter_opt"), all_names,
                                           checkpoint["exploiter"], exploiter_changed)
    imitation = dict(checkpoint.get("imitation_state", {}))
    if "optimizer" in imitation:
        imitation["optimizer"] = expand_optimizer(imitation["optimizer"], policy_names,
                                                  checkpoint["model"], main_changed)
    out["imitation_state"] = imitation
    report = {"migrated": main_changed, "architecture": 4,
              "observation_version": P.OBSERVATION_VERSION, "scalars": P.N_SCALARS,
              "opponent_features": P.ENEMY_FEATURES, "new_parameters": 256 if main_changed else 0,
              "update_before": checkpoint.get("update"), "update_after": out.get("update"),
              "imitation_steps": imitation.get("cumulative_steps", 0),
              "optimizer_moments": "preserved; new feature columns zero-padded",
              "new_feature_weights_zero": bool(torch.count_nonzero(model[EXPANDED][:, 8:]) == 0)}
    out["input_migration"] = report if main_changed else checkpoint.get("input_migration", report)
    return out, report


def migrate_file(path: Path, expected_sha256: str | None = None) -> dict:
    from .imitate import RunLock, atomic_checkpoint

    path = path.resolve()
    lock = RunLock(path.parent)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("The checkpoint changed during verification; refusing to overwrite newer training")
        source = torch.load(path, map_location="cpu", weights_only=True)
        converted, report = convert_checkpoint(source)
        if report["migrated"]:
            atomic_checkpoint(path, converted)
        return {**report, "checkpoint": str(path), "source_sha256": digest,
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(migrate_file(args.checkpoint, args.expected_sha256), indent=2))


if __name__ == "__main__":
    main()
