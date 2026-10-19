"""Explicitly add persistent, action/time-aware memory to the shared V4 checkpoint.

The pure converter retains learned tensors, counters and optimizer moments. File
migration takes the normal writer lock and atomically replaces only its requested
checkpoint; callers may pass a verified fingerprint to reject a concurrent change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .model import MEMORY_CONTEXT, MEMORY_HIDDEN, MEMORY_VERSION, make_net, net_version
from .protocol import OBSERVATION_VERSION

MEMORY_PREFIXES = ("memory_gru.", "memory_projection.")
EXPANDED = "memory_gru.weight_ih_l0"


def _marker(value, expected):
    return (isinstance(value, torch.Tensor) and value.ndim == 0
            and value.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            and int(value.item()) == expected)


def _expand_model(source, template):
    if not _marker(source.get("net_version"), 4) or not _marker(source.get("observation_version"), OBSERVATION_VERSION):
        raise ValueError("Memory migration requires V4 observation schema 2; migrate the inputs first")
    have_memory = "memory_version" in source
    if have_memory and not _marker(source["memory_version"], MEMORY_VERSION):
        raise ValueError("Unsupported source memory version")
    expected = set(template) if have_memory else {k for k in template
                                                if k != "memory_version" and not k.startswith(MEMORY_PREFIXES)}
    if set(source) != expected:
        raise ValueError("Source V4 layers are missing, partial or unexpected")
    out = dict(source)
    expanded = False
    for name, value in source.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Model tensor {name} is invalid")
        if tuple(value.shape) == tuple(template[name].shape):
            continue
        if name == EXPANDED and tuple(value.shape) == (3 * MEMORY_HIDDEN, 768):
            out[name] = torch.cat([value, value.new_zeros(3 * MEMORY_HIDDEN, MEMORY_CONTEXT)], 1)
            expanded = True
        else:
            raise ValueError(f"Model tensor {name} has an incompatible shape")
    if not have_memory:
        dtype = source["pi.weight"].dtype
        for name, value in template.items():
            if name == "memory_version" or name.startswith(MEMORY_PREFIXES):
                out[name] = value.clone().to(dtype=dtype if value.is_floating_point() else value.dtype)
    return out, not have_memory or expanded


def _expand_optimizer(source, old_names, new_names, old_model, new_model):
    if source is None:
        return None
    out = copy.deepcopy(source)
    groups = out.get("param_groups", [])
    ids = [pid for group in groups for pid in group.get("params", [])]
    if not groups or len(ids) != len(old_names) or len(set(ids)) != len(ids):
        raise ValueError("Cannot safely map optimizer groups to the source V4 parameters")
    if any(not isinstance(pid, int) or pid < 0 for pid in ids):
        raise ValueError("Optimizer parameter identifiers must be nonnegative integers")
    if any("param_names" in group for group in groups):
        names = [name for group in groups for name in group.get("param_names", [])]
        if len(names) != len(ids) or set(names) != set(old_names):
            raise ValueError("Optimizer parameter names do not match the source model")
    else:
        names = old_names
    by_id = dict(zip(ids, names))
    states = out.setdefault("state", {})
    for pid, moments in states.items():
        if pid not in by_id:
            raise ValueError("Optimizer contains state for an unknown parameter")
        name = by_id[pid]
        old_shape = tuple(old_model[name].shape)
        for key, value in list(moments.items()):
            if not isinstance(value, torch.Tensor) or (key == "step" and value.numel() == 1):
                continue
            if tuple(value.shape) != old_shape:
                raise ValueError(f"Optimizer moment {name}/{key} has an incompatible shape")
            if name == EXPANDED and old_shape != tuple(new_model[name].shape):
                moments[key] = torch.cat([value, value.new_zeros(3 * MEMORY_HIDDEN, MEMORY_CONTEXT)], 1)
    # Old modules retain their registration order. Append the memory parameters to
    # the final group, preserving every existing group's options and identifiers.
    added = [name for name in new_names if name not in old_names]
    next_id = max(ids, default=-1) + 1
    group = groups[-1]
    for name in added:
        group["params"].append(next_id)
        if "param_names" in group:
            group["param_names"].append(name)
        weight = new_model[name]
        states[next_id] = {"step": torch.tensor(0.), "exp_avg": torch.zeros_like(weight),
                           "exp_avg_sq": torch.zeros_like(weight)}
        if group.get("amsgrad", False):
            states[next_id]["max_exp_avg_sq"] = torch.zeros_like(weight)
        next_id += 1
    return out


def convert_checkpoint(checkpoint: dict) -> tuple[dict, dict]:
    """Pure conversion, including main/helper and imitation AdamW continuity."""
    if checkpoint.get("trainer") != "teach" or "exploiter" not in checkpoint:
        raise ValueError("Migrate the complete shared Blacklight training checkpoint")
    with torch.random.fork_rng(devices=[]):
        # Only the CPU template generator is needed. torch.manual_seed would
        # also reseed an already active MPS/CUDA caller outside this CPU fork.
        torch.random.default_generator.manual_seed(0)
        net = make_net()
    template = net.state_dict()
    model, changed = _expand_model(checkpoint["model"], template)
    exploiter, helper_changed = _expand_model(checkpoint["exploiter"], template)
    if changed != helper_changed:
        raise ValueError("Main and helper models have different memory revisions")
    net.load_state_dict(model, strict=True)
    net.load_state_dict(exploiter, strict=True)
    net_version(model)
    all_names = [name for name, _ in net.named_parameters()]
    old_names = [name for name in all_names if name in checkpoint["model"]]
    out = dict(checkpoint)
    out["model"], out["exploiter"] = model, exploiter
    out["opt"] = _expand_optimizer(checkpoint.get("opt"), old_names, all_names, checkpoint["model"], model)
    helper_names = [name for name in all_names if name in checkpoint["exploiter"]]
    out["exploiter_opt"] = _expand_optimizer(checkpoint.get("exploiter_opt"), helper_names, all_names,
                                            checkpoint["exploiter"], exploiter)
    imitation = dict(checkpoint.get("imitation_state", {}))
    if "optimizer" in imitation:
        policy_names = [name for name in all_names if not name.startswith(("v.", "aux.", "own."))]
        old_policy_names = [name for name in policy_names if name in checkpoint["model"]]
        imitation["optimizer"] = _expand_optimizer(imitation["optimizer"], old_policy_names, policy_names,
                                                   checkpoint["model"], model)
    out["imitation_state"] = imitation
    old_count = sum(checkpoint["model"][name].numel() for name in old_names)
    count = sum(p.numel() for p in net.parameters())
    report = {"migrated": changed, "memory_version": MEMORY_VERSION, "memory_hidden": MEMORY_HIDDEN,
              "memory_context": MEMORY_CONTEXT, "parameters_before": old_count, "parameters_after": count,
              "new_parameters": count - old_count, "update_before": checkpoint.get("update"),
              "update_after": out.get("update"), "imitation_steps": imitation.get("cumulative_steps", 0),
              "optimizer_moments": "preserved; new memory parameters start with zero Adam moments",
              "initial_residual_zero": bool(torch.count_nonzero(model["memory_projection.weight"]) == 0
                                            and torch.count_nonzero(model["memory_projection.bias"]) == 0)}
    out["memory_migration"] = report if changed else checkpoint.get("memory_migration", report)
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
