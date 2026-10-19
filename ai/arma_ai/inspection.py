"""Measured network structure and repeatable input substitutions for the browser inspector.

These describe an actual loaded model. Substituting an input is a sensitivity test on one
observation, not a causal explanation, and its inputs can be outside the training distribution.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from . import protocol as P
from .model import GPoolResBlock, MEMORY_CONTEXT


def describe(net, layers: list[dict], stage_for_parameter) -> dict:
    """Enrich stages measured from the loaded model and describe their actual dependencies."""
    counts = defaultdict(int)
    for name, parameter in net.named_parameters():
        counts[stage_for_parameter(name, net)] += parameter.numel()
    by_id = {layer["id"]: layer for layer in layers}
    branches = []
    context_stages = ((["nearby_modulation", "close_modulation", "opponents_query"]))
    towers = tuple((name, name.title() + " view", tower, maps)
                   for name, tower, maps in zip(net.tower_names, net.towers, net.tower_maps))
    map_ids = ("local_in", "close_in", "global_in", "territory_in")
    for group, name, tower, maps in towers:
        stages = [layer["id"] for layer in layers if layer["group"] == group
                  and layer["kind"] in ("conv", "dense") and layer["id"] not in context_stages]
        branches.append({"id": group, "name": name, "inputs": [map_ids[m] for m in maps], "stages": stages})
        for lid in stages:
            layer = by_id[lid]
            suffix = lid[len(group) + 1:]
            if layer["kind"] == "dense":
                layer.update(short=f"{layer['shape'][0]} features", operation="Linear + ReLU",
                             explanation="Flattens the spatial features into a fixed-length summary for fusion.")
            elif suffix.startswith("a"):
                index = int(suffix[1:])
                attention = tower.attn[index - 1]
                layer.update(short=f"Attention {index}",
                             operation=f"Self-attention · {attention.heads} heads" + (" + position embedding" if index == 1 else ""),
                             explanation=f"Each arena cell can read information from every other cell through {attention.heads} attention heads.")
            elif suffix == "c1":
                if hasattr(tower, "pre"):
                    layer.update(short="Conv → 32×32", operation="3×3 convolution · stride 2 + ReLU",
                                 explanation="Reads overlapping local patches at 32×32 before further reduction; Nearby and Close learn separate filters for their different physical scales.")
                else:
                    layer.update(short="Patch → 16×16", operation="4×4 convolution · stride 4 + ReLU",
                                 explanation="Reads each 4×4 patch and compresses it into learned channels. Spatial resolution falls from 64×64 to 16×16 here; whether fine geometry survives needs testing.")
            elif suffix == "pre":
                layer.update(short="Residual · 32×32", operation="Residual · 2 × 3×3 convolutions + ReLU",
                             explanation="Processes nearby wall and gap geometry at 32×32, with a skip connection, before the next reduction.")
            elif suffix == "reduce":
                layer.update(short="Conv → 16×16", operation="3×3 convolution · stride 2 + ReLU",
                             explanation="Reduces the processed 32×32 local features to 16×16 for the deeper residual blocks.")
            elif suffix == "condition":
                layer.update(short="Arena-conditioned features", operation="Bounded FiLM + ReLU",
                             explanation="The arena summary adjusts each local channel's gain and bias after the residual blocks. Gain is 1 + 0.1 × tanh(gain); bias is 0.1 × tanh(bias).")
            elif (suffix.startswith("c")) and (2 <= int(suffix[1:]) <= len(tower.res) + 1):
                index = int(suffix[1:]) - 1
                block = tower.res[index - 1]
                pooled = isinstance(block, GPoolResBlock)
                layer.update(short=f"Global pool · {index}" if pooled else f"Residual {index}",
                             operation="Residual · 2 × 3×3 convolutions" + (" + global mean/max pooling" if pooled else ""),
                             explanation="Two convolutions read nearby patterns; a skip connection preserves the incoming features."
                             + (" Pooled board statistics also condition every cell." if pooled else ""))
            else:
                side = layer["shape"][-1]
                layer.update(short=f"Conv → {side}×{side}", operation="3×3 convolution + ReLU" + (" · stride 2" if side < 16 else ""),
                             explanation="Extracts spatial patterns and, when strided, reduces the map before its summary.")
    branches.append({"id": "scalars", "name": "Numbers", "inputs": ["scalars_in"],
                     "stages": [layer["id"] for layer in layers if layer["group"] == "scalars" and layer["kind"] == "dense"]})
    branches.append({"id": "opponents", "name": "Opponents", "inputs": ["opponents_in"],
                     "stages": ["opponents_h1", "opponents_h2", "opponents_weights", "opponents_pool"]})
    for layer in layers:
        lid = layer["id"]
        layer["parameters"] = counts[lid]
        layer.setdefault("short", layer["name"])
        if layer["kind"] == "input":
            layer.update(operation="Observation", explanation="The exact observation sent by the engine. Map bits are unpacked into separate feature planes before convolution.")
            if lid == "opponents_in":
                layer.update(operation="Scalar slices 64:88 + 96:108 → 3 × 12",
                             explanation="The three nearest enemies' measurements from the same observation, grouped by enemy. Each slot contains relative position, distance, heading, speed, braking, presence, brake reserve, rubber used relative to its budget, and waits before left and right turns.")
        elif lid.startswith("scalars_h"):
            module = net.scalars[0 if lid.endswith("1") else 2]
            layer.update(short=f"{module.in_features} → {module.out_features}", operation="Linear + ReLU",
                         explanation=(("Processes exact speed, rubber, brake, timing, ray and arena measurements. Enemy-slot measurements use the separate shared opponent encoder; the action mask is applied separately.")))
        elif lid.endswith("_modulation"):
            layer.update(short="Arena → 64 gains + 64 biases", operation="Linear · 256 → 128",
                         explanation="Reads the arena summary and produces raw gain and bias coefficients for one local view. The following conditioning operation bounds them with tanh.")
        elif lid in ("opponents_h1", "opponents_h2"):
            layer.update(short="3 opponents × 64 features", operation="Shared linear + ReLU",
                         explanation="Applies the same encoder to each nearest-enemy slot. The display flattens the three 64-feature tokens; absent slots are excluded when routing weights are formed.")
        elif lid == "opponents_query":
            layer.update(short="Tactical context → 64", operation="Linear · 896 → 64 → tanh",
                         explanation="Reads the Nearby, Close and Arena summaries plus the scalar branch to form a query for opponent selection.")
        elif lid == "opponents_weights":
            layer.update(short="3 routing weights", operation="Token-query dot product / 8 → presence mask → softmax",
                         explanation="Weights the three opponent tokens using the tactical query. Absent slots receive exactly zero; all three weights are zero when no enemies are present. These are routing weights, not kill probabilities.")
        elif lid == "opponents_pool":
            layer.update(short="64 opponent features", operation="Weighted sum of 3 tokens",
                         explanation="Combines the opponent tokens using their routing weights. The result is exactly zero when every opponent slot is absent.")
        elif lid.startswith("trunk"):
            module = net.trunk[0 if lid.endswith("1") else 2]
            explanation = (("Combines three spatial summaries, the exact-number branch and the selected opponent features. Arena context has already conditioned both local views."))
            layer.update(short=("Fusion · " if lid.endswith("1") else "") + f"{module.in_features} → {module.out_features}",
                         operation="Linear + ReLU", explanation=explanation)
        elif lid == "memory_in":
            layer.update(short=f"Previous memory · {net.state_size}", operation="Retained state from the preceding decision",
                         explanation="The same player's GRU state immediately before this decision. It starts at zero for a new life and resets after a gap, identity change or checkpoint change.")
        elif lid == "memory_context":
            layer.update(short="Previous control + elapsed time", operation="6 executed-action indicators + seconds / 0.05",
                         explanation="The actual preceding steering/braking control and the time since that observation. The control indicators are all zero when no previous control executed, including the dropped time-zero input.")
        elif lid == "memory_state":
            layer.update(short=f"GRU · {net.state_size} memory units", operation=f"GRU · 775 inputs → {net.state_size} hidden units",
                         explanation="Reads the current 768 combat features, previous control and elapsed time together with the retained state. Its output is kept for this player's next decision; inspection and input tests use copies.")
        elif lid == "memory_residual":
            layer.update(short="Memory → 768 features", operation=f"Linear · {net.state_size} → 768",
                         explanation="Converts recurrent memory into a learned correction. Migration starts this projection at zero, preserving the existing policy before memory training.")
        elif lid == "memory_output":
            layer.update(short="Combat features + memory", operation="768 trunk features + memory correction",
                         explanation="The adapted features used by the move, value and auxiliary heads. Current spatial features retain a direct path around memory.")
        elif lid == "policy":
            layer.update(short=f"Policy · {net.n_actions} actions", operation="Linear → legal-action mask → softmax",
                         explanation="Produces the move distribution. Forbidden actions receive zero probability; the match can sample an action or take the largest probability."
                         + ((" Six joint actions allow steering and braking together.")))
        elif lid == "value":
            layer.update(short="Value · 1 score", operation=("Linear → tanh"),
                         explanation=(("A learned discounted round-outcome estimate bounded to [-1, 1] by tanh. It is not a calibrated probability of winning.")))
    names = ["Territory share in 2s", "Death within 2s"]
    explanation = "Predicts territory share and death within two seconds."
    names.append("Credited kill AND survival within 2s")
    explanation += " The third prediction requires a credited kill within two seconds AND survival through that horizon."
    layers.extend([
        {"id": "aux", "group": "head", "kind": "auxiliary", "shape": [net.n_aux], "parameters": counts["aux"],
         "name": "auxiliary predictions", "short": "Auxiliary · training", "names": names,
         "label": ("territory, danger and safe kill"),
         "operation": "Linear → sigmoid", "explanation": explanation + " These supervise training but do not choose the deployed action."},
        {"id": "own", "group": "head", "kind": "auxiliary", "shape": [3, 16, 16], "parameters": counts["own"],
         "name": "future ownership", "short": "Ownership · training", "label": "3 classes · 16×16",
         "operation": "1×1 convolution → class softmax", "explanation": "Reads the arena features before downsampling. Predicts who reaches each cell first two seconds ahead; training upsamples its logits to 64×64."},
    ])
    edges = []
    for branch in branches:
        first, *rest = branch["stages"]
        edges.extend([source, first] for source in branch["inputs"])
        edges.extend([a, b] for a, b in zip(branch["stages"], rest))
        edges.append([branch["stages"][-1], "trunk1"])
    for group in net.tower_names[:2]:
        edges.extend([["arena_fc", f"{group}_modulation"], [f"{group}_modulation", f"{group}_condition"]])
    edges.extend([[f"{group}_fc", "opponents_query"] for group in net.tower_names])
    edges.extend([["scalars_h2", "opponents_query"], ["opponents_query", "opponents_weights"],
                  ["opponents_h2", "opponents_pool"]])
    memory_stages = ["memory_in", "memory_context", "memory_state", "memory_residual", "memory_output"]
    edges.extend([["trunk1", "trunk2"], ["trunk2", "memory_state"], ["memory_in", "memory_state"],
                  ["memory_context", "memory_state"], ["memory_state", "memory_residual"],
                  ["trunk2", "memory_output"], ["memory_residual", "memory_output"],
                  ["memory_output", "policy"], ["memory_output", "value"]])
    arena = next(branch["stages"] for branch in branches if branch["id"] == "arena")
    feature_stage = arena[-4]  # last residual/attention stage, before down0, down1 and fc
    edges.extend([["memory_output", "aux"], [feature_stage, "own"]])
    return {"parameters": sum(parameter.numel() for parameter in net.parameters()), "branches": branches, "edges": edges,
            "observation_version": int(net.observation_version.item()), "scalar_count": P.N_SCALARS,
            "context_stages": context_stages, "history_frames": (2), "decision_interval": 0.05,
            "memory_stages": memory_stages,
            "memory": {"kind": "GRU", "hidden": net.state_size, "persistent": True,
                       "previous_action": True, "elapsed_time": True,
                       "reset": "new life, round, missing decision, player or checkpoint change"},
            "heads": {"auxiliary": hasattr(net, "aux"), "ownership": hasattr(net, "own"),
                      "attention": bool((any(len(tower.attn) for tower in net.towers))),
                      "opponents": True}}


def activation_stats(layers: list[tuple[str, np.ndarray]]) -> dict:
    """Absolute statistics, before the display's brightness scaling and 8-bit quantisation."""
    return {name: {"min": round(float(x.min()), 5), "max": round(float(x.max()), 5),
                   "mean": round(float(x.mean()), 5), "std": round(float(x.std()), 5),
                   "zero": round(float((x == 0).mean()), 4), "signed": bool((x < 0).any())}
            for name, x in layers}


@torch.inference_mode()
def input_sensitivity(net, maps: np.ndarray, scalars: np.ndarray, mask: np.ndarray, *,
                      state: torch.Tensor | None = None, context: torch.Tensor | None = None) -> dict:
    """Compare input substitutions with independent copies of the exact predecision memory.

    Ordinary map/sensor substitutions keep memory and legality fixed. Explicit memory/control
    substitutions clear only that input and never advance the live player's recurrent state.
    """
    names = ((0, "Nearby map"), (1, "Close map"), (2, "Arena map"), (3, "Territory map"))
    reads = (range(P.N_MAPS))
    variants = [("baseline", "Baseline", maps.copy(), scalars.copy())]
    for m, label in names:
        if m not in reads:
            continue
        x = maps.copy()
        x[m] = 0
        if x.shape[0] >= 2 * P.N_MAPS:
            x[P.N_MAPS + m] = 0
        variants.append((f"map_{m}", label, x, scalars.copy()))
    indices = torch.arange(P.N_SCALARS)[None]
    own_indices = net.scalar_input(indices).flatten().numpy()
    enemy_indices = net.opponent_input(indices).flatten().numpy()
    numbers_removed = scalars.copy()
    numbers_removed[own_indices] = 0
    variants.append(("numbers", ("Own / arena numbers"), maps.copy(), numbers_removed))
    enemies_removed = scalars.copy()
    enemies_removed[enemy_indices] = 0
    variants.append(("opponents", "Enemy measurements", maps.copy(), enemies_removed))
    x = maps.copy()
    x[P.N_MAPS:] = x[:P.N_MAPS]
    variants.append(("history", "Previous frame → current", x, scalars.copy()))
    if state is not None:
        variants.append(("memory", "Retained memory → zero", maps.copy(), scalars.copy()))
    if context is not None:
        variants.append(("controls", "Previous control / time → zero", maps.copy(), scalars.copy()))
    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype
    x = torch.from_numpy(np.stack([v[2] for v in variants])).to(device)
    s = torch.from_numpy(np.stack([v[3] for v in variants])).to(device, dtype=dtype)
    m = torch.from_numpy(np.repeat(np.asarray(mask, bool)[None, :net.n_actions], len(variants), axis=0)).to(device)
    memory = net.initial_state(1) if state is None else state.detach().to(device=device, dtype=dtype).clone()
    controls = s.new_zeros(1, MEMORY_CONTEXT) if context is None else context.detach().to(device=device, dtype=dtype).clone()
    if memory.shape != (1, net.state_size) or controls.shape != (1, MEMORY_CONTEXT):
        raise ValueError("input tests require the single decision's predecision memory and context")
    memory = memory.repeat(len(variants), 1)
    controls = controls.repeat(len(variants), 1)
    for i, (key, _, _, _) in enumerate(variants):
        if key == "memory":
            memory[i].zero_()
        elif key == "controls":
            controls[i].zero_()
    logits, values, _ = net.forward_step(x, s, m, memory, context=controls)
    probs = torch.softmax(logits.float(), 1).cpu().numpy()
    values = values.float().cpu().numpy()
    rows = []
    for i, (key, label, _, _) in enumerate(variants[1:], 1):
        rows.append({"id": key, "label": label, "probs": probs[i].tolist(),
                     "policy_shift": float(np.abs(probs[i] - probs[0]).sum() / 2),
                     "preferred": int(probs[i].argmax()), "value_delta": float(values[i] - values[0])})
    return {"baseline": probs[0].tolist(), "value": float(values[0]), "rows": rows}
