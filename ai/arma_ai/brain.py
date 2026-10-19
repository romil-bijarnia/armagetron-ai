"""Watch an actual training arena, inspect a policy separately, and measure fixed-opponent results.

    uv run arma-brain --no-open
    uv run arma-brain --source inspection --no-open
    uv run arma-brain --source inspection --greedy --no-open

Live training reads the actor's watch.json feed. Freeze and step affect displayed samples only;
the trainer continues at its own pace. Each move records its actual search, sampling or opponent
logic and the checkpoint used for its root prediction. Policy inspection is a separate real-engine
match, with explicitly selectable sampled or highest-probability moves and match playback speed.
Network shows the architecture; activations and input tests come only from policy inspection.
Progress separates rolling training outcomes from bounded real-engine checks of frozen weights.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import queue
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import protocol as P
from .engine import ArenaConfig, EnginePool, _parse_step
from .inspection import activation_stats, describe, input_sensitivity
from .model import MEMORY_CONTEXT, PolicyNet, _masked, make_net, net_version
from .show import Match, WALLS_STAY_UP
from .play import PolicyHistory
from .watch_server import TrainingFeed, read_record

PROJECT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("web") / "brain.html"
MAX_FPS = 20  # decisions sent to the page per second of wall time
LOCAL_PLANES = ["walls", "own trail", "enemy trails", "team trails", "enemy heads", "enemy paths", "outside",
                "team heads"]
GLOBAL_PLANES = ["walls", "own trail", "enemy trails", "own head", "enemy heads", "outside", "team trails",
                 "team heads"]
MOVES = ["Straight", "Left", "Right", "Brake", "Left + brake", "Right + brake"]


def scalar_names() -> list[str]:
    """What each of the exact numbers is, in the order gNeural.cpp packs them."""
    n = ["speed", "rubber used", "brake reservoir", "braking", "time since last turn", "can turn left",
         "can turn right", "turns queued", "arena width", "arena height", "position across the arena",
         "position along the arena", "round time", "enemies alive", "teammates alive", "base speed"]
    for k in range(16):  # 16 rays, clockwise from straight ahead
        n += [f"ray {k * 22.5:g}° distance", f"ray {k * 22.5:g}° closeness"]
    for k in range(4):
        n += [f"ray {k * 90}° hits {h}" for h in ("the rim", "an enemy wall", "a teammate's wall", "own wall")]
    for e in range(3):  # the three nearest enemies
        n += [f"enemy {e + 1} {w}" for w in ("ahead", "to the side", "distance", "heading ahead",
                                           "heading sideways", "speed", "braking", "present")]
    n += ["unused"] * (96 - len(n))
    for e in range(3):
        n += [f"enemy {e + 1} {w}" for w in ("brake reserve", "rubber used / budget",
                                            "left turn wait", "right turn wait")]
    n += ["unused"] * (P.N_SCALARS - len(n))
    return n


# ------------------------------------------------------------------------------------------- network

MAP_IDS = ("local_in", "close_in", "global_in", "territory_in")
MAP_LABELS = ("what it sees nearby", "up close", "the whole arena", "who gets there first")
MAP_PLANE_NAMES = (LOCAL_PLANES, LOCAL_PLANES, GLOBAL_PLANES,
                   ["mine first", "enemy first", "d1", "d2", "d4", "d8", "d16", "d32"])
MAP_GROUPS = ("local", "close", "global", "territory")  # the page's colour band of each input map
GROUP_WORD = {"player": "player view", "arena": "arena view", "local": "nearby", "global": "arena",
              "nearby": "nearby view", "close": "close view", "opponents": "opponents",
              "scalars": "numbers", "trunk": "trunk", "memory": "persistent memory"}
def input_maps(net: nn.Module) -> tuple[int, ...]:
    """Indices of the four maps read by the policy."""
    return tuple(range(P.N_MAPS))


def stacked(maps: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """(2*N_MAPS, G, G): this decision's maps, then the previous decision's (or these again at a spawn)."""
    return np.concatenate([maps, maps if prev is None else prev])


@dataclass
class Thought:
    """One decision, with everything the page shows."""
    layers: list[tuple[str, np.ndarray]]  # (name, activations) after each stage, in forward order
    probs: np.ndarray  # the odds of each move
    value: float
    own: np.ndarray | None = None  # (3, 16, 16): nobody / me / an enemy holds the cell two seconds on
    aux: np.ndarray | None = None  # territory, death, credited kill AND survival within 2s
    attention: list[np.ndarray] = field(default_factory=list)  # per arena layer: (heads, cells, cells)
    opponents: np.ndarray | None = None  # routing weights of the three nearest enemy slots
    state: torch.Tensor | None = None  # private next recurrent state, retained by the live controller


def attend(layer: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnLayer.forward, also handing back the attention weights (B, heads, N, N)."""
    B, N, C = x.shape
    h = layer.heads
    q, k, v = layer.qkv(layer.ln1(x)).view(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
    att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(C // h), -1)
    x = x + layer.proj((att @ v).transpose(1, 2).reshape(B, N, C))
    return x + layer.fc2(torch.nn.functional.gelu(layer.fc1(layer.ln2(x)))), att


@torch.no_grad()
def think(net: PolicyNet, maps: np.ndarray, scalars: np.ndarray, mask: np.ndarray,
          state: torch.Tensor | None = None, context: torch.Tensor | None = None) -> Thought:
    """Run one decision through NET and keep every stage's output and every head's prediction.

    MAPS: (2*N_MAPS, GRID, GRID) uint8 (current then previous frame), SCALARS: (N_SCALARS,) float32,
    MASK: (N_ACTIONS,) bool. STATE and CONTEXT are frozen predecision inputs; no caller-owned
    tensor is changed. Omitting them gives the same zero-state diagnostic as net.forward."""
    out: list[tuple[str, torch.Tensor]] = []
    attention: list[np.ndarray] = []
    own = None
    opponents = None
    parameter = next(net.parameters())
    x_in = torch.from_numpy(np.array(maps, np.uint8))[None].to(parameter.device)
    s_in = torch.from_numpy(np.array(scalars, np.float32))[None].to(parameter.device, dtype=parameter.dtype)

    def array(x: torch.Tensor) -> np.ndarray:
        return x.float().cpu().numpy()

    def mlp(prefix: str, seq: nn.Sequential, h: torch.Tensor) -> torch.Tensor:
        n = 0
        for m in seq:
            h = m(h)
            if isinstance(m, nn.ReLU):
                n += 1
                out.append((f"{prefix}{n}", h[0]))
        return h

    def spatial(t: int, arena_summary: torch.Tensor | None = None) -> torch.Tensor:
        nonlocal own
        group = net.tower_names[t]
        tower = net.towers[t]
        x = torch.relu(tower.stem(net.tower_input(x_in, t)))
        out.append((f"{group}_c1", x[0]))
        if hasattr(tower, "pre"):
            x = tower.pre(x)
            out.append((f"{group}_pre", x[0]))
            x = torch.relu(tower.reduce(x))
            out.append((f"{group}_reduce", x[0]))
        n = 1
        for b in tower.res:
            x = b(x)  # a plain residual block or global-pooling block
            n += 1
            out.append((f"{group}_c{n}", x[0]))
        if len(getattr(tower, "attn", ())):
            # Each attention layer is a stage of its own.
            B, C, H, W = x.shape
            cells = x.flatten(2).transpose(1, 2) + tower.pos
            for a_i, layer in enumerate(tower.attn):
                cells, att = attend(layer, cells)
                attention.append(array(att[0]))
                x = cells.transpose(1, 2).reshape(B, C, H, W)
                out.append((f"{group}_a{a_i + 1}", x[0]))
        if (group == "arena"):
            own = array(torch.softmax(net.own(x)[0], 0))
        if arena_summary is not None:
            # Capture the affine output from the very call used by the model's FiLM operation.
            handle = net.conditioning[t].register_forward_hook(
                lambda _module, _args, result: out.append((f"{group}_modulation", result[0])))
            try:
                x = net.condition(x, arena_summary, t)
            finally:
                handle.remove()
            out.append((f"{group}_condition", x[0]))
        x = torch.relu(tower.down0(x))
        out.append((f"{group}_c{n + 1}", x[0]))
        x = torch.relu(tower.down1(x))
        out.append((f"{group}_c{n + 2}", x[0]))
        h = torch.relu(tower.fc(x.flatten(1)))
        out.append((f"{group}_fc", h[0]))
        return h

    arena = spatial(2)
    parts = [spatial(0, arena), spatial(1, arena), arena]
    hs = mlp("scalars_h", net.scalars, (net.scalar_input(s_in)))
    fused = torch.cat(parts + [hs], 1)
    handle = net.opponents[1].register_forward_hook(
        lambda _module, _args, result: out.append(("opponents_h1", result[0].flatten())))
    try:
        pool, weights, tokens, query = net.attend_opponents(s_in, fused)
    finally:
        handle.remove()
    out.extend([("opponents_h2", tokens[0].flatten()), ("opponents_query", query[0]),
                ("opponents_weights", weights[0]), ("opponents_pool", pool[0])])
    opponents = array(weights[0])
    fused = torch.cat([fused, pool], 1)
    h = mlp("trunk", net.trunk, fused)
    before = net.initial_state(1) if state is None else state.detach().to(h).clone()
    controls = h.new_zeros(1, MEMORY_CONTEXT) if context is None else context.detach().to(h).clone()
    if before.shape != (1, net.state_size) or controls.shape != (1, MEMORY_CONTEXT):
        raise ValueError("one inspection decision requires one memory state and one control/time context")
    out.extend([("memory_in", before[0]), ("memory_context", controls[0])])
    residual: list[torch.Tensor] = []
    handle = net.memory_projection.register_forward_hook(
        lambda _module, _args, result: residual.append(result[0, 0]))
    try:
        adapted, next_state = net._remember(h[:, None], before, context=controls[:, None])
    finally:
        handle.remove()
    h = adapted[:, 0]
    out.extend([("memory_state", next_state[0]), ("memory_residual", residual[0]), ("memory_output", h[0])])
    logits = net.pi(h)[0]
    allowed = torch.from_numpy(np.asarray(mask, bool)[:net.n_actions]).to(logits.device)
    logits = logits.masked_fill(~allowed, _masked(logits.dtype))
    probs = torch.softmax(logits, 0)
    aux = (array(torch.sigmoid(net.aux(h)[0])))
    value = net.v(h)[0, 0]
    value = torch.tanh(value)
    return Thought([(k, array(v)) for k, v in out], array(probs), float(value), own, aux, attention, opponents,
                   next_state.detach().clone())


def trace(net: PolicyNet, maps: np.ndarray, scalars: np.ndarray, mask: np.ndarray,
          state: torch.Tensor | None = None, context: torch.Tensor | None = None):
    """(layers, probs, value) of one decision: every stage's output in forward order, the move odds, the value."""
    t = think(net, maps, scalars, mask, state=state, context=context)
    return t.layers, t.probs, t.value


def layer_table(net: PolicyNet) -> list[dict]:
    """Every stage the page draws, in frame order."""
    blank = np.zeros((2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
    outs = trace(net, blank, np.zeros(P.N_SCALARS, np.float32), np.ones(P.N_ACTIONS, bool))[0]
    names, shapes = [k for k, _ in outs], dict(outs)
    layers = []
    for m in input_maps(net):
        mid, label, planes, group = MAP_IDS[m], MAP_LABELS[m], MAP_PLANE_NAMES[m], MAP_GROUPS[m]
        layers.append({"id": mid, "group": group, "kind": "input", "shape": [P.GRID, P.GRID],
                       "label": label, "name": label, "planes": planes})
    indices = torch.arange(P.N_SCALARS)[None]
    scalar_indices = net.scalar_input(indices).flatten().tolist()
    opponent_input = net.opponent_input(indices)
    opponent_indices = opponent_input.flatten().tolist()
    names_by_index = scalar_names()
    layers.append({"id": "scalars_in", "group": "scalars", "kind": "input", "shape": [len(scalar_indices)],
                   "label": f"{len(scalar_indices)} exact numbers", "name": "exact numbers",
                   "names": [names_by_index[i] for i in scalar_indices], "scalar_indices": scalar_indices})
    layers.append({"id": "opponents_in", "group": "opponents", "kind": "input", "shape": [len(opponent_indices)],
                   "label": f"{opponent_input.shape[1]} opponents × {opponent_input.shape[2]} measurements",
                   "name": "nearest opponents", "names": [names_by_index[i] for i in opponent_indices],
                   "scalar_indices": opponent_indices})
    convs: dict[str, int] = {}
    for name in names:
        shape = list(shapes[name].shape)
        group = name.split("_")[0] if not name.startswith("trunk") else "trunk"
        word = GROUP_WORD[group]
        if len(shape) == 3:
            kind = "conv"
            if "_a" in name:  # attention output: the same cells after reading their context
                label = f"{shape[0]} channels · {shape[1]}×{shape[2]}"
                human = f"{word} · attention {name.split('_a')[1]}"
            else:
                convs[group] = convs.get(group, 0) + 1
                k = int(name.rsplit("_c", 1)[1]) if "_c" in name and name.rsplit("_c", 1)[1].isdigit() else convs[group]
                label = f"{shape[0]} filters · {shape[1]}×{shape[2]}"
                if name.endswith("_pre"):
                    stage = "residual at 32×32"
                elif name.endswith("_reduce"):
                    stage = "reduce to 16×16"
                elif name.endswith("_condition"):
                    stage = "arena-conditioned features"
                elif shape[1] < P.GRID // 4:
                    stage = (f"down to {shape[1]}×{shape[2]}")
                else:
                    stage = "stem" if k == 1 else f"block {k - 1}"
                human = f"{word} · {stage}"
        else:
            label = f"{shape[0]} neurons"
            human = {"scalars_h1": "numbers · layer 1", "scalars_h2": "numbers · layer 2",
                     "trunk1": "trunk · layer 1", "trunk2": "trunk · layer 2",
                     "nearby_modulation": "nearby view · arena modulation",
                     "close_modulation": "close view · arena modulation",
                     "opponents_h1": "opponents · shared encoder 1", "opponents_h2": "opponents · shared encoder 2",
                     "opponents_query": "opponents · tactical query", "opponents_weights": "opponents · routing weights",
                     "opponents_pool": "opponents · weighted summary",
                     "memory_in": "memory · before this decision", "memory_context": "memory · previous controls and time",
                     "memory_state": "memory · GRU state", "memory_residual": "memory · learned correction",
                     "memory_output": "memory · adapted combat features"}.get(name, f"{word} · summary")
            kind = "dense"
        layers.append({"id": name, "group": group, "kind": kind, "shape": shape, "label": label, "name": human})
    layers.append({"id": "policy", "group": "head", "kind": "output", "shape": [net.n_actions],
                   "label": "move", "name": "move", "names": MOVES[:net.n_actions]})
    layers.append({"id": "value", "group": "head", "kind": "output", "shape": [1], "label": "value", "name": "value"})
    return layers


def layer_of(param: str, net: PolicyNet) -> str | None:
    """The page's stage a parameter belongs to (None for what the page does not draw)."""
    p = param.split(".")
    if p[0] == "towers":
        group, n, part = net.tower_names[int(p[1])], len(net.towers[int(p[1])].res), p[2]
        if part == "stem":
            return f"{group}_c1"
        if part == "res":
            return f"{group}_c{int(p[3]) + 2}"
        if part in ("pre", "reduce"):
            return f"{group}_{part}"
        if part == "pos":
            return f"{group}_a1"
        if part == "attn":
            return f"{group}_a{int(p[3]) + 1}"
        if part in ("down0", "down1"):
            return f"{group}_c{n + 2 + (part == 'down1')}"
        if part == "fc":
            return f"{group}_fc"
        return None
    elif p[0] == "conditioning":
        return f"{net.tower_names[int(p[1])]}_modulation"
    elif p[0] == "opponents":
        return {"0": "opponents_h1", "2": "opponents_h2"}.get(p[1])
    elif p[0] == "opponent_query":
        return "opponents_query"
    elif p[0] == "memory_gru":
        return "memory_state"
    elif p[0] == "memory_projection":
        return "memory_residual"
    head = ".".join(p[:2]) if p[0] in ("scalars", "trunk") else p[0]
    return {"scalars.0": "scalars_h1", "scalars.2": "scalars_h2", "trunk.0": "trunk1", "trunk.2": "trunk2",
            "pi": "policy", "v": "value", "aux": "aux", "own": "own"}.get(head)


def learned(net: PolicyNet, new: dict, old: dict) -> dict[str, float]:
    """How much each stage's weights moved since OLD, relative to their size, scaled so the stage that
    changed most is 1."""
    acc: dict[str, float] = {}
    for name, w in new.items():
        if name not in old or not w.is_floating_point() or w.ndim < 2:
            continue  # biases and norms are not what a stage is made of
        if w.shape != old[name].shape:
            continue  # an input-layout extension is not a measured learning update
        lid = layer_of(name, net)
        if lid is None:
            continue
        w, o = w.float(), old[name].float()
        d = (w - o).abs().mean().item() / (w.abs().mean().item() + 1e-8)
        acc[lid] = max(acc.get(lid, 0.0), d)
    top = max(acc.values(), default=0.0)
    return {k: round(v / top, 3) for k, v in acc.items() if v > 0} if top > 0 else {}


def _u8(x: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.asarray(x, np.float32) * 255 + 0.5, 0, 255).astype(np.uint8).tobytes()).decode()


def attention_payload(att: list[np.ndarray], cell: int | None, global_map: np.ndarray) -> dict:
    """What the page paints: where one cell looks in every attention layer, head by head (the hovered
    cell, else the cell my head is in), and how much each cell is looked at over all."""
    side = math.isqrt(att[0].shape[-1])
    if cell is None or not 0 <= cell < side * side:
        patch = P.GRID // side
        heads = np.argwhere(global_map & (1 << GLOBAL_PLANES.index("own head")))
        cell = (int(heads[0][0]) // patch * side + int(heads[0][1]) // patch) if len(heads) else side * side // 2 + side // 2
    rows = np.stack([a[:, cell] for a in att])  # (layers, heads, cells)
    rows = rows / np.maximum(rows.max(-1, keepdims=True), 1e-9)
    got = np.stack([a.mean((0, 1)) for a in att])  # (layers, cells): attention received
    got = got / np.maximum(got.max(-1, keepdims=True), 1e-9)
    return {"cell": int(cell), "side": side, "heads": int(att[0].shape[0]), "rows": _u8(rows), "got": _u8(got)}


class Scales:
    """Per-layer brightness scale: quick to rise, slow to fall, so a layer neither saturates nor fades."""

    def __init__(self):
        self.s: dict[str, float] = {}

    def __call__(self, name: str, a: np.ndarray) -> np.ndarray:
        a = a.reshape(-1)
        p = float(np.percentile(np.abs(a), 99.5)) if a.size > 64 else float(np.abs(a).max())
        s = self.s.get(name, 0.0)
        s = p if p > s else 0.97 * s + 0.03 * p
        self.s[name] = s = max(s, 1e-3)
        return a / s


# ------------------------------------------------------------------------------------------ web page

class Hub:
    """Fans server-sent events out to every open page and holds the page's controls (pause, step,
    speed). Slow pages drop frames, never the topology or the control state."""

    def __init__(self, speed: float, *, source: str = "inspection", selection: str = "sampled",
                 feed: TrainingFeed | None = None, benchmarks=None):
        self.session = f"{time.time_ns():x}"
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        self.sticky: dict[str, bytes] = {}  # a new page gets topology, metrics and the latest decision
        self.paused = False
        self.steps = 0  # decisions still to let through while paused
        self.speed = speed
        self.source = source
        self.selection = selection
        self.arena = 0
        self.n_arenas = None
        self.source_revision = 0
        self.feed = feed
        self.benchmarks = benchmarks
        self.training_status = {}
        self.cell = None  # the arena cell whose attention the page wants (None: the cell my head is in)
        self.inspection: dict | None = None
        self.probe_requested: dict | None = None
        self.state_msg = self._state()

    # ------------------------------------------------------------------ controls
    def _state(self) -> bytes:
        return f"event: state\ndata: {json.dumps(self.state())}\n\n".encode()

    def state(self) -> dict:
        return {"paused": self.paused, "speed": self.speed, "source": self.source,
                "selection": self.selection, "arena": self.arena}

    def control(self, body: dict) -> None:
        with self.lock:
            source = body.get("source", self.source)
            selection = body.get("selection", self.selection)
            if source not in ("training", "inspection"):
                raise ValueError("Choose live training or policy inspection.")
            if selection not in ("sampled", "greedy"):
                raise ValueError("Choose sampled or highest-probability policy moves.")
            arena = body.get("arena", self.arena)
            if (not isinstance(arena, int) or isinstance(arena, bool) or arena < 0
                    or arena >= (self.n_arenas or 4096)):
                raise ValueError("Choose an available training arena.")
            speed = float(body.get("speed", self.speed))
            if not math.isfinite(speed):
                raise ValueError("Match speed must be finite.")
            if arena != self.arena and self.feed is not None:
                self.feed.request_arena(arena)
            if source != self.source or arena != self.arena:
                self.source_revision += 1
                self.paused = False
                self.steps = 0
                self.inspection = self.probe_requested = None
                self.sticky.pop("frame", None)
                self.sticky.pop("training_frame", None)
            self.source, self.selection, self.arena = source, selection, arena
            if "paused" in body:
                self.paused = bool(body["paused"])
                if not self.paused:
                    self.steps = 0
            if body.get("step"):
                self.paused = True
                self.steps += int(body["step"])
            if "speed" in body:
                self.speed = min(16.0, max(0.05, speed))
            if "cell" in body:
                self.cell = int(body["cell"]) if body["cell"] is not None else None
            self.state_msg = self._state()
        self._send(self.state_msg)

    def wait_if_paused(self, meanwhile) -> bool:
        """Block while paused with no step pending, calling MEANWHILE every 20 ms; say whether it waited."""
        waited = False
        while self.source == "inspection" and self.paused and self.steps <= 0:
            waited = True
            time.sleep(0.02)
            meanwhile()
        return waited

    def frame_shown(self) -> None:
        """A decision reached the page: when stepping, that was the step."""
        with self.lock:
            if self.paused and self.steps > 0:
                self.steps -= 1

    def request_probe(self, body: dict) -> None:
        with self.lock:
            if self.source != "inspection":
                raise ValueError("Input tests are available in the separate policy inspection match.")
            snapshot = self.inspection
            if not self.paused:
                raise ValueError("Pause the match before testing inputs.")
            if snapshot is None or int(body.get("decision", -1)) != snapshot["decision"]:
                raise ValueError("Input tests require the latest decision. Return to Live and pause.")
            if int(body.get("update", -1)) != snapshot["update"]:
                raise ValueError("The checkpoint changed. Step to a decision from the new weights.")
            self.probe_requested = snapshot

    # ------------------------------------------------------------------ events
    def publish(self, event: str, data: str) -> None:
        msg = f"event: {event}\ndata: {data}\n\n".encode()
        if event in ("topology", "metrics", "frame", "training_frame", "training_status", "benchmark"):
            with self.lock:
                self.sticky[event] = msg
        self._send(msg)

    def _send(self, msg: bytes) -> None:
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            if msg.startswith((b"event: frame\n", b"event: training_frame\n")):
                # A slow browser should jump to the newest actual sample, not play a queue
                # of old game positions. Preserve control/status events already waiting.
                prefix = msg.split(b"\n", 1)[0] + b"\n"
                with q.mutex:
                    replaced = False
                    for i, pending in enumerate(q.queue):
                        if pending.startswith(prefix):
                            del q.queue[i]
                            q.queue.append(msg)
                            replaced = True
                            break
                    if not replaced and (q.maxsize <= 0 or len(q.queue) < q.maxsize):
                        q.queue.append(msg)
                        q.unfinished_tasks += 1
                    q.not_empty.notify()
                continue
            try:
                q.put_nowait(msg)
            except queue.Full:
                if msg.startswith((b"event: frame", b"event: training_frame")):
                    continue  # the page is behind; it gets the next frame
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
                q.put_nowait(msg)

    def handler(hub):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                route = self.path.split("?")[0]
                page = PAGE.with_name("live.html") if hub.source == "training" else PAGE
                assets = {"/": (page, "text/html"), "/index.html": (page, "text/html"),
                          "/live.css": (PAGE.with_name("live.css"), "text/css"),
                          "/live.js": (PAGE.with_name("live.js"), "text/javascript"),
                          "/arena3d.js": (PAGE.with_name("arena3d.js"), "text/javascript"),
                          "/brain.css": (PAGE.with_name("brain.css"), "text/css"),
                          "/brain.js": (PAGE.with_name("brain.js"), "text/javascript"),
                          "/render.js": (PAGE.with_name("render.js"), "text/javascript"),
                          "/architecture3d.js": (PAGE.with_name("architecture3d.js"), "text/javascript"),
                          "/vendor/three.module.js": (PAGE.parent / "vendor/three.module.js", "text/javascript"),
                          "/vendor/three.core.js": (PAGE.parent / "vendor/three.core.js", "text/javascript"),
                          "/vendor/OrbitControls.js": (PAGE.parent / "vendor/OrbitControls.js", "text/javascript")}
                if route in assets:
                    path, mime = assets[route]
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime + "; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/status":
                    body = json.dumps({**hub.state(), "training": hub.training_status}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    q: queue.Queue = queue.Queue(maxsize=6)
                    with hub.lock:
                        hub.clients.add(q)
                        first = hub.state_msg + b"".join(hub.sticky.values())
                    try:
                        self.wfile.write(first)
                        self.wfile.flush()
                        while True:
                            try:
                                msg = q.get(timeout=10)
                            except queue.Empty:
                                msg = b": ping\n\n"
                            self.wfile.write(msg)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        with hub.lock:
                            hub.clients.discard(q)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path not in ("/control", "/probe", "/benchmark"):
                    self.send_error(404)
                    return
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("Expected a JSON object.")
                    if self.path == "/benchmark":
                        if hub.benchmarks is None:
                            raise ValueError("Benchmarking is unavailable for this server.")
                        if body.get("action", "start") == "cancel":
                            result = hub.benchmarks.cancel()
                        elif body.get("action", "start") == "start":
                            result = hub.benchmarks.start(
                                rounds=body.get("rounds", 20), time_budget_s=body.get("time_budget_s", 60),
                                size=body.get("size", -3), sample=body.get("sample", False))
                        else:
                            raise ValueError("Unknown benchmark action.")
                        hub.publish("benchmark", json.dumps(result))
                    elif self.path == "/probe":
                        hub.request_probe(body)
                    else:
                        hub.control(body)
                except (ValueError, TypeError, OSError) as exc:
                    response = json.dumps({"error": str(exc)}).encode()
                    self.send_response(409 if self.path == "/probe" else 400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                self.send_response(204)
                self.end_headers()
        return Handler


def serve(hub: Hub, port: int) -> tuple[ThreadingHTTPServer, int]:
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), hub.handler())
        except OSError:
            continue
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, int(srv.server_address[1])
    raise OSError(f"no free port from {port} to {port + 19}")


# ----------------------------------------------------------------------------------------------- run

class Metrics:
    """The trainer's metrics.jsonl (one row per update), re-read whenever it grows."""

    KEEP = 600  # updates of history the page gets for its charts
    KEYS = ("update", "steps", "samples", "trained", "sps", "lr", "pg", "vf", "ent", "kl", "clipfrac", "grad_norm",
            "value_explained", "ep_len", "policy_lag", "collect_s", "learn_s", "time",
            "aux", "safe_kill", "territory", "danger", "search_kl", "evals_s", "replay", "games", "own")

    def __init__(self, path: Path):
        self.path = path
        self.size = -1
        self.rows: list[dict] = []
        self.saved_config = {}
        self.config_mtime = None

    def configuration(self) -> dict:
        progress = read_record(self.path.parent / "progress.json")
        if progress.get("config"):
            return {"values": progress["config"], "source": "trainer", "time": progress.get("time")}
        checkpoint = self.path.parent / "latest.pt"
        try:
            stamp = checkpoint.stat().st_mtime_ns
            if stamp != self.config_mtime:
                ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.saved_config = ck.get("config", {})
                self.config_mtime = stamp
        except (OSError, RuntimeError, ValueError):
            pass
        return {"values": self.saved_config, "source": "saved checkpoint"}

    def poll(self) -> bool:
        """Re-read the tail of the file if it changed; say whether there is anything to show."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return False
        if size == self.size:
            return False
        self.size = size
        tail = 300_000
        try:
            with open(self.path, "rb") as f:
                f.seek(max(0, size - tail))
                lines = f.read().decode(errors="replace").split("\n")
        except OSError:
            return False
        rows = []
        for line in lines[1 if size > tail else 0:]:  # the first line of a cut tail is partial
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        self.rows = rows[-self.KEEP:]
        return bool(self.rows)

    def payload(self) -> str:
        last = self.rows[-1]
        wins = sorted(k for k in last if k.startswith("win_"))
        series = {k: [r.get(k) for r in self.rows] for k in (*self.KEYS, *wins) if k in last}
        # the run's own rhythm, so the page can tell "training" from "stopped" at any pace
        times = [r["time"] for r in self.rows[-8:] if "time" in r]
        gaps = sorted(b - a for a, b in zip(times, times[1:]))
        interval = gaps[len(gaps) // 2] if gaps else 60.0
        return json.dumps({"last": last, "series": series, "interval": interval,
                           "configuration": self.configuration()}, separators=(",", ":"))


class Weights:
    """The network, reloaded whenever its checkpoint file changes."""

    def __init__(self, path: Path):
        self.path = path
        self.net = make_net().eval()
        self.mtime = 0.0
        self.update = 0
        self.changed_at = 0.0
        self.state: dict | None = None

    def poll(self) -> dict | None:
        """Load the checkpoint if it changed; return the previous weights then (None the first time
        and when nothing changed)."""
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            return None
        if m == self.mtime:
            return None
        try:
            ck = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception:
            return None  # being replaced right now; next time
        old = self.state
        net_version(ck["model"])
        try:
            self.net.load_state_dict(ck["model"])
        except RuntimeError:
            raise SystemExit(f"{self.path} has weights incompatible with the V4 architecture") from None
        self.state = {k: v.clone() for k, v in self.net.state_dict().items()}
        self.update = int(ck.get("update", 0))
        if self.mtime:
            self.changed_at = time.time()
        self.mtime = m
        return old if old is not None else {}


def default_checkpoint(project: Path = PROJECT) -> Path:
    return project / "runs/blacklight/latest.pt"


def inspect_match(args, hub, net, weights, port, follow_training) -> None:
    """Run the separately labelled policy match only while that source is selected."""
    arena = ArenaConfig(slots=args.ais, builtin_ais=args.bots, size_factor=args.size, walls_length=args.walls, extra={
        "NEURAL_SPECTATE": "1", "NEURAL_END_ROUND_WITHOUT_NEURAL": "0", "NEURAL_CONTROL_AFTER_ROUND": "1"})
    scales = Scales()
    m = Match(args.walls)
    with EnginePool([arena], workdir=PROJECT / "runtime" / f"brain-{port}",
                    base_port=47950 + (port - 8765) % 1000, log_engines=True) as pool:
        conn = pool.conns[0]
        conn.setblocking(True)
        clock = None
        speed = hub.speed
        last_frame = last_poll = 0.0
        history = PolicyHistory(net)
        decision = 0
        try:
            while hub.source == "inspection":
                now = time.monotonic()
                if now - last_poll > 1.0:
                    last_poll = now
                    follow_training()
                mtype, payload = pool._read_msg(conn)
                if mtype != P.MSG_WORLD:
                    continue
                rid = m.round_id
                step_follows = m.update(payload)
                if m.round_id != rid or clock is None:
                    clock = (time.monotonic(), m.time)
                acts: list[int] = []
                if step_follows:
                    mtype, payload = pool._read_msg(conn)
                    if mtype == P.MSG_STEP:
                        step = _parse_step(0, payload)
                        acts = [0] * len(step.slots)
                        identities = {cycle.slot: cycle.pid for cycle in m.cycles.values()
                                      if cycle.slot != 255 and cycle.alive}
                        duplicate = history.begin(step, identities=identities, generation=weights.mtime)
                        if duplicate is not None:
                            acts = duplicate
                        for k, s in enumerate(step.slots):
                            if duplicate is not None or not s.needs_action:
                                continue
                            snapshot = history.snapshot(step, k)
                            seen, mask = snapshot.maps, snapshot.mask
                            th = think(net, seen, s.scalars, mask,
                                       state=snapshot.state, context=snapshot.context)
                            probs = th.probs
                            a = int(probs.argmax()) if hub.selection == "greedy" else int(np.random.choice(len(probs), p=probs / probs.sum()))
                            acts[k] = a
                            history.commit(step, k, a, th.state)
                            # every decision counts while paused (stepping must not skip one)
                            if k == 0 and (hub.paused or time.monotonic() - last_frame >= 1.0 / MAX_FPS):
                                last_frame = time.monotonic()
                                decision += 1
                                stats = activation_stats(th.layers)
                                acts_q = np.concatenate([np.clip(scales(n, v) * 0.5 + 0.5 if stats[n]["signed"]
                                                                else scales(n, v), 0, 1) for n, v in th.layers])
                                for name, stat in stats.items():
                                    stat["scale"] = round(scales.s[name], 5)
                                me = next((c for c in m.cycles.values() if c.slot == 0), None)
                                frame = {
                                    "source": "inspection", "selection": hub.selection,
                                    "opponent_names": ["Built-in AI"] * args.bots,
                                    "decision": decision, "stats": stats,
                                    "update": weights.update, "trained_ago": (time.time() - weights.changed_at)
                                    if weights.changed_at else None,
                                    "acts": _u8(acts_q),
                                    # All four maps are the inputs captured for this decision.
                                    "maps": {MAP_IDS[i]: base64.b64encode(np.ascontiguousarray(s.maps[i]).tobytes()).decode()
                                             for i in range(P.N_MAPS)},
                                    "scalars": _u8(np.tanh(np.abs(s.scalars))),
                                    "numbers": [float(x) for x in s.scalars],
                                    "probs": [float(x) for x in probs], "mask": mask.tolist(),
                                    "action": a, "value": th.value,
                                    "memory": {"context": snapshot.context[0].float().cpu().tolist(),
                                               "before_norm": float(snapshot.state.float().norm()),
                                               "after_norm": float(th.state.float().norm()),
                                               "gap_resets": history.memory_gap_resets},
                                    "round": m.round_id, "time": round(m.time, 2), "bounds": m.bounds,
                                    "walls": args.walls,
                                    "me": me.pid if me else None,
                                    "heading": [round(me.dx, 3), round(me.dy, 3)] if me else None,
                                    "cycles": [[c.pid, round(c.x, 1), round(c.y, 1), int(c.alive), c.color, c.slot]
                                               for c in m.cycles.values()],
                                    "result": m.last_result,
                                }
                                trails = []
                                for c in m.cycles.values():
                                    if not c.alive and c.died_at is not None and m.time - c.died_at > WALLS_STAY_UP:
                                        continue
                                    points = []
                                    for x, y in c.trail:
                                        point = [round(x, 2), round(y, 2)]
                                        if len(points) > 1 and ((points[-2][0] == points[-1][0] == point[0])
                                                               or (points[-2][1] == points[-1][1] == point[1])):
                                            points[-1] = point
                                        else:
                                            points.append(point)
                                    trails.append({"id": c.pid, "name": c.label, "color": c.color, "alive": c.alive,
                                                   "points": points, "heading": [c.dx, c.dy]})
                                frame["trails"] = trails
                                if th.own is not None:
                                    frame["own"] = _u8(th.own[1:])  # me, then an enemy; nobody is the rest
                                frame["previous"] = {
                                    MAP_IDS[i]: base64.b64encode(np.ascontiguousarray(seen[P.N_MAPS + i]).tobytes()).decode()
                                    for i in range(P.N_MAPS)}
                                if th.aux is not None:
                                    frame["aux"] = [round(float(x), 3) for x in th.aux]
                                if th.opponents is not None:
                                    frame["opponents"] = [float(x) for x in th.opponents]
                                if th.attention:
                                    frame["attn"] = attention_payload(th.attention, hub.cell, s.maps[2])
                                with hub.lock:
                                    hub.inspection = {"decision": decision, "update": weights.update, "maps": seen.copy(),
                                                      "scalars": s.scalars.copy(), "mask": mask.copy(), "probs": probs.copy(),
                                                      "state": snapshot.state.detach().clone(),
                                                      "context": snapshot.context.detach().clone(),
                                                      "generation": weights.mtime,
                                                      "attention": th.attention, "attention_cell": hub.cell}
                                hub.publish("frame", json.dumps(frame, separators=(",", ":")))
                                hub.frame_shown()
                        history.finish(step, acts)
                # paused: hold the match here (the engine waits for its actions); a step lets it run
                # until the next decision has been shown
                if hub.wait_if_paused(follow_training) or hub.speed != speed:
                    speed = hub.speed
                    clock = (time.monotonic(), m.time)
                if hub.source != "inspection":
                    break
                if not m.over:
                    delay = clock[0] + (m.time - clock[1]) / speed - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                body = bytes([len(acts)]) + bytes(acts)
                conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, len(body)) + body)
        except KeyboardInterrupt:
            raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ais", type=int, default=1, help="cycles driven by the network (the brain shown is AI 1's)")
    ap.add_argument("--bots", type=int, default=1, help="built-in AI opponents")
    ap.add_argument("--size", type=float, default=-2, help="arena SIZE_FACTOR (-3 small, 0 full-size)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    ap.add_argument("--walls", type=float, default=600, help="trail length in metres (-1 = endless)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="weights to show (default: runs/blacklight/latest.pt)")
    ap.add_argument("--source", choices=("training", "inspection"), default="training",
                    help="live trainer arena (default) or a separate policy inspection match")
    ap.add_argument("--greedy", action="store_true", help="choose the top move in policy inspection")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
    args = ap.parse_args()
    # one decision at a time needs no thread pool; a wide one only fights the trainer for the CPU
    torch.set_num_threads(2)

    path = args.checkpoint or default_checkpoint()
    weights = Weights(path)
    if weights.poll() is None:
        raise SystemExit(f"no checkpoint at {path}")
    run_dir = path.parent.parent if path.parent.name == "pool" else path.parent
    metrics = Metrics(run_dir / "metrics.jsonl")
    net = weights.net
    layers = layer_table(net)
    architecture = describe(net, layers, layer_of)
    from .benchmarks import BenchmarkManager
    feed = TrainingFeed(run_dir)
    feed.request_arena(0)
    benchmarks = BenchmarkManager(run_dir, path)
    hub = Hub(args.speed, source=args.source, selection="greedy" if args.greedy else "sampled",
              feed=feed, benchmarks=benchmarks)
    last_follow = 0.0
    last_benchmark = None

    def publish_topology(old: dict | None) -> None:
        hub.publish("topology", json.dumps({
            "layers": layers, "version": net.version, "reads": [MAP_IDS[m] for m in input_maps(net)],
            "observation_version": int(net.observation_version.item()),
            "actions": MOVES[:net.n_actions],
            "session": hub.session,
            "architecture": architecture,
            "learned": learned(net, weights.state, old) if old else {},
            "update": weights.update, "checkpoint": str(path.relative_to(PROJECT) if path.is_relative_to(PROJECT)
                                                        else path)}))

    def follow_training() -> None:
        nonlocal last_follow, last_benchmark
        now = time.monotonic()
        if now - last_follow < 1.0:
            return
        last_follow = now
        feed.poll(hub)
        benchmark = json.dumps(benchmarks.poll(), separators=(",", ":"))
        if benchmark != last_benchmark:
            hub.publish("benchmark", benchmark)
            last_benchmark = benchmark
        old = weights.poll()
        if old is not None:
            publish_topology(old)
        if metrics.poll():
            hub.publish("metrics", metrics.payload())
        with hub.lock:
            snapshot, hub.probe_requested = hub.probe_requested, None
        if snapshot is not None:
            if (snapshot["update"] != weights.update
                    or snapshot.get("generation", weights.mtime) != weights.mtime):
                result = {"error": "Weights changed. Step to a decision from the new checkpoint."}
            else:
                result = input_sensitivity(net, snapshot["maps"], snapshot["scalars"], snapshot["mask"],
                                           state=snapshot.get("state"), context=snapshot.get("context"))
                if not np.allclose(result["baseline"], snapshot["probs"], atol=1e-5):
                    result = {"error": "The decision no longer matches these weights. Step and retry."}
            result.update(decision=snapshot["decision"], update=snapshot["update"])
            hub.publish("probe", json.dumps(result, separators=(",", ":")))
        snapshot = hub.inspection
        if (hub.paused and snapshot and snapshot["attention"]
                and hub.cell != snapshot["attention_cell"]):
            attn = attention_payload(snapshot["attention"], hub.cell, snapshot["maps"][2])
            snapshot["attention_cell"] = hub.cell
            hub.publish("attention", json.dumps({"decision": snapshot["decision"], "attn": attn}))

    publish_topology(None)
    if metrics.poll():
        hub.publish("metrics", metrics.payload())
    srv, port = serve(hub, args.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"brain: {url}  (update {weights.update}, {path.name}); Ctrl-C quits", flush=True)
    if not args.no_open:
        webbrowser.open(url)

    try:
        while True:
            follow_training()
            if hub.source == "training":
                feed.poll(hub)
                time.sleep(0.1)
            else:
                inspect_match(args, hub, net, weights, port, follow_training)
    except KeyboardInterrupt:
        pass
    finally:
        benchmarks.close()
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
