"""Watch the network think: what it sees, what it believes and what it does, live in the browser.

    ./trainctl brain                    # the AI against the game's best bot
    ./trainctl brain --ais 2 --bots 0   # two copies of the AI; the brain shown is AI 1's
    ./trainctl brain --speed 0.5        # half speed

A real match runs in a headless engine, as in ./trainctl show. Every time AI 1 decides, the page
(http://127.0.0.1:8765) gets the decision: the arena with the network's beliefs painted on it (who
will hold each cell two seconds from now, where its attention goes, what its rays feel), the odds
of each move, its win chance and danger reading, the four maps it sees, and every layer's output
drawn as the pictures they are. Space pauses the match, the right arrow steps it one decision at
a time, [ and ] change the speed. While training runs, the page follows the newest weights
(runs/v3/actor.pt) and shows which layers each update changed most; until v3 has weights it shows
the trained v1 brain (runs/main). Ctrl-C quits.
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
from .model import TOWER_NAMES, PolicyNet, make_net, net_version
from .show import Match
from .train import MASK_TABLE

PROJECT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("web") / "brain.html"
MAX_FPS = 20  # decisions sent to the page per second of wall time
LOCAL_PLANES = ["walls", "own trail", "enemy trails", "team trails", "enemy heads", "enemy paths", "outside",
                "team heads"]
GLOBAL_PLANES = ["walls", "own trail", "enemy trails", "own head", "enemy heads", "outside", "team trails",
                 "team heads"]
MOVES = ["Straight", "Left", "Right", "Brake"]


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
    n += ["unused"] * (P.N_SCALARS - len(n))
    return n


# ------------------------------------------------------------------------------------------- network

MAP_IDS = ("local_in", "close_in", "global_in", "territory_in")
MAP_LABELS = ("what it sees nearby", "up close", "the whole arena", "who gets there first")
MAP_PLANE_NAMES = (LOCAL_PLANES, LOCAL_PLANES, GLOBAL_PLANES,
                   ["mine first", "enemy first", "d1", "d2", "d4", "d8", "d16", "d32"])
MAP_GROUPS = ("local", "close", "global", "territory")  # the page's colour band of each input map
GROUP_WORD = {"player": "player view", "arena": "arena view", "local": "nearby", "global": "arena",
              "scalars": "numbers", "trunk": "trunk"}
TOWER_GROUPS = TOWER_NAMES  # one conv tower per frame of reference: ("player", "arena")
# v1 (runs/main): one plain conv tower per map, fed only the 2 m view and the minimap
V1_TOWERS = (("local", "local", "unpack_local", 0), ("global", "globl", "unpack_global", 2))  # group, attr, unpack, map


def input_maps(net: nn.Module) -> tuple[int, ...]:
    """Indices of the maps NET reads (v1 ignores the close view and territory)."""
    return tuple(m for *_, m in V1_TOWERS) if net.version == 1 else tuple(range(P.N_MAPS))


def stacked(maps: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """(2*N_MAPS, G, G): this decision's maps, then the previous decision's (or these again at a spawn)."""
    return np.concatenate([maps, maps if prev is None else prev])


@dataclass
class Thought:
    """One decision, with everything the page shows."""
    layers: list[tuple[str, np.ndarray]]  # (name, activations) after each stage, in forward order
    probs: np.ndarray  # the odds of each move
    value: float
    own: np.ndarray | None = None  # (3, 16, 16): nobody / me / an enemy holds the cell two seconds on (v2+)
    aux: np.ndarray | None = None  # (2,): my territory share in two seconds, dead within two seconds (v2+)
    attention: list[np.ndarray] = field(default_factory=list)  # per arena attention layer: (heads, cells, cells) (v3)


def attend(layer: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnLayer.forward, also handing back the attention weights (B, heads, N, N)."""
    B, N, C = x.shape
    h = layer.heads
    q, k, v = layer.qkv(layer.ln1(x)).view(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
    att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(C // h), -1)
    x = x + layer.proj((att @ v).transpose(1, 2).reshape(B, N, C))
    return x + layer.fc2(torch.nn.functional.gelu(layer.fc1(layer.ln2(x)))), att


@torch.no_grad()
def think(net: PolicyNet, maps: np.ndarray, scalars: np.ndarray, mask: np.ndarray) -> Thought:
    """Run one decision through NET and keep every stage's output and every head's prediction.

    MAPS: (2*N_MAPS, GRID, GRID) uint8 (current then previous frame), SCALARS: (N_SCALARS,) float32,
    MASK: (N_ACTIONS,) bool."""
    out: list[tuple[str, torch.Tensor]] = []
    attention: list[np.ndarray] = []
    own = None
    x_in = torch.from_numpy(np.array(maps, np.uint8))[None]

    def mlp(prefix: str, seq: nn.Sequential, h: torch.Tensor) -> torch.Tensor:
        n = 0
        for m in seq:
            h = m(h)
            if isinstance(m, nn.ReLU):
                n += 1
                out.append((f"{prefix}{n}", h[0]))
        return h

    def plain(group: str, tower, unpack, m: int) -> torch.Tensor:  # a v1 tower: convs, then fc
        x = unpack(x_in[:, m])
        n = 0
        for mod in tower.net:
            x = mod(x)
            if isinstance(mod, nn.ReLU):
                n += 1
                out.append((f"{group}_c{n}", x[0]))
        h = torch.relu(tower.fc(x.flatten(1)))
        out.append((f"{group}_fc", h[0]))
        return h

    parts = []
    if net.version == 1:
        parts = [plain(group, getattr(net, attr), getattr(net, unpack), m) for group, attr, unpack, m in V1_TOWERS]
    for t, group in enumerate(TOWER_GROUPS if net.version >= 2 else ()):
        tower = net.towers[t]
        x = torch.relu(tower.stem(net.tower_input(x_in, t)))
        out.append((f"{group}_c1", x[0]))
        n = 1
        for b in tower.res:
            x = b(x)  # a plain residual block, or v3's pooling block
            n += 1
            out.append((f"{group}_c{n}", x[0]))
        if len(getattr(tower, "attn", ())):
            # v3: the cells talk to each other; each attention layer is a stage of its own
            B, C, H, W = x.shape
            cells = x.flatten(2).transpose(1, 2) + tower.pos
            for a_i, layer in enumerate(tower.attn):
                cells, att = attend(layer, cells)
                attention.append(att[0].numpy())
                x = cells.transpose(1, 2).reshape(B, C, H, W)
                out.append((f"{group}_a{a_i + 1}", x[0]))
        if t == 1 and hasattr(net, "own"):
            own = torch.softmax(net.own(x)[0], 0).numpy()  # the arena tower's 16x16 features feed the ownership head
        x = torch.relu(tower.down0(x))
        out.append((f"{group}_c{n + 1}", x[0]))
        x = torch.relu(tower.down1(x))
        out.append((f"{group}_c{n + 2}", x[0]))
        h = torch.relu(tower.fc(x.flatten(1)))
        out.append((f"{group}_fc", h[0]))
        parts.append(h)
    hs = mlp("scalars_h", net.scalars, torch.from_numpy(np.array(scalars, np.float32))[None])
    h = mlp("trunk", net.trunk, torch.cat(parts + [hs], 1))
    logits = net.pi(h)[0].masked_fill(~torch.from_numpy(np.asarray(mask, bool)), -1e8)
    probs = torch.softmax(logits, 0)
    aux = torch.sigmoid(net.aux(h)[0]).numpy() if hasattr(net, "aux") else None
    return Thought([(k, v.numpy()) for k, v in out], probs.numpy(), float(net.v(h)[0, 0]), own, aux, attention)


def trace(net: PolicyNet, maps: np.ndarray, scalars: np.ndarray, mask: np.ndarray):
    """(layers, probs, value) of one decision: every stage's output in forward order, the move odds, the value."""
    t = think(net, maps, scalars, mask)
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
    layers.append({"id": "scalars_in", "group": "scalars", "kind": "input", "shape": [P.N_SCALARS],
                   "label": f"{P.N_SCALARS} exact numbers", "name": "exact numbers", "names": scalar_names()})
    convs: dict[str, int] = {}
    for name in names:
        shape = list(shapes[name].shape)
        group = name.split("_")[0] if not name.startswith("trunk") else "trunk"
        word = GROUP_WORD[group]
        if len(shape) == 3:
            kind = "conv"
            if "_a" in name:  # v3 attention output: the same cells, after looking at each other
                label = f"{shape[0]} channels · {shape[1]}×{shape[2]}"
                human = f"{word} · attention {name.split('_a')[1]}"
            else:
                convs[group] = convs.get(group, 0) + 1
                k = convs[group]
                label = f"{shape[0]} filters · {shape[1]}×{shape[2]}"
                if net.version == 1 or shape[1] < P.GRID // 4:
                    stage = f"conv {k}" if net.version == 1 else f"down to {shape[1]}×{shape[2]}"
                else:
                    stage = "stem" if k == 1 else f"block {k - 1}"
                human = f"{word} · {stage}"
        else:
            label = f"{shape[0]} neurons"
            human = {"scalars_h1": "numbers · layer 1", "scalars_h2": "numbers · layer 2",
                     "trunk1": "trunk · layer 1", "trunk2": "trunk · layer 2"}.get(name, f"{word} · summary")
            kind = "dense"
        layers.append({"id": name, "group": group, "kind": kind, "shape": shape, "label": label, "name": human})
    layers.append({"id": "policy", "group": "head", "kind": "output", "shape": [P.N_ACTIONS],
                   "label": "move", "name": "move", "names": MOVES})
    layers.append({"id": "value", "group": "head", "kind": "output", "shape": [1], "label": "value", "name": "value"})
    return layers


def layer_of(param: str, net: PolicyNet) -> str | None:
    """The page's stage a parameter belongs to (None for what the page does not draw)."""
    p = param.split(".")
    if net.version == 1:
        if p[0] not in ("local", "globl"):
            group = None
        else:
            group = "local" if p[0] == "local" else "global"
            if p[1] == "fc":
                return f"{group}_fc"
            tower = getattr(net, p[0])
            convs = [i for i, m in enumerate(tower.net) if isinstance(m, nn.Conv2d)]
            return f"{group}_c{convs.index(int(p[2])) + 1}" if p[1] == "net" and int(p[2]) in convs else None
    elif p[0] == "towers":
        group, n, part = TOWER_GROUPS[int(p[1])], len(net.towers[int(p[1])].res), p[2]
        if part == "stem":
            return f"{group}_c1"
        if part == "res":
            return f"{group}_c{int(p[3]) + 2}"
        if part == "pos":
            return f"{group}_a1"
        if part == "attn":
            return f"{group}_a{int(p[3]) + 1}"
        if part in ("down0", "down1"):
            return f"{group}_c{n + 2 + (part == 'down1')}"
        if part == "fc":
            return f"{group}_fc"
        return None
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
        p = float(np.percentile(a, 99.5)) if a.size > 64 else float(a.max())
        s = self.s.get(name, 0.0)
        s = p if p > s else 0.97 * s + 0.03 * p
        self.s[name] = s = max(s, 1e-3)
        return a / s


# ------------------------------------------------------------------------------------------ web page

class Hub:
    """Fans server-sent events out to every open page and holds the page's controls (pause, step,
    speed). Slow pages drop frames, never the topology or the control state."""

    def __init__(self, speed: float):
        self.clients: set[queue.Queue] = set()
        self.lock = threading.Lock()
        self.sticky: dict[str, bytes] = {}  # topology and metrics: a new page gets the latest of each
        self.paused = False
        self.steps = 0  # decisions still to let through while paused
        self.speed = speed
        self.cell = None  # the arena cell whose attention the page wants (None: the cell my head is in)
        self.state_msg = self._state()

    # ------------------------------------------------------------------ controls
    def _state(self) -> bytes:
        return f"event: state\ndata: {json.dumps({'paused': self.paused, 'speed': self.speed})}\n\n".encode()

    def control(self, body: dict) -> None:
        with self.lock:
            if "paused" in body:
                self.paused = bool(body["paused"])
                if not self.paused:
                    self.steps = 0
            if body.get("step"):
                self.paused = True
                self.steps += int(body["step"])
            if "speed" in body:
                self.speed = min(16.0, max(0.05, float(body["speed"])))
            if "cell" in body:
                self.cell = int(body["cell"]) if body["cell"] is not None else None
            self.state_msg = self._state()
        self._send(self.state_msg)

    def wait_if_paused(self, meanwhile) -> bool:
        """Block while paused with no step pending, calling MEANWHILE every 20 ms; say whether it waited."""
        waited = False
        while self.paused and self.steps <= 0:
            waited = True
            time.sleep(0.02)
            meanwhile()
        return waited

    def frame_shown(self) -> None:
        """A decision reached the page: when stepping, that was the step."""
        with self.lock:
            if self.paused and self.steps > 0:
                self.steps -= 1

    # ------------------------------------------------------------------ events
    def publish(self, event: str, data: str) -> None:
        msg = f"event: {event}\ndata: {data}\n\n".encode()
        if event in ("topology", "metrics"):
            self.sticky[event] = msg
        self._send(msg)

    def _send(self, msg: bytes) -> None:
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                if msg.startswith(b"event: frame"):
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
                if self.path.split("?")[0] in ("/", "/index.html"):   # ?hd=0 and the like are the page's business
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
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
                        first = b"".join(hub.sticky.values()) + hub.state_msg
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
                if self.path != "/control":
                    self.send_error(404)
                    return
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    hub.control(body if isinstance(body, dict) else {})
                except (ValueError, TypeError):
                    self.send_error(400)
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
        return srv, p
    raise OSError(f"no free port from {port} to {port + 19}")


# ----------------------------------------------------------------------------------------------- run

class Metrics:
    """The trainer's metrics.jsonl (one row per update), re-read whenever it grows."""

    KEEP = 600  # updates of history the page gets for its charts
    KEYS = ("update", "steps", "sps", "lr", "pg", "vf", "ent", "kl", "clipfrac", "grad_norm",
            "value_explained", "ep_len", "policy_lag", "collect_s", "learn_s", "time",
            "aux", "search_kl", "evals_s", "replay", "games", "own")

    def __init__(self, path: Path):
        self.path = path
        self.size = -1
        self.rows: list[dict] = []

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
        return json.dumps({"last": last, "series": series, "interval": interval}, separators=(",", ":"))


class Weights:
    """The network, reloaded whenever its checkpoint file changes (every update while training)."""

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
        version = net_version(ck["model"])
        if version != self.net.version:
            if old is not None:
                raise SystemExit(f"{self.path} now holds a v{version} network; restart the brain page")
            self.net = make_net(version).eval()
        try:
            self.net.load_state_dict(ck["model"])
        except RuntimeError:
            raise SystemExit(f"{self.path} holds an older v2 layout (one tower per map); the page shows the "
                             "two-tower network") from None
        self.state = {k: v.clone() for k, v in self.net.state_dict().items()}
        self.update = int(ck.get("update", 0))
        if self.mtime:
            self.changed_at = time.time()
        self.mtime = m
        return old if old is not None else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ais", type=int, default=1, help="cycles driven by the network (the brain shown is AI 1's)")
    ap.add_argument("--bots", type=int, default=1, help="built-in AI opponents")
    ap.add_argument("--size", type=float, default=-2, help="arena SIZE_FACTOR (-3 small, 0 full-size)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    ap.add_argument("--walls", type=float, default=600, help="trail length in metres (-1 = endless)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="weights to show (default: runs/v3/actor.pt, which training rewrites every update; "
                         "runs/main, the trained v1 brain, while v3 has none)")
    ap.add_argument("--greedy", action="store_true", help="always take the top move")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
    args = ap.parse_args()
    # one decision at a time needs no thread pool; a wide one only fights the trainer for the CPU
    torch.set_num_threads(2)

    path = args.checkpoint or next((PROJECT / run / f for run in ("runs/v3", "runs/main") for f in ("actor.pt", "latest.pt")
                                    if (PROJECT / run / f).exists()), PROJECT / "runs/v3/actor.pt")
    weights = Weights(path)
    if weights.poll() is None:
        raise SystemExit(f"no checkpoint at {path}")
    run_dir = path.parent.parent if path.parent.name == "pool" else path.parent
    metrics = Metrics(run_dir / "metrics.jsonl")
    net = weights.net
    layers = layer_table(net)
    hub = Hub(args.speed)

    def publish_topology(old: dict | None) -> None:
        hub.publish("topology", json.dumps({
            "layers": layers, "version": net.version, "reads": [MAP_IDS[m] for m in input_maps(net)],
            "learned": learned(net, weights.state, old) if old else {},
            "update": weights.update, "checkpoint": str(path.relative_to(PROJECT) if path.is_relative_to(PROJECT)
                                                        else path)}))

    def follow_training() -> None:
        old = weights.poll()
        if old is not None:
            publish_topology(old)
        if metrics.poll():
            hub.publish("metrics", metrics.payload())

    publish_topology(None)
    if metrics.poll():
        hub.publish("metrics", metrics.payload())
    srv, port = serve(hub, args.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"brain: {url}  (update {weights.update}, {path.name}); Ctrl-C quits", flush=True)
    if not args.no_open:
        webbrowser.open(url)

    arena = ArenaConfig(slots=args.ais, builtin_ais=args.bots, size_factor=args.size, walls_length=args.walls, extra={
        "NEURAL_SPECTATE": "1", "NEURAL_END_ROUND_WITHOUT_NEURAL": "0", "NEURAL_CONTROL_AFTER_ROUND": "1"})
    scales = Scales()
    m = Match(args.walls)
    with EnginePool([arena], workdir=PROJECT / "runtime" / "brain", base_port=47950, log_engines=True) as pool:
        conn = pool.conns[0]
        conn.setblocking(True)
        clock = None
        speed = hub.speed
        last_frame = last_poll = 0.0
        prev_maps: dict[int, np.ndarray] = {}  # slot -> maps of its previous decision
        try:
            while True:
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
                    prev_maps.clear()
                acts: list[int] = []
                if step_follows:
                    mtype, payload = pool._read_msg(conn)
                    if mtype == P.MSG_STEP:
                        step = _parse_step(0, payload)
                        acts = [0] * len(step.slots)
                        for k, s in enumerate(step.slots):
                            if not s.needs_action:
                                continue
                            mask = MASK_TABLE[s.mask] > 0.5
                            if s.flags & (P.FLAG_SPAWNED | P.FLAG_DIED):
                                prev_maps.pop(k, None)
                            th = think(net, stacked(s.maps, prev_maps.get(k)), s.scalars, mask)
                            probs = th.probs
                            prev_maps[k] = np.array(s.maps)
                            a = int(probs.argmax()) if args.greedy else int(np.random.choice(len(probs), p=probs / probs.sum()))
                            acts[k] = a
                            # every decision counts while paused (stepping must not skip one)
                            if k == 0 and (hub.paused or time.monotonic() - last_frame >= 1.0 / MAX_FPS):
                                last_frame = time.monotonic()
                                acts_q = np.concatenate([np.clip(scales(n, v), 0, 1) for n, v in th.layers])
                                me = next((c for c in m.cycles.values() if c.slot == 0), None)
                                frame = {
                                    "update": weights.update, "trained_ago": (time.time() - weights.changed_at)
                                    if weights.changed_at else None,
                                    "acts": _u8(acts_q),
                                    # all four maps, even the ones a v1 brain never reads: the page says so
                                    "maps": {MAP_IDS[i]: base64.b64encode(np.ascontiguousarray(s.maps[i]).tobytes()).decode()
                                             for i in range(P.N_MAPS)},
                                    "scalars": _u8(np.tanh(np.abs(s.scalars))),
                                    "numbers": [round(float(x), 3) for x in s.scalars],
                                    "probs": [round(float(x), 4) for x in probs], "mask": mask.tolist(),
                                    "action": a, "value": round(th.value, 3),
                                    "round": m.round_id, "time": round(m.time, 2), "bounds": m.bounds,
                                    "walls": args.walls,
                                    "me": me.pid if me else None,
                                    "heading": [round(me.dx, 3), round(me.dy, 3)] if me else None,
                                    "cycles": [[c.pid, round(c.x, 1), round(c.y, 1), int(c.alive), c.color, c.slot]
                                               for c in m.cycles.values()],
                                    "result": m.last_result,
                                }
                                if th.own is not None:
                                    frame["own"] = _u8(th.own[1:])  # me, then an enemy; nobody is the rest
                                if th.aux is not None:
                                    frame["aux"] = [round(float(x), 3) for x in th.aux]
                                if th.attention:
                                    frame["attn"] = attention_payload(th.attention, hub.cell, s.maps[2])
                                hub.publish("frame", json.dumps(frame, separators=(",", ":")))
                                hub.frame_shown()
                # paused: hold the match here (the engine waits for its actions); a step lets it run
                # until the next decision has been shown
                if hub.wait_if_paused(follow_training) or hub.speed != speed:
                    speed = hub.speed
                    clock = (time.monotonic(), m.time)
                if not m.over:
                    delay = clock[0] + (m.time - clock[1]) / speed - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                body = bytes([len(acts)]) + bytes(acts)
                conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, len(body)) + body)
        except KeyboardInterrupt:
            pass
        finally:
            srv.shutdown()


if __name__ == "__main__":
    main()
