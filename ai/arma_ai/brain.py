"""Watch the network think: every neuron of the policy network, drawn live in the browser.

    ./trainctl brain                    # the AI against the game's best bot
    ./trainctl brain --ais 2 --bots 0   # two copies of the AI; the brain shown is AI 1's
    ./trainctl brain --speed 0.5        # half speed

A real match runs in a headless engine, as in ./trainctl show. Every time AI 1 decides, the
activation of every neuron is sent to a page (http://127.0.0.1:8765) that draws the network in 3D:
one slab of points per layer, lit by how strongly each neuron fires, and the strongest connections
between layers, glowing where signal flows through them. Space pauses the match, the right arrow
steps it one decision at a time, [ and ] change the speed, and hovering a neuron says what it is.
While training runs, the page follows the newest weights (runs/v2/actor.pt) and flashes the
connections each update changed. Ctrl-C quits.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import protocol as P
from .engine import ArenaConfig, EnginePool, _parse_step
from .model import TOWER_MAPS, TOWER_NAMES, PolicyNet, make_net, net_version
from .show import Match
from .train import MASK_TABLE

PROJECT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("web") / "brain.html"
MAX_FPS = 20  # activation frames sent per second of wall time
CONV_LINES = 900  # neurons per conv layer that get lines drawn to their strongest inputs
DENSE_LINES = 600  # likewise for fully connected layers
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
GROUP_WORD = {"player": "player view", "arena": "arena view", "scalars": "numbers", "trunk": "trunk"}
TOWER_GROUPS = TOWER_NAMES  # one conv tower per frame of reference: ("player", "arena")


def stacked(maps: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    """(2*N_MAPS, G, G): this decision's maps, then the previous decision's (or these again at a spawn)."""
    return np.concatenate([maps, maps if prev is None else prev])


@torch.no_grad()
def trace(net: PolicyNet, maps: np.ndarray, scalars: np.ndarray, mask: np.ndarray):
    """Run one decision through NET and keep every layer's output.

    MAPS: (2*N_MAPS, GRID, GRID) uint8 (current then previous frame), SCALARS: (N_SCALARS,) float32,
    MASK: (N_ACTIONS,) bool. Returns (layers, probs, value): layers is a list of (name, activations)
    after each ReLU, in forward order."""
    out: list[tuple[str, torch.Tensor]] = []
    x_in = torch.from_numpy(np.array(maps, np.uint8))[None]

    def mlp(prefix: str, seq: nn.Sequential, h: torch.Tensor) -> torch.Tensor:
        n = 0
        for m in seq:
            h = m(h)
            if isinstance(m, nn.ReLU):
                n += 1
                out.append((f"{prefix}{n}", h[0]))
        return h

    parts = []
    for t, group in enumerate(TOWER_GROUPS):
        tower = net.towers[t]
        x = torch.relu(tower.stem(net.tower_input(x_in, t)))
        out.append((f"{group}_c1", x[0]))
        n = 1
        for b in tower.res:
            y = torch.relu(b.c0(x))
            x = torch.relu(x + b.c1(y))
            n += 1
            out.append((f"{group}_c{n}", x[0]))
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
    return [(k, v.numpy()) for k, v in out], probs.numpy(), float(net.v(h)[0, 0])


def layer_table(net: PolicyNet) -> list[dict]:
    """Every layer the page draws, in frame order, with where it sits in the picture."""
    blank = np.zeros((2 * P.N_MAPS, P.GRID, P.GRID), np.uint8)
    outs = trace(net, blank, np.zeros(P.N_SCALARS, np.float32), np.ones(P.N_ACTIONS, bool))[0]
    names, shapes = [k for k, _ in outs], dict(outs)
    layers = []
    for mid, label, planes, group in zip(MAP_IDS, MAP_LABELS, MAP_PLANE_NAMES, MAP_GROUPS):
        layers.append({"id": mid, "group": group, "kind": "input", "shape": [P.GRID, P.GRID], "col": 0,
                       "label": label, "name": label, "planes": planes})
    layers.append({"id": "scalars_in", "group": "scalars", "kind": "input", "shape": [P.N_SCALARS], "col": 0,
                   "label": f"{P.N_SCALARS} exact numbers", "name": "exact numbers", "names": scalar_names()})
    seen16: dict[str, int] = {}
    convs: dict[str, int] = {}
    for name in names:
        shape = list(shapes[name].shape)
        group = name.split("_")[0] if not name.startswith("trunk") else "trunk"
        word = GROUP_WORD[group]
        if len(shape) == 3:
            if shape[1] == P.GRID // 4:  # full patch resolution, one column each
                seen16[group] = seen16.get(group, 0) + 1
                col = seen16[group]
            else:
                col = 5 if shape[1] == P.GRID // 8 else 6
            convs[group] = convs.get(group, 0) + 1
            label = f"{shape[0]} filters · {shape[1]}×{shape[2]}"
            human = f"{word} · conv {convs[group]}"
            kind = "conv"
        else:
            col = {"scalars_h1": 3, "scalars_h2": 7, "trunk1": 8, "trunk2": 9}.get(name, 7)
            label = f"{shape[0]} neurons"
            human = {"scalars_h1": "numbers · layer 1", "scalars_h2": "numbers · layer 2",
                     "trunk1": "trunk · layer 1", "trunk2": "trunk · layer 2"}.get(name, f"{word} · summary")
            kind = "dense"
        layers.append({"id": name, "group": group, "kind": kind, "shape": shape, "col": col,
                       "label": label, "name": human})
    layers.append({"id": "policy", "group": "head", "kind": "output", "shape": [P.N_ACTIONS], "col": 10,
                   "label": "move", "name": "move", "names": MOVES})
    layers.append({"id": "value", "group": "head", "kind": "output", "shape": [1], "col": 10,
                   "label": "value", "name": "value"})
    return layers


def _top(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    return np.argpartition(-scores, k - 1)[:k]


def edges(net: PolicyNet, layers: list[dict], rng: np.random.Generator) -> list[dict]:
    """The strongest connections into each layer, as index pairs into the layers' units.

    Each group carries the weight tensor and flat parameter index of every line, so the same lines
    can be re-weighed when new weights arrive."""
    by_id = {L["id"]: L for L in layers}
    groups = []

    def add(src, dst, si, di, param, pidx):
        groups.append({"src": src, "dst": dst, "si": np.asarray(si, np.int64), "di": np.asarray(di, np.int64),
                       "param": param, "pidx": np.asarray(pidx, np.int64)})

    def stem(t, dst, mod, param):
        """Lines from the tower's input maps into its first layer: each map is drawn as one point
        per cell (all its planes together, current frame only), and every sampled stem unit gets
        a line from the cell of each map it weighs most."""
        W = mod.weight.detach().numpy()
        co_n, ci_n, k, _ = W.shape
        s = mod.stride[0]
        A = np.abs(W)
        _, ho, wo = by_id[dst]["shape"]
        units = rng.choice(co_n * ho * wo, size=min(CONV_LINES, co_n * ho * wo), replace=False)
        first = 0
        for m in TOWER_MAPS[t]:
            planes = P.MAP_PLANES[m]
            Am = A[:, first:first + planes]  # this map's current-frame planes
            si, di, pidx = [], [], []
            for u in units:
                co, rem = divmod(int(u), ho * wo)
                y, x = divmod(rem, wo)
                flat = Am[co].reshape(-1)  # (planes, k, k): stride == kernel, so every tap is inside
                tap = int(flat.argmax())
                ci, rem2 = divmod(tap, k * k)
                ky, kx = divmod(rem2, k)
                si.append((y * s + ky) * P.GRID + x * s + kx)
                di.append(int(u))
                pidx.append(((co * ci_n + first + ci) * k + ky) * k + kx)
            add(MAP_IDS[m], dst, si, di, param, pidx)
            first += planes

    def conv(src, dst, mod, param):
        W = mod.weight.detach().numpy()
        co_n, ci_n, k, _ = W.shape
        s, p = mod.stride[0], mod.padding[0]
        A = np.abs(W)
        _, hi, wi = by_id[src]["shape"]
        _, ho, wo = by_id[dst]["shape"]
        units = rng.choice(co_n * ho * wo, size=min(CONV_LINES, co_n * ho * wo), replace=False)
        si, di, pidx = [], [], []
        for u in units:
            co, rem = divmod(int(u), ho * wo)
            y, x = divmod(rem, wo)
            ys, xs = y * s - p + np.arange(k), x * s - p + np.arange(k)
            valid = ((ys >= 0) & (ys < hi))[:, None] & ((xs >= 0) & (xs < wi))[None, :]
            cand = np.where(valid[None], A[co], -1.0)  # (ci, k, k)
            flat = cand.reshape(-1)
            for t in _top(flat, 2):
                if flat[t] < 0:
                    continue
                ci, rem2 = divmod(int(t), k * k)
                ky, kx = divmod(rem2, k)
                si.append(ci * hi * wi + int(ys[ky]) * wi + int(xs[kx]))
                di.append(int(u))
                pidx.append(((co * ci_n + ci) * k + ky) * k + kx)
        add(src, dst, si, di, param, pidx)

    def dense(srcs, dst, W, param, per_unit, units=DENSE_LINES):
        """SRCS: list of (layer id, size) the input vector is made of, in order. Lines go from the
        PER_UNIT strongest inputs of up to UNITS neurons of DST."""
        A = np.abs(W.detach().numpy())
        bounds = np.cumsum([0] + [n for _, n in srcs])
        lines = {sid: ([], [], []) for sid, _ in srcs}
        rows = np.arange(A.shape[0]) if A.shape[0] <= units else np.sort(rng.choice(A.shape[0], units, replace=False))
        for j in rows:
            for t in _top(A[j], per_unit):
                part = int(np.searchsorted(bounds, t, side="right") - 1)
                sid = srcs[part][0]
                lines[sid][0].append(int(t - bounds[part]))
                lines[sid][1].append(int(j))
                lines[sid][2].append(int(j) * A.shape[1] + int(t))
        for sid, (si, di, pidx) in lines.items():
            if si:
                add(sid, dst, si, di, param, pidx)

    fused_srcs = []
    for m, group in enumerate(TOWER_GROUPS):
        tower = net.towers[m]
        seq = [(f"towers.{m}.stem.weight", tower.stem, True)]
        for b_i, b in enumerate(tower.res):
            seq += [(f"towers.{m}.res.{b_i}.c0.weight", b.c0, False), (f"towers.{m}.res.{b_i}.c1.weight", b.c1, False)]
        seq += [(f"towers.{m}.down0.weight", tower.down0, False), (f"towers.{m}.down1.weight", tower.down1, False)]
        # the page shows one slab per ReLU output: stem, each residual block (after its 2nd conv), two downs
        names = [f"{group}_c1"] + [f"{group}_c{2 + i}" for i in range(len(tower.res))] + \
                [f"{group}_c{2 + len(tower.res)}", f"{group}_c{3 + len(tower.res)}"]
        # stem: from the tower's input maps
        stem(m, names[0], tower.stem, seq[0][0])
        prev_name = names[0]
        for b_i, b in enumerate(tower.res):
            # a residual block is drawn as one slab; its lines come from the block's second conv
            conv(prev_name, names[1 + b_i], b.c1, f"towers.{m}.res.{b_i}.c1.weight")
            prev_name = names[1 + b_i]
        conv(prev_name, names[-2], tower.down0, f"towers.{m}.down0.weight")
        conv(names[-2], names[-1], tower.down1, f"towers.{m}.down1.weight")
        last = by_id[names[-1]]["shape"]
        dense([(names[-1], int(np.prod(last)))], f"{group}_fc", tower.fc.weight, f"towers.{m}.fc.weight", 2)
        fused_srcs.append((f"{group}_fc", tower.fc.out_features))
    dense([("scalars_in", P.N_SCALARS)], "scalars_h1", net.scalars[0].weight, "scalars.0.weight", 2)
    dense([("scalars_h1", 256)], "scalars_h2", net.scalars[2].weight, "scalars.2.weight", 2)
    dense(fused_srcs + [("scalars_h2", 256)], "trunk1", net.trunk[0].weight, "trunk.0.weight", 3)
    dense([("trunk1", net.trunk[2].weight.shape[1])], "trunk2", net.trunk[2].weight, "trunk.2.weight", 2)
    dense([("trunk2", net.pi.weight.shape[1])], "policy", net.pi.weight, "pi.weight", 48)
    dense([("trunk2", net.v.weight.shape[1])], "value", net.v.weight, "v.weight", 48)
    return groups


def _u8(x: np.ndarray) -> str:
    return base64.b64encode(np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8).tobytes()).decode()


def weigh(groups: list[dict], state: dict, old: dict | None) -> list[dict]:
    """Line strengths (and how much each changed since OLD) for the page, normalised per group."""
    out = []
    for g in groups:
        W = state[g["param"]].detach().cpu().numpy().reshape(-1)
        w = np.abs(W[g["pidx"]])
        w = np.sqrt(w / max(float(w.max()), 1e-12))  # sqrt so weaker lines stay visible
        item = {"src": g["src"], "dst": g["dst"],
                "si": base64.b64encode(g["si"].astype(np.uint32).tobytes()).decode(),
                "di": base64.b64encode(g["di"].astype(np.uint32).tobytes()).decode(),
                "w": _u8(w)}
        if old is not None:
            d = np.abs(W[g["pidx"]] - old[g["param"]].detach().cpu().numpy().reshape(-1)[g["pidx"]])
            item["dw"] = _u8(d / max(float(d.max()), 1e-12))
        out.append(item)
    return out


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
                if self.path in ("/", "/index.html"):
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

    KEEP = 150  # updates of history the page gets for its sparklines
    KEYS = ("update", "steps", "sps", "lr", "pg", "vf", "ent", "kl", "clipfrac", "grad_norm",
            "value_explained", "ep_len", "policy_lag", "collect_s", "learn_s", "time")

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
        if net_version(ck["model"]) != 2:
            raise SystemExit("the brain page shows v2 networks; export/convert older checkpoints first")
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
                    help="weights to show (default: runs/v2/actor.pt, which training rewrites every update)")
    ap.add_argument("--greedy", action="store_true", help="always take the top move")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
    args = ap.parse_args()
    # one decision at a time needs no thread pool; a wide one only fights the trainer for the CPU
    torch.set_num_threads(2)

    run = PROJECT / "runs/v2"
    path = args.checkpoint or next((f for f in (run / "actor.pt", run / "latest.pt") if f.exists()), run / "actor.pt")
    weights = Weights(path)
    if weights.poll() is None:
        raise SystemExit(f"no checkpoint at {path}")
    run_dir = path.parent.parent if path.parent.name == "pool" else path.parent
    metrics = Metrics(run_dir / "metrics.jsonl")
    net = weights.net
    rng = np.random.default_rng(0)
    layers = layer_table(net)
    groups = edges(net, layers, rng)
    hub = Hub(args.speed)

    def publish_topology(old: dict | None) -> None:
        hub.publish("topology", json.dumps({
            "layers": layers, "edges": weigh(groups, weights.state, old or None),
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
                            outs, probs, value = trace(net, stacked(s.maps, prev_maps.get(k)), s.scalars, mask)
                            prev_maps[k] = np.array(s.maps)
                            a = int(probs.argmax()) if args.greedy else int(np.random.choice(len(probs), p=probs / probs.sum()))
                            acts[k] = a
                            # every decision counts while paused (stepping must not skip one)
                            if k == 0 and (hub.paused or time.monotonic() - last_frame >= 1.0 / MAX_FPS):
                                last_frame = time.monotonic()
                                acts_q = np.concatenate([np.clip(scales(n, v), 0, 1) for n, v in outs])
                                me = next((c for c in m.cycles.values() if c.slot == 0), None)
                                hub.publish("frame", json.dumps({
                                    "update": weights.update, "trained_ago": (time.time() - weights.changed_at)
                                    if weights.changed_at else None,
                                    "acts": _u8(acts_q),
                                    "maps": {mid: base64.b64encode(np.ascontiguousarray(s.maps[i]).tobytes()).decode()
                                             for i, mid in enumerate(MAP_IDS)},
                                    "scalars": _u8(np.tanh(np.abs(s.scalars))),
                                    "numbers": [round(float(x), 3) for x in s.scalars],
                                    "probs": [round(float(x), 4) for x in probs], "mask": mask.tolist(),
                                    "action": a, "value": round(value, 3),
                                    "round": m.round_id, "time": round(m.time, 2), "bounds": m.bounds,
                                    "walls": args.walls,
                                    "me": me.pid if me else None,
                                    "cycles": [[c.pid, round(c.x, 1), round(c.y, 1), int(c.alive), c.color, c.slot]
                                               for c in m.cycles.values()],
                                    "result": m.last_result,
                                }, separators=(",", ":")))
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
