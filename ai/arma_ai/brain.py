"""Watch the network think: every neuron of the policy network, drawn live in the browser.

    ./trainctl brain                    # the AI against the game's best bot
    ./trainctl brain --ais 2 --bots 0   # two copies of the AI; the brain shown is AI 1's
    ./trainctl brain --speed 0.5        # half speed

A real match runs in a headless engine, as in ./trainctl show. Every time AI 1 decides, the
activation of every neuron is sent to a page (http://127.0.0.1:8765) that draws the network in 3D:
one slab of points per layer, lit by how strongly each neuron fires, and the strongest connections
between layers, glowing where signal flows through them. Space pauses the match, the right arrow
steps it one decision at a time, [ and ] change the speed, and hovering a neuron says what it is.
While training runs, the page follows the newest weights (runs/main/actor.pt) and flashes the
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
from .model import PolicyNet
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
GROUP_WORD = {"local": "nearby", "global": "arena", "scalars": "numbers", "trunk": "trunk"}


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

@torch.no_grad()
def trace(net: PolicyNet, local: np.ndarray, globl: np.ndarray, scalars: np.ndarray, mask: np.ndarray):
    """Run one decision through NET and keep every layer's output.

    local/globl: (GRID, GRID) uint8 bit-planes, scalars: (N_SCALARS,) float32, mask: (N_ACTIONS,) bool.
    Returns (layers, probs, value) where layers is a list of (name, activations) after each ReLU."""
    out: list[tuple[str, torch.Tensor]] = []

    def tower(prefix: str, mod, unpack, packed: np.ndarray) -> torch.Tensor:
        x = unpack(torch.from_numpy(np.array(packed, np.uint8))[None])
        n = 0
        for m in mod.net:
            x = m(x)
            if isinstance(m, nn.ReLU):
                n += 1
                out.append((f"{prefix}_c{n}", x[0]))
        h = torch.relu(mod.fc(x.flatten(1)))
        out.append((f"{prefix}_fc", h[0]))
        return h

    def mlp(prefix: str, seq: nn.Sequential, h: torch.Tensor) -> torch.Tensor:
        n = 0
        for m in seq:
            h = m(h)
            if isinstance(m, nn.ReLU):
                n += 1
                out.append((f"{prefix}{n}", h[0]))
        return h

    hl = tower("local", net.local, net.unpack_local, local)
    hg = tower("global", net.globl, net.unpack_global, globl)
    hs = mlp("scalars_h", net.scalars, torch.from_numpy(np.array(scalars, np.float32))[None])
    h = mlp("trunk", net.trunk, torch.cat([hl, hg, hs], 1))
    logits = net.pi(h)[0].masked_fill(~torch.from_numpy(np.asarray(mask, bool)), -1e8)
    probs = torch.softmax(logits, 0)
    return [(k, v.numpy()) for k, v in out], probs.numpy(), float(net.v(h)[0, 0])


def layer_table(net: PolicyNet) -> list[dict]:
    """Every layer the page draws, in frame order, with where it sits in the picture."""
    blank = np.zeros((P.GRID, P.GRID), np.uint8)
    outs = trace(net, blank, blank, np.zeros(P.N_SCALARS, np.float32), np.ones(P.N_ACTIONS, bool))[0]
    names, shapes = [k for k, _ in outs], dict(outs)
    layers = [
        {"id": "local_in", "group": "local", "kind": "input", "shape": [P.GRID, P.GRID], "col": 0,
         "label": "what it sees nearby", "name": "what it sees nearby", "planes": LOCAL_PLANES},
        {"id": "global_in", "group": "global", "kind": "input", "shape": [P.GRID, P.GRID], "col": 0,
         "label": "the whole arena", "name": "the whole arena", "planes": GLOBAL_PLANES},
        {"id": "scalars_in", "group": "scalars", "kind": "input", "shape": [P.N_SCALARS], "col": 0,
         "label": f"{P.N_SCALARS} exact numbers", "name": "exact numbers", "names": scalar_names()},
    ]
    seen16 = {"local": 0, "global": 0}
    convs = {"local": 0, "global": 0}
    for name in names:
        shape = list(shapes[name].shape)
        group = name.split("_")[0] if not name.startswith("trunk") else "trunk"
        word = GROUP_WORD[group]
        if len(shape) == 3:
            if shape[1] == P.GRID // 4:  # full patch resolution, one column each
                seen16[group] += 1
                col = seen16[group]
            else:
                col = 4 if shape[1] == P.GRID // 8 else 5
            convs[group] += 1
            label = f"{shape[0]} filters · {shape[1]}×{shape[2]}"
            human = f"{word} · conv {convs[group]}"
            kind = "conv"
        else:
            col = {"local_fc": 6, "global_fc": 6, "scalars_h1": 3, "scalars_h2": 6,
                   "trunk1": 7, "trunk2": 8}[name]
            label = f"{shape[0]} neurons"
            human = {"local_fc": "nearby · summary", "global_fc": "arena · summary",
                     "scalars_h1": "numbers · layer 1", "scalars_h2": "numbers · layer 2",
                     "trunk1": "trunk · layer 1", "trunk2": "trunk · layer 2"}[name]
            kind = "dense"
        layers.append({"id": name, "group": group, "kind": kind, "shape": shape, "col": col,
                       "label": label, "name": human})
    layers.append({"id": "policy", "group": "head", "kind": "output", "shape": [P.N_ACTIONS], "col": 9,
                   "label": "move", "name": "move", "names": MOVES})
    layers.append({"id": "value", "group": "head", "kind": "output", "shape": [1], "col": 9,
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

    def conv(src, dst, mod, param, composite=False):
        W = mod.weight.detach().numpy()
        co_n, ci_n, k, _ = W.shape
        s, p = mod.stride[0], mod.padding[0]
        if composite:  # the input is drawn as one point per cell, all planes together
            hi = wi = P.GRID
        else:
            _, hi, wi = by_id[src]["shape"]
        _, ho, wo = by_id[dst]["shape"]
        A = np.abs(W)
        units = rng.choice(co_n * ho * wo, size=min(CONV_LINES, co_n * ho * wo), replace=False)
        si, di, pidx = [], [], []
        for u in units:
            co, rem = divmod(int(u), ho * wo)
            y, x = divmod(rem, wo)
            ys, xs = y * s - p + np.arange(k), x * s - p + np.arange(k)
            valid = ((ys >= 0) & (ys < hi))[:, None] & ((xs >= 0) & (xs < wi))[None, :]
            cand = np.where(valid[None], A[co], -1.0)  # (ci, k, k)
            if composite:
                best_ci = cand.argmax(0)
                cand = cand.max(0)[None]
            flat = cand.reshape(-1)
            for t in _top(flat, 2):
                if flat[t] < 0:
                    continue
                ci, rem2 = divmod(int(t), k * k)
                ky, kx = divmod(rem2, k)
                if composite:
                    ci_w = int(best_ci[ky, kx])
                    si.append(int(ys[ky]) * wi + int(xs[kx]))
                else:
                    ci_w = ci
                    si.append(ci * hi * wi + int(ys[ky]) * wi + int(xs[kx]))
                di.append(int(u))
                pidx.append(((co * ci_n + ci_w) * k + ky) * k + kx)
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

    for prefix, tower in (("local", net.local), ("global", net.globl)):
        attr = "local" if prefix == "local" else "globl"
        convs = [m for m in tower.net if isinstance(m, nn.Conv2d)]
        names = [f"{prefix}_c{i + 1}" for i in range(len(convs))]
        params = [n for n, m in tower.net.named_modules() if isinstance(m, nn.Conv2d)]
        conv(f"{prefix}_in", names[0], convs[0], f"{attr}.net.{params[0]}.weight", composite=True)
        for i in range(1, len(convs)):
            conv(names[i - 1], names[i], convs[i], f"{attr}.net.{params[i]}.weight")
        last = by_id[names[-1]]["shape"]
        dense([(names[-1], int(np.prod(last)))], f"{prefix}_fc", tower.fc.weight, f"{attr}.fc.weight", 2)
    dense([("scalars_in", P.N_SCALARS)], "scalars_h1", net.scalars[0].weight, "scalars.0.weight", 2)
    dense([("scalars_h1", 256)], "scalars_h2", net.scalars[2].weight, "scalars.2.weight", 2)
    dense([("local_fc", 256), ("global_fc", 256), ("scalars_h2", 256)], "trunk1", net.trunk[0].weight,
          "trunk.0.weight", 3)
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
        self.topology: bytes | None = None
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
        if event == "topology":
            self.topology = msg
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
                        first = (hub.topology or b"") + hub.state_msg
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

class Weights:
    """The network, reloaded whenever its checkpoint file changes (every update while training)."""

    def __init__(self, path: Path):
        self.path = path
        self.net = PolicyNet().eval()
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
        self.net.load_state_dict(ck["model"])
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
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="weights to show (default: runs/main/actor.pt, which training rewrites every update)")
    ap.add_argument("--greedy", action="store_true", help="always take the top move")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the page in the browser")
    args = ap.parse_args()

    run = PROJECT / "runs/main"
    path = args.checkpoint or next((f for f in (run / "actor.pt", run / "latest.pt") if f.exists()), run / "actor.pt")
    weights = Weights(path)
    if weights.poll() is None:
        raise SystemExit(f"no checkpoint at {path}")
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

    def follow_weights() -> None:
        old = weights.poll()
        if old is not None:
            publish_topology(old)

    publish_topology(None)
    srv, port = serve(hub, args.port)
    url = f"http://127.0.0.1:{port}/"
    print(f"brain: {url}  (update {weights.update}, {path.name}); Ctrl-C quits", flush=True)
    if not args.no_open:
        webbrowser.open(url)

    arena = ArenaConfig(slots=args.ais, builtin_ais=args.bots, size_factor=args.size, extra={
        "NEURAL_SPECTATE": "1", "NEURAL_END_ROUND_WITHOUT_NEURAL": "0", "NEURAL_CONTROL_AFTER_ROUND": "1"})
    scales = Scales()
    m = Match()
    with EnginePool([arena], workdir=PROJECT / "runtime" / "brain", base_port=47950, log_engines=True) as pool:
        conn = pool.conns[0]
        conn.setblocking(True)
        clock = None
        speed = hub.speed
        last_frame = last_poll = 0.0
        try:
            while True:
                now = time.monotonic()
                if now - last_poll > 1.0:
                    last_poll = now
                    follow_weights()
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
                        for k, s in enumerate(step.slots):
                            if not s.needs_action:
                                continue
                            mask = MASK_TABLE[s.mask] > 0.5
                            outs, probs, value = trace(net, s.local, s.globl, s.scalars, mask)
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
                                    "local": base64.b64encode(np.ascontiguousarray(s.local).tobytes()).decode(),
                                    "global": base64.b64encode(np.ascontiguousarray(s.globl).tobytes()).decode(),
                                    "scalars": _u8(np.tanh(np.abs(s.scalars))),
                                    "numbers": [round(float(x), 3) for x in s.scalars],
                                    "probs": [round(float(x), 4) for x in probs], "mask": mask.tolist(),
                                    "action": a, "value": round(value, 3),
                                    "round": m.round_id, "time": round(m.time, 2), "bounds": m.bounds,
                                    "me": me.pid if me else None,
                                    "cycles": [[c.pid, round(c.x, 1), round(c.y, 1), int(c.alive), c.color, c.slot]
                                               for c in m.cycles.values()],
                                    "result": m.last_result,
                                }, separators=(",", ":")))
                                hub.frame_shown()
                # paused: hold the match here (the engine waits for its actions); a step lets it run
                # until the next decision has been shown
                if hub.wait_if_paused(follow_weights) or hub.speed != speed:
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
