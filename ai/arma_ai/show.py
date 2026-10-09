"""Watch the trained AI play, drawn live in the terminal (no game window).

    ./trainctl show                     # two copies of the AI against each other
    ./trainctl show --bots 1 --ais 1    # the AI against the game's best bot
    ./trainctl show --ais 2 --bots 2 --size -1 --speed 2

A real match runs in a headless engine; every cycle's trail is drawn with braille dots in its own
colour, with a scoreboard beside the arena. Ctrl-C quits.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import protocol as P
from .engine import ArenaConfig, EnginePool, _parse_step
from .play import Brain

PROJECT = Path(__file__).resolve().parent.parent
WALLS_STAY_UP = 8.0  # seconds a dead cycle's trail stays solid (the game's SP_WALLS_STAY_UP_DELAY)
_DOT = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
AI_COLORS = ["#22d3ee", "#f472b6", "#facc15", "#4ade80"]
BOT_COLORS = ["#fb923c", "#a78bfa", "#94a3b8", "#f87171"]


def dim(hex_color: str, f: float = 0.45) -> str:
    return "#" + "".join(f"{round(int(hex_color[i:i + 2], 16) * f):02x}" for i in (1, 3, 5))


@dataclass
class Cycle:
    pid: int
    name: str
    slot: int
    color: str
    alive: bool = True
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0
    dy: float = 1.0
    speed: float = 0.0
    trail: deque = field(default_factory=deque)  # (x, y) points, oldest first
    length: float = 0.0  # metres of trail currently kept
    died_at: float | None = None

    @property
    def label(self) -> str:
        return f"AI {self.slot + 1}" if self.slot != 255 else f"{self.name.capitalize()} (bot)"


class Match:
    def __init__(self, walls_length: float = -1):
        self.walls_length = walls_length  # the engine's WALLS_LENGTH; <= 0 means endless
        self.round_id = -1
        self.time = 0.0
        self.over = False
        self.bounds = (0.0, 0.0, 500.0, 500.0)
        self.cycles: dict[int, Cycle] = {}
        self.wins: dict[int, int] = {}
        self.draws = 0
        self.rounds = 0
        self.last_result = ""
        self._colors: dict[int, str] = {}

    def _extend(self, c: Cycle, p: tuple) -> None:
        """Add a point and forget the far end, so the drawn trail is the wall the engine still has."""
        if c.trail:
            lx, ly = c.trail[-1]
            c.length += math.hypot(p[0] - lx, p[1] - ly)
        c.trail.append(p)
        if self.walls_length > 0:
            while len(c.trail) > 1:
                (x0, y0), (x1, y1) = c.trail[0], c.trail[1]
                seg = math.hypot(x1 - x0, y1 - y0)
                if c.length - seg < self.walls_length:
                    break
                c.length -= seg
                c.trail.popleft()

    def update(self, payload: bytes) -> bool:
        """Apply one WORLD message; return whether a STEP follows."""
        rid, t, over, step_follows, lx, ly, hx, hy, n = P.WORLD_HEAD.unpack_from(payload, 0)
        off = P.WORLD_HEAD.size
        if rid != self.round_id:
            self.round_id = rid
            self.rounds += 1
            self.cycles = {}
            self.over = False
        self.time, self.bounds = t, (lx, ly, hx, hy)
        for _ in range(n):
            pid, slot, alive, x, y, dx, dy, speed, raw = P.WORLD_CYCLE.unpack_from(payload, off)
            off += P.WORLD_CYCLE.size
            c = self.cycles.get(pid)
            if c is None:
                if pid not in self._colors:
                    ais = sum(1 for v in self._colors.values() if v in AI_COLORS)
                    bots = len(self._colors) - ais
                    self._colors[pid] = (AI_COLORS[slot % 4] if slot != 255 else BOT_COLORS[bots % 4])
                c = Cycle(pid, raw.split(b"\0")[0].decode(errors="replace"), slot, self._colors[pid])
                c.trail.append((x, y))
                c.dx, c.dy = dx, dy
                self.cycles[pid] = c
            elif c.alive:
                if (round(dx), round(dy)) != (round(c.dx), round(c.dy)):
                    # it turned since the last tick: put the corner in so the trail stays axis-aligned
                    corner = (x, c.y) if abs(c.dx) > abs(c.dy) else (c.x, y)
                    self._extend(c, corner)
                self._extend(c, (x, y))
            if c.alive and not alive:
                c.died_at = t
            c.alive, c.x, c.y, c.dx, c.dy, c.speed = bool(alive), x, y, dx, dy, speed
        if over and not self.over:
            self.over = True
            winners = [c for c in self.cycles.values() if c.alive]
            if winners:
                for c in winners:
                    self.wins[c.pid] = self.wins.get(c.pid, 0) + 1
                self.last_result = ", ".join(c.label for c in winners) + " won"
            else:
                self.draws += 1
                self.last_result = "draw"
        return bool(step_follows)


class Arena:
    """Top-down view of the arena; the panel border around it is the rim."""

    def __init__(self, match: Match):
        self.m = match

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        cols, rows = options.max_width, options.height or 20
        lx, ly, hx, hy = self.m.bounds
        W, H = cols * 2, rows * 4
        sx, sy = (W - 1) / max(hx - lx, 1e-6), (H - 1) / max(hy - ly, 1e-6)
        bits = [[0] * cols for _ in range(rows)]
        color = [[None] * cols for _ in range(rows)]

        def dot(x: int, y: int, c: str) -> None:
            if 0 <= x < W and 0 <= y < H:
                bits[y // 4][x // 2] |= _DOT[y % 4][x % 2]
                color[y // 4][x // 2] = c

        def to_dots(px: float, py: float) -> tuple[int, int]:
            return round((px - lx) * sx), round((hy - py) * sy)

        cycles = sorted(self.m.cycles.values(), key=lambda c: c.alive)  # living trails on top
        for c in cycles:
            if not c.alive and c.died_at is not None and self.m.time - c.died_at > WALLS_STAY_UP:
                continue  # the game has removed this wall
            col = c.color if c.alive else dim(c.color)
            pts = [to_dots(*p) for p in c.trail]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                dx, dy = abs(x1 - x0), -abs(y1 - y0)
                stepx, stepy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
                err = dx + dy
                while True:
                    dot(x0, y0, col)
                    if x0 == x1 and y0 == y1:
                        break
                    e2 = 2 * err
                    if e2 >= dy:
                        err += dy
                        x0 += stepx
                    if e2 <= dx:
                        err += dx
                        y0 += stepy
            if len(pts) == 1:
                dot(*pts[0], col)
        heads = {}
        for c in self.m.cycles.values():
            if c.alive:
                x, y = to_dots(c.x, c.y)
                heads[(min(max(y // 4, 0), rows - 1), min(max(x // 2, 0), cols - 1))] = c.color
        for r in range(rows):
            line = Text(no_wrap=True)
            for k in range(cols):
                if (r, k) in heads:
                    line.append("⣿", style=f"bold {heads[(r, k)]}")
                elif bits[r][k]:
                    line.append(chr(0x2800 + bits[r][k]), style=color[r][k])
                else:
                    line.append(" ")
            yield line


def scoreboard(m: Match, brain: Brain, speed: float) -> Group:
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(no_wrap=True, width=1)
    t.add_column(no_wrap=True, ratio=1, overflow="ellipsis")
    t.add_column(no_wrap=True, justify="right")
    t.add_column(no_wrap=True, justify="right", width=4)
    t.add_row("", Text("player", style="bold #d0d0d0"), Text("now", style="bold #d0d0d0"),
              Text("wins", style="bold #d0d0d0"))
    for c in sorted(m.cycles.values(), key=lambda c: (c.slot == 255, c.slot, c.name)):
        status = Text(f"{c.speed:.0f} m/s", style="#5fd75f") if c.alive else Text("crashed", style="#ff5f5f")
        t.add_row(Text("■", style=c.color), Text(c.label, style="bold #e4e4e4" if c.alive else "#8a8a8a"),
                  status, Text(str(m.wins.get(c.pid, 0)), style="bold #e4e4e4"))
    info = Text(no_wrap=True, overflow="ellipsis")
    info.append(f"\nround {m.rounds}   ", style="#9e9e9e")
    info.append(f"{int(m.time // 60)}:{int(m.time % 60):02d}", style="bold #e4e4e4")
    info.append(f"   draws {m.draws}\n", style="#9e9e9e")
    if m.last_result:
        info.append("last round: ", style="#9e9e9e")
        info.append(m.last_result + "\n", style="bold #5fafd7")
    info.append(f"\nAI = training update {brain.update}\n", style="#6c6c6c")
    info.append(f"speed {speed:g}x", style="#6c6c6c")
    return Group(t, info)


def build(m: Match, brain: Brain, speed: float, width: int, height: int) -> Layout:
    lx, ly, hx, hy = m.bounds
    aspect = (hx - lx) / max(hy - ly, 1e-6)
    side = 34
    rows = max(height - 3, 6)  # arena panel border (2) + footer (1)
    cols = round(rows * 2 * aspect)  # braille dots are square when a cell is twice as tall as wide
    if cols + 2 + side > width:
        cols = max(width - 2 - side, 10)
        rows = max(round(cols / (2 * aspect)), 4)
    title = Text()
    title.append("¹", style="bold #ffffff")
    title.append("arena", style="bold #2fb9b0")
    title.append(f"  {hx - lx:.0f} m", style="#9e9e9e")
    arena = Panel(Arena(m), title=title, title_align="left", box=box.ROUNDED, border_style="#2fb9b0", padding=0)
    t2 = Text()
    t2.append("²", style="bold #ffffff")
    t2.append("match", style="bold #d1589f")
    board = Panel(scoreboard(m, brain, speed), title=t2, title_align="left", box=box.ROUNDED,
                  border_style="#d1589f", padding=(0, 1))
    body = Layout()
    body.split_row(Layout(arena, size=cols + 2), Layout(board, size=side), Layout(Text(" "), ratio=1))
    lay = Layout()
    lay.split_column(Layout(body, size=rows + 2), Layout(Text(" ctrl-c quits", style="#6c6c6c"), size=1),
                     Layout(Text(" "), ratio=1))
    return lay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ais", type=int, default=2, help="cycles driven by the network")
    ap.add_argument("--bots", type=int, default=0, help="built-in AI opponents")
    ap.add_argument("--size", type=float, default=-2, help="arena SIZE_FACTOR (-3 small, 0 full-size)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    ap.add_argument("--walls", type=float, default=600, help="trail length in metres (-1 = endless)")
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/v2/latest.pt")
    ap.add_argument("--greedy", action="store_true",
                    help="always take the top move (copies of the AI then tend to mirror each other)")
    ap.add_argument("--snapshot", type=float, default=0,
                    help="testing: play this many seconds without the live view, then print one frame")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()

    brain = Brain(args.checkpoint, sample=not args.greedy)
    arena = ArenaConfig(slots=args.ais, builtin_ais=args.bots, size_factor=args.size, walls_length=args.walls, extra={
        "NEURAL_SPECTATE": "1", "NEURAL_END_ROUND_WITHOUT_NEURAL": "0", "NEURAL_CONTROL_AFTER_ROUND": "1"})
    console = Console(width=args.width, height=args.height)
    m = Match(args.walls)
    with EnginePool([arena], workdir=PROJECT / "runtime" / "show", base_port=47900, log_engines=True) as pool:
        conn = pool.conns[0]
        conn.setblocking(True)
        clock = None  # (wall time, game time) at the start of the current round
        frame = 0
        class Snapshot:  # stands in for Live when testing
            def update(self, *_, **__):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                console.print(build(m, brain, args.speed, *console.size), height=console.size.height)

        view = Snapshot() if args.snapshot else Live(build(m, brain, args.speed, *console.size), console=console,
                                                     screen=True, auto_refresh=False)
        started = time.monotonic()
        try:
            with view as live:
                while not args.snapshot or time.monotonic() - started < args.snapshot:
                    mtype, payload = pool._read_msg(conn)
                    if mtype != P.MSG_WORLD:
                        continue
                    rid = m.round_id
                    step_follows = m.update(payload)
                    if m.round_id != rid or clock is None:
                        clock = (time.monotonic(), m.time)
                    if step_follows:
                        mtype, payload = pool._read_msg(conn)
                        acts = brain.actions(_parse_step(0, payload)) if mtype == P.MSG_STEP else []
                    else:
                        acts = []
                    frame += 1
                    if not m.over or frame % 4 == 0:  # after the round is decided, fast-forward
                        live.update(build(m, brain, args.speed, *console.size), refresh=True)
                    if not m.over:
                        delay = clock[0] + (m.time - clock[1]) / args.speed - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                    body = bytes([len(acts)]) + bytes(acts)
                    conn.sendall(P.HEADER.pack(P.MAGIC, P.MSG_ACTIONS, len(body)) + body)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
