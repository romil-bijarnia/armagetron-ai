"""Watch the trained AI play, drawn live in the terminal (no game window).

    uv run arma-show                     # two copies of the AI against each other
    uv run arma-show --bots 1 --ais 1    # the AI against the game's best bot
    uv run arma-show --ais 2 --bots 2 --size -1 --speed 2

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
AI_COLORS = ["#8fafcd", "#c9a977", "#a0b8a3", "#b4a2c3"]
BOT_COLORS = ["#c28d81", "#9ab9b8", "#b9bca0", "#b6a4a8"]
DIM = "#8b949e"
RIM = "#48515d"


def dim(hex_color: str, f: float = 0.45) -> str:
    return "#" + "".join(f"{round(int(hex_color[i:i + 2], 16) * f):02x}" for i in (1, 3, 5))


def direction(cycle: Cycle) -> str:
    if abs(cycle.dx) > abs(cycle.dy):
        return "→" if cycle.dx >= 0 else "←"
    return "↑" if cycle.dy >= 0 else "↓"


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
        if self.slot == 255:
            return f"{self.name.capitalize()} (bot)"
        return "Blacklight" if self.slot == 0 else f"Blacklight {self.slot + 1}"


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
        cols, rows = max(1, options.max_width), max(1, options.height or 20)
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
                heads[(min(max(y // 4, 0), rows - 1), min(max(x // 2, 0), cols - 1))] = (c.color, direction(c))
        for r in range(rows):
            line = Text(no_wrap=True)
            for k in range(cols):
                if (r, k) in heads:
                    head_color, heading = heads[(r, k)]
                    line.append(heading, style=f"bold {head_color}")
                elif bits[r][k]:
                    line.append(chr(0x2800 + bits[r][k]), style=color[r][k])
                else:
                    line.append(" ")
            yield line


def scoreboard(m: Match, brain: Brain, speed: float, *, compact: bool = False) -> Group:
    if not m.cycles:
        return Group(Text("Waiting for players", style=DIM))
    if compact:
        alive = sum(c.alive for c in m.cycles.values())
        return Group(Text(f"{alive}/{len(m.cycles)} alive  ·  {m.draws} draws", style=DIM))
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(no_wrap=True, ratio=1, overflow="ellipsis")
    t.add_column(no_wrap=True)
    t.add_column(no_wrap=True, justify="right")
    t.add_column(no_wrap=True, justify="right")
    t.add_row(Text("Player", style=DIM), Text("State", style=DIM), Text("Speed", style=DIM),
              Text("Wins", style=DIM))
    for c in sorted(m.cycles.values(), key=lambda c: (c.slot == 255, c.slot, c.name)):
        label = f"{direction(c)} {c.label}" if c.alive else c.label
        t.add_row(Text(label, style=c.color if c.alive else DIM),
                  Text("alive" if c.alive else "out", style=DIM),
                  Text(f"{c.speed:.0f} m/s" if c.alive else "—", style=DIM),
                  Text(str(m.wins.get(c.pid, 0))))
    info = Text(f"Draws {m.draws}", style=DIM, no_wrap=True, overflow="ellipsis")
    if m.last_result:
        info.append(f"\nLast: {m.last_result}", style=DIM)
    return Group(t, info)


def build(m: Match, brain: Brain, speed: float, width: int, height: int) -> Group:
    """Fit the arena's proportions, with a side table only when there is room for it."""
    width, height = max(20, min(width, 120)), max(8, height)
    lx, ly, hx, hy = m.bounds
    aspect = max((hx - lx) / max(hy - ly, 1e-6), .1)
    wide = width >= 80 and height >= 16
    compact = not wide and height < len(m.cycles) + 9
    board_height = 1 if compact or not m.cycles else len(m.cycles) + 2 + bool(m.last_result)
    side, gap = (34, 3) if wide else (0, 0)
    available_cols = width - side - gap - 2
    available_rows = height - 5 - (0 if wide else board_height)
    rows = max(2, min(available_rows, int(available_cols / (2 * aspect))))
    cols = max(1, min(available_cols, round(rows * 2 * aspect)))
    content_width = cols + 2 + side + gap if wide else min(width, max(cols + 2, 46))
    header = Table.grid(expand=True)
    header.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    header.add_column(justify="right", no_wrap=True)
    clock = f"{int(m.time // 60)}:{int(m.time % 60):02d}"
    header.add_row(Text("Blacklight / match", style="bold"), Text(f"Round {m.rounds}  ·  {clock}", style=DIM))
    identity = Text(f"Blacklight u{brain.update:,}  ·  {speed:g}x  ·  {hx - lx:.0f} x {hy - ly:.0f} m",
                    style=DIM, no_wrap=True, overflow="ellipsis")
    arena = Panel(Arena(m), box=box.SQUARE, border_style=RIM, padding=0, width=cols + 2, height=rows + 2)
    board = scoreboard(m, brain, speed, compact=compact)
    if wide:
        body = Table.grid(padding=(0, gap))
        body.add_column(width=cols + 2)
        body.add_column(width=side)
        body.add_row(arena, board)
    else:
        body = Group(arena, board)
    footer = Text("Ctrl-C ends this match.", style=DIM, no_wrap=True, overflow="ellipsis")
    bounded = Table.grid()
    bounded.add_column(width=content_width)
    for item in (header, identity, body, footer):
        bounded.add_row(item)
    return Group(bounded)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ais", type=int, default=2, help="cycles driven by the network")
    ap.add_argument("--bots", type=int, default=0, help="built-in AI opponents")
    ap.add_argument("--size", type=float, default=-2, help="arena SIZE_FACTOR (-3 small, 0 full-size)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    ap.add_argument("--walls", type=float, default=600, help="trail length in metres (-1 = endless)")
    ap.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/blacklight/latest.pt")
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
                                                     screen=False, auto_refresh=False, vertical_overflow="crop")
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
