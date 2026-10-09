"""Live btop-style terminal dashboard for a training run.

    ./trainctl watch            (or: uv run arma-dash)

Everything shown comes from the run itself (runs/<run>/metrics.jsonl, its checkpoints and its saved
past versions) and redraws every two seconds. Ctrl-C closes the dashboard; training keeps running.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT = Path(__file__).resolve().parent.parent
DECISIONS_PER_SECOND = 20  # the engine asks for a move every 0.05 s of game time
TRAINCTL = str(PROJECT / "trainctl").replace(str(Path.home()), "~", 1)

# braille cells are 2 dots wide and 4 high; bit for (row, column) inside a cell
_DOT = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

# btop-like palette: each gradient runs from near the axis to far from it
GRAD = {
    "teal": ["#0d3b45", "#137a85", "#22b5b0", "#7fe8d8", "#d5fff6"],
    "green": ["#16351a", "#246b2c", "#3aa64a", "#86dd78", "#e0ffd0"],
    "pink": ["#3a1530", "#7a2766", "#c2449e", "#ef8fd0", "#ffd9f2"],
    "blue": ["#14254a", "#21468c", "#3b78d4", "#86b4ff", "#dbe9ff"],
    "purple": ["#24183f", "#4b3290", "#7d5bd6", "#b9a1ff", "#ece4ff"],
}
BORDER = {"head": "#5f87af", "survival": "#2fb9b0", "bot": "#d1589f", "self": "#5fbf5f",
          "learn": "#6f9de8", "speed": "#9b7be6", "arenas": "#b0b0b0", "updates": "#8a8a8a"}
SUPER = "⁰¹²³⁴⁵⁶⁷⁸⁹"

ARENAS = {  # key in metrics -> (name, opponents, even-share baseline)
    "vs_ai_small": ("best bot · small", "1 built-in bot", 0.5),
    "vs_ai_std": ("best bot · full-size", "1 built-in bot", 0.5),
    "mixed_ffa": ("4-player with bots", "2 bots, 1 self", 0.25),
    "vs_ai_ffa": ("4-player vs 3 bots", "3 built-in bots", 0.25),
    "duel_small": ("duel · small", "self / past self", 0.5),
    "duel_mid": ("duel · medium", "self / past self", 0.5),
    "duel_std": ("duel · full-size", "self / past self", 0.5),
    "ffa4": ("4-player", "self / past selves", 0.25),
}


def lerp_hex(colors: list[str], f: float) -> str:
    f = min(max(f, 0.0), 1.0) * (len(colors) - 1)
    i = min(int(f), len(colors) - 2)
    t = f - i
    a = [int(colors[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(colors[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def resample(values: list[float | None], n: int) -> list[float | None]:
    """Values at n evenly spaced points, linearly interpolated between the valid samples."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    if not pts:
        return [None] * n
    if len(pts) == 1:
        return [pts[0][1] if k * (len(values) - 1) / max(n - 1, 1) >= pts[0][0] else None for k in range(n)]
    span = max(len(values) - 1, 1)
    out, j = [], 0
    for k in range(n):
        x = k * span / max(n - 1, 1)
        if x < pts[0][0]:
            out.append(None)  # before this series had any data
            continue
        while j < len(pts) - 2 and pts[j + 1][0] < x:
            j += 1
        (x0, y0), (x1, y1) = pts[j], pts[j + 1]
        out.append(y1 if x >= x1 else y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return out


class AreaGraph:
    """Filled braille area graph. With a second series it mirrors btop's network panel: the first
    series grows up from the middle line, the second grows down from it."""

    def __init__(self, up: list[float | None], up_grad: list[str], down: list[float | None] | None = None,
                 down_grad: list[str] | None = None, hi: float | None = None, down_hi: float | None = None,
                 baseline: float | None = None, down_baseline: float | None = None, fmt: str = "{:.0f}"):
        self.up, self.up_grad, self.down, self.down_grad = up, up_grad, down, down_grad
        self.hi, self.down_hi = hi, down_hi
        self.baseline, self.down_baseline = baseline, down_baseline
        self.fmt = fmt

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield from self.render(options.max_width, options.height or 6)

    @staticmethod
    def _top(values: list[float | None], hi: float | None) -> float:
        if hi is not None:
            return hi
        vals = [v for v in values if v is not None and math.isfinite(v)]
        return max(max(vals) * 1.12, 1e-6) if vals else 1.0

    @staticmethod
    def _half(values, grad, hi, baseline, rows, cols, grow_up: bool) -> list[Text]:
        W, H = cols * 2, rows * 4
        cells = [[0] * cols for _ in range(rows)]
        filled = [[False] * cols for _ in range(rows)]
        for x, v in enumerate(resample(values, W)):
            if v is None:
                continue
            h = round(min(max(v / hi, 0.0), 1.0) * H)
            for d in range(h):  # d = distance from the axis in dots
                y = H - 1 - d if grow_up else d
                cells[y // 4][x // 2] |= _DOT[y % 4][x % 2]
                filled[y // 4][x // 2] = True
        base_cells = set()
        if baseline is not None and 0 < baseline < hi:
            d = round(baseline / hi * (H - 1))
            y = H - 1 - d if grow_up else d
            for x in range(0, W, 4):
                if not filled[y // 4][x // 2]:
                    cells[y // 4][x // 2] |= _DOT[y % 4][x % 2]
                    base_cells.add((y // 4, x // 2))
        lines = []
        for r in range(rows):
            dist = (rows - 1 - r) if grow_up else r  # rows away from the axis
            color = lerp_hex(grad, (dist + 0.5) / rows)
            line = Text(no_wrap=True)
            for c in range(cols):
                bits = cells[r][c]
                if not bits:
                    line.append(" ")
                elif (r, c) in base_cells:
                    line.append(chr(0x2800 + bits), style="#4a4a4a")
                else:
                    line.append(chr(0x2800 + bits), style=color)
            lines.append(line)
        return lines

    def render(self, width: int, height: int) -> list[Text]:
        mirrored = self.down is not None
        hi_up = self._top(self.up, self.hi)
        hi_dn = self._top(self.down, self.down_hi) if mirrored else 0.0
        labels = [self.fmt.format(hi_up), self.fmt.format(0)] + ([self.fmt.format(hi_dn)] if mirrored else [])
        gutter = max(len(s) for s in labels) + 1
        cols = max(width - gutter, 4)
        if not mirrored:
            body = self._half(self.up, self.up_grad, hi_up, self.baseline, height, cols, True)
            marks = {0: labels[0], height - 1: labels[1]}
        else:
            top_rows = max(height // 2, 1)
            bot_rows = max(height - top_rows, 1)
            body = (self._half(self.up, self.up_grad, hi_up, self.baseline, top_rows, cols, True)
                    + self._half(self.down, self.down_grad, hi_dn, self.down_baseline, bot_rows, cols, False))
            marks = {0: labels[0], top_rows - 1: labels[1], height - 1: labels[2]}
        out = []
        for r, line in enumerate(body[:height]):
            row = Text(marks.get(r, "").rjust(gutter - 1) + " ", style="#6c6c6c", no_wrap=True)
            row.append_text(line)
            out.append(row)
        return out


def meter(value: float | None, width: int, grad: list[str], baseline: float | None = None) -> Text:
    """btop-style bar of ■ cells; ▪ marks the even-share point."""
    t = Text(no_wrap=True)
    n = 0 if value is None else round(min(max(value, 0.0), 1.0) * width)
    mark = None if baseline is None else min(round(baseline * width), width - 1)
    for i in range(width):
        if i < n:
            t.append("■", style=lerp_hex(grad, (i + 0.5) / width))
        elif i == mark:
            t.append("▪", style="#8a8a8a")
        else:
            t.append("■", style="#3a3a3a")
    return t


def pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.0f} days ago"


def trend(values: list[float | None], window: int = 10) -> float | None:
    vals = [v for v in values if v is not None]
    window = min(window, len(vals) // 2)
    if window < 2:
        return None
    return sum(vals[-window:]) / window - sum(vals[-2 * window:-window]) / window


def arrow(delta: float | None, scale: float) -> Text:
    if delta is None:
        return Text(" ")
    if delta > scale:
        return Text("▲", style="#5fd75f")
    if delta < -scale:
        return Text("▼", style="#ff5f5f")
    return Text("•", style="#8a8a8a")


def read_metrics(path: Path, state: dict) -> list[dict]:
    """Incrementally read metrics.jsonl; keep the latest record per update (resumes repeat some)."""
    by_update = state.setdefault("by_update", {})
    if not path.exists():
        return []
    with open(path) as f:
        f.seek(state.get("offset", 0))
        for line in f:
            if not line.endswith("\n"):
                break  # half-written line; read it next time
            state["offset"] = state.get("offset", 0) + len(line.encode())
            try:
                rec = json.loads(line)
                by_update[rec["update"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return [by_update[u] for u in sorted(by_update)]


def is_training(run_dir: Path) -> bool:
    pidfile = run_dir / "trainctl.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text().strip()), 0)
            return True
        except (ValueError, OSError):
            pass
    return bool(subprocess.run(["pgrep", "-f", "bin/arma-train"], capture_output=True, text=True).stdout.strip())


def trained_hours(recs: list[dict]) -> float:
    total = 0.0
    for a, b in zip(recs, recs[1:]):
        gap = b.get("time", 0) - a.get("time", 0)
        if 0 < gap < 600:  # longer gaps are pauses between sessions
            total += gap
    return total / 3600


def duel(rec: dict) -> float | None:
    vals = [rec[k] for k in ("win_duel_small", "win_duel_mid", "win_duel_std") if k in rec]
    return sum(vals) / len(vals) if vals else None


def box_panel(num: int, title: str, body, color: str, note: str = "", extra: str = "") -> Panel:
    t = Text()
    t.append(SUPER[num], style="bold #ffffff")
    t.append(title, style=f"bold {color}")
    if extra:
        t.append(f"  {extra}", style="bold #e4e4e4")
    sub = Text(note, style="#9e9e9e") if note else None
    return Panel(body, title=t, title_align="left", subtitle=sub, subtitle_align="right",
                 box=box.ROUNDED, border_style=color, padding=(0, 1))


def side(rows: list[tuple[str, Text | str]], note: str = "") -> Group:
    """Stats column beside a graph: label/value rows, then a dim one-line note."""
    g = Table.grid(padding=(0, 1))
    g.add_column(style="#9e9e9e", no_wrap=True)
    g.add_column(no_wrap=True, justify="right")
    for k, v in rows:
        g.add_row(k, v if isinstance(v, Text) else Text(v, style="bold #e4e4e4"))
    return Group(g, Text(note, style="#6c6c6c", no_wrap=True, overflow="ellipsis")) if note else Group(g)


def labelled_meter(value: float | None, grad: list[str], baseline: float | None = None, width: int = 10) -> Text:
    return Text.assemble(meter(value, width, grad, baseline), " ", Text(f"{pct(value):>4}", style="bold #e4e4e4"))


def graph_with_side(graph: AreaGraph, stats: Group, side_width: int, show_side: bool) -> Layout:
    lay = Layout()
    if show_side:
        lay.split_row(Layout(graph, ratio=1), Layout(stats, size=side_width))
    else:
        lay.update(graph)
    return lay


def win_text(v: float | None, even: float) -> Text:
    if v is None:
        return Text("–", style="#6c6c6c")
    c = "#5fd75f" if v > even + 0.05 else ("#ff5f5f" if v < even - 0.05 else "#e4e4e4")
    return Text(f"{100 * v:.0f}%", style=c)


def build(run_dir: Path, recs: list[dict], width: int = 160, height: int = 48) -> Layout:
    running = is_training(run_dir)
    last = recs[-1] if recs else {}
    latest = run_dir / "latest.pt"
    pool = sorted((run_dir / "pool").glob("u*.pt"))
    wide = width >= 130 and height >= 38  # full btop layout with speed, arenas and updates

    # ---- header bar
    head = Table.grid(expand=True)
    head.add_column(no_wrap=True, ratio=1, overflow="ellipsis")
    head.add_column(no_wrap=True, justify="right")
    left = Text(no_wrap=True, overflow="ellipsis")
    left.append(" armagetron-ai ", style="bold #000000 on #5f87af")
    left.append("  ")
    left.append("● TRAINING" if running else "■ STOPPED", style="bold #5fd75f" if running else "bold #ff5f5f")
    compact = width < 100
    if not compact:
        left.append(f"   run {run_dir.name}", style="#9e9e9e")
    if recs:
        left.append(f"   update {last['update']}", style="bold #e4e4e4")
        left.append(f"   {last['steps'] / 1e6:.2f} M {'dec' if compact else 'decisions'}", style="#e4e4e4")
        left.append(f"   {trained_hours(recs):.1f} h{'' if compact else ' trained'}", style="#e4e4e4")
    right = Text(no_wrap=True)
    right.append(f"saved {ago(time.time() - latest.stat().st_mtime)}" if latest.exists() else "no checkpoint yet",
                 style="#5fafd7")
    if width >= 100:
        right.append(f"   {len(pool)} past versions", style="#9e9e9e")
    right.append(f"   {time.strftime('%H:%M:%S')} ", style="#6c6c6c")
    head.add_row(left, right)
    header = Layout(Panel(head, box=box.ROUNDED, border_style=BORDER["head"], padding=(0, 0)), size=3)

    # ---- series
    survival = [r.get("ep_len", 0) / DECISIONS_PER_SECOND for r in recs]
    bot_small = [r.get("win_vs_ai_small") for r in recs]
    bot_full = [r.get("win_vs_ai_std") for r in recs]
    duels = [duel(r) for r in recs]
    ffa = [r.get("win_ffa4") for r in recs]
    decisive = [1 - r["ent"] / math.log(4) if "ent" in r else None for r in recs]
    predict = [max(r["value_explained"], 0) if "value_explained" in r else None for r in recs]
    speed = [r.get("sps") for r in recs]
    now = lambda xs: xs[-1] if xs else None  # noqa: E731
    span = f"updates {recs[0]['update']}–{last['update']}" if recs else "no updates yet"

    surv_vals = [v for v in survival if v]
    surv_stats = side([
        ("now", f"{survival[-1]:.0f} s" if recs else "–"),
        ("best", f"{max(surv_vals):.0f} s" if surv_vals else "–"),
        ("trend", arrow(trend(survival), 1.0)),
    ], "higher = stays alive longer")
    bot_stats = side([
        ("small", labelled_meter(now(bot_small), GRAD["green"], 0.5)),
        ("full", labelled_meter(now(bot_full), GRAD["pink"], 0.5)),
    ], "aim for 100%")
    self_stats = side([
        ("duels", labelled_meter(now(duels), GRAD["green"], 0.5, 8)),
        ("4-player", labelled_meter(now(ffa), GRAD["blue"], 0.25, 8)),
    ], "▪ = an even share")
    learn_stats = side([
        ("decisive", labelled_meter(now(decisive), GRAD["blue"], None, 8)),
        ("predicts", labelled_meter(now(predict), GRAD["purple"], None, 8)),
    ], "both rise slowly")
    sp_vals = [v for v in speed if v]
    recent_sp = sp_vals[-10:]
    speed_stats = side([
        ("now", f"{speed[-1]:,}/s" if recs and speed[-1] else "–"),
        ("avg", f"{sum(sp_vals) / len(sp_vals):,.0f}/s" if sp_vals else "–"),
        ("per hour", f"{sum(recent_sp) / len(recent_sp) * 3600 / 1e6:.1f} M" if recent_sp else "–"),
    ])

    g_surv = AreaGraph(survival, GRAD["teal"], fmt="{:.0f}s")
    g_bot = AreaGraph(bot_small, GRAD["green"], bot_full, GRAD["pink"], hi=1, down_hi=1,
                      baseline=0.5, down_baseline=0.5, fmt="{:.0%}")
    g_self = AreaGraph(duels, GRAD["green"], ffa, GRAD["blue"], hi=1, down_hi=1,
                       baseline=0.5, down_baseline=0.25, fmt="{:.0%}")
    g_learn = AreaGraph(decisive, GRAD["blue"], predict, GRAD["purple"], hi=1, down_hi=1, fmt="{:.0%}")
    g_speed = AreaGraph(speed, GRAD["purple"], fmt="{:.0f}")

    # a stats column only where the graph keeps at least 30 columns; otherwise the title carries the numbers
    def fits(panel_width: float, side_width: int) -> bool:
        return panel_width - 4 - side_width >= 30

    w_surv, w_bot, w_row2 = (width * 0.6, width * 0.4, width / 3) if wide else (width / 2, width / 2, width / 2)
    s_surv, s_bot = fits(w_surv, 27), fits(w_bot, 24)
    s_self, s_learn = fits(w_row2, 24), fits(w_row2, 24)
    p_surv = box_panel(1, "survival", graph_with_side(g_surv, surv_stats, 27, s_surv), BORDER["survival"],
                       f"seconds per round · {span}" if s_surv else "seconds per round",
                       "" if s_surv else (f"{survival[-1]:.0f} s" if recs else ""))
    p_bot = box_panel(2, "vs best bot", graph_with_side(g_bot, bot_stats, 24, s_bot), BORDER["bot"],
                      "▲ small arena  ▼ full-size", "" if s_bot else f"{pct(now(bot_small))} / {pct(now(bot_full))}")
    p_self = box_panel(3, "vs past self", graph_with_side(g_self, self_stats, 24, s_self), BORDER["self"],
                       "▲ duels  ▼ 4-player", "" if s_self else f"{pct(now(duels))} / {pct(now(ffa))}")
    p_learn = box_panel(4, "learning", graph_with_side(g_learn, learn_stats, 24, s_learn), BORDER["learn"],
                        "▲ decisive  ▼ predicts", "" if s_learn else f"{pct(now(decisive))} / {pct(now(predict))}")

    layout = Layout()
    if not wide:
        top, bottom = Layout(), Layout()
        top.split_row(Layout(p_surv), Layout(p_bot))
        bottom.split_row(Layout(p_self), Layout(p_learn))
        layout.split_column(
            header, Layout(top, ratio=1), Layout(bottom, ratio=1),
            Layout(Text(f" ctrl-c closes this · stop training: {TRAINCTL} stop", style="#6c6c6c",
                        no_wrap=True, overflow="ellipsis"), size=1),
        )
        return layout

    # ---- arenas: one row per arena type, like btop's process list
    arenas = Table(box=None, expand=True, show_edge=False, pad_edge=False, header_style="bold #d0d0d0")
    arenas.add_column("arena", no_wrap=True, ratio=3, overflow="ellipsis")
    arenas.add_column("against", style="#9e9e9e", no_wrap=True, ratio=3, overflow="ellipsis")
    arenas.add_column("win", justify="right", no_wrap=True, width=5)
    arenas.add_column("", no_wrap=True, width=1)
    arenas.add_column("", no_wrap=True, width=12)
    for key, (name, against, even) in ARENAS.items():
        k = f"win_{key}"
        if k not in last:
            continue
        arenas.add_row(name, against, win_text(last[k], even), arrow(trend([r.get(k) for r in recs]), 0.03),
                       meter(last[k], 12, GRAD["green"] if even == 0.5 else GRAD["blue"], even))

    # ---- recent updates, newest first
    updates = Table(box=None, expand=True, show_edge=False, pad_edge=False, header_style="bold #d0d0d0")
    for name in ("update", "time", "decisions", "per s", "survive", "bot S", "bot F", "duels", "4p",
                 "decisive", "kl"):
        updates.add_column(name, justify="right", no_wrap=True)
    third = (height - 3) // 3
    shown = max(third - 3, 1)
    for r in reversed(recs[-shown:]):
        updates.add_row(
            Text(str(r["update"]), style="bold #e4e4e4"),
            Text(time.strftime("%H:%M", time.localtime(r.get("time", 0))), style="#9e9e9e"),
            f"{r['steps'] / 1e6:.2f} M",
            Text(f"{r.get('sps', 0):,}", style="#b9a1ff"),
            Text(f"{r.get('ep_len', 0) / DECISIONS_PER_SECOND:.0f} s", style="#7fe8d8"),
            win_text(r.get("win_vs_ai_small"), 0.5), win_text(r.get("win_vs_ai_std"), 0.5),
            win_text(duel(r), 0.5), win_text(r.get("win_ffa4"), 0.25),
            Text(f"{100 * (1 - r['ent'] / math.log(4)):.0f}%" if "ent" in r else "–", style="#86b4ff"),
            Text(f"{r.get('kl', 0):.4f}", style="#9e9e9e"),
        )

    p_speed = box_panel(5, "speed", graph_with_side(g_speed, speed_stats, 20, True), BORDER["speed"],
                        "decisions per second")
    p_arenas = box_panel(6, "arenas", arenas, BORDER["arenas"], "rolling win rate per arena type")
    p_updates = box_panel(7, "updates", updates, BORDER["updates"], f"ctrl-c closes this · stop: {TRAINCTL} stop")

    row1, row2, row3 = Layout(), Layout(), Layout()
    row1.split_row(Layout(p_surv, ratio=3), Layout(p_bot, ratio=2))
    row2.split_row(Layout(p_self), Layout(p_learn), Layout(p_speed))
    row3.split_row(Layout(p_arenas, ratio=2), Layout(p_updates, ratio=3))
    layout.split_column(header, Layout(row1, ratio=1), Layout(row2, ratio=1), Layout(row3, ratio=1))
    return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/v2")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()
    run_dir = (PROJECT / args.run).resolve()
    state: dict = {}
    console = Console(width=args.width, height=args.height)

    def frame() -> Layout:
        w, h = console.size
        return build(run_dir, read_metrics(run_dir / "metrics.jsonl", state), w, h)

    if args.once:
        console.print(frame(), height=console.size.height)
        return
    try:
        with Live(frame(), console=console, screen=True, refresh_per_second=1) as live:
            while True:
                time.sleep(2)
                live.update(frame())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
