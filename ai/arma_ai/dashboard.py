"""Live terminal dashboard for a training run.

    ./trainctl watch            (or: uv run arma-dash)

Redraws every two seconds from runs/<run>/metrics.jsonl. Ctrl-C closes the dashboard; training
keeps running.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT = Path(__file__).resolve().parent.parent
DECISIONS_PER_SECOND = 20  # the engine asks for a move every 0.05 s of game time

# braille cells are 2 dots wide and 4 high; bit for (row, column) inside a cell
_DOT = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


class Chart:
    """Line chart drawn with braille dots so it is smooth even in a small terminal."""

    def __init__(self, series: list[tuple[str, list[float | None], str]], xs: list[int],
                 ymin: float | None = None, ymax: float | None = None, fmt: str = "{:.2f}",
                 baseline: float | None = None):
        self.series = series
        self.xs = xs
        self.ymin, self.ymax = ymin, ymax
        self.fmt = fmt
        self.baseline = baseline

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        height = options.height or 8
        yield from self.render(width, height)

    def render(self, width: int, height: int) -> list[Text]:
        legend = Text()
        for label, _, color in self.series:
            legend.append("● ", style=color)
            legend.append(label + "   ", style="dim")
        values = [v for _, vals, _ in self.series for v in vals if v is not None and math.isfinite(v)]
        rows = max(height - 2, 3)
        if not values or len(self.xs) < 2:
            out = [legend] + [Text("") for _ in range(rows)]
            out[rows // 2 + 1] = Text("waiting for more updates", style="dim", justify="center")
            return out + [Text("")]
        lo = self.ymin if self.ymin is not None else min(values)
        hi = self.ymax if self.ymax is not None else max(values)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        if self.ymin is None or self.ymax is None:  # a little headroom for auto-scaled axes
            pad = (hi - lo) * 0.08
            lo = lo - pad if self.ymin is None else lo
            hi = hi + pad if self.ymax is None else hi
        labels = [self.fmt.format(hi), self.fmt.format((hi + lo) / 2), self.fmt.format(lo)]
        gutter = max(len(s) for s in labels) + 1
        cols = max(width - gutter - 1, 8)
        W, H = cols * 2, rows * 4
        bits = [[0] * cols for _ in range(rows)]
        color = [[None] * cols for _ in range(rows)]

        def dot(x: int, y: int, c: str) -> None:
            if 0 <= x < W and 0 <= y < H:
                r, k = y // 4, x // 2
                bits[r][k] |= _DOT[y % 4][x % 2]
                color[r][k] = c

        def to_y(v: float) -> int:
            return round((hi - v) / (hi - lo) * (H - 1))

        n = len(self.xs)
        if self.baseline is not None and lo <= self.baseline <= hi:
            y = to_y(self.baseline)
            for x in range(0, W, 3):
                dot(x, y, "grey35")
        for _, vals, c in self.series:
            prev = None
            for i, v in enumerate(vals):
                if v is None or not math.isfinite(v):
                    continue
                x, y = round(i * (W - 1) / (n - 1)), to_y(min(max(v, lo), hi))
                if prev is None:
                    dot(x, y, c)
                else:  # Bresenham line from the previous point
                    x0, y0 = prev
                    dx, dy = abs(x - x0), -abs(y - y0)
                    sx, sy = (1 if x > x0 else -1), (1 if y > y0 else -1)
                    err = dx + dy
                    while True:
                        dot(x0, y0, c)
                        if x0 == x and y0 == y:
                            break
                        e2 = 2 * err
                        if e2 >= dy:
                            err += dy
                            x0 += sx
                        if e2 <= dx:
                            err += dx
                            y0 += sy
                prev = (x, y)

        out = [legend]
        label_rows = {0: labels[0], rows // 2: labels[1], rows - 1: labels[2]}
        for r in range(rows):
            line = Text(label_rows.get(r, "").rjust(gutter - 1) + " ", style="dim")
            line.append("┤" if r in label_rows else "│", style="grey50")
            for k in range(cols):
                if bits[r][k]:
                    line.append(chr(0x2800 + bits[r][k]), style=color[r][k])
                else:
                    line.append(" ")
            out.append(line)
        first, last = f" update {self.xs[0]} ", f" update {self.xs[-1]} "
        fill = max(cols - len(first) - len(last), 0)
        out.append(Text(" " * gutter + "└" + first + "─" * fill + last, style="grey50"))
        return out[:height] if height >= 3 else out


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


def training_status(run_dir: Path) -> tuple[bool, str]:
    pidfile = run_dir / "trainctl.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text().strip()), 0)
            return True, "training"
        except (ValueError, OSError):
            pass
    found = subprocess.run(["pgrep", "-f", "bin/arma-train"], capture_output=True, text=True).stdout.strip()
    return (True, "training") if found else (False, "stopped")


def ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.0f} days ago"


def trained_hours(recs: list[dict]) -> float:
    total = 0.0
    for a, b in zip(recs, recs[1:]):
        gap = b.get("time", 0) - a.get("time", 0)
        if 0 < gap < 600:  # longer gaps are pauses between runs
            total += gap
    return total / 3600


def avg(rec: dict, keys: list[str]) -> float | None:
    vals = [rec[k] for k in keys if k in rec]
    return sum(vals) / len(vals) if vals else None


def pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def build(run_dir: Path, recs: list[dict]) -> Layout:
    running, state = training_status(run_dir)
    latest_ck = run_dir / "latest.pt"
    xs = [r["update"] for r in recs]
    last = recs[-1] if recs else {}

    head = Table.grid(expand=True)
    head.add_column(ratio=1)
    head.add_column(justify="right")
    status = Text("● TRAINING" if running else "■ STOPPED", style="bold green" if running else "bold red")
    status.append(f"   run {run_dir.name}", style="dim")
    saved = f"saved {ago(time.time() - latest_ck.stat().st_mtime)}" if latest_ck.exists() else "no checkpoint yet"
    head.add_row(status, Text(saved, style="cyan"))
    if recs:
        facts = Text()
        facts.append(f"update {last['update']}", style="bold")
        facts.append(f"   {last['steps'] / 1e6:.2f} M decisions of practice", style="")
        facts.append(f"   {trained_hours(recs):.1f} h trained", style="")
        facts.append(f"   {last.get('sps', 0):,} decisions/s" if running else "", style="dim")
        head.add_row(facts, Text(time.strftime("%H:%M:%S"), style="dim"))

    survival = [r.get("ep_len", 0) / DECISIONS_PER_SECOND for r in recs]
    vs_bot = [("small arena", [r.get("win_vs_ai_small") for r in recs], "magenta"),
              ("full-size arena", [r.get("win_vs_ai_std") for r in recs], "yellow")]
    duel_keys = ["win_duel_small", "win_duel_mid", "win_duel_std"]
    vs_self = [("duels", [avg(r, duel_keys) for r in recs], "green"),
               ("4-player", [r.get("win_ffa4") for r in recs], "cyan")]
    health = [("decisiveness", [1 - r["ent"] / math.log(4) if "ent" in r else None for r in recs], "blue"),
              ("outcome prediction", [max(r["value_explained"], 0) if "value_explained" in r else None
                                      for r in recs], "bright_magenta")]

    def panel(title: str, chart: Chart) -> Panel:
        return Panel(chart, title=title, title_align="left", border_style="grey35", padding=(0, 1))

    charts = Layout()
    top, bottom = Layout(name="top"), Layout(name="bottom")
    top.split_row(
        Layout(panel("Survival time per round (seconds)",
                     Chart([("average", survival, "bright_cyan")], xs, ymin=0, fmt="{:.0f}"))),
        Layout(panel("Win rate vs the game's best bot",
                     Chart(vs_bot, xs, ymin=0, ymax=1, fmt="{:.0%}", baseline=0.5))),
    )
    bottom.split_row(
        Layout(panel("Win rate vs its past self",
                     Chart(vs_self, xs, ymin=0, ymax=1, fmt="{:.0%}", baseline=0.5))),
        Layout(panel("Learning health",
                     Chart(health, xs, ymin=0, ymax=1, fmt="{:.0%}"))),
    )
    charts.split_column(top, bottom)

    guide = Table.grid(expand=True, padding=(0, 2))
    guide.add_column(style="bold", no_wrap=True)
    guide.add_column(justify="right", no_wrap=True)
    guide.add_column(style="dim", no_wrap=True, overflow="ellipsis")
    dec = 1 - last["ent"] / math.log(4) if "ent" in last else None
    pred = max(last["value_explained"], 0) if "value_explained" in last else None
    guide.add_row("Survival", f"{survival[-1]:.0f} s" if recs else "–", "rising = it is learning to stay alive")
    guide.add_row("vs best bot", f"{pct(last.get('win_vs_ai_small'))} / {pct(last.get('win_vs_ai_std'))}",
                  "small / full-size arena; should climb towards 100%")
    guide.add_row("vs past self", pct(avg(last, duel_keys)), "above 50% = it beats its older versions")
    guide.add_row("4-player", pct(last.get("win_ffa4")), "25% = an even share between four players")
    guide.add_row("Health", f"{pct(dec)} / {pct(pred)}", "decisiveness / outcome prediction; both should slowly rise")
    guide.add_row("Speed", f"{last.get('sps', 0):,}/s" if recs else "–", "decisions per second; more = faster progress")

    layout = Layout()
    layout.split_column(
        Layout(Panel(head, border_style="green" if running else "red", padding=(0, 1)), size=4),
        Layout(charts, name="charts"),
        Layout(Panel(guide, title="Latest", title_align="left", border_style="grey35", padding=(0, 1)), size=8),
        Layout(Text(" Ctrl-C closes this view; training keeps running", style="dim"), size=1),
    )
    return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()
    run_dir = (PROJECT / args.run).resolve()
    state: dict = {}
    console = Console(width=args.width, height=args.height)
    if args.once:
        console.print(build(run_dir, read_metrics(run_dir / "metrics.jsonl", state)), height=args.height or 40)
        return
    try:
        with Live(build(run_dir, read_metrics(run_dir / "metrics.jsonl", state)), console=console,
                  screen=True, refresh_per_second=1) as live:
            while True:
                time.sleep(2)
                live.update(build(run_dir, read_metrics(run_dir / "metrics.jsonl", state)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
