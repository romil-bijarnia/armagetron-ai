"""Live training activity and recorded learning results in a full-screen terminal view.

    uv run arma-dash

Reads progress.json, metrics.jsonl and V4 checkpoints. Ctrl-C closes the view; training continues.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .progress_state import process_started, warmup_state

PROJECT = Path(__file__).resolve().parent.parent
DECISIONS_PER_SECOND = 20  # the engine asks for a move every 0.05 s of game time
HEARTBEAT_TIMEOUT = 30

# braille cells are 2 dots wide and 4 high; bit for (row, column) inside a cell
_DOT = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

ACCENT = "#8aa8c7"
DIM = "#8b949e"
BORDER = "#414b57"
WARNING = "#d3b775"
BLUE = ["#526b83", ACCENT]
NEUTRAL = ["#555e68", "#a0a9b2"]

ARENAS = {
    "duel_-3": "duel · size -3",
    "duel_-1.5": "duel · size -1.5",
    "duel_0": "duel · size 0",
    "ffa4_-1": "4-player · size -1",
    "ffa4_0": "4-player · size 0",
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
    """Recorded values on a shared time axis; paired series use separate upper/lower scales."""

    def __init__(self, up: list[float | None], up_grad: list[str], down: list[float | None] | None = None,
                 down_grad: list[str] | None = None, hi: float | None = None, down_hi: float | None = None,
                 baseline: float | None = None, down_baseline: float | None = None, fmt: str = "{:.0f}",
                 lo: float = 0):
        self.up, self.up_grad, self.down, self.down_grad = up, up_grad, down, down_grad
        self.hi, self.down_hi = hi, down_hi
        self.baseline, self.down_baseline = baseline, down_baseline
        self.fmt = fmt
        self.lo = lo

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
        if width <= 0 or height <= 0:
            return []
        if not any(v is not None and math.isfinite(v) for v in self.up + (self.down or [])):
            return [Text("No data yet" if r == height // 2 else "", style="#6c6c6c") for r in range(height)]
        mirrored = self.down is not None
        hi_up = self._top(self.up, self.hi)
        hi_dn = self._top(self.down, self.down_hi) if mirrored else 0.0
        labels = [self.fmt.format(hi_up), self.fmt.format(0)] + ([self.fmt.format(hi_dn)] if mirrored else [])
        gutter = max(len(s) for s in labels) + 1
        cols = max(width - gutter, 1)
        if not mirrored and self.lo < 0:
            # Signed learning statistics need an actual zero axis: a negative explained
            # variance must remain visible, rather than becoming a misleading zero.
            labels = [self.fmt.format(hi_up), self.fmt.format(0), self.fmt.format(self.lo)]
            gutter = max(len(s) for s in labels) + 1
            cols = max(width - gutter, 1)
            W, H = cols * 2, max(height * 4, 1)
            scale = max(hi_up - self.lo, 1e-9)
            dot = lambda v: min(max(round((hi_up - v) / scale * (H - 1)), 0), H - 1)
            zero = dot(0)
            cells = [[0] * cols for _ in range(height)]
            for x in range(0, W, 4):
                cells[zero // 4][x // 2] |= _DOT[zero % 4][x % 2]
            for x, v in enumerate(resample(self.up, W)):
                if v is None:
                    continue
                y = dot(v)
                for d in range(min(y, zero), max(y, zero) + 1):
                    cells[d // 4][x // 2] |= _DOT[d % 4][x % 2]
            body = [Text("".join(chr(0x2800 + bits) if bits else " " for bits in row),
                         style=lerp_hex(self.up_grad, 0.65), no_wrap=True) for row in cells]
            marks = {zero // 4: labels[1], 0: labels[0], height - 1: labels[2]}
        elif not mirrored:
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
            row.truncate(width)
            out.append(row)
        return out


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


def read_progress(path: Path) -> dict:
    """A trainer atomically replaces this heartbeat, including before its first update."""
    try:
        progress = json.loads(path.read_text())
        return progress if isinstance(progress, dict) else {}
    except (OSError, ValueError):
        return {}


PROCESS_CACHE_SECONDS = 3
_process_cache: dict = {"time": 0.0, "processes": None}


def _list_processes() -> dict[int, str] | None:
    """One process snapshot, or None when the listing itself failed. Zombie/reused PID files are
    not training evidence. ps runs in its own session so the terminal's title does not flash its
    name every time the view refreshes."""
    try:
        result = subprocess.run(["ps", "-axww", "-o", "pid=,stat=,command="], capture_output=True,
                                text=True, timeout=5, check=False, start_new_session=True)
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and not parts[1].startswith("Z"):
            found[int(parts[0])] = parts[2]
    return found


def _processes() -> dict[int, str]:
    """The process list, refreshed at most every few seconds. A listing that fails (full-mode
    training saturates every core, and ps can be slow once) keeps the previous snapshot rather
    than reporting a live trainer as stopped."""
    now = time.monotonic()
    cached = _process_cache["processes"]
    if cached is not None and now - _process_cache["time"] < PROCESS_CACHE_SECONDS:
        return cached
    listed = _list_processes()
    if listed is not None:
        _process_cache.update(time=now, processes=listed)
        return listed
    return cached if cached is not None else {}


def _process_role(command: str, run_dir: Path, *, heartbeat_pid: bool = False) -> str | None:
    try:
        args = shlex.split(command)
    except ValueError:
        return None
    names = {Path(a).name for a in args}
    teacher = "arma-teach" in names or "arma_ai.teach" in args
    imitation = bool({"imitate", "arma-imitate"} & names) or "arma_ai.imitate" in args
    supervised = "trainctl" in names and "_supervise" in args
    if not (teacher or imitation or supervised):
        return None
    # A fresh heartbeat is written by this run's trainer and identifies its exact PID.
    # Other processes must also name this run, not merely be trainers somewhere else.
    if not heartbeat_pid:
        run = "runs/blacklight"
        for i, arg in enumerate(args):
            if arg == "--run" and i + 1 < len(args):
                run = args[i + 1]
            elif arg.startswith("--run="):
                run = arg.split("=", 1)[1]
        if (PROJECT / run).resolve() != run_dir.resolve():
            return None
    return "trainer" if (teacher or imitation) and Path(args[0]).name != "uv" else "supervisor"


def _imitation_process(command: str) -> bool:
    try:
        args = shlex.split(command)
    except ValueError:
        return False
    return bool({"imitate", "arma-imitate"} & {Path(a).name for a in args}) or "arma_ai.imitate" in args


@dataclass(frozen=True)
class TrainingStatus:
    label: str
    detail: str
    running: bool = False
    fresh: bool = False
    method: str | None = None


def training_status(run_dir: Path, progress: dict, recs: list[dict]) -> TrainingStatus:
    processes = _processes()
    pid = progress.get("pid")
    heartbeat_command = processes.get(pid, "") if isinstance(pid, int) else ""
    recorded_run = progress.get("run")
    same_run = not recorded_run or Path(recorded_run).resolve() == run_dir.resolve()
    heartbeat_live = same_run and _process_role(heartbeat_command, run_dir, heartbeat_pid=True) == "trainer"
    heartbeat_time = progress.get("time", 0)
    age = time.time() - heartbeat_time if isinstance(heartbeat_time, (int, float)) else math.inf
    fresh = heartbeat_live and -5 <= age <= HEARTBEAT_TIMEOUT
    phase = progress.get("phase")
    method = "imitate" if heartbeat_live and _imitation_process(heartbeat_command) else progress.get("trainer")
    if heartbeat_live:
        if phase in ("stopped", "failed"):
            return TrainingStatus(phase.upper(), "Trainer has finished; showing its last recorded progress.", method=method)
        if not fresh:
            return TrainingStatus("STALE", f"Trainer process exists; heartbeat {ago(max(age, 0))}.", method=method)
        if method == "imitate":
            labels = {"collecting": "COLLECTING", "learning": "IMITATING", "validating": "VALIDATING",
                      "starting": "STARTING", "stopping": "SAVING"}
            detail = {"collecting": "Recording native teacher decisions.", "learning": "Learning the teacher's actions.",
                      "validating": "Checking held-out teacher lives.", "stopping": "Saving the current V4 checkpoint."}
            return TrainingStatus(labels.get(phase, "IMITATING"), detail.get(phase, "Imitation heartbeat is live."),
                                  True, True, method)
        labels = {"starting": "STARTING", "warming_up": "WARMING UP", "learning": "TRAINING",
                  "stopping": "SAVING"}
        detail = {"starting": "Loading the trainer and preparing self-play.",
                  "warming_up": "Collecting replay and completed search targets before learning.",
                  "learning": "Learning from self-play and search targets.",
                  "stopping": "Finishing training and saving the checkpoint."}.get(phase, "Trainer heartbeat is live.")
        return TrainingStatus(labels.get(phase, "TRAINING"), detail, True, True, method)
    roles = {_process_role(command, run_dir) for command in processes.values()}
    trainers = [command for command in processes.values() if _process_role(command, run_dir) == "trainer"]
    if trainers:
        method = "imitate" if any(_imitation_process(command) for command in trainers) else "teach"
    if "trainer" in roles:
        if progress:
            return TrainingStatus("STARTING", "Trainer process is live; waiting for its current heartbeat.", True, method=method)
        if method == "imitate":
            return TrainingStatus("STARTING", "Imitation is starting; waiting for its first heartbeat.", True, method=method)
        return TrainingStatus("TRAINING", "Waiting for first metrics." if not recs else
                              "Trainer process is live; showing its logged updates.", True, method=method)
    if "supervisor" in roles:
        return TrainingStatus("STARTING", "Supervisor is live; waiting for the trainer.", method=method)
    if phase == "failed":
        return TrainingStatus("FAILED", "Trainer stopped after an error; see train.log.", method=method)
    return TrainingStatus("STOPPED", "No live trainer for this run." +
                          (" Showing recorded results." if recs or progress else " No training data yet."), method=method)


def is_training(run_dir: Path) -> bool:
    return training_status(run_dir, read_progress(run_dir / "progress.json"), []).running


def policy_focus(rec: dict, run_dir: Path, progress: dict | None = None) -> float | None:
    """Entropy normalized over the six controls; this is not a strength score."""
    if "ent" not in rec:
        return None
    return min(max(1 - rec["ent"] / math.log(6), 0.0), 1.0)


def _line(value: str, style: str = DIM) -> Text:
    return Text(value, style=style, no_wrap=True, overflow="ellipsis")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value, fmt: str = ",.0f") -> str:
    return format(value, fmt) if _finite(value) else "not reported"


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"


def ago(seconds: float) -> str:
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02}s ago"
    return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02}m ago"


def _percent(fraction: float | None) -> str:
    if fraction is None:
        return "unavailable"
    # Rounding an incomplete gate to 100% hides the last requirement.
    return "99%" if .995 <= fraction < 1 else f"{min(max(fraction, 0), 1):.0%}"


def _bar(label: str, fraction: float | None, detail: str, width: int, *, due: bool = False) -> Text:
    prefix = f"{label:<18} "
    suffix = f" {_percent(fraction):>5}  {detail}" + (" due" if due else "")
    size = min(28, max(3, width - len(prefix) - len(suffix)))
    if fraction is None:
        return _line(f"{label}: {detail or 'counts unavailable'}")
    filled = round(min(max(fraction, 0), 1) * size)
    row = Text.assemble((prefix, ACCENT), ("━" * filled, WARNING if due else ACCENT),
                        ("━" * (size - filled), BORDER), (suffix, DIM))
    row.no_wrap, row.overflow = True, "ellipsis"
    return row


def _panel(title: str, body, note: str = "") -> Panel:
    return Panel(body, title=_line(title, ACCENT), title_align="left", box=box.ROUNDED,
                 border_style=BORDER, padding=(0, 1),
                 subtitle=_line(note) if note else None, subtitle_align="left")


def _eligible(records: list[dict], progress: dict) -> list[dict]:
    if "update" not in progress:
        return records
    return [row for row in records if row.get("update", 0) <= progress["update"]
            and row.get("time", 0) <= progress.get("time", math.inf) + 1]


def _activity(progress: dict, current: dict, status: TrainingStatus, width: int,
              now: float, started: float | None, saved_at: float | None) -> tuple[Panel, int]:
    cfg = progress.get("config", {})
    live = status.running and status.fresh
    update = current.get("update")
    title = "Live activity" if live else "Recorded activity"
    lead = f"Update {update:,}" if _finite(update) else "Update not reported"
    lead += (f" · {_number(progress.get('sps'))} decisions/s" if live else " · no live rate")
    if live and started is not None:
        lead += f" · elapsed {_duration(now - started)}"
    lines = [_line(lead, "bold")]
    counters = [f"{current[key]:,} {label}" for key, label in
                (("steps", "decisions"), ("trained", "trained"), ("games", "rounds")) if _finite(current.get(key))]
    if counters:
        lines.append(_line(" · ".join(counters)))
    warming = progress.get("phase") in ("starting", "warming_up")
    inner_width = width - 4
    if warming:
        state = warmup_state(progress)
        lines.append(_bar("Warmup readiness", state["fraction"], "", inner_width))
        for gate in state["gates"]:
            detail = (f"{gate['current']:,}/{gate['target']:,}" if gate["current"] is not None and gate["target"] is not None
                      else "counts unavailable")
            lines.append(_bar(gate["label"], gate["fraction"], detail, inner_width))
        waiting = "Waiting for " + ", ".join(label.lower() for label in state["waiting"])
        lines.append(_line(waiting if not state["complete"] else "All targets ready; awaiting first learner update.",
                           DIM if live else WARNING))
    elif live and progress.get("phase") == "learning":
        ready, capacity = progress.get("ready_samples"), cfg.get("ring")
        if _finite(ready) and _finite(capacity) and capacity > 4096:
            lines.append(_bar("Replay ready", ready / (capacity - 4096), f"{ready:,}/{capacity - 4096:,}", inner_width))
        every = cfg.get("log_every")
        if _finite(update) and isinstance(every, int) and every > 0:
            lines.append(_bar("Next report", update % every / every, f"{update % every}/{every} updates", inner_width))
        interval = cfg.get("save_seconds")
        if saved_at is not None and _finite(interval) and interval > 0:
            elapsed = max(0, now - saved_at)
            lines.append(_bar("Next save", elapsed / interval, f"{_duration(elapsed)}/{_duration(interval)}",
                              inner_width, due=elapsed >= interval))
        snapshot = cfg.get("snapshot_every")
        if _finite(update) and isinstance(snapshot, int) and snapshot > 0:
            lines.append(_bar("Next snapshot", update % snapshot / snapshot,
                              f"u{(update // snapshot + 1) * snapshot:,}", inner_width))
        target = progress.get("max_updates", cfg.get("max_updates"))
        if _finite(update) and _finite(target) and target > 0:
            lines.append(_bar("Update target", update / target, f"{update:,}/{target:,}", inner_width))
        elif target == 0:
            lines.append(_line("Training continues without a fixed update limit."))
    if not live or not current:
        lines.append(_line(status.detail, WARNING if status.label in ("STALE", "FAILED") else DIM))
    return _panel(title, Group(*lines)), len(lines) + 2


def _score_pairs(last: dict) -> list[tuple[str, str]]:
    pairs = [("win_vs_hunter", "Space fighter"), ("win_vs_past", "V4 snapshots"), ("win_exploiter", "Exploiter vs target")]
    if "win_vs_exploiter" in last or "win_vs_exploiter" in last.get("match_results", {}):
        pairs.append(("win_vs_exploiter", "Main vs exploiter"))  # reported once such a round has finished
    return pairs


def _scores(last: dict, *, bars: bool = False) -> Table:
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(ratio=2, no_wrap=True, overflow="ellipsis")
    if bars:
        table.add_column(width=12, no_wrap=True)
    for _ in range(3):
        table.add_column(justify="right", no_wrap=True)
    header = ["Opponent"] + ([""] if bars else []) + ["Score", "W / D / L", "Rounds"]
    table.add_row(*[_line(value) for value in header])
    for key, name in _score_pairs(last):
        counts = last.get("match_results", {}).get(key)
        score = counts.get("score", last.get(key)) if isinstance(counts, dict) else last.get(key)
        if isinstance(counts, dict) and counts.get("games", 0) > 0:
            score = (counts["wins"] + .5 * counts["draws"]) / counts["games"]
        cells = [_line(name, "default")]
        if bars:
            if _finite(score):
                filled = round(min(max(score, 0), 1) * 12)
                cells.append(Text.assemble(("━" * filled, ACCENT), ("━" * (12 - filled), BORDER)))
            else:
                cells.append(_line(""))
        cells.append(_line(f"{score:.1%}" if _finite(score) else "—", "default"))
        if isinstance(counts, dict):
            cells.extend([_line(f"{counts['wins']} / {counts['draws']} / {counts['losses']}"),
                          _line(f"n={counts['games']}")])
        else:
            cells.extend([_line("count unavailable" if _finite(score) else "—"), _line("—")])
        table.add_row(*cells)
    return table


def _chart(title: str, graph: AreaGraph, note: str = "") -> Panel:
    return _panel(title, graph, note)


def _arenas(last: dict) -> Panel:
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(no_wrap=True, ratio=1)
    table.add_column(no_wrap=True, justify="right")
    table.add_row(_line("Self-play arena"), _line("Mean round duration"))
    for key, name in ARENAS.items():
        value = last.get(f"len_{key}")
        table.add_row(_line(name, "default"), _line(f"{value / DECISIONS_PER_SECOND:.1f} s" if _finite(value) else "—"))
    return _panel("Arena reports", table, "Game time per completed round")


def _reports(records: list[dict], rows: int, now: float) -> Panel:
    table = Table.grid(padding=(0, 2), expand=True)
    for _ in range(7):
        table.add_column(no_wrap=True, justify="right")
    table.add_row(*[_line(value) for value in ("Update", "Age", "Policy loss", "Value loss", "Gain", "Value EV",
                                               "Decisions/s")])
    for row in reversed(records[-rows:]):
        table.add_row(_line(f"{row['update']:,}", "default"), _line(ago(now - row.get("time", now))),
                      _line(_number(row.get("pg"), ".4f")), _line(_number(row.get("vf"), ".4f")),
                      _line(_number(row.get("search_kl"), ".3f")),
                      _line(_number(row.get("value_explained"), ".3f")), _line(_number(row.get("sps"))))
    return _panel("Recent saved reports", table)


def build(run_dir: Path, recs: list[dict], width: int = 160, height: int = 48,
          progress: dict | None = None) -> Layout:
    progress = read_progress(run_dir / "progress.json") if progress is None else progress
    status = training_status(run_dir, progress, recs)
    now = time.time()
    started = progress.get("started_at")
    if not _finite(started):
        started = process_started(progress.get("pid")) if status.running else None
    if width < 76 or height < 22 or (status.method or progress.get("trainer")) == "imitate":
        from .cli_progress import build as compact_view
        return Layout(compact_view(run_dir, progress, recs, status, width=width, height=height,
                                   now=now, started=started))
    history = _eligible(recs, progress)
    last = history[-1] if history else {}
    current = progress or last
    cfg = progress.get("config", {})
    try:
        saved_at = (run_dir / "latest.pt").stat().st_mtime
    except OSError:
        saved_at = None
    live = status.running and status.fresh
    wide = width >= 130 and height >= 38
    header = Table.grid(expand=True, padding=(0, 1))
    header.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    header.add_column(justify="right", no_wrap=True)
    color = ACCENT if live else WARNING if status.label in ("STARTING", "STALE", "FAILED") else DIM
    header.add_row(_line(f"armagetron / {run_dir.name}", "bold"), _line(status.label, color))
    settings = [str(progress.get("mode", cfg.get("mode", ""))), str(cfg.get("device", "")).upper()]
    if _finite(cfg.get("games")):
        settings.append(f"{cfg['games']} arenas")
    if _finite(cfg.get("combat_prob")):
        settings.append(f"combat starts {cfg['combat_prob']:.0%}")
    saved = ago(now - saved_at) if saved_at is not None else "not saved"
    if width < 100 and saved_at is not None:
        saved = saved.removesuffix(" ago").replace(" ", "")
    header.add_row(_line(" · ".join(value for value in settings if value)),
                   _line(f"Checkpoint {saved}"))
    activity, activity_height = _activity(progress, current, status, width, now, started, saved_at)
    if last:
        if started is not None and last.get("time", 0) < started:
            session = "previous session"
        else:
            session = "current session" if live and started is not None else "session unknown" if live else "recorded"
        report_label = f"Saved report u{last['update']:,} · {session} · {ago(now - last.get('time', now))}"
        if width >= 100:
            report_label += f" · history u{history[0]['update']:,}–u{last['update']:,}"
    else:
        report_label = "Saved report: none yet. Charts await the first logged update."
    report = Layout(_line(report_label), size=1)
    heartbeat = ago(now - progress["time"]) if _finite(progress.get("time")) else "unavailable"
    footer = Layout(_line(f"Ctrl-C closes view; trainer unaffected. · Heartbeat {heartbeat}"), size=1)
    duration = [row.get("ep_len", math.nan) / DECISIONS_PER_SECOND for row in history]
    explained = [row.get("value_explained") for row in history]
    variance_low = min(-1, min((v for v in explained if _finite(v)), default=0) * 1.1)
    round_value = f" · {last['ep_len'] / DECISIONS_PER_SECOND:.1f} s" if _finite(last.get("ep_len")) else ""
    ev_value = f" · {last['value_explained']:.3f}" if _finite(last.get("value_explained")) else ""
    focus = policy_focus(last, run_dir, progress)
    p_duration = _chart("Mean round duration" + round_value, AreaGraph(duration, BLUE, fmt="{:.0f}s"),
                        "Recorded game seconds per round")
    p_value = _chart("Value EV" + ev_value, AreaGraph(explained, BLUE, hi=1, lo=variance_low, fmt="{:.2f}"),
                     (f"Policy focus {focus:.0%} · " if focus is not None and width >= 100 else "") + "0 = mean prediction")
    p_losses = _chart("Policy / value loss", AreaGraph([row.get("pg") for row in history], BLUE,
                       [row.get("vf") for row in history], NEUTRAL, fmt="{:.3f}"), "Policy above · value below")
    p_speed = _chart("Logged throughput", AreaGraph([row.get("sps") for row in history], BLUE),
                     "Decisions/s in saved reports")
    gain_value = f" · {last['search_kl']:.3f}" if _finite(last.get("search_kl")) else ""
    p_gain = _chart("Search gain" + gain_value, AreaGraph([row.get("search_kl") for row in history], BLUE,
                    [row.get("ent") for row in history], NEUTRAL, fmt="{:.2f}"),
                    "KL above · entropy below")
    scores = _scores(last, bars=wide)
    score_rows = 1 + len(_score_pairs(last))  # the header and one row per reported matchup
    layout = Layout()
    base = [Layout(header, size=2), Layout(activity, size=activity_height), report]
    if wide:
        matchup_body = Layout()
        matchup_body.split_column(Layout(scores, size=score_rows), Layout(AreaGraph(
            [row.get("win_vs_hunter") for row in history], BLUE,
            [row.get("win_vs_past") for row in history], NEUTRAL, hi=1, down_hi=1, fmt="{:.0%}")))
        p_scores = _panel("Training matches", matchup_body, "Space fighter above · V4 snapshots below · draws = ½")
        first, second, third = Layout(), Layout(), Layout()
        first.split_row(Layout(p_scores, ratio=3), Layout(p_gain, ratio=2), Layout(p_duration, ratio=2))
        second.split_row(Layout(p_value), Layout(p_losses), Layout(p_speed))
        third.split_row(Layout(_arenas(last), ratio=2), Layout(_reports(history, 7, now), ratio=3))
        layout.split_column(*base, Layout(first, ratio=1), Layout(second, ratio=1), Layout(third, size=10), footer)
    else:
        charts = Layout()
        charts.split_row(Layout(p_duration), Layout(p_value))
        p_scores = _panel("Training matches", scores, "Observed training outcomes · draws = ½")
        sections = [*base, Layout(p_scores, size=score_rows + 2), Layout(charts, ratio=1)]
        if height >= 30:
            extra = Layout()
            extra.split_row(Layout(p_losses), Layout(p_gain), Layout(p_speed))
            sections.append(Layout(extra, ratio=1))
        layout.split_column(*sections, footer)
    return layout

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/blacklight")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--simple", action="store_true", help="compact inline training console")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()
    run_dir = (PROJECT / args.run).resolve()
    state: dict = {}
    console = Console(width=args.width, height=args.height)

    if args.simple:
        from .cli_progress import run
        run(run_dir, console, once=args.once)
        return

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
