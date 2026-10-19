"""One updating training line shared by foreground learning and its terminal monitor."""

from __future__ import annotations

import math

from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn
from rich.table import Column
from rich.text import Text

BLUE = "#8aa8c7"
DIM = "#8b949e"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def elapsed_text(seconds):
    seconds = max(0, int(seconds))
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours}:{minutes:02}:{seconds:02}"


class Percentage(ProgressColumn):
    def render(self, task):
        if not task.fields.get("measured") or task.total is None:
            return Text("—", style=DIM)
        percent = 100 if task.completed >= task.total else min(round(task.percentage), 99)
        return Text(f"{percent:3}%", style="progress.percentage")


class MeasuredBar(BarColumn):
    def render(self, task):
        return super().render(task) if task.fields.get("measured") else Text("…", style=DIM)


def progress_bar(console):
    return Progress(
        TextColumn("{task.description}", table_column=Column(no_wrap=True, overflow="ellipsis", max_width=44)),
        MeasuredBar(bar_width=None, complete_style=BLUE, finished_style=BLUE), Percentage(),
        TextColumn("{task.fields[counts]}", table_column=Column(no_wrap=True)),
        TextColumn("{task.fields[elapsed]}", style="progress.elapsed", table_column=Column(no_wrap=True)),
        TextColumn("{task.fields[saved]}", style=DIM, table_column=Column(no_wrap=True)),
        console=console, expand=True)


def add_task(bar):
    return bar.add_task("Resuming Blacklight", total=None, measured=False, counts="", elapsed="", saved="")


def update_bar(bar, task, *, stage, update, completed=None, total=None, elapsed=None,
               saved_update=None, counts=None):
    width = bar.console.width
    suffix = f" · u{update:,}" if finite(update) else ""
    if width < 90:
        stage = {"Recording strong bots": "Recording bots", "Reading demonstrations": "Reading examples",
                 "Learning from bots": "Learning", "Checking learned moves": "Checking",
                 "Warmup: ready targets": "Warmup: targets", "Warmup: search targets": "Warmup: search"}.get(stage, stage)
    if width < 60:
        suffix = f" u{update:,.0f}".replace(",", "") if finite(update) else ""
        stage = stage.split()[0].rstrip(":")[:max(4, 16 - len(suffix))]
    measured = finite(completed) and finite(total) and total > 0
    count_text = counts if measured and counts is not None else f"{completed:,.0f}/{total:,.0f}" if measured else ""
    if width < 60:
        count_text = count_text.replace(",", "").replace(" to report", "")
    bar.update(task, description=stage + suffix, completed=min(completed, total) if measured else 0,
               total=total if measured else None, measured=measured, counts=count_text,
               elapsed=elapsed_text(elapsed) if finite(elapsed) and width >= 70 else "",
               saved=f"saved u{saved_update:,}" if finite(saved_update) and width >= 90 else "",
               refresh=True)
