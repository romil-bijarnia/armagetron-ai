"""Compact inline training console, driven by the trainer's actual telemetry."""

from __future__ import annotations

import math
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .dashboard import read_metrics, read_progress, training_status
from .progress_state import process_started, warmup_state

ACCENT = "#8aa8c7"
DIM = "#8b949e"
WARNING = "#d3b775"
TRACK = "#414b57"
REPLAY_READ_MARGIN = 4096


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"


def age(timestamp: float, now: float) -> str:
    seconds = max(0, int(now - timestamp))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02}s ago"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02}m ago"


def number(value, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "not reported"


def nonnegative(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def progress_rows(label: str, completed: float, total: float, width: int, *, detail: str | None = None,
                  due: bool = False) -> list[Text]:
    """Keep the measured counts visible when a narrow terminal leaves less room for a bar."""
    fraction = min(max(completed / total, 0), 1)
    percent = 100 if fraction == 1 else min(round(fraction * 100), 99)
    detail = detail or f"{completed:,.0f}/{total:,.0f}"
    suffix = f" {percent:3}%  {detail}" + (" due" if due else "")
    prefix = f"{label:<13}  " if width >= 60 else f"{label}  "
    available = width - len(prefix) - len(suffix)
    rows = []
    if available < 4:
        rows.append(Text(label, style=ACCENT, no_wrap=True))
        prefix = ""
        available = width - len(suffix)
        if available < 2:
            suffix = f" {percent}% {detail.replace(',', '')}" + (" due" if due else "")
            available = width - len(suffix)
    bar_width = min(26 if width >= 76 else 16, max(1, available))
    filled = round(fraction * bar_width)
    color = WARNING if due else ACCENT
    rows.append(Text.assemble((prefix, ACCENT), ("━" * filled, color),
                              ("━" * (bar_width - filled), TRACK), (suffix, DIM)))
    for row in rows:
        row.no_wrap, row.overflow = True, "ellipsis"
    return rows


def eligible_records(records: list[dict], progress: dict) -> list[dict]:
    """Exclude log entries ahead of the checkpoint a restarted trainer loaded."""
    if "update" not in progress:
        return records
    return [r for r in records if r.get("update", 0) <= progress["update"]
            and r.get("time", 0) <= progress.get("time", math.inf) + 1]


def update_rate(progress: dict, records: list[dict], started: float | None) -> float | None:
    now = progress.get("time", 0)
    eligible = [r for r in records if 10 <= now - r.get("time", 0) <= 180
                and r.get("time", 0) >= (started or 0)
                and r.get("update", 0) <= progress.get("update", 0)]
    if not eligible:
        return None
    first = eligible[0]
    return 60 * (progress["update"] - first["update"]) / (now - first["time"])


def report_lines(progress: dict, records: list[dict], status, *, width: int, now: float,
                 started: float | None = None) -> list[Text]:
    """The latest saved report under the activity line: what training is doing to the policy.

    The match scores carry their counts; losses describe the learning objectives."""
    def line(value: str, style: str = DIM) -> Text:
        return Text(value, style=style, no_wrap=True, overflow="ellipsis")

    history = eligible_records(records, progress)
    if not history:
        return [line("No saved report yet")]
    last = history[-1]
    live = status.running and status.fresh
    narrow = width < 90
    sep = " · "
    head = [f"Report u{last.get('update', 0):,}", age(last.get("time", now), now)]
    if started and last.get("time", 0) < started:
        head.append("previous session")
    numbers = [("clip", last.get("clip_fraction"), 3), ("gain", last.get("search_kl"), 3),
               ("policy", last.get("pg"), 3), ("value", last.get("vf"), 3)]
    if not narrow:
        numbers += [("entropy", last.get("ent"), 2), ("value EV", last.get("value_explained"), 2)]
    shown = [f"{label} {value:.{decimals}f}" for label, value, decimals in numbers
             if isinstance(value, (int, float)) and math.isfinite(value)]
    lines = [line(sep.join(head + shown), "default")]
    matches = []
    for key, label, short in (("win_vs_hunter", "Space fighter", "Hunter"), ("win_vs_past", "Past selves", "Past"),
                              ("win_exploiter", "Exploiter", "Exploiter"),
                              ("win_vs_exploiter", "Main vs exploiter", "VsExpl")):
        counts = last.get("match_results", {}).get(key)
        score = last.get(key)
        if isinstance(counts, dict) and counts.get("games"):
            score = (counts["wins"] + .5 * counts["draws"]) / counts["games"]
            detail = f" ({counts['wins']}W {counts['draws']}D {counts['losses']}L)"
        elif isinstance(score, (int, float)) and math.isfinite(score):
            detail = ""
        else:
            continue
        matches.append(f"{short if narrow else label} {score:.0%}{detail}")
    if matches:
        lines.append(line(sep.join(matches)))
    if live:
        rates = []
        if nonnegative(progress.get("sps")):
            rates.append(f"{progress['sps']:,.0f} decisions/s")
        per_minute = update_rate(progress, history, started)
        if per_minute is not None:
            rates.append(f"{per_minute:.1f} updates/min")
        every, update = progress.get("config", {}).get("snapshot_every"), progress.get("update")
        if isinstance(every, int) and every > 0 and nonnegative(update):
            rates.append(f"next snapshot u{(update // every + 1) * every:,}")
        if rates:
            lines.append(line(sep.join(rates)))
    return lines


def build_imitation(run_dir: Path, progress: dict, status, *, width: int, height: int,
                    now: float, started: float | None = None) -> Group:
    """Imitation has phase work and cumulative updates, not an RL replay buffer or value loss."""
    width = max(20, min(width, 116))
    live = status.running and status.fresh
    cfg = progress.get("config", {})

    def line(value: str, style: str = DIM) -> Text:
        return Text(value, style=style, no_wrap=True, overflow="ellipsis")

    color = ACCENT if live else WARNING if status.label in ("STALE", "FAILED") else DIM
    header = Table.grid(expand=True)
    header.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    header.add_column(justify="right", no_wrap=True)
    header.add_row(line(f"armagetron / {run_dir.name}", "bold"), line(status.label, color))
    body = [header]
    identity = ["imitation"]
    epochs = cfg.get("epochs")
    if nonnegative(epochs):
        identity.append("continuous" if epochs == 0 else f"{epochs:g} cycles")
    if cfg.get("device"):
        identity.append(str(cfg["device"]).upper())
    body.append(line(" · ".join(identity)))
    update = progress.get("update")
    text = f"Update {update:,}" if nonnegative(update) else "Update not reported"
    began = progress.get("started_at", started)
    if live and nonnegative(began):
        text += f" · elapsed {duration(now - began)}"
    body.append(line(text, "bold"))
    counters = []
    if nonnegative(progress.get("session_updates")):
        counters.append(f"Session +{progress['session_updates']:,}")
    if nonnegative(progress.get("imitation_steps")):
        counters.append(f"imitation total {progress['imitation_steps']:,}")
    if counters:
        joined = " · ".join(counters)
        body.extend([line(joined)] if len(joined) <= width else [line(value) for value in counters])
    stage = str(progress.get("stage") or progress.get("phase", "Waiting for first heartbeat").capitalize())
    completed, total = progress.get("completed"), progress.get("total")
    if nonnegative(completed) and nonnegative(total) and total > 0:
        if live:
            body.extend(progress_rows(stage, completed, total, width))
        else:
            body.extend([line(f"Recorded stage: {stage}"), line(f"Work {completed:,.0f}/{total:,.0f}")])
    else:
        body.append(line(stage))
    if nonnegative(progress.get("loss")):
        body.append(line(f"Imitation loss {number(progress['loss'])} · {age(progress.get('time', now), now)}"))
    try:
        saved = age((run_dir / "latest.pt").stat().st_mtime, now).removesuffix(" ago").replace(" ", "")
    except OSError:
        saved = "not saved"
    heartbeat = (age(progress["time"], now).removesuffix(" ago").replace(" ", "")
                 if nonnegative(progress.get("time")) else "waiting")
    if status.label in ("STALE", "FAILED"):
        body.append(line(status.detail, WARNING))
    footer = [line(f"Checkpoint {saved} · heartbeat {heartbeat}"),
              line("Ctrl-C closes view; stop saves training." if width < 76 else
                   "Ctrl-C closes this view. stop saves and stops training.")]
    body = body[:max(6, height) - len(footer)] + footer
    bounded = Table.grid()
    bounded.add_column(width=width, no_wrap=True)
    for item in body:
        bounded.add_row(item)
    return Group(bounded)


def build(run_dir: Path, progress: dict, records: list[dict], status, *, width: int = 110,
          height: int = 30, now: float | None = None, started: float | None = None,
          changed_at: float = 0) -> Group:
    """Stable, bounded rows; logged results always carry their own update and age."""
    now = time.time() if now is None else now
    if (getattr(status, "method", None) or progress.get("trainer")) == "imitate":
        current = progress if progress.get("trainer") == "imitate" else {}
        return build_imitation(run_dir, current, status, width=width, height=height, now=now, started=started)
    width = max(20, min(width, 116))
    narrow = width < 76
    history = eligible_records(records, progress)
    last = history[-1] if history else {}
    current = progress or last
    cfg = current.get("config", {})
    update = current.get("update")
    live = status.running and status.fresh
    warming = live and current.get("phase") in ("starting", "warming_up")
    learning = live and current.get("phase") == "learning"
    try:
        saved_at = (run_dir / "latest.pt").stat().st_mtime
    except OSError:
        saved_at = None
    status_style = ACCENT if status.fresh else DIM
    if status.label in ("STARTING", "STALE"):
        status_style = WARNING
    elif status.label == "FAILED":
        status_style = "#cf8686"

    def line(value: str, style: str = DIM) -> Text:
        return Text(value, style=style, no_wrap=True, overflow="ellipsis")

    header = Table.grid(expand=True)
    header.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    header.add_column(justify="right", no_wrap=True)
    header.add_row(line(f"armagetron / {run_dir.name}", "bold"), line(status.label, status_style))
    body = [header]
    totals_line = status_line = None
    loss_lines, score_items, compact_scores = [], [], []
    replay_rows, report_rows, save_rows, target_rows = [], [], [], []
    identity = []
    mode = current.get("mode", cfg.get("mode"))
    if mode:
        identity.append(mode)
    if cfg.get("device"):
        identity.append(str(cfg["device"]).upper())
    combat = cfg.get("combat_prob", current.get("hyperparameters", {}).get("combat_prob"))
    if isinstance(combat, (int, float)) and math.isfinite(combat):
        identity.append(f"combat starts {combat:.0%}")
    if identity:
        body.append(line("  ·  ".join(identity)))
    update_text = f"Update {update:,}" if update is not None else "Update not reported"
    if started and live:
        update_text += f"  ·  elapsed {duration(now - started)}"
    body.append(line(update_text, "bold"))
    rate = current.get("sps") if live else None
    actor = f"{rate:,} decisions/s" if rate is not None else "waiting decisions/s"
    learner_rate = update_rate(progress, history, started) if live and not warming else None
    learner = (f"{learner_rate:.1f} updates/min" if learner_rate is not None else
               "warming up" if warming else "rate pending" if live else "no live rate")
    body.append(line(f"{actor}  ·  {learner}"))
    totals = []
    if "trained" in current:
        totals.append(f"{current['trained']:,} trained")
    if "games" in current:
        totals.append(f"{current['games']:,} rounds")
    if totals:
        totals_line = line("  ·  ".join(totals))
        body.append(totals_line)
    if "ready_samples" in current:
        searched = (f"  ·  {current['searched_samples']:,} searched" if
                    current.get("warmup_searched_target", current.get("searched_samples", 0)) > 0 else "")
        body.append(line(f"Replay {current['ready_samples']:,} ready{searched}"))
    else:
        body.append(line("Replay not reported"))

    if warming:
        readiness = warmup_state(current)
        if readiness["fraction"] is not None:
            body.extend(progress_rows("WARMUP", readiness["fraction"], 1, width, detail="readiness"))
            collected, ready, *extra = readiness["gates"]
            body.append(line(f"Positions {collected['current']:,}/{collected['target']:,}"))
            targets = f"Targets {ready['current']:,}/{ready['target']:,}"
            if extra:
                searched = extra[0]
                targets += f"  ·  searched {searched['current']:,}/{searched['target']:,}"
            body.append(line(targets))
            if readiness["complete"]:
                body.append(line("Ready; waiting for the first update."))
        else:
            body.append(line("WARMUP  Waiting for replay target counts.", ACCENT))
    if learning:
        capacity, ready = cfg.get("ring"), current.get("ready_samples")
        if nonnegative(capacity) and capacity > REPLAY_READ_MARGIN and nonnegative(ready):
            replay_rows = progress_rows("Replay ready", ready, capacity - REPLAY_READ_MARGIN, width)
            body.extend(replay_rows)
        every = cfg.get("log_every")
        if isinstance(every, int) and every > 0 and nonnegative(update):
            report_rows = progress_rows("Next report", update % every, every, width)
            body.extend(report_rows)
        interval = cfg.get("save_seconds")
        if nonnegative(interval) and interval > 0:
            if saved_at is not None:
                elapsed = max(0, now - saved_at)
                save_rows = progress_rows("Next save", elapsed, interval, width,
                                          detail=f"{duration(elapsed)}/{duration(interval)}", due=elapsed >= interval)
            else:
                save_rows = [line("Next save: waiting for first checkpoint")]
            body.extend(save_rows)
    if not warming and history:
        stamp = f"u{last.get('update', 0):,}  ·  {age(last.get('time', now), now)}"
        previous = "previous session" if started and last.get("time", 0) < started else "recorded" if not status.fresh else "reported"
        gain = f"  ·  Gain {number(last['search_kl'])}" if nonnegative(last.get("search_kl")) else ""
        loss_lines = [line(f"Loss {stamp}  ·  {previous}"),
                      line(f"Policy {number(last.get('pg'))}  ·  Value {number(last.get('vf'))}{gain}", "default")]
        body.extend(loss_lines)
    elif not warming:
        body.append(line("Loss: awaiting first report"))
    target = current.get("max_updates", cfg.get("max_updates"))
    if nonnegative(target) and target > 0 and nonnegative(update):
        if live:
            target_rows = progress_rows("Update target", update, target, width)
            body.extend(target_rows)
        else:
            body.append(line(f"Update target {update:,}/{target:,}"))

    pairs = (("win_vs_hunter", "Space fighter"), ("win_vs_past", "Past selves"),
             ("win_exploiter", "Exploiter" if narrow else "Exploiter vs frozen target"),
             ("win_vs_exploiter", "Main vs exploiter"))
    reported = [(key, label) for key, label in pairs
                if key in last or key in last.get("match_results", {})]
    if reported:
        matchups = Table.grid(padding=(0, 1))
        matchups.add_column(no_wrap=True, overflow="ellipsis")
        matchups.add_column(justify="right", no_wrap=True)
        matchups.add_column(justify="right", no_wrap=True)
        matchups.add_row(line("Opponent"), line("Score"), line("W/D/L · rounds"))
        for key, label in reported:
            if width < 50 and key == "win_vs_hunter":
                label = "Hunter"
            value = last.get(key)
            counts = last.get("match_results", {}).get(key)
            if counts:
                detail = (f"{counts['wins']}/{counts['draws']}/{counts['losses']} n{counts['games']}" if narrow
                          else f"{counts['wins']}W {counts['draws']}D {counts['losses']}L · n={counts['games']}")
            else:
                detail = "count unavailable"
            score = f"{value:.1%}" if value is not None else "unreported"
            matchups.add_row(line(label, "default"), line(score, "default"),
                             line(detail))
            short = {"win_vs_hunter": "Hunter", "win_vs_past": "Past", "win_exploiter": "Exploiter",
                     "win_vs_exploiter": "VsExpl"}[key]
            compact_count = (f"{counts['wins']}W/{counts['draws']}D/{counts['losses']}L n{counts['games']}"
                             if counts else "count unavailable")
            compact_scores.append(line(f"Train {short} {score}  {compact_count}"))
        caption = ("Training scores; draw = ½." if width < 50 else
                   "Training scores (draw = ½), not fixed benchmarks.")
        score_items = [matchups, line(caption)]
        if warming:
            stamp = line(f"Scores u{last.get('update', 0):,}  ·  {age(last.get('time', now), now)}")
            score_items.append(stamp)
            compact_scores.insert(0, stamp)
        body.extend(score_items)
    else:
        body.append(line("Training matches: no results yet"))
    saved = age(saved_at, now) if saved_at is not None else "not saved"
    heartbeat = age(progress.get("time", now), now) if progress else "waiting"
    saved_short = saved.removesuffix(" ago").replace(" ", "") if saved != "not saved" else saved
    heartbeat_short = heartbeat.removesuffix(" ago").replace(" ", "")
    body.append(line(f"Checkpoint {saved_short} · heartbeat {heartbeat_short}"))
    if status.label in ("STOPPED", "FAILED", "STALE"):
        status_line = line(status.detail, WARNING if status.label != "STOPPED" else DIM)
        body.append(status_line)
    body.append(line("Ctrl-C closes view; trainer unaffected." if narrow else
                     "Ctrl-C closes this view. stop saves and stops training."))
    # Preserve the live status and footer in short terminals. Tables have a known row count;
    # every text row above is deliberately single-line. No terminal crop hides the controls.
    budget = max(6, height)

    def rows() -> int:
        return sum(len(item.rows) if isinstance(item, Table) else 1 for item in body)

    if rows() > budget and totals_line is not None:
        body = [item for item in body if item is not totals_line]
    if rows() > budget and score_items:
        index = next(i for i, item in enumerate(body) if item is score_items[0])
        body = [item for item in body if all(item is not old for old in score_items)]
        body[index:index] = compact_scores
    for optional_rows in (report_rows, save_rows, replay_rows, target_rows):
        if rows() <= budget:
            break
        body = [item for item in body if all(item is not old for old in optional_rows)]
    if rows() > budget and status_line is not None:
        body = [item for item in body if item is not status_line]
    for score_row in reversed(compact_scores):
        if rows() <= budget:
            break
        body = [item for item in body if item is not score_row]
    if rows() > budget and loss_lines:
        body = [item for item in body if item is not loss_lines[1]]
    if rows() > budget:
        body = body[:budget - 2] + body[-2:]
    # A grid bounds every child without a decorative panel or terminal-wide padding.
    bounded = Table.grid()
    bounded.add_column(width=width, no_wrap=True)
    for item in body:
        bounded.add_row(item)
    return Group(bounded)


def inline_activity(progress: dict, status) -> dict:
    """Give the single bar one real unit of work, never overall AI completeness."""
    from .inline_ui import finite
    method = getattr(status, "method", None)
    current = progress if not method or not progress.get("trainer") or progress.get("trainer") == method else {}
    out = {"stage": status.label.capitalize(), "update": current.get("update"),
           "saved_update": current.get("saved_update"), "completed": None, "total": None}
    if not status.running or not status.fresh:
        return out
    if current.get("trainer") == "imitate":
        return {**out, "stage": str(current.get("stage") or "Preparing imitation"),
                "completed": current.get("completed"), "total": current.get("total")}
    phase = current.get("phase")
    if phase in ("starting", "warming_up"):
        readiness = warmup_state(current)
        if readiness["complete"]:
            return {**out, "stage": "Starting learner"}
        if readiness["fraction"] is not None:
            gate = min(readiness["gates"], key=lambda g: g["fraction"])
            return {**out, "stage": "Warmup: " + gate["label"].lower(),
                    "completed": gate["current"], "total": gate["target"]}
        return {**out, "stage": "Preparing replay"}
    if phase == "learning":
        cfg = current.get("config", {})
        update = current.get("update")
        target = current.get("max_updates", cfg.get("max_updates"))
        if finite(update) and finite(target) and target > 0:
            return {**out, "stage": "Learning", "completed": update, "total": target}
        every = cfg.get("log_every")
        if finite(update) and finite(every) and every > 0:
            completed = update % every
            return {**out, "stage": "Learning", "completed": completed, "total": every,
                    "counts": f"{completed:,.0f}/{every:,.0f} to report"}
        return {**out, "stage": "Learning"}
    return {**out, "stage": "Saving Blacklight" if phase == "stopping" else "Preparing training"}


def run(run_dir: Path, console: Console, *, once: bool = False) -> None:
    from .inline_ui import add_task, finite, progress_bar, update_bar

    progress, status, started, pid = {}, None, None, None
    records, metrics_state = [], {}
    bar = progress_bar(console)
    task = add_task(bar)

    def refresh():
        nonlocal progress, status, pid, started, records
        progress = read_progress(run_dir / "progress.json")
        records = read_metrics(run_dir / "metrics.jsonl", metrics_state)
        status = training_status(run_dir, progress, records)
        if progress.get("pid") != pid or (started is None and status.running):
            pid = progress.get("pid")
            started = (progress.get("started_at") or process_started(pid)) if status.running else None

    def frame():
        info = inline_activity(progress, status)
        now = time.time()
        # Imitation reports elsewhere; otherwise the last RL report rides under the activity line.
        imitating = status.running and (getattr(status, "method", None) or progress.get("trainer")) == "imitate"
        report = [] if imitating else report_lines(progress, records, status, width=console.width, now=now,
                                                   started=started)
        if not status.running or not status.fresh:
            label = info["stage"]
            if finite(info["update"]):
                label += f" · Blacklight update {info['update']:,}"
            if finite(info["saved_update"]):
                label += f" · saved u{info['saved_update']:,}"
            return Group(Text(label, style=WARNING if status.label in ("STALE", "FAILED") else DIM,
                              no_wrap=True, overflow="ellipsis"), *report)
        update_bar(bar, task, **info, elapsed=now - started if started else None)
        return Group(bar.get_renderable(), *report)

    refresh()
    cfg = progress.get("config", {})
    method = getattr(status, "method", None) or progress.get("trainer")
    mode = "imitation" if method == "imitate" else progress.get("mode", cfg.get("mode", "training"))
    identity = ["Blacklight", mode]
    if cfg.get("device"):
        identity.append(str(cfg["device"]).upper())
    console.print(" · ".join(identity), highlight=False)
    checkpoint = run_dir / "latest.pt"
    project = Path(__file__).resolve().parent.parent
    try:
        checkpoint = (Path("runs/blacklight/latest.pt") if run_dir.resolve() == (project / "runs/blacklight").resolve()
                      else checkpoint.relative_to(project))
    except ValueError:
        pass
    console.print(f"Checkpoint: {checkpoint}", style=DIM, highlight=False)
    console.print("Ctrl-C closes this view. stop saves and stops training.", style=DIM, highlight=False)
    if once or not console.is_terminal:
        console.print(frame())
        return
    def following() -> bool:
        return status.running or status.label in ("STARTING", "STALE")

    if not following():
        console.print(frame())
        return
    try:
        with Live(frame(), console=console, screen=False, auto_refresh=False, vertical_overflow="crop") as live:
            misses = 0  # the view closes once the trainer has been gone for several refreshes, not one
            while misses < 4:
                time.sleep(.5)
                refresh()
                live.update(frame(), refresh=True)
                misses = 0 if following() else misses + 1
    except KeyboardInterrupt:
        pass
