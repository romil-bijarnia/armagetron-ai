"""Replay readiness and trainer timing shared by the training views."""

from __future__ import annotations

import math
import os
import subprocess
import time


def replay_requirements(warmup: int, batch: int) -> tuple[int, int, int]:
    """Collected positions, ready targets, and searched targets needed for a batch."""
    return warmup, max(warmup // 2, batch - batch // 2), batch // 2


def _count(value) -> int | None:
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 and int(value) == value):
        return int(value)
    return None


def warmup_state(progress: dict) -> dict:
    """Report every startup gate; collecting enough positions alone is not readiness.

    New heartbeats publish the targets explicitly. Older V4 heartbeats can derive
    them from the same warmup and batch settings, without inventing missing counts.
    """
    cfg = progress.get("config", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    warmup = _count(progress.get("warmup_target", cfg.get("warmup")))
    batch = _count(progress.get("batch", cfg.get("batch")))
    ready = _count(progress.get("warmup_ready_target"))
    searched = _count(progress.get("warmup_searched_target"))
    if warmup is not None and batch is not None and batch > 0:
        _, fallback_ready, fallback_searched = replay_requirements(warmup, batch)
        ready = fallback_ready if ready is None else ready
        searched = fallback_searched if searched is None else searched
    specs = (
        ("collected", "Positions", "warmup_samples", warmup),
        ("ready", "Ready targets", "ready_samples", ready),
        ("searched", "Search targets", "searched_samples", searched),
    )
    gates = []
    for key, label, field, target in specs:
        current = _count(progress.get(field))
        fraction = None if current is None or target is None else min(current / target, 1.0) if target else 1.0
        gates.append({"id": key, "label": label, "current": current, "target": target,
                      "fraction": fraction, "complete": fraction == 1.0})
    fraction = min(g["fraction"] for g in gates) if all(g["fraction"] is not None for g in gates) else None
    return {"fraction": fraction, "complete": all(g["complete"] for g in gates), "gates": gates,
            "waiting": [g["label"] for g in gates if not g["complete"]]}


def process_started(pid: int | None) -> float | None:
    """Use trainer start time, rather than checkpoint age or time the view opened."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True,
                                text=True, timeout=2, env={**os.environ, "LC_ALL": "C"})
        return time.mktime(time.strptime(result.stdout.strip(), "%a %b %d %H:%M:%S %Y"))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
