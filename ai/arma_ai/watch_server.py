"""Read the actor's observation feed without controlling or slowing training."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text())
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


def process_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


class TrainingFeed:
    """Only the selected arena request is written; pause and speed never reach the actor."""

    def __init__(self, run: Path):
        self.run = run
        self.sent = None
        self.last_status = 0.0
        self.status_key = None

    def request_arena(self, arena: int) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        path = self.run / "watch_request.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"arena": arena}) + "\n")
        tmp.replace(path)

    def poll(self, hub, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        frame = read_record(self.run / "watch.json")
        progress = read_record(self.run / "progress.json")
        count = progress.get("config", {}).get("games") or frame.get("n_arenas")
        if isinstance(count, int) and count > 0:
            hub.n_arenas = count
            if hub.arena >= count:
                hub.control({"arena": 0})
        heartbeat = progress.get("time", 0)
        phase = progress.get("phase", "unknown")
        running = (now - heartbeat < 30 and phase not in ("stopped", "stopping", "failed")
                   and process_alive(progress.get("pid")))
        stamp = frame.get("timestamp", 0)
        fresh = (bool(frame) and 0 <= now - stamp < 5 and frame.get("status") != "stopped"
                 and process_alive(frame.get("actor_pid"))
                 and frame.get("trainer_pid", progress.get("pid")) == progress.get("pid"))
        selected = frame.get("arena") == hub.arena
        available = running and fresh and selected
        if not running:
            message = "Training is stopped or its heartbeat is stale. Start a training mode to watch live."
        elif not frame:
            message = "Training is running; waiting for its live arena feed."
        elif not selected:
            message = f"Waiting for training arena {hub.arena + 1}."
        elif not fresh:
            message = "Waiting for a fresh sample from the training actor."
        else:
            message = "Live arena from the running trainer."
            if phase == "warming_up":
                message = "Live training arena. Replay is warming up before learning resumes."
        status = {"available": available, "stale": bool(frame) and not available,
                  "message": message, "timestamp": stamp or None, "received_at": now,
                  "phase": phase, "update": progress.get("update"),
                  "arena": hub.arena, "n_arenas": count, "running": running,
                  "steps": progress.get("steps"), "games": progress.get("games"),
                  "ready_samples": progress.get("ready_samples"),
                  "warmup_samples": progress.get("warmup_samples"),
                  "warmup_target": progress.get("warmup_target"),
                  "mode": progress.get("mode"), "config": progress.get("config", {}),
                  "hyperparameters": progress.get("hyperparameters", {}),
                  "resources": progress.get("resources", {}),
                  "sps": progress.get("sps"), "evals_s": progress.get("evals_s"),
                  "samples": progress.get("samples"), "trained": progress.get("trained")}
        hub.training_status = status
        if not available:
            with hub.lock:
                hub.sticky.pop("training_frame", None)
        status_key = (available, status["stale"], phase, hub.arena, count)
        if now - self.last_status >= 1 or status_key != self.status_key:
            hub.publish("training_status", json.dumps(status))
            self.last_status = now
            self.status_key = status_key
        key = (hub.source_revision, stamp, frame.get("actor_pid"), frame.get("arena"))
        if (available and hub.source == "training" and (not hub.paused or hub.steps > 0)
                and key != self.sent):
            hub.publish("training_frame", json.dumps(frame, separators=(",", ":")))
            hub.frame_shown()
            self.sent = key
        return status
