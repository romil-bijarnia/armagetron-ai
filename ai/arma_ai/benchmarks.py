"""Bounded real-engine checks using temporary frozen weights, independent of training."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .watch_server import read_record

PROJECT = Path(__file__).resolve().parent.parent


def discard_snapshot(directory: Path | None) -> None:
    """Remove only the temporary weights owned by this benchmark after its worker stops."""
    if directory is None:
        return
    (directory / "checkpoint.pt").unlink(missing_ok=True)
    try:
        directory.rmdir()
    except FileNotFoundError:
        pass


def atomic_record(path: Path, record: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, allow_nan=False) + "\n")
    tmp.replace(path)


def load_results(run: Path, limit: int = 20) -> list[dict]:
    """Only measured fixed-opponent games belong here; never infer them from training scores."""
    results = []
    for name in ("eval.jsonl", "ladder.jsonl"):
        try:
            lines = (run / name).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or not all(k in row for k in ("wins", "draws", "losses")):
                continue
            if not all(isinstance(row[k], int) and row[k] >= 0 for k in ("wins", "draws", "losses")):
                continue
            count = row["wins"] + row["draws"] + row["losses"]
            if row.get("rounds", row.get("games", count)) != count:
                continue
            row = dict(row)
            row.setdefault("opponent", row.get("b", "unknown"))
            row.setdefault("update", row.get("a"))
            row.setdefault("size_factor", row.get("size"))
            row.setdefault("environment", "real_engine")
            row.setdefault("search", False)
            row.setdefault("policy_mode", "unknown")
            row.setdefault("rounds_requested", count)
            row["rounds_completed"] = row["rounds"] = count
            row.setdefault("partial", count < row["rounds_requested"])
            row.setdefault("status", "complete")
            results.append(row)
    return sorted(results, key=lambda row: row.get("time", 0), reverse=True)[:limit]


class BenchmarkManager:
    def __init__(self, run_dir: Path, checkpoint: Path):
        self.run_dir, self.checkpoint = run_dir.resolve(), checkpoint.resolve()
        self.lock = threading.RLock()
        self.proc = None
        self.job = None
        self.completed = None
        self.cancelling = False
        self.snapshot_dir = None

    def start(self, rounds=20, time_budget_s=60, size=-3.0, sample=False) -> dict:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise ValueError("A fixed-opponent check is already running.")
            if self.proc is not None:
                self.poll()
            if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 400:
                raise ValueError("Choose between 1 and 400 games.")
            if (not isinstance(time_budget_s, (float, int)) or not math.isfinite(time_budget_s)
                    or not 5 <= time_budget_s <= 600):
                raise ValueError("The check must have a time budget between 5 and 600 seconds.")
            if not isinstance(size, (float, int)) or not math.isfinite(size) or not -5 <= size <= 2:
                raise ValueError("Choose an arena size between -5 and 2.")
            if not isinstance(sample, bool):
                raise ValueError("The policy sampling setting must be true or false.")
            import torch
            ck = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.job = self.run_dir / "benchmarks" / f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
            self.job.mkdir(parents=True)
            # Ownership crosses a process boundary. Avoid TemporaryDirectory's GC finalizer:
            # losing the manager must not remove weights while its worker is still evaluating.
            self.snapshot_dir = Path(tempfile.mkdtemp(prefix="arma-benchmark-snapshot-"))
            self.proc = None
            try:
                frozen = self.snapshot_dir / "checkpoint.pt"
                torch.save({"model": ck["model"], "update": ck.get("update", 0)}, frozen)
                config = {"checkpoint": str(frozen), "snapshot_dir": str(self.snapshot_dir),
                          "source_checkpoint": str(self.checkpoint),
                          "update": ck.get("update", 0), "run": str(self.run_dir),
                          "rounds": rounds, "time_budget_s": time_budget_s,
                          "size": size, "sample": sample, "started_at": time.time()}
                atomic_record(self.job / "config.json", config)
                self.completed = None
                self.cancelling = False
                env = os.environ.copy()
                env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
                with (self.job / "worker.log").open("ab") as log:
                    self.proc = subprocess.Popen([sys.executable, "-m", "arma_ai.benchmarks", "--worker", str(self.job)],
                                                 cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=log, start_new_session=True)
            except BaseException:
                discard_snapshot(self.snapshot_dir)
                self.snapshot_dir = None
                raise
            return self.poll()

    def poll(self) -> dict:
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            progress = read_record(self.job / "progress.json") if self.job else {}
            config = read_record(self.job / "config.json") if self.job else {}
            if self.job and not running:
                discard_snapshot(self.snapshot_dir)
                self.snapshot_dir = None
                self.completed = read_record(self.job / "result.json") or self.completed
                if self.completed is None:
                    self.completed = {**progress, "status": "cancelled" if self.cancelling else "error",
                                      "partial": True, "update": config.get("update"), "time": time.time()}
                    if self.cancelling:
                        # SIGTERM can arrive before the worker installs its cancellation handler.
                        count = sum(progress.get(k, 0) for k in ("wins", "draws", "losses"))
                        self.completed.update(
                            wins=progress.get("wins", 0), draws=progress.get("draws", 0),
                            losses=progress.get("losses", 0), rounds=count, rounds_completed=count,
                            rounds_requested=config.get("rounds"), opponent="builtin",
                            checkpoint=config.get("source_checkpoint"), source_checkpoint=config.get("source_checkpoint"),
                            sample=config.get("sample"), size_factor=config.get("size"),
                            policy_mode="sampled" if config.get("sample") else "greedy",
                            environment="real_engine", search=False, stopped_reason="cancelled",
                            benchmark_id=self.job.name)
                        atomic_record(self.job / "result.json", self.completed)
                        with (self.run_dir / "eval.jsonl").open("a") as log:
                            log.write(json.dumps(self.completed) + "\n")
                    else:
                        self.completed["error"] = "The check stopped before a final result was saved."
            if running and not progress:
                progress = {"status": "starting", "update": config.get("update"),
                            "rounds_requested": config.get("rounds"), "rounds_completed": 0,
                            "wins": 0, "draws": 0, "losses": 0, "opponent": "builtin",
                            "sample": config.get("sample"), "size_factor": config.get("size"),
                            "environment": "real_engine", "search": False}
            results = load_results(self.run_dir)
            return {"running": running,
                    "status": "cancelling" if running and self.cancelling else
                              progress.get("status", "starting") if running else
                              (self.completed or {}).get("status", "idle"),
                    "progress": progress if running else self.completed,
                    "latest": results[0] if results else None, "results": results}

    def cancel(self) -> dict:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.cancelling = True
                self.proc.terminate()
            return self.poll()

    def close(self) -> None:
        self.cancel()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=5)
        self.poll()  # reap the frozen weights even if the worker needed time to stop


def worker(job: Path) -> None:
    import torch
    from .evaluate import EvaluationCancelled, evaluate
    cfg = read_record(job / "config.json")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def cancel(*_):
        raise EvaluationCancelled()

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    def progress(row):
        atomic_record(job / "progress.json", row)

    try:
        try:
            with tempfile.TemporaryDirectory(prefix="arma-benchmark-") as runtime:
                result = evaluate(Path(cfg["checkpoint"]), None, cfg["rounds"], 1, cfg["size"],
                                  cfg["sample"], Path(runtime), time_budget_s=cfg["time_budget_s"],
                                  progress=progress, device="cpu", base_port=52000 + os.getpid() % 1000)
        except Exception as exc:
            result = {**read_record(job / "progress.json"), "status": "error", "partial": True,
                      "error": str(exc) or type(exc).__name__, "time": time.time(),
                      "update": cfg["update"], "opponent": "builtin"}
        result.update(checkpoint=cfg["source_checkpoint"], source_checkpoint=cfg["source_checkpoint"],
                      benchmark_id=job.name)
        atomic_record(job / "result.json", result)
        # The result is append-only; a partial check never pretends the requested games completed.
        with (Path(cfg["run"]) / "eval.jsonl").open("a") as log:
            log.write(json.dumps(result, allow_nan=False) + "\n")
    finally:
        # The worker owns this cleanup too, so a completed check leaves no frozen weights even
        # if its inspector was closed or its manager is never polled again.
        snapshot = Path(cfg["snapshot_dir"]) if cfg.get("snapshot_dir") else None
        if snapshot is not None and Path(cfg["checkpoint"]) == snapshot / "checkpoint.pt":
            discard_snapshot(snapshot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", type=Path, required=True)
    worker(ap.parse_args().worker)
