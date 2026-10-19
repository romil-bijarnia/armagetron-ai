"""Imitation shares the V4 controller while keeping its progress separate from RL reports."""

import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from arma_ai import cli_progress, dashboard


def heartbeat(run, phase="learning"):
    return {"trainer": "imitate", "run": str(run), "pid": 321, "started_at": time.time() - 60,
            "time": time.time(), "phase": phase, "update": 8000, "imitation_steps": 486,
            "session_updates": 24, "completed": 25, "total": 100, "stage": "Teacher actions",
            "config": {"epochs": 0, "device": "mps"}, "loss": .125}


@pytest.mark.parametrize("command", ["python /tmp/bin/imitate", "python /tmp/bin/arma-imitate",
                                     "python -m arma_ai.imitate"])
@pytest.mark.parametrize("phase,label", [("collecting", "COLLECTING"), ("learning", "IMITATING"),
                                         ("validating", "VALIDATING")])
def test_imitation_process_forms_and_phases_are_live(tmp_path, monkeypatch, command, phase, label):
    monkeypatch.setattr(dashboard, "_processes", lambda: {321: f"{command} --run {tmp_path}"})
    status = dashboard.training_status(tmp_path, heartbeat(tmp_path, phase), [])
    assert status.label == label and status.method == "imitate" and status.running and status.fresh


def test_starting_imitation_does_not_inherit_the_previous_rl_session(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "_processes", lambda: {654: f"python -m arma_ai.imitate --run {tmp_path}"})
    old = {"trainer": "teach", "pid": 123, "time": 1, "phase": "stopped", "update": 7000}
    status = dashboard.training_status(tmp_path, old, [])
    stream = io.StringIO()
    Console(file=stream, width=80).print(cli_progress.build(tmp_path, old, [{"pg": 12345}], status))
    output = stream.getvalue()
    assert status.label == "STARTING" and status.method == "imitate"
    assert "imitation" in output and "12345" not in output and "Update 7,000" not in output


@pytest.mark.parametrize("width", [40, 60, 80, 110])
@pytest.mark.parametrize("height", [12, 16, 30])
@pytest.mark.parametrize("live", [False, True])
def test_imitation_view_preserves_counters_and_omits_stale_rl_metrics(tmp_path, width, height, live):
    progress = heartbeat(tmp_path)
    status = SimpleNamespace(method="imitate", fresh=live, running=live, label="IMITATING" if live else "STOPPED",
                             detail="No live trainer.")
    old_rl = [{"trainer": "teach", "update": 7999, "time": time.time() - 2, "pg": 12345., "vf": 54321.,
               "win_vs_hunter": .99, "sps": 999}]
    stream = io.StringIO()
    Console(file=stream, width=width).print(cli_progress.build(
        tmp_path, progress, old_rl, status, width=width, height=height))
    output = stream.getvalue()
    assert len(output.splitlines()) <= height and all(len(line) <= width for line in output.splitlines())
    for text in ("imitation", "continuous", "MPS", "Update 8,000", "Session +24", "imitation total 486",
                 "Teacher actions", "25/100", "Checkpoint", "heartbeat", "stop saves"):
        assert text in output
    for text in ("12345", "54321", "Replay", "Space fighter", "999", "Value loss"):
        assert text not in output
    assert ("━" in output) == live


def test_large_dashboard_uses_imitation_progress_instead_of_rl_charts(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "_processes", lambda: {321: f"python /tmp/imitate --run {tmp_path}"})
    stream = io.StringIO()
    Console(file=stream, width=160, height=48).print(dashboard.build(
        tmp_path, [{"trainer": "teach", "update": 7000, "pg": 12345}], 160, 48, heartbeat(tmp_path)), height=48)
    output = stream.getvalue()
    assert "IMITATING" in output and "Teacher actions" in output
    assert "Value EV" not in output and "Space fighter" not in output and "12345" not in output


def controller(tmp_path):
    script = tmp_path / "trainctl"
    shutil.copy(Path(__file__).resolve().parents[1] / "trainctl", script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args = tmp_path / "uv-args.json"
    stubs = {"pgrep": "#!/bin/sh\nexit 1\n",
             "uv": f"#!{sys.executable}\nimport json,os,sys\n"
                   "open(os.environ['CONTROL_ARGS'],'w').write(json.dumps(sys.argv[1:]))\n"
                   "print('IMITATION_PROGRESS')\n"}
    for name, source in stubs.items():
        path = bin_dir / name
        path.write_text(source)
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "CONTROL_ARGS": str(args)}
    return script, env, args


def fake_session(tmp_path, name="imitate"):
    run = tmp_path / "run with spaces"
    run.mkdir()
    script = tmp_path / name
    script.write_text("import os,signal,sys,time\nfrom pathlib import Path\n"
                      "run=Path(sys.argv[1]); pidfile=run/'imitation.pid'\n"
                      "def stop(*_):\n (run/'latest.pt').write_text('saved')\n pidfile.unlink(missing_ok=True)\n sys.exit(0)\n"
                      "signal.signal(signal.SIGTERM,stop)\n"
                      "pidfile.write_text(str(os.getpid()))\n"
                      "while True: time.sleep(.01)\n")
    process = subprocess.Popen([sys.executable, str(script), str(run)])
    deadline = time.monotonic() + 3
    while not (run / "imitation.pid").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert (run / "imitation.pid").exists()
    return process, run


@pytest.mark.parametrize("mode", ["full", "background"])
def test_rl_launch_refuses_a_live_imitation_session(tmp_path, mode):
    script, env, args = controller(tmp_path)
    process, run = fake_session(tmp_path)
    try:
        result = subprocess.run([str(script), mode, "--run", str(run), "--no-watch"], env=env,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 1 and "already running" in result.stdout
        assert not args.exists() and process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize("name", ["imitate", "arma-imitate"])
def test_stop_signals_imitation_and_waits_for_its_checkpoint(tmp_path, name):
    script, env, _ = controller(tmp_path)
    process, run = fake_session(tmp_path, name)
    try:
        result = subprocess.run([str(script), "stop", "--run", str(run)], env=env,
                                capture_output=True, text=True, timeout=5)
        process.wait(timeout=5)
        assert result.returncode == 0 and "stopped; checkpoint saved" in result.stdout
        assert (run / "latest.pt").read_text() == "saved"
        assert not (run / "imitation.pid").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stop_ignores_a_reused_imitation_pid(tmp_path):
    script, env, _ = controller(tmp_path)
    process, run = fake_session(tmp_path, "viewer")
    try:
        result = subprocess.run([str(script), "stop", "--run", str(run)], env=env,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0 and "not training" in result.stdout
        assert process.poll() is None and not (run / "latest.pt").exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_status_does_not_print_prior_rl_log_during_imitation(tmp_path):
    script, env, args = controller(tmp_path)
    process, run = fake_session(tmp_path)
    (run / "train.log").write_text("u7000 old_RL_loss=12345\n")
    try:
        result = subprocess.run([str(script), "status", "--run", str(run)], env=env,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0 and "imitation (pid" in result.stdout
        assert "IMITATION_PROGRESS" in result.stdout and "old_RL_loss" not in result.stdout
        assert "--simple" in json.loads(args.read_text()) and "--once" in json.loads(args.read_text())
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_foreground_imitation_controller_preserves_arguments(tmp_path):
    script, env, args = controller(tmp_path)
    run = tmp_path / "run with spaces"
    result = subprocess.run([str(script), "imitate", "--run", str(run), "--epochs", "2"], env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert json.loads(args.read_text()) == ["run", "-q", "arma-imitate", "--run", str(run), "--epochs", "2"]
