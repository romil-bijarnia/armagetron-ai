"""Mode resolution, launch quoting and genuine actor/learner work/rest pacing."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import torch

from arma_ai.teach import DutyPacer, FULL_SIM_THREADS, Learner, parse_config


def test_profiles_have_distinct_workloads_and_no_replay_growth():
    background = parse_config(["--mode", "background"])
    full = parse_config(["--mode", "full"])
    assert (background.games, background.threads, background.batch) == (96, 4, 256)
    assert (full.games, full.threads, full.batch) == (256, FULL_SIM_THREADS, 512)
    assert background.duty_cycle == .35 and full.duty_cycle == 1
    assert background.torch_threads == full.torch_threads == 2
    assert (background.ring, background.exploiter_ring) == (full.ring, full.exploiter_ring) == (200_000, 60_000)
    assert background.resume and full.resume


@pytest.mark.parametrize("mode_position", ["first", "last"])
def test_explicit_flags_override_every_preset_in_any_position(mode_position):
    overrides = ["--games", "12", "--threads", "3", "--batch", "16", "--torch-threads", "1",
                 "--duty-cycle", ".7", "--lr", ".0003", "--run", "runs/with spaces", "--resume", "false"]
    mode = ["--mode", "background"]
    cfg = parse_config(mode + overrides if mode_position == "first" else overrides + mode)
    assert (cfg.games, cfg.threads, cfg.batch, cfg.torch_threads, cfg.duty_cycle) == (12, 3, 16, 1, .7)
    assert cfg.lr == .0003 and cfg.run == "runs/with spaces" and cfg.resume is False


def test_last_mode_selects_defaults_without_overwriting_explicit_values():
    cfg = parse_config(["--mode", "background", "--games", "7", "--mode=full"])
    assert cfg.mode == "full" and cfg.games == 7
    assert cfg.duty_cycle == 1 and cfg.batch == 512


@pytest.mark.parametrize("args", [["--duty-cycle", "0"], ["--duty-cycle", "1.1"],
                                  ["--duty-cycle", "nan"], ["--torch-threads", "0"],
                                  ["--threads", "0"], ["--mode", "other"]])
def test_invalid_resource_settings_fail_during_argument_parsing(args):
    with pytest.raises(SystemExit):
        parse_config(args)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []
        self.stopped = False

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        if not self.stopped:
            self.now += seconds
        return self.stopped


def test_pacer_accounts_for_finished_work_and_never_banks_idle_credit():
    clock = Clock()
    pacer = DutyPacer(.35, clock)
    clock.now = .35
    pacer.pause(0, clock)
    assert clock.waits == pytest.approx([.65])
    assert clock.now == pytest.approx(1)
    assert pacer.rest_seconds == pytest.approx(.65)
    # Natural idle time does not buy an unbounded later burst.
    clock.now = 100
    started = clock()
    clock.now += .7
    pacer.pause(started, clock)
    assert clock.waits[-1] == pytest.approx(1.3)
    assert pacer.rest_seconds == pytest.approx(1.95)


def test_full_mode_never_waits_and_background_stop_is_immediate():
    clock = Clock()
    clock.now = 10
    full = DutyPacer(1, clock)
    full.pause(0, clock)
    assert not clock.waits and full.rest_seconds == 0
    clock.stopped = True
    background = DutyPacer(.35, clock)
    background.pause(0, clock)
    assert clock.now == 10 and background.rest_seconds == 0


@pytest.mark.parametrize("value", [0, -1, 1.01, float("nan"), float("inf")])
def test_programmatic_pacer_rejects_invalid_targets(value):
    with pytest.raises(ValueError, match="duty_cycle"):
        DutyPacer(value)


@pytest.mark.parametrize("command,mode,wrapper", [("background", "background", "nice"),
                                                 ("gentle", "background", "nice"),
                                                 ("full", "full", "caffeinate"),
                                                 ("start", "full", "caffeinate")])
def test_trainctl_aliases_preserve_arguments_and_resume_defaults(tmp_path, command, mode, wrapper):
    args, wrappers, output = _launch_stub(tmp_path, command)
    assert args[:2] == ["run", "arma-teach"]
    assert wrappers == [wrapper]
    assert "--no-watch" not in args
    cfg = parse_config(args[2:])
    assert cfg.mode == mode and cfg.games == 7 and cfg.lr == .0003 and cfg.resume
    assert cfg.run == str(tmp_path / "run with spaces")
    assert f"training started ({mode})" in output


def test_trainctl_explicit_mode_controls_priority_and_equals_run_path(tmp_path):
    args, wrappers, output = _launch_stub(tmp_path, "background", override=True)
    assert wrappers == ["caffeinate"]
    cfg = parse_config(args[2:])
    assert cfg.mode == "full" and cfg.games == 7 and cfg.duty_cycle == 1
    assert cfg.run == str(tmp_path / "run with spaces")
    assert "training started (full)" in output


def test_stopping_training_does_not_kill_the_independent_live_viewer(tmp_path):
    script = tmp_path / "trainctl"
    shutil.copy(Path(__file__).resolve().parents[1] / "trainctl", script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "kills.txt"
    for name, text in {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "pkill": '#!/bin/sh\nprintf "%s\\n" "$*" >> "$MODE_KILLS"\nexit 1\n',
    }.items():
        p = bin_dir / name
        p.write_text(text)
        p.chmod(0o755)
    result = subprocess.run([str(script), "stop", "--run", str(tmp_path / "run")],
                            env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                                 "MODE_KILLS": str(record)}, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and "not training" in result.stdout
    assert not record.exists()


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU is unavailable"))])
def test_background_training_paces_both_workers_and_can_resume_full(tmp_path, device):
    """Two real updates in a disposable run verify that pacing is wired to both processes."""
    old_threads = torch.get_num_threads()
    flags = ["--run", str(tmp_path / "run"), "--device", device, "--games", "2", "--threads", "2",
             "--torch-threads", "2", "--sims", "2", "--search-prob", "1", "--batch", "8",
             "--warmup", "16", "--ring", "8192", "--exploiter-ring", "8192", "--value-steps", "4",
             "--hunter-prob", "0", "--exploiter-prob", "0", "--log-every", "1"]
    try:
        background = Learner(parse_config(["--mode", "background", "--max-updates", "2", *flags]))
        initial = json.loads((background.run_dir / "progress.json").read_text())
        assert initial["mode"] == "background" and initial["config"]["games"] == 2
        assert initial["net"] == 4 and initial["actions"] == 6
        assert initial["hyperparameters"]["batch"] == 8
        background.run()
        progress = json.loads((background.run_dir / "progress.json").read_text())
        assert progress["phase"] == "stopped" and progress["update"] == 2
        assert progress["resources"]["duty_cycle"] == .35
        assert progress["resources"]["learner_throttle_seconds"] > 0
        assert progress["resources"]["actor_throttle_seconds"] > 0
        assert background.trained[0] == 16
        assert all(r.shm.buf is None for r in background.rings)

        full = Learner(parse_config(["--mode", "full", "--max-updates", "3", "--lr", ".0003",
                                     "--weight-decay", ".0002", *flags]))
        assert full.update == 2 and full.trained[0] == 16
        assert full.hyperparameters()["lr"] == .0003
        assert full.hyperparameters()["weight_decay"] == .0002
        assert all(group["weight_decay"] == .0002 for group in full.xopt.param_groups)
        full.run()
        stopped = json.loads((full.run_dir / "progress.json").read_text())
        assert stopped["phase"] == "stopped" and stopped["update"] == 3
        assert stopped["mode"] == "full" and stopped["resources"]["duty_cycle"] == 1
        assert stopped["resources"]["learner_throttle_seconds"] == 0
        assert stopped["resources"]["actor_throttle_seconds"] == 0
        assert stopped["hyperparameters"]["lr"] == .0003
        assert stopped["hyperparameters"]["optimizer"] == "AdamW"
        assert full.trained[0] == 24 and all(r.shm.buf is None for r in full.rings)
    finally:
        torch.set_num_threads(old_threads)


def _launch_stub(tmp_path, command, override=False):
    """Exercise the real shell supervisor with a fake trainer; never allocate a GPU or run."""
    source = Path(__file__).resolve().parents[1] / "trainctl"
    script = tmp_path / "trainctl"
    shutil.copy(source, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "arguments.json"
    wrappers = tmp_path / "wrappers.txt"
    stubs = {
        "pgrep": "#!/bin/sh\nexit 1\n",
        "pkill": "#!/bin/sh\nexit 1\n",
        "nice": '#!/bin/sh\nprintf "nice\\n" >> "$MODE_WRAPPERS"\nshift 2\nexec "$@"\n',
        "caffeinate": '#!/bin/sh\nprintf "caffeinate\\n" >> "$MODE_WRAPPERS"\nshift\nexec "$@"\n',
        "uv": (f"#!{os.sys.executable}\nimport json, os, sys\n"
               "with open(os.environ['MODE_RECORD'], 'w') as f: json.dump(sys.argv[1:], f)\n"),
    }
    for name, text in stubs.items():
        f = bin_dir / name
        f.write_text(text)
        f.chmod(0o755)
    run = tmp_path / "run with spaces"
    flags = [f"--run={run}"] if override else ["--run", str(run)]
    flags += ["--games", "7", "--lr", ".0003", "--no-watch"]
    if override:
        flags += ["--mode=full"]
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "MODE_RECORD": str(record), "MODE_WRAPPERS": str(wrappers)}
    result = subprocess.run([str(script), command, *flags], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        # The supervisor removes its pid file only after the fake trainer has exited.
        if record.exists() and not (run / "trainctl.pid").exists():
            break
        time.sleep(.01)
    assert record.exists(), (run / "train.log").read_text()
    assert not (run / "trainctl.pid").exists()
    return json.loads(record.read_text()), wrappers.read_text().splitlines(), result.stdout
