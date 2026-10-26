"""Native bot RL learns its own decisions and preserves the shared checkpoint."""

import hashlib
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rich.console import Console

from arma_ai import protocol as P
from arma_ai import cli_progress, dashboard
from arma_ai.bot_profiles import HELD_OUT_PROFILES, make_roster
from arma_ai.model import make_net
from arma_ai.native_train import (Config, Episode, Experience, NativeTrainer, OBJECTIVE,
                                 OPPONENTS, outcome, parse_config, returns)


def test_returns_use_complete_outcomes_and_stay_in_value_head_range():
    np.testing.assert_allclose(returns("loss", 3, .9), [-.81, -.9, -1])
    np.testing.assert_allclose(returns("win", 3, .9), [.81, .9, 1])
    np.testing.assert_array_equal(returns("draw", 3, .9), [0, 0, 0])
    assert returns("win", 0, .9).shape == (0,)
    with pytest.raises(ValueError):
        returns("unfinished", 3, .9)


def test_native_terminal_outcomes_do_not_reward_suicidal_mutual_deaths():
    def step(flags, over=False):
        return SimpleNamespace(slots=[SimpleNamespace(flags=flags, alive=bool(flags & P.FLAG_ALIVE))],
                               round_over=over)
    assert outcome(step(P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION)) is None
    assert outcome(step(P.FLAG_ALIVE | P.FLAG_WON, True)) == "win"
    assert outcome(step(P.FLAG_DIED)) == "loss"
    assert outcome(step(P.FLAG_ALIVE, True)) == "draw"


def test_rosters_are_native_bot_only_and_cover_the_configured_training_pool():
    trainer = object.__new__(NativeTrainer)
    trainer.cfg = Config(engines=12)
    trainer.rng = np.random.default_rng(17)
    trainer.profile_cursor = 0
    configs, episodes = trainer.arenas()
    assert [ep.opponents[0] for ep in episodes] == list(OPPONENTS) * 2
    for arena, episode in zip(configs, episodes):
        assert arena.slots == 1 and arena.builtin_ais == len(episode.opponents)
        assert arena.builtin_ais in (1, 3)
        assert not set(episode.opponents) & set(HELD_OUT_PROFILES)
        roster = make_roster("fighter_rush", episode.opponents)
        assert arena.ai_characters == roster.text
        assert arena.builtin_profiles == roster.builtin_names
        assert "NEURAL_TEACHER" not in arena.settings()
        for profile in roster.manifest["opponents"]:
            assert profile["values"][2] == 20
            assert all(profile["values"][i] == 10 for i in (0, 1, 4, 5, 7, 9))


def test_background_help_presets_are_bounded_and_full_is_explicit(tmp_path):
    cfg = parse_config(["--run", str(tmp_path)])
    assert cfg.mode == "background" and cfg.engines == 2 and cfg.duty_cycle == .35
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    full = parse_config(["--mode", "full", "--engines", "3"])
    assert full.engines == 3 and full.duty_cycle == 1


def test_bot_launcher_survives_the_start_command_and_uses_its_own_session(tmp_path):
    controller = tmp_path / "trainctl"
    shutil.copy(Path(__file__).resolve().parents[1] / "trainctl", controller)
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "pgrep").write_text("#!/bin/sh\nexit 1\n")
    (binary / "uv").write_text("#!/bin/sh\nsleep 10\n")
    for path in binary.iterdir():
        path.chmod(0o755)
    run = tmp_path / "run with spaces"
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"]}
    subprocess.run([str(controller), "bots", "--run", str(run), "--no-watch"],
                   env=env, capture_output=True, timeout=10, check=True)
    pid = int((run / "trainctl.pid").read_text())
    try:
        os.kill(pid, 0)
        assert os.getpgid(pid) == pid
        assert os.getpgid(pid) != os.getpgrp()
    finally:
        os.killpg(pid, signal.SIGTERM)


@pytest.fixture
def run(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    net = make_net()
    checkpoint = {"trainer": "teach", "model": net.state_dict(), "update": 42,
                  "last_training_method": "rl", "trained": [100, 7], "generated": [200, 8],
                  "decisions": 200, "games": 20, "exploiter": net.state_dict(),
                  "xupdates": 3, "generation": 1, "exploiter_opt": {}}
    torch.save(checkpoint, tmp_path / "latest.pt")
    return tmp_path


def synthetic_episode(trainer):
    """Numerical replay fixture; predictive strength is evaluated separately in real games."""
    trainer.net.eval()
    state = trainer.net.initial_state(1)
    maps = np.full((2 * P.N_MAPS, P.GRID, P.GRID), 17, np.uint8)
    scalars = np.full(P.N_SCALARS, .25, np.float32)
    mask = np.ones(P.N_ACTIONS, bool)
    context = np.zeros(7, np.float32)
    episode = Episode((OPPONENTS[0],), -3, result="loss")
    for _ in range(4):
        with torch.no_grad():
            a, logp, value, next_state = trainer.net.act_step(
                torch.from_numpy(maps)[None],
                torch.from_numpy(np.concatenate([scalars, mask]))[None], state,
                context=torch.from_numpy(context)[None])
        episode.steps.append(Experience(maps.copy(), scalars.copy(), mask.copy(),
                                       state[0].numpy().copy(), context.copy(),
                                       int(a[0]), float(logp[0]), float(value[0])))
        state = next_state
    return episode


def test_own_policy_replay_updates_weights_and_resumes_optimizer(run):
    trainer = NativeTrainer(Config(run=str(run), device="cpu", sequence_burn_in=0))
    episode = synthetic_episode(trainer)
    original = trainer.net.v.weight.detach().clone()
    metrics = trainer.step(episode, 0, 4, returns("loss", 4, trainer.cfg.gamma))
    assert abs(metrics["policy_kl"]) < 1e-5
    assert not torch.equal(original, trainer.net.v.weight)
    trainer.close()
    saved = torch.load(run / "latest.pt", weights_only=True)
    assert saved["update"] == 43 and saved["last_training_method"] == "native_rl"
    assert saved["rl_objective"] == OBJECTIVE and saved["trained"] == [104, 7]
    assert saved["xupdates"] == 3 and saved["generation"] == 1
    assert (run / "snapshots/before-native-bots-u00000042.pt").exists()
    resumed = NativeTrainer(Config(run=str(run), device="cpu"))
    assert resumed.update == 43 and resumed.optimizer.state
    for key, value in saved["model"].items():
        torch.testing.assert_close(resumed.net.state_dict()[key], value)
    resumed.close()
    saved["last_training_method"] = "imitation"
    torch.save(saved, run / "latest.pt")
    switched = NativeTrainer(Config(run=str(run), device="cpu"))
    assert not switched.optimizer.state
    for key, value in saved["model"].items():
        torch.testing.assert_close(switched.net.state_dict()[key], value)
    switched.close()


def test_failed_training_preserves_the_last_valid_checkpoint(run):
    before = hashlib.sha256((run / "latest.pt").read_bytes()).hexdigest()
    trainer = NativeTrainer(Config(run=str(run), device="cpu"))
    with pytest.raises(RuntimeError, match="already being trained"):
        NativeTrainer(Config(run=str(run), device="cpu"))
    trainer.step(synthetic_episode(trainer), 0, 4, returns("win", 4, .997))
    trainer.close(failed=True)
    assert hashlib.sha256((run / "latest.pt").read_bytes()).hexdigest() == before


def test_stop_requested_during_startup_is_honoured_before_collection(run):
    trainer = NativeTrainer(Config(run=str(run), device="cpu"))
    (run / "trainctl.stop").touch()
    assert trainer.stopping()
    trainer.train()
    assert trainer.update == 42 and trainer.session_games == 0
    trainer.close()


def test_native_heartbeat_and_console_do_not_claim_self_play(tmp_path, monkeypatch):
    import time
    progress = {"trainer": "native_rl", "run": str(tmp_path), "pid": 321,
                "phase": "collecting", "time": time.time(), "stage": "Playing native bots",
                "update": 43, "saved_update": 42, "session_updates": 1,
                "session_games": 4, "native_decisions": 300,
                "wins": 1, "draws": 0, "losses": 3, "completed": 1, "total": 2,
                "config": {"mode": "background", "engines": 2, "device": "mps"}}
    monkeypatch.setattr(dashboard, "_processes", lambda: {
        321: f"python -m arma_ai.native_train --run {tmp_path}"})
    status = dashboard.training_status(tmp_path, progress, [])
    assert status.running and status.fresh and status.method == "native_rl"
    activity = cli_progress.inline_activity(progress, status)
    assert activity["stage"] == "Playing native bots" and activity["completed"] == 1
    stream = io.StringIO()
    Console(file=stream, width=80).print(cli_progress.build(tmp_path, progress, [], status, width=80))
    output = stream.getvalue()
    assert "native bot RL" in output and "1W / 0D / 3L" in output
    assert "self-play" not in output and "Replay" not in output
