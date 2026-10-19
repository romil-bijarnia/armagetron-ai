"""The terminal dashboard must reflect actual trainer progress and the metrics it logs."""

import io
import json
import math
import os
import time
from pathlib import Path

import pytest
from rich.console import Console

from arma_ai import dashboard as dash


def render(run, records, progress, width=80, height=24):
    stream = io.StringIO()
    console = Console(file=stream, width=width, height=height, color_system=None)
    console.print(dash.build(run, records, width, height, progress), height=height)
    output = stream.getvalue()
    assert len(output.splitlines()) == height
    assert all(len(line) <= width for line in output.splitlines())
    return output


@pytest.fixture
def live_teacher(tmp_path, monkeypatch):
    run = tmp_path / "v4"
    run.mkdir()
    monkeypatch.setattr(dash, "_processes", lambda: {321: f"python /tmp/bin/arma-teach --run {run}"})
    return run, {"trainer": "teach", "pid": 321, "time": time.time(), "phase": "warming_up",
                 "update": 0, "max_updates": 0, "steps": 6000, "games": 20,
                 "samples": 6000, "trained": 0, "warmup_samples": 6000, "warmup_target": 20000,
                 "ready_samples": 5000, "searched_samples": 2500, "actions": 6, "net": 4, "batch": 512,
                 "started_at": time.time() - 300, "mode": "full",
                 "config": {"device": "mps", "games": 256, "combat_prob": .25, "ring": 200000,
                            "log_every": 25, "save_seconds": 300, "batch": 512}}


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 48)])
def test_warmup_is_visible_before_any_training_metrics(live_teacher, size):
    run, progress = live_teacher
    # The caller need not pass progress: normal frames read the live heartbeat from disk.
    (run / "progress.json").write_text(json.dumps(progress))
    output = render(run, [], None, *size)
    for text in ("WARMING UP", "Warmup readiness", "6,000/20,000", "30%", "5,000/10,000", "2,500/256"):
        assert text in output
    assert "No data yet" in output
    assert "Space fighter" in output
    assert "best bot" not in output


def test_full_sample_buffer_does_not_claim_learning_started(live_teacher):
    run, progress = live_teacher
    progress.update(warmup_samples=20000, ready_samples=2500, searched_samples=100)
    output = render(run, [], progress)
    assert "WARMING UP" in output
    assert "Waiting for ready targets, search targets" in output
    readiness = next(line for line in output.splitlines() if "Warmup readiness" in line)
    assert "25%" in readiness and "100%" not in readiness
    assert "TRAINING" not in output


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 48)])
def test_bounded_updates_have_a_real_progress_bar(live_teacher, size):
    run, progress = live_teacher
    progress.update(phase="learning", update=25, max_updates=100, trained=12800)
    output = render(run, [], progress, *size)
    assert "TRAINING" in output
    assert "Update target" in output and "25/100" in output
    assert "25%" in output
    assert "12,800 trained" in output
    assert "Warmup readiness" not in output


def test_unbounded_training_shows_counts_not_overall_percent(live_teacher):
    run, progress = live_teacher
    progress.update(phase="learning", update=25, max_updates=0, trained=12800)
    output = render(run, [], progress)
    assert "Update 25" in output and "without a fixed update limit" in output
    assert "6,000 decisions" in output
    assert "12,800 trained" in output
    assert "Update target" not in output


@pytest.mark.parametrize("change,processes,label", [
    ({}, {}, "STOPPED"),
    ({}, {321: "sleep 1000"}, "STOPPED"),
    ({"time": 1}, None, "STALE"),
    ({"phase": "stopped"}, {}, "STOPPED"),
    ({"phase": "failed"}, {}, "FAILED"),
    ({"phase": "starting"}, None, "STARTING"),
    ({"phase": "stopping"}, None, "SAVING"),
])
def test_status_needs_a_live_matching_process(live_teacher, monkeypatch, change, processes, label):
    run, progress = live_teacher
    progress.update(change)
    if processes is not None:
        monkeypatch.setattr(dash, "_processes", lambda: processes)
    status = dash.training_status(run, progress, [])
    assert status.label == label
    assert status.running == (label in ("STARTING", "SAVING"))


def test_process_listing_is_cached_and_a_failed_listing_keeps_the_last_snapshot(monkeypatch):
    listings = [{321: "python arma-teach --run runs/blacklight"}, None, {}]
    calls = []

    def list_processes():
        calls.append(1)
        return listings.pop(0)

    monkeypatch.setattr(dash, "_list_processes", list_processes)
    monkeypatch.setattr(dash, "_process_cache", {"time": 0.0, "processes": None})
    first = dash._processes()
    assert first == {321: "python arma-teach --run runs/blacklight"} and len(calls) == 1
    assert dash._processes() == first and len(calls) == 1  # within the cache window: no second ps
    dash._process_cache["time"] = 0.0
    assert dash._processes() == first and len(calls) == 2  # ps failed: the trainer is not declared gone
    dash._process_cache["time"] = 0.0
    assert dash._processes() == {} and len(calls) == 3  # a real empty listing is believed


def test_a_failed_listing_with_no_snapshot_reports_nothing_running(monkeypatch):
    monkeypatch.setattr(dash, "_list_processes", lambda: None)
    monkeypatch.setattr(dash, "_process_cache", {"time": 0.0, "processes": None})
    assert dash._processes() == {}


def test_other_runs_and_reused_supervisor_pid_cannot_look_like_training(tmp_path, monkeypatch):
    run = tmp_path / "idle"
    run.mkdir()
    (run / "trainctl.pid").write_text("123\n")
    monkeypatch.setattr(dash, "_processes", lambda: {
        123: "sleep 1000", 456: f"python /tmp/bin/arma-teach --run {tmp_path / 'other'}",
    })
    assert not dash.is_training(run)
    output = render(run, [], {})
    assert "STOPPED" in output
    assert "No training data yet" in output
    assert "TRAINING" not in output


def test_teacher_without_heartbeat_still_detects_its_own_run(tmp_path, monkeypatch):
    run = tmp_path / "v4"
    monkeypatch.setattr(dash, "_processes", lambda: {123: f"python /tmp/bin/arma-teach --run={run}"})
    assert dash.is_training(run)
    assert not dash.is_training(tmp_path / "other")


def test_supervisor_waiting_for_child_is_starting_not_training(tmp_path, monkeypatch):
    run = tmp_path / "v4"
    monkeypatch.setattr(dash, "_processes", lambda: {123: f"/bin/sh /tmp/trainctl _supervise teach --run {run}"})
    status = dash.training_status(run, {}, [])
    assert status.label == "STARTING"
    assert not status.running


@pytest.mark.parametrize("size", [(100, 30), (160, 48)])
def test_search_gain_is_charted_with_entropy_and_listed(tmp_path, monkeypatch, size):
    monkeypatch.setattr(dash, "_processes", lambda: {})
    recs = [{"trainer": "teach", "update": update, "time": time.time(), "ep_len": 400, "sps": 60, "ent": .3,
             "search_kl": .138 - .01 * i, "pg": .25, "vf": .2, "value_explained": .5}
            for i, update in enumerate((25, 50, 75))]
    output = render(tmp_path / "v4", recs, {}, *size)
    assert "Search gain · 0.118" in output and "KL above · entropy below" in output
    if size[0] == 160:
        assert "Gain" in output and "0.118" in output and "0.138" in output


def test_entropy_is_normalized_over_six_controls(tmp_path):
    assert dash.policy_focus({"ent": math.log(6), "actions": 6}, tmp_path) == pytest.approx(0)
    assert dash.policy_focus({"ent": math.log(6)}, tmp_path) == pytest.approx(0)
    assert dash.policy_focus({"ent": 0}, tmp_path) == pytest.approx(1)
    assert dash.policy_focus({}, tmp_path) is None


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 48)])
def test_teacher_results_preserve_meaning_and_negative_value_explained(tmp_path, monkeypatch, size):
    monkeypatch.setattr(dash, "_processes", lambda: {})
    rec = {"trainer": "teach", "update": 24, "steps": 18640, "time": time.time(), "ep_len": 428,
           "sps": 800, "ent": math.log(6), "actions": 6, "value_explained": -0.42,
           "win_vs_hunter": .2, "win_vs_past": .6, "win_exploiter": .7,
           "len_duel_-3": 300, "len_duel_-1.5": 400, "len_duel_0": 500,
           "len_ffa4_-1": 600, "len_ffa4_0": 700}
    output = render(tmp_path / "v4", [rec], {}, *size)
    for text in ("STOPPED", "Space fighter", "V4 snapshots", "Exploiter", "20.0%", "60.0%", "70.0%",
                 "Value EV", "-0.420", "-1.00", "Mean round duration", "21.4 s", "0 = mean prediction"):
        assert text in output
    if size[0] >= 100:
        assert "focus 0%" in output
    assert "best bot" not in output
    assert "survival" not in output.lower()
    assert "past versions" not in output
    if size[0] == 160:
        for name in dash.ARENAS.values():
            assert name in output
        assert "15.0 s" in output and "35.0 s" in output
        assert "Game time per completed round" in output
        assert "-0.420" in output




def test_missing_or_invalid_heartbeat_is_no_progress(tmp_path):
    path = tmp_path / "progress.json"
    assert dash.read_progress(path) == {}
    for content in ('{"phase":', '[]'):
        path.write_text(content)
        assert dash.read_progress(path) == {}


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 48)])
def test_screenshot_warmup_reports_the_waiting_gate(live_teacher, size):
    run, progress = live_teacher
    progress.update(update=5666, warmup_samples=31656, ready_samples=7193, searched_samples=1614)
    text = render(run, [], progress, *size)
    overall = next(line for line in text.splitlines() if "Warmup readiness" in line)
    assert "72%" in overall and "100%" not in overall
    for label, count in (("Positions", "31,656/20,000"), ("Ready targets", "7,193/10,000"),
                         ("Search targets", "1,614/256")):
        row = next(line for line in text.splitlines() if label in line)
        assert count in row and "━" in row
    assert "158%" not in text and "Waiting for ready targets" in text
    assert "Ctrl-C closes view" in text.splitlines()[-1]


def test_last_incomplete_target_never_rounds_to_complete(live_teacher):
    run, progress = live_teacher
    progress.update(warmup_samples=40000, ready_samples=9999, searched_samples=300)
    text = render(run, [], progress)
    row = next(line for line in text.splitlines() if "Warmup readiness" in line)
    assert "99%" in row and "100%" not in row
    assert "Waiting for ready targets" in text


def test_complete_replay_gates_wait_for_the_actual_first_update(live_teacher):
    run, progress = live_teacher
    progress.update(warmup_samples=40000, ready_samples=10000, searched_samples=300)
    text = render(run, [], progress)
    row = next(line for line in text.splitlines() if "Warmup readiness" in line)
    assert "100%" in row and "awaiting first learner update" in text
    assert "WARMING UP" in text and "TRAINING" not in text


def test_missing_gate_counts_are_unavailable_not_empty(live_teacher):
    run, progress = live_teacher
    del progress["ready_samples"]
    text = render(run, [], progress)
    row = next(line for line in text.splitlines() if "Ready targets" in line)
    assert "counts unavailable" in row and "0/" not in row
    assert "Warmup readiness: counts unavailable" in text


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (160, 48)])
@pytest.mark.parametrize("state", ["learning", "warming_up", "stale", "stopped", "failed", "empty"])
def test_layout_states_fit_and_keep_controls(live_teacher, monkeypatch, size, state):
    run, progress = live_teacher
    now = time.time()
    progress.update(phase="learning" if state == "stale" else state, sps=127,
                    update=100, trained=123456, started_at=now - 300)
    records = [{"trainer": "teach", "update": 75, "time": now - 60, "pg": .1234, "vf": .0123,
                "ep_len": 428, "value_explained": -.42, "sps": 99, "win_vs_hunter": .475,
                "match_results": {"win_vs_hunter": {"wins": 8, "draws": 3, "losses": 9, "games": 20}}}]
    if state == "stale":
        progress["time"] = now - 60
    if state in ("stopped", "failed", "empty"):
        monkeypatch.setattr(dash, "_processes", lambda: {})
    if state == "empty":
        progress, records = {}, []
    text = render(run, records, progress, *size)
    assert "Ctrl-C closes view; trainer unaffected" in text.splitlines()[-1]
    assert "Checkpoint" in text and "Heartbeat" in text
    if state != "empty":
        assert "Update 100" in text and "Saved report u75" in text
        assert "8 / 3 / 9" in text and "n=20" in text and "47.5%" in text
        assert "Mean round duration" in text and "-0.420" in text
    if state in ("stale", "stopped", "failed", "empty"):
        assert "127 decisions/s" not in text and "Live activity" not in text
    else:
        assert "127 decisions/s" in text and "Live activity" in text


@pytest.mark.parametrize("size", [(40, 18), (60, 20), (100, 18)])
def test_small_terminals_fall_back_to_the_compact_console(live_teacher, size):
    run, progress = live_teacher
    progress.update(warmup_samples=31656, ready_samples=7193, searched_samples=1614)
    text = render(run, [], progress, *size)
    assert "WARMING UP" in text and "72%" in text
    assert "Ctrl-C closes" in text.rstrip().splitlines()[-1]
    assert "158%" not in text


def test_reports_from_a_previous_session_are_explicit(live_teacher):
    run, progress = live_teacher
    progress.update(update=100, phase="warming_up", started_at=time.time() - 30)
    record = {"update": 100, "time": time.time() - 120, "sps": 9999, "pg": .123,
              "value_explained": -.42, "ep_len": 428}
    text = render(run, [record], progress)
    assert "Saved report u100 · previous session" in text
    assert "9999 decisions/s" not in text
    assert "Mean round duration · 21.4 s" in text


def test_metrics_ahead_of_the_loaded_checkpoint_stay_hidden(live_teacher):
    run, progress = live_teacher
    progress.update(update=100, phase="learning")
    records = [{"update": 100, "time": time.time() - 10, "ep_len": 428, "value_explained": -.42},
               {"update": 200, "time": time.time() - 1, "ep_len": 999999, "value_explained": 77}]
    text = render(run, records, progress, 160, 48)
    assert "Saved report u100" in text and "-0.420" in text
    assert "77.000" not in text and "49,999.95" not in text


def test_checkpoint_bar_uses_disk_save_time_and_stays_due(live_teacher, monkeypatch):
    run, progress = live_teacher
    now = time.time()
    monkeypatch.setattr(dash.time, "time", lambda: now)
    progress.update(phase="learning", update=100, time=now)
    saved = run / "latest.pt"
    saved.touch()
    os.utime(saved, (now - 360, now - 360))
    text = render(run, [], progress)
    assert "Next save" in text and "100%" in text and "6:00/5:00 due" in text
    os.utime(saved, (now - 30, now - 30))
    text = render(run, [], progress)
    assert "0:30/5:00" in text and " due" not in text


def test_checkpoint_header_does_not_inventory_retired_pool_files(live_teacher):
    run, progress = live_teacher
    pool = run / "pool"
    pool.mkdir()
    for name in ("u0000100.pt", "x0001-u0000100.pt"):
        (pool / name).touch()
    (run / "latest.pt").touch()
    text = render(run, [], progress)
    assert "Checkpoint" in text and "2 snapshots" not in text and "past versions" not in text


def test_narrow_header_keeps_settings_separate_from_checkpoint(live_teacher, monkeypatch):
    run, progress = live_teacher
    now = time.time()
    monkeypatch.setattr(dash.time, "time", lambda: now)
    checkpoint = run / "latest.pt"
    checkpoint.touch()
    os.utime(checkpoint, (now - 145, now - 145))
    text = render(run, [], progress, 80, 24)
    row = next(line for line in text.splitlines() if "Checkpoint" in line)
    assert "combat starts 25%" in row and "Checkpoint 2m25s" in row
    assert row.index("Checkpoint") - (row.index("combat starts 25%") + len("combat starts 25%")) >= 2
    assert "0 = mean prediction" in text


def test_signed_graph_has_negative_values_and_respects_its_dimensions():
    graph = dash.AreaGraph([.5, -.75, -.42], dash.BLUE, hi=1, lo=-1, fmt="{:.2f}")
    rows = graph.render(18, 7)
    assert len(rows) == 7 and all(row.cell_len <= 18 for row in rows)
    assert any("-1.00" in row.plain for row in rows)
    assert any(any(0x2800 <= ord(char) <= 0x28ff for char in row.plain) for row in rows[4:])
    assert graph.render(0, 0) == []
