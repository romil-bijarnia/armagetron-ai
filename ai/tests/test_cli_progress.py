"""Compact terminal reporting keeps denominators and stale/warmup states honest."""

import io
import os
from types import SimpleNamespace

import pytest
from rich.console import Console

from arma_ai.cli_progress import build, report_lines


def test_report_lines_show_gain_match_counts_and_live_rates():
    record = {"trainer": "teach", "update": 11300, "time": 100, "search_kl": .176, "pg": .276, "vf": .374,
              "ent": .33, "value_explained": .26, "win_vs_hunter": 0., "win_exploiter": 1.,
              "match_results": {"win_vs_hunter": {"wins": 0, "draws": 0, "losses": 2, "games": 2},
                                "win_exploiter": {"wins": 1, "draws": 0, "losses": 0, "games": 1}}}
    progress = {"trainer": "teach", "update": 11305, "time": 120, "phase": "learning", "sps": 16,
                "config": {"snapshot_every": 1000, "log_every": 25}}
    live = SimpleNamespace(fresh=True, running=True, label="TRAINING", detail="", method="teach")
    lines = [line.plain for line in report_lines(progress, [record], live, width=110, now=120, started=80)]
    assert lines == ["Report u11,300 · 20s ago · gain 0.176 · policy 0.276 · value 0.374 · entropy 0.33 · value EV 0.26",
                     "Space fighter 0% (0W 0D 2L) · Exploiter 100% (1W 0D 0L)",
                     "16 decisions/s · 15.0 updates/min · next snapshot u12,000"]
    narrow = [line.plain for line in report_lines(progress, [record], live, width=60, now=120, started=80)]
    assert narrow[:2] == ["Report u11,300 · 20s ago · gain 0.176 · policy 0.276 · value 0.374",
                          "Hunter 0% (0W 0D 2L) · Exploiter 100% (1W 0D 0L)"]
    stopped = SimpleNamespace(fresh=False, running=False, label="STOPPED", detail="", method="teach")
    assert len(report_lines(progress, [record], stopped, width=110, now=120)) == 2  # no live rates
    assert report_lines(progress, [], stopped, width=110, now=120)[0].plain == "No saved report yet"


def test_terminal_shows_search_gain_with_the_losses(tmp_path):
    text = output(tmp_path, {"trainer": "teach", "update": 100, "time": 119, "pg": .123, "vf": .456, "search_kl": .078},
                  {"trainer": "teach", "update": 101, "time": 120, "phase": "learning"})
    assert "Policy 0.1230  ·  Value 0.4560  ·  Gain 0.0780" in text


def output(tmp_path, record, progress, width=110, fresh=True, *, running=None, height=32, now=120):
    stream = io.StringIO()
    console = Console(file=stream, width=width, color_system=None)
    running = fresh if running is None else running
    status = SimpleNamespace(fresh=fresh, running=running,
                             label="TRAINING" if fresh and running else "STALE" if running else "STOPPED",
                             detail="The trainer heartbeat is stale.")
    console.print(build(tmp_path, progress, [record], status, width=width, height=height,
                        now=now, started=80))
    text = stream.getvalue()
    assert all(len(line) <= width for line in text.splitlines())
    return text


@pytest.mark.parametrize("width", [60, 80, 110])
def test_terminal_displays_measured_counts_with_score(tmp_path, width):
    row = {"trainer": "teach", "update": 100, "time": 119, "win_vs_hunter": 1 / 33,
           "match_results": {"win_vs_hunter": {"wins": 0, "draws": 2, "losses": 31, "games": 33}}}
    progress = {"trainer": "teach", "update": 101, "time": 120, "phase": "learning", "sps": 90}
    text = output(tmp_path, row, progress, width)
    assert "3.0%" in text and ("0/2/31 n33" if width < 76 else "0W 2D 31L · n=33") in text
    assert "not fixed benchmarks" in text if width >= 80 else "W/D/L" in text


def test_terminal_never_infers_games_from_old_percentages(tmp_path):
    text = output(tmp_path, {"trainer": "teach", "update": 100, "time": 119, "win_vs_hunter": .2},
                  {"update": 100, "time": 120, "phase": "learning"})
    assert "20.0%" in text and "count unavailable" in text


def test_terminal_hides_prior_losses_during_replay_warmup(tmp_path):
    text = output(tmp_path, {"update": 100, "time": 119, "pg": .123, "vf": .456},
                  {"update": 100, "time": 120, "phase": "warming_up", "warmup_target": 20000,
                   "warmup_samples": 20000, "ready_samples": 500, "searched_samples": 100, "batch": 512})
    assert "WARMUP" in text and "500/10,000" in text and "100/256" in text
    assert ".123" not in text and ".456" not in text


def test_warmup_percentage_tracks_ready_targets_after_collection_is_full(tmp_path):
    progress = {"update": 5595, "time": 120, "phase": "warming_up", "warmup_target": 20000,
                "warmup_samples": 31656, "ready_samples": 7193, "searched_samples": 1614, "batch": 512}
    text = output(tmp_path, {}, progress)
    assert "WARMUP" in text and "72%" in text and "158%" not in text
    assert "31,656/20,000" in text and "7,193/10,000" in text and "1,614/256" in text
    assert "Ready; waiting" not in text


def test_warmup_does_not_round_an_unmet_requirement_to_complete(tmp_path):
    progress = {"update": 5595, "time": 120, "phase": "warming_up", "warmup_target": 20000,
                "warmup_samples": 31656, "ready_samples": 9999, "searched_samples": 1614, "batch": 512}
    text = output(tmp_path, {}, progress)
    warmup = next(line for line in text.splitlines() if "WARMUP" in line)
    assert "99%" in warmup and "100%" not in warmup
    assert "Ready; waiting" not in text
    progress["ready_samples"] = 10000
    complete = output(tmp_path, {}, progress)
    assert "100%" in complete and "Ready; waiting for the first update." in complete


def test_terminal_does_not_show_stale_speed_as_live(tmp_path):
    text = output(tmp_path, {"update": 100, "time": 90},
                  {"update": 101, "time": 90, "phase": "learning", "sps": 999}, fresh=False, running=True)
    assert "STALE" in text and "waiting decisions/s" in text and "999 decisions/s" not in text


@pytest.mark.parametrize("width", [40, 60, 80, 110])
@pytest.mark.parametrize("height", [12, 16, 30])
@pytest.mark.parametrize("phase", ["learning", "warming_up", "stale"])
def test_resizing_keeps_status_settings_and_footer(tmp_path, width, height, phase):
    progress = {"trainer": "teach", "update": 101, "time": 120, "phase": phase, "sps": 90,
                "mode": "background", "config": {"device": "mps", "combat_prob": .25, "ring": 200000,
                                                    "log_every": 25, "save_seconds": 100},
                "trained": 123456, "games": 123, "ready_samples": 500, "searched_samples": 100,
                "warmup_target": 20000, "warmup_samples": 12000, "batch": 512}
    record = {"trainer": "teach", "update": 100, "time": 119, "pg": .123, "vf": .456,
              "win_vs_hunter": 1 / 33, "win_vs_past": .5, "win_exploiter": .25,
              "match_results": {key: {"wins": 0, "draws": 2, "losses": 31, "games": 33}
                                for key in ("win_vs_hunter", "win_vs_past", "win_exploiter")}}
    status = SimpleNamespace(fresh=phase != "stale", running=True,
                             label="STALE" if phase == "stale" else "WARMING UP" if phase == "warming_up" else "TRAINING",
                             detail="The trainer heartbeat is stale.")
    checkpoint = tmp_path / "latest.pt"
    checkpoint.touch()
    os.utime(checkpoint, (50, 50))
    stream = io.StringIO()
    console = Console(file=stream, width=width, height=height, color_system=None)
    console.print(build(tmp_path, progress, [record], status, width=width, height=height, now=120, started=80))
    text = stream.getvalue()
    lines = text.splitlines()
    assert len(lines) <= height and all(len(line) <= width for line in lines)
    assert status.label in text and "Update 101" in text
    assert "background" in text and "combat starts 25%" in text
    assert "Checkpoint" in text and "heartbeat" in text and "Ctrl-C closes" in lines[-1]
    assert "500 ready" in text and "100 searched" in text
    assert "3.0%" in text and ("0W/2D/31L n33" in text or "0/2/31 n33" in text or "0W 2D 31L · n=33" in text)
    if phase == "warming_up":
        assert "500/10,000" in text and "100/256" in text and "Policy" not in text
    else:
        assert "Loss u100" in text and "Policy 0.1230" in text


def test_empty_console_does_not_invent_progress(tmp_path):
    stream = io.StringIO()
    status = SimpleNamespace(fresh=False, running=False, label="STOPPED", detail="No live trainer.")
    Console(file=stream, width=40).print(build(tmp_path, {}, [], status, width=40, height=12, now=120))
    text = stream.getvalue()
    assert "Update not reported" in text and "awaiting first report" in text
    assert "0 decisions" not in text and "0%" not in text and "0.0000" not in text


def test_update_changes_do_not_flash_and_future_records_stay_hidden(tmp_path):
    progress = {"trainer": "teach", "update": 100, "time": 120, "phase": "learning"}
    records = [{"trainer": "teach", "update": 100, "time": 119, "pg": .25, "vf": .5},
               {"trainer": "teach", "update": 101, "time": 121, "pg": 99, "vf": 99}]
    status = SimpleNamespace(fresh=True, running=True, label="TRAINING", detail="")
    rendered = []
    for changed in (0, 119.5):
        stream = io.StringIO()
        console = Console(file=stream, width=80, force_terminal=True, color_system="truecolor")
        console.print(build(tmp_path, progress, records, status, width=80, now=120, changed_at=changed))
        rendered.append(stream.getvalue())
    assert rendered[0] == rendered[1] and "99.0000" not in rendered[0]


def learning_progress(**overrides):
    return {"trainer": "teach", "update": 37, "time": 120, "phase": "learning", "sps": 90,
            "ready_samples": 97952, "searched_samples": 20000, "max_updates": 0,
            "config": {"ring": 200000, "log_every": 25, "save_seconds": 100}, **overrides}


@pytest.mark.parametrize("width", [20, 28, 40, 60, 110])
def test_learning_bars_keep_real_counts_and_percentages_in_narrow_terminals(tmp_path, width):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.touch()
    os.utime(checkpoint, (70, 70))
    text = output(tmp_path, {"update": 25, "time": 110}, learning_progress(), width, height=40)
    assert "Replay ready" in text and "97952/195904" in text.replace(",", "")
    assert "Next report" in text and "12/25" in text and "48%" in text
    assert "Next save" in text and "0:50/1:40" in text
    assert text.count("50%") == 2 and "━" in text
    assert "Update target" not in text  # an unbounded learner has no completion percentage


def test_checkpoint_interval_stays_due_until_a_real_save(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.touch()
    os.utime(checkpoint, (10, 10))
    first = output(tmp_path, {}, learning_progress())
    later = output(tmp_path, {}, learning_progress(time=220), now=220)
    assert "100%  1:50/1:40 due" in first
    assert "100%  3:30/1:40 due" in later
    os.utime(checkpoint, (215, 215))
    saved = output(tmp_path, {}, learning_progress(time=220), now=220)
    assert "5%  0:05/1:40" in saved and " due" not in saved


@pytest.mark.parametrize("update,counts,percent", [(24, "24/25", "96%"), (25, "0/25", "0%")])
def test_next_report_bar_tracks_the_configured_reporting_interval(tmp_path, update, counts, percent):
    text = output(tmp_path, {}, learning_progress(update=update))
    row = next(row for row in text.splitlines() if row.startswith("Next report"))
    assert counts in row and percent in row


@pytest.mark.parametrize("fresh,running", [(False, True), (True, False), (False, False)])
def test_runtime_bars_and_rates_require_a_fresh_live_trainer(tmp_path, fresh, running):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.touch()
    os.utime(checkpoint, (10, 10))
    text = output(tmp_path, {}, learning_progress(max_updates=100), fresh=fresh, running=running)
    assert "━" not in text and "Next report" not in text and "Next save" not in text
    assert "90 decisions/s" not in text and "no live rate" in text
    assert "Update target 37/100" in text


def test_update_target_bar_requires_a_configured_bound(tmp_path):
    text = output(tmp_path, {}, learning_progress(max_updates=100))
    row = next(row for row in text.splitlines() if row.startswith("Update target"))
    assert "37%" in row and "37/100" in row


def test_missing_measurements_do_not_invent_empty_progress(tmp_path):
    progress = {"time": 120, "phase": "learning", "config": {"ring": 200000, "log_every": 25,
                                                              "save_seconds": 100}}
    text = output(tmp_path, {}, progress)
    assert "waiting for first checkpoint" in text and "Checkpoint not saved" in text
    assert "━" not in text and "%" not in text
    progress = {"time": 120, "phase": "warming_up", "warmup_target": 20000}
    text = output(tmp_path, {}, progress)
    assert "Waiting for replay target counts" in text and "━" not in text and "%" not in text
