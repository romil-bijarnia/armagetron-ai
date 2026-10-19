"""Benchmark outcomes remain honest under duplicates, timeouts and changing live weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arma_ai import evaluate as E
from arma_ai import protocol as P
from arma_ai.benchmarks import BenchmarkManager, load_results, worker


def state(round_id, flags, alive, over=True, *, tick=1, kills=0, engine=0):
    slot = SimpleNamespace(flags=flags, alive=bool(flags & P.FLAG_ALIVE), kills=kills)
    return SimpleNamespace(engine=engine, round_id=round_id, tick=tick, slots=[slot], n_alive=alive,
                           round_over=over, round_time=2.5)


class FakeBrain:
    def __init__(self, *_a, **_kw): self.update = 77
    def actions(self, _step): return [0]


class FakePool:
    batches = []
    closed = False
    def __init__(self, *_a, **_kw): self.iterator = iter(self.batches)
    def __enter__(self): return self
    def __exit__(self, *_a): type(self).closed = True
    def act(self, *_a): pass
    def poll(self, **_kw):
        item = next(self.iterator)
        if isinstance(item, BaseException): raise item
        return item


def test_benchmark_counts_each_terminal_round_once(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "Brain", FakeBrain)
    monkeypatch.setattr(E, "_EvaluationPool", FakePool)
    win = state(1, P.FLAG_ALIVE | P.FLAG_WON, 1)
    loss = state(2, P.FLAG_DIED, 1)
    draw = state(3, P.FLAG_DIED, 0)
    FakePool.batches = [[win, win], [win, loss, loss], [draw]]
    changes = []
    result = E.evaluate(tmp_path / "model.pt", None, 3, 1, -3, progress=changes.append)
    assert (result["wins"], result["draws"], result["losses"]) == (1, 1, 1)
    assert result["rounds_completed"] == 3 and not result["partial"]
    assert result["score"] == .5 and result["update"] == 77 and FakePool.closed
    assert [r["rounds_completed"] for r in changes] == [0, 1, 2, 3, 3]


def test_time_budget_keeps_completed_games_and_never_fills_missing_games(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "Brain", FakeBrain)
    monkeypatch.setattr(E, "_EvaluationPool", FakePool)
    FakePool.batches = [[state(1, P.FLAG_DIED, 0)], TimeoutError()]
    result = E.evaluate(tmp_path / "model.pt", None, 20, 1, -3, time_budget_s=60)
    assert result["rounds_completed"] == 1 and result["rounds_requested"] == 20
    assert result["draws"] == 1 and result["wins"] == result["losses"] == 0
    assert result["partial"] and result["status"] == "time_limit" and result["score"] == .5
    assert FakePool.closed
    zero = E.evaluate(tmp_path / "model.pt", None, 20, 1, -3, time_budget_s=0)
    assert zero["rounds_completed"] == 0 and zero["score"] is None


def test_combat_counts_credited_wins_separately_from_uncredited_wins_and_dead_killers(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "Brain", FakeBrain)
    monkeypatch.setattr(E, "_EvaluationPool", FakePool)
    attack = state(1, P.FLAG_ALIVE, 1, False, kills=1)
    attack_win = state(1, P.FLAG_ALIVE | P.FLAG_WON, 1, tick=2)
    uncredited_win = state(2, P.FLAG_ALIVE | P.FLAG_WON, 1)
    dead_killer = state(3, P.FLAG_DIED, 0, kills=1)
    loss = state(4, P.FLAG_DIED, 1)
    FakePool.batches = [[attack, attack], [attack_win, attack_win], [uncredited_win], [dead_killer, loss]]
    result = E.evaluate(tmp_path / "model.pt", None, 4, 1, -3)
    assert (result["wins"], result["draws"], result["losses"]) == (2, 1, 1)
    assert result["combat"] == {
        "completed_rounds": 4, "credited_kills": 2, "deaths": 2,
        "surviving_kill_rounds": 1, "surviving_kill_rate": .25,
        "wins_with_credited_kill": 1, "wins_without_credited_kill": 1,
        "kills_per_round": .5, "mean_win_s": 2.5,
    }


def test_combat_excludes_unfinished_games_and_keeps_engines_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "Brain", FakeBrain)
    monkeypatch.setattr(E, "_EvaluationPool", FakePool)
    unfinished = state(1, P.FLAG_ALIVE, 1, False, kills=1, engine=0)
    completed = state(1, P.FLAG_ALIVE | P.FLAG_WON, 1, engine=1)
    FakePool.batches = [[unfinished, completed], TimeoutError()]
    result = E.evaluate(tmp_path / "model.pt", None, 20, 2, -3, time_budget_s=60)
    assert result["partial"] and result["combat"]["completed_rounds"] == 1
    assert result["combat"]["credited_kills"] == result["combat"]["surviving_kill_rounds"] == 0
    assert result["combat"]["wins_without_credited_kill"] == 1


def test_combat_tracks_the_checkpoint_when_spawn_slots_alternate(monkeypatch, tmp_path):
    class DuelBrain(FakeBrain):
        def actions(self, _step): return [0, 0]

    monkeypatch.setattr(E, "Brain", DuelBrain)
    monkeypatch.setattr(E, "_EvaluationPool", FakePool)
    dead = SimpleNamespace(flags=P.FLAG_DIED, alive=False, kills=0)
    odd = state(1, P.FLAG_ALIVE | P.FLAG_WON, 1, kills=1)
    odd.slots.insert(0, dead)  # hero in slot 1
    even = state(2, P.FLAG_DIED, 1)
    even.slots.append(odd.slots[1])  # villain's kill must not count for the hero in slot 0
    FakePool.batches = [[odd], [even]]
    result = E.evaluate(tmp_path / "hero.pt", tmp_path / "villain.pt", 2, 1, -3)
    assert (result["wins"], result["losses"]) == (1, 1)
    assert result["combat"]["credited_kills"] == result["combat"]["surviving_kill_rounds"] == 1
    assert result["combat"]["deaths"] == 1


def test_results_loader_only_uses_measured_outcomes(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"win_vs_hunter": .8, "games": 900}) + "\n")
    assert load_results(tmp_path) == []
    rows = [{"wins": 2, "draws": 1, "losses": 0, "rounds": 3, "update": 10, "time": 1},
            {"wins": 0, "draws": 0, "losses": 0, "rounds": 20}, {"win_rate": .8, "rounds": 20}]
    (tmp_path / "eval.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n{")
    results = load_results(tmp_path)
    assert len(results) == 1 and results[0]["rounds_completed"] == 3
    assert results[0]["policy_mode"] == "unknown"


def test_manager_freezes_checkpoint_and_rejects_concurrent_check(monkeypatch, tmp_path):
    class Process:
        def __init__(self, *_a, **_kw): self.code = None
        def poll(self): return self.code
        def terminate(self): self.code = 0
        def wait(self, **_kw): return self.code
    monkeypatch.setattr("arma_ai.benchmarks.subprocess.Popen", Process)
    source = tmp_path / "latest.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}, "update": 7}, source)
    manager = BenchmarkManager(tmp_path, source)
    status = manager.start(time_budget_s=5)
    assert status["running"] and status["progress"]["update"] == 7
    torch.save({"model": {"weight": torch.tensor([2.0])}, "update": 8}, source)
    config = json.loads((manager.job / "config.json").read_text())
    snapshot = Path(config["checkpoint"])
    assert not snapshot.is_relative_to(tmp_path)
    assert not list(manager.job.glob("*.pt"))
    frozen = torch.load(snapshot, weights_only=True)
    assert frozen["update"] == 7 and frozen["model"]["weight"].item() == 1
    with pytest.raises(ValueError, match="already"):
        manager.start()
    manager.close()
    assert not snapshot.exists() and not snapshot.parent.exists()
    assert source.exists()


def test_manager_keeps_snapshot_until_cancelled_worker_stops(monkeypatch, tmp_path):
    class Process:
        def __init__(self, *_a, **_kw): self.code = None
        def poll(self): return self.code
        def terminate(self): pass  # SIGTERM was sent; the worker still needs its weights.
        def wait(self, **_kw): self.code = 0

    monkeypatch.setattr("arma_ai.benchmarks.subprocess.Popen", Process)
    source = tmp_path / "latest.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}, "update": 7}, source)
    manager = BenchmarkManager(tmp_path, source)
    manager.start(time_budget_s=5)
    snapshot = manager.snapshot_dir / "checkpoint.pt"
    assert manager.cancel()["status"] == "cancelling"
    assert snapshot.exists()
    manager.close()
    assert not snapshot.exists() and not snapshot.parent.exists()
    result = json.loads((manager.job / "result.json").read_text())
    assert result["status"] == "cancelled" and result["checkpoint"] == str(source)


def test_manager_cleans_snapshot_if_worker_cannot_launch(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("cannot launch worker")

    monkeypatch.setattr("arma_ai.benchmarks.subprocess.Popen", fail)
    source = tmp_path / "latest.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}, "update": 7}, source)
    manager = BenchmarkManager(tmp_path, source)
    with pytest.raises(OSError, match="cannot launch"):
        manager.start(time_budget_s=5)
    config = json.loads((manager.job / "config.json").read_text())
    assert not Path(config["snapshot_dir"]).exists()
    assert manager.snapshot_dir is None and manager.proc is None
    assert source.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_worker_cleans_its_snapshot_without_manager_polling(monkeypatch, tmp_path, fail):
    class Process:
        def __init__(self, *_a, **_kw): self.code = None
        def poll(self): return self.code

    monkeypatch.setattr("arma_ai.benchmarks.subprocess.Popen", Process)
    monkeypatch.setattr("arma_ai.benchmarks.signal.signal", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    source = tmp_path / "latest.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}, "update": 7}, source)
    manager = BenchmarkManager(tmp_path, source)
    manager.start(time_budget_s=5)
    snapshot = manager.snapshot_dir / "checkpoint.pt"

    def evaluate(checkpoint, *args, **kwargs):
        assert checkpoint == snapshot and checkpoint.exists()
        assert torch.load(checkpoint, weights_only=True)["update"] == 7
        if fail:
            raise RuntimeError("engine failed")
        return {"status": "complete", "wins": 1, "draws": 0, "losses": 0,
                "rounds": 1, "update": 7, "partial": False, "time": 1}

    monkeypatch.setattr(E, "evaluate", evaluate)
    worker(manager.job)
    assert not snapshot.exists() and not snapshot.parent.exists()
    assert source.exists()
    result = json.loads((manager.job / "result.json").read_text())
    assert result["status"] == ("error" if fail else "complete")
    assert result["update"] == 7 and result["checkpoint"] == str(source)
    manager.proc.code = 0
    manager.poll()  # Reaping after the worker already cleaned up is harmless.
