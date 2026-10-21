"""Warmup reflects every learner gate, including the screenshot's overfull replay case."""

import numpy as np
import pytest

from arma_ai.progress_state import replay_requirements, warmup_state


def progress(**updates):
    return {"warmup_target": 20000, "batch": 512, "warmup_samples": 31656,
            "ready_samples": 7193, "searched_samples": 1614, **updates}


def test_overfull_collection_reports_the_actual_unmet_gate():
    state = warmup_state(progress())
    assert state["fraction"] == pytest.approx(.7193)
    assert not state["complete"] and state["waiting"] == ["Ready targets"]
    assert [g["fraction"] for g in state["gates"]] == [1, .7193, 1]
    assert [g["current"] for g in state["gates"]] == [31656, 7193, 1614]
    assert [g["target"] for g in state["gates"]] == [20000, 10000, 256]


@pytest.mark.parametrize("changes,fraction,waiting", [
    ({"warmup_samples": 16000, "ready_samples": 12000}, .8, ["Positions"]),
    ({"ready_samples": 12000, "searched_samples": 128}, .5, ["Search targets"]),
    ({"ready_samples": 10000, "searched_samples": 256}, 1, []),
])
def test_all_three_requirements_must_be_met(changes, fraction, waiting):
    state = warmup_state(progress(**changes))
    assert state["fraction"] == pytest.approx(fraction)
    assert state["waiting"] == waiting
    assert state["complete"] == (not waiting)


def test_explicit_targets_and_saved_configuration_are_supported():
    cfg = {"config": {"warmup": 20000, "batch": 512}, "warmup_samples": 31656,
           "ready_samples": 7193, "searched_samples": 1614}
    assert warmup_state(cfg) == warmup_state(progress())
    cfg.update(warmup_ready_target=5000, warmup_searched_target=200)
    assert warmup_state(cfg)["complete"]


@pytest.mark.parametrize("bad", [None, -1, True, float("nan"), float("inf"), "10000"])
def test_missing_or_invalid_counts_do_not_invent_readiness(bad):
    state = warmup_state(progress(ready_samples=bad))
    assert state["fraction"] is None and not state["complete"]
    assert state["gates"][1]["current"] is None


def test_requirements_match_odd_batches_and_small_warmup():
    assert replay_requirements(20000, 511) == (20000, 10000, 0)
    assert replay_requirements(16, 31) == (16, 31, 0)
    assert replay_requirements(0, 3) == (0, 3, 0)
    assert replay_requirements(20000, 511, search=True) == (20000, 10000, 255)


@pytest.mark.parametrize("ready,searched,allowed", [(7193, 0, False), (10000, 1, False),
                                                     (10000, 0, True)])
def test_display_requirements_agree_with_the_real_batch_gate(ready, searched, allowed):
    from arma_ai.teach import Config, Learner

    class BatchAccepted(Exception):
        pass

    class Ring:
        maps = np.zeros((ready, 1), np.uint8)
        gen = np.arange(ready)
        has_policy = (np.arange(ready) < searched).astype(np.uint8)
        policy_update = np.zeros(ready, np.int64)

        def window(self):
            return np.arange(ready)

        def sample(self, n, _rng, policy_only=False, **_kwargs):
            return np.arange(min(n, searched if policy_only else ready))

        def sequences(self, *_args, **_kwargs):
            raise BatchAccepted()

    learner = Learner.__new__(Learner)
    learner.cfg = Config(warmup=20000, batch=512)
    ring = Ring()
    learner.rings, learner.update = [ring], 0
    learner.rng = np.random.default_rng(0)
    assert warmup_state(progress(ready_samples=ready-searched, searched_samples=0,
                                 warmup_searched_target=0))["complete"] == allowed
    if allowed:
        with pytest.raises(BatchAccepted):
            learner.batch(ring, 512)
    else:
        assert learner.batch(ring, 512) is None
