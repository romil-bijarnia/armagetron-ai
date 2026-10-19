"""One canonical checkpoint preserves learning across RL and imitation objectives."""

import fcntl
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from arma_ai import protocol as P
from arma_ai.model import make_net
from arma_ai.teach import Config, Learner, _half


def same_state(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected.cpu(), actual.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            same_state(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            same_state(left, right)
    else:
        assert expected == actual


def batch():
    steps = 3  # one burn-in observation and two learned decisions
    policy = torch.zeros(1, steps, P.N_ACTIONS)
    policy[..., 4] = 1
    return dict(maps=torch.zeros(1, steps, 2 * P.N_MAPS, P.GRID, P.GRID, dtype=torch.uint8),
                scalars=torch.zeros(1, steps, P.N_SCALARS), mask=torch.ones(1, steps, P.N_ACTIONS, dtype=torch.bool),
                context=torch.zeros(1, steps, 7), reset=torch.tensor([[True, False, False]]),
                policy=policy, has_policy=torch.ones(1, steps, dtype=torch.bool),
                learn=torch.tensor([[False, True, True]]), value=torch.tensor([[0., .25, -.25]]),
                aux=torch.full((1, steps, 3), .5), own=torch.zeros(1, steps, P.GRID, P.GRID, dtype=torch.long),
                own_ok=torch.zeros(1, steps, dtype=torch.bool))


@pytest.mark.parametrize("method", ["rl", "imitation"])
def test_canonical_resume_preserves_weights_and_resets_only_cross_method_moments(tmp_path, method):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with TemporaryDirectory(dir=tmp_path) as directory:
            run = Path(directory)
            net = make_net()
            optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4, eps=1e-6)
            net.pi.bias.grad = torch.ones_like(net.pi.bias)
            optimizer.step()
            old_model = {name: value.clone() for name, value in net.state_dict().items()}
            with torch.no_grad():
                net.pi.bias.add_(.2)  # the canonical policy is newer than the saved helper
            imitation = {"cumulative_steps": 12, "cumulative_samples": 96,
                         "rng_state": {"seed": 17}, "torch_rng_state": torch.get_rng_state()}
            if method == "imitation":
                imitation["optimizer"] = optimizer.state_dict()
            ck = {"trainer": "teach", "last_training_method": method,
                  "model": net.state_dict(), "opt": optimizer.state_dict(),
                  "exploiter": old_model, "exploiter_opt": optimizer.state_dict(),
                  "update": 41, "xupdates": 9, "generation": 3,
                  "trained": [2048, 128], "generated": [3000, 500], "decisions": 9000, "games": 90,
                  "imitation_state": imitation}
            torch.save(ck, run / "latest.pt")
            torch.save({"model": old_model}, run / "target.pt")
            learner = Learner(Config(run=str(run), device="cpu", bf16=False, games=2, threads=2,
                                     ring=8192, exploiter_ring=8192, warmup=16, batch=2,
                                     sequence_burn_in=1, sequence_unroll=2))
            try:
                same_state(ck["model"], learner.main.state_dict())
                same_state(_half(ck["model"]), torch.load(run / "target.pt", weights_only=True)["model"])
                assert (learner.update, learner.generation, learner.decisions, learner.games) == (41, 3, 9000, 90)
                assert learner.trained == [2048, 128] and learner.generated == [3000, 500]
                same_state(imitation, learner.imitation_state)
                if method == "imitation":
                    assert not learner.opt.state and not learner.xopt.state and learner.xupdates == 0
                    same_state(learner.main.state_dict(), learner.exploiter.state_dict())
                else:
                    same_state(ck["opt"], learner.opt.state_dict())
                    same_state(ck["exploiter_opt"], learner.xopt.state_dict())
                    same_state(old_model, learner.exploiter.state_dict())
                    assert learner.xupdates == 9

                # Opening and saving without a main-policy update must preserve imitation momentum.
                learner.save()
                untouched = torch.load(run / "latest.pt", weights_only=True)
                assert untouched["last_training_method"] == method
                same_state(imitation, untouched["imitation_state"])
                learner.step(learner.exploiter, learner.xopt, batch())
                assert learner.last_training_method == method
                same_state(imitation, learner.imitation_state)

                before = learner.main.pi.bias.detach().clone()
                learner.step(learner.main, learner.opt, batch())
                assert not torch.equal(before, learner.main.pi.bias)
                assert learner.last_training_method == "rl"
                assert "optimizer" not in learner.imitation_state
                assert learner.imitation_state["cumulative_steps"] == 12
                assert learner.imitation_state["cumulative_samples"] == 96
                same_state(imitation["rng_state"], learner.imitation_state["rng_state"])
                same_state(imitation["torch_rng_state"], learner.imitation_state["torch_rng_state"])
                learner.save()
                saved = torch.load(run / "latest.pt", weights_only=True)
                assert saved["trainer"] == "teach" and saved["last_training_method"] == "rl"
                assert "optimizer" not in saved["imitation_state"]
                assert not (run / "previous.pt").exists() and not (run / "pool").exists()
            finally:
                learner.cleanup_runtime_models()
                for ring in learner.rings:
                    ring.close(unlink=True)
                learner.log.close()
                learner._lock.close()
            assert sorted(path.name for path in run.glob("*.pt")) == ["latest.pt"]
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("history", ["progress", "metrics"])
def test_missing_canonical_checkpoint_never_restarts_a_trained_run(tmp_path, monkeypatch, history):
    import arma_ai.sim as sim
    import arma_ai.teach as teach

    run = tmp_path / "established"
    run.mkdir()
    if history == "progress":
        (run / "progress.json").write_text(json.dumps({"update": 123, "phase": "stopped"}))
    else:
        (run / "metrics.jsonl").write_text('{"update":0}\n{"update":123}\n{"partial":')
    (run / "previous.pt").write_bytes(b"retired checkpoint must never be loaded")
    monkeypatch.setattr(sim, "build", lambda: None)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("A missing canonical checkpoint reached model allocation")

    monkeypatch.setattr(teach, "make_net", unexpected_allocation)
    with pytest.raises(FileNotFoundError, match="missing, but this run has trained updates"):
        Learner(Config(run=str(run), device="cpu", ring=8192, exploiter_ring=8192, warmup=16, batch=2))
    assert not (run / "latest.pt").exists()
    with (run / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # rejection released the shared run lock
