"""Bounded simulator actor/learner proof for the combat architecture and checkpoint resume."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arma_ai import protocol as P
from arma_ai.model import net_version
from arma_ai.teach import Config, Learner, _load_net


class RecordingLearner(Learner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.steps = []

    def step(self, net, opt, data):
        out = super().step(net, opt, data)
        self.steps.append({
            "losses": {k: float(v) for k, v in out.items()},
            "compound": float(net.pi.weight.grad[4:].abs().sum()),
            "safe_kill": float(net.aux.weight.grad[2].abs().sum()),
            "conditioning": sum(float(layer.weight.grad.abs().sum()) for layer in net.conditioning),
            "opponents": float(net.opponent_query.weight.grad.abs().sum()),
            "memory_projection": float(net.memory_projection.weight.grad.abs().sum()),
            "memory_context": float(net.memory_gru.weight_ih_l0.grad[:, -7:].abs().sum()),
        })
        return out


def assert_same_state(expected, actual):
    """Resuming must preserve every parameter and optimiser moment exactly."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected.cpu(), actual.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_same_state(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_same_state(left, right)
    else:
        assert expected == actual


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                                reason="Apple GPU is not available")),
])
def test_v4_teacher_actor_learner_checkpoint_resume(tmp_path: Path, device: str):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        kwargs = dict(run=str(tmp_path / "v4"), games=2, sims=2, macro=2, search_prob=1.0,
                      threads=2, ring=8192, exploiter_ring=8192, batch=8, warmup=16,
                      replay_ratio=1.0, hunter_prob=0.0, exploiter_prob=0.0,
                      sequence_burn_in=2, sequence_unroll=4,
                      value_steps=4, combat_prob=1, device=device, bf16=device == "mps", log_every=1)
        first = RecordingLearner(Config(**kwargs, max_updates=2, resume=False))
        assert first.main.version == 4
        progress_path = tmp_path / "v4" / "progress.json"
        starting = json.loads(progress_path.read_text())
        assert starting["phase"] == "starting" and starting["warmup_samples"] == 0
        assert starting["warmup_target"] == kwargs["warmup"] and starting["actions"] == 6
        assert starting["warmup_ready_target"] == 8 and starting["warmup_searched_target"] == 4
        assert 0 < starting["started_at"] <= starting["time"]
        first.run()
        assert first.update == 2
        assert first.generated[0] > 0 and first.decisions > 0
        assert first.log.closed and first._lock.closed
        assert all(r.shm.buf is None for r in first.rings)
        stopped = json.loads(progress_path.read_text())
        assert stopped["phase"] == "stopped" and stopped["update"] == 2
        assert stopped["samples"] == first.generated[0] and stopped["steps"] == first.decisions
        first_trained = sum(int(step["losses"]["n"]) for step in first.steps)
        assert 0 < stopped["trained"] == first_trained <= 2 * kwargs["batch"]
        assert stopped["max_updates"] == 2
        for step in first.steps:
            assert np.isfinite(list(step["losses"].values())).all()
            assert step["losses"]["safe_kill"] > 0
        for head in ("compound", "safe_kill", "conditioning", "opponents", "memory_projection", "memory_context"):
            assert any(step[head] > 0 for step in first.steps), f"no gradient through {head}"

        path = tmp_path / "v4" / "latest.pt"
        ck = torch.load(path, map_location="cpu", weights_only=True)
        assert net_version(ck["model"]) == 4 and ck["update"] == 2
        assert not {"net", "bootstrap", "bootstrap_trained", "warm_value_every"} & ck["config"].keys()
        assert "bootstrapping" not in ck
        assert ck["last_training_method"] == "rl"
        assert sorted(path.name for path in first.run_dir.glob("*.pt")) == ["latest.pt"]
        assert not (first.run_dir / "pool").exists()
        assert ck["opt"]["state"]  # the checkpoint includes resumable optimiser moments
        loaded, _ = _load_net(path, torch.device("cpu"))
        with torch.no_grad():
            logits, values = loaded(torch.zeros(1, 2 * P.N_MAPS, P.GRID, P.GRID, dtype=torch.uint8),
                                    torch.zeros(1, P.N_SCALARS), torch.ones(1, 6, dtype=torch.bool))
        assert logits.shape == (1, 6) and torch.isfinite(logits).all()
        assert torch.isfinite(values).all() and (values.abs() <= 1).all()

        # Existing V4 checkpoints may carry settings from the retired trainers. They are
        # ignored metadata and must not change how the current fighter resumes.
        ck["config"].update(net=4, bootstrap="runs/removed/latest.pt", bootstrap_trained=1_500_000,
                            warm_value_every=8)
        ck["bootstrapping"] = True
        ck.pop("last_training_method")  # older canonical RL checkpoints still resume exactly
        ck.pop("imitation_state")
        torch.save(ck, path)
        resumed = RecordingLearner(Config(**kwargs, max_updates=3, resume=True))
        assert resumed.update == 2
        assert_same_state(ck["model"], resumed.main.state_dict())
        assert_same_state(ck["opt"], resumed.opt.state_dict())
        assert_same_state(ck["exploiter"], resumed.exploiter.state_dict())
        assert_same_state(ck["exploiter_opt"], resumed.xopt.state_dict())
        restarting = json.loads(progress_path.read_text())
        assert restarting["warmup_samples"] == 0 and restarting["samples"] == stopped["samples"]
        resumed.run()
        assert resumed.update == 3 and len(resumed.steps) == 1
        assert resumed.trained[0] == first_trained + int(resumed.steps[0]["losses"]["n"])
        rows = [json.loads(line) for line in (tmp_path / "v4" / "metrics.jsonl").read_text().splitlines()]
        assert [row["update"] for row in rows] == [1, 2, 3]
        assert all(row["trainer"] == "teach" and row["actions"] == 6 and row["net"] == 4 for row in rows)
        final = json.loads(progress_path.read_text())
        assert final["phase"] == "stopped" and final["update"] == 3
        assert all(np.isfinite([row[k] for k in ("pg", "vf", "aux", "own", "safe_kill", "grad_norm")]).all()
                   for row in rows)
        assert torch.load(path, map_location="cpu", weights_only=True)["update"] == 3
        assert sorted(file.name for file in resumed.run_dir.glob("*.pt")) == ["latest.pt"]
    finally:
        torch.set_num_threads(old_threads)
