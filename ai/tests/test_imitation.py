"""Expert provenance, deployment history and isolation of imitation checkpoints."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from arma_ai import protocol as P
from arma_ai.imitate import Config, Demonstrations, ImitationTrainer, episode_samples, metrics, validate_provenance


def provenance(rounds=5):
    return {"schema_version": 1, "observation_version": P.OBSERVATION_VERSION, "n_scalars": P.N_SCALARS, "teacher_type": "engine_builtin", "iq": 100,
            "decision_interval": .05, "action_encoding": "steer_brake_6",
            "observation_timing": "before_all_teacher_actions", "label_source": "native_executed_control",
            "teacher_schedule": "policy_decision_boundaries", "emergency_help": False,
            "teacher_slots": 1, "builtin_opponents": 1,
            "engine_sha256": "a" * 64, "source_fingerprint": {"engine": "b" * 64},
            "rounds": [{"complete": True, "truncated": False} for _ in range(rounds)]}


def episode(seed=0):
    flags = np.full((6, 1), P.FLAG_ALIVE | P.FLAG_NEEDS_ACTION, np.uint8)
    flags[0, 0] |= P.FLAG_SPAWNED
    maps = np.zeros((6, 1, 4, 64, 64), np.uint8)
    for tick in range(6):
        maps[tick] = seed * 10 + tick
    return {"times": np.arange(6, dtype=np.float64) * .05, "maps": maps,
            "scalars": np.full((6, 1, P.N_SCALARS), seed / 10, np.float32),
            "masks": np.full((6, 1), 63, np.uint8), "actions": np.array([[0], [1], [2]] * 2, np.int8),
            "flags": flags}


def recording(path, duplicate=False):
    rounds = [episode(i) for i in range(5)]
    if duplicate:
        rounds.append(episode(0))
    contents = {f"r{i}_{key}": val for i, r in enumerate(rounds) for key, val in r.items()}
    np.savez_compressed(path, n_rounds=np.array(len(rounds)), metadata=np.array(json.dumps(provenance(len(rounds)))),
                        **contents)
    return path


@pytest.mark.parametrize("key,value", [("teacher_type", "scripted"), ("iq", 101),
                                      ("label_source", "inferred_position"),
                                      ("observation_timing", "after_action"), ("decision_interval", .1),
                                      ("emergency_help", True), ("teacher_slots", 2), ("builtin_opponents", 0)])
def test_reject_untrusted_or_misaligned_experts(key, value):
    meta = provenance()
    meta[key] = value
    with pytest.raises(ValueError):
        validate_provenance(meta)


def test_history_is_prior_observation_and_resets_at_spawn_gap_and_dead_row():
    r = episode()
    r["flags"][2] = P.FLAG_DIED
    r["actions"][2] = -1
    r["masks"][2] = 0
    r["flags"][4] |= P.FLAG_SPAWNED
    r["times"][5] += .05
    rows = episode_samples(r, 7)
    assert rows.tolist() == [[7, 0, 0, 0], [7, 1, 0, 0], [7, 3, 0, 3], [7, 4, 0, 4], [7, 5, 0, 5]]


def test_refuses_illegal_or_missing_teacher_label():
    r = episode()
    r["masks"][1] = 1
    with pytest.raises(ValueError, match="illegal"):
        episode_samples(r, 0)
    r = episode()
    r["actions"][1] = -1
    with pytest.raises(ValueError, match="exactly one"):
        episode_samples(r, 0)


def test_teacher_death_finishes_a_life_without_inventing_a_finished_four_player_match(tmp_path):
    path = recording(tmp_path / "expert.npz")
    with np.load(path, allow_pickle=False) as z:
        arrays = dict(z)
    meta = json.loads(str(arrays["metadata"]))
    meta["rounds"][0]["complete"] = False
    arrays["r0_flags"][-1] = P.FLAG_DIED
    arrays["r0_actions"][-1] = -1
    arrays["r0_masks"][-1] = 0
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    assert len(Demonstrations([path]).episodes) == 5
    arrays["r0_flags"][-1] = P.FLAG_ALIVE
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="complete teacher lives"):
        Demonstrations([path])


def test_truncated_recording_is_rejected_even_after_a_terminal_flag(tmp_path):
    path = recording(tmp_path / "expert.npz")
    with np.load(path, allow_pickle=False) as z:
        arrays = dict(z)
    meta = json.loads(str(arrays["metadata"]))
    meta["rounds"][0]["truncated"] = True
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="truncated"):
        Demonstrations([path])


def test_split_keeps_players_and_frames_together_and_deduplicates_across_files(tmp_path):
    path = recording(tmp_path / "expert.npz", duplicate=True)
    data = Demonstrations([path, path])
    assert len(data.episodes) == 5
    assert data.summary["duplicate_rounds_removed"] == 7
    assert set(data.rows[data.training, 0]).isdisjoint(data.rows[data.validation, 0])
    assert len(data.training) + len(data.validation) == 30
    x, scalars, mask, labels = data.batch(np.array([0, 1, 2]), torch.device("cpu"))
    assert x.shape == (3, 8, 64, 64)
    assert x[:, 0, 0, 0].tolist() == [0, 1, 2]
    assert x[:, 4, 0, 0].tolist() == [0, 0, 1]
    assert mask.all() and labels.tolist() == [0, 1, 2]


def test_metrics_expose_straight_only_and_excessive_turn_policies():
    labels = np.array([0] * 9 + [1, 2])
    logits = np.full((11, 6), -10., np.float32)
    logits[:, 0] = 0
    result = metrics(logits, labels, np.ones(6))
    assert result["accuracy"] == result["straight_only_accuracy"] == 9 / 11
    assert result["turn_accuracy"] == result["false_turn_rate"] == 0
    logits[:, 1] = 1
    result = metrics(logits, labels, np.ones(6))
    assert result["false_turn_rate"] == 1
    assert result["action_recall"][3] is None


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("net_version", torch.tensor(4))
        self.register_buffer("observation_version", torch.tensor(P.OBSERVATION_VERSION))
        self.register_buffer("memory_version", torch.tensor(1))
        self.trunk = nn.Linear(P.N_SCALARS + 1, 8)
        self.pi = nn.Linear(8, 6)
        self.v = nn.Linear(8, 1)
        self.aux = nn.Linear(8, 3)
        self.memory_gru = nn.GRU(15, 8, batch_first=True)
        self.memory_projection = nn.Linear(8, 8)
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)

    def initial_state(self, batch, device=None):
        return torch.zeros(batch, 8, device=device)

    def remembered(self, maps, scalars, state, reset=None, context=None):
        features = torch.cat([maps.float().mean((2, 3, 4))[..., None] / 255, scalars], -1)
        features = torch.tanh(self.trunk(features))
        if context is None:
            context = features.new_zeros(*features.shape[:2], 7)
        outputs = []
        for tick in range(features.shape[1]):
            if reset is not None:
                state = state * (~reset[:, tick])[:, None]
            output, hidden = self.memory_gru(torch.cat([features[:, tick], context[:, tick]], -1)[:, None],
                                              state[None])
            state = hidden[0]
            outputs.append(features[:, tick] + self.memory_projection(output[:, 0]))
        return torch.stack(outputs, 1), state

    def heads_sequence(self, maps, scalars, mask, state, reset=None, context=None):
        h, state = self.remembered(maps, scalars, state, reset, context)
        return (self.pi(h).masked_fill(~mask, -1e9), torch.tanh(self.v(h)).squeeze(-1),
                self.aux(h), h.new_zeros(*h.shape[:2], 3, 1, 1), state)

    def forward_sequence(self, maps, scalars, mask, state, reset=None, context=None):
        logits, value, _, _, state = self.heads_sequence(maps, scalars, mask, state, reset, context)
        return logits, value, state

    def forward(self, maps, scalars, mask):
        features = torch.cat([maps.float().mean((1, 2, 3))[:, None] / 255, scalars], 1)
        h = torch.tanh(self.trunk(features))
        return self.pi(h).masked_fill(~mask, -1e9), torch.tanh(self.v(h)).squeeze(1)


def shared_checkpoint(path):
    net = TinyPolicy()
    optimizer = torch.optim.AdamW(net.parameters())
    state = net.state_dict()
    torch.save({"trainer": "teach", "model": state, "opt": optimizer.state_dict(),
                "exploiter": state, "exploiter_opt": optimizer.state_dict(),
                "update": 7514, "trained": [1000, 0], "generated": [2000, 0],
                "decisions": 900, "games": 10}, path)
    return state


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_state(a, b)
    else:
        assert left == right


def test_two_sessions_continue_exact_weights_optimizer_and_counters_in_one_file(tmp_path, monkeypatch):
    from arma_ai import imitate
    monkeypatch.setattr(imitate, "make_net", lambda _: TinyPolicy())
    data = Demonstrations([recording(tmp_path / "expert.npz")])
    run = tmp_path / "v4"
    run.mkdir()
    initial = shared_checkpoint(run / "latest.pt")
    cfg = Config(epochs=1, steps_per_epoch=2, batch=8, device="cpu", learning_rate=.01)
    with ImitationTrainer(run, cfg) as first:
        assert first.update == 7514 and not first.optimizer_resumed
        first.learn(data)
        assert first.update == 7516
    saved = torch.load(run / "latest.pt", weights_only=True)
    assert saved["imitation_state"]["cumulative_steps"] == 2
    assert saved["trained"] == [1016, 0]
    assert saved["opt"]["state"] == {}
    assert saved["last_training_method"] == "imitation"
    assert not torch.equal(saved["model"]["pi.weight"], initial["pi.weight"])
    for name in ("v.weight", "v.bias", "aux.weight", "aux.bias"):
        assert torch.equal(saved["model"][name], initial[name])
    with ImitationTrainer(run, cfg) as second:
        assert second.update == 7516 and second.steps == 2 and second.optimizer_resumed
        same_state(saved["model"], second.net.state_dict())
        same_state(saved["imitation_state"]["optimizer"], second.optimizer.state_dict())
        second.learn(data)
    latest = torch.load(run / "latest.pt", weights_only=True)
    assert latest["update"] == 7518
    assert latest["imitation_state"]["cumulative_steps"] == 4
    assert latest["imitation_state"]["cumulative_samples"] == 32
    assert latest["trained"] == [1032, 0]
    assert all(float(state["step"]) == 4 for state in latest["imitation_state"]["optimizer"]["state"].values())
    assert [p.name for p in run.rglob("*.pt")] == ["latest.pt"]
    assert not (run / "imitation.pid").exists()
    progress = json.loads((run / "progress.json").read_text())
    assert progress["phase"] == "stopped" and progress["update"] == 7518


def test_second_writer_cannot_touch_the_shared_checkpoint(tmp_path, monkeypatch):
    from arma_ai import imitate
    monkeypatch.setattr(imitate, "make_net", lambda _: TinyPolicy())
    shared_checkpoint(tmp_path / "latest.pt")
    original = (tmp_path / "latest.pt").read_bytes()
    with ImitationTrainer(tmp_path, Config(device="cpu")):
        with pytest.raises(RuntimeError, match="already being trained"):
            ImitationTrainer(tmp_path, Config(device="cpu"))
    assert (tmp_path / "latest.pt").read_bytes() == original


def test_missing_checkpoint_does_not_silently_create_a_new_fighter(tmp_path):
    with pytest.raises(FileNotFoundError, match="will not silently start"):
        ImitationTrainer(tmp_path, Config(device="cpu"))
    assert not (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "imitation.pid").exists()


def test_interrupt_saves_completed_updates_in_the_main_checkpoint(tmp_path, monkeypatch):
    from arma_ai import imitate
    monkeypatch.setattr(imitate, "make_net", lambda _: TinyPolicy())
    data = Demonstrations([recording(tmp_path / "expert.npz")])
    shared_checkpoint(tmp_path / "latest.pt")

    def interrupt(phase, done, total, info):
        if phase == "learning" and done == 2:
            raise KeyboardInterrupt

    with ImitationTrainer(tmp_path, Config(epochs=1, steps_per_epoch=4, batch=8, device="cpu"), interrupt) as trainer:
        trainer.learn(data)
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert latest["update"] == 7516
    assert latest["imitation_state"]["cumulative_steps"] == 2
    assert latest["imitation_state"]["validation_update"] is None
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "stopped"


def test_noop_resume_does_not_rewrite_or_reset_the_checkpoint(tmp_path, monkeypatch):
    from arma_ai import imitate
    monkeypatch.setattr(imitate, "make_net", lambda _: TinyPolicy())
    data = Demonstrations([recording(tmp_path / "expert.npz")])
    shared_checkpoint(tmp_path / "latest.pt")
    cfg = Config(epochs=1, steps_per_epoch=1, batch=8, device="cpu")
    with ImitationTrainer(tmp_path, cfg) as trainer:
        trainer.learn(data)
    original = (tmp_path / "latest.pt").read_bytes()
    with ImitationTrainer(tmp_path, cfg) as trainer:
        assert trainer.optimizer_resumed
    assert (tmp_path / "latest.pt").read_bytes() == original
