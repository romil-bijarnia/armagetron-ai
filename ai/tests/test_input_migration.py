"""Schema upgrades retain learned behavior, optimizer moments and training continuity."""

import copy
import hashlib

import pytest
import torch

from arma_ai.migrate_inputs import EXPANDED, convert_checkpoint, migrate_file
from arma_ai.model import PolicyNet, make_net, net_version
from arma_ai import protocol as P


@pytest.fixture(scope="module")
def legacy_checkpoint():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(8)
    net = make_net()
    names = [name for name, _ in net.named_parameters()]
    policy_names = [name for name in names if not name.startswith(("v.", "aux.", "own."))]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    imitation_opt = torch.optim.AdamW([p for n, p in net.named_parameters() if n in policy_names], lr=1e-5)
    net.opponents[0].weight.grad = torch.full_like(net.opponents[0].weight, .2)
    net.pi.weight.grad = torch.full_like(net.pi.weight, .3)
    opt.step()
    imitation_opt.step()
    model = {k: v.detach().clone() for k, v in net.state_dict().items() if k != "observation_version"}
    model[EXPANDED] = model[EXPANDED][:, :8].clone()

    def legacy_optimizer(optimizer, ordered):
        state = copy.deepcopy(optimizer.state_dict())
        pid = state["param_groups"][0]["params"][ordered.index(EXPANDED)]
        for key in ("exp_avg", "exp_avg_sq"):
            state["state"][pid][key] = state["state"][pid][key][:, :8].clone()
        return state

    checkpoint = {"trainer": "teach", "model": model, "exploiter": copy.deepcopy(model),
                  "opt": legacy_optimizer(opt, names), "exploiter_opt": legacy_optimizer(opt, names),
                  "update": 11280, "trained": [3895936, 0], "decisions": 1269816, "games": 4072,
                  "last_training_method": "imitation", "imitation_state": {
                      "cumulative_steps": 3766, "cumulative_samples": 241024,
                      "optimizer": legacy_optimizer(imitation_opt, policy_names)}}
    yield checkpoint
    torch.set_num_threads(old_threads)


class LegacyReference(PolicyNet):
    """The old input adapter only; every visual and decision layer is the same V4."""

    def __init__(self):
        super().__init__()
        self.opponents[0] = torch.nn.Linear(8, 64)
        del self._buffers["observation_version"]

    @staticmethod
    def opponent_input(scalars):
        assert scalars.shape[1] == 96
        return scalars[:, 64:88].reshape(-1, 3, 8)


def test_existing_weights_and_all_adam_moments_are_preserved(legacy_checkpoint):
    old = legacy_checkpoint
    converted, report = convert_checkpoint(old)
    assert report["migrated"] and report["new_parameters"] == 256
    assert report["update_before"] == report["update_after"] == 11280
    assert converted["trained"] == old["trained"]
    assert converted["imitation_state"]["cumulative_steps"] == 3766
    for model in ("model", "exploiter"):
        assert net_version(converted[model]) == 4
        for name, value in old[model].items():
            actual = converted[model][name]
            if name == EXPANDED:
                assert actual.shape == (64, 12) and torch.count_nonzero(actual[:, 8:]) == 0
                assert torch.equal(actual[:, :8], value)
            else:
                assert torch.equal(actual, value)
        assert "observation_version" not in old[model]
    for a, b in ((old["opt"], converted["opt"]),
                 (old["exploiter_opt"], converted["exploiter_opt"]),
                 (old["imitation_state"]["optimizer"], converted["imitation_state"]["optimizer"])):
        assert a["param_groups"] == b["param_groups"]
        for pid, values in a["state"].items():
            for key, value in values.items():
                actual = b["state"][pid][key]
                if value.shape == (64, 8):
                    assert torch.equal(actual[:, :8], value) and torch.count_nonzero(actual[:, 8:]) == 0
                else:
                    assert torch.equal(actual, value)


def test_new_measurements_initially_preserve_old_policy_value_and_aux_outputs(legacy_checkpoint):
    converted, _ = convert_checkpoint(legacy_checkpoint)
    before, after = LegacyReference().eval(), make_net().eval()
    before.load_state_dict(legacy_checkpoint["model"])
    after.load_state_dict(converted["model"])
    torch.manual_seed(9)
    maps = torch.randint(0, 256, (4, 8, 64, 64), dtype=torch.uint8)
    original = torch.randn(4, 96)
    original[:, [71, 79, 87]] = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    expanded = torch.cat([original, torch.rand(4, 12)], 1)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[1, 1:3] = False
    with torch.no_grad():
        a, b = before.heads(maps, original, mask), after.heads(maps, expanded, mask)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=2e-6)
    assert a[0].argmax(1).tolist() == b[0].argmax(1).tolist()
    assert sum(p.numel() for p in after.parameters()) == 8_155_373


def test_file_migration_is_explicit_atomic_and_idempotent(tmp_path, legacy_checkpoint):
    path = tmp_path / "latest.pt"
    torch.save(legacy_checkpoint, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="changed during verification"):
        migrate_file(path, "wrong fingerprint")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    report = migrate_file(path, before)
    assert report["migrated"] and report["update_after"] == 11280
    after = path.read_bytes()
    assert not migrate_file(path)["migrated"] and path.read_bytes() == after
    assert [p.name for p in tmp_path.glob("*.pt")] == ["latest.pt"]


def test_malformed_optimizer_cannot_replace_a_good_checkpoint(tmp_path, legacy_checkpoint):
    ck = copy.deepcopy(legacy_checkpoint)
    pid = next(iter(ck["imitation_state"]["optimizer"]["state"]))
    ck["imitation_state"]["optimizer"]["state"][pid]["exp_avg"] = torch.zeros(2, 2)
    path = tmp_path / "latest.pt"; torch.save(ck, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible shape"):
        migrate_file(path)
    assert path.read_bytes() == original
