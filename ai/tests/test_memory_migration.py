"""Memory migration preserves the fighter and every optimizer's learned history."""

import copy
import hashlib

import pytest
import torch

from arma_ai.migrate_memory import EXPANDED, MEMORY_PREFIXES, convert_checkpoint, migrate_file
from arma_ai.model import make_net, net_version


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def old_checkpoint():
    torch.manual_seed(43)
    net = make_net()
    old = [(name, p) for name, p in net.named_parameters() if not name.startswith(MEMORY_PREFIXES)]
    split = len(old) // 2
    opt = torch.optim.AdamW([{"params": [p for _, p in old[:split]], "lr": 1e-4},
                             {"params": [p for _, p in old[split:]], "lr": 2e-4}], amsgrad=True)
    policy = [(name, p) for name, p in old if not name.startswith(("v.", "aux.", "own."))]
    imitation = torch.optim.AdamW([p for _, p in policy], lr=3e-5)
    net.pi.weight.grad = torch.full_like(net.pi.weight, .3)
    net.opponents[0].weight.grad = torch.full_like(net.opponents[0].weight, .2)
    opt.step()
    imitation.step()
    state = {name: value.detach().clone() for name, value in net.state_dict().items()
             if name != "memory_version" and not name.startswith(MEMORY_PREFIXES)}
    return {"trainer": "teach", "model": state, "exploiter": copy.deepcopy(state),
            "opt": opt.state_dict(), "exploiter_opt": copy.deepcopy(opt.state_dict()),
            "update": 11280, "trained": [3895936, 256], "decisions": 1269816, "games": 4072,
            "last_training_method": "imitation", "imitation_state": {"cumulative_steps": 3766,
                "cumulative_samples": 241024, "optimizer": imitation.state_dict()}}


def test_preserves_all_old_tensors_counters_groups_and_adam_moments(old_checkpoint):
    old = old_checkpoint
    torch.manual_seed(99)
    rng = torch.get_rng_state().clone()
    converted, report = convert_checkpoint(old)
    assert torch.equal(rng, torch.get_rng_state())
    assert report["migrated"] and report["new_parameters"] == 990720
    assert report["parameters_after"] == 8155373 and report["initial_residual_zero"]
    assert converted["update"] == old["update"] == 11280
    assert converted["trained"] == old["trained"]
    assert converted["imitation_state"]["cumulative_steps"] == 3766
    for model in ("model", "exploiter"):
        assert net_version(converted[model]) == 4
        for name, value in old[model].items():
            assert torch.equal(converted[model][name], value)
        assert "memory_version" not in old[model]
    pairs = [(old["opt"], converted["opt"]), (old["exploiter_opt"], converted["exploiter_opt"]),
             (old["imitation_state"]["optimizer"], converted["imitation_state"]["optimizer"])]
    for before, after in pairs:
        assert len(before["param_groups"]) == len(after["param_groups"])
        for a, b in zip(before["param_groups"], after["param_groups"]):
            assert {k: v for k, v in a.items() if k != "params"} == {k: v for k, v in b.items() if k != "params"}
            assert b["params"][:len(a["params"])] == a["params"]
        for pid, values in before["state"].items():
            for name, value in values.items():
                assert torch.equal(value, after["state"][pid][name])
        added = set(after["state"]) - set(before["state"])
        assert len(added) == 6
        for pid in added:
            for value in after["state"][pid].values():
                assert torch.count_nonzero(value) == 0


def test_zero_residual_preserves_policy_value_aux_and_maps_exactly(old_checkpoint):
    converted, _ = convert_checkpoint(old_checkpoint)
    net = make_net().eval()
    net.load_state_dict(converted["model"])
    maps = torch.randint(0, 256, (2, 8, 64, 64), dtype=torch.uint8)
    scalars = torch.randn(2, 108)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        h, arena = net.trunk_and_arena(maps, scalars)
        expected = (*net._policy_value(h, mask), net.aux(h),
                    torch.nn.functional.interpolate(net.own(arena), size=(64, 64), mode="bilinear", align_corners=False))
        actual = net.heads_step(maps, scalars, mask, torch.randn(2, 256), context=torch.randn(2, 7))
    for before, after in zip(expected, actual[:4]):
        assert torch.equal(before, after)


def test_adam_continues_existing_steps_and_initializes_new_memory(old_checkpoint):
    converted, _ = convert_checkpoint(old_checkpoint)
    net = make_net()
    net.load_state_dict(converted["model"])
    old_names = [name for name, _ in net.named_parameters() if name in old_checkpoint["model"]]
    old_ids = old_checkpoint["opt"]["param_groups"][0]["params"]
    first = set(old_names[:len(old_ids)])
    optimizer = torch.optim.AdamW([{"params": [p for n, p in net.named_parameters() if n in first]},
                                   {"params": [p for n, p in net.named_parameters() if n not in first]}])
    optimizer.load_state_dict(converted["opt"])
    before = optimizer.state[net.pi.weight]["step"].clone()
    net.pi.weight.grad = torch.ones_like(net.pi.weight)
    net.memory_projection.weight.grad = torch.ones_like(net.memory_projection.weight)
    optimizer.step()
    assert optimizer.state[net.pi.weight]["step"] == before + 1
    assert optimizer.state[net.memory_projection.weight]["step"] == 1
    assert net.memory_projection.weight.abs().sum() > 0


def test_file_conversion_is_fingerprinted_atomic_and_idempotent(tmp_path, old_checkpoint):
    path = tmp_path / "latest.pt"
    torch.save(old_checkpoint, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="changed during verification"):
        migrate_file(path, "wrong")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert migrate_file(path, before)["migrated"]
    after = path.read_bytes()
    assert not migrate_file(path)["migrated"] and path.read_bytes() == after
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["latest.pt"]


def test_bad_optimizer_or_partial_source_cannot_replace_checkpoint(tmp_path, old_checkpoint):
    ck = copy.deepcopy(old_checkpoint)
    pid = next(iter(ck["opt"]["state"]))
    ck["opt"]["state"][pid]["exp_avg"] = torch.zeros(2, 2)
    path = tmp_path / "latest.pt"
    torch.save(ck, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible shape"):
        migrate_file(path)
    assert path.read_bytes() == before
    ck = copy.deepcopy(old_checkpoint)
    ck["model"]["memory_version"] = torch.tensor(1)
    with pytest.raises(ValueError, match="partial"):
        convert_checkpoint(ck)


def test_intermediate_768_input_memory_expands_moments_with_zeros(old_checkpoint):
    converted, _ = convert_checkpoint(old_checkpoint)
    net = make_net()
    names = [name for name, _ in net.named_parameters()]
    intermediate = copy.deepcopy(converted)
    for key in ("model", "exploiter"):
        intermediate[key][EXPANDED] = intermediate[key][EXPANDED][:, :768].clone()
    for key in ("opt", "exploiter_opt"):
        ids = [pid for group in intermediate[key]["param_groups"] for pid in group["params"]]
        pid = ids[names.index(EXPANDED)]
        for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            intermediate[key]["state"][pid][field] = torch.ones(768, 768)
    policy_names = [name for name in names if not name.startswith(("v.", "aux.", "own."))]
    opt = intermediate["imitation_state"]["optimizer"]
    pid = opt["param_groups"][0]["params"][policy_names.index(EXPANDED)]
    for field in ("exp_avg", "exp_avg_sq"):
        opt["state"][pid][field] = torch.ones(768, 768)
    actual, report = convert_checkpoint(intermediate)
    assert report["migrated"] and report["new_parameters"] == 5376
    assert torch.equal(actual["model"][EXPANDED][:, :768], intermediate["model"][EXPANDED])
    assert torch.count_nonzero(actual["model"][EXPANDED][:, 768:]) == 0
    moments = actual["imitation_state"]["optimizer"]["state"][pid]
    assert torch.equal(moments["exp_avg"][:, :768], torch.ones(768, 768))
    assert torch.count_nonzero(moments["exp_avg"][:, 768:]) == 0
