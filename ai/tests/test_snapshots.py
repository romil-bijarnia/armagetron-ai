"""RL keeps dated copies of the checkpoint to fall back on; latest.pt alone is resumed."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from arma_ai.teach import Config, Learner


def config(run: Path, **overrides) -> Config:
    return Config(run=str(run), device="cpu", bf16=False, games=2, threads=2, ring=8192, exploiter_ring=8192,
                  warmup=16, batch=2, sequence_burn_in=1, sequence_unroll=2, **overrides)


def close(learner: Learner) -> None:
    learner.cleanup_runtime_models()
    for ring in learner.rings:
        ring.close(unlink=True)
    learner.log.close()
    learner._lock.close()


def snapshots(run: Path) -> list[str]:
    folder = run / "snapshots"
    return sorted(path.name for path in folder.iterdir()) if folder.exists() else []


def test_snapshots_copy_the_checkpoint_once_per_interval(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with TemporaryDirectory(dir=tmp_path) as directory:
            run = Path(directory)
            learner = Learner(config(run, snapshot_every=100))
            try:
                learner.save()  # update 0 has not crossed the first interval
                assert snapshots(run) == []
                learner.update = 137
                learner.save()
                learner.update = 180
                learner.save()  # the same interval: no second copy
                assert snapshots(run) == ["update-00000137.pt"]
                copy = torch.load(run / "snapshots" / "update-00000137.pt", weights_only=True)
                assert copy["trainer"] == "teach" and copy["update"] == 137
                learner.update = 300
                learner.save()
                assert snapshots(run) == ["update-00000137.pt", "update-00000300.pt"]
            finally:
                close(learner)
            assert sorted(path.name for path in run.glob("*.pt")) == ["latest.pt"]
            # Resuming counts from the saved update: the interval it is in is not copied again.
            learner = Learner(config(run, snapshot_every=100))
            try:
                assert learner.update == 300
                learner.save()
                assert snapshots(run) == ["update-00000137.pt", "update-00000300.pt"]
                learner.update = 401
                learner.save()
                assert snapshots(run) == ["update-00000137.pt", "update-00000300.pt", "update-00000401.pt"]
            finally:
                close(learner)
    finally:
        torch.set_num_threads(old_threads)


def test_snapshots_can_be_switched_off_but_not_made_negative(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with TemporaryDirectory(dir=tmp_path) as directory:
            run = Path(directory)
            learner = Learner(config(run, snapshot_every=0))
            try:
                learner.update = 5000
                learner.save()
                assert snapshots(run) == [] and (run / "latest.pt").exists()
            finally:
                close(learner)
        with pytest.raises(ValueError):
            Learner(config(tmp_path / "negative", snapshot_every=-1))
    finally:
        torch.set_num_threads(old_threads)
