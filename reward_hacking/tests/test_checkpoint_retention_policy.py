"""Conservative checkpoint retention and pre-write disk sabotage controls."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments

from reward_hacking.checkpoint_retention import (
    CheckpointDiskSpaceError,
    CheckpointRetentionMixin,
    checkpoint_is_complete,
    main,
    prune_checkpoints,
)


def checkpoint(root: Path, step: int, *, epoch: float = 0.5) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "epoch": epoch}))
    (path / "adapter_config.json").write_text("{}")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (path / name).write_bytes(b"state")
    return path


def test_running_retains_latest_epochs_pins_and_middle(tmp_path: Path) -> None:
    for step in range(1, 21):
        checkpoint(tmp_path, step, epoch=step / 5)
    (tmp_path / "checkpoint-1" / ".keep").touch()
    torn = checkpoint(tmp_path, 21)
    (torn / "optimizer.pt").unlink()
    unrelated = tmp_path / "checkpoint-garbage"
    unrelated.mkdir()
    dry = prune_checkpoints(tmp_path, dry_run=True)
    assert len(list(tmp_path.glob("checkpoint-*"))) == 22
    deleted = prune_checkpoints(tmp_path)
    assert deleted == dry
    kept = {path.name for path in tmp_path.iterdir()}
    assert {
        "checkpoint-1",
        "checkpoint-5",
        "checkpoint-10",
        "checkpoint-15",
        "checkpoint-19",
        "checkpoint-20",
        "checkpoint-21",
        unrelated.name,
    } <= kept
    assert len(kept) <= 11


def test_finished_retains_final_three_spaced_epochs_and_pins(tmp_path: Path) -> None:
    for step in range(1, 21):
        checkpoint(tmp_path, step, epoch=step / 5)
    (tmp_path / "checkpoint-2" / ".keep").touch()
    prune_checkpoints(tmp_path, finished=True)
    assert {path.name for path in tmp_path.iterdir()} == {
        f"checkpoint-{step}" for step in (2, 5, 10, 15, 20)
    }


@pytest.mark.parametrize(
    "missing",
    [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
    ],
)
def test_torn_checkpoint_never_deleted(tmp_path: Path, missing: str) -> None:
    torn = checkpoint(tmp_path, 1)
    (torn / missing).unlink()
    for step in range(2, 10):
        checkpoint(tmp_path, step)
    assert not checkpoint_is_complete(torn)
    prune_checkpoints(tmp_path, finished=True)
    assert torn.is_dir()


def test_full_model_and_sharded_model_completeness(tmp_path: Path) -> None:
    path = checkpoint(tmp_path, 1)
    (path / "adapter_config.json").unlink()
    (path / "adapter_model.safetensors").unlink()
    (path / "config.json").write_text("{}")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001.safetensors"}})
    )
    assert not checkpoint_is_complete(path)
    (path / "model-00001.safetensors").write_bytes(b"weight")
    assert checkpoint_is_complete(path)


@pytest.mark.parametrize("location", ["root", "checkpoint", "inside"])
def test_symlink_sabotage_refuses_before_any_deletion(tmp_path: Path, location: str) -> None:
    root = tmp_path / "run"
    root.mkdir()
    for step in range(1, 9):
        checkpoint(root, step)
    if location == "root":
        target = tmp_path / "alias"
        target.symlink_to(root, target_is_directory=True)
    elif location == "checkpoint":
        (root / "checkpoint-9").symlink_to(root / "checkpoint-1", target_is_directory=True)
        target = root
    else:
        (root / "checkpoint-1" / "unexpected").symlink_to(tmp_path)
        target = root
    with pytest.raises(ValueError, match="symlink"):
        prune_checkpoints(target, finished=True)
    assert all((root / f"checkpoint-{step}").is_dir() for step in range(1, 9))


def test_mismatched_state_and_invalid_limits_refuse(tmp_path: Path) -> None:
    path = checkpoint(tmp_path, 1)
    (path / "trainer_state.json").write_text('{"global_step": 99, "epoch": 1}')
    assert not checkpoint_is_complete(path)
    with pytest.raises(ValueError, match="latest"):
        prune_checkpoints(tmp_path, keep_latest=1)


class SavingTrainer:
    def __init__(self, root: Path) -> None:
        self.args = SimpleNamespace(
            output_dir=str(root),
            num_train_epochs=4,
            save_total_limit=1,
            save_only_model=False,
            should_save=True,
            max_steps=-1,
            save_strategy="steps",
        )
        self.state = SimpleNamespace(global_step=0, max_steps=10, epoch=0.0)
        self.model = torch.nn.Linear(2, 2)
        self.saved = False

    def _get_output_dir(self, trial: Any) -> str:
        del trial
        return self.args.output_dir

    def _save_checkpoint(self, model: Any, trial: Any) -> None:
        del model, trial
        checkpoint(Path(self.args.output_dir), self.state.global_step)
        self.saved = True

    def train(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "result"


class RetainingTrainer(CheckpointRetentionMixin, SavingTrainer):
    pass


def test_disk_sabotage_stops_before_save_and_preserves_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = checkpoint(tmp_path, 1)
    trainer = RetainingTrainer(tmp_path)
    trainer.state.global_step = 2
    monkeypatch.setattr(
        "reward_hacking.checkpoint_retention.shutil.disk_usage", lambda _: SimpleNamespace(free=0)
    )
    with pytest.raises(CheckpointDiskSpaceError, match="resume"):
        trainer._save_checkpoint(trainer.model, None)
    assert not trainer.saved
    assert resume.is_dir()
    assert not (tmp_path / "checkpoint-2").exists()


def test_incomplete_training_does_not_apply_finished_policy(tmp_path: Path) -> None:
    for step in range(1, 9):
        checkpoint(tmp_path, step)
    trainer = RetainingTrainer(tmp_path)
    trainer.state.global_step = 8
    assert trainer.train() == "result"
    assert len(list(tmp_path.iterdir())) == 8
    trainer.state.global_step = 10
    checkpoint(tmp_path, 10)
    assert trainer.train() == "result"
    assert len(list(tmp_path.iterdir())) == 4
    assert trainer.args.save_total_limit is None


def test_newer_torn_never_counts_toward_two_resumes(tmp_path: Path) -> None:
    for step in range(1, 7):
        checkpoint(tmp_path, step)
    for step in range(7, 10):
        torn = checkpoint(tmp_path, step)
        (torn / "optimizer.pt").unlink()
    prune_checkpoints(tmp_path, keep_middle=0)
    assert {path.name for path in tmp_path.iterdir()} == {
        f"checkpoint-{step}" for step in range(5, 10)
    }


def test_nested_pin_and_custom_torn_preserved(tmp_path: Path) -> None:
    for step in range(1, 10):
        checkpoint(tmp_path, step)
    pin = tmp_path / "checkpoint-1" / "nested"
    pin.mkdir()
    (pin / ".keep").touch()
    prune_checkpoints(
        tmp_path, finished=True, keep_middle=0, is_complete=lambda path: path.name != "checkpoint-2"
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "checkpoint-1",
        "checkpoint-2",
        "checkpoint-9",
    }


def test_projection_sabotage_above_disk_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = RetainingTrainer(tmp_path)
    monkeypatch.setattr("reward_hacking.checkpoint_retention.minimum_free_disk_bytes", lambda: 100)
    monkeypatch.setattr(
        "reward_hacking.checkpoint_retention.shutil.disk_usage", lambda _: SimpleNamespace(free=101)
    )
    with pytest.raises(CheckpointDiskSpaceError):
        trainer.train()
    assert not list(tmp_path.iterdir())


def test_measured_checkpoint_expands_disk_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = RetainingTrainer(tmp_path)
    saved = checkpoint(tmp_path, 1)
    (saved / "optimizer.pt").write_bytes(b"0" * (2 * 1024**2))
    monkeypatch.setattr("reward_hacking.checkpoint_retention.minimum_free_disk_bytes", lambda: 0)
    monkeypatch.setattr(
        "reward_hacking.checkpoint_retention.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2 * 1024**2),
    )
    with pytest.raises(CheckpointDiskSpaceError):
        trainer.check_checkpoint_disk_space()
    assert saved.is_dir()


def test_cli_finished_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for step in range(1, 10):
        checkpoint(tmp_path, step)
    monkeypatch.setattr("sys.argv", ["checkpoint_retention", "prune", str(tmp_path), "--dry-run"])
    with caplog.at_level("INFO"):
        main()
    assert "would remove" in caplog.text
    assert len(list(tmp_path.iterdir())) == 9


class InterruptedTrainingError(RuntimeError):
    """Deliberate interruption after a fully persisted checkpoint."""


class InterruptAfterSave(TrainerCallback):
    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        if state.global_step == 2:
            raise InterruptedTrainingError("interrupt after complete checkpoint")


class TinyRetainingTrainer(CheckpointRetentionMixin, Trainer):
    pass


def test_real_hf_checkpoint_interruption_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("reward_hacking.checkpoint_retention.minimum_free_disk_bytes", lambda: 0)
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=12, n_positions=8, n_embd=8, n_layer=1, n_head=1))
    rows = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 4, "labels": [[1, 2, 3, 4]] * 4})
    arguments = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=8,
        per_device_train_batch_size=1,
        save_steps=1,
        save_total_limit=1,
        report_to="none",
        disable_tqdm=True,
    )
    interrupted = TinyRetainingTrainer(
        model=model, args=arguments, train_dataset=rows, callbacks=[InterruptAfterSave()]
    )
    with pytest.raises(InterruptedTrainingError):
        interrupted.train()
    assert {path.name for path in tmp_path.glob("checkpoint-*")} == {"checkpoint-1", "checkpoint-2"}
    assert checkpoint_is_complete(tmp_path / "checkpoint-2")
    resumed = TinyRetainingTrainer(model=model, args=arguments, train_dataset=rows)
    resumed.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    checkpoints = sorted(tmp_path.glob("checkpoint-*"))
    assert len(checkpoints) == 4
    assert checkpoint_is_complete(tmp_path / "checkpoint-8")
    state = json.loads((tmp_path / "checkpoint-8" / "trainer_state.json").read_text())
    assert state["global_step"] == 8
    optimizer = torch.load(tmp_path / "checkpoint-8" / "optimizer.pt", weights_only=True)
    assert {int(entry["step"]) for entry in optimizer["state"].values()} == {8}


def test_sequential_saves_retain_epoch_boundaries_and_final_middles(tmp_path: Path) -> None:
    for step in range(1, 21):
        checkpoint(tmp_path, step, epoch=step / 5)
        prune_checkpoints(tmp_path)
        assert {f"checkpoint-{boundary}" for boundary in range(5, step + 1, 5)} <= {
            path.name for path in tmp_path.iterdir()
        }
        assert len(list(tmp_path.iterdir())) <= 2 + 3 + step // 5
    prune_checkpoints(tmp_path, finished=True)
    assert {path.name for path in tmp_path.iterdir()} == {
        f"checkpoint-{step}" for step in (5, 10, 15, 20)
    }


@pytest.mark.parametrize("extra", ["complete.json", "distill_state.json"])
def test_cli_inventory_preserves_special_torn_checkpoint(tmp_path: Path, extra: str) -> None:
    for step in range(1, 10):
        path = checkpoint(tmp_path, step)
        (path / extra).write_text("{}")
        if extra == "complete.json":
            (path / "rollout_records.json").write_text("[]")
    (tmp_path / "checkpoint-1" / extra).unlink()
    prune_checkpoints(tmp_path, finished=True, keep_middle=0)
    assert {path.name for path in tmp_path.iterdir()} == {"checkpoint-1", "checkpoint-9"}


def test_symlink_shard_and_traversal_index_refuse(tmp_path: Path) -> None:
    path = checkpoint(tmp_path, 1)
    (path / "adapter_config.json").unlink()
    (path / "adapter_model.safetensors").unlink()
    (path / "config.json").write_text("{}")
    (path / "model.safetensors.index.json").write_text('{"weight_map":{"x":"../outside"}}')
    with pytest.raises(ValueError, match="outside"):
        checkpoint_is_complete(path)


def test_failed_train_keeps_running_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for step in range(1, 10):
        checkpoint(tmp_path, step)
    trainer = RetainingTrainer(tmp_path)
    trainer.state.global_step = 10
    monkeypatch.setattr("reward_hacking.checkpoint_retention.minimum_free_disk_bytes", lambda: 0)

    def fail_training(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise InterruptedTrainingError("failed training")

    monkeypatch.setattr(SavingTrainer, "train", fail_training)
    with pytest.raises(InterruptedTrainingError):
        trainer.train()
    assert len(list(tmp_path.iterdir())) == 9


def test_max_steps_epoch_projection_exceeds_declared_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = RetainingTrainer(tmp_path)
    trainer.args.max_steps = 20
    trainer.args.gradient_accumulation_steps = 1
    monkeypatch.setattr(RetainingTrainer, "get_train_dataloader", lambda _: [1], raising=False)
    monkeypatch.setattr("reward_hacking.checkpoint_retention.minimum_free_disk_bytes", lambda: 0)
    monkeypatch.setattr(
        "reward_hacking.checkpoint_retention.shutil.disk_usage",
        lambda _: SimpleNamespace(free=10 * 1024**2),
    )
    with pytest.raises(CheckpointDiskSpaceError, match="projected_checkpoint_count=26"):
        trainer.check_checkpoint_disk_space(preflight=True)


def test_full_model_estimate_includes_frozen_weights(tmp_path: Path) -> None:
    trainer = RetainingTrainer(tmp_path)
    trainer.model = torch.nn.Sequential(torch.nn.Embedding(1_000_000, 1), torch.nn.Linear(1, 1))
    trainer.model[0].weight.requires_grad_(False)
    assert trainer._estimated_checkpoint_bytes() > 5 * 1024**2


def test_online_middle_reservoir_uses_planned_final_step(tmp_path: Path) -> None:
    for step in range(1, 57):
        checkpoint(tmp_path, step)
        prune_checkpoints(tmp_path, planned_final_step=56)
        assert len(list(tmp_path.iterdir())) <= 5
    prune_checkpoints(tmp_path, finished=True)
    assert {path.name for path in tmp_path.iterdir()} == {
        f"checkpoint-{step}" for step in (14, 28, 42, 56)
    }
