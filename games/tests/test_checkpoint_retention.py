"""Tests for run-owned checkpoint retention and disk accounting.

The fixtures use the files ``transformers.Trainer`` writes, but tiny payloads so the tests stay
CPU-only.  The pruning tests deliberately provide a loader callback: pruning is allowed only after
the retained checkpoints have been read by the caller's real loader and readout artifacts exist.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - fixtures construct runtime paths
from typing import TYPE_CHECKING, cast

import pytest

from games.checkpoint_retention import (
    ADAPTER_WEIGHT_FILENAMES,
    REQUIRED_CHECKPOINT_FILES,
    CheckpointRetentionCallback,
    checkpoint_disk_usage,
    forecast_checkpoint_disk_usage,
    inspect_checkpoints,
    missing_checkpoint_files,
    prune_redundant_checkpoints,
    validate_retention_settings,
    write_retention_manifest,
)

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


def write_checkpoint(
    run_root: Path, step: int, *, omit: tuple[str, ...] = (), payload: bytes = b"state"
) -> Path:
    """Create a small checkpoint with the same state-file names as a real Trainer save."""
    checkpoint = run_root / f"checkpoint-{step}"
    checkpoint.mkdir()
    for filename in (*REQUIRED_CHECKPOINT_FILES, ADAPTER_WEIGHT_FILENAMES[0]):
        if filename in omit:
            continue
        contents = (
            json.dumps({"global_step": step}).encode()
            if filename == "trainer_state.json"
            else json.dumps({"peft_type": "LORA"}).encode()
            if filename == "adapter_config.json"
            else payload
        )
        (checkpoint / filename).write_bytes(contents)
    return checkpoint


class TestCheckpointInventory:
    def test_inventory_records_relative_paths_steps_sizes_and_hashes(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(tmp_path, 3, payload=b"three")

        snapshots = inspect_checkpoints(tmp_path)

        assert [snapshot.relative_path for snapshot in snapshots] == ["checkpoint-3"]
        snapshot = snapshots[0]
        assert snapshot.step == 3
        assert snapshot.complete is True
        assert snapshot.missing == ()
        assert snapshot.total_bytes == sum(path.stat().st_size for path in checkpoint.iterdir())
        assert {entry.relative_path for entry in snapshot.files} == {
            f"checkpoint-3/{path.name}" for path in checkpoint.iterdir()
        }
        assert all(len(entry.sha256) == 64 for entry in snapshot.files)


class TestExplicitRetentionPolicy:
    def test_per_step_no_rotation_policy_is_accepted(self) -> None:
        validate_retention_settings(max_steps=20, save_steps=1, save_total_limit=0)

    def test_coarser_cadence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="save_steps=1"):
            validate_retention_settings(max_steps=20, save_steps=5, save_total_limit=0)

    def test_rotation_that_would_delete_a_checkpoint_is_refused(self) -> None:
        with pytest.raises(ValueError, match="oldest of 20 checkpoints"):
            validate_retention_settings(max_steps=20, save_steps=1, save_total_limit=10)

    def test_negative_rotation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            validate_retention_settings(max_steps=20, save_steps=1, save_total_limit=-1)

    def test_train_config_requires_the_policy_when_manifest_recording_is_enabled(
        self, tmp_path: Path
    ) -> None:
        from games.train import GameTrainConfig  # noqa: PLC0415

        with pytest.raises(ValueError, match="save_steps=1"):
            GameTrainConfig(
                arm="twin-pd-group",
                generate_fresh=True,
                output_dir=str(tmp_path),
                max_steps=2,
                record_retention_manifest=True,
                save_steps=2,
                save_total_limit=0,
            )

    def test_smoke_manifest_recording_uses_per_step_no_rotation(self, tmp_path: Path) -> None:
        from games.train import _parse_args  # noqa: PLC0415

        config = _parse_args(
            [
                "--arm",
                "twin-pd-group",
                "--smoke",
                "--output-dir",
                str(tmp_path),
                "--record-retention-manifest",
            ]
        )

        assert config.save_steps == 1
        assert config.save_total_limit == 0

    def test_inventory_marks_an_incomplete_newest_checkpoint_without_guessing(
        self, tmp_path: Path
    ) -> None:
        write_checkpoint(tmp_path, 2)
        write_checkpoint(tmp_path, 3, omit=("optimizer.pt",))

        snapshots = inspect_checkpoints(tmp_path)

        newest = snapshots[-1]
        assert newest.relative_path == "checkpoint-3"
        assert newest.complete is False
        assert newest.missing == ("optimizer.pt",)

    def test_nested_state_file_does_not_satisfy_a_direct_checkpoint_requirement(
        self, tmp_path: Path
    ) -> None:
        checkpoint = write_checkpoint(tmp_path, 3, omit=("optimizer.pt",))
        nested = checkpoint / "nested"
        nested.mkdir()
        (nested / "optimizer.pt").write_bytes(b"bogus")

        assert missing_checkpoint_files(checkpoint) == ["optimizer.pt"]

    def test_adapter_config_is_required_for_a_resumable_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(tmp_path, 3, omit=("adapter_config.json",))

        assert missing_checkpoint_files(checkpoint) == ["adapter_config.json"]

    def test_empty_optimizer_state_is_not_a_complete_resume(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(tmp_path, 3)
        (checkpoint / "optimizer.pt").write_bytes(b"")

        assert missing_checkpoint_files(checkpoint) == ["optimizer.pt"]

    def test_trainer_step_mismatch_is_not_a_complete_resume(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(tmp_path, 3)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 2}), encoding="utf-8"
        )

        assert missing_checkpoint_files(checkpoint) == ["trainer_state.json (step mismatch)"]

    def test_invalid_adapter_config_is_not_a_complete_resume(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(tmp_path, 3)
        (checkpoint / "adapter_config.json").write_text("not-json", encoding="utf-8")

        assert missing_checkpoint_files(checkpoint) == ["adapter_config.json (invalid JSON)"]

    def test_disk_usage_counts_only_run_owned_checkpoint_directories(self, tmp_path: Path) -> None:
        first = write_checkpoint(tmp_path, 1, payload=b"first")
        second = write_checkpoint(tmp_path, 2, payload=b"second")
        (tmp_path / "shared-base-cache").mkdir()
        (tmp_path / "shared-base-cache" / "weights.bin").write_bytes(b"do not count")

        expected = sum(path.stat().st_size for path in (*first.iterdir(), *second.iterdir()))

        assert checkpoint_disk_usage(tmp_path) == expected


class TestCheckpointDiskForecast:
    def test_forecast_uses_the_largest_observed_complete_checkpoint(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1, payload=b"one")
        write_checkpoint(tmp_path, 2, payload=b"two" * 10)

        forecast = forecast_checkpoint_disk_usage(tmp_path, total_checkpoints=4)

        assert forecast.observed_complete_bytes > 0
        assert forecast.bytes_per_checkpoint == forecast.observed_complete_bytes
        assert forecast.projected_checkpoint_bytes == 4 * forecast.observed_complete_bytes

    def test_forecast_requires_an_observed_checkpoint_or_explicit_size(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="no complete checkpoint"):
            forecast_checkpoint_disk_usage(tmp_path, total_checkpoints=2)

        forecast = forecast_checkpoint_disk_usage(
            tmp_path, total_checkpoints=2, bytes_per_checkpoint=123
        )
        assert forecast.bytes_per_checkpoint == 123
        assert forecast.projected_checkpoint_bytes == 246


class TestRetentionManifestRecording:
    def test_manifest_records_complete_and_incomplete_checkpoints_without_pruning(
        self, tmp_path: Path
    ) -> None:
        write_checkpoint(tmp_path, 1)
        write_checkpoint(tmp_path, 2, omit=("rng_state.pth",))

        manifest = write_retention_manifest(tmp_path)

        assert [snapshot.step for snapshot in manifest.retained] == [1]
        assert [snapshot.step for snapshot in manifest.incomplete] == [2]
        assert manifest.pruned == ()
        assert (tmp_path / "checkpoint-1").is_dir()
        assert (tmp_path / "checkpoint-2").is_dir()

    def test_trainer_callback_writes_the_run_owned_manifest_after_save(
        self, tmp_path: Path
    ) -> None:
        write_checkpoint(tmp_path, 1)

        CheckpointRetentionCallback(tmp_path).on_save(
            cast("TrainingArguments", None),
            cast("TrainerState", None),
            cast("TrainerControl", None),
        )

        written = json.loads((tmp_path / "retention_manifest.json").read_text(encoding="utf-8"))
        assert written["status"] == "complete"
        assert written["retained"][0]["step"] == 1


class TestCheckpointPruning:
    def test_empty_retention_is_refused_even_when_there_is_only_one_checkpoint(
        self, tmp_path: Path
    ) -> None:
        write_checkpoint(tmp_path, 1)
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="retain at least one"):
            prune_redundant_checkpoints(
                tmp_path,
                retain_steps=(),
                durable_readouts=(readout,),
                load_checkpoint=lambda _path: None,
                validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
            )

        assert (tmp_path / "checkpoint-1").is_dir()

    def test_the_sole_complete_resume_is_preserved(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1)
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")

        manifest = prune_redundant_checkpoints(
            tmp_path,
            retain_steps=(1,),
            durable_readouts=(readout,),
            load_checkpoint=lambda _path: None,
            validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
        )

        assert (tmp_path / "checkpoint-1").is_dir()
        assert manifest.pruned == ()

    def test_pruning_keeps_two_complete_resumes_and_records_hashes(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1, payload=b"early")
        prior = write_checkpoint(tmp_path, 2, payload=b"prior")
        final = write_checkpoint(tmp_path, 3, payload=b"final")
        readout = tmp_path / "readout" / "final.json"
        readout.parent.mkdir()
        readout.write_text("{}", encoding="utf-8")
        loader_calls: list[Path] = []

        def record_loaded(path: Path) -> None:
            loader_calls.append(path)

        manifest = prune_redundant_checkpoints(
            tmp_path,
            retain_steps=(2, 3),
            durable_readouts=(readout,),
            load_checkpoint=record_loaded,
            validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
        )

        assert not (tmp_path / "checkpoint-1").exists()
        assert prior.is_dir()
        assert final.is_dir()
        assert loader_calls == [prior, final]
        assert [entry.relative_path for entry in manifest.retained] == [
            "checkpoint-2",
            "checkpoint-3",
        ]
        assert [entry.relative_path for entry in manifest.pruned] == ["checkpoint-1"]
        assert manifest.pruned[0].step == 1
        assert all(len(file.sha256) == 64 for file in manifest.pruned[0].files)
        written = json.loads((tmp_path / "retention_manifest.json").read_text(encoding="utf-8"))
        assert written["status"] == "complete"
        assert written["retained"][1]["relative_path"] == "checkpoint-3"

    def test_pruning_refuses_to_leave_only_one_complete_resume(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1)
        write_checkpoint(tmp_path, 2)
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")
        before = sorted(path.name for path in tmp_path.iterdir())

        with pytest.raises(ValueError, match="two newest complete checkpoints"):
            prune_redundant_checkpoints(
                tmp_path,
                retain_steps=(2,),
                durable_readouts=(readout,),
                load_checkpoint=lambda _path: None,
                validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
            )

        assert sorted(path.name for path in tmp_path.iterdir()) == before

    def test_pruning_refuses_without_durable_readout(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1)
        write_checkpoint(tmp_path, 2)
        write_checkpoint(tmp_path, 3)

        with pytest.raises(ValueError, match="readout"):
            prune_redundant_checkpoints(
                tmp_path,
                retain_steps=(2, 3),
                durable_readouts=(tmp_path / "missing.json",),
                load_checkpoint=lambda _path: None,
                validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
            )

        assert (tmp_path / "checkpoint-1").is_dir()

    def test_pruning_refuses_an_incomplete_retained_checkpoint(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1)
        write_checkpoint(tmp_path, 2, omit=("scheduler.pt",))
        write_checkpoint(tmp_path, 3)
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match=r"scheduler[.]pt"):
            prune_redundant_checkpoints(
                tmp_path,
                retain_steps=(2, 3),
                durable_readouts=(readout,),
                load_checkpoint=lambda _path: None,
                validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
            )

        assert (tmp_path / "checkpoint-1").is_dir()

    def test_pruning_does_not_touch_a_sibling_cache_or_another_run(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1)
        write_checkpoint(tmp_path, 2)
        write_checkpoint(tmp_path, 3)
        other_run = tmp_path.parent / "other-run"
        other_run.mkdir()
        other_checkpoint = write_checkpoint(other_run, 1)
        cache = tmp_path / "shared-base-cache"
        cache.mkdir()
        cache_file = cache / "weights.bin"
        cache_file.write_bytes(b"keep")
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")

        prune_redundant_checkpoints(
            tmp_path,
            retain_steps=(2, 3),
            durable_readouts=(readout,),
            load_checkpoint=lambda _path: None,
            validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
        )

        assert other_checkpoint.is_dir()
        assert cache_file.read_bytes() == b"keep"

    def test_pruning_keeps_an_early_milestone_as_adapter_only(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1, payload=b"early")
        write_checkpoint(tmp_path, 2, payload=b"prior")
        write_checkpoint(tmp_path, 3, payload=b"final")
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")
        loaded: list[Path] = []

        manifest = prune_redundant_checkpoints(
            tmp_path,
            retain_steps=(2, 3),
            adapter_steps=(1,),
            durable_readouts=(readout,),
            load_checkpoint=loaded.append,
            validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
        )

        assert loaded == [
            tmp_path / "checkpoint-2",
            tmp_path / "checkpoint-3",
            tmp_path / "checkpoint-1",
        ]
        adapter_dir = tmp_path / "adapters" / "checkpoint-1"
        assert not (tmp_path / "checkpoint-1").exists()
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "adapter_model.safetensors").is_file()
        assert [entry.relative_path for entry in manifest.adapter_retained] == [
            "adapters/checkpoint-1"
        ]

    def test_pruning_sabotage_loader_failure_leaves_all_checkpoints(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 1)
        write_checkpoint(tmp_path, 2)
        write_checkpoint(tmp_path, 3)
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")

        def loader(path: Path) -> None:
            if path.name == "checkpoint-3":
                raise ValueError("simulated load failure")

        with pytest.raises(ValueError, match="simulated load failure"):
            prune_redundant_checkpoints(
                tmp_path,
                retain_steps=(2, 3),
                durable_readouts=(readout,),
                load_checkpoint=loader,
                validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
            )

        assert all((tmp_path / f"checkpoint-{step}").is_dir() for step in (1, 2, 3))

    def test_pruning_refuses_when_a_retained_checkpoint_changes_during_load(
        self, tmp_path: Path
    ) -> None:
        write_checkpoint(tmp_path, 1)
        write_checkpoint(tmp_path, 2)
        write_checkpoint(tmp_path, 3)
        readout = tmp_path / "readout.json"
        readout.write_text("{}", encoding="utf-8")

        def mutating_loader(path: Path) -> None:
            if path.name == "checkpoint-2":
                (path / "optimizer.pt").write_bytes(b"changed")

        with pytest.raises(RuntimeError, match="changed while it was being loaded"):
            prune_redundant_checkpoints(
                tmp_path,
                retain_steps=(2, 3),
                durable_readouts=(readout,),
                load_checkpoint=mutating_loader,
                validate_readout=lambda path: json.loads(path.read_text(encoding="utf-8")),
            )

        assert all((tmp_path / f"checkpoint-{step}").is_dir() for step in (1, 2, 3))
