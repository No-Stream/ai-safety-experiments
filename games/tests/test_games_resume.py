"""Pin resume-from-checkpoint, the thing that stops a 30-hour cloud run being hostage to one kill.

Offline: no trainer is constructed and no weights are loaded. What is tested is the resolution
step -- turning the `--resume-from-checkpoint` setting into the argument
`transformers.Trainer.train` receives -- plus the guard against the one combination that would fail
silently.

What resuming *restores* is inherited rather than implemented here, and was read out of the
installed source rather than assumed: `GRPOTrainer` does not override `train()` and contains no
reference to `resume_from_checkpoint`, so HF's path runs, which reloads the LoRA adapter via
`model.load_adapter(..., is_trainable=True)` (`transformers/trainer.py:3440`), the optimizer and
scheduler, the RNG states, `global_step` from `trainer_state.json`, and fast-forwards the dataloader
with `skip_first_batches`. Those are transformers' own behaviours and are not re-tested here.

:class:`TestTheSilentFailureIsRefused` is the class with teeth. Asking for the newest checkpoint
without naming the run directory cannot work -- the directory is timestamped fresh each launch --
and the failure mode is a restart that quietly begins at step 0 and reports success, which is worse
than a crash because the resulting trace looks like a complete run.

Two things live next door rather than here. Whether the resolved checkpoint actually reaches
`Trainer.train` used to be checked here by searching `games/train.py` for the call, because
constructing a trainer needed a GPU; `games/tests/test_games_train_wiring.py` now runs the real
`train_game_arm` against a stubbed trainer and asserts what `train()` was asked for, which a text
search cannot distinguish from a rename. The same file holds the checks that a resume is refused
when it disagrees with the launch its checkpoint came from.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from games.train import (
    ADAPTER_WEIGHT_FILENAMES,
    INCOMPLETE_CHECKPOINTS_DIRNAME,
    REQUIRED_CHECKPOINT_FILES,
    RESUME_LATEST,
    TRAINER_STATE_FILENAME,
    GameTrainConfig,
    ResumeResolution,
    assert_resume_is_addressable,
    missing_checkpoint_files,
    resolve_complete_resume_checkpoint,
    resolve_resume_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

ARM = "twin-pd-group"
STAMP = datetime(2026, 9, 2, 18, 15, 0, tzinfo=UTC)
COMPLETE_FILES = (*REQUIRED_CHECKPOINT_FILES, ADAPTER_WEIGHT_FILENAMES[0])


def make_checkpoint(run_dir: Path, step: int, *, with_state: bool = True) -> Path:
    """Lay down a checkpoint directory shaped like one Trainer.save_state would write."""
    checkpoint = run_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    if with_state:
        (checkpoint / TRAINER_STATE_FILENAME).write_text(json.dumps({"global_step": step}))
    return checkpoint


def write_checkpoint(run_dir: Path, step: int, *, omit: tuple[str, ...] = ()) -> Path:
    """Lay down `checkpoint-<step>` with every file a resume needs, minus the ones named in `omit`.

    Empty files, because the gate reads names and nothing else.
    """
    checkpoint = run_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in COMPLETE_FILES:
        if name not in omit:
            (checkpoint / name).write_bytes(b"")
    return checkpoint


def resolve_latest(run_dir: Path) -> ResumeResolution:
    return resolve_complete_resume_checkpoint(RESUME_LATEST, str(run_dir), now=STAMP)


class TestResolvingTheLatestCheckpoint:
    def test_it_picks_the_highest_step_not_the_lexically_last(self, tmp_path: Path) -> None:
        """checkpoint-100 sorts before checkpoint-20 as text, which would resume 80 steps early."""
        for step in (10, 20, 100):
            make_checkpoint(tmp_path, step)
        resolved = resolve_resume_checkpoint(RESUME_LATEST, str(tmp_path))
        assert resolved is not None
        assert resolved.endswith("checkpoint-100")

    def test_an_empty_run_directory_starts_fresh_rather_than_failing(self, tmp_path: Path) -> None:
        """This is what makes "re-run the same command" right on both first launch and restart."""
        assert resolve_resume_checkpoint(RESUME_LATEST, str(tmp_path)) is None

    def test_a_missing_run_directory_starts_fresh(self, tmp_path: Path) -> None:
        assert resolve_resume_checkpoint(RESUME_LATEST, str(tmp_path / "not-yet")) is None

    def test_non_checkpoint_entries_do_not_confuse_it(self, tmp_path: Path) -> None:
        make_checkpoint(tmp_path, 30)
        (tmp_path / "completions").mkdir()
        (tmp_path / "run_config.json").write_text("{}")
        resolved = resolve_resume_checkpoint(RESUME_LATEST, str(tmp_path))
        assert resolved is not None
        assert resolved.endswith("checkpoint-30")


class TestResolvingAnExplicitCheckpoint:
    def test_a_real_checkpoint_passes_through(self, tmp_path: Path) -> None:
        checkpoint = make_checkpoint(tmp_path, 40)
        assert resolve_resume_checkpoint(str(checkpoint), str(tmp_path)) == str(checkpoint)

    def test_a_path_that_does_not_exist_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is not a directory"):
            resolve_resume_checkpoint(str(tmp_path / "checkpoint-40"), str(tmp_path))

    def test_a_directory_without_trainer_state_raises(self, tmp_path: Path) -> None:
        """A typo here would restart at step 0 with a warm optimizer and lie about the steps."""
        checkpoint = make_checkpoint(tmp_path, 40, with_state=False)
        with pytest.raises(ValueError, match=f"has no {TRAINER_STATE_FILENAME}"):
            resolve_resume_checkpoint(str(checkpoint), str(tmp_path))

    def test_an_empty_setting_means_a_fresh_run(self, tmp_path: Path) -> None:
        make_checkpoint(tmp_path, 40)
        assert resolve_resume_checkpoint("", str(tmp_path)) is None


class TestMissingCheckpointFiles:
    def test_a_complete_checkpoint_is_missing_nothing(self, tmp_path: Path) -> None:
        assert missing_checkpoint_files(write_checkpoint(tmp_path, 1)) == []

    def test_the_legacy_bin_adapter_name_also_counts(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(tmp_path, 1, omit=("adapter_model.safetensors",))
        (checkpoint / "adapter_model.bin").write_bytes(b"")
        assert missing_checkpoint_files(checkpoint) == []

    def test_every_missing_file_is_named(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(
            tmp_path, 1, omit=("optimizer.pt", "rng_state.pth", "adapter_model.safetensors")
        )
        assert missing_checkpoint_files(checkpoint) == [
            "optimizer.pt",
            "rng_state.pth",
            "adapter_model.safetensors or adapter_model.bin",
        ]


class TestLatestFallsBackToTheNewestCompleteCheckpoint:
    """The gate with teeth: a reclaim mid-sync leaves `checkpoint-3` without `optimizer.pt`.

    Sabotage that proved this class fails for the right reason: with `missing_checkpoint_files`
    made to return nothing, `test_a_torn_newest_checkpoint_is_stepped_past` resolves `checkpoint-3`
    and goes red (recorded in the group report for this change).
    """

    def test_a_complete_latest_checkpoint_is_resumed_and_nothing_is_set_aside(
        self, tmp_path: Path
    ) -> None:
        write_checkpoint(tmp_path, 2)
        newest = write_checkpoint(tmp_path, 3)
        resolution = resolve_latest(tmp_path)
        assert resolution.checkpoint == str(newest)
        assert resolution.set_aside == ()
        assert not (tmp_path / INCOMPLETE_CHECKPOINTS_DIRNAME).exists()

    def test_a_torn_newest_checkpoint_is_stepped_past(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        complete = write_checkpoint(tmp_path, 2)
        torn = write_checkpoint(tmp_path, 3, omit=("optimizer.pt",))
        with caplog.at_level("WARNING", logger="games.train"):
            resolution = resolve_latest(tmp_path)
        assert resolution.checkpoint == str(complete)
        moved_to = tmp_path / INCOMPLETE_CHECKPOINTS_DIRNAME / "20260902T181500Z" / "checkpoint-3"
        assert resolution.set_aside == (
            {
                "checkpoint": "checkpoint-3",
                "step": 3,
                "missing": ["optimizer.pt"],
                "moved_to": str(moved_to),
            },
        )
        assert not torn.exists()
        assert (moved_to / TRAINER_STATE_FILENAME).is_file()
        assert "INCOMPLETE CHECKPOINT: checkpoint-3 lacks ['optimizer.pt']" in caplog.text
        assert "falling back to checkpoint-2 (step 2)" in caplog.text

    def test_several_torn_checkpoints_are_all_stepped_past_newest_first(
        self, tmp_path: Path
    ) -> None:
        complete = write_checkpoint(tmp_path, 10)
        write_checkpoint(tmp_path, 20, omit=("scheduler.pt",))
        write_checkpoint(tmp_path, 30, omit=("optimizer.pt", "scheduler.pt"))
        resolution = resolve_latest(tmp_path)
        assert resolution.checkpoint == str(complete)
        assert [entry["checkpoint"] for entry in resolution.set_aside] == [
            "checkpoint-30",
            "checkpoint-20",
        ]
        assert resolution.set_aside[0]["missing"] == ["optimizer.pt", "scheduler.pt"]

    def test_older_torn_checkpoints_below_a_complete_one_are_left_alone(
        self, tmp_path: Path
    ) -> None:
        """An optimizer-light retention policy prunes exactly those on purpose."""
        write_checkpoint(tmp_path, 1, omit=("optimizer.pt",))
        complete = write_checkpoint(tmp_path, 2)
        resolution = resolve_latest(tmp_path)
        assert resolution.checkpoint == str(complete)
        assert resolution.set_aside == ()
        assert (tmp_path / "checkpoint-1").is_dir()

    def test_a_relaunch_after_the_fallback_resumes_the_same_checkpoint_cleanly(
        self, tmp_path: Path
    ) -> None:
        """The torn directory is gone from the glob, so the second launch has nothing to set aside."""
        complete = write_checkpoint(tmp_path, 2)
        write_checkpoint(tmp_path, 3, omit=("optimizer.pt",))
        resolve_latest(tmp_path)
        second = resolve_latest(tmp_path)
        assert second.checkpoint == str(complete)
        assert second.set_aside == ()

    def test_no_complete_checkpoint_at_all_raises_rather_than_starting_fresh(
        self, tmp_path: Path
    ) -> None:
        write_checkpoint(tmp_path, 1, omit=("optimizer.pt",))
        write_checkpoint(tmp_path, 2, omit=("rng_state.pth",))
        with pytest.raises(RuntimeError, match="none is a complete checkpoint"):
            resolve_latest(tmp_path)
        assert not (tmp_path / INCOMPLETE_CHECKPOINTS_DIRNAME).exists()

    def test_an_empty_or_missing_run_directory_still_starts_fresh(self, tmp_path: Path) -> None:
        assert resolve_latest(tmp_path) == ResumeResolution(checkpoint=None)
        assert resolve_latest(tmp_path / "not-yet") == ResumeResolution(checkpoint=None)


class TestAnExplicitCheckpointIsRefusedWhenIncomplete:
    def test_a_complete_explicit_checkpoint_passes_through(self, tmp_path: Path) -> None:
        checkpoint = write_checkpoint(tmp_path, 4)
        resolution = resolve_complete_resume_checkpoint(str(checkpoint), str(tmp_path))
        assert resolution == ResumeResolution(checkpoint=str(checkpoint))

    def test_a_torn_explicit_checkpoint_is_refused_not_substituted(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 3)
        torn = write_checkpoint(tmp_path, 4, omit=("optimizer.pt",))
        with pytest.raises(ValueError, match=r"is missing \['optimizer.pt'\]"):
            resolve_complete_resume_checkpoint(str(torn), str(tmp_path))

    def test_an_empty_setting_means_a_fresh_run(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path, 3)
        assert resolve_complete_resume_checkpoint("", str(tmp_path)) == ResumeResolution(
            checkpoint=None
        )


class TestTheSilentFailureIsRefused:
    def test_latest_without_an_output_dir_raises_at_config_time(self) -> None:
        with pytest.raises(ValueError, match="needs an explicit --output-dir"):
            GameTrainConfig(arm=ARM, generate_fresh=True, resume_from_checkpoint=RESUME_LATEST)

    def test_latest_with_an_output_dir_is_accepted(self, tmp_path: Path) -> None:
        config = GameTrainConfig(
            arm=ARM,
            generate_fresh=True,
            resume_from_checkpoint=RESUME_LATEST,
            output_dir=str(tmp_path),
        )
        assert config.resume_from_checkpoint == RESUME_LATEST

    def test_an_explicit_path_needs_no_output_dir(self) -> None:
        """Only the "newest in the run directory" form depends on knowing the run directory."""
        assert_resume_is_addressable("/runs/twin-pd/checkpoint-40", None)

    def test_a_fresh_run_needs_no_output_dir(self) -> None:
        assert_resume_is_addressable("", None)


class TestTheConfigCarriesItIntoTheRunRecord:
    def test_it_defaults_to_a_fresh_run(self) -> None:
        config = GameTrainConfig(arm=ARM, generate_fresh=True)
        assert config.resume_from_checkpoint == ""

    def test_it_lands_in_the_serialised_config(self, tmp_path: Path) -> None:
        """`run_config.json` is written from the dataclass, so the field is recorded for free.

        That matters for reading a trace later: a run that resumed at step 40 and one that trained
        40 steps from scratch are different experiments with the same step numbers.
        """
        import dataclasses  # noqa: PLC0415

        config = GameTrainConfig(
            arm=ARM,
            generate_fresh=True,
            resume_from_checkpoint=RESUME_LATEST,
            output_dir=str(tmp_path),
        )
        assert dataclasses.asdict(config)["resume_from_checkpoint"] == RESUME_LATEST


class TestTheCliFlag:
    def test_the_flag_reaches_the_config(self, tmp_path: Path) -> None:
        from games.train import _parse_args  # noqa: PLC0415

        config = _parse_args(
            [
                "--arm",
                ARM,
                "--generate-fresh",
                "--output-dir",
                str(tmp_path),
                "--resume-from-checkpoint",
                RESUME_LATEST,
            ]
        )
        assert config.resume_from_checkpoint == RESUME_LATEST
        assert config.output_dir == str(tmp_path)

    def test_omitting_the_flag_leaves_a_fresh_run(self, tmp_path: Path) -> None:
        from games.train import _parse_args  # noqa: PLC0415

        config = _parse_args(["--arm", ARM, "--generate-fresh", "--output-dir", str(tmp_path)])
        assert config.resume_from_checkpoint == ""
