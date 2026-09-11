"""Pin the checkpoint-to-S3 sync, entirely offline: the sync command is injected, no AWS is called.

The invariant with teeth is that **a failed sync never fails the run**. Training is the expensive
thing; an upload problem is logged and the step continues. So every way a sync can go wrong is
committed here as the exact violation -- a non-zero exit, a hang against the cap, a command that
cannot be executed at all -- and asserted to come back as an outcome rather than as an exception.
That matters because `transformers.trainer_callback.CallbackHandler.call_event` wraps callbacks in
no try/except, so anything raised out of `on_save` aborts `trainer.train()`: the failure this
callback exists to prevent would be the failure it causes, on every relaunch.

The other invariant is coverage: the run directory ships wholesale, because TRL writes the
per-completion rollout parquet inside it and a narrowed sync drops that silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from transformers import TrainerControl, TrainerState

from games.s3_sync import (
    DEFAULT_SYNC_COMMAND,
    UNRUNNABLE_SYNC_RETURNCODE,
    WALL_CLOCK_CAP_RETURNCODE,
    S3SyncCallback,
    build_sync_command,
    sync_directory,
)

if TYPE_CHECKING:
    from transformers import TrainingArguments


def fire(callback: S3SyncCallback, hook: str, *, step: int) -> None:
    """Invoke a callback hook with real trainer state, standing in only for the unused args.

    `TrainerState` and `TrainerControl` are cheap dataclasses so the real ones are used;
    `TrainingArguments` does device detection in __post_init__ and the callback never reads it.
    """
    getattr(callback, hook)(
        cast("TrainingArguments", None), TrainerState(global_step=step), TrainerControl()
    )


class TestS3SyncCommand:
    def test_the_command_shape(self, tmp_path: Path) -> None:
        argv = build_sync_command(tmp_path, "s3://bucket/runs/arm")
        assert argv == (
            *DEFAULT_SYNC_COMMAND,
            str(tmp_path),
            "s3://bucket/runs/arm",
            "--only-show-errors",
        )

    def test_a_non_s3_destination_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be an s3:// URI"):
            build_sync_command(tmp_path, "/mnt/backups/runs")

    def test_the_whole_run_directory_ships_so_the_rollout_trace_does_too(
        self, tmp_path: Path
    ) -> None:
        """A narrowed sync loses the per-completion parquet, and nothing reports the loss.

        TRL writes `<output_dir>/completions/completions_<step>.parquet` from inside `log`
        (`grpo_trainer.py`, `df_base.to_parquet`). On 2026-08-19 a comparison of two generation
        backends had to recover its rollouts from a rich-rendered table in the training log, with
        the completion text ellipsis-truncated beyond repair, because the sync in use covered a log
        directory rather than the run. So the assertion is on coverage: the source is the run root,
        and no filter flag stands between it and the parquet.
        """
        trace = tmp_path / "completions" / "completions_00001.parquet"
        trace.parent.mkdir()
        trace.write_bytes(b"parquet")
        argv = build_sync_command(tmp_path, "s3://bucket/runs/arm")
        source = Path(argv[len(DEFAULT_SYNC_COMMAND)])
        assert trace.relative_to(source)
        assert not [flag for flag in argv if flag.startswith(("--exclude", "--include"))]

    def test_a_missing_local_dir_is_skipped_not_failed(self, tmp_path: Path) -> None:
        """The first save can fire before anything has been written."""
        outcome = sync_directory(tmp_path / "absent", "s3://bucket/runs")
        assert outcome.ran is False
        assert outcome.skipped_reason == "missing-local-dir"

    def test_a_successful_sync_runs_the_injected_command(self, tmp_path: Path) -> None:
        outcome = sync_directory(tmp_path, "s3://bucket/runs", sync_command=("true",))
        assert outcome.succeeded is True
        assert outcome.command[0] == "true"
        assert outcome.failure_reason is None

    def test_a_failing_sync_reports_rather_than_raising(self, tmp_path: Path) -> None:
        """A transient upload error must never throw away the GPU hours it was protecting."""
        outcome = sync_directory(tmp_path, "s3://bucket/runs", sync_command=("false",))
        assert outcome.ran is True
        assert outcome.succeeded is False
        assert outcome.returncode != 0

    def test_a_hanging_sync_is_bounded_and_reported_rather_than_raised(
        self, tmp_path: Path
    ) -> None:
        """The cap must kill the upload, not the run.

        This assertion was `pytest.raises(subprocess.TimeoutExpired)` until 2026-08-19, i.e. it
        pinned the bug: `subprocess.run` raises `TimeoutExpired` after killing the child whatever
        `check=False` says, and nothing between here and `trainer.train()` catches it.
        """
        outcome = sync_directory(
            tmp_path,
            "s3://bucket/runs",
            sync_command=("sh", "-c", "sleep 5"),
            timeout_seconds=1,
        )
        assert outcome.ran is True
        assert outcome.succeeded is False
        assert outcome.returncode == WALL_CLOCK_CAP_RETURNCODE
        assert outcome.failure_reason is not None
        assert "timed out" in outcome.failure_reason

    def test_a_missing_aws_binary_is_reported_rather_than_raised(self, tmp_path: Path) -> None:
        """`aws` absent from a container's PATH is an upload problem, not a training problem."""
        outcome = sync_directory(
            tmp_path, "s3://bucket/runs", sync_command=("aws-that-is-not-installed",)
        )
        assert outcome.ran is True
        assert outcome.succeeded is False
        assert outcome.returncode == UNRUNNABLE_SYNC_RETURNCODE
        assert outcome.failure_reason is not None
        assert "aws-that-is-not-installed" in outcome.failure_reason


class TestS3SyncCallback:
    def test_on_save_syncs_the_run_directory(self, tmp_path: Path) -> None:
        callback = S3SyncCallback(
            local_dir=tmp_path, s3_dest="s3://bucket/runs", sync_command=("true",)
        )
        fire(callback, "on_save", step=20)
        assert len(callback.outcomes) == 1
        assert callback.outcomes[0].succeeded is True

    def test_on_train_end_syncs_once_more(self, tmp_path: Path) -> None:
        callback = S3SyncCallback(
            local_dir=tmp_path, s3_dest="s3://bucket/runs", sync_command=("true",)
        )
        fire(callback, "on_save", step=10)
        fire(callback, "on_train_end", step=70)
        assert len(callback.outcomes) == 2

    def test_a_failed_sync_does_not_raise_out_of_the_callback(self, tmp_path: Path) -> None:
        callback = S3SyncCallback(
            local_dir=tmp_path, s3_dest="s3://bucket/runs", sync_command=("false",)
        )
        fire(callback, "on_save", step=10)
        assert callback.outcomes[0].succeeded is False

    def test_a_hanging_sync_does_not_take_the_run_down_with_it(self, tmp_path: Path) -> None:
        """A network partition at step 65 of 70 must not be what ends the run."""
        callback = S3SyncCallback(
            local_dir=tmp_path,
            s3_dest="s3://bucket/runs",
            sync_command=("sh", "-c", "sleep 5"),
            timeout_seconds=1,
        )
        fire(callback, "on_save", step=65)
        assert callback.outcomes[0].succeeded is False
        assert callback.outcomes[0].returncode == WALL_CLOCK_CAP_RETURNCODE

    def test_an_unrunnable_sync_command_does_not_take_the_run_down_either(
        self, tmp_path: Path
    ) -> None:
        callback = S3SyncCallback(
            local_dir=tmp_path,
            s3_dest="s3://bucket/runs",
            sync_command=("aws-that-is-not-installed",),
        )
        fire(callback, "on_train_end", step=70)
        assert callback.outcomes[0].succeeded is False
        assert callback.outcomes[0].returncode == UNRUNNABLE_SYNC_RETURNCODE
