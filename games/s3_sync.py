"""Ship a run's directory to S3 every time the trainer saves a checkpoint.

Batch jobs die: spot reclamation, an OOM at step 43, a wall-clock timeout, an instance going
unhealthy. The container's disk goes with them. `cloud/entrypoint.sh` installs an exit-time sync
that covers a clean-ish shutdown, but a hard kill can skip even that, and a nine-hour run that
leaves nothing behind has to be paid for twice. So the sync also runs on every save: after each
`save_steps` interval the checkpoints are already off the instance, and the worst case degrades
from "lose the run" to "lose the steps since the last save".

Two deliberate choices.

**Shelling out to `aws s3 sync` rather than reimplementing it.** Multipart uploads, retries,
parallelism, and skip-if-unchanged are exactly the parts that are tedious to get right and awful to
debug at 2am against a 20 GB checkpoint directory. The command is injectable, which is also what
makes this testable without AWS.

**A failed sync never fails the run.** Training is the expensive thing; an upload problem is
logged loudly and the step continues. The opposite policy would let a transient S3 error throw away
the GPU hours it was supposed to protect.

One thing here must not be narrowed: the run directory ships wholesale. TRL's `log_completions`
writes the per-completion rollout trace to `<output_dir>/completions/*.parquet`, which is the raw
material the research reads and the one artifact whose absence announces itself nowhere. Syncing a
file list, or adding `--exclude`, silently drops it; `TestS3SyncCommand` asserts against that.

Two things now ride on that coverage rather than one. `games/stage_runner.py` writes each stage's
log into `<output_dir>/logs/` as well as the shared `artifacts/games/logs/`, precisely because this
sync reaches the run directory and nothing reaches the shared one -- so a narrowing here would take
the sweep, regrade and training logs with it.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

DEFAULT_SYNC_COMMAND: tuple[str, ...] = ("aws", "s3", "sync")
DEFAULT_SYNC_TIMEOUT_SECONDS = 30 * 60

# Stand-ins for a sync that produced no exit code, spelled as the shells spell them.
WALL_CLOCK_CAP_RETURNCODE = 124
UNRUNNABLE_SYNC_RETURNCODE = 127


@dataclass
class SyncOutcome:
    """What one sync attempt did, so the caller can assert on it and the log can report it."""

    command: tuple[str, ...]
    returncode: int
    skipped_reason: str | None = None
    failure_reason: str | None = None

    @property
    def ran(self) -> bool:
        """Whether a sync was actually attempted."""
        return self.skipped_reason is None

    @property
    def succeeded(self) -> bool:
        """Whether the sync ran and exited cleanly."""
        return self.ran and self.returncode == 0


def build_sync_command(
    local_dir: Path, s3_dest: str, *, sync_command: Sequence[str] = DEFAULT_SYNC_COMMAND
) -> tuple[str, ...]:
    """Build the argv for one directory-to-S3 sync.

    `--only-show-errors` keeps a per-file progress dump out of the training log, where it would
    bury the reward curve under thousands of upload lines.
    """
    if not s3_dest.startswith("s3://"):
        raise ValueError(f"s3_dest must be an s3:// URI, got {s3_dest!r}.")
    return (*sync_command, str(local_dir), s3_dest, "--only-show-errors")


def sync_directory(
    local_dir: Path,
    s3_dest: str,
    *,
    sync_command: Sequence[str] = DEFAULT_SYNC_COMMAND,
    timeout_seconds: int = DEFAULT_SYNC_TIMEOUT_SECONDS,
) -> SyncOutcome:
    """Sync one directory to S3, returning what happened rather than raising.

    A missing directory is a skip, not an error: the first save can fire before anything has been
    written, and a run that has produced nothing yet has nothing to lose.

    Nothing reaches the caller as an exception, which takes two catches rather than none:
    `check=False` suppresses a non-zero exit but not `TimeoutExpired` (raised after the child is
    killed) and not the `OSError` an unrunnable command raises. Both matter because
    `transformers`' `CallbackHandler.call_event` wraps callbacks in no try/except, so anything
    raised out of `on_save` aborts `trainer.train()`: a network partition at step 65 of 70 would
    end the run this sync exists to protect, and would end it again on every relaunch.
    """
    argv = build_sync_command(local_dir, s3_dest, sync_command=sync_command)
    if not local_dir.is_dir():
        logger.warning(f"skipping s3 sync, {local_dir=} does not exist yet")
        return SyncOutcome(command=argv, returncode=0, skipped_reason="missing-local-dir")

    logger.info(f"syncing to s3, {local_dir=} {s3_dest=}")
    try:
        finished = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        reason = f"the sync timed out after {timeout_seconds}s and was killed"
        logger.exception(f"s3 sync failed and the run continues: {reason}, {argv=}")
        return SyncOutcome(
            command=argv, returncode=WALL_CLOCK_CAP_RETURNCODE, failure_reason=reason
        )
    except OSError as error:
        reason = f"the sync command could not be run: {error}"
        logger.exception(f"s3 sync failed and the run continues: {reason}, {argv=}")
        return SyncOutcome(
            command=argv, returncode=UNRUNNABLE_SYNC_RETURNCODE, failure_reason=reason
        )
    if finished.returncode != 0:
        logger.error(
            f"s3 sync failed and the run continues, {finished.returncode=} "
            f"stderr={finished.stderr.strip()[:2000]}"
        )
    return SyncOutcome(command=argv, returncode=finished.returncode)


def build_restore_command(
    s3_dest: str, local_dir: Path, *, sync_command: Sequence[str] = DEFAULT_SYNC_COMMAND
) -> tuple[str, ...]:
    """Build the argv for one S3-to-directory restore: the same tool, the arguments reversed."""
    if not s3_dest.startswith("s3://"):
        raise ValueError(f"s3_dest must be an s3:// URI, got {s3_dest!r}.")
    return (*sync_command, s3_dest, str(local_dir), "--only-show-errors")


def restore_directory(
    s3_dest: str,
    local_dir: Path,
    *,
    sync_command: Sequence[str] = DEFAULT_SYNC_COMMAND,
    timeout_seconds: int = DEFAULT_SYNC_TIMEOUT_SECONDS,
) -> SyncOutcome:
    """Pull a run directory back from S3, and RAISE when the pull fails.

    The missing half of recoverability, and the reason it is worth its own function rather than a
    reversed call: for years this repository only ever uploaded, so a job resubmitted onto a fresh
    instance after a spot reclaim found an empty output directory, resumed from nothing, and started
    again at step 0 with every signal green. A reclaim takes the instance's disk with it, so
    `--resume-from-checkpoint latest` means "latest of what is on this disk", which is nothing.

    The failure policy is the OPPOSITE of `sync_directory`'s, deliberately. A failed upload is worth
    swallowing because training is the expensive thing and the next save will try again. A failed
    restore is not: the run would silently retrain steps it had already paid for, and every step
    number in its trace afterwards would be a lie about which data the adapter had seen. So this
    raises, and the operator relaunches once the bucket is reachable.

    An empty or absent prefix is NOT a failure -- `aws s3 sync` exits 0 having copied nothing, which
    is exactly right for the first launch of a run, where there is nothing to restore yet.
    """
    argv = build_restore_command(s3_dest, local_dir, sync_command=sync_command)
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"restoring from s3, {s3_dest=} {local_dir=}")
    try:
        finished = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"the restore from {s3_dest} timed out after {timeout_seconds}s and was killed. "
            f"Training would have resumed from an incomplete directory, so nothing was started."
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"the restore command could not be run ({error}), so nothing says whether this run has "
            f"earlier checkpoints in {s3_dest}. Training was not started."
        ) from error
    if finished.returncode != 0:
        raise RuntimeError(
            f"the restore from {s3_dest} failed with exit {finished.returncode}: "
            f"{finished.stderr.strip()[:2000]}. Resuming from a partial directory would retrain "
            f"steps already paid for and mislabel every step number after them, so nothing was "
            f"started."
        )
    return SyncOutcome(command=argv, returncode=finished.returncode)


@dataclass
class S3SyncCallback(TrainerCallback):
    """Sync the run directory to S3 on every checkpoint save, and once when training ends.

    Wire it up from `games/train.py` with one line, only when a destination is configured:

        if config.s3_dest:
            callbacks.append(S3SyncCallback(local_dir=Path(config.output_dir),
                                            s3_dest=config.s3_dest))

    Nothing here touches the trainer's log dict. The metric-logging path in this repo goes through
    TRL's injected `log_metric`, because a callback that mutates the dict `Trainer.log` handed it is
    writing into a copy -- the bug that lost a whole run's verifier accuracy.
    """

    local_dir: Path
    s3_dest: str
    sync_command: Sequence[str] = DEFAULT_SYNC_COMMAND
    timeout_seconds: int = DEFAULT_SYNC_TIMEOUT_SECONDS
    outcomes: list[SyncOutcome] = field(default_factory=list)

    def _sync(self, reason: str) -> SyncOutcome:
        """Run one sync and remember the outcome."""
        logger.info(f"s3 sync triggered, {reason=}")
        outcome = sync_directory(
            self.local_dir,
            self.s3_dest,
            sync_command=self.sync_command,
            timeout_seconds=self.timeout_seconds,
        )
        self.outcomes.append(outcome)
        return outcome

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Ship everything written so far, right after the trainer writes a checkpoint."""
        del args, control, kwargs
        self._sync(f"on_save at step {state.global_step}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Catch whatever the final save left behind, including the trainer state."""
        del args, control, kwargs
        self._sync(f"on_train_end at step {state.global_step}")
