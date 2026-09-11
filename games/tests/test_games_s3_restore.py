"""The S3 restore: the direction this repository never had, and the one that must RAISE.

Offline. The sync command is injected, so nothing here talks to AWS.

Why the direction matters enough to test separately from ``sync_directory``. Every checkpoint has
always been uploaded on save, but nothing ever downloaded one -- so a job resubmitted onto a fresh
instance after a spot reclaim found an empty output directory, resumed from nothing, and began again
at step 0 with every signal green. The failure policy is the mirror image of the upload's on purpose:
a failed upload is worth swallowing because the next save will try again, while a failed restore
would silently retrain steps already paid for and mislabel every step number after them.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from games.s3_sync import build_restore_command, restore_directory

if TYPE_CHECKING:
    from pathlib import Path

DESTINATION = "s3://bucket/prefix/misspecified-Qwen-Qwen3.5-4B"


class RecordingCommand:
    """A stand-in for `aws s3 sync` that records its argv and answers with a chosen exit code."""

    def __init__(self, returncode: int = 0, *, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        self.calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, self.returncode, "", self.stderr)


class TestRestoreCommand:
    def test_the_arguments_are_the_upload_reversed(self, tmp_path: Path):
        argv = build_restore_command(DESTINATION, tmp_path)
        assert argv[:3] == ("aws", "s3", "sync")
        assert argv[3] == DESTINATION
        assert argv[4] == str(tmp_path)

    def test_a_non_s3_destination_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="must be an s3:// URI"):
            build_restore_command("/local/path", tmp_path)


class TestRestoreDirectory:
    def test_a_clean_restore_reports_success_and_creates_the_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        command = RecordingCommand()
        monkeypatch.setattr(subprocess, "run", command)
        target = tmp_path / "runs" / "arm"
        outcome = restore_directory(DESTINATION, target)
        assert outcome.succeeded
        assert target.is_dir()
        assert command.calls[0][3:5] == (DESTINATION, str(target))

    def test_a_failed_restore_raises_rather_than_letting_training_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Sabotage: a non-zero exit, which is what an unreachable bucket or a bad prefix produces.

        Watched to fail: without this the run would resume from a partial directory, retrain steps it
        had already paid for, and every step number in its trace afterwards would be a lie about
        which data the adapter had seen.
        """
        monkeypatch.setattr(subprocess, "run", RecordingCommand(1, stderr="Access Denied"))
        with pytest.raises(RuntimeError, match="failed with exit 1"):
            restore_directory(DESTINATION, tmp_path / "arm")

    def test_a_timed_out_restore_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        def timing_out(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            del kwargs
            raise subprocess.TimeoutExpired(argv, 1)

        monkeypatch.setattr(subprocess, "run", timing_out)
        with pytest.raises(RuntimeError, match="timed out"):
            restore_directory(DESTINATION, tmp_path / "arm")

    def test_an_unrunnable_restore_command_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def missing(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            del argv, kwargs
            raise OSError(2, "No such file or directory")

        monkeypatch.setattr(subprocess, "run", missing)
        with pytest.raises(RuntimeError, match="could not be run"):
            restore_directory(DESTINATION, tmp_path / "arm")

    def test_an_empty_prefix_is_not_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The first launch of a run: `aws s3 sync` exits 0 having copied nothing, correctly."""
        monkeypatch.setattr(subprocess, "run", RecordingCommand(0))
        assert restore_directory(DESTINATION, tmp_path / "fresh").succeeded
