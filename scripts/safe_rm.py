#!/usr/bin/python3
"""Delete explicitly selected temporary paths after checking every safety guard."""

from __future__ import annotations

import argparse
import datetime
import getpass
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if __package__:
    from scripts.ttl_marker import TTLMarkerError, read_ttl_marker
else:
    from ttl_marker import TTLMarkerError, read_ttl_marker

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
VAR_TMP = Path("/var/tmp")  # noqa: S108 - intentional cleanup allowlist root
TMP = Path("/tmp")  # noqa: S108 - intentional cleanup allowlist root
LOG_PATH = Path.home() / "logs" / "safe_rm.log"


class RemovalRefusalError(Exception):
    """A validation failure with the byte count inspected before refusing."""

    def __init__(self, reason: str, byte_count: int = 0) -> None:
        """Create a refusal with the partial byte count gathered during validation."""
        super().__init__(reason)
        self.reason = reason
        self.byte_count = byte_count


class Assessment(NamedTuple):
    """Validation result for one command-line path."""

    argument: str
    path: Path
    byte_count: int
    refusal: str | None


class RemovalRoots(NamedTuple):
    """Resolved roots that define which paths safe_rm may consider."""

    repo_root: Path
    var_tmp: Path
    pytest_root: Path
    artifact_root: Path


def is_within(path: Path, root: Path) -> bool:
    """Return whether path is a strict descendant of root."""
    return path != root and root in path.parents


def _roots(repo_root: Path, var_tmp: Path, tmp_root: Path) -> RemovalRoots:
    return RemovalRoots(
        repo_root.resolve(),
        var_tmp.resolve(),
        (tmp_root / f"pytest-of-{getpass.getuser()}").resolve(),
        (repo_root / "artifacts").resolve(),
    )


def _marker_exists(path: Path, byte_count: int) -> bool:
    try:
        os.lstat(str(path))
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as error:
        raise RemovalRefusalError(
            f"cannot inspect protection marker: {error}", byte_count
        ) from error
    return True


def _marked_artifact_tree(path: Path, artifact_root: Path, byte_count: int) -> bool:
    if not is_within(path, artifact_root):
        return False
    for ancestor in (path, *path.parents):
        if ancestor == artifact_root:
            break
        marker = ancestor / ".disposable"
        try:
            marker_metadata = os.lstat(str(marker))
        except (FileNotFoundError, NotADirectoryError):
            marker_metadata = None
        except OSError as error:
            raise RemovalRefusalError(
                f"cannot inspect disposable marker: {error}", byte_count
            ) from error
        if (
            marker_metadata is not None
            and stat.S_ISREG(marker_metadata.st_mode)
            and marker_metadata.st_uid == os.getuid()
        ):
            return True
        ttl_marker = ancestor / ".ttl"
        try:
            os.lstat(str(ttl_marker))
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as error:
            raise RemovalRefusalError(f"cannot inspect TTL marker: {error}", byte_count) from error
        try:
            read_ttl_marker(ttl_marker)
        except TTLMarkerError as error:
            raise RemovalRefusalError(f"invalid TTL marker: {error}", byte_count) from error
        return True
    return False


def _marked_root(path: Path, byte_count: int) -> bool:
    """Return whether path itself carries a valid marker for exceptional tree entries."""
    disposable_marker = path / ".disposable"
    try:
        disposable_metadata = os.lstat(str(disposable_marker))
    except (FileNotFoundError, NotADirectoryError):
        disposable_metadata = None
    except OSError as error:
        raise RemovalRefusalError(
            f"cannot inspect disposable marker: {error}", byte_count
        ) from error
    if (
        disposable_metadata is not None
        and stat.S_ISREG(disposable_metadata.st_mode)
        and disposable_metadata.st_uid == os.getuid()
    ):
        return True

    ttl_marker = path / ".ttl"
    try:
        os.lstat(str(ttl_marker))
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as error:
        raise RemovalRefusalError(f"cannot inspect TTL marker: {error}", byte_count) from error
    try:
        read_ttl_marker(ttl_marker)
    except TTLMarkerError as error:
        raise RemovalRefusalError(f"invalid TTL marker: {error}", byte_count) from error
    return True


def _protected_var_tmp_path(path: Path, temporary_root: Path) -> str | None:
    if not is_within(path, temporary_root):
        return None
    relative_parts = path.relative_to(temporary_root).parts
    protected_paths = {
        "s3-mirror": ("var", "tmp", "s3-mirror"),
        "cpython-runtime": ("var", "tmp", "cpython-runtime"),
        "atlas-train-extra-deps": ("var", "tmp", "atlas-train-extra-deps"),
    }
    if relative_parts[0] in protected_paths:
        return "protected path: /" + "/".join(protected_paths[relative_parts[0]])
    if relative_parts[0].startswith("flashinfer-"):
        return "protected path: /var/tmp/flashinfer-*"
    if len(relative_parts) == 1 and relative_parts[0].endswith(".lock"):
        return "protected path: /var/tmp/*.lock"
    return None


def _allowed_root(
    path: Path, original_path: Path, roots: RemovalRoots, byte_count: int
) -> str | None:
    for checked_path in (path, original_path):
        if checked_path == roots.var_tmp:
            return "allowlisted root itself"
        protected_reason = _protected_var_tmp_path(checked_path, roots.var_tmp)
        if protected_reason:
            return protected_reason

    if path in (roots.pytest_root, roots.artifact_root):
        return "allowlisted root itself"
    if is_within(path, roots.var_tmp) or is_within(path, roots.pytest_root):
        return None
    if _marked_artifact_tree(path, roots.artifact_root, byte_count):
        return None
    return "outside deletion allowlist"


def owns_entry(path: Path, metadata: os.stat_result) -> bool:
    """Return whether this entry belongs to the current user."""
    del path
    return metadata.st_uid == os.getuid()


def registered_worktrees(repo_root: Path) -> list[Path]:
    """Read Git's registered worktree roots; inability to inspect them blocks deletion."""
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RemovalRefusalError("git executable unavailable")
    result = subprocess.run(  # noqa: S603 - fixed Git command; repo_root comes from this script
        [git_executable, "-C", str(repo_root), "worktree", "list", "--porcelain"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RemovalRefusalError(f"git worktree list failed: {result.stderr.strip()}")
    return [
        Path(line[len("worktree ") :]).resolve()
        for line in result.stdout.splitlines()
        if line.startswith("worktree ")
    ]


def _check_worktree(path: Path, repo_root: Path, byte_count: int) -> None:
    try:
        worktrees = registered_worktrees(repo_root)
    except RemovalRefusalError as error:
        raise RemovalRefusalError(error.reason, byte_count) from error
    except OSError as error:
        raise RemovalRefusalError(f"git worktree list unavailable: {error}", byte_count) from error
    if any(path == worktree or path in worktree.parents for worktree in worktrees):
        raise RemovalRefusalError("path is or contains a registered git worktree", byte_count)


def _check_keep_ancestors(path: Path, original_path: Path, byte_count: int) -> None:
    if path.name == ".keep" or original_path.name == ".keep":
        raise RemovalRefusalError(".keep protection", byte_count)
    ancestors = set(path.parents) | set(original_path.parents)
    for ancestor in ancestors:
        if _marker_exists(ancestor / ".keep", byte_count):
            raise RemovalRefusalError(".keep protection in path ancestor", byte_count)


def _entry_metadata(path: Path, byte_count: int) -> os.stat_result:
    try:
        return os.lstat(str(path))
    except OSError as error:
        raise RemovalRefusalError(
            f"tree changed or cannot be inspected: {error}", byte_count
        ) from error


def _check_keep_child(name: str, byte_count: int) -> None:
    if name == ".keep":
        raise RemovalRefusalError(".keep protection in descendant", byte_count)


def _check_git_entry(path: Path, byte_count: int) -> None:
    if path.name == ".git":
        raise RemovalRefusalError("path contains .git metadata", byte_count)


def _is_mount_point(path: Path) -> bool:
    return os.path.ismount(str(path))


def _scan_tree(path: Path, *, allow_marked_entries: bool) -> int:  # noqa: C901, PLR0912 - explicit fail-closed tree checks
    """Inspect the entire tree without following links and count regular-file bytes."""
    byte_count = 0
    pending = [path]
    while pending:
        current = pending.pop()
        metadata = _entry_metadata(current, byte_count)
        _check_owner(current, metadata, byte_count)
        if _is_mount_point(current):
            raise RemovalRefusalError("path contains a mount point", byte_count)
        if stat.S_ISREG(metadata.st_mode):
            byte_count += metadata.st_size
            continue
        if stat.S_ISLNK(metadata.st_mode):
            byte_count += metadata.st_size
            raise RemovalRefusalError("path contains a symlink", byte_count)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RemovalRefusalError("path contains a special file", byte_count)
        if not allow_marked_entries:
            _check_git_entry(current, byte_count)

        try:
            with os.scandir(str(current)) as entries:
                children = list(entries)
        except OSError as error:
            raise RemovalRefusalError(f"tree cannot be scanned: {error}", byte_count) from error

        for entry in children:
            child = current / entry.name
            try:
                child_metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise RemovalRefusalError(
                    f"tree changed or cannot be inspected: {error}", byte_count
                ) from error
            _check_keep_child(entry.name, byte_count)
            if not allow_marked_entries:
                _check_git_entry(child, byte_count)
            _check_owner(child, child_metadata, byte_count)
            if not stat.S_ISLNK(child_metadata.st_mode) and _is_mount_point(child):
                raise RemovalRefusalError("path contains a mount point", byte_count)
            if stat.S_ISREG(child_metadata.st_mode):
                byte_count += child_metadata.st_size
            elif stat.S_ISLNK(child_metadata.st_mode):
                byte_count += child_metadata.st_size
                if not allow_marked_entries:
                    raise RemovalRefusalError("path contains a symlink", byte_count)
            elif stat.S_ISDIR(child_metadata.st_mode):
                pending.append(child)
            else:
                raise RemovalRefusalError("path contains a special file", byte_count)
    return byte_count


def _check_owner(path: Path, metadata: os.stat_result, byte_count: int) -> None:
    if not owns_entry(path, metadata):
        raise RemovalRefusalError(
            "path contains an entry not owned by the current user", byte_count
        )


def assess_path(
    argument: str,
    repo_root: Path | None = None,
    var_tmp: Path | None = None,
    tmp_root: Path | None = None,
) -> Assessment:
    """Resolve and validate one requested path without changing the filesystem."""
    try:
        original_path = Path(argument).expanduser().absolute()
        metadata = os.lstat(str(original_path))
    except (OSError, ValueError) as error:
        return Assessment(argument, Path(argument), 0, f"path cannot be inspected: {error}")
    if stat.S_ISLNK(metadata.st_mode):
        return Assessment(argument, original_path, metadata.st_size, "path itself is a symlink")

    root_values = _roots(repo_root or REPO_ROOT, var_tmp or VAR_TMP, tmp_root or TMP)
    path = Path(os.path.realpath(str(original_path)))
    try:
        allowed_root_reason = _allowed_root(path, original_path, root_values, 0)
        if allowed_root_reason:
            byte_count = 0
        else:
            _check_keep_ancestors(path, original_path, 0)
            _check_worktree(path, root_values.repo_root, 0)
            # pytest scratch is test output only; its links are unlinked by rmtree, never followed.
            byte_count = _scan_tree(
                path,
                allow_marked_entries=_marked_root(path, 0)
                or is_within(path, root_values.pytest_root),
            )
    except RemovalRefusalError as error:
        return Assessment(argument, path, error.byte_count, error.reason)
    except (OSError, ValueError) as error:
        return Assessment(argument, path, 0, f"validation failed: {error}")
    return Assessment(argument, path, byte_count, allowed_root_reason)


def _refuse_overlapping_paths(assessments: list[Assessment]) -> list[Assessment]:
    """Refuse overlapping arguments so one deletion cannot consume another requested path."""
    refusals: dict[int, str] = {}
    for first_index, first in enumerate(assessments):
        if first.refusal:
            continue
        for second_index in range(first_index + 1, len(assessments)):
            second = assessments[second_index]
            if second.refusal:
                continue
            overlap = (
                first.path == second.path
                or first.path in second.path.parents
                or second.path in first.path.parents
            )
            if overlap:
                refusals[first_index] = "overlapping requested paths"
                refusals[second_index] = "overlapping requested paths"
    return [
        assessment._replace(refusal=refusals[index]) if index in refusals else assessment
        for index, assessment in enumerate(assessments)
    ]


def _write_record(assessment: Assessment, outcome: str, reason: str) -> None:
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 - Python 3.8
        "path": assessment.argument,
        "resolved_path": str(assessment.path),
        "bytes": assessment.byte_count,
        "outcome": outcome,
        "reason": reason,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(LOG_PATH), os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    LOGGER.info(json.dumps(record, sort_keys=True))


def _raise_rmtree_error(
    function: Callable[..., object],
    failed_path: str,
    error_info: tuple[type[BaseException], BaseException, TracebackType | None],
) -> None:
    del function, failed_path
    error = error_info[1]
    traceback = error_info[2]
    if traceback is None:
        raise error
    raise error.with_traceback(traceback)


def _delete(path: Path) -> None:
    metadata = os.lstat(str(path))
    if stat.S_ISLNK(metadata.st_mode):
        raise OSError("path became a symlink after validation")
    if not owns_entry(path, metadata) or _is_mount_point(path):
        raise OSError("path ownership or mount status changed after validation")
    if stat.S_ISDIR(metadata.st_mode):
        shutil.rmtree(str(path), onerror=_raise_rmtree_error)  # pyright: ignore[reportDeprecated]
    elif stat.S_ISREG(metadata.st_mode):
        path.unlink()
    else:
        raise OSError("path type changed after validation")


def main(argv: list[str] | None = None) -> int:
    """Validate each requested path, log its outcome, and remove only safe targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="check and log paths without deleting"
    )
    parser.add_argument("paths", nargs="+", help="paths to validate and delete")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    assessments = _refuse_overlapping_paths(
        [assess_path(path, REPO_ROOT, VAR_TMP, TMP) for path in arguments.paths]
    )
    failed = False
    for assessment in assessments:
        if assessment.refusal:
            _write_record(assessment, "refused", assessment.refusal)
            failed = True
            continue
        if arguments.dry_run:
            _write_record(assessment, "dry-run", "all safety checks passed; no change made")
            continue
        current = assess_path(assessment.argument, REPO_ROOT, VAR_TMP, TMP)
        if current.refusal:
            _write_record(current, "refused", current.refusal)
            failed = True
            continue
        if current.path != assessment.path or current.byte_count != assessment.byte_count:
            _write_record(current, "refused", "path changed after its initial safety check")
            failed = True
            continue
        _write_record(assessment, "deleting", "all safety checks passed; deletion starting")
        try:
            _delete(assessment.path)
        except OSError as error:
            _write_record(assessment, "failed", f"deletion failed: {error}")
            failed = True
            continue
        _write_record(assessment, "deleted", "removed after all safety checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
