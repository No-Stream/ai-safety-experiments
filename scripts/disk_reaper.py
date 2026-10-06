#!/usr/bin/python3
"""Conservative, marker-based cleanup. Default invocation only reports candidates."""

import argparse
import contextlib
import datetime
import fcntl
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
TEMPORARY = Path("/var/tmp")
OVERFLOW = Path("/mnt/d/ml-exp-overflow")
GIB = 1024**3
RECENTLY_MODIFIED = "modified within 6 hours"
REPORT_FLOOR_GIB = 1
# rule: (action, reason), in the order choose_rule tries them. Names never grant deletion.
RULES = {
    "compiler-cache": ("MOVE", "~/.cache torch/inductor or triton untouched for 14 days"),
    "disposable-marker": ("REMOVE", ".disposable marker older than 24 hours"),
    "tool-cache": ("REMOVE", "~/.cache uv or pip, emptied by the tool's own cleaner"),
    "tool-cache-no-cleaner": ("REPORT ONLY", "uv or pip cache whose cleaner is not on PATH"),
    "repo-cache": ("REMOVE", "__pycache__, .pytest_cache or .ruff_cache under a workspace repo"),
    "gate-worktree": ("REPORT ONLY", "/var/tmp/gate-* worktree: owner runs git worktree remove"),
    "old-log": ("REMOVE", "*.log under /var/tmp or ~/logs older than 14 days"),
    "overflow-expiry": ("REMOVE", "overflow YYYY-MM-DD directory older than 30 days"),
    "research-data": ("REPORT ONLY", "run data or hub model over 1 GiB: owner review"),
    "unmatched": ("REPORT ONLY", "unmatched data over 1 GiB: owner review"),
}


def roots(home: Path, temporary: Path = TEMPORARY, overflow: Path = OVERFLOW) -> List[Path]:
    """Return explicit action roots, never the workspace as a whole."""
    return [
        *(home / "workspace").glob("*/artifacts"),
        home / ".cache",
        home / "logs",
        temporary,
        overflow,
    ]


def inventory(path: Path) -> Tuple[int, float, List[Path]]:
    """Count regular files and links without following links; reject special files."""
    entries = [path]
    if path.is_dir() and not path.is_symlink():
        for directory, folders, files in os.walk(
            str(path), followlinks=False, onerror=raise_walk_error
        ):
            entries.extend(Path(directory) / name for name in folders + files)
    total, newest = 0, 0.0
    for entry in entries:
        metadata = entry.lstat()
        if not (
            stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        ):
            message = "special file: {}".format(entry)
            raise ValueError(message)
        total += 0 if stat.S_ISDIR(metadata.st_mode) else metadata.st_size
        newest = max(newest, metadata.st_mtime)
    return total, newest, entries


def raise_walk_error(error: OSError) -> None:
    """Make os.walk permission failures visible rather than silently incomplete."""
    raise error


def open_handles(path: Path) -> str:
    """Any diagnostic or timeout makes the handle check inconclusive and blocks cleanup."""
    lsof = shutil.which("lsof")
    tool = lsof or shutil.which("fuser")
    if tool is None:
        return "neither lsof nor fuser available"
    # fuser is not recursive: inspect every directory and file separately.
    entries = [path] if lsof else inventory(path)[2]
    for entry in entries:
        command = [tool, "--", str(entry)]
        if lsof:
            command = (
                [tool, "-t", "+D", str(path)] if path.is_dir() else [tool, "-t", "--", str(path)]
            )
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=10
            )
        except subprocess.TimeoutExpired:
            return "open-handle check timed out"
        if result.stderr or result.returncode not in (0, 1):
            return "open-handle check unavailable: " + result.stderr.strip()
        if result.stdout:
            return "open file handles"
    return ""


def present(path: Path) -> bool:
    """Unlike exists(), propagate access failures and recognize dangling protection markers."""
    try:
        path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def protect(path: Path, allowed: List[Path], now: float, *, check_handles: bool = True) -> str:  # noqa: C901, PLR0911, PLR0912 - linear safety checks
    """Fail closed on scope, markers, learned assets, tracking and active use."""
    if not any(root in path.parents for root in allowed):
        return "outside allowlist (or root itself)"
    if path.resolve() != path.absolute():
        return "symlink in path"
    forbidden = [
        "docs",
        ".claude",
        ".ssh",
        ".codex",
        "cpython-runtime",
        "s3-mirror",
        "atlas-train-extra-deps",
    ]
    if any(part in forbidden for part in path.parts):
        return "hard protected path"
    _, newest, entries = inventory(path)
    if any(present(parent / ".keep") for parent in [path, *path.parents]) or any(
        entry.name == ".keep" for entry in entries
    ):
        return ".keep protection"
    if any(
        entry.name == "adapter_config.json"
        or entry.name.startswith("adapter_model.")
        or entry.name.startswith("base_results_and_detectors_")
        for entry in entries
    ):
        return "adapter or trained probe protection"
    # A child of a learned-asset directory is protected too.
    for parent in path.parents:
        if (
            present(parent / "adapter_config.json")
            or any(child.name.startswith("adapter_model.") for child in parent.iterdir())
            or parent.name.startswith("base_results_and_detectors_")
        ):
            return "adapter or trained probe protection"
    if any(entry.is_symlink() for entry in entries):
        return "tree contains symlinks (owner review required)"
    if newest > now - 6 * 3600:
        return RECENTLY_MODIFIED
    if any(entry.name == ".git" for entry in entries):
        return "git repository metadata (owner cleanup required)"
    repository = next((parent for parent in path.parents if present(parent / ".git")), None)
    if repository is not None:
        result = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z", "--", str(path)],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            return "git tracking check unavailable"
        if result.stdout:
            return "git-tracked files"
    return open_handles(path) if check_handles else ""


def choose_rule(  # noqa: PLR0911 - linear rule list
    path: Path, home: Path, temporary: Path, overflow: Path, now: float
) -> str:
    """Return the name of the first rule in RULES that applies to path."""
    newest = inventory(path)[1]
    cold = newest < now - 14 * 86400
    if path in (home / ".cache/torch/inductor", home / ".cache/triton") and cold:
        return "compiler-cache"
    marker = path / ".disposable"
    if marker.is_file() and marker.stat().st_mtime < now - 86400:
        return "disposable-marker"
    if path in (home / ".cache/uv", home / ".cache/pip"):
        return "tool-cache" if shutil.which(path.name) else "tool-cache-no-cleaner"
    caches = ("__pycache__", ".pytest_cache", ".ruff_cache")
    if path.name in caches and home / "workspace" in path.parents:
        return "repo-cache"
    if path.parent == temporary and path.name.startswith("gate-") and path.is_dir():
        return "gate-worktree"
    log_root = temporary in path.parents or home / "logs" in path.parents
    if path.suffix == ".log" and log_root and cold:
        return "old-log"
    if path.parent == overflow and overflow_date(path) < now - 30 * 86400:
        return "overflow-expiry"
    if "artifacts" in path.parts or path.name.startswith("models--"):
        return "research-data"
    return "unmatched"


def overflow_date(path: Path) -> float:
    """Timestamp of a YYYY-MM-DD overflow folder name; infinity for any other name."""
    try:
        date = datetime.datetime.strptime(path.name, "%Y-%m-%d").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        return float("inf")
    return date.timestamp()


def execute(  # noqa: C901, PLR0911, PLR0913, PLR0917 - explicit action inputs
    path: Path,
    rule: str,
    home: Path,
    overflow: Path,
    ledger: Path,
    allowed: List[Path],
    now: float,
) -> str:
    """Recheck protections at action time and act; return why it skipped, or "" if it acted."""
    action = RULES[rule][0]
    reason = protect(path, allowed, now)
    if reason or action == "REPORT ONLY":
        return reason
    if (action == "MOVE" or overflow in path.parents) and not os.path.ismount(overflow.parent):
        return "overflow drive not mounted"
    ledger_descriptor = os.open(
        str(ledger), os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(ledger_descriptor, "a") as stream:
        size, newest, entries = inventory(path)
        destination = None
        if action == "MOVE":
            if any(present(parent / ".keep") for parent in [overflow, *overflow.parents]):
                return ".keep protects overflow"
            if not overflow.is_dir():
                return "overflow directory missing"
            date = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date().isoformat()
            destination = overflow / date / path.relative_to(home)
            if present(destination) or any(
                parent.is_symlink() for parent in [destination, *destination.parents]
            ):
                return "overflow destination exists or has symlink ancestor"
            destination.mkdir(parents=True)
            rsync = ["rsync", "-a", "--", str(path) + "/", str(destination) + "/"]
            subprocess.run(rsync, check=True)
            if inventory(destination)[0] != size or inventory(path) != (size, newest, entries):
                raise RuntimeError("copy byte totals differ or source changed; source retained")
            reason = protect(path, allowed, now)
            if reason:
                return reason
        record = {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "path": str(path),
            "bytes": size,
            "rule": rule,
            "action": action,
            "destination": str(destination) if destination else None,
            "status": "intent",
        }
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        if rule == "tool-cache":
            command = (
                ["uv", "cache", "prune", "--cache-dir", str(path)]
                if path.name == "uv"
                else ["pip", "cache", "purge"]
            )
            subprocess.run(
                command,
                check=True,
                env=dict(os.environ, PIP_CACHE_DIR=str(path)),
                stdout=subprocess.DEVNULL,
            )
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        record["status"] = "complete"
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return ""


def run(home: Path, temporary: Path, overflow: Path, ledger: Path, *, apply: bool = False) -> None:
    """Report disjoint candidates and apply only eligible actions."""
    now = time.time()
    allowed = roots(home, temporary, overflow)
    pending = [child for root in allowed if root.is_dir() for child in root.iterdir()]
    totals = {"MOVE": 0, "REMOVE": 0, "REPORT ONLY": 0}
    LOGGER.info("path | size | rule | action | reason")
    while pending:
        path = pending.pop()
        try:
            size, _, entries = inventory(path)
            rule = choose_rule(path, home, temporary, overflow, now)
            action, reason = RULES[rule]
            blocked = protect(path, allowed, now, check_handles=action != "REPORT ONLY")
            if blocked:
                action, reason = "REPORT ONLY", blocked
            # Generic rules only report; look inside small or marker-holding trees for children
            # that a specific rule acts on, and give a row only to what is large.
            if rule in ("unmatched", "research-data"):
                small = size < REPORT_FLOOR_GIB * GIB
                holds_marked_child = blocked in ("", RECENTLY_MODIFIED) and any(
                    entry.name == ".disposable" and entry.parent != path for entry in entries
                )
                if path.is_dir() and (holds_marked_child or (small and not blocked)):
                    pending.extend(path.iterdir())
                    continue
                if small:
                    continue
        except (FileNotFoundError, PermissionError, ValueError) as error:
            LOGGER.info("%s | unknown | inspection | REPORT ONLY | %s", path, error)
            continue
        if apply and action != "REPORT ONLY":
            try:
                skipped = execute(path, rule, home, overflow, ledger, allowed, now)
            except FileNotFoundError as error:
                LOGGER.info("%s | unknown | action | REPORT ONLY | path vanished: %s", path, error)
                continue
            if skipped:
                action, reason = "REPORT ONLY", skipped
        LOGGER.info(
            "%s | %.3f GiB (%d B) | %s | %s | %s", path, size / GIB, size, rule, action, reason
        )
        totals[action] += size
    LOGGER.info(
        "Totals (candidate bytes, cleaners may reclaim less): %s",
        ", ".join("{}={:.3f} GiB".format(action, size / GIB) for action, size in totals.items()),
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Serialize apply invocations and stop early above the free-space threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--if-free-below-gb", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if (
        args.if_free_below_gb is not None
        and shutil.disk_usage("/").free / GIB > args.if_free_below_gb
    ):
        LOGGER.info("Free space exceeds threshold; no scan or actions.")
        return 0
    with contextlib.ExitStack() as stack:
        if args.apply:
            descriptor = os.open(
                str(TEMPORARY / "disk-reaper.lock"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            lock = stack.enter_context(os.fdopen(descriptor, "a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        run(
            Path.home(),
            TEMPORARY,
            OVERFLOW,
            TEMPORARY / "disk-reaper-ledger.jsonl",
            apply=args.apply,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
