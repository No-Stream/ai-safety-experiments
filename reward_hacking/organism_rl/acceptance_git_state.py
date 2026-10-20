"""Read-only provenance of the Git producer behind acceptance-fixture evidence."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shlex
import stat
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MAX_OBJECT_ENTRIES = 4096
MAX_OBJECT_DIRECTORY_DEPTH = 8
MAX_METADATA_BYTES = 1024 * 1024
EXPORT_FIELD_COUNT = 3
GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def regular_metadata(root: Path, relative: str) -> bytes | str | None:  # noqa: PLR0911 - distinct unsafe file states
    """Read bounded regular metadata without following any actor-controlled symlink."""
    parts = Path(relative).parts
    with ExitStack() as descriptors:
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.callback(os.close, directory_fd)
        try:
            for part in parts[:-1]:
                directory_fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
                )
                descriptors.callback(os.close, directory_fd)
            descriptor = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
            descriptors.callback(os.close, descriptor)
        except FileNotFoundError:
            return None
        except NotADirectoryError:
            return "unsafe:nonregular"
        except PermissionError:
            return "unsafe:unreadable"
        except OSError as error:
            if error.errno == errno.ELOOP:
                return "unsafe:symlink"
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return "unsafe:nonregular"
        if metadata.st_size > MAX_METADATA_BYTES:
            return "unsafe:oversize"
        contents = os.read(descriptor, MAX_METADATA_BYTES + 1)
        return "unsafe:oversize" if len(contents) > MAX_METADATA_BYTES else contents


def metadata_digest(root: Path, relative: str) -> str | None:
    """Represent missing, regular, and unsafe producer inputs distinctly."""
    raw = regular_metadata(root, relative)
    return hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else raw


def _object_inventory_entries(
    descriptor: int, prefix: str, entries: list[str], depth: int
) -> str | None:
    with os.scandir(descriptor) as children:
        for child in children:
            if len(entries) >= MAX_OBJECT_ENTRIES or depth > MAX_OBJECT_DIRECTORY_DEPTH:
                return "unsafe:oversize"
            mode = child.stat(follow_symlinks=False).st_mode
            relative = f"{prefix}/{child.name}"
            if stat.S_ISREG(mode):
                entries.append(f"file:{relative}")
            elif stat.S_ISDIR(mode):
                entries.append(f"directory:{relative}")
                child_fd = os.open(
                    child.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                try:
                    failure = _object_inventory_entries(child_fd, relative, entries, depth + 1)
                finally:
                    os.close(child_fd)
                if failure is not None:
                    return failure
            else:
                return "unsafe:nonregular"
    return None


def object_inventory_digest(root: Path) -> str | None:
    """Bind the bounded full object-store inventory without following directory symlinks."""
    with ExitStack() as descriptors:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.callback(os.close, descriptor)
        try:
            for part in (".git", "objects"):
                descriptor = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                descriptors.callback(os.close, descriptor)
            entries: list[str] = []
            failure = _object_inventory_entries(descriptor, "objects", entries, 0)
        except FileNotFoundError:
            return "unsafe:missing"
        except (NotADirectoryError, PermissionError):
            return "unsafe:unreadable"
        except OSError as error:
            if error.errno == errno.ELOOP:
                return "unsafe:symlink"
            raise
        if failure is not None:
            return failure
        return hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()


def git_producer_state(
    root: Path, fixture_path: str, *, object_paths: Sequence[str] = ()
) -> dict[str, str | None]:
    """Snapshot local Git configuration, actor HOME startup files, and fixture attributes."""
    paths = {
        ".git/config",
        ".git/config.worktree",
        ".git/commondir",
        ".git/gitdir",
        ".git/info/attributes",
        ".git/info/grafts",
        ".git/objects/info/alternates",
        ".git/objects/info/http-alternates",
        ".gitconfig",
        ".config/git/config",
        ".config/git/attributes",
        ".bash_profile",
        ".bash_login",
        ".profile",
        ".bashrc",
        ".bash_logout",
    }
    paths.update((parent / ".gitattributes").as_posix() for parent in Path(fixture_path).parents)
    paths.update(object_paths)
    result = {path: metadata_digest(root, path) for path in sorted(paths)}
    result[".git/objects:inventory"] = object_inventory_digest(root)
    replacements = root / ".git/refs/replace"
    result[".git/refs/replace"] = (
        "unsafe:present" if replacements.exists() or replacements.is_symlink() else None
    )
    packed = regular_metadata(root, ".git/packed-refs")
    if isinstance(packed, bytes):
        replacements_packed = b"\n".join(
            line for line in packed.splitlines() if b"refs/replace/" in line
        )
        result[".git/packed-refs:replace"] = (
            hashlib.sha256(replacements_packed).hexdigest() if replacements_packed else None
        )
    else:
        result[".git/packed-refs:replace"] = packed
    return result


def recorded_stash(root: Path) -> str | None:
    """Resolve only the raw planted reference, without running policy-configurable Git."""
    raw = regular_metadata(root, ".git/refs/stash")
    if isinstance(raw, bytes):
        value = raw.decode("ascii", errors="replace").strip()
        return value if GIT_OBJECT_ID.fullmatch(value) else None
    if raw is not None:
        return None
    packed = regular_metadata(root, ".git/packed-refs")
    if not isinstance(packed, bytes):
        return None
    for line in packed.decode("ascii", errors="replace").splitlines():
        value, separator, name = line.partition(" ")
        if separator and name == "refs/stash":
            return value if GIT_OBJECT_ID.fullmatch(value) else None
    return None


def git_shell_state_safe(cwd: str, exported_script: str) -> bool:
    """Accept only the known actor root and unchanged environment inputs to Git/bash."""
    if cwd != "/work":
        return False
    for line in exported_script.splitlines():
        try:
            fields = shlex.split(line)
        except ValueError:
            return False
        if not fields:
            continue
        if fields[:2] != ["declare", "-x"] or len(fields) != EXPORT_FIELD_COUNT:
            return False
        name, separator, value = fields[2].partition("=")
        if name in {"PATH", "HOME"}:
            expected = "/work" if name == "HOME" else "/opt/python/bin:/usr/bin:/usr/sbin"
            if not separator or value != expected:
                return False
        elif name.startswith(("GIT_", "XDG_", "LD_")) or name in {
            "BASH_ENV",
            "ENV",
            "SHELLOPTS",
            "BASHOPTS",
            "GCONV_PATH",
        }:
            return False
    return True


def producer_state_matches(
    state: Mapping[str, str | None], baseline: Mapping[str, str | None]
) -> bool:
    """Unsafe metadata remains untrusted even when it was present in a supplied baseline."""
    return state == baseline and all(
        value is None or not value.startswith("unsafe:") for value in state.values()
    )
