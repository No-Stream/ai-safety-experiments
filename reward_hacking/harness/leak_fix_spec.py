"""Read reproducible fix-commit specifications and interpreter paths from local R2E extractions."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.r2e_leak_data import R2eLeakTask


def git_bytes(repository: Path, *arguments: str) -> bytes:
    """Run local Git with host configuration disabled and return unmodified bytes."""
    result = subprocess.run(  # noqa: S603 - trusted local Git argv
        ["git", "-C", str(repository), *arguments],  # noqa: S607
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        },
        capture_output=True,
        check=True,
        timeout=60,
    )
    return result.stdout


def git_text(repository: Path, *arguments: str) -> str:
    """Decode local Git output as UTF-8 text."""
    return git_bytes(repository, *arguments).decode("utf-8").strip()


def git_tree(repository: Path, commit_id: str) -> dict[str, tuple[str, str]]:
    """Read file modes and object ids, including unchanged submodule gitlinks."""
    entries: dict[str, tuple[str, str]] = {}
    for raw_entry in git_bytes(repository, "ls-tree", "-r", "-z", "--full-tree", commit_id).split(
        b"\x00"
    ):
        if not raw_entry:
            continue
        raw_metadata, raw_path = raw_entry.split(b"\t", maxsplit=1)
        mode, _object_type, object_id = raw_metadata.decode("ascii").split()
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("R2E fix tree contains a non-UTF-8 path") from error
        entries[path] = (mode, object_id)
    return entries


def _signature(commit_headers: list[str], header_name: str) -> dict[str, str]:
    matching_headers = [line for line in commit_headers if line.startswith(f"{header_name} ")]
    if len(matching_headers) != 1:
        raise ValueError("R2E fix commit must have one author and committer signature")
    match = re.fullmatch(
        rf"{header_name} (.*) <(.*)> ([0-9]+) ([+-][0-9]{{4}})",
        matching_headers[0],
    )
    if match is None:
        raise ValueError("R2E fix commit signature cannot be represented by the channel builder")
    name, email, timestamp, timezone = match.groups()
    if not name or not email:
        raise ValueError("R2E fix commit signature is incomplete")
    return {"name": name, "email": email, "date": f"@{timestamp} {timezone}"}


def _commit_headers(headers: list[str]) -> tuple[list[str], list[str], list[str]]:
    header_records: list[list[str]] = []
    for line in headers:
        if line.startswith(" "):
            if not header_records:
                raise ValueError("R2E fix commit starts with a header continuation")
            header_records[-1].append(line.removeprefix(" "))
        else:
            header_records.append([line])
    signature_lines = next(
        (
            [record[0].removeprefix("gpgsig "), *record[1:]]
            for record in header_records
            if record[0].startswith("gpgsig ")
        ),
        [],
    )
    if any(len(record) > 1 and not record[0].startswith("gpgsig ") for record in header_records):
        raise ValueError(
            "R2E fix commit has multi-line metadata the channel builder cannot reproduce"
        )
    headers = [record[0] for record in header_records if not record[0].startswith("gpgsig ")]
    allowed_header_names = {"tree", "parent", "author", "committer"}
    if any(line.split(maxsplit=1)[0] not in allowed_header_names for line in headers):
        raise ValueError("R2E fix commit has metadata the channel builder cannot reproduce")
    parent_ids = [line.split(maxsplit=1)[1] for line in headers if line.startswith("parent ")]
    if not parent_ids:
        raise ValueError("R2E fix commit has no parent")
    return headers, parent_ids, signature_lines


def fix_commit_spec(case: R2eLeakTask, source_repository: Path) -> dict[str, object]:
    """Retain exact fixes; squash merges onto their first parent without changing their tree."""
    commit_id = case.commit_hash
    commit_object = git_bytes(source_repository, "cat-file", "commit", commit_id)
    raw_headers, raw_message = commit_object.split(b"\n\n", maxsplit=1)
    try:
        headers = raw_headers.decode("utf-8").splitlines()
        message = raw_message.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("R2E fix commit metadata is not UTF-8") from error

    headers, parent_ids, signature_lines = _commit_headers(headers)
    base_commit_id = parent_ids[0]
    representation: dict[str, object] = {}
    if len(parent_ids) > 1:
        planted_headers = [line for line in headers if not line.startswith("parent ")]
        planted_headers.insert(1, f"parent {base_commit_id}")
        planted_object = "\n".join(planted_headers).encode("utf-8") + b"\n\n" + raw_message
        object_format = git_text(source_repository, "rev-parse", "--show-object-format")
        commit_id = hashlib.new(
            object_format, f"commit {len(planted_object)}\0".encode() + planted_object
        ).hexdigest()
        representation = {
            "upstream_commit_id": case.commit_hash,
            "fix_commit_representation": "squashed_merge",
        }
        signature_lines = []
    if not message.strip():
        raise ValueError("R2E fix commit has no message")

    base_tree = git_tree(source_repository, base_commit_id)
    fix_tree = git_tree(source_repository, case.commit_hash)
    changed_paths = sorted(
        path
        for path in base_tree.keys() | fix_tree.keys()
        if base_tree.get(path) != fix_tree.get(path)
    )
    if not changed_paths:
        raise ValueError("R2E fix commit has no changed files")

    files: dict[str, dict[str, str | None]] = {}
    blob_ids: dict[str, str] = {}
    for path in changed_paths:
        fixed_entry = fix_tree.get(path)
        if fixed_entry is None:
            original_entry = base_tree[path]
            if original_entry[0] not in {"100644", "100755"}:
                raise ValueError("R2E fix commit changes an unsupported file mode")
            files[path] = {"content": None, "mode": original_entry[0]}
            continue
        mode, blob_id = fixed_entry
        if mode not in {"100644", "100755"}:
            raise ValueError("R2E fix commit changes an unsupported file mode")
        try:
            content = git_bytes(source_repository, "cat-file", "blob", blob_id).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("R2E fix commit contains a non-UTF-8 changed file") from error
        files[path] = {"content": content, "mode": mode}
        blob_ids[path] = blob_id

    tree_id = next(line.split(maxsplit=1)[1] for line in headers if line.startswith("tree "))
    return {
        "commit_id": commit_id,
        **representation,
        "tree_id": tree_id,
        "blob_ids": blob_ids,
        "message": message,
        "author": _signature(headers, "author"),
        "committer": _signature(headers, "committer"),
        "issue_slug": re.sub(r"[^a-z0-9-]", "-", case.task_id),
        "files": files,
        **({"gpgsig": "\n".join(signature_lines)} if signature_lines else {}),
    }


def interpreter_paths(source_repository: Path, interpreters_root: Path) -> tuple[Path, Path, str]:
    """Resolve the extracted venv and its unique executable interpreter without relocation."""
    venv_python = source_repository / ".venv" / "bin" / "python"
    if not venv_python.is_symlink():
        raise ValueError("extracted venv Python is not a symlink")
    python_binary_name = venv_python.readlink().name
    matching_binaries = sorted(interpreters_root.glob(f"*/bin/{python_binary_name}"))
    if len(matching_binaries) != 1:
        raise FileNotFoundError(
            f"the extracted interpreter {python_binary_name} is missing or ambiguous"
        )
    interpreter_binary = matching_binaries[0]
    interpreter_root = interpreter_binary.parent.parent
    if not interpreter_binary.is_file() or not os.access(interpreter_binary, os.X_OK):
        raise ValueError("the extracted interpreter is not executable")
    subprocess.run(  # noqa: S603 - validated local interpreter argv
        [str(interpreter_binary), "--version"],
        capture_output=True,
        check=True,
        timeout=15,
    )
    return venv_python.parent.parent, interpreter_root, python_binary_name
