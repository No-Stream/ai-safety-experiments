"""Narrow shared-box deletion guards; every fixture path is temporary and owned by this test."""

# ruff: noqa: S108, S603 - exercise protected path names and run only the local entry point

from __future__ import annotations

import getpass
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pytest

from scripts import safe_rm

SYSTEM_PYTHON = Path("/usr/bin/python3")
REPO_ROOT = Path(__file__).resolve().parents[1]


class SafeScope(NamedTuple):
    repo: Path
    var_tmp: Path
    tmp_root: Path
    pytest_root: Path
    home: Path
    log: Path


@pytest.fixture
def safe_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SafeScope:
    repo = tmp_path / "repo"
    var_tmp = tmp_path / "var-tmp"
    tmp_root = tmp_path / "tmp"
    pytest_root = tmp_root / f"pytest-of-{getpass.getuser()}"
    home = tmp_path / "home"
    log = tmp_path / "logs" / "safe_rm.log"
    for directory in (repo / "artifacts", var_tmp, pytest_root, home):
        directory.mkdir(parents=True)
    monkeypatch.setattr(safe_rm, "REPO_ROOT", repo)
    monkeypatch.setattr(safe_rm, "VAR_TMP", var_tmp)
    monkeypatch.setattr(safe_rm, "TMP", tmp_root)
    monkeypatch.setattr(safe_rm, "LOG_PATH", log)
    monkeypatch.setattr(safe_rm, "registered_worktrees", lambda repo_root: [])
    return SafeScope(repo, var_tmp, tmp_root, pytest_root, home, log)


def _make_file(path: Path, contents: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _records(log: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def _assert_refused(scope: SafeScope, path: Path, reason: str) -> None:
    assert safe_rm.main([str(path)]) != 0
    assert path.exists() or path.is_symlink()
    record = _records(scope.log)[-1]
    assert record["path"] == str(path)
    assert record["outcome"] == "refused"
    assert reason in str(record["reason"])
    assert isinstance(record["bytes"], int)
    assert record["bytes"] >= 0
    assert "timestamp" in record


def test_refuses_var_tmp_root(safe_scope: SafeScope) -> None:
    _assert_refused(safe_scope, safe_scope.var_tmp, "allowlisted root itself")


def test_refuses_symlink_to_home_and_does_not_follow_it(safe_scope: SafeScope) -> None:
    protected_file = _make_file(safe_scope.home / "valuable.bin", b"keep")
    escaping_link = safe_scope.var_tmp / "escape"
    escaping_link.symlink_to(safe_scope.home, target_is_directory=True)

    _assert_refused(safe_scope, escaping_link, "symlink")

    assert protected_file.read_bytes() == b"keep"


def test_refuses_realpath_outside_var_tmp_through_parent_symlink(safe_scope: SafeScope) -> None:
    protected_file = _make_file(safe_scope.home / "valuable.bin")
    (safe_scope.var_tmp / "escape").symlink_to(safe_scope.home, target_is_directory=True)

    _assert_refused(safe_scope, safe_scope.var_tmp / "escape" / protected_file.name, "allowlist")

    assert protected_file.exists()


def test_refuses_s3_mirror_alias_even_when_realpath_is_an_allowed_sibling(
    safe_scope: SafeScope,
) -> None:
    protected_file = _make_file(safe_scope.var_tmp / "mirror-target" / "valuable.bin")
    (safe_scope.var_tmp / "s3-mirror").symlink_to(
        safe_scope.var_tmp / "mirror-target", target_is_directory=True
    )

    _assert_refused(
        safe_scope,
        safe_scope.var_tmp / "s3-mirror" / protected_file.name,
        "protected path",
    )

    assert protected_file.exists()


def test_refuses_symlink_to_an_allowed_var_tmp_sibling(safe_scope: SafeScope) -> None:
    protected_file = _make_file(safe_scope.var_tmp / "target" / "valuable.bin")
    link = safe_scope.var_tmp / "alias"
    link.symlink_to(protected_file.parent, target_is_directory=True)

    _assert_refused(safe_scope, link, "symlink")

    assert protected_file.exists()


def test_refuses_s3_mirror_tree(safe_scope: SafeScope) -> None:
    candidate = _make_file(safe_scope.var_tmp / "s3-mirror" / "run" / "trace.jsonl")

    _assert_refused(safe_scope, candidate.parents[1], "protected path")


@pytest.mark.parametrize(
    ("relative_path", "reason"),
    [
        ("checkpoint.lock", "/var/tmp/*.lock"),
        ("cpython-runtime", "/var/tmp/cpython-runtime"),
        ("flashinfer-cache", "/var/tmp/flashinfer-*"),
    ],
)
def test_refuses_other_explicit_var_tmp_exclusions(
    safe_scope: SafeScope, relative_path: str, reason: str
) -> None:
    if relative_path.endswith(".lock"):
        candidate = _make_file(safe_scope.var_tmp / relative_path)
    else:
        candidate = _make_file(safe_scope.var_tmp / relative_path / "payload.bin").parent

    _assert_refused(safe_scope, candidate, reason)


def test_refuses_keep_marker_in_ancestor(safe_scope: SafeScope) -> None:
    protected_directory = safe_scope.var_tmp / "keep-parent"
    _make_file(protected_directory / ".keep", b"")
    candidate = _make_file(protected_directory / "nested" / "result.bin")

    _assert_refused(safe_scope, candidate.parent, ".keep")


def test_refuses_keep_marker_in_descendant_without_partial_deletion(safe_scope: SafeScope) -> None:
    candidate = safe_scope.var_tmp / "kept-tree"
    marker = _make_file(candidate / "nested" / ".keep", b"")
    result = _make_file(candidate / "result.bin")

    _assert_refused(safe_scope, candidate, ".keep")

    assert marker.exists()
    assert result.exists()


def test_refuses_registered_worktree(
    safe_scope: SafeScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = safe_scope.var_tmp / "worktree"
    result = _make_file(candidate / "tracked-like-file")
    monkeypatch.setattr(safe_rm, "registered_worktrees", lambda repo_root: [candidate.resolve()])

    _assert_refused(safe_scope, candidate, "registered git worktree")

    assert result.exists()


def test_registered_worktree_list_reads_the_repository() -> None:
    assert REPO_ROOT.resolve() in safe_rm.registered_worktrees(REPO_ROOT)


def test_refuses_git_metadata_inside_tree(safe_scope: SafeScope) -> None:
    candidate = safe_scope.var_tmp / "repository-copy"
    metadata = _make_file(candidate / ".git" / "config")

    _assert_refused(safe_scope, candidate, ".git")

    assert metadata.exists()


@pytest.mark.parametrize(
    ("marker_name", "marker_contents"),
    [
        (".ttl", b'{"expires":"2000-01-01","created_by":"test","purpose":"cleanup acceptance"}'),
        (".ttl", b'{"expires":"2999-12-31","created_by":"test","purpose":"cleanup acceptance"}'),
        (".disposable", b""),
    ],
)
def test_deletes_marked_tree_with_git_metadata_and_symlinks_without_following_them(
    safe_scope: SafeScope, marker_name: str, marker_contents: bytes
) -> None:
    candidate = safe_scope.var_tmp / "marked-repository"
    internal_target = _make_file(candidate / "internal" / "payload.bin", b"internal")
    external_target = _make_file(safe_scope.home / "external-payload.bin", b"external")
    _make_file(candidate / ".git" / "config", b"git metadata")
    (candidate / "internal-link").symlink_to(internal_target.parent, target_is_directory=True)
    (candidate / "external-link").symlink_to(external_target)
    _make_file(candidate / marker_name, marker_contents)

    assert safe_rm.main([str(candidate)]) == 0

    assert not candidate.exists()
    assert not internal_target.exists()
    assert external_target.read_bytes() == b"external"


def test_deletes_pytest_run_with_symlinks_without_following_them(safe_scope: SafeScope) -> None:
    candidate = safe_scope.pytest_root / "pytest-42"
    external_target = _make_file(safe_scope.home / "external-payload.bin", b"external")
    _make_file(candidate / "test_case0" / "repo" / ".git" / "config", b"git metadata")
    (candidate / "test_case0" / "venv-link").symlink_to(external_target)
    (candidate / "test_case0-current").symlink_to(
        candidate / "test_case0", target_is_directory=True
    )

    assert safe_rm.main([str(candidate)]) == 0

    assert not candidate.exists()
    assert external_target.read_bytes() == b"external"


def test_registered_worktree_stays_protected_when_marked(
    safe_scope: SafeScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = safe_scope.var_tmp / "registered-worktree"
    _make_file(candidate / ".git" / "config")
    _make_file(candidate / ".ttl", b"")
    monkeypatch.setattr(safe_rm, "registered_worktrees", lambda repo_root: [candidate.resolve()])

    _assert_refused(safe_scope, candidate, "registered git worktree")

    assert candidate.exists()


def test_refuses_unmarked_artifact_directory(safe_scope: SafeScope) -> None:
    candidate = _make_file(safe_scope.repo / "artifacts" / "unmarked" / "result.bin")

    _assert_refused(safe_scope, candidate.parent, "outside deletion allowlist")


def test_refuses_foreign_owned_entry_when_owner_check_reports_it(
    safe_scope: SafeScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _make_file(safe_scope.var_tmp / "foreign.bin")
    monkeypatch.setattr(safe_rm, "owns_entry", lambda path, metadata: False)

    _assert_refused(safe_scope, candidate, "current user")


def test_refuses_foreign_owned_descendant(
    safe_scope: SafeScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = safe_scope.var_tmp / "mixed-ownership"
    foreign_file = _make_file(candidate / "foreign.bin")
    monkeypatch.setattr(
        safe_rm,
        "owns_entry",
        lambda path, metadata: path != foreign_file,
    )

    _assert_refused(safe_scope, candidate, "current user")
    assert foreign_file.exists()


def test_refuses_mount_point(safe_scope: SafeScope, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _make_file(safe_scope.var_tmp / "mounted.bin")
    monkeypatch.setattr(
        safe_rm.os.path,
        "ismount",
        lambda path: Path(path) == candidate,
    )

    _assert_refused(safe_scope, candidate, "mount point")


def test_refuses_mount_point_inside_tree(
    safe_scope: SafeScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = safe_scope.var_tmp / "tree-with-mount"
    mounted_directory = candidate / "mounted"
    _make_file(mounted_directory / "payload.bin")
    monkeypatch.setattr(
        safe_rm.os.path,
        "ismount",
        lambda path: Path(path) == mounted_directory,
    )

    _assert_refused(safe_scope, candidate, "mount point")
    assert (mounted_directory / "payload.bin").exists()


def test_deletes_allowed_tree_and_logs_path_bytes_reason_and_timestamp(
    safe_scope: SafeScope,
) -> None:
    candidate = safe_scope.var_tmp / "old-run"
    payload = _make_file(candidate / "result.bin", b"known bytes")

    assert safe_rm.main([str(candidate)]) == 0

    assert not candidate.exists()
    record = _records(safe_scope.log)[-1]
    assert record["path"] == str(candidate)
    assert record["bytes"] == len(b"known bytes")
    assert record["outcome"] == "deleted"
    assert record["reason"] == "removed after all safety checks passed"
    datetime.fromisoformat(str(record["timestamp"]))
    assert not payload.exists()


def test_dry_run_logs_and_keeps_allowed_file(safe_scope: SafeScope) -> None:
    candidate = _make_file(safe_scope.var_tmp / "inspect-only.bin")

    assert safe_rm.main(["--dry-run", str(candidate)]) == 0

    assert candidate.exists()
    assert _records(safe_scope.log)[-1]["outcome"] == "dry-run"


def test_deletes_single_file_under_marked_artifact_tree(safe_scope: SafeScope) -> None:
    artifact_run = safe_scope.repo / "artifacts" / "smoke-run"
    _make_file(artifact_run / ".disposable", b"")
    candidate = _make_file(artifact_run / "result.bin")

    assert safe_rm.main([str(candidate)]) == 0

    assert not candidate.exists()
    assert artifact_run.exists()
    assert _records(safe_scope.log)[-1]["bytes"] == len(b"data")


def test_refuses_overlapping_arguments_before_deleting_either(safe_scope: SafeScope) -> None:
    parent = safe_scope.var_tmp / "overlap"
    child = _make_file(parent / "nested" / "result.bin")

    assert safe_rm.main([str(parent), str(child.parent)]) != 0

    assert child.exists()
    records = _records(safe_scope.log)
    assert [record["outcome"] for record in records] == ["refused", "refused"]
    assert all(record["reason"] == "overlapping requested paths" for record in records)


@pytest.mark.skipif(not SYSTEM_PYTHON.exists(), reason=f"no system interpreter at {SYSTEM_PYTHON}")
def test_safe_rm_executes_under_system_python_on_a_tmp_path(tmp_path: Path) -> None:
    candidate = _make_file(tmp_path / "safe-rm-probe.bin")
    home = tmp_path / "script-home"
    home.mkdir()
    environment = {**os.environ, "HOME": str(home), "USER": getpass.getuser()}

    completed = subprocess.run(
        [
            str(SYSTEM_PYTHON),
            str(REPO_ROOT / "scripts" / "safe_rm.py"),
            "--dry-run",
            str(candidate),
        ],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert "dry-run" in completed.stdout
    assert candidate.exists()
    assert (home / "logs" / "safe_rm.log").is_file()
