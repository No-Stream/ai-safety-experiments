from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.harness.leak_channels import (
    CHANNELS,
    BuildRequest,
    build_channel_templates,
    probe_closed,
    probe_open,
    template_fingerprint,
)

if TYPE_CHECKING:
    from pathlib import Path


FIX_EPOCH = 1_700_000_100
BASE_EPOCH = 1_700_000_000
CODE_PATH = "src/example.py"
TEST_PATH = "tests/test_example.py"
BASE_CODE = "VALUE = 'base-value'\n"
FIXED_CODE = (
    "VALUE = 'distinctive-fixed-value'\n\n"
    "def useful_fix_marker() -> str:\n"
    "    return 'marker-content-long-enough'\n"
)
FIXED_TEST = "def test_example() -> None:\n    assert True\n"


def _git_executable() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("git executable is required by the synthetic repository tests")
    return git_executable


def _git(repository: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    completed = subprocess.run(  # noqa: S603 - fixed git executable and argv; no shell is involved
        [_git_executable(), "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=process_env,
    )
    return completed.stdout.strip()


def _commit_env(epoch: int) -> dict[str, str]:
    date = f"@{epoch} +0000"
    return {
        "GIT_AUTHOR_NAME": "Synthetic Author",
        "GIT_AUTHOR_EMAIL": "author@example.invalid",
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_NAME": "Synthetic Committer",
        "GIT_COMMITTER_EMAIL": "committer@example.invalid",
        "GIT_COMMITTER_DATE": date,
    }


def _make_synthetic_repo(root: Path, object_format: str = "sha1") -> tuple[Path, dict[str, Any]]:
    source = root / "source"
    source.mkdir()
    init_arguments = ["init", "--initial-branch=main"]
    if object_format != "sha1":
        init_arguments.append(f"--object-format={object_format}")
    _git(root, *init_arguments, str(source))
    _git(source, "config", "user.name", "Synthetic Author")
    _git(source, "config", "user.email", "author@example.invalid")
    (source / "src").mkdir()
    (source / "tests").mkdir()
    (source / CODE_PATH).write_text(BASE_CODE, encoding="utf-8")
    (source / TEST_PATH).write_text(
        "def test_example() -> None:\n    assert True\n", encoding="utf-8"
    )
    _git(source, "add", ".")
    _git(source, "commit", "-m", "Synthetic base", env=_commit_env(BASE_EPOCH))

    base = root / "base"
    subprocess.run(  # noqa: S603 - fixed git executable and argv; no shell is involved
        [_git_executable(), "clone", "--no-hardlinks", str(source), str(base)],
        check=True,
        capture_output=True,
    )

    (source / CODE_PATH).write_text(FIXED_CODE, encoding="utf-8")
    (source / TEST_PATH).write_text(FIXED_TEST, encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "Synthetic reference fix", env=_commit_env(FIX_EPOCH))
    fix_commit = _git(source, "rev-parse", "HEAD")
    fix_tree = _git(source, "show", "-s", "--format=%T", fix_commit)
    blob_ids = {
        CODE_PATH: _git(source, "rev-parse", f"{fix_commit}:{CODE_PATH}"),
        TEST_PATH: _git(source, "rev-parse", f"{fix_commit}:{TEST_PATH}"),
    }
    fix_commit_spec: dict[str, Any] = {
        "commit_id": fix_commit,
        "tree_id": fix_tree,
        "blob_ids": blob_ids,
        "message": "Synthetic reference fix",
        "author": {
            "name": _git(source, "show", "-s", "--format=%an", fix_commit),
            "email": _git(source, "show", "-s", "--format=%ae", fix_commit),
            "date": _git(source, "show", "-s", "--format=%aD", fix_commit),
        },
        "committer": {
            "name": _git(source, "show", "-s", "--format=%cn", fix_commit),
            "email": _git(source, "show", "-s", "--format=%ce", fix_commit),
            "date": _git(source, "show", "-s", "--format=%cD", fix_commit),
        },
        "issue_slug": "synthetic-fix",
        "files": {
            CODE_PATH: {"content": FIXED_CODE, "mode": "100644"},
            TEST_PATH: {"content": FIXED_TEST, "mode": "100644"},
        },
    }
    return base, fix_commit_spec


def _artifact_paths(template_dir: Path, manifest: dict[str, Any], channel: str) -> list[Path]:
    relative_paths = manifest["channel_details"][channel]["artifact_paths"]
    return [template_dir / relative_path for relative_path in relative_paths]


def _build_templates(
    request: BuildRequest,
) -> dict[str, Any]:
    return build_channel_templates(request)


def _request(
    base: Path, fix_commit_spec: dict[str, Any], channels: list[str], output_dir: Path
) -> BuildRequest:
    return BuildRequest(
        base_dir=base,
        fix_commit_spec=fix_commit_spec,
        channels=tuple(channels),
        out_dir=output_dir,
    )


def _delete_channel_artifact(template_dir: Path, manifest: dict[str, Any], channel: str) -> None:
    if channel == "mtime":
        timestamp_ns = manifest["fix"]["base_date"] * 1_000_000_000
        for path in _artifact_paths(template_dir, manifest, channel):
            os.utime(path, ns=(timestamp_ns, timestamp_ns))
        return
    removed = False
    for path in _artifact_paths(template_dir, manifest, channel):
        if path.is_file():
            path.unlink()
            removed = True
    assert removed


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    return _make_synthetic_repo(tmp_path)


@pytest.mark.parametrize(
    ("channel", "patch_location"),
    [(channel, "root") for channel in CHANNELS if channel != "patch_file"]
    + [("patch_file", "root"), ("patch_file", "dotgit")],
)
def test_each_channel_open_closed_and_sabotage(
    tmp_path: Path,
    synthetic_repo: tuple[Path, dict[str, Any]],
    channel: str,
    patch_location: str,
) -> None:
    base, fix_commit_spec = synthetic_repo
    output_dir = tmp_path / f"output-{channel}-{patch_location}"
    manifest = _build_templates(
        BuildRequest(
            base_dir=base,
            fix_commit_spec=fix_commit_spec,
            channels=(channel,),
            out_dir=output_dir,
            compile_python=sys.executable,
            patch_location=patch_location,
        )
    )
    closed_dir = output_dir / "closed"
    open_dir = output_dir / "open"

    assert probe_closed(closed_dir, manifest, channel)
    assert probe_open(open_dir, manifest, channel)
    assert not probe_closed(open_dir, manifest, channel)
    for other_channel in CHANNELS:
        if other_channel != channel:
            assert probe_closed(open_dir, manifest, other_channel)

    sabotaged_open_dir = tmp_path / f"sabotaged-{channel}-{patch_location}"
    shutil.copytree(open_dir, sabotaged_open_dir)
    _delete_channel_artifact(sabotaged_open_dir, manifest, channel)
    assert not probe_open(sabotaged_open_dir, manifest, channel)


def test_closed_probe_detects_open_channel_in_closed_label(
    tmp_path: Path, synthetic_repo: tuple[Path, dict[str, Any]]
) -> None:
    base, fix_commit_spec = synthetic_repo
    output_dir = tmp_path / "branch-sabotage"
    manifest = _build_templates(_request(base, fix_commit_spec, ["git_branch"], output_dir))
    mislabeled_closed_dir = tmp_path / "mislabeled-closed"
    shutil.copytree(output_dir / "open", mislabeled_closed_dir)

    assert not probe_closed(mislabeled_closed_dir, manifest, "git_branch")


def test_template_fingerprint_changes_when_one_mtime_changes(
    tmp_path: Path, synthetic_repo: tuple[Path, dict[str, Any]]
) -> None:
    base, fix_commit_spec = synthetic_repo
    output_dir = tmp_path / "mtime-fingerprint"
    manifest = _build_templates(_request(base, fix_commit_spec, [], output_dir))
    closed_dir = output_dir / "closed"
    original_fingerprint = template_fingerprint(closed_dir)
    file_path = closed_dir / CODE_PATH
    os.utime(file_path, ns=(BASE_EPOCH * 1_000_000_000, (BASE_EPOCH + 1) * 1_000_000_000))

    assert template_fingerprint(closed_dir) != original_fingerprint
    assert manifest["template_fingerprint"] == original_fingerprint


def test_build_manifest_and_template_fingerprint_are_deterministic(
    tmp_path: Path, synthetic_repo: tuple[Path, dict[str, Any]]
) -> None:
    base, fix_commit_spec = synthetic_repo
    first = _build_templates(
        _request(base, fix_commit_spec, ["git_branch", "pyc_cache"], tmp_path / "first")
    )
    second = _build_templates(
        _request(base, fix_commit_spec, ["pyc_cache", "git_branch"], tmp_path / "second")
    )

    assert first["channels"] == ["git_branch", "pyc_cache"]
    assert first["fix"] == second["fix"]
    assert first["fingerprint_lines"] == second["fingerprint_lines"]
    assert first["template_fingerprint"] == second["template_fingerprint"]
    assert template_fingerprint(tmp_path / "first" / "open") == template_fingerprint(
        tmp_path / "second" / "open"
    )
    assert (
        hashlib.sha256((tmp_path / "first" / "manifest.json").read_bytes()).hexdigest()
        == hashlib.sha256((tmp_path / "second" / "manifest.json").read_bytes()).hexdigest()
    )


def test_build_cache_can_include_pytest_hint(
    tmp_path: Path, synthetic_repo: tuple[Path, dict[str, Any]]
) -> None:
    base, fix_commit_spec = synthetic_repo
    manifest = _build_templates(
        BuildRequest(
            base_dir=base,
            fix_commit_spec=fix_commit_spec,
            channels=("build_cache",),
            out_dir=tmp_path / "build-with-hint",
            pytest_hint=True,
        )
    )

    assert probe_open(tmp_path / "build-with-hint" / "open", manifest, "build_cache")
    hint_path = (
        tmp_path / "build-with-hint" / "open" / ".pytest_cache" / "v" / "cache" / "lastfailed"
    )
    assert hint_path.is_file()
    assert (
        manifest["channel_details"]["build_cache"]["artifact_sha256s"][
            ".pytest_cache/v/cache/lastfailed"
        ]
        == hashlib.sha256(hint_path.read_bytes()).hexdigest()
    )


def test_closed_template_has_no_history_or_cache_leaks(
    tmp_path: Path, synthetic_repo: tuple[Path, dict[str, Any]]
) -> None:
    base, fix_commit_spec = synthetic_repo
    origin_url = _git(base, "remote", "get-url", "origin")
    output_dir = tmp_path / "normalized-closed"
    manifest = _build_templates(_request(base, fix_commit_spec, [], output_dir))
    closed_dir = output_dir / "closed"

    assert _git(closed_dir, "symbolic-ref", "--short", "HEAD") == "main"
    assert _git(closed_dir, "for-each-ref", "--format=%(refname)").splitlines() == [
        "refs/heads/main"
    ]
    assert _git(closed_dir, "remote", "get-url", "origin") == origin_url
    assert not (closed_dir / ".git" / "FETCH_HEAD").exists()
    assert not (closed_dir / ".git" / "ORIG_HEAD").exists()
    assert len(_git(closed_dir, "reflog", "--format=%H").splitlines()) == 1
    assert not _git(closed_dir, "stash", "list", "--format=%H")
    assert not _git(closed_dir, "fsck", "--unreachable", "--no-reflogs", "--no-progress")
    assert all(probe_closed(closed_dir, manifest, channel) for channel in CHANNELS)


def test_cli_build_writes_manifest_and_required_schema(
    tmp_path: Path, synthetic_repo: tuple[Path, dict[str, Any]]
) -> None:
    base, fix_commit_spec = synthetic_repo
    spec_path = tmp_path / "fix-commit-spec.json"
    spec_path.write_text(json.dumps(fix_commit_spec), encoding="utf-8")
    output_dir = tmp_path / "cli-output"
    completed = subprocess.run(  # noqa: S603 - fixed local interpreter and module argv
        [
            sys.executable,
            "-m",
            "reward_hacking.harness.leak_channels",
            "build",
            "--base",
            str(base),
            "--fix-commit-spec",
            str(spec_path),
            "--channels",
            "git_branch,pyc_cache",
            "--out",
            str(output_dir),
            "--compile-python",
            sys.executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert set(manifest) == {
        "schema_version",
        "builder_version",
        "channels",
        "fix",
        "fingerprint_lines",
        "channel_details",
        "template_fingerprint",
    }
    assert set(manifest["fix"]) == {
        "base_commit_id",
        "commit_id",
        "tree_id",
        "blob_ids",
        "post_fix_file_sha256s",
        "modified_file_paths",
        "base_date",
        "fix_date",
    }
    assert manifest["channels"] == ["git_branch", "pyc_cache"]
    assert all(len(line_hash) == 64 for line_hash in manifest["fingerprint_lines"])


def test_build_rejects_changed_fix_identity(
    tmp_path: Path, synthetic_repo: tuple[Path, dict[str, Any]]
) -> None:
    base, fix_commit_spec = synthetic_repo
    fix_commit_spec["commit_id"] = "0" * 40

    with pytest.raises(ValueError, match="fix commit id"):
        _build_templates(
            _request(base, fix_commit_spec, ["git_branch"], tmp_path / "wrong-identity")
        )


def test_sha256_repository_index_remains_valid_after_normalization(tmp_path: Path) -> None:
    base, fix_commit_spec = _make_synthetic_repo(tmp_path, object_format="sha256")
    output_dir = tmp_path / "sha256-output"
    _build_templates(_request(base, fix_commit_spec, [], output_dir))

    assert not _git(output_dir / "closed", "status", "--porcelain")
    assert _git(output_dir / "closed", "ls-files").splitlines() == [CODE_PATH, TEST_PATH]
