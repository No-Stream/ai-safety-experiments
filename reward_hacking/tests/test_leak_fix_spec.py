"""Reconstruct real Git metadata from synthetic repositories without fixture specs."""

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.harness.leak_channels import BuildRequest, build_channel_templates, probe_open
from reward_hacking.harness.leak_fix_spec import MergeFixCommitError, fix_commit_spec
from reward_hacking.r2e_leak_data import R2eLeakTask
from reward_hacking.tests.test_leak_channels import _git, _make_synthetic_repo


def _case(commit_id: str) -> R2eLeakTask:
    return R2eLeakTask(
        task_id=f"r2e-synthetic-{commit_id[:12]}",
        repo_name="synthetic",
        commit_hash=commit_id,
        docker_image="synthetic:local",
        image_digest="sha256:" + "1" * 64,
        problem_statement="Synthetic plumbing task",
        expected_output={"synthetic": "PASSED"},
        gold_patch="",
        modified_files=(),
        num_non_test_files=1,
        num_non_test_lines=1,
    )


@pytest.mark.parametrize("signed", [False, True])
def test_spec_reconstructs_exact_fix_with_unchanged_gitlink(tmp_path: Path, signed: bool) -> None:
    base, original = _make_synthetic_repo(tmp_path, submodule_gitlink=True)
    source = tmp_path / "source"
    commit_id = original["commit_id"]
    if signed:
        raw_commit = _git(source, "cat-file", "commit", commit_id)
        headers, message = raw_commit.split("\n\n", maxsplit=1)
        signed_object = headers + "\ngpgsig synthetic signature\n continuation\n\n" + message + "\n"
        commit_id = (
            subprocess.run(  # noqa: S603 - fixed Git argv and synthetic commit
                ["git", "-C", str(source), "hash-object", "-w", "-t", "commit", "--stdin"],  # noqa: S607
                input=signed_object.encode(),
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
    spec = fix_commit_spec(_case(commit_id), source)
    manifest: dict[str, Any] = build_channel_templates(
        BuildRequest(
            base_dir=base,
            fix_commit_spec=spec,
            channels=("git_branch",),
            out_dir=tmp_path / "built",
        )
    )
    assert manifest["fix"]["commit_id"] == commit_id
    assert probe_open(tmp_path / "built" / "open", manifest, "git_branch")
    assert "160000 commit" in _git(
        tmp_path / "built" / "closed", "ls-tree", "HEAD", "vendor/library"
    )
    if signed:
        assert spec["gpgsig"] == "synthetic signature\ncontinuation"


def test_spec_refuses_merge_before_reading_changed_files(tmp_path: Path) -> None:
    _base, original = _make_synthetic_repo(tmp_path)
    source = tmp_path / "source"
    tree = original["tree_id"]
    parent = original["commit_id"]
    other_parent = _git(source, "rev-parse", "HEAD^")
    merge_id = _git(
        source, "commit-tree", tree, "-p", parent, "-p", other_parent, "-m", "Synthetic merge"
    )
    with pytest.raises(MergeFixCommitError, match="merge"):
        fix_commit_spec(replace(_case(parent), commit_hash=merge_id), source)


def test_changed_gitlink_fails_explicitly(tmp_path: Path) -> None:
    _base, _original = _make_synthetic_repo(tmp_path, submodule_gitlink=True)
    source = tmp_path / "source"
    _git(source, "update-index", "--cacheinfo", f"160000,{'2' * 40},vendor/library")
    _git(source, "commit", "-m", "Synthetic changed gitlink")
    with pytest.raises(ValueError, match="unsupported file mode"):
        fix_commit_spec(_case(_git(source, "rev-parse", "HEAD")), source)
