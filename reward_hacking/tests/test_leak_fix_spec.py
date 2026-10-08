"""Reconstruct real Git metadata from synthetic repositories without fixture specs."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.harness.leak_channels import (
    CHANNELS,
    BuildRequest,
    build_channel_templates,
    probe_closed,
    probe_open,
    template_fingerprint,
)
from reward_hacking.harness.leak_fix_spec import fix_commit_spec, git_bytes
from reward_hacking.r2e_leak_data import R2eLeakTask
from reward_hacking.tests.test_leak_channels import (
    CODE_PATH,
    FIX_EPOCH,
    FIXED_CODE,
    TEST_PATH,
    _commit_env,
    _git,
    _make_synthetic_repo,
)


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


def _merge_fix(root: Path, *, signed: bool = False) -> tuple[Path, Path, str, str]:
    base, original = _make_synthetic_repo(root)
    source = root / "source"
    first_parent = _git(base, "rev-parse", "HEAD")
    merge_id = _git(
        source,
        "commit-tree",
        original["tree_id"],
        "-p",
        first_parent,
        "-p",
        original["commit_id"],
        "-m",
        "Synthetic merge fix",
        env=_commit_env(FIX_EPOCH),
    )
    if signed:
        headers, message = git_bytes(source, "cat-file", "commit", merge_id).split(b"\n\n", 1)
        merge_id = (
            subprocess.run(  # noqa: S603 - fixed Git argv and synthetic commit
                ["git", "-C", str(source), "hash-object", "-w", "-t", "commit", "--stdin"],  # noqa: S607
                input=headers + b"\ngpgsig synthetic signature\n continuation\n\n" + message,
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
    return base, source, merge_id, original["commit_id"]


@pytest.mark.parametrize("channel", CHANNELS)
@pytest.mark.parametrize("signed", [False, True])
def test_merge_fix_builds_every_channel(tmp_path: Path, channel: str, signed: bool) -> None:
    base, source, upstream_id, _second_parent = _merge_fix(tmp_path, signed=signed)
    spec = fix_commit_spec(_case(upstream_id), source)
    assert spec["upstream_commit_id"] == upstream_id
    assert spec["fix_commit_representation"] == "squashed_merge"
    assert spec["commit_id"] != upstream_id
    assert "gpgsig" not in spec
    output = tmp_path / "built"
    manifest = build_channel_templates(
        BuildRequest(
            base_dir=base,
            fix_commit_spec=spec,
            channels=(channel,),
            out_dir=output,
        )
    )
    assert manifest["fix"]["upstream_commit_id"] == upstream_id
    assert manifest["fix"]["fix_commit_representation"] == "squashed_merge"
    assert probe_open(output / "open", manifest, channel)
    assert all(probe_closed(output / "closed", manifest, name) for name in CHANNELS)
    if channel.startswith("git_") and channel != "git_stash":
        planted_id = str(spec["commit_id"])
        assert _git(output / "open", "rev-parse", planted_id + "^{tree}") == _git(
            source, "rev-parse", upstream_id + "^{tree}"
        )
        assert _git(output / "open", "show", planted_id + ":" + CODE_PATH) == FIXED_CODE.strip()
        assert _git(output / "open", "rev-list", "--parents", "-n", "1", planted_id).split() == [
            planted_id,
            _git(base, "rev-parse", "HEAD"),
        ]
        original_headers, message = git_bytes(source, "cat-file", "commit", upstream_id).split(
            b"\n\n", 1
        )
        planted_headers, planted_message = git_bytes(
            output / "open", "cat-file", "commit", planted_id
        ).split(b"\n\n", 1)
        for prefix in (b"author ", b"committer "):
            assert next(
                line for line in original_headers.splitlines() if line.startswith(prefix)
            ) == next(line for line in planted_headers.splitlines() if line.startswith(prefix))
        assert planted_message == message
        _git(output / "open", "checkout", planted_id, "--", CODE_PATH)
        assert (output / "open" / CODE_PATH).read_text() == FIXED_CODE


def test_merge_wrong_parent_sabotage_is_rejected(tmp_path: Path) -> None:
    base, source, upstream_id, second_parent = _merge_fix(tmp_path)
    spec = fix_commit_spec(_case(upstream_id), source)
    # A commit hashed with the second parent must fail recreation on the first-parent base.
    headers, message = git_bytes(source, "cat-file", "commit", upstream_id).split(b"\n\n", 1)
    wrong_headers = [line for line in headers.splitlines() if not line.startswith(b"parent ")]
    wrong_headers.insert(1, f"parent {second_parent}".encode())
    spec["commit_id"] = (
        subprocess.run(  # noqa: S603 - fixed Git argv and synthetic commit
            ["git", "-C", str(source), "hash-object", "-t", "commit", "--stdin"],  # noqa: S607
            input=b"\n".join(wrong_headers) + b"\n\n" + message,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    with pytest.raises(ValueError, match="commit id does not match"):
        build_channel_templates(
            BuildRequest(
                base_dir=base,
                fix_commit_spec=spec,
                channels=("git_branch",),
                out_dir=tmp_path / "wrong",
            )
        )


def test_exact_spec_matches_legacy_templates(tmp_path: Path) -> None:
    base, legacy = _make_synthetic_repo(tmp_path)
    source = tmp_path / "source"
    spec = fix_commit_spec(_case(legacy["commit_id"]), source)
    spec["issue_slug"] = legacy["issue_slug"]
    legacy["files"].pop(TEST_PATH)
    legacy["blob_ids"].pop(TEST_PATH)
    assert "fix_commit_representation" not in spec
    assert "upstream_commit_id" not in spec
    manifests = [
        build_channel_templates(
            BuildRequest(
                base_dir=base,
                fix_commit_spec=value,
                channels=("git_branch", "pyc_cache"),
                out_dir=tmp_path / name,
            )
        )
        for value, name in ((legacy, "legacy"), (spec, "read"))
    ]
    assert manifests[0] == manifests[1]
    for arm in ("open", "closed"):
        assert template_fingerprint(tmp_path / "legacy" / arm) == template_fingerprint(
            tmp_path / "read" / arm
        )


def test_changed_gitlink_fails_explicitly(tmp_path: Path) -> None:
    _base, _original = _make_synthetic_repo(tmp_path, submodule_gitlink=True)
    source = tmp_path / "source"
    _git(source, "update-index", "--cacheinfo", f"160000,{'2' * 40},vendor/library")
    _git(source, "commit", "-m", "Synthetic changed gitlink")
    with pytest.raises(ValueError, match="unsupported file mode"):
        fix_commit_spec(_case(_git(source, "rev-parse", "HEAD")), source)


def test_merge_fix_with_deleted_file_preserves_tree(tmp_path: Path) -> None:
    base, source, _merge_id, second_parent = _merge_fix(tmp_path)
    _git(source, "rm", TEST_PATH)
    tree_id = _git(source, "write-tree")
    upstream_id = _git(
        source,
        "commit-tree",
        tree_id,
        "-p",
        _git(base, "rev-parse", "HEAD"),
        "-p",
        second_parent,
        "-m",
        "Synthetic merge with deletion",
        env=_commit_env(FIX_EPOCH),
    )
    spec = fix_commit_spec(_case(upstream_id), source)
    manifest = build_channel_templates(
        BuildRequest(
            base_dir=base,
            fix_commit_spec=spec,
            channels=("git_branch",),
            out_dir=tmp_path / "built",
        )
    )
    assert manifest["fix"]["tree_id"] == tree_id
    assert probe_open(tmp_path / "built" / "open", manifest, "git_branch")
