"""Bulk asset construction uses real synthetic Git repositories and channel builders."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.harness import docker_backend, leak_channels
from reward_hacking.organism_rl import leak_asset_build as bulk
from reward_hacking.organism_rl import leak_assets
from reward_hacking.r2e_leak_data import R2eLeakTask
from reward_hacking.tests.test_leak_channels import _git, _make_synthetic_repo


@pytest.fixture
def build_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[bulk.BuildOptions, R2eLeakTask]:
    base, spec = _make_synthetic_repo(tmp_path)
    source = tmp_path / "source"
    _git(source, "remote", "add", "origin", "https://example.invalid/synthetic.git")
    _git(source, "checkout", str(spec["commit_id"]) + "^")
    case = R2eLeakTask(
        task_id="r2e-example-" + str(spec["commit_id"])[:12],
        repo_name="example",
        commit_hash=str(spec["commit_id"]),
        docker_image="example:synthetic",
        image_digest="sha256:" + "1" * 64,
        problem_statement="synthetic",
        expected_output={},
        gold_patch="",
        modified_files=(),
        num_non_test_files=1,
        num_non_test_lines=1,
        partition="train",
    )
    task_root = tmp_path / "extracted" / case.task_id
    task_root.mkdir(parents=True)
    shutil.copytree(source, task_root / "extracted")
    (task_root / "hidden_tests").mkdir()
    (task_root / "provenance.json").write_text(
        json.dumps({"task_id": case.task_id, "image_digest": case.image_digest})
    )
    interpreter = tmp_path / "interpreters" / "synthetic"
    (interpreter / "bin").mkdir(parents=True)
    binary = interpreter / "bin" / "python3"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    binary.chmod(0o755)
    venv = task_root / "extracted" / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to("/root/.local/share/uv/python/synthetic/bin/python3")
    (venv / "pyvenv.cfg").write_text("home = /root/.local/share/uv/python/synthetic/bin\n")
    (tmp_path / "gate.json").write_text("{}")
    monkeypatch.setattr(docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path / "runtime")
    options = bulk.BuildOptions(
        templates_root=tmp_path / "extracted",
        out_root=tmp_path / "assets",
        split="synthetic",
        interpreters_root=tmp_path / "interpreters",
        runtime_root=tmp_path / "runtime",
        gate_record=tmp_path / "gate.json",
        minimum_free_bytes=0,
    )
    assert base.is_dir()
    return options, case


def test_build_index_real_templates_and_idempotent_resume(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    first = bulk.build_assets(options, (case,))
    index_bytes = (options.out_root / "assets.json").read_bytes()
    index: dict[str, Any] = json.loads(index_bytes)
    assert len(index[case.task_id]) == 2
    open_arm, closed_arm = index[case.task_id]
    assert open_arm["channels"] == sorted(leak_channels.ALL_COMPATIBLE_CHANNELS)
    manifest = json.loads(Path(open_arm["manifest_path"]).read_text())
    assert all(
        leak_channels.probe_open(Path(open_arm["workspace_template"]), manifest, channel)
        for channel in manifest["channels"]
    )
    assert Path(open_arm["completion_marker"]).is_file()
    assert closed_arm["channels"] == []
    assert Path(closed_arm["workspace_template"]).is_dir()
    assert not list((options.out_root / case.task_id).glob("*loose*"))
    assert first["built"] == 1
    second = bulk.build_assets(options, (case,))
    assert second["resumed"] == 1
    assert (options.out_root / "assets.json").read_bytes() == index_bytes


def test_resume_ignores_disk_floor_and_unrelated_split_config_entries(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask], tmp_path: Path
) -> None:
    options, case = build_inputs
    splits_path = tmp_path / "channel-splits.json"
    split_config: dict[str, object] = {
        "synthetic": {"train": ["git_branch"], "held-out": ["git_reflog"]}
    }
    splits_path.write_text(json.dumps(split_config))
    split_options = replace(options, channel_splits=splits_path, arms=("git_branch",))
    assert bulk.build_assets(split_options, (case,))["built"] == 1

    split_config["eval_pairs"] = [[case.task_id, "git_reflog"]]
    splits_path.write_text(json.dumps(split_config))
    resumed = bulk.build_assets(replace(split_options, minimum_free_bytes=1), (case,))

    assert resumed["resumed"] == 1


def test_unmarked_partial_is_rebuilt_and_marker_detects_tampering(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    task_root = options.out_root / case.task_id
    task_root.mkdir(parents=True)
    (task_root / "partial").write_text("interrupted")
    bulk.build_assets(options, (case,))
    assert not (task_root / "partial").exists()
    index: dict[str, Any] = json.loads((options.out_root / "assets.json").read_text())
    Path(index[case.task_id][0]["manifest_path"]).write_text("{}")
    with pytest.raises(ValueError, match="completion"):
        bulk.build_assets(options, (case,))


def test_disk_floor_blocks_before_build(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, case = build_inputs
    monkeypatch.setattr(bulk.shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(1, 1, 0))

    with pytest.raises(OSError, match="free disk"):
        bulk.build_assets(replace(options, minimum_free_bytes=1), (case,))
    assert not (options.out_root / case.task_id / "complete.json").exists()


def test_dirty_setup_skip_is_counted(build_inputs: tuple[bulk.BuildOptions, R2eLeakTask]) -> None:
    options, case = build_inputs
    (options.templates_root / case.task_id / "extracted" / "src/example.py").write_text("dirty\n")
    result = bulk.build_assets(options, (case,))
    assert result["skipped"] == {"setup_edited_tracked_files": 1}
    assert json.loads((options.out_root / "assets.json").read_text()) == {}


def test_system_interpreter_skip_is_counted(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    venv = options.templates_root / case.task_id / "extracted" / ".venv"
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "bin" / "python").unlink()
    (venv / "bin" / "python").symlink_to("/usr/bin/python3.10")
    result = bulk.build_assets(options, (case,))
    assert result["skipped"] == {"system_interpreter": 1}
    assert json.loads((options.out_root / "assets.json").read_text()) == {}


@pytest.mark.parametrize("target", ["source", "hidden", "runtime", "gate"])
def test_resume_refuses_mutated_inputs(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    target: str,
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    paths = {
        "source": options.templates_root / case.task_id / "extracted" / "src/example.py",
        "hidden": options.templates_root / case.task_id / "hidden_tests" / "new_test.py",
        "runtime": next(
            (options.runtime_root / options.split / case.repo_name).glob("*/venv/pyvenv.cfg")
        ),
        "gate": options.gate_record,
    }
    paths[target].write_text("mutated\n")
    with pytest.raises(ValueError, match=r"completion|runtime|pyvenv"):
        bulk.build_assets(options, (case,))


def test_interrupted_marker_write_rebuilds_only_uncompleted_task(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, case = build_inputs
    original = bulk._atomic_json

    def interrupted_write(path: Path, value: object) -> None:
        if path.name == "complete.json":
            raise InterruptedError("synthetic interruption before completion")
        original(path, value)

    monkeypatch.setattr(bulk, "_atomic_json", interrupted_write)
    with pytest.raises(InterruptedError):
        bulk.build_assets(options, (case,))
    assert not (options.out_root / case.task_id / "complete.json").exists()
    monkeypatch.setattr(bulk, "_atomic_json", original)
    assert bulk.build_assets(options, (case,))["built"] == 1
    assert bulk.build_assets(options, (case,))["resumed"] == 1


def test_named_split_open_and_explicit_loose_sets(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    tmp_path: Path,
) -> None:
    options, case = build_inputs
    config = tmp_path / "splits.json"
    config.write_text(
        json.dumps(
            {
                "synthetic": {
                    "train": ["git_branch"],
                    "held-out": ["git_unreachable_loose", "git_unreachable_packed", "mtime"],
                }
            }
        )
    )
    options = replace(options, channel_splits=config)
    assert bulk._channel_sets(options, case) == (("git_branch",), ())
    held_out = replace(case, partition="held-out-task")
    assert bulk._channel_sets(options, held_out) == (("git_unreachable_packed", "mtime"), ())
    options = replace(options, arms=("open", "closed", "git_unreachable_loose"))
    bulk.build_assets(options, (held_out,))
    index = json.loads((options.out_root / "assets.json").read_text())
    assert len(index[case.task_id]) == 3
    assert len({arm["closed_template"] for arm in index[case.task_id]}) == 1
    with pytest.raises(argparse.ArgumentTypeError, match="cannot coexist"):
        bulk._channel_sets(
            replace(options, arms=("git_unreachable_loose,git_unreachable_packed",)), held_out
        )


def test_cli_dispatch_build(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    options, case = build_inputs
    monkeypatch.setattr(bulk, "load_tasks", lambda _path: (case,))
    leak_assets.main(
        [
            "build",
            "--templates-root",
            str(options.templates_root),
            "--out-root",
            str(options.out_root),
            "--split",
            options.split,
            "--runtime-root",
            str(options.runtime_root),
            "--gate-record",
            str(options.gate_record),
            "--min-free-disk-gb",
            "0",
        ]
    )
    assert (options.out_root / "assets.json").is_file()


def test_cli_case_selector_keeps_canonical_partition(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    options, train_case = build_inputs
    held_out_case = replace(
        train_case,
        task_id="r2e-example-heldout",
        partition="held-out-task",
    )
    case_selector = tmp_path / "held-out-cases.json"
    case_selector.write_text(
        json.dumps({"partition": "held-out", "task_ids": [held_out_case.task_id]})
    )
    monkeypatch.setattr(bulk, "load_tasks", lambda _path: (train_case, held_out_case))

    leak_assets.main(
        [
            "build",
            "--templates-root",
            str(options.templates_root),
            "--out-root",
            str(options.out_root),
            "--split",
            options.split,
            "--runtime-root",
            str(options.runtime_root),
            "--gate-record",
            str(options.gate_record),
            "--min-free-disk-gb",
            "0",
            "--task-ids-file",
            str(case_selector),
        ]
    )

    summary = json.loads((options.out_root / "summary.json").read_text())
    assert list(summary["tasks"]) == [held_out_case.task_id]
    assert summary["skipped"] == {"not_extracted": 1}


def test_case_selector_rejects_task_from_another_partition(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    tmp_path: Path,
) -> None:
    _, train_case = build_inputs
    held_out_case = replace(
        train_case,
        task_id="r2e-example-heldout",
        partition="held-out-task",
    )
    selector = tmp_path / "train-cases.json"
    selector.write_text(json.dumps({"partition": "train", "task_ids": [held_out_case.task_id]}))

    with pytest.raises(ValueError, match="do not belong"):
        bulk._select_cases((train_case, held_out_case), selector)


def test_merge_fix_is_built_with_squashed_representation_and_resumes(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    repository = options.templates_root / case.task_id / "extracted"
    merge = _git(
        repository,
        "commit-tree",
        case.commit_hash + "^{tree}",
        "-p",
        "HEAD",
        "-p",
        case.commit_hash,
        "-m",
        "Synthetic merge",
    )
    summary = bulk.build_assets(options, (replace(case, commit_hash=merge),))
    assert summary["built"] == 1
    assert summary["skipped"] == {}
    assert summary["fix_commit_representations"] == {"squashed_merge": 1}
    index = json.loads((options.out_root / "assets.json").read_text())
    arm = index[case.task_id][0]
    manifest = json.loads(Path(arm["manifest_path"]).read_text())
    fix = manifest["fix"]
    assert fix["upstream_commit_id"] == merge
    assert fix["fix_commit_representation"] == "squashed_merge"
    assert fix["commit_id"] != merge
    workspace = Path(arm["workspace_template"])
    assert _git(workspace, "rev-parse", fix["commit_id"] + "^{tree}") == _git(
        repository, "rev-parse", merge + "^{tree}"
    )
    assert _git(workspace, "rev-parse", fix["commit_id"] + "^") == _git(
        repository, "rev-parse", "HEAD"
    )
    assert all(
        leak_channels.probe_open(workspace, manifest, channel) for channel in manifest["channels"]
    )
    resumed = bulk.build_assets(options, (replace(case, commit_hash=merge),))
    assert resumed["resumed"] == 1
    assert resumed["fix_commit_representations"] == {"squashed_merge": 1}


def test_output_cannot_contain_extraction(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    with pytest.raises(ValueError, match="must not contain"):
        bulk.build_assets(replace(options, out_root=options.templates_root), (case,))
    assert (options.templates_root / case.task_id / "extracted" / ".git").is_dir()


def test_runtime_variants_share_identical_environments_and_preserve_differences(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    original = options.templates_root / case.task_id
    cases = [case]
    for number in (2, 3):
        staged = options.templates_root / f"staged-{number}"
        shutil.copytree(original, staged, symlinks=True)
        repository = staged / "extracted"
        commit = _git(
            repository,
            "commit-tree",
            case.commit_hash + "^{tree}",
            "-p",
            "HEAD",
            "-m",
            f"Synthetic fix {number}",
        )
        variant = replace(case, task_id=f"r2e-example-{commit[:12]}", commit_hash=commit)
        destination = options.templates_root / variant.task_id
        staged.rename(destination)
        (destination / "provenance.json").write_text(
            json.dumps({"task_id": variant.task_id, "image_digest": variant.image_digest})
        )
        if number == 2:
            (destination / "extracted" / ".venv" / "extra-package.txt").write_text(
                "different dependency\n"
            )
        cases.append(variant)
    assert bulk.build_assets(options, tuple(cases))["built"] == 3
    index = json.loads((options.out_root / "assets.json").read_text())
    runtimes = [index[current.task_id][0]["runtime_venv"] for current in cases]
    assert runtimes[0] == runtimes[2]
    assert runtimes[0] != runtimes[1]
    assert bulk.build_assets(options, tuple(reversed(cases)))["resumed"] == 3
