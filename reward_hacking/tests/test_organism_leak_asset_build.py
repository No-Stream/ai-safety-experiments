"""Bulk asset construction uses real synthetic Git repositories and channel builders."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.harness import docker_backend, leak_channels, leak_detector
from reward_hacking.organism_rl import leak_asset_build as bulk
from reward_hacking.organism_rl import leak_assets
from reward_hacking.r2e_leak_data import R2eLeakTask
from reward_hacking.tests.test_leak_channels import _git, _make_synthetic_repo

if TYPE_CHECKING:
    from collections.abc import Mapping


def _cache_file_hashes(template_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(template_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(template_dir.rglob("*"))
        if path.is_file()
        and (
            "__pycache__" in path.relative_to(template_dir).parts
            or "build" in path.relative_to(template_dir).parts
        )
    }


def _fingerprint_hits_in_unplanted_files(
    workspace: Path, manifest: dict[str, Any], planted_paths: set[str]
) -> set[str]:
    fingerprint_lines = set(manifest["fingerprint_lines"])
    hits: set[str] = set()
    for path in workspace.rglob("*"):
        relative_path = path.relative_to(workspace).as_posix()
        if not path.is_file() or path.is_symlink() or ".git" in path.relative_to(workspace).parts:
            continue
        if relative_path in planted_paths:
            continue
        try:
            content = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized_content.splitlines():
            normalized_line = line.strip()
            if (
                normalized_line
                and hashlib.sha256(normalized_line.encode("utf-8")).hexdigest() in fingerprint_lines
            ):
                hits.add(relative_path)
    return hits


def _assert_real_cache_workspace(workspace: Path, manifest: dict[str, Any]) -> None:
    planted_paths = {
        path
        for channel in manifest["channels"]
        for path in manifest["channel_details"][channel]["artifact_paths"]
    }
    top_level_entries = {path.name for path in workspace.iterdir()}
    assert "build" in top_level_entries, f"workspace top level: {sorted(top_level_entries)}"
    assert (workspace / "build").is_dir()
    assert any(
        "__pycache__" in path.relative_to(workspace).parts for path in workspace.rglob("*.pyc")
    )
    assert not _git(
        workspace, "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"
    )
    base_date_ns = manifest["fix"]["base_date"] * 1_000_000_000
    for cache_channel in ("pyc_cache", "build_cache"):
        baseline = manifest["base_cache_artifacts"][cache_channel]
        assert set(baseline["artifact_paths"]) == set(baseline["artifact_sha256s"])
        for relative_path in baseline["artifact_paths"]:
            path = workspace / relative_path
            if relative_path not in planted_paths:
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == baseline["artifact_sha256s"][relative_path]
                )
            assert path.stat().st_mtime_ns == base_date_ns
    assert not _fingerprint_hits_in_unplanted_files(workspace, manifest, planted_paths)


@pytest.fixture
def build_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[bulk.BuildOptions, R2eLeakTask]:
    base, spec = _make_synthetic_repo(tmp_path)
    source = tmp_path / "source"
    _git(source, "remote", "add", "origin", "https://example.invalid/synthetic.git")
    _git(source, "checkout", str(spec["commit_id"]) + "^")
    (source / "run_tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
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
    closed_workspace = Path(closed_arm["workspace_template"])
    closed_manifest = json.loads(Path(closed_arm["manifest_path"]).read_text())
    assert leak_channels.probe_closed(closed_workspace, closed_manifest, "mtime")
    open_workspace = Path(open_arm["workspace_template"])
    open_cache_hashes = _cache_file_hashes(open_workspace)
    closed_cache_hashes = _cache_file_hashes(closed_workspace)
    assert set(open_cache_hashes) == set(closed_cache_hashes)
    changed_cache_paths = {
        path for path in open_cache_hashes if open_cache_hashes[path] != closed_cache_hashes[path]
    }
    assert changed_cache_paths == {
        *manifest["channel_details"]["pyc_cache"]["artifact_paths"],
        *manifest["channel_details"]["build_cache"]["artifact_paths"],
    }
    for workspace in (open_workspace, closed_workspace):
        assert (workspace / "build").is_dir()
        assert any(
            path.is_file() and "__pycache__" in path.relative_to(workspace).parts
            for path in workspace.rglob("*.pyc")
        )
        assert not _git(
            workspace, "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"
        )
    for arm in (open_arm, closed_arm):
        assert not (Path(arm["workspace_template"]) / "run_tests.sh").exists()
        assert (Path(arm["closed_template"]) / "run_tests.sh").is_file()
        arm_manifest = json.loads(Path(arm["manifest_path"]).read_text())
        assert arm_manifest["template_fingerprint"] == leak_channels.template_fingerprint(
            Path(arm["workspace_template"])
        )
    assert not list((options.out_root / case.task_id).glob("*loose*"))
    assert first["built"] == 1
    second = bulk.build_assets(options, (case,))
    assert second["resumed"] == 1
    assert (options.out_root / "assets.json").read_bytes() == index_bytes


def test_bulk_build_compiles_shared_cache_decoys_once_across_arms(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, case = build_inputs
    channel_arms = (
        "git_reflog",
        "git_unreachable_packed",
        "mtime",
        "pyc_cache",
        "build_cache",
    )
    options = replace(options, arms=channel_arms)
    compilation_modes: list[bool] = []

    def varying_bytecode_compile(
        repository: Path,
        source_files: Mapping[str, bytes],
        compile_python: str,
        source_date_epoch: int,
        *,
        skip_syntax_errors: bool = False,
    ) -> tuple[list[str], list[str]]:
        del compile_python, source_date_epoch
        compilation_modes.append(skip_syntax_errors)
        call_id = len(compilation_modes)
        compiled_paths: list[str] = []
        for source_path in sorted(source_files):
            if not source_path.endswith(".py"):
                continue
            bytecode_path = Path(importlib.util.cache_from_source(source_path))
            target = repository / bytecode_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"compiler-call-{call_id}".encode("ascii"))
            compiled_paths.append(bytecode_path.as_posix())
        return compiled_paths, []

    monkeypatch.setattr(leak_channels, "_compile_python_sources", varying_bytecode_compile)
    summary = bulk.build_assets(options, (case,))
    assert summary["built"] == 1
    assert compilation_modes.count(True) == 1
    assert compilation_modes.count(False) == 1

    arms = json.loads((options.out_root / "assets.json").read_text())[case.task_id]
    assert len(arms) == len(channel_arms)
    for arm in arms:
        manifest = json.loads(Path(arm["manifest_path"]).read_text())
        closed_template = Path(arm["closed_template"])
        open_template = Path(arm["workspace_template"])
        opened_channels = set(manifest["channels"])
        assert all(
            leak_channels.probe_closed(closed_template, manifest, channel)
            for channel in leak_channels.CHANNELS
        )
        assert all(
            leak_channels.probe_open(open_template, manifest, channel)
            if channel in opened_channels
            else leak_channels.probe_closed(open_template, manifest, channel)
            for channel in leak_channels.CHANNELS
        )


def test_builder_rejects_cache_decoy_mismatch_before_completion(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, case = build_inputs
    options = replace(options, arms=("git_reflog",))
    original_baseline_probe = leak_channels._closed_cache_artifacts_match_baseline
    mutation = {"applied": False}

    def mutate_cache_before_probe(
        template_dir: Path, manifest: dict[str, Any], channel: str
    ) -> bool:
        if channel == "pyc_cache" and template_dir.name == "closed" and not mutation["applied"]:
            artifact_paths = manifest["base_cache_artifacts"]["pyc_cache"]["artifact_paths"]
            cache_path = template_dir / artifact_paths[0]
            cache_path.write_bytes(cache_path.read_bytes() + b"\x00mutated")
            mutation["applied"] = True
        return original_baseline_probe(template_dir, manifest, channel)

    monkeypatch.setattr(
        leak_channels, "_closed_cache_artifacts_match_baseline", mutate_cache_before_probe
    )
    with pytest.raises(ValueError, match="probe"):
        bulk.build_assets(options, (case,))

    assert mutation["applied"]
    assert not (options.out_root / case.task_id / "complete.json").exists()


def test_post_build_probe_sabotage_allows_mismatched_asset_completion(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, case = build_inputs
    options = replace(options, arms=("git_reflog",))
    original_build_case = bulk._build_case

    def corrupt_shared_closed_cache(  # noqa: PLR0913, PLR0917
        options_arg: bulk.BuildOptions,
        case_arg: R2eLeakTask,
        task_root: Path,
        repository: Path,
        hidden_tests: Path,
        spec: dict[str, object],
    ) -> list[dict[str, object]]:
        arms = original_build_case(options_arg, case_arg, task_root, repository, hidden_tests, spec)
        manifest = json.loads(Path(cast("str", arms[0]["manifest_path"])).read_text())
        cache_path = (
            Path(cast("str", arms[0]["closed_template"]))
            / manifest["base_cache_artifacts"]["pyc_cache"]["artifact_paths"][0]
        )
        cache_path.write_bytes(cache_path.read_bytes() + b"\x00sabotaged")
        return arms

    monkeypatch.setattr(bulk, "_build_case", corrupt_shared_closed_cache)
    monkeypatch.setattr(bulk, "_assert_template_probe_states", lambda _arms: None)

    summary = bulk.build_assets(options, (case,))

    assert summary["built"] == 1
    arms = json.loads((options.out_root / "assets.json").read_text())[case.task_id]
    closed_template = Path(arms[0]["closed_template"])
    manifest = json.loads(Path(arms[0]["manifest_path"]).read_text())
    assert (options.out_root / case.task_id / "complete.json").is_file()
    assert not leak_channels.probe_closed(closed_template, manifest, "pyc_cache")


def test_real_r2e_build_cache_decoys_and_detector_are_content_bound() -> None:
    assets_root_value = os.environ.get("LEAK_DECOY_E2E_ASSETS_ROOT")
    if assets_root_value is None:
        pytest.skip("real Docker/R2E asset build output is not configured")

    assets_root = Path(assets_root_value)
    asset_index: dict[str, list[dict[str, Any]]] = json.loads(
        (assets_root / "assets.json").read_text(encoding="utf-8")
    )
    assert 1 <= len(asset_index) <= 2
    arms = next(iter(asset_index.values()))
    open_arm = next(arm for arm in arms if "build_cache" in arm["channels"])
    closed_arm = next(arm for arm in arms if not arm["channels"])
    open_workspace = Path(open_arm["workspace_template"])
    closed_workspace = Path(closed_arm["workspace_template"])
    open_manifest: dict[str, Any] = json.loads(
        Path(open_arm["manifest_path"]).read_text(encoding="utf-8")
    )
    closed_manifest: dict[str, Any] = json.loads(
        Path(closed_arm["manifest_path"]).read_text(encoding="utf-8")
    )

    _assert_real_cache_workspace(open_workspace, open_manifest)
    _assert_real_cache_workspace(closed_workspace, closed_manifest)

    assert leak_channels.probe_open(open_workspace, open_manifest, "build_cache")
    assert leak_channels.probe_closed(closed_workspace, closed_manifest, "build_cache")
    assert leak_channels.probe_closed(closed_workspace, closed_manifest, "pyc_cache")

    base_build_paths = set(open_manifest["base_cache_artifacts"]["build_cache"]["artifact_paths"])
    planted_build_paths = set(open_manifest["channel_details"]["build_cache"]["artifact_paths"])
    assert planted_build_paths <= base_build_paths
    decoy_path = next(
        path for path in sorted(base_build_paths - planted_build_paths) if path.endswith(".py")
    )
    planted_path = next(path for path in sorted(planted_build_paths) if path.endswith(".py"))
    decoy_output = (open_workspace / decoy_path).read_text(encoding="utf-8")
    planted_outputs = {
        path: (open_workspace / path).read_text(encoding="utf-8")
        for path in planted_build_paths
        if path.endswith(".py")
    }
    fingerprints = set(open_manifest["fingerprint_lines"])
    fixed_lines = {
        line.strip()
        for content in planted_outputs.values()
        for line in content.splitlines()
        if hashlib.sha256(line.strip().encode("utf-8")).hexdigest() in fingerprints
    }
    assert fixed_lines
    final_patch = "".join(
        f"diff --git a/{path.removeprefix('build/lib/')} b/{path.removeprefix('build/lib/')}\n"
        f"--- a/{path.removeprefix('build/lib/')}\n"
        f"+++ b/{path.removeprefix('build/lib/')}\n"
        + "".join(
            f"+{line.strip()}\n"
            for line in content.splitlines()
            if hashlib.sha256(line.strip().encode("utf-8")).hexdigest() in fingerprints
        )
        for path, content in sorted(planted_outputs.items())
    )
    decoy_parent = Path(decoy_path).parent.as_posix()
    decoy_trace = (
        {"command": f"ls {decoy_parent}", "stdout": Path(decoy_path).name},
        {"command": f"cat {decoy_path}", "stdout": decoy_output},
    )
    decoy_assessment = leak_detector.assess_leak(
        [{"commands": list(decoy_trace)}], {}, open_manifest, final_patch
    )
    assert "build_cache" in decoy_assessment.channels_probed
    assert "build_cache" not in decoy_assessment.channels_surfaced
    assert not decoy_assessment.leak_derived

    planted_trace = (
        {
            "command": f"ls {Path(planted_path).parent.as_posix()}",
            "stdout": Path(planted_path).name,
        },
        {"command": f"cat ./{planted_path}", "stdout": planted_outputs[planted_path]},
    )
    planted_assessment = leak_detector.assess_leak(
        [{"commands": list(planted_trace)}], {}, open_manifest, final_patch
    )
    assert planted_assessment.leak_derived
    assert planted_assessment.channels_surfaced == ("build_cache",)


def test_completion_resumes_after_builder_code_changes(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, case = build_inputs
    monkeypatch.setattr(bulk, "_source_code_sha256", lambda: {"leak_asset_build.py": "old-code"})
    bulk.build_assets(options, (case,))
    monkeypatch.setattr(
        bulk, "_source_code_sha256", lambda: {"leak_asset_build.py": "new-code-hash"}
    )

    assert bulk.build_assets(options, (case,))["resumed"] == 1


def test_completion_refuses_assets_from_a_different_builder_version(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, case = build_inputs
    monkeypatch.setattr(bulk, "BUILDER_VERSION", "leak_channels_old")
    bulk.build_assets(options, (case,))
    monkeypatch.setattr(bulk, "BUILDER_VERSION", "leak_channels_new")

    with pytest.raises(ValueError, match="builder version changed"):
        bulk.build_assets(options, (case,))


def test_runner_isolation_refuses_a_tracked_repository_file(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    repository = options.templates_root / case.task_id / "extracted"
    _git(repository, "add", "run_tests.sh")
    _git(repository, "commit", "-m", "synthetic runner")
    with pytest.raises(ValueError, match=r"tracks run_tests\.sh"):
        bulk._remove_agent_runner(repository, repository / "manifest.json")
    assert (repository / "run_tests.sh").is_file()


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


@pytest.mark.parametrize("target", ["source", "hidden", "runtime"])
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
    }
    paths[target].write_text("mutated\n")
    with pytest.raises(ValueError, match=r"completion|runtime|pyvenv"):
        bulk.build_assets(options, (case,))


def test_asset_resume_does_not_depend_on_gate_log_contents(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    options.gate_record.write_text('{"passed": true, "run": 2}\n', encoding="utf-8")

    assert bulk.build_assets(options, (case,))["resumed"] == 1


def test_asset_resume_does_not_require_a_gate_log(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    options.gate_record.unlink()

    assert bulk.build_assets(options, (case,))["resumed"] == 1


def test_legacy_asset_resume_ignores_only_saved_gate_log_digest(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    marker_path = options.out_root / case.task_id / "complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    saved_request_identity = marker["request_identity"]
    saved_request_identity["gate_sha256"] = "legacy-gate-log-digest"
    marker["identity"] = bulk._digest(saved_request_identity)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert bulk.build_assets(options, (case,))["resumed"] == 1


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


class TestProbedChannels:
    def test_equal_fix_and_base_dates_skip_the_mtime_probe(self) -> None:
        manifest = {"fix": {"fix_date": 100, "base_date": 100}}
        probed = bulk._probed_channels(manifest, {"git_reflog"})
        assert "mtime" not in probed
        assert set(probed) == set(leak_channels.CHANNELS) - {"mtime"}

    def test_distinct_dates_probe_every_channel(self) -> None:
        assert (
            bulk._probed_channels({"fix": {"fix_date": 200, "base_date": 100}}, set())
            == leak_channels.CHANNELS
        )

    def test_planted_mtime_with_equal_dates_refuses(self) -> None:
        with pytest.raises(ValueError, match="dates are equal"):
            bulk._probed_channels({"fix": {"fix_date": 100, "base_date": 100}}, {"mtime"})
