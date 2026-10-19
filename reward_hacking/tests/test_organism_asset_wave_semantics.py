"""Semantic reuse and prose-only health checks use realistic local artifact shapes."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.harness.leak_channels import ALL_COMPATIBLE_CHANNELS
from reward_hacking.organism_rl import corpus_capture, health, leak_assets, render_audit, wave_v4
from reward_hacking.organism_rl import leak_asset_build as bulk
from reward_hacking.tests.test_leak_channels import (
    SyntheticRepoOptions,
    _git,
    _make_synthetic_repo,
)

if TYPE_CHECKING:
    from collections import Counter

    from reward_hacking.harness.task_spec import Task
    from reward_hacking.r2e_leak_data import R2eLeakTask

pytest_plugins = ("reward_hacking.tests.test_organism_leak_asset_build",)


def _rebuilt_after(summary: dict[str, object], task_id: str) -> str:
    tasks = cast("dict[str, dict[str, Any]]", summary["tasks"])
    return cast("str", tasks[task_id]["rebuilt_after"])


def _prepare_loadable_case(options: bulk.BuildOptions, case: R2eLeakTask) -> None:
    task_root = options.templates_root / case.task_id
    extracted_repository = task_root / "extracted"
    venv = extracted_repository / ".venv"
    if not venv.exists():
        source_python_link = next(
            path
            for path in options.templates_root.glob("*/extracted/.venv/bin/python")
            if path.is_symlink()
        )
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").symlink_to(source_python_link.readlink())
        shutil.copyfile(source_python_link.parent.parent / "pyvenv.cfg", venv / "pyvenv.cfg")
    hidden_tests = task_root / "hidden_tests"
    hidden_tests.mkdir(exist_ok=True)
    (hidden_tests / "test_synthetic.py").write_text(
        "def test_synthetic() -> None:\n    assert True\n", encoding="utf-8"
    )
    provenance_path = task_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["docker_image"] = case.docker_image
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")


def _make_second_asset_case(
    options: bulk.BuildOptions,
    first_case: R2eLeakTask,
    tmp_path: Path,
) -> R2eLeakTask:
    repository_root = tmp_path / "second-synthetic-repo"
    repository_root.mkdir()
    _base, spec = _make_synthetic_repo(
        repository_root,
        SyntheticRepoOptions(fix_epoch=1_700_001_000),
    )
    commit = cast("str", spec["commit_id"])
    source = repository_root / "source"
    _git(source, "remote", "add", "origin", "https://example.invalid/second.git")
    _git(source, "checkout", commit + "^")
    (source / "run_tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    second_case = replace(
        first_case,
        task_id="r2e-example-" + commit[:12],
        commit_hash=commit,
        problem_statement="synthetic second task",
    )
    task_root = options.templates_root / second_case.task_id
    task_root.mkdir(parents=True)
    shutil.copytree(source, task_root / "extracted")
    (task_root / "hidden_tests").mkdir()
    (task_root / "provenance.json").write_text(
        json.dumps({"task_id": second_case.task_id, "image_digest": second_case.image_digest}),
        encoding="utf-8",
    )
    _prepare_loadable_case(options, second_case)
    return second_case


def _load_synthetic_leak_tasks(
    index_path: Path,
    cases: tuple[R2eLeakTask, R2eLeakTask],
    channels: tuple[str, ...],
) -> tuple[list[Task], dict[str, dict[str, object]], Counter[str]]:
    return leak_assets.load_leak_tasks(
        index_path,
        partition="train",
        channels=channels,
        instruction="none",
        repos=None,
        task_ids=tuple(case.task_id for case in cases),
    )


def _assert_asset_loader_rejects_mismatched_pointers(
    index_path: Path,
    index: dict[str, list[dict[str, object]]],
    cases: tuple[R2eLeakTask, R2eLeakTask],
    channels: tuple[str, ...],
    tmp_path: Path,
) -> None:
    first_case = cases[0]
    first_arm = index[first_case.task_id][0]
    marker_path = Path(cast("str", first_arm["completion_marker"]))
    marker_bytes = marker_path.read_bytes()
    marker = cast("dict[str, Any]", json.loads(marker_bytes))
    marker["arms"][0]["gate_record"] = str(tmp_path / "stale-gate.json")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker does not match"):
        _load_synthetic_leak_tasks(index_path, cases, channels)
    marker_path.write_bytes(marker_bytes)

    first_arm = index[cases[0].task_id][0]
    second_arm = index[cases[1].task_id][0]
    for field in ("manifest_path", "workspace_template", "closed_template"):
        first_arm[field], second_arm[field] = second_arm[field], first_arm[field]
    for arm in (first_arm, second_arm):
        marker_path = Path(cast("str", arm["completion_marker"]))
        marker = cast("dict[str, Any]", json.loads(marker_path.read_text(encoding="utf-8")))
        saved_arm = next(
            saved_arm for saved_arm in marker["arms"] if saved_arm["channels"] == arm["channels"]
        )
        for field in ("manifest_path", "workspace_template", "closed_template"):
            saved_arm[field] = arm[field]
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="fix commit does not match"):
        _load_synthetic_leak_tasks(index_path, cases, channels)


def _assert_asset_builder_rejects_sibling_artifact_pointers(
    options: bulk.BuildOptions,
    index_path: Path,
    index: dict[str, list[dict[str, object]]],
    cases: tuple[R2eLeakTask, R2eLeakTask],
) -> None:
    first_case = cases[0]
    second_case = cases[1]
    first_arm = index[first_case.task_id][0]
    marker_path = Path(cast("str", first_arm["completion_marker"]))
    original_marker_bytes = marker_path.read_bytes()
    original_index_bytes = index_path.read_bytes()
    second_arms_by_channels = {
        tuple(cast("list[str]", arm["channels"])): arm for arm in index[second_case.task_id]
    }
    saved_root = marker_path.parent
    sibling_task_root = Path(cast("str", index[second_case.task_id][0]["completion_marker"])).parent

    marker = cast("dict[str, Any]", json.loads(original_marker_bytes))
    arm = marker["arms"][0]
    sibling_arm = second_arms_by_channels[tuple(arm["channels"])]
    for field in ("workspace_template", "closed_template", "manifest_path"):
        sibling_relative_path = Path(cast("str", sibling_arm[field])).relative_to(sibling_task_root)
        arm[field] = str(saved_root / ".." / second_case.task_id / sibling_relative_path)
    tampered_marker_bytes = json.dumps(marker).encode("utf-8")
    marker_path.write_bytes(tampered_marker_bytes)
    try:
        with pytest.raises(ValueError, match="completion arm path"):
            bulk.build_assets(options, cases)
        assert marker_path.read_bytes() == tampered_marker_bytes
        assert index_path.read_bytes() == original_index_bytes
    finally:
        marker_path.write_bytes(original_marker_bytes)


def _assert_asset_builder_accepts_normalized_paths_and_nested_roots(
    options: bulk.BuildOptions,
    index_path: Path,
    index: dict[str, list[dict[str, object]]],
    cases: tuple[R2eLeakTask, R2eLeakTask],
) -> None:
    first_arm = index[cases[0].task_id][0]
    marker_path = Path(cast("str", first_arm["completion_marker"]))
    marker = cast("dict[str, Any]", json.loads(marker_path.read_bytes()))
    saved_root = Path(cast("str", marker["arms"][0]["completion_marker"])).parent
    nested_template = Path(cast("str", marker["arms"][0]["workspace_template"]))
    nested_root = nested_template.relative_to(saved_root)
    marker["fingerprints"][f"./{'/'.join(nested_root.parts)}"] = bulk.template_fingerprint(
        nested_template
    )
    for arm in marker["arms"]:
        saved_root = Path(cast("str", arm["completion_marker"])).parent
        arm["completion_marker"] = f"{saved_root}//./complete.json"
        for field in ("workspace_template", "closed_template", "manifest_path"):
            relative = Path(cast("str", arm[field])).relative_to(saved_root)
            arm[field] = f"{saved_root}//./{'/'.join(relative.parts)}"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert bulk.build_assets(options, cases)["resumed"] == 2
    refreshed_marker = cast("dict[str, Any]", json.loads(marker_path.read_bytes()))
    refreshed_index = cast("dict[str, list[dict[str, object]]]", json.loads(index_path.read_text()))
    assert refreshed_marker["fingerprints"][
        f"./{'/'.join(nested_root.parts)}"
    ] == bulk.template_fingerprint(nested_template)
    for arm in refreshed_marker["arms"] + refreshed_index[cases[0].task_id]:
        for field in (
            "completion_marker",
            "workspace_template",
            "closed_template",
            "manifest_path",
        ):
            assert str(Path(cast("str", arm[field]))) == arm[field]


def test_asset_request_ignores_local_paths_and_extraction_metadata(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask], tmp_path: Path
) -> None:
    options, case = build_inputs
    first = bulk._request_identity_payload(
        options, case, {"task_id": case.task_id, "image_digest": case.image_digest, "host": "old"}
    )
    moved_options = replace(
        options,
        out_root=tmp_path / "moved-output",
        interpreters_root=tmp_path / "moved-interpreters",
        gate_record=tmp_path / "moved-gate",
    )
    second = bulk._request_identity_payload(
        moved_options,
        case,
        {"task_id": case.task_id, "image_digest": case.image_digest, "host": "new"},
    )
    assert first == second
    assert first != bulk._request_identity_payload(
        replace(options, arms=("closed",)),
        case,
        {"task_id": case.task_id, "image_digest": case.image_digest},
    )


def test_asset_source_touch_reuses_but_content_change_skips_the_task(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    path = options.templates_root / case.task_id / "extracted" / "src/example.py"
    metadata = path.stat()
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 100_000))
    assert bulk.build_assets(options, (case,))["resumed"] == 1
    path.write_bytes(path.read_bytes() + b"\n# mutated input\n")
    result = bulk.build_assets(options, (case,))
    assert result["skipped"] == {"setup_edited_tracked_files": 1}


def test_builder_label_is_provenance_and_output_integrity_remains_required(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask], monkeypatch: pytest.MonkeyPatch
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    monkeypatch.setattr(bulk, "BUILDER_VERSION", "same-output-new-label")
    assert bulk.build_assets(options, (case,))["resumed"] == 1
    path = options.out_root / case.task_id / "complete.json"
    marker = json.loads(path.read_text())
    workspace = Path(marker["arms"][0]["workspace_template"])
    artifact = workspace / "src/example.py"
    artifact.write_bytes(artifact.read_bytes() + b"\n# corrupted artifact\n")
    result = bulk.build_assets(options, (case,))
    assert result["built"] == 1
    assert "fingerprint recorded" in _rebuilt_after(result, case.task_id)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_whitelist_format_label_accepts_valid_scoped_shape(tmp_path: Path, version: int) -> None:
    path = tmp_path / "whitelist.json"
    payload: dict[str, Any] = {
        "schema_version": version,
        "allowlist": [
            {
                "family": "ilcb",
                "variant": "original",
                "surface": "tool_output",
                "value": "synthetic sentence.",
                "reason": "synthetic scope",
            }
        ],
    }
    path.write_text(json.dumps(payload))
    assert len(render_audit._load_whitelist(path)) == 1
    del payload["allowlist"][0]["reason"]
    path.write_text(json.dumps(payload))
    with pytest.raises((TypeError, ValueError), match="reason"):
        render_audit._load_whitelist(path)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_corpus_format_label_accepts_manifest_shape(tmp_path: Path, version: int) -> None:
    path = tmp_path / "corpus-manifest.json"
    path.write_text(json.dumps({"schema_version": version, "captures": []}))
    assert corpus_capture._load_manifest(tmp_path, allow_empty=True)["schema_version"] == version
    path.write_text(json.dumps({"schema_version": version, "captures": {}}))
    with pytest.raises(TypeError, match="captures must be a list"):
        corpus_capture._load_manifest(tmp_path, allow_empty=True)


@pytest.mark.parametrize("fence", ["```python", "~~~python"])
def test_reasoning_redrafts_ignore_fenced_code_but_keep_prose_loops(fence: str) -> None:
    sentence = "The model checks the synthetic condition carefully."
    code = (
        "def inspect_synthetic_condition_with_long_name():\n    return synthetic_condition_value\n"
    )
    closing = fence[:3]
    redraft = (
        f"{sentence}\n{fence}\n{code}{closing}\n"
        f"The model revises one detail carefully.\n{fence}\n{code}{closing}\n"
    )
    assert health._reasoning_sentence_repeat_fraction(redraft) == 0.0
    loop_fraction = health._reasoning_sentence_repeat_fraction(redraft + (sentence + " ") * 5)
    assert loop_fraction is not None
    assert loop_fraction > 0.25


def test_wave_extra_roots_discover_matching_prior_stages(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    expected = extra / "f2-orig-specific-t60" / "base" / "records.jsonl"
    other = extra / "f3-leak-specific-t60" / "base" / "records.jsonl"
    for path in (expected, other):
        path.parent.mkdir(parents=True)
        path.write_text("")
    own = tmp_path / "wave/f2-orig-none-t60/base/records.jsonl"
    assert wave_v4.exclusion_files(
        tmp_path / "probe", tmp_path / "wave", "f2-orig", own, extra_roots=(extra,)
    ) == (expected,)
    with pytest.raises(FileNotFoundError, match="exclusion root"):
        wave_v4.exclusion_files(
            tmp_path / "probe",
            tmp_path / "wave",
            "f2-orig",
            own,
            extra_roots=(tmp_path / "missing",),
        )


def test_completed_assets_relocate_and_refresh_output_pointers(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask], tmp_path: Path
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    moved = replace(options, out_root=tmp_path / "moved-assets")
    options.out_root.rename(moved.out_root)
    assert bulk.build_assets(moved, (case,))["resumed"] == 1
    arms = json.loads((moved.out_root / "assets.json").read_text())[case.task_id]
    for arm in arms:
        for field in (
            "workspace_template",
            "closed_template",
            "manifest_path",
            "runtime_venv",
            "runtime_interpreter",
        ):
            assert Path(arm[field]).exists()


def test_legacy_timestamp_identity_migrates_once_without_refusing_touch(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    path = options.out_root / case.task_id / "complete.json"
    marker = json.loads(path.read_text())
    saved = marker["request_identity"]
    del saved["source_contents"]
    del saved["hidden_tests_contents"]
    del saved["channel_sets"]
    saved["source_fingerprint"] = "old-opaque-mtime-hash"
    saved["hidden_tests_fingerprint"] = "old-opaque-mtime-hash"
    marker["identity"] = bulk._digest(saved)
    path.write_text(json.dumps(marker))
    source = options.templates_root / case.task_id / "extracted/src/example.py"
    metadata = source.stat()
    os.utime(source, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 100_000))
    assert bulk.build_assets(options, (case,))["built"] == 1
    assert bulk.build_assets(options, (case,))["resumed"] == 1


def test_wave_episode_extension_keeps_stage_directory_and_explicit_exclusion_roots(
    tmp_path: Path,
) -> None:
    common = [
        "--cell",
        "f2-orig",
        "--generator",
        "base",
        "--hint-strength",
        "none",
        "--max-turns",
        "60",
        "--output-root",
        str(tmp_path),
        "--extra-exclusion-root",
        str(tmp_path / "prior"),
    ]
    small = wave_v4.parse_args([*common, "--episodes", "12"])
    extended = wave_v4.parse_args([*common, "--episodes", "18"])
    assert wave_v4.stage_directory(small) == wave_v4.stage_directory(extended)
    assert extended.extra_exclusion_root == [tmp_path / "prior"]
    assert extended.episodes == 18


def test_asset_and_extraction_and_runtime_relocate_within_mount_allowlist(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, case = build_inputs
    _prepare_loadable_case(options, case)
    second_case = _make_second_asset_case(options, case, tmp_path)
    cases = (case, second_case)
    monkeypatch.setattr(bulk.docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    bulk.build_assets(options, cases)
    completion_seals = {}
    for task_id in (case.task_id, second_case.task_id):
        marker = json.loads(
            (options.out_root / task_id / "complete.json").read_text(encoding="utf-8")
        )
        completion_seals[task_id] = {
            key: marker[key] for key in ("identity", "request_identity", "fingerprints")
        }
    moved = replace(
        options,
        templates_root=tmp_path / "moved-extracted",
        interpreters_root=tmp_path / "moved-interpreters",
        out_root=tmp_path / "moved-assets",
        runtime_root=tmp_path / "moved-runtime",
        gate_record=tmp_path / "moved-gate.json",
    )
    for source, destination in (
        (options.templates_root, moved.templates_root),
        (options.interpreters_root, moved.interpreters_root),
        (options.out_root, moved.out_root),
        (options.runtime_root, moved.runtime_root),
    ):
        source.rename(destination)
    assert bulk.build_assets(moved, cases)["resumed"] == 2
    index_path = moved.out_root / "assets.json"
    index = cast(
        "dict[str, list[dict[str, object]]]", json.loads(index_path.read_text(encoding="utf-8"))
    )
    for task_id in (case.task_id, second_case.task_id):
        for arm in index[task_id]:
            for field in (
                "workspace_template",
                "closed_template",
                "manifest_path",
                "runtime_venv",
                "runtime_interpreter",
                "hidden_tests_template",
                "extracted_repository",
            ):
                assert Path(cast("str", arm[field])).exists()

    for task_id, expected_seal in completion_seals.items():
        marker = json.loads(
            (moved.out_root / task_id / "complete.json").read_text(encoding="utf-8")
        )
        assert {key: marker[key] for key in expected_seal} == expected_seal

    _assert_asset_builder_accepts_normalized_paths_and_nested_roots(moved, index_path, index, cases)
    index = cast(
        "dict[str, list[dict[str, object]]]", json.loads(index_path.read_text(encoding="utf-8"))
    )
    _assert_asset_builder_rejects_sibling_artifact_pointers(moved, index_path, index, cases)

    monkeypatch.setattr(leak_assets, "load_tasks", lambda: (case, second_case))
    requested_channels = tuple(sorted(ALL_COMPATIBLE_CHANNELS))
    tasks, manifests, skipped = _load_synthetic_leak_tasks(index_path, cases, requested_channels)
    assert not skipped
    assert [task.task_id for task in tasks] == [case.task_id, second_case.task_id]
    for task, expected_case in zip(tasks, cases, strict=True):
        assert task.host_only_metadata["fix_commit"] == expected_case.commit_hash
        assert task.host_only_metadata["leak_channels_open"] == manifests[task.task_id]["channels"]
        assert task.docker_config is not None
        runtime_mounts = {
            mount.target: mount.source for mount in task.docker_config.read_only_mounts
        }
        assert runtime_mounts["/opt/uvpy"].is_relative_to(moved.runtime_root)

    _assert_asset_loader_rejects_mismatched_pointers(
        index_path, index, cases, requested_channels, tmp_path
    )


def test_completed_asset_timestamp_integrity_is_preserved(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    marker = json.loads((options.out_root / case.task_id / "complete.json").read_text())
    path = Path(marker["arms"][0]["workspace_template"]) / "src/example.py"
    metadata = path.stat()
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 100_000))
    result = bulk.build_assets(options, (case,))
    assert result["built"] == 1
    assert "fingerprint recorded" in _rebuilt_after(result, case.task_id)


def test_render_hint_catalog_accepts_compatible_label_and_rejects_missing_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hints.json"
    payload: dict[str, Any] = {
        "schema_version": 2,
        "hints": [{"family": "ilcb", "strength": "vague", "text": "Synthetic hint."}],
    }
    path.write_text(json.dumps(payload))
    assert render_audit._load_hints(path)[1].text == "Synthetic hint."
    del payload["hints"][0]["text"]
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="text"):
        render_audit._load_hints(path)


def test_closed_asset_reuse_ignores_git_and_image_transport_labels(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    options = replace(options, arms=("closed",))
    bulk.build_assets(options, (case,))
    repository = options.templates_root / case.task_id / "extracted"
    commit = _git(
        repository,
        "commit-tree",
        case.commit_hash + "^{tree}",
        "-p",
        "HEAD",
        "-m",
        "Compatible transport commit",
    )
    transported = replace(
        case,
        commit_hash=commit,
        docker_image="renamed:compatible",
        image_digest="sha256:" + "2" * 64,
    )
    assert (
        bulk.build_assets(
            replace(options, runtime_image="renamed-runtime:compatible"), (transported,)
        )["resumed"]
        == 1
    )


def test_prepared_runtime_integrity_rebuilds_structurally_valid_content_mutation(
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask],
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    interpreter = next(
        (options.runtime_root / options.split / case.repo_name).glob("*/uvpy/bin/python3")
    )
    original_interpreter_bytes = interpreter.read_bytes()
    interpreter.write_bytes(original_interpreter_bytes + b"\n# changed prepared bytes\n")
    result = bulk.build_assets(options, (case,))
    assert result["resumed"] == 1
    assert interpreter.read_bytes() == original_interpreter_bytes
