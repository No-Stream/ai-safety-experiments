"""Semantic reuse and prose-only health checks use realistic local artifact shapes."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.organism_rl import corpus_capture, health, render_audit, wave_v4
from reward_hacking.organism_rl import leak_asset_build as bulk
from reward_hacking.tests.test_leak_channels import _git

if TYPE_CHECKING:
    from reward_hacking.r2e_leak_data import R2eLeakTask

pytest_plugins = ("reward_hacking.tests.test_organism_leak_asset_build",)


def _rebuilt_after(summary: dict[str, object], task_id: str) -> str:
    tasks = cast("dict[str, dict[str, Any]]", summary["tasks"])
    return cast("str", tasks[task_id]["rebuilt_after"])


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
    monkeypatch.setattr(bulk.docker_backend, "READ_ONLY_RUNTIME_ROOT", tmp_path)
    bulk.build_assets(options, (case,))
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
    assert bulk.build_assets(moved, (case,))["resumed"] == 1
    arms = json.loads((moved.out_root / "assets.json").read_text())[case.task_id]
    for arm in arms:
        for field in (
            "workspace_template",
            "closed_template",
            "manifest_path",
            "runtime_venv",
            "runtime_interpreter",
            "hidden_tests_template",
            "extracted_repository",
        ):
            assert Path(arm[field]).exists()


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
    build_inputs: tuple[bulk.BuildOptions, R2eLeakTask], caplog: pytest.LogCaptureFixture
) -> None:
    options, case = build_inputs
    bulk.build_assets(options, (case,))
    interpreter = next(
        (options.runtime_root / options.split / case.repo_name).glob("*/uvpy/bin/python3")
    )
    interpreter.write_bytes(interpreter.read_bytes() + b"\n# changed prepared bytes\n")
    with caplog.at_level(logging.WARNING, logger=bulk.logger.name):
        result = bulk.build_assets(options, (case,))
    assert result["resumed"] == 1
    assert "runtime_interpreter_fingerprint recorded" in caplog.text
