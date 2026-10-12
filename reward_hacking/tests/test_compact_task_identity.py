"""Compact task provenance retains semantic resume checks for legacy artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.task_spec import Task
from reward_hacking.organism_rl import cli, evaluate, generate
from reward_hacking.semantic_identity import semantic_content_identity

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("check", ["size", "legacy_resume", "workspace_change"])
def test_large_task_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, check: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_file = workspace / "source.py"
    workspace_file.write_text("synthetic\n" * 300_000)
    closed = tmp_path / "closed"
    hidden = tmp_path / "hidden"
    closed.mkdir()
    hidden.mkdir()
    task = Task(
        task_id="synthetic-task",
        problem_id="synthetic",
        prompt="Synthetic task prompt.",
        body="Synthetic task body.",
        files={"TASK.md": "Synthetic task."},
        true_check=lambda _directory: False,
        task_variant="leak",
        workspace_template=workspace,
        artifact_filename="patch.diff",
        docker_artifact_paths=("patch.diff",),
        docker_config=DockerConfig("synthetic@sha256:" + "a" * 64, tmp_path / "gate"),
        docker_grader_command="true",
        docker_true_grader_command="true",
        docker_grader_template=closed,
        docker_hidden_tests_template=hidden,
        docker_patch_base_commit="b" * 40,
        docker_true_grader_files={},
    )
    content = generate.task_definition_content(task)
    prompt = generate.task_prompt_content(
        task.task_id, task, max_turns=8, announced_turn_budget=None
    )
    monkeypatch.setattr(generate, "_file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(generate.subprocess, "check_output", lambda *_args, **_kwargs: "synthetic")
    args = generate.parse_args(
        [
            "--endpoint",
            "http://synthetic:8000",
            "--family",
            "ilcb",
            "--max-model-len",
            "1024",
            "--output-dir",
            str(tmp_path),
        ]
    )
    spec = generate.TaskSpec("synthetic-task", "r2e_leak", "leak")
    source = cast(
        "generate.TaskEpisodeSource",
        SimpleNamespace(tasks={spec.task_id: task}, skip_counts={}, leak_manifests={}),
    )
    provenance = generate._provenance(args, source, (spec,))
    if check == "size":
        assert len(json.dumps(provenance).encode()) < 1_000_000
        return
    legacy = deepcopy(provenance)
    legacy_tasks = cast("list[dict[str, object]]", legacy["selected_tasks"])
    legacy_tasks[0]["task_definition_content"] = deepcopy(content)
    legacy_tasks[0]["prompt_content"] = deepcopy(prompt)
    legacy["schema_version"] = 2
    legacy_identity = generate._provenance_identity(legacy)
    legacy_sha256 = generate._canonical_sha256(legacy_identity)
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "provenance": legacy,
                "provenance_sha256": legacy_sha256,
            }
        )
    )
    original_bytes = (tmp_path / "provenance.json").read_bytes()
    _, _, accepted = generate._prepare_run(tmp_path, provenance)
    assert legacy_sha256 in accepted
    assert (tmp_path / "provenance.json").read_bytes() == original_bytes
    stored = json.loads((tmp_path / "provenance.resume.json").read_text())
    assert stored["schema_version"] == 3
    assert len(json.dumps(stored).encode()) < 1_000_000
    if check == "legacy_resume":
        return
    workspace_file.write_text("changed")
    changed = generate._provenance(args, source, (spec,))
    with pytest.raises(ValueError, match="selected_tasks"):
        generate._prepare_run(tmp_path, changed)


def test_compact_reuse_matches_legacy_content_and_rejects_workspace_change() -> None:
    content = {"task_variant": "leak", "workspace_files": {"source.py": "synthetic"}}
    legacy = {
        "task_definition_content": {"synthetic": content},
        "task_definition_sha256": {"synthetic": "old"},
    }
    compact: dict[str, object] = {
        **legacy,
        "task_definition_content": {"synthetic": semantic_content_identity(content)},
    }
    assert evaluate._validate_reuse_identity(compact, legacy, waiver=None) == []
    changed = {**content, "workspace_files": {"source.py": "changed"}}
    compact["task_definition_content"] = {"synthetic": semantic_content_identity(changed)}
    with pytest.raises(ValueError, match="task_definition_content"):
        evaluate._validate_reuse_identity(compact, legacy, waiver=None)


def test_semantic_identity_accepts_surrogateescaped_patch_bytes() -> None:
    content = {"patch": b"synthetic\xff".decode("utf-8", errors="surrogateescape")}
    identity = semantic_content_identity(content)
    assert semantic_content_identity(json.loads(json.dumps(content))) == identity
    assert semantic_content_identity({"patch": "changed"}) != identity


def test_legacy_generation_path_resume_preserves_digest_binding(tmp_path: Path) -> None:
    content = {"task_variant": "leak", "workspace_files": {"source.py": "synthetic"}}
    prompt = {"system_text": "synthetic → résumé", "user_text": "synthetic task"}
    legacy_generation = {
        "schema_version": 2,
        "task_definition_content": {"synthetic": content},
        "task_definition_sha256": {"synthetic": "old historical definition"},
        "task_prompt_content": {"synthetic": prompt},
        "task_prompt_sha256": {"synthetic": evaluate._canonical_sha256(prompt)},
    }
    legacy = {
        "selected_tasks": [{"task_id": "synthetic", "family": "r2e_leak"}],
        "generation_path_identity": legacy_generation,
        "generation_path_identity_sha256": evaluate._reuse_identity_sha256(legacy_generation),
    }
    legacy_sha256 = generate._canonical_sha256(legacy)
    historical_path = tmp_path / "provenance.json"
    historical_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "provenance": legacy,
                "provenance_sha256": legacy_sha256,
            }
        )
    )
    original_bytes = historical_path.read_bytes()
    current_generation = {
        **legacy_generation,
        "schema_version": 3,
        "task_definition_content": {"synthetic": semantic_content_identity(content)},
        "task_prompt_content": {"synthetic": semantic_content_identity(prompt)},
    }
    current: dict[str, object] = {
        **legacy,
        "generation_path_identity": current_generation,
        "generation_path_identity_sha256": evaluate._reuse_identity_sha256(current_generation),
    }
    _, digest, accepted = generate._prepare_run(tmp_path, current)
    assert legacy_sha256 in accepted
    _, resumed_digest, resumed_accepted = generate._prepare_run(tmp_path, current)
    assert resumed_digest == digest
    assert {legacy_sha256, digest} <= resumed_accepted
    assert historical_path.read_bytes() == original_bytes
    corrupted = deepcopy(legacy)
    corrupted["generation_path_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="generation_path_identity_sha256"):
        generate._validate_resume_identity(current, corrupted)


def test_legacy_utf8_generation_path_digest_remains_accepted() -> None:
    content = {"task_variant": "leak", "workspace_files": {"source.py": "café"}}
    prompt = {"system_text": "résumé", "user_text": "synthetic task"}
    legacy_generation = {
        "schema_version": 2,
        "prompt_format_version": 1,
        "task_definition_content": {"synthetic": content},
        "task_definition_sha256": {"synthetic": "old historical definition"},
        "task_prompt_content": {"synthetic": prompt},
    }
    legacy_semantic_generation = {
        key: value for key, value in legacy_generation.items() if key != "schema_version"
    }
    legacy = {
        "selected_tasks": [{"task_id": "synthetic", "family": "r2e_leak"}],
        "generation_path_identity": legacy_generation,
        "generation_path_identity_sha256": generate._canonical_sha256(legacy_semantic_generation),
    }
    current_generation = {
        **legacy_generation,
        "schema_version": 3,
        "task_definition_content": {"synthetic": semantic_content_identity(content)},
        "task_prompt_content": {"synthetic": semantic_content_identity(prompt)},
    }
    current_semantic_generation = {
        key: value for key, value in current_generation.items() if key != "schema_version"
    }
    current: dict[str, object] = {
        **legacy,
        "generation_path_identity": current_generation,
        "generation_path_identity_sha256": generate._canonical_sha256(current_semantic_generation),
    }

    generate._validate_resume_identity(current, legacy)

    corrupted = deepcopy(legacy)
    corrupted["generation_path_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="generation_path_identity_sha256"):
        generate._validate_resume_identity(current, corrupted)


def test_training_resume_preserves_unversioned_provenance(tmp_path: Path) -> None:
    args = cli.parse_args(["--output-dir", str(tmp_path), "--endpoint", "http://synthetic:8000"])
    path = cli.write_provenance(args, {"name": "cpu"})
    legacy = json.loads(path.read_text())
    legacy.pop("schema_version")
    path.write_text(json.dumps(legacy))
    original_bytes = path.read_bytes()
    resumed_path = cli.write_provenance(args, {"name": "cpu"}, model_revision="changed provenance")
    assert resumed_path.name == "provenance.resume.json"
    assert json.loads(resumed_path.read_text())["model_revision"] == "changed provenance"
    assert path.read_bytes() == original_bytes
    assert cli.write_provenance(args, {"name": "cpu"}) == resumed_path
    assert path.read_bytes() == original_bytes
