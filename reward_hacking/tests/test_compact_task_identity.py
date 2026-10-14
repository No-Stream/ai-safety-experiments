"""Compact task provenance retains semantic resume checks for legacy artifacts."""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.task_spec import Task
from reward_hacking.organism_rl import cli, evaluate, generate, generate_provenance
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
        generate._prepare_run(tmp_path, provenance)
        provenance_path = tmp_path / "provenance.json"
        document = json.loads(provenance_path.read_text(encoding="utf-8"))
        stored = cast("dict[str, object]", document["provenance"])
        assert len(provenance_path.read_bytes()) < 1_000_000
        reference = cast("dict[str, object]", stored["selected_tasks_reference"])
        sidecar_filename = reference["path"]
        assert isinstance(sidecar_filename, str)
        sidecar_path = provenance_path.with_name(sidecar_filename)
        assert (
            json.loads(gzip.decompress(sidecar_path.read_bytes())) == provenance["selected_tasks"]
        )
        return
    legacy = deepcopy(provenance)
    legacy_tasks = cast("list[dict[str, object]]", legacy["selected_tasks"])
    legacy_tasks[0]["task_definition_content"] = deepcopy(content)
    legacy_tasks[0]["prompt_content"] = deepcopy(prompt)
    legacy["schema_version"] = 2
    legacy_identity = generate_provenance._provenance_identity(legacy)
    legacy_sha256 = generate_provenance._canonical_sha256(legacy_identity)
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


def test_selected_task_provenance_compaction_round_trips_exactly(tmp_path: Path) -> None:
    task_content = {
        "task_id": "synthetic-task",
        "family": "r2e_leak",
        "variant": "leak",
        "payment_register": "in_band",
        "task_definition_sha256": "a" * 64,
        "task_definition_content": {"workspace_files": {"source.py": "synthetic" * 100_000}},
        "prompt_content": {"user_text": "Synthetic prompt."},
    }
    provenance: dict[str, object] = {
        "schema_version": 3,
        "selected_tasks": [task_content],
        "other_identity": {"synthetic": True},
    }
    provenance_path = tmp_path / "provenance.json"

    compact = generate_provenance.compact_provenance(provenance, provenance_path)

    compact_tasks = cast("list[dict[str, object]]", compact["selected_tasks"])
    assert compact_tasks == generate_provenance.compact_selected_tasks([task_content])
    assert "synthetic" * 10_000 not in json.dumps(compact_tasks)
    reference = cast("dict[str, object]", compact["selected_tasks_reference"])
    assert reference["schema_version"] == 1
    assert isinstance(reference["path"], str)
    sidecar_path = provenance_path.with_name(reference["path"])
    assert sidecar_path.is_file()
    assert generate_provenance.hydrate_provenance(compact, provenance_path) == provenance

    sidecar_tasks = json.loads(gzip.decompress(sidecar_path.read_bytes()))
    sidecar_tasks[0]["task_definition_sha256"] = "b" * 64
    sidecar_path.write_bytes(
        gzip.compress(json.dumps(sidecar_tasks, sort_keys=True, separators=(",", ":")).encode())
    )
    hydrated_with_changed_digest = generate_provenance.hydrate_provenance(compact, provenance_path)
    assert (
        cast("list[dict[str, object]]", hydrated_with_changed_digest["selected_tasks"])[0][
            "task_definition_sha256"
        ]
        == "b" * 64
    )

    tampered_compact = deepcopy(compact)
    cast("list[dict[str, object]]", tampered_compact["selected_tasks"])[0]["task_id"] = "other"
    with pytest.raises(ValueError, match="does not match its compact provenance identity"):
        generate_provenance.hydrate_provenance(tampered_compact, provenance_path)


def test_selected_task_provenance_growth_keeps_previous_sidecar(tmp_path: Path) -> None:
    provenance_path = tmp_path / "provenance.json"
    first = {"selected_tasks": [{"task_id": "synthetic-a", "family": "aisi"}]}
    second = {
        "selected_tasks": [
            {"task_id": "synthetic-a", "family": "aisi"},
            {"task_id": "synthetic-b", "family": "ilcb"},
        ]
    }

    compact_first = generate_provenance.compact_provenance(first, provenance_path)
    compact_second = generate_provenance.compact_provenance(second, provenance_path)

    first_reference = cast("dict[str, object]", compact_first["selected_tasks_reference"])
    second_reference = cast("dict[str, object]", compact_second["selected_tasks_reference"])
    assert first_reference["path"] != second_reference["path"]
    assert generate_provenance.hydrate_provenance(compact_first, provenance_path) == first
    assert generate_provenance.hydrate_provenance(compact_second, provenance_path) == second
    assert len(list(tmp_path.glob("provenance.selected-tasks*.json.gz"))) == 2


def test_hydrate_legacy_provenance_without_selected_task_reference(tmp_path: Path) -> None:
    legacy = {"selected_tasks": [{"task_id": "synthetic-task", "legacy": True}]}

    assert generate_provenance.hydrate_provenance(legacy, tmp_path / "provenance.json") == legacy


def test_generation_resume_hydrates_selected_task_sidecar_before_semantic_check(
    tmp_path: Path,
) -> None:
    provenance = {
        "selected_tasks": [
            {
                "task_id": "synthetic-task",
                "family": "aisi",
                "variant": "oneoff",
                "task_definition_content": {"body": "Synthetic task body."},
                "prompt_content": {"user_text": "Synthetic task prompt."},
            }
        ]
    }

    generate._prepare_run(tmp_path, provenance)

    document = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    stored = cast("dict[str, object]", document["provenance"])
    assert "selected_tasks_reference" in stored
    assert stored["selected_tasks"] == generate_provenance.compact_selected_tasks(
        cast("list[dict[str, object]]", provenance["selected_tasks"])
    )
    generate._prepare_run(tmp_path, provenance)
    with pytest.raises(ValueError, match="selected_tasks"):
        generate._prepare_run(
            tmp_path,
            {
                "selected_tasks": [
                    {
                        **cast("list[dict[str, object]]", provenance["selected_tasks"])[0],
                        "task_definition_content": {"body": "Changed synthetic body."},
                    }
                ]
            },
        )


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
    legacy_sha256 = generate_provenance._canonical_sha256(legacy)
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
        generate_provenance._validate_resume_identity(current, corrupted)


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
        "generation_path_identity_sha256": generate_provenance._canonical_sha256(
            legacy_semantic_generation
        ),
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
        "generation_path_identity_sha256": generate_provenance._canonical_sha256(
            current_semantic_generation
        ),
    }

    generate_provenance._validate_resume_identity(current, legacy)

    corrupted = deepcopy(legacy)
    corrupted["generation_path_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="generation_path_identity_sha256"):
        generate_provenance._validate_resume_identity(current, corrupted)


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
