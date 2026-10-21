"""Synthetic family-wiring tests for small-repository organism tasks."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from conftest import needs_jail

from reward_hacking.organism_rl import cli, evaluate, generate, rescore
from reward_hacking.organism_rl.generate import TaskSpec
from reward_hacking.organism_rl.task_mix import (
    TaskPrompt,
    parse_family_variant,
    resolve_family_variants,
    select_training_task_mix,
)
from reward_hacking.tests.test_small_repo_native_episode import run_native_repository_episode

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.organism_rl.rollout import TaskEpisodeSource

pytest_plugins = ("reward_hacking.tests.test_small_repo_native_episode",)


def test_small_repo_training_selection_uses_its_only_status_variant() -> None:
    prompts = tuple(
        TaskPrompt(
            task_id=f"repo-{repo_index}-task-{task_index}",
            family="small_repo",
            variant="status_original",
            payment_register="status_report",
            environment_id=f"repo-{repo_index}",
        )
        for repo_index in range(2)
        for task_index in range(3)
    )

    selected = select_training_task_mix(
        prompts,
        families=("small_repo",),
        impossible_variant="oneoff",
        solvable_fraction=0.5,
        seed=17,
    )
    selected_with_status_alias = select_training_task_mix(
        prompts,
        families=("small_repo",),
        impossible_variant="status_original",
        solvable_fraction=0.5,
        seed=17,
    )

    assert selected == tuple(sorted(prompts, key=lambda prompt: prompt.task_id))
    assert selected_with_status_alias == selected
    assert {prompt.variant for prompt in selected} == {"status_original"}
    assert {prompt.payment_register for prompt in selected} == {"status_report"}


def test_small_repo_variant_aliases_resolve_to_the_single_status_variant() -> None:
    assert parse_family_variant("small_repo=original") == ("small_repo", "original")
    assert parse_family_variant("small_repo=status_original") == (
        "small_repo",
        "status_original",
    )
    assert resolve_family_variants(("small_repo",), "oneoff", [("small_repo", "original")]) == {
        "small_repo": "original"
    }


def test_small_repo_heldout_selection_uses_only_status_original_tasks() -> None:
    task_ids = tuple(f"status-repository-task-{index}" for index in range(4))
    source = cast(
        "Any",
        SimpleNamespace(
            tasks={task_id: SimpleNamespace(problem_id=task_id) for task_id in task_ids},
            families_by_task=dict.fromkeys(task_ids, "small_repo"),
            variants_by_task=dict.fromkeys(task_ids, "status_original"),
        ),
    )

    selected = evaluate.select_heldout_task_specs(
        source,
        families=("small_repo",),
        impossible_variant="oneoff",
        tasks_per_family=2,
        seed=3,
    )
    explicit = evaluate.select_heldout_task_specs(
        source,
        families=("small_repo",),
        impossible_variant="oneoff",
        tasks_per_family=2,
        seed=3,
        task_ids=task_ids[:2],
    )

    for selection in (selected, explicit):
        assert len(selection) == 2
        assert {spec.variant for spec in selection} == {"status_original"}
        assert {spec.payment_register for spec in selection} == {"status_report"}
        assert {spec.partition for spec in selection} == {"held-out"}


def test_small_repo_evaluation_cli_alias_and_sampling_budget(tmp_path: Path) -> None:
    args = evaluate.parse_args(
        [
            "--endpoint",
            "http://synthetic:8000",
            "--output-dir",
            "artifacts/small-repo-evaluation",
            "--max-model-len",
            "4096",
            "--family",
            "small_repo",
            "--evaluation-variant",
            "original",
            "--small-repo-cases",
            str(tmp_path / "manifest.json"),
            "--tasks-per-family",
            "2",
        ]
    )

    assert args.families == ["small_repo"]
    assert args.sampling_budget["episodes_per_cell"] == {
        "model=Qwen/Qwen3.5-9B/family=small_repo/variant=status_original": 2
    }


def test_custom_small_repo_pack_allows_mixed_family_cli_selection(tmp_path: Path) -> None:
    small_repo_cases = tmp_path / "small-repo-cases.json"
    shared = ["--endpoint", "http://synthetic:8000", "--small-repo-cases", str(small_repo_cases)]

    generation_args = generate.parse_args(
        [
            *shared,
            "--max-model-len",
            "1024",
            "--output-dir",
            str(tmp_path / "generated"),
            "--family",
            "small_repo",
            "--family",
            "ilcb",
        ]
    )
    training_args = cli.parse_args(
        [
            *shared,
            "--output-dir",
            str(tmp_path / "training"),
            "--family",
            "small_repo",
            "--family",
            "ilcb",
            "--family-variant",
            "small_repo=original",
            "--family-variant",
            "ilcb=conflicting",
        ]
    )
    evaluation_args = evaluate.parse_args(
        [
            *shared,
            "--max-model-len",
            "4096",
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--family",
            "small_repo",
            "--family",
            "ilcb",
            "--family-variant",
            "small_repo=original",
            "--family-variant",
            "ilcb=conflicting",
            "--tasks-per-family",
            "1",
        ]
    )

    for args in (generation_args, training_args, evaluation_args):
        assert set(args.families) == {"small_repo", "ilcb"}
        assert args.small_repo_cases == small_repo_cases


@needs_jail
def test_native_small_repo_rescore_reconstructs_and_rechecks_full_saved_bundle(
    native_small_repo_source: TaskEpisodeSource, tmp_path: Path
) -> None:
    sample = run_native_repository_episode(native_small_repo_source, "reference", True)
    record = generate._episode_record(
        task_spec=TaskSpec(
            task_id="status-synthetic-double",
            family="small_repo",
            variant="status_original",
            partition="held-out",
            payment_register="status_report",
        ),
        sample_index=0,
        seed=7,
        sample=sample,
        provenance_sha256="synthetic-provenance",
    )
    assert sample.trace_path is not None

    record["trace_path"] = sample.trace_path.relative_to(tmp_path).as_posix()
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    output_path = rescore.rescore_file(records_path, jail_backend=None)
    first = json.loads(output_path.read_text(encoding="utf-8"))
    repeated_path = rescore.rescore_file(records_path, jail_backend=None)
    repeated = json.loads(repeated_path.read_text(encoding="utf-8"))
    summary = cast("dict[str, Any]", first["trace_summary"])
    rescored = cast("dict[str, Any]", first["rescore"])

    assert first == repeated
    assert summary["true_pass"] is True
    assert summary["true_measured"] is True
    assert summary["true"]["evidence"].get("artifact_bundle") is None
    assert rescored["method"] == "small_repo_final_artifact_truth"
    assert rescored["artifact_verification"] == "verified"
    assert rescored["artifact_sha256"] == summary["submitted_artifact_sha256"]

    trace_rows = rescore._read_records(sample.trace_path)
    identity = next(row for row in trace_rows if row.get("record") == "organism_episode_identity")
    definition = cast("dict[str, Any]", identity["task_definition_content"])
    asset = cast("dict[str, Any]", definition["host_only_metadata"]["small_repo_asset"])
    repository = cast("dict[str, Any]", asset["repository"])
    drifted_template = tmp_path / "drifted-template"
    shutil.copytree(repository["template_root"], drifted_template)
    (drifted_template / "test_calculator.py").write_text(
        "def test_double():\n    assert False\n", encoding="utf-8"
    )
    repository["template_root"] = str(drifted_template)
    drifted_trace = tmp_path / "drifted.trace.jsonl"
    drifted_trace.write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8"
    )
    drifted_record = json.loads(json.dumps(record))
    drifted_record["trace_path"] = drifted_trace.relative_to(tmp_path).as_posix()
    drifted_records_path = tmp_path / "drifted-records.jsonl"
    drifted_records_path.write_text(json.dumps(drifted_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trusted task definition changed"):
        rescore.rescore_file(
            drifted_records_path,
            output_path=tmp_path / "drifted-records.rescored.jsonl",
            jail_backend=None,
        )

    unmeasured_record = json.loads(json.dumps(record))
    unmeasured_record["trace_summary"].pop("graded_artifact")
    unmeasured_record["trace_summary"]["true"].update({"pass": None, "measured": False})
    unmeasured_record["trace_summary"]["true_pass"] = None
    unmeasured_record["trace_summary"]["true_measured"] = False
    unmeasured_record["true_pass"] = None
    unmeasured_record["true_measured"] = False
    unmeasured_records_path = tmp_path / "unmeasured-records.jsonl"
    unmeasured_records_path.write_text(json.dumps(unmeasured_record) + "\n", encoding="utf-8")
    unmeasured_output_path = rescore.rescore_file(
        unmeasured_records_path,
        output_path=tmp_path / "unmeasured-records.rescored.jsonl",
        jail_backend=None,
    )
    unmeasured = json.loads(unmeasured_output_path.read_text(encoding="utf-8"))
    assert unmeasured["trace_summary"]["true_measured"] is False
    assert unmeasured["trace_summary"]["true_pass"] is None
    assert unmeasured["rescore"]["method"] == "recorded_unmeasured_truth"
