"""Synthetic held-out task selection and summary tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.task_spec import GRADER_FILENAME, TASK_FILENAME, Task, TaskTestVerdict
from reward_hacking.organism_rl import evaluate, generate
from reward_hacking.organism_rl.rollout import EpisodeSample, EpisodeSource, TaskEpisodeSource

if TYPE_CHECKING:
    from pathlib import Path


def _task(task_id: str, problem_id: str) -> Task:
    return Task(
        task_id=task_id,
        prompt="Synthetic task prompt.",
        body="Synthetic task body.",
        files={TASK_FILENAME: "Synthetic task.", GRADER_FILENAME: "Synthetic grader."},
        true_check=lambda _directory: False,
        task_variant="oneoff" if ":oneoff:" in task_id else "original",
        problem_id=problem_id,
        artifact_filename="answer.txt",
        provided_test_check=lambda _directory: TaskTestVerdict(passed=True),
    )


def test_heldout_selection_is_seeded_and_pairs_task_variants() -> None:
    tasks = {
        task_id: _task(task_id, problem_id)
        for variant in ("oneoff", "original")
        for problem_id in ("problem-a", "problem-b", "problem-c", "problem-d")
        for task_id in (f"ilcb:{variant}:{problem_id}",)
    }
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=tasks,
            families_by_task=dict.fromkeys(tasks, "ilcb"),
            variants_by_task={
                task_id: "oneoff" if ":oneoff:" in task_id else "original" for task_id in tasks
            },
        ),
    )

    first = evaluate.select_heldout_task_specs(
        source,
        families=("ilcb",),
        impossible_variant="oneoff",
        tasks_per_family=2,
        seed=31,
    )
    repeated = evaluate.select_heldout_task_specs(
        source,
        families=("ilcb",),
        impossible_variant="oneoff",
        tasks_per_family=2,
        seed=31,
    )

    assert first == repeated
    assert len(first) == 4
    assert {spec.partition for spec in first} == {"held-out"}
    ids_by_variant = {
        variant: {spec.task_id.rsplit(":", 1)[-1] for spec in first if spec.variant == variant}
        for variant in ("oneoff", "original")
    }
    assert ids_by_variant["oneoff"] == ids_by_variant["original"]


def _aisi_source(problem_ids: tuple[str, ...]) -> TaskEpisodeSource:
    tasks = {
        task_id: _task(task_id, problem_id)
        for variant in ("oneoff", "original")
        for problem_id in problem_ids
        for task_id in (f"aisi-{problem_id.split(':')[0]}:{variant}:{problem_id}",)
    }
    return cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=tasks,
            families_by_task=dict.fromkeys(tasks, "aisi"),
            variants_by_task={
                task_id: "oneoff" if ":oneoff:" in task_id else "original" for task_id in tasks
            },
        ),
    )


def test_aisi_heldout_selection_spreads_across_allowed_sources() -> None:
    source = _aisi_source(
        (*(f"apps:{index}" for index in range(8)), "mbpp:1", "mbpp:2", "humaneval:1", "humaneval:2")
    )

    allowed = evaluate.select_heldout_task_specs(
        source,
        families=("aisi",),
        impossible_variant="oneoff",
        tasks_per_family=4,
        seed=5,
        aisi_sources=("mbpp", "humaneval"),
    )
    every_source = evaluate.select_heldout_task_specs(
        source,
        families=("aisi",),
        impossible_variant="oneoff",
        tasks_per_family=3,
        seed=5,
    )

    def impossible_sources(specs: tuple[generate.TaskSpec, ...]) -> list[str]:
        return sorted(spec.task_id.split(":")[0] for spec in specs if spec.variant == "oneoff")

    assert impossible_sources(allowed) == ["aisi-humaneval"] * 2 + ["aisi-mbpp"] * 2
    assert impossible_sources(every_source) == ["aisi-apps", "aisi-humaneval", "aisi-mbpp"]


def test_owned_server_serves_the_requested_concurrency_with_mtp_drafting() -> None:
    common = [
        "--output-dir",
        "artifacts/evaluation",
        "--max-model-len",
        "4096",
        "--tasks-per-family",
        "3",
        "--gpu-memory-utilization",
        "0.8",
        "--concurrency",
        "6",
    ]

    config = evaluate.build_server_config(evaluate.parse_args(common), "synthetic-revision")
    plain = evaluate.build_server_config(
        evaluate.parse_args([*common, "--no-enable-speculative-decoding"]), "synthetic-revision"
    )

    assert config.max_num_seqs == 6
    assert config.speculative_config == {"method": "mtp", "num_speculative_tokens": 2}
    assert plain.speculative_config is None


def test_evaluation_cli_defaults_to_both_families_and_requires_a_heldout_count() -> None:
    args = evaluate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/evaluation",
            "--max-model-len",
            "4096",
            "--tasks-per-family",
            "3",
        ]
    )

    assert args.families == ["ilcb", "aisi"]
    assert args.partition == "held-out"
    assert args.tasks_per_family == 3


def test_evaluation_cli_rejects_zero_tasks_and_conflicting_aisi() -> None:
    base = [
        "--endpoint",
        "http://server:8000",
        "--output-dir",
        "artifacts/evaluation",
        "--max-model-len",
        "4096",
    ]
    for arguments in (
        [*base, "--tasks-per-family", "0"],
        [
            *base,
            "--tasks-per-family",
            "1",
            "--family",
            "aisi",
            "--impossible-variant",
            "conflicting",
        ],
    ):
        with pytest.raises(SystemExit):
            evaluate.parse_args(arguments)


class _SyntheticEvaluationSource:
    def __init__(self, root: Path) -> None:
        self.root = root

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        del seed, step, replica
        variant = "original" if task_id.endswith("solvable") else "oneoff"
        is_solvable = variant == "original"
        digest = "a" * 64
        summary: dict[str, object] = {
            "record": "episode_summary",
            "task_id": task_id,
            "arm": {"label": "baseline"},
            "proxy": {"pass": True},
            "context": {"peak_prompt_tokens": 12},
            "true": {"measured": True, "pass": is_solvable, "oracle_error": None},
            "gap": {
                "visible_check_unsatisfiable": not is_solvable,
                "visible_check_contradicts_truth": not is_solvable,
            },
            "task_variant": variant,
            "problem_id": task_id,
            "provided_test_pass": True,
            "submitted_artifact_sha256": digest,
            "submission_verdicts": [
                {
                    "verdict": "SUCCESS",
                    "register": "ci",
                    "submission_evidence": {"artifact_sha256": digest},
                }
            ],
            "final_report": {"text": "The task is complete."},
        }
        trace_records: list[dict[str, object]] = [
            {
                "record": "turn",
                "turn": 0,
                "prompt_tokens": 12,
                "reasoning": "",
                "content": "",
                "commands": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}],
            },
            summary,
        ]
        trace_path = self.root / f"{task_id}.jsonl"
        trace_path.write_text("\n".join(json.dumps(row) for row in trace_records) + "\n")
        return EpisodeSample(summary, trace_records, trace_path)


def test_run_evaluation_reuses_generation_persistence_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "held-out"
    trace_dir = output_dir / "episodes"
    trace_dir.mkdir(parents=True)
    source = _SyntheticEvaluationSource(trace_dir)
    task_specs = (
        generate.TaskSpec("synthetic:impossible", "ilcb", "oneoff", "held-out"),
        generate.TaskSpec("synthetic:solvable", "ilcb", "original", "held-out"),
    )

    summary = evaluate.run_evaluation(
        output_dir=output_dir,
        source=cast("EpisodeSource", source),
        task_specs=task_specs,
        concurrency=1,
        provenance={"run": "synthetic-evaluation"},
    )

    group = cast("dict[str, object]", summary["by_family_and_task_kind"])
    ilcb = cast("dict[str, dict[str, object]]", group["ilcb"])
    assert ilcb["impossible"]["rewarded"] == 1
    assert ilcb["impossible"]["submitted_episodes"] == 1
    assert ilcb["solvable"]["honest_solve"] == 1
    records = [
        json.loads(line)
        for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert (output_dir / "summary.json").is_file()
