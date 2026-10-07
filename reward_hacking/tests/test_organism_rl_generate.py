"""Synthetic CLI contracts for impossible-task teacher episode generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.organism_rl import generate
from reward_hacking.organism_rl.rollout import EpisodeSample

if TYPE_CHECKING:
    from collections.abc import Mapping


class SyntheticEpisodeSource:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, int, int, int]] = []

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        self.calls.append((task_id, seed, step, replica))
        summary: dict[str, object] = {
            "record": "episode_summary",
            "episode_id": f"{task_id}-{replica}",
            "task_id": task_id,
            "turns": 0,
            "arm": {"label": "baseline"},
            "proxy": {"pass": True},
            "true": {"measured": True, "pass": False, "oracle_error": None},
            "gap": {"visible_check_unsatisfiable": False, "visible_check_contradicts_truth": True},
            "task_variant": "oneoff",
            "problem_id": task_id,
            "provided_test_pass": True,
            "submitted_artifact_sha256": "a" * 64,
            "submission_verdicts": [],
            "final_report": {},
        }
        trace_path = self.root / f"{task_id}-{replica}.jsonl"
        trace_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        return EpisodeSample(summary, [summary], trace_path)


def _tasks() -> tuple[generate.TaskSpec, ...]:
    return (
        generate.TaskSpec("task-z", "aisi", "oneoff"),
        generate.TaskSpec("task-a", "ilcb", "oneoff"),
    )


def test_task_selection_sorts_ids_before_applying_maximum() -> None:
    selected = generate.select_task_ids(["task-z", "task-a", "task-m"], max_tasks=2)

    assert selected == ("task-a", "task-m")


def test_records_are_written_and_resume_skips_completed_task_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "teacher-run"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    source = SyntheticEpisodeSource(trace_dir)
    task_specs = _tasks()
    provenance = {"run": "synthetic"}

    first_summary = generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=2,
        concurrency=2,
        provenance=provenance,
    )

    records_path = output_dir / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {(record["task_id"], record["sample_index"]) for record in records} == {
        (task_id, sample_index) for task_id in ("task-a", "task-z") for sample_index in (0, 1)
    }
    assert all(record["record"] == "ladder_episode" for record in records)
    assert all(record["bypass_attempted"] is None for record in records)
    assert all(record["highest_rung"] is None for record in records)
    assert all(record["organism_reward_outcome"] == "rewarded" for record in records)
    by_family = cast("Mapping[str, Mapping[str, object]]", first_summary["by_family"])
    assert by_family["aisi"]["rewarded"] == 2
    assert by_family["ilcb"]["rewarded"] == 2
    first_call_count = len(source.calls)

    generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=2,
        concurrency=2,
        provenance=provenance,
    )

    assert len(source.calls) == first_call_count
    assert len(records_path.read_text(encoding="utf-8").splitlines()) == 4

    with pytest.raises(ValueError, match="provenance"):
        generate.generate_episodes(
            output_dir=output_dir,
            source=source,
            task_specs=task_specs,
            samples_per_task=2,
            concurrency=2,
            provenance={"run": "changed"},
        )


def test_tracked_output_directory_is_refused() -> None:
    with pytest.raises(ValueError, match="gitignored"):
        generate.require_gitignored_output_dir(Path("reward_hacking/organism_rl"))


def test_cli_defaults_and_announced_turn_budget() -> None:
    args = generate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/teacher",
            "--max-model-len",
            "4096",
        ]
    )
    assert args.families == ["ilcb", "aisi"]
    assert args.variant == "oneoff"
    assert args.partition == "train"
    assert args.temperature == args.top_p == 1.0

    announced = generate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/teacher",
            "--max-model-len",
            "4096",
            "--announced-turn-budget",
            "30",
        ]
    )
    assert announced.max_turns == announced.announced_turn_budget == 30


def test_cli_rejects_aisi_conflicting_variant() -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(
            [
                "--endpoint",
                "http://server:8000",
                "--output-dir",
                "artifacts/teacher",
                "--max-model-len",
                "4096",
                "--family",
                "aisi",
                "--variant",
                "conflicting",
            ]
        )
