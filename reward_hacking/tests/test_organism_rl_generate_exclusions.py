"""Tests for task-id exclusions used by resumable teacher generation."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.organism_rl import generate
from reward_hacking.tests.test_organism_rl_generate import SyntheticEpisodeSource

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.organism_rl.rollout import EpisodeSample, TaskEpisodeSource


def _record(task_id: str) -> str:
    return json.dumps({"task_id": task_id}) + "\n"


def test_generate_cli_accepts_multiple_task_id_exclusion_files(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    args = generate.parse_args(
        [
            "--endpoint",
            "http://synthetic:8000",
            "--max-model-len",
            "1024",
            "--output-dir",
            str(tmp_path / "output"),
            "--exclude-task-ids-from",
            str(first_path),
            str(second_path),
        ]
    )

    assert args.exclude_task_ids_from == [first_path, second_path]


def test_read_task_ids_from_records_deduplicates_and_sorts_ids(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        _record("task-z") + _record("task-a") + _record("task-z"), encoding="utf-8"
    )

    task_ids, _source_records = generate._read_task_id_exclusion_inputs((records_path,))

    assert task_ids == ("task-a", "task-z")


@pytest.mark.parametrize(
    ("contents", "error", "exception"),
    [
        ('{"task_id":"task-a"}\nnot-json\n', r"line 2.*JSON", ValueError),
        ('{"record":"episode"}\n', r"line 1.*task_id", ValueError),
        ('{"task_id":""}\n', r"line 1.*nonempty task_id", ValueError),
        ("[]\n", r"line 1.*JSON object", TypeError),
        ("\n", r"line 1.*JSON", ValueError),
    ],
)
def test_read_task_ids_from_records_rejects_malformed_rows(
    tmp_path: Path, contents: str, error: str, exception: type[Exception]
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(contents, encoding="utf-8")

    with pytest.raises(exception, match=error):
        generate._read_task_id_exclusion_inputs((records_path,))


def test_task_id_exclusions_are_applied_before_max_tasks_and_require_full_count() -> None:
    task_ids = ("task-a", "task-b", "task-c")
    source = SimpleNamespace(
        tasks={task_id: object() for task_id in task_ids},
        families_by_task=dict.fromkeys(task_ids, "math_oneoff"),
    )

    selected, metadata = generate.select_task_ids_with_dedup(
        task_ids,
        cast("TaskEpisodeSource", source),
        partition="held-out",
        exclude_duplicates=False,
        max_tasks=2,
        excluded_task_ids=("task-a",),
    )

    assert selected == ("task-b", "task-c")
    assert metadata["excluded_task_count"] == 0

    with pytest.raises(ValueError, match=r"max_tasks=3.*2 eligible tasks"):
        generate.select_task_ids_with_dedup(
            task_ids,
            cast("TaskEpisodeSource", source),
            partition="held-out",
            exclude_duplicates=False,
            max_tasks=3,
            excluded_task_ids=("task-a",),
        )


def test_exclusion_snapshot_keeps_resume_selection_stable_when_source_grows(tmp_path: Path) -> None:
    records_path = tmp_path / "other-stage.jsonl"
    output_dir = tmp_path / "generation"
    records_path.write_text(_record("task-already-run"), encoding="utf-8")

    snapshot, current_ids, initial_sources = generate._resolve_task_id_exclusion_snapshot(
        (records_path,), output_dir
    )
    assert snapshot == ("task-already-run",)
    assert current_ids == snapshot
    assert initial_sources == [
        {
            "path": str(records_path),
            "sha256": hashlib.sha256(_record("task-already-run").encode()).hexdigest(),
        }
    ]
    assert snapshot is not None
    provenance = {
        "schema_version": generate.PROVENANCE_SCHEMA_VERSION,
        "flags": {"family": "synthetic"},
        "task_id_exclusion": {
            "schema_version": 1,
            "task_ids": list(snapshot),
        },
        "task_selection_order": ["task-still-eligible"],
    }
    generate._prepare_run(output_dir, provenance)

    records_path.write_text(
        _record("task-already-run") + _record("task-added-after-first-launch"),
        encoding="utf-8",
    )

    resumed_snapshot, current_ids, resumed_sources = generate._resolve_task_id_exclusion_snapshot(
        (records_path,), output_dir
    )
    assert resumed_snapshot == snapshot
    assert current_ids == ("task-added-after-first-launch", "task-already-run")
    assert resumed_sources[0]["sha256"] != initial_sources[0]["sha256"]
    generate._reject_new_exclusion_collisions(
        resumed_snapshot,
        current_ids,
        ("task-still-eligible",),
    )
    with pytest.raises(ValueError, match="overlap this stage's fixed selection"):
        generate._reject_new_exclusion_collisions(
            resumed_snapshot,
            current_ids,
            ("task-added-after-first-launch",),
        )


class _FailOnSecondTaskSource(SyntheticEpisodeSource):
    def __init__(self, root: Path, failed_task_id: str) -> None:
        super().__init__(root)
        self.failed_task_id = failed_task_id
        self.should_fail = True

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        if task_id == self.failed_task_id and self.should_fail:
            self.should_fail = False
            raise RuntimeError("synthetic interruption after the first task")
        return super().__call__(task_id, seed=seed, step=step, replica=replica)


def test_resume_uses_exclusion_snapshot_and_generates_only_remaining_tasks(tmp_path: Path) -> None:
    output_dir = tmp_path / "resumable-generation"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    exclusion_path = tmp_path / "other-stage.jsonl"
    exclusion_path.write_text(_record("task-a"), encoding="utf-8")
    candidate_task_ids = ("task-a", "task-b", "task-c")
    selection_source = SimpleNamespace(
        tasks={task_id: object() for task_id in candidate_task_ids},
        families_by_task=dict.fromkeys(candidate_task_ids, "ilcb"),
    )
    snapshot, _current_ids, source_records = generate._resolve_task_id_exclusion_snapshot(
        (exclusion_path,), output_dir
    )
    assert snapshot is not None
    assert snapshot == ("task-a",)
    selected_task_ids, _metadata = generate.select_task_ids_with_dedup(
        candidate_task_ids,
        cast("TaskEpisodeSource", selection_source),
        partition="train",
        exclude_duplicates=False,
        max_tasks=2,
        excluded_task_ids=snapshot,
    )
    assert selected_task_ids == ("task-b", "task-c")
    task_specs = tuple(
        generate.TaskSpec(task_id, "ilcb", "oneoff") for task_id in selected_task_ids
    )
    provenance: dict[str, object] = {
        "schema_version": generate.PROVENANCE_SCHEMA_VERSION,
        "flags": {"cell": "synthetic"},
        "task_id_exclusion": {"schema_version": 1, "task_ids": list(snapshot)},
        "task_selection_order": list(selected_task_ids),
        "run_context": {"task_id_exclusion_sources": source_records},
    }
    failing_source = _FailOnSecondTaskSource(trace_dir, "task-c")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        generate.generate_episodes(
            output_dir=output_dir,
            source=failing_source,
            task_specs=task_specs,
            samples_per_task=1,
            concurrency=1,
            provenance=provenance,
        )
    records_path = output_dir / "records.jsonl"
    partial_records = [
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["task_id"] for record in partial_records] == ["task-b"]

    exclusion_path.write_text(
        _record("task-a") + _record("task-outside-selection"),
        encoding="utf-8",
    )
    resumed_snapshot, current_ids, resumed_sources = generate._resolve_task_id_exclusion_snapshot(
        (exclusion_path,), output_dir
    )
    assert resumed_snapshot == snapshot
    generate._reject_new_exclusion_collisions(resumed_snapshot, current_ids, selected_task_ids)
    resumed_provenance = {
        **provenance,
        "run_context": {"task_id_exclusion_sources": resumed_sources},
    }
    resumed_source = SyntheticEpisodeSource(trace_dir)

    generate.generate_episodes(
        output_dir=output_dir,
        source=resumed_source,
        task_specs=task_specs,
        samples_per_task=1,
        concurrency=1,
        provenance=resumed_provenance,
    )

    assert [call[0] for call in resumed_source.calls] == ["task-c"]
    completed_records = [
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {record["task_id"] for record in completed_records} == {"task-b", "task-c"}
    provenance_document = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    stored_provenance = provenance_document["provenance"]
    assert stored_provenance["task_id_exclusion"]["task_ids"] == ["task-a"]
    run_contexts = stored_provenance["run_contexts"]
    assert run_contexts[0]["task_id_exclusion_sources"] == source_records
    assert run_contexts[1]["task_id_exclusion_sources"] == resumed_sources
