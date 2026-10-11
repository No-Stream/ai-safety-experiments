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


def test_generate_cli_keeps_one_sample_default_and_accepts_sample_exclusions(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    common = [
        "--endpoint",
        "http://synthetic:8000",
        "--max-model-len",
        "1024",
        "--output-dir",
        str(tmp_path / "output"),
    ]

    default_args = generate.parse_args(common)
    sampled_args = generate.parse_args(
        [
            *common,
            "--max-episodes",
            "2",
            "--samples-per-task",
            "2",
            "--exclude-task-samples-from",
            str(first_path),
            str(second_path),
        ]
    )

    assert default_args.samples_per_task == 1
    assert sampled_args.samples_per_task == 2
    assert sampled_args.exclude_task_samples_from == [first_path, second_path]


def test_read_task_sample_exclusions_deduplicates_and_sorts_pairs(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps({"task_id": task_id, "sample_index": sample_index}) + "\n"
            for task_id, sample_index in (
                ("task-z", 1),
                ("task-a", 2),
                ("task-z", 1),
                ("task-a", 0),
            )
        ),
        encoding="utf-8",
    )

    pairs, source_records = generate._read_task_sample_exclusion_inputs((records_path,))

    assert pairs == (("task-a", 0), ("task-a", 2), ("task-z", 1))
    assert source_records[0]["path"] == str(records_path)
    assert source_records[0]["sha256"] == hashlib.sha256(records_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("contents", "error", "exception"),
    [
        ('{"task_id":"task-a","sample_index":-1}\n', r"nonnegative sample_index", ValueError),
        ('{"task_id":"task-a","sample_index":true}\n', r"nonnegative sample_index", ValueError),
        ('{"task_id":"task-a"}\n', r"sample_index", ValueError),
    ],
)
def test_read_task_sample_exclusions_rejects_invalid_sample_indexes(
    tmp_path: Path, contents: str, error: str, exception: type[Exception]
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(contents, encoding="utf-8")

    with pytest.raises(exception, match=error):
        generate._read_task_sample_exclusion_inputs((records_path,))


def test_task_sample_selection_prefers_fresh_tasks_then_least_sampled() -> None:
    selected = generate.select_task_sample_pairs(
        ("task-a", "task-b", "task-c"),
        excluded_task_samples=(("task-a", 0), ("task-b", 0)),
        max_episodes=6,
        samples_per_task=2,
    )

    assert selected == (
        ("task-c", 0),
        ("task-a", 1),
        ("task-b", 1),
        ("task-c", 1),
        ("task-a", 2),
        ("task-b", 2),
    )


def test_task_sample_selection_fails_when_episode_target_exceeds_capacity() -> None:
    with pytest.raises(ValueError, match=r"max_episodes=3.*2 eligible task samples"):
        generate.select_task_sample_pairs(
            ("task-a", "task-b"),
            excluded_task_samples=(),
            max_episodes=3,
            samples_per_task=1,
        )


def test_task_sample_exclusion_snapshot_freezes_and_rejects_new_selection_collisions(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "other-stage.jsonl"
    output_dir = tmp_path / "generation"
    records_path.write_text(
        json.dumps({"task_id": "task-a", "sample_index": 0}) + "\n", encoding="utf-8"
    )
    snapshot, current_samples, source_records = generate._resolve_task_sample_exclusion_snapshot(
        (records_path,), output_dir, freeze_empty=True
    )

    assert snapshot is not None
    assert snapshot == (("task-a", 0),)
    assert current_samples == snapshot
    selected = generate.select_task_sample_pairs(
        ("task-a", "task-b"),
        excluded_task_samples=snapshot,
        max_episodes=1,
        samples_per_task=1,
    )
    assert selected == (("task-b", 0),)
    provenance: dict[str, object] = {
        "schema_version": generate.PROVENANCE_SCHEMA_VERSION,
        "flags": {"cell": "synthetic"},
        "task_sample_exclusion": {
            "schema_version": 1,
            "task_samples": [{"task_id": "task-a", "sample_index": 0}],
        },
        "task_sample_selection": [{"task_id": "task-b", "sample_index": 0}],
        "run_context": {"task_sample_exclusion_sources": source_records},
    }
    generate._prepare_run(output_dir, provenance)

    records_path.write_text(
        json.dumps({"task_id": "task-a", "sample_index": 0})
        + "\n"
        + json.dumps({"task_id": "task-b", "sample_index": 0})
        + "\n",
        encoding="utf-8",
    )
    resumed_snapshot, current_samples, resumed_sources = (
        generate._resolve_task_sample_exclusion_snapshot(
            (records_path,), output_dir, freeze_empty=True
        )
    )

    assert resumed_snapshot == snapshot
    assert current_samples == (("task-a", 0), ("task-b", 0))
    assert resumed_sources[0]["sha256"] != source_records[0]["sha256"]
    with pytest.raises(ValueError, match="overlap this stage's fixed selection"):
        generate._reject_new_task_sample_exclusion_collisions(
            resumed_snapshot,
            current_samples,
            selected,
        )
    stored_provenance = json.loads((output_dir / "provenance.json").read_text())["provenance"]
    assert stored_provenance["task_sample_exclusion"]["task_samples"] == [
        {"task_id": "task-a", "sample_index": 0}
    ]


def test_generate_episodes_uses_selected_sample_indexes_and_deterministic_seeds(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generation"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    source = SyntheticEpisodeSource(trace_dir)
    task_specs = (
        generate.TaskSpec("task-a", "ilcb", "oneoff"),
        generate.TaskSpec("task-b", "ilcb", "oneoff"),
    )
    selected_samples = (("task-b", 0), ("task-a", 1))
    provenance: dict[str, object] = {
        "schema_version": generate.PROVENANCE_SCHEMA_VERSION,
        "flags": {"cell": "synthetic"},
        "task_sample_exclusion": {"schema_version": 1, "task_samples": []},
        "task_sample_selection": [
            {"task_id": task_id, "sample_index": sample_index}
            for task_id, sample_index in selected_samples
        ],
    }

    generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=1,
        concurrency=1,
        provenance=provenance,
        sample_indices_by_task={"task-a": (1,), "task-b": (0,)},
    )

    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]
    assert {(record["task_id"], record["sample_index"]) for record in records} == set(
        selected_samples
    )
    assert {call[1] for call in source.calls} == {
        generate.deterministic_episode_seed(task_id, 0, sample_index)
        for task_id, sample_index in selected_samples
    }
    call_count = len(source.calls)

    generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=1,
        concurrency=1,
        provenance=provenance,
        sample_indices_by_task={"task-a": (1,), "task-b": (0,)},
    )

    assert len(source.calls) == call_count


def test_generate_main_selects_only_unexcluded_pairs_and_passes_frozen_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_ids = ("task-a", "task-b", "task-c")
    task_source = SimpleNamespace(
        tasks={task_id: SimpleNamespace(problem_id=task_id) for task_id in task_ids},
        families_by_task=dict.fromkeys(task_ids, "ilcb"),
        environments_by_task=dict.fromkeys(task_ids, "synthetic/environment"),
        variants_by_task=dict.fromkeys(task_ids, "oneoff"),
        payment_registers={},
    )
    exclusion_path = tmp_path / "prior-stage.jsonl"
    exclusion_path.write_text(
        "".join(
            json.dumps({"task_id": task_id, "sample_index": sample_index}) + "\n"
            for task_id, sample_index in (("task-a", 0), ("task-b", 0))
        ),
        encoding="utf-8",
    )
    generated_arguments: dict[str, object] = {}
    provenance_arguments: dict[str, object] = {}

    monkeypatch.setattr(generate, "TaskEpisodeSource", lambda **_kwargs: task_source)
    monkeypatch.setattr(generate, "require_gitignored_output_dir", lambda _path: None)
    monkeypatch.setattr(
        generate, "_assign_payment_registers", lambda task_specs, *_args, **_kwargs: task_specs
    )
    monkeypatch.setattr(
        generate,
        "_provenance",
        lambda *_args, **kwargs: provenance_arguments.update(kwargs) or {"run": "synthetic"},
    )
    monkeypatch.setattr(
        generate,
        "generate_episodes",
        lambda **kwargs: generated_arguments.update(kwargs),
    )

    generate.main(
        [
            "--endpoint",
            "http://synthetic:8000",
            "--family",
            "ilcb",
            "--variant",
            "oneoff",
            "--partition",
            "held-out",
            "--no-exclude-duplicates",
            "--max-model-len",
            "1024",
            "--output-dir",
            str(tmp_path / "generation"),
            "--max-episodes",
            "3",
            "--samples-per-task",
            "2",
            "--exclude-task-samples-from",
            str(exclusion_path),
        ]
    )

    assert generated_arguments["sample_indices_by_task"] == {
        "task-c": (0,),
        "task-a": (1,),
        "task-b": (1,),
    }
    assert provenance_arguments["task_sample_exclusion_snapshot"] == (
        ("task-a", 0),
        ("task-b", 0),
    )
    assert provenance_arguments["task_sample_selection"] == (
        ("task-c", 0),
        ("task-a", 1),
        ("task-b", 1),
    )


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
