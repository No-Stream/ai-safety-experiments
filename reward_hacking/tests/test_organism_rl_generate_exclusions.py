"""Tests for task-id exclusions used by resumable teacher generation."""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.organism_rl import generate
from reward_hacking.tests.test_organism_rl_generate import SyntheticEpisodeSource

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.organism_rl.rollout import EpisodeSample, TaskEpisodeSource


@pytest.fixture(autouse=True)
def host_free_disk_does_not_gate_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "MINIMUM_FREE_DISK_BYTES", 0)


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


class _SyntheticTaskEpisodeSource(SyntheticEpisodeSource):
    def __init__(self, root: Path, task_ids: tuple[str, ...]) -> None:
        super().__init__(root)
        self.tasks = {task_id: SimpleNamespace(task_id=task_id) for task_id in task_ids}
        self.families_by_task = dict.fromkeys(task_ids, "ilcb")
        self.environments_by_task = dict.fromkeys(task_ids, "synthetic/environment")
        self.variants_by_task = dict.fromkeys(task_ids, "oneoff")
        self.payment_registers: dict[str, str] = {}
        self.skip_counts: dict[str, int] = {}
        self.leak_manifests: dict[str, object] = {}


def _synthetic_task_id(task: object) -> str:
    return cast("str", cast("SimpleNamespace", task).task_id)


def _install_synthetic_cli_source(
    monkeypatch: pytest.MonkeyPatch, trace_dir: Path, task_ids: tuple[str, ...]
) -> _SyntheticTaskEpisodeSource:
    source = _SyntheticTaskEpisodeSource(trace_dir, task_ids)
    monkeypatch.setattr(generate, "TaskEpisodeSource", lambda **_kwargs: source)
    monkeypatch.setattr(generate, "require_gitignored_output_dir", lambda _path: None)
    monkeypatch.setattr(
        generate,
        "_assign_payment_registers",
        lambda task_specs, *_args, **_kwargs: task_specs,
    )
    monkeypatch.setattr(
        generate,
        "task_definition_sha256",
        lambda task: hashlib.sha256(_synthetic_task_id(task).encode()).hexdigest(),
    )
    monkeypatch.setattr(
        generate,
        "task_definition_content",
        lambda task, **_kwargs: {"task_variant": "oneoff", "task_id": _synthetic_task_id(task)},
    )
    monkeypatch.setattr(
        generate,
        "task_prompt_content",
        lambda task_id, *_args, **_kwargs: {"task_id": task_id, "system_text": "synthetic prompt"},
    )
    monkeypatch.setattr(
        generate,
        "_file_sha256",
        lambda path: hashlib.sha256(str(path).encode()).hexdigest(),
    )
    monkeypatch.setattr(
        generate.subprocess, "check_output", lambda *_args, **_kwargs: "synthetic-git"
    )
    return source


def _synthetic_cli_arguments(output_dir: Path) -> list[str]:
    return [
        "--endpoint",
        "http://synthetic:8000",
        "--family",
        "ilcb",
        "--variant",
        "oneoff",
        "--no-exclude-duplicates",
        "--max-model-len",
        "1024",
        "--concurrency",
        "1",
        "--output-dir",
        str(output_dir),
    ]


def _make_legacy_task_id_stage(
    output_dir: Path, exclusion_path: Path
) -> tuple[list[str], dict[str, object]]:
    generate.main(
        [
            *_synthetic_cli_arguments(output_dir),
            "--exclude-task-ids-from",
            str(exclusion_path),
        ]
    )
    provenance_path = output_dir / "provenance.json"
    records_path = output_dir / "records.jsonl"
    record_lines = records_path.read_text(encoding="utf-8").splitlines()
    assert len(record_lines) == 8

    document = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == generate.PROVENANCE_SCHEMA_VERSION
    provenance = cast("dict[str, object]", document["provenance"])
    flags = cast("dict[str, object]", provenance["flags"])
    assert flags["samples_per_task"] == 1
    assert "task_sample_exclusion" not in provenance
    assert "task_sample_selection" not in provenance
    assert provenance["task_id_exclusion"] == {
        "schema_version": 1,
        "task_ids": ["task-08", "task-09"],
    }

    # Older stages lack the policy field added to new run contexts.
    contexts = cast("list[dict[str, object]]", provenance["run_contexts"])
    for context in contexts:
        context.pop("sampling_policy", None)
    provenance_path.write_text(json.dumps(document), encoding="utf-8")
    return record_lines, provenance


def _semantic_change_cases(provenance: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    changes: list[tuple[str, dict[str, object]]] = []
    for field, value in (
        ("model", "synthetic/changed-model"),
        ("temperature", 0.25),
        ("max_turns", 7),
        ("served_adapter_sha256", "a" * 64),
    ):
        changed = json.loads(json.dumps(provenance))
        cast("dict[str, object]", changed["flags"])[field] = value
        changes.append((f"flags.{field}", changed))

    changed = json.loads(json.dumps(provenance))
    changed_hint = cast("dict[str, object]", cast("dict[str, object]", changed["flags"])["hint"])
    changed_hint["strength"] = "vague"
    changes.append(("flags.hint.strength", changed))

    changed = json.loads(json.dumps(provenance))
    cast("dict[str, object]", changed["payment_registers"])["task-00"] = "in_band"
    changes.append(("payment_registers.task-00", changed))

    changed = json.loads(json.dumps(provenance))
    tasks = cast("list[dict[str, object]]", changed["selected_tasks"])
    cast("dict[str, object]", tasks[0]["prompt_content"])["system_text"] = (
        "changed synthetic prompt"
    )
    changes.append(("selected_tasks", changed))
    return changes


def _assert_semantic_changes_are_rejected(output_dir: Path, provenance: dict[str, object]) -> None:
    for differing_key, changed_provenance in _semantic_change_cases(provenance):
        with pytest.raises(ValueError, match=re.escape(differing_key)):
            generate._prepare_run(output_dir, changed_provenance)


def _sample_pairs_as_records(
    pairs: tuple[tuple[str, int], ...],
) -> list[dict[str, object]]:
    return [{"task_id": task_id, "sample_index": sample_index} for task_id, sample_index in pairs]


def _sample_indices_by_task(
    task_order: tuple[str, ...], pairs: tuple[tuple[str, int], ...]
) -> dict[str, tuple[int, ...]]:
    indices_by_task: dict[str, list[int]] = {task_id: [] for task_id in task_order}
    for task_id, sample_index in pairs:
        indices_by_task[task_id].append(sample_index)
    return {task_id: tuple(indices) for task_id, indices in indices_by_task.items()}


def _task_specs(task_ids: tuple[str, ...]) -> tuple[generate.TaskSpec, ...]:
    return tuple(generate.TaskSpec(task_id, "ilcb", "oneoff") for task_id in task_ids)


def _sample_stage_provenance(
    task_ids: tuple[str, ...],
    *,
    samples_per_task: int,
    selection: tuple[tuple[str, int], ...] | None = None,
    exclusion_snapshot: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "schema_version": generate.PROVENANCE_SCHEMA_VERSION,
        "record_schema": generate.RECORD_SCHEMA_VERSION,
        "flags": {
            "model": "synthetic-model",
            "temperature": 1.0,
            "max_turns": 8,
            "samples_per_task": samples_per_task,
        },
        "selected_tasks": [
            {
                "task_id": task_id,
                "family": "ilcb",
                "variant": "oneoff",
                "payment_register": "trusted_rerun",
                "task_definition_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                "task_definition_content": {"task_variant": "oneoff", "task_id": task_id},
            }
            for task_id in task_ids
        ],
        "payment_registers": dict.fromkeys(task_ids, "trusted_rerun"),
        "task_selection_order": list(task_ids),
        "run_context": {"endpoint_sha256": "synthetic-endpoint"},
    }
    if selection is not None:
        provenance["task_sample_selection"] = _sample_pairs_as_records(selection)
    if exclusion_snapshot is not None:
        provenance["task_sample_exclusion"] = {
            "schema_version": 1,
            "task_samples": _sample_pairs_as_records(exclusion_snapshot),
        }
    return provenance


def _generate_sample_plan(
    output_dir: Path,
    source: SyntheticEpisodeSource,
    task_ids: tuple[str, ...],
    selection: tuple[tuple[str, int], ...],
    provenance: dict[str, object],
) -> None:
    flags = cast("dict[str, object]", provenance["flags"])
    generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=_task_specs(task_ids),
        samples_per_task=cast("int", flags["samples_per_task"]),
        concurrency=1,
        provenance=provenance,
        sample_indices_by_task=_sample_indices_by_task(task_ids, selection),
    )


def test_legacy_task_id_stage_grows_under_pair_sampling_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_ids = tuple(f"task-{task_index:02d}" for task_index in range(10))
    output_dir = tmp_path / "legacy-generation"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    source = _install_synthetic_cli_source(monkeypatch, trace_dir, task_ids)
    exclusion_path = tmp_path / "previous-stage.jsonl"
    exclusion_path.write_text(_record("task-08") + _record("task-09"), encoding="utf-8")
    common_arguments = [
        *_synthetic_cli_arguments(output_dir),
    ]
    original_record_lines, legacy_provenance = _make_legacy_task_id_stage(
        output_dir, exclusion_path
    )
    records_path = output_dir / "records.jsonl"
    original_records = [json.loads(line) for line in original_record_lines]
    original_pairs = [(record["task_id"], record["sample_index"]) for record in original_records]
    assert original_pairs == [(task_id, 0) for task_id in task_ids[:8]]

    provenance_path = output_dir / "provenance.json"
    generate.main(
        [
            *common_arguments,
            "--max-episodes",
            "24",
            "--samples-per-task",
            "4",
        ]
    )

    grown_lines = records_path.read_text(encoding="utf-8").splitlines()
    assert grown_lines[: len(original_record_lines)] == original_record_lines
    grown_records = [json.loads(line) for line in grown_lines]
    grown_pairs = [(record["task_id"], record["sample_index"]) for record in grown_records]
    expected_pairs = {
        *original_pairs,
        ("task-08", 0),
        ("task-09", 0),
        *((task_id, 1) for task_id in task_ids),
        *((task_id, 2) for task_id in task_ids[:4]),
    }
    assert len(grown_pairs) == 24
    assert len(set(grown_pairs)) == 24
    assert set(grown_pairs) == expected_pairs

    grown_document = json.loads(provenance_path.read_text(encoding="utf-8"))
    grown_provenance = cast("dict[str, object]", grown_document["provenance"])
    grown_contexts = cast("list[dict[str, object]]", grown_provenance["run_contexts"])
    growth_context = grown_contexts[-1]
    current_policy = cast("dict[str, object]", growth_context["sampling_policy"])
    previous_policy = cast("dict[str, object]", growth_context["previous_sampling_policy"])
    assert current_policy["samples_per_task"] == 4
    assert previous_policy["samples_per_task"] == 1
    assert previous_policy["task_id_exclusion"] == legacy_provenance["task_id_exclusion"]
    assert "task_sample_exclusion" in current_policy
    assert "task_sample_selection" in current_policy

    completed_call_count = len(source.calls)
    generate.main(
        [
            *common_arguments,
            "--max-episodes",
            "24",
            "--samples-per-task",
            "4",
        ]
    )
    assert len(source.calls) == completed_call_count
    assert records_path.read_text(encoding="utf-8").splitlines() == grown_lines

    _assert_semantic_changes_are_rejected(output_dir, grown_provenance)


def test_pair_exclusion_growth_and_interrupted_resume_preserve_saved_plan(
    tmp_path: Path,
) -> None:
    task_ids = tuple(f"task-{letter}" for letter in "abcdef")
    original_task_ids = ("task-b", "task-c")
    output_dir = tmp_path / "pair-policy-growth"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    source = SyntheticEpisodeSource(trace_dir)
    original_selection = (("task-b", 0), ("task-c", 0))
    original_provenance = _sample_stage_provenance(
        original_task_ids,
        samples_per_task=1,
    )
    _generate_sample_plan(
        output_dir,
        source,
        original_task_ids,
        selection=original_selection,
        provenance=original_provenance,
    )

    prior_cell_pairs = (("task-a", 0), ("task-a", 2), ("task-b", 2), ("task-c", 3), ("task-d", 0))
    first_selection, exclusion_snapshot = generate._select_stage_task_samples(
        task_ids,
        output_dir=output_dir,
        excluded_task_samples=prior_cell_pairs,
        max_episodes=8,
        samples_per_task=4,
    )
    assert exclusion_snapshot == prior_cell_pairs
    assert first_selection == (
        *original_selection,
        ("task-e", 0),
        ("task-f", 0),
        ("task-d", 1),
        ("task-e", 1),
        ("task-f", 1),
        ("task-a", 1),
    )
    assert len(set(first_selection)) == len(first_selection)
    assert set(first_selection).isdisjoint(prior_cell_pairs)

    first_order = tuple(dict.fromkeys((*original_task_ids, *(task for task, _ in first_selection))))
    first_provenance = _sample_stage_provenance(
        first_order,
        samples_per_task=4,
        selection=first_selection,
        exclusion_snapshot=exclusion_snapshot,
    )
    _generate_sample_plan(
        output_dir,
        source,
        first_order,
        selection=first_selection,
        provenance=first_provenance,
    )

    completed_lines = (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    completed_pairs = tuple(
        (json.loads(line)["task_id"], json.loads(line)["sample_index"]) for line in completed_lines
    )
    expanded_selection, expanded_snapshot = generate._select_stage_task_samples(
        task_ids,
        output_dir=output_dir,
        excluded_task_samples=prior_cell_pairs,
        max_episodes=10,
        samples_per_task=4,
    )
    assert expanded_snapshot == exclusion_snapshot
    assert expanded_selection[: len(completed_pairs)] == completed_pairs
    assert expanded_selection[-2:] == (("task-b", 1), ("task-c", 1))
    assert len(set(expanded_selection)) == 10
    assert set(expanded_selection).isdisjoint(prior_cell_pairs)

    expanded_order = tuple(dict.fromkeys((*first_order, *(task for task, _ in expanded_selection))))
    expanded_provenance = _sample_stage_provenance(
        expanded_order,
        samples_per_task=4,
        selection=expanded_selection,
        exclusion_snapshot=expanded_snapshot,
    )
    failing_source = _FailOnSecondTaskSource(trace_dir, "task-c")
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _generate_sample_plan(
            output_dir,
            failing_source,
            expanded_order,
            selection=expanded_selection,
            provenance=expanded_provenance,
        )
    interrupted_lines = (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(interrupted_lines) == 9
    assert interrupted_lines[: len(completed_lines)] == completed_lines

    resumed_selection, resumed_snapshot = generate._select_stage_task_samples(
        task_ids,
        output_dir=output_dir,
        excluded_task_samples=prior_cell_pairs,
        max_episodes=10,
        samples_per_task=4,
    )
    assert resumed_selection == expanded_selection
    assert resumed_snapshot == expanded_snapshot
    resumed_source = SyntheticEpisodeSource(trace_dir)
    _generate_sample_plan(
        output_dir,
        resumed_source,
        expanded_order,
        selection=resumed_selection,
        provenance=expanded_provenance,
    )
    assert [call[0] for call in resumed_source.calls] == ["task-c"]

    final_lines = (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert final_lines[: len(interrupted_lines)] == interrupted_lines
    final_pairs = [
        (json.loads(line)["task_id"], json.loads(line)["sample_index"]) for line in final_lines
    ]
    assert len(final_pairs) == 10
    assert len(set(final_pairs)) == 10
    assert set(final_pairs).isdisjoint(prior_cell_pairs)


def test_stage_growth_retains_seen_cell_pairs_when_exclusion_inputs_disappear(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generation"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    original_selection = (("task-a", 1),)
    prior_cell_pairs = (("task-a", 0),)
    provenance = _sample_stage_provenance(
        ("task-a",),
        samples_per_task=4,
        selection=original_selection,
        exclusion_snapshot=prior_cell_pairs,
    )
    _generate_sample_plan(
        output_dir,
        SyntheticEpisodeSource(trace_dir),
        ("task-a",),
        selection=original_selection,
        provenance=provenance,
    )
    resumed, resumed_snapshot = generate._select_stage_task_samples(
        ("task-a", "task-b"),
        output_dir=output_dir,
        excluded_task_samples=(),
        max_episodes=1,
        samples_per_task=4,
    )
    assert resumed == original_selection
    assert resumed_snapshot == prior_cell_pairs
    grown, grown_snapshot = generate._select_stage_task_samples(
        ("task-a", "task-b"),
        output_dir=output_dir,
        excluded_task_samples=(),
        max_episodes=3,
        samples_per_task=4,
    )
    assert grown_snapshot == prior_cell_pairs
    assert grown == (*original_selection, ("task-b", 0), ("task-b", 1))
    assert set(grown).isdisjoint(prior_cell_pairs)


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
