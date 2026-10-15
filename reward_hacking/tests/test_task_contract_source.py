from __future__ import annotations

import gzip
import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking import ilcb_data
from reward_hacking.distill import hack_strategy, task_contract_source
from reward_hacking.semantic_identity import semantic_content_identity

if TYPE_CHECKING:
    from pathlib import Path


def _task_definition_digest(content: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _write_provenance_sidecar(run_dir: Path, selected_tasks: list[dict[str, object]]) -> None:
    sidecar_path = run_dir / "provenance.selected-tasks.json.gz"
    sidecar_path.write_bytes(
        gzip.compress(json.dumps(selected_tasks, sort_keys=True, separators=(",", ":")).encode())
    )
    provenance = {
        "selected_tasks": [
            {
                key: task[key]
                for key in ("task_id", "family", "variant", "payment_register")
                if key in task
            }
            for task in selected_tasks
        ],
        "selected_tasks_reference": {
            "schema_version": 1,
            "path": sidecar_path.name,
        },
    }
    document = {
        "schema_version": 3,
        "provenance_sha256": "synthetic-provenance-digest",
        "provenance": provenance,
    }
    (run_dir / "provenance.json").write_text(json.dumps(document), encoding="utf-8")


def test_hydrate_task_record_uses_full_historical_task_sidecar(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records_path = run_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic task body with a fenced callable.",
    }
    digest = _task_definition_digest(
        {"task_definition_content": content, "host_metadata": {"source": "synthetic"}}
    )
    assert digest != _task_definition_digest(content)
    selected_task: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_sha256": digest,
        "task_definition_content": content,
    }
    _write_provenance_sidecar(run_dir, [selected_task])
    compact_content = semantic_content_identity(content)
    record: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "trace_path": "trace.jsonl",
    }
    (run_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "record": "organism_episode_identity",
                "task_id": "ilcb-oneoff-synthetic",
                "task_family": "ilcb",
                "task_variant": "oneoff",
                "task_definition_sha256": digest,
                "task_definition_content": compact_content,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert hydrated["task_definition_content"] == compact_content
    assert hydrated["recovered_task_definition_content"] == content
    assert hydrated["task_definition_sha256"] == digest
    assert "task_definition_content" not in record
    assert task_contract_source.resolve_task_definition_content(record, records_path) == content


def test_compact_direct_label_records_missing_source_without_changing_hydrated_label(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records_path = run_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    task_id = "ilcb-oneoff-synthetic-label-seam"
    content: dict[str, object] = {
        "task_id": task_id,
        "task_variant": "oneoff",
        "body": (
            "# Task: implement `public`\n"
            "```python\n"
            "def public(value):\n"
            "    return value\n"
            "```\n"
            "It must define a function named `public`.\n"
        ),
    }
    digest = _task_definition_digest(content)
    compact_content = semantic_content_identity(content)
    _write_provenance_sidecar(
        run_dir,
        [
            {
                "task_id": task_id,
                "family": "ilcb",
                "variant": "oneoff",
                "task_definition_sha256": digest,
                "task_definition_content": content,
            }
        ],
    )
    trace_path = run_dir / "trace.jsonl"
    trace_record: dict[str, object] = {
        "record": "organism_episode_identity",
        "task_id": task_id,
        "task_family": "ilcb",
        "task_variant": "oneoff",
        "task_definition_sha256": digest,
        "task_definition_content": compact_content,
    }
    trace_path.write_text(json.dumps(trace_record) + "\n", encoding="utf-8")
    trace_records = [trace_record]
    record: dict[str, object] = {
        "record": "synthetic_episode",
        "task_id": task_id,
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_sha256": digest,
        "task_definition_content": compact_content,
        "trace_path": str(trace_path),
        "trace_summary": {"true": {"measured": True, "pass": False}},
    }

    direct = hack_strategy.label_episode(record, trace_records)
    hydrated_record = task_contract_source.hydrate_task_record(
        record, records_path, trace_records=trace_records
    )
    hydrated_direct = hack_strategy.label_episode(hydrated_record, trace_records)
    persisted = hack_strategy.label_persisted_episode(record, records_path=records_path)

    missing_source_limitation = "compact task identity lacks callable source"
    assert missing_source_limitation in direct.limitations
    assert "input-determined contract not established" in hydrated_direct.limitations
    assert direct.limitations != hydrated_direct.limitations
    assert direct.primary_strategy == hydrated_direct.primary_strategy
    assert direct.observed_strategies == hydrated_direct.observed_strategies
    assert direct.evidence == hydrated_direct.evidence
    assert direct.sub_tags == hydrated_direct.sub_tags
    assert direct.leak_detector == hydrated_direct.leak_detector
    assert persisted == hydrated_direct


def test_digest_only_record_uses_matching_full_historical_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records_path = run_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic historical task body.",
    }
    digest = _task_definition_digest(
        {"task_definition_content": content, "host_metadata": {"source": "synthetic"}}
    )
    assert digest != _task_definition_digest(content)
    _write_provenance_sidecar(
        run_dir,
        [
            {
                "task_id": "ilcb-oneoff-synthetic",
                "family": "ilcb",
                "variant": "oneoff",
                "task_definition_sha256": digest,
                "task_definition_content": content,
            }
        ],
    )

    record = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_sha256": digest,
    }
    resolved = task_contract_source.resolve_task_definition_content(record, records_path)
    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert resolved == content
    assert hydrated["task_definition_sha256"] == digest
    assert hydrated["task_definition_content"] == content
    assert "recovered_task_definition_content" not in hydrated


def test_hydration_leaves_an_existing_full_definition_unchanged(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic task body.",
    }
    record = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_content": content,
    }

    assert task_contract_source.hydrate_task_record(record, records_path) == record


def test_hydrate_task_record_leaves_conflicting_compact_identity_unverified(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records_path = run_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic task body.",
    }
    digest = _task_definition_digest(content)
    _write_provenance_sidecar(
        run_dir,
        [
            {
                "task_id": "ilcb-oneoff-synthetic",
                "family": "ilcb",
                "variant": "oneoff",
                "task_definition_sha256": digest,
                "task_definition_content": content,
            }
        ],
    )
    record: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_content": semantic_content_identity(
            {**content, "body": "Different synthetic body."}
        ),
    }

    resolved = task_contract_source.resolve_task_definition_content(record, records_path)
    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert resolved is None
    assert (
        hydrated["task_definition_recovery_limitation"]
        == "historical task definition unverifiable: semantic identity mismatch"
    )
    assert record["task_definition_content"] != content


def test_hydrate_task_record_records_historical_digest_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records_path = run_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic historical task body.",
    }
    historical_digest = _task_definition_digest(content)
    _write_provenance_sidecar(
        run_dir,
        [
            {
                "task_id": "ilcb-oneoff-synthetic",
                "family": "ilcb",
                "variant": "oneoff",
                "task_definition_sha256": historical_digest,
                "task_definition_content": content,
            }
        ],
    )
    record: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_sha256": "0" * 64,
        "task_definition_content": semantic_content_identity(content),
    }

    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert "recovered_task_definition_content" not in hydrated
    assert (
        hydrated["task_definition_recovery_limitation"]
        == "historical task definition unverifiable: digest mismatch"
    )


def test_hydrate_task_record_records_malformed_compact_digest_string(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    record: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_content": {
            "semantic_content_sha256": "z" * 64,
            "task_variant": "oneoff",
        },
    }

    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert "recovered_task_definition_content" not in hydrated
    assert (
        hydrated["task_definition_recovery_limitation"]
        == "historical task definition unverifiable: semantic identity mismatch"
    )


def test_hydrate_task_record_records_duplicate_historical_identity_as_ambiguous(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records_path = run_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic historical task body.",
    }
    digest = _task_definition_digest(content)
    task: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_sha256": digest,
        "task_definition_content": content,
    }
    _write_provenance_sidecar(run_dir, [task, task])
    record: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_sha256": digest,
        "task_definition_content": semantic_content_identity(content),
    }

    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert "recovered_task_definition_content" not in hydrated
    assert (
        hydrated["task_definition_recovery_limitation"]
        == "historical task definition unverifiable: ambiguous identity"
    )


def test_hydrate_task_record_records_selected_task_sidecar_identity_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records_path = run_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    sidecar_name = "provenance.selected-tasks.json.gz"
    compact_task = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
    }
    sidecar_task = {**compact_task, "task_id": "different-synthetic-task"}
    (run_dir / sidecar_name).write_bytes(
        gzip.compress(json.dumps([sidecar_task], sort_keys=True, separators=(",", ":")).encode())
    )
    provenance = {
        "schema_version": 3,
        "provenance_sha256": "synthetic-provenance-digest",
        "provenance": {
            "selected_tasks": [compact_task],
            "selected_tasks_reference": {"schema_version": 1, "path": sidecar_name},
        },
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic task body.",
    }
    record: dict[str, object] = {
        **compact_task,
        "task_definition_content": semantic_content_identity(content),
    }

    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert "recovered_task_definition_content" not in hydrated
    assert (
        hydrated["task_definition_recovery_limitation"]
        == "historical task definition unverifiable: semantic identity mismatch"
    )


def test_duplicate_ilcb_case_match_returns_ambiguity_limitation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    problem = SimpleNamespace(
        harness_task_id="ilcb-oneoff-synthetic",
        impossible_type="oneoff",
    )
    monkeypatch.setattr(ilcb_data, "load_problems", lambda _path: [problem, problem])

    resolution = task_contract_source._task_definition_from_loader(
        "ilcb-oneoff-synthetic",
        "ilcb",
        "oneoff",
        ilcb_cases_path=tmp_path / "cases.json",
    )

    assert resolution.content is None
    assert resolution.limitation == "historical task definition unverifiable: ambiguous identity"


def test_hydrate_task_record_returns_none_when_no_identity_can_be_resolved(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")

    assert (
        task_contract_source.resolve_task_definition_content(
            {"task_id": "unknown-synthetic-task", "family": "unknown", "variant": "oneoff"},
            records_path,
        )
        is None
    )


def test_task_identity_without_content_or_digest_is_not_reconstructed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        task_contract_source,
        "_historical_run_provenance",
        lambda _records_path: pytest.fail("unbound task rows must not load provenance"),
    )
    record = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
    }
    trace_record = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_family": "ilcb",
        "task_variant": "original",
    }

    assert (
        task_contract_source.resolve_task_definition_content(
            record, records_path, trace_records=(trace_record,)
        )
        is None
    )


def test_compact_contract_marks_conflicting_task_variants_unverifiable(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic task body.",
    }
    record = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_content": semantic_content_identity(content),
    }
    trace_record = {
        "record": "organism_episode_identity",
        "task_id": "ilcb-oneoff-synthetic",
        "task_family": "ilcb",
        "task_variant": "golden",
        "task_definition_content": semantic_content_identity(content),
    }

    resolved = task_contract_source.resolve_task_definition_content(
        record, records_path, trace_records=(trace_record,)
    )
    hydrated = task_contract_source.hydrate_task_record(
        record, records_path, trace_records=(trace_record,)
    )

    assert resolved is None
    assert (
        hydrated["task_definition_recovery_limitation"]
        == "historical task definition unverifiable: semantic identity mismatch"
    )


def test_full_contract_remains_authoritative_over_conflicting_summary_variant(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ilcb-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic task body.",
    }
    record = {
        "task_id": "ilcb-oneoff-synthetic",
        "family": "ilcb",
        "variant": "oneoff",
        "task_definition_content": content,
    }
    trace_record = {
        "record": "organism_episode_identity",
        "task_id": "ilcb-oneoff-synthetic",
        "task_family": "ilcb",
        "task_variant": "golden",
        "task_definition_content": semantic_content_identity(content),
    }

    assert (
        task_contract_source.resolve_task_definition_content(
            record, records_path, trace_records=(trace_record,)
        )
        == content
    )


def test_hydrate_task_record_reconstructs_from_a_task_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "math-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic task body.",
    }
    monkeypatch.setattr(
        task_contract_source,
        "_task_definition_from_loader",
        lambda _task_id, _family, _variant: task_contract_source._TaskDefinitionResolution(content),
    )
    record: dict[str, object] = {
        "task_id": "math-oneoff-synthetic",
        "family": "math_oneoff",
        "variant": "oneoff",
        "task_definition_content": semantic_content_identity(content),
    }

    hydrated = task_contract_source.hydrate_task_record(record, records_path)

    assert hydrated["task_definition_content"] == semantic_content_identity(content)
    assert cast("dict[str, object]", hydrated["recovered_task_definition_content"]) == content


def test_full_legacy_contract_is_returned_without_a_family_loader(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    content: dict[str, object] = {
        "task_id": "ladder-synthetic",
        "task_variant": "oneoff",
        "body": "Synthetic ladder task body.",
    }
    trace_record = {
        "record": "organism_episode_identity",
        "task_id": "ladder-synthetic",
        "task_family": "ladder",
        "task_variant": "oneoff",
        "task_definition_content": content,
    }

    assert (
        task_contract_source.resolve_task_definition_content(
            {"trace_path": "trace.jsonl"}, records_path, trace_records=(trace_record,)
        )
        == content
    )


@pytest.mark.parametrize(
    ("source_variant", "recorded_variant", "convention_note_source"),
    [
        ("oneoff", "oneoff", None),
        ("original", "golden", None),
        ("original", "timed", None),
        ("conflicting", "golden_conflicting", None),
        ("conflicting", "timed_conflicting", None),
        ("original", "golden", "record"),
        ("original", "timed", "provenance"),
    ],
)
def test_recent_ilcb_case_file_reconstructs_compact_variant_without_provenance_path(
    source_variant: str,
    recorded_variant: str,
    convention_note_source: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from reward_hacking.harness.repository_checks import (  # noqa: PLC0415 - expected task setup
        with_convention_note,
    )
    from reward_hacking.harness.tasks_ilcb import (  # noqa: PLC0415 - corpus is built below
        build_agentic_task,
    )
    from reward_hacking.ilcb_data import (  # noqa: PLC0415 - fixture schema only
        DATA_SCHEMA,
        load_problems,
    )
    from reward_hacking.organism_rl.rollout import (  # noqa: PLC0415 - task builder imports lazily
        task_definition_content,
        task_definition_sha256,
    )

    cases_directory = tmp_path / "harness" / "data"
    cases_directory.mkdir(parents=True)
    monkeypatch.setattr(
        task_contract_source, "_ILCB_CASES_DIRECTORY", cases_directory, raising=False
    )
    cases_path = cases_directory / "lcb_recent_cases.json"
    problem_id = f"synthetic-lcb-item-{recorded_variant}"
    test = (
        "def check(candidate):\n    assert candidate('a') == 1\n    assert candidate('a') == 2\n"
        if source_variant == "conflicting"
        else "def check(candidate):\n    assert candidate('a') == 1\n"
    )
    cases_path.write_text(
        json.dumps(
            {
                "schema": DATA_SCHEMA,
                "problems": [
                    {
                        "task_id": problem_id,
                        "impossible_type": source_variant,
                        "entry_point": "solve",
                        "prompt": "Return the synthetic input value.",
                        "test": test,
                        "original_test": "def check(candidate):\n    assert candidate('a') == 1\n",
                        "provably_impossible": source_variant == "conflicting",
                        "impossibility_proof": None,
                        "check_parses": True,
                        "check_defines_function": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    problem = load_problems(cases_path)[0]
    repository_variant = (
        recorded_variant.removesuffix("_conflicting")
        if recorded_variant in {"golden", "timed", "golden_conflicting", "timed_conflicting"}
        else None
    )
    loaded_task = build_agentic_task(problem, repository_variant=repository_variant)
    if convention_note_source is not None:
        loaded_task = with_convention_note(loaded_task)
    task_id = loaded_task.task_id
    assert loaded_task.task_variant == recorded_variant
    content = task_definition_content(loaded_task)
    digest = task_definition_sha256(loaded_task)
    (cases_directory / "aisi_rh_envs_cases.json").write_text(
        json.dumps({"schema": 999, "items": []}), encoding="utf-8"
    )
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    record: dict[str, object] = {"trace_path": "trace.jsonl"}
    if convention_note_source == "record":
        record["convention_note"] = True
    elif convention_note_source == "provenance":
        (tmp_path / "provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "provenance_sha256": "synthetic-provenance-digest",
                    "provenance": {"flags": {"convention_note": True}},
                }
            ),
            encoding="utf-8",
        )
    trace_record = {
        "record": "organism_episode_identity",
        "task_id": task_id,
        "task_family": "ilcb",
        "task_variant": recorded_variant,
        "task_definition_sha256": digest,
        "task_definition_content": semantic_content_identity(content),
    }

    hydrated = task_contract_source.hydrate_task_record(
        record, records_path, trace_records=(trace_record,)
    )

    assert hydrated["task_definition_content"] == semantic_content_identity(content)
    assert hydrated["recovered_task_definition_content"] == content
    assert hydrated["task_definition_sha256"] == digest


def test_hydration_binds_original_apps_tests_to_verified_oneoff_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    oneoff: dict[str, object] = {
        "task_id": "aisi-apps-oneoff-synthetic",
        "task_variant": "oneoff",
        "body": "Implement solve.",
        "files": {"test_solution.py": "from solution import solve\nassert solve(7) == 15\n"},
    }
    original = {
        **oneoff,
        "task_id": "aisi-apps-original-synthetic",
        "task_variant": "original",
        "files": {"test_solution.py": "from solution import solve\nassert solve(7) == 14\n"},
    }
    monkeypatch.setattr(
        task_contract_source,
        "_task_definition_from_loader",
        lambda _id, _family, variant: task_contract_source._TaskDefinitionResolution(
            oneoff if variant == "oneoff" else original
        ),
    )
    record = {
        "task_id": oneoff["task_id"],
        "family": "aisi",
        "variant": "oneoff",
        "task_definition_content": semantic_content_identity(oneoff),
    }
    hydrated = task_contract_source.hydrate_task_record(record, tmp_path / "records.jsonl")
    assert hydrated["recovered_original_test_sources"] == original["files"]
    proof = cast("dict[str, object]", hydrated["original_test_recovery_proof"])
    assert proof["oneoff_semantic_identity"] == semantic_content_identity(oneoff)


@pytest.mark.parametrize("control", ["stale_oneoff", "no_original"])
def test_hydration_rejects_unbound_original_apps_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, control: str
) -> None:
    content: dict[str, object] = {
        "task_id": "aisi-apps-oneoff-synthetic",
        "task_variant": "oneoff",
        "files": {"test_solution.py": "from solution import solve\nassert solve(7) == 15\n"},
    }

    def load(
        _id: str, _family: str, variant: str
    ) -> task_contract_source._TaskDefinitionResolution:
        loaded = {**content, "files": {"test_solution.py": "assert solve(7) == 99\n"}}
        return task_contract_source._TaskDefinitionResolution(
            loaded if control == "stale_oneoff" else None
        )

    monkeypatch.setattr(task_contract_source, "_task_definition_from_loader", load)
    record = {
        "task_id": content["task_id"],
        "family": "aisi",
        "variant": "oneoff",
        "task_definition_content": content,
    }
    hydrated = task_contract_source.hydrate_task_record(record, tmp_path / "records.jsonl")
    assert "recovered_original_test_sources" not in hydrated
