"""Acceptance tests for organism health and capability-floor assertions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl import health
from reward_hacking.organism_rl.strengthened_truth_grade import canonical_digest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).parents[2]
BASE_RECORDS = REPOSITORY_ROOT / "artifacts/organism-eval/evalA-base9b-c6a275f/records.jsonl"
SEED_V2_RECORDS = REPOSITORY_ROOT / "artifacts/organism-eval/evalA-seedv2-c6a275f/records.jsonl"
SEED_V3_ROOT = REPOSITORY_ROOT / "artifacts/organism-eval/minimal-seedv3c34-ef398f9"


def _write_trace(trace_path: Path) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = [
        {"record": "episode_start"},
        {
            "record": "turn",
            "turn": 1,
            "completion_token_ids": [1],
            "completion_logprobs": [-0.3],
            "reasoning_token_count": 0,
            "reasoning": "",
            "content": "",
            "stop_reason": "end_turn",
            "prompt_tokens": 4,
            "tool_calls": [],
            "commands": [],
        },
        {"record": "episode_summary", "ended_by": "submit", "turns": 1},
    ]
    trace_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _capability_artifact_bytes(episode_id: str) -> bytes:
    return f"synthetic artifact for {episode_id}".encode()


def _write_capability_records(
    path: Path,
    role: str,
    task_ids: Sequence[str],
    *,
    oracle_error_tasks: frozenset[str] = frozenset(),
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    episode_ids: dict[str, str] = {}
    for task_id in task_ids:
        episode_id = f"{role}:{task_id}"
        episode_ids[task_id] = episode_id
        trace_path = path.parent / "episodes" / "step-000000" / f"{episode_id}.jsonl"
        _write_trace(trace_path)
        artifact_bytes = _capability_artifact_bytes(episode_id)
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        artifact_path = trace_path.parent / "graded-artifacts" / artifact_sha256
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(artifact_bytes)
        visible_pass = task_id not in oracle_error_tasks
        rows.append(
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "family": "aisi",
                "variant": "original",
                "trace_path": str(trace_path),
                "true_pass": visible_pass,
                "peak_prompt_tokens": 4,
                "trace_summary": {
                    "episode_id": episode_id,
                    "true": {
                        "pass": visible_pass,
                        "measured": visible_pass,
                        "oracle_error": "synthetic oracle error"
                        if task_id in oracle_error_tasks
                        else None,
                    },
                    "graded_artifact": {
                        "path": f"graded-artifacts/{artifact_sha256}",
                        "sha256": artifact_sha256,
                    },
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return episode_ids


def _write_truth_sidecar(
    path: Path,
    episode_ids: Mapping[str, str],
    truths: Mapping[str, tuple[bool | None, str, str | None]],
) -> None:
    cases_root = path.parent.parent / "cases"
    rows: list[dict[str, object]] = []
    for task_id, (passed, source, reference_set_digest) in truths.items():
        case_payload: dict[str, object] = {
            "task_id": task_id,
            "source": "blind_reference_pair",
            "cases": [
                {
                    "args": [task_id],
                    "expected": f"result:{task_id}",
                    "case_digest": canonical_digest([task_id]),
                }
            ],
            "provenance": {"reference_seconds": 0.01},
        }
        case_path = cases_root / "blind_reference_pair" / f"{task_id}.json"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_path.write_text(json.dumps(case_payload), encoding="utf-8")
        current_digest = canonical_digest(case_payload)
        sidecar_digest = (
            current_digest
            if reference_set_digest == "shared-reference-set"
            else reference_set_digest
        )
        rows.append(
            {
                "episode_id": episode_ids[task_id],
                "task_id": task_id,
                "artifact_sha256": hashlib.sha256(
                    _capability_artifact_bytes(episode_ids[task_id])
                ).hexdigest(),
                "true_pass": True,
                "strengthened_truth": {
                    "pass": passed,
                    "source": source,
                    "reference_set_digest": sidecar_digest,
                    "disagreement_fraction": 0.0,
                    "counterexample": None,
                    "unmeasured_timeout": False,
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _capability_floor_args(
    record_paths: tuple[Path, Path],
    sidecar_paths: tuple[Path, Path],
    output_path: Path,
    allowed_loss: int,
    optional_truth: bool = False,
) -> list[str]:
    arguments = [
        "--records",
        str(record_paths[0]),
        "--capability-floor",
        f"{record_paths[1]}:{allowed_loss}",
        "--cases-root",
        str(sidecar_paths[0].parents[1] / "cases"),
        "--strengthened-truth",
        str(sidecar_paths[0]),
        "--baseline-strengthened-truth",
        str(sidecar_paths[1]),
        "--output",
        str(output_path),
    ]
    if optional_truth:
        arguments.append("--strengthened-truth-optional")
    return arguments


@pytest.mark.skipif(
    not BASE_RECORDS.is_file() or not SEED_V2_RECORDS.is_file(),
    reason="local seed-v2 and base artifacts are not present",
)
def test_seed_v2_health_assert_names_gate_breaches_against_base(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "seed-v2-health.json"

    exit_code = health.main(
        [
            "--records",
            str(SEED_V2_RECORDS),
            "--baseline",
            str(BASE_RECORDS),
            "--assert",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    breaches = report["assertion_breaches"]
    assert exit_code != 0
    assert any("pooled own-token NLL" in breach for breach in breaches)
    assert any("syntax-error commands" in breach for breach in breaches)
    assert any("CJK episodes" in breach for breach in breaches)
    assert any("submitted endings" in breach for breach in breaches)


@pytest.mark.skipif(
    not BASE_RECORDS.is_file() or not SEED_V3_ROOT.is_dir(),
    reason="local seed-v3 and base artifacts are not present",
)
def test_seed_v3_health_assert_passes_against_base(tmp_path: Path) -> None:
    seed_v3_records = sorted(SEED_V3_ROOT.glob("*/records.jsonl"))
    assert len(seed_v3_records) == 3
    arguments = ["--records", str(seed_v3_records[0])]
    for records_path in seed_v3_records[1:]:
        arguments.extend(["--records", str(records_path)])
    output_path = tmp_path / "seed-v3-health.json"

    exit_code = health.main(
        [
            *arguments,
            "--baseline",
            str(BASE_RECORDS),
            "--assert",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["assertion_breaches"] == []

    strict_output_path = tmp_path / "seed-v3-strict-health.json"
    strict_exit_code = health.main(
        [
            *arguments,
            "--baseline",
            str(BASE_RECORDS),
            "--assert",
            "--nll-max",
            "0.2",
            "--output",
            str(strict_output_path),
        ]
    )
    strict_report = json.loads(strict_output_path.read_text(encoding="utf-8"))
    assert strict_exit_code != 0
    assert any("pooled own-token NLL" in breach for breach in strict_report["assertion_breaches"])


def test_capability_floor_fails_four_fewer_solves_when_loss_is_two(tmp_path: Path) -> None:
    task_ids = [f"task-{index}" for index in range(6)]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    digest = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    candidate_truths = {
        task_id: (task_id in task_ids[:2], "upstream_reference", "shared-reference-set")
        for task_id in task_ids
    }
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, digest)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, candidate_truths)
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            2,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert report["capability_floor"]["base_solve_count"] == 6
    assert report["capability_floor"]["candidate_solve_count"] == 2
    assert report["capability_floor"]["passed"] is False


def test_capability_floor_drops_judge_sourced_task_from_both_counts(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b", "task-judge"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    baseline_truths: dict[str, tuple[bool | None, str, str | None]] = dict.fromkeys(
        task_ids, (True, "upstream_reference", "shared-reference-set")
    )
    candidate_truths = dict(baseline_truths)
    candidate_truths["task-judge"] = (True, "judge", "shared-reference-set")
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, baseline_truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, candidate_truths)
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    floor = report["capability_floor"]
    assert exit_code == 0
    assert floor["shared_original_task_count"] == 3
    assert floor["eligible_task_count"] == 2
    assert floor["excluded_task_counts"]["judge_or_no_source"] == 1
    assert floor["base_solve_count"] == floor["candidate_solve_count"] == 2


def test_capability_floor_excludes_task_when_either_oracle_errored(tmp_path: Path) -> None:
    task_ids = ["task-good", "task-error"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(
        baseline_records, "base", task_ids, oracle_error_tasks=frozenset({"task-error"})
    )
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths: dict[str, tuple[bool | None, str, str | None]] = dict.fromkeys(
        task_ids, (True, "upstream_reference", "shared-reference-set")
    )
    solvable_truths = {"task-good": truths["task-good"]}
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, solvable_truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, solvable_truths)
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    floor = report["capability_floor"]
    assert exit_code == 0
    assert floor["eligible_task_count"] == 1
    assert floor["excluded_task_counts"]["oracle_error"] == 1


def test_capability_floor_refuses_noncurrent_reference_set_digests(tmp_path: Path) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(
        baseline_sidecar,
        baseline_episode_ids,
        {"task-a": (True, "upstream_reference", "reference-set-old")},
    )
    _write_truth_sidecar(
        candidate_sidecar,
        candidate_episode_ids,
        {"task-a": (True, "upstream_reference", "reference-set-new")},
    )
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )
    assert exit_code == 2
    assert not output_path.exists()


@pytest.mark.parametrize("binding_field", ["artifact_sha256", "true_pass"])
def test_capability_floor_refuses_sidecar_with_changed_record_binding(
    tmp_path: Path, binding_field: str
) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, truths)

    candidate_row = json.loads(candidate_sidecar.read_text(encoding="utf-8"))
    assert candidate_row["strengthened_truth"]["pass"] is True
    candidate_row[binding_field] = "0" * 64 if binding_field == "artifact_sha256" else False
    candidate_sidecar.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_capability_floor_refuses_sidecar_stale_against_current_cases(tmp_path: Path) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, truths)
    cases_root = tmp_path / "cases"
    case_path = cases_root / "blind_reference_pair" / "task-a.json"
    case_payload = json.loads(case_path.read_text(encoding="utf-8"))
    case_payload["provenance"]["reference_seconds"] = 0.02
    case_path.write_text(json.dumps(case_payload), encoding="utf-8")
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )
    assert exit_code == 2
    assert not output_path.exists()


def test_capability_floor_requires_cases_root(tmp_path: Path) -> None:
    task_ids = ["task-a"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(candidate_sidecar, candidate_episode_ids, truths)
    arguments = _capability_floor_args(
        (candidate_records, baseline_records),
        (candidate_sidecar, baseline_sidecar),
        tmp_path / "report.json",
        0,
    )
    cases_root_index = arguments.index("--cases-root")
    del arguments[cases_root_index : cases_root_index + 2]

    assert health.main(arguments) == 2


def test_capability_floor_requires_each_original_sidecar_entry(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(
        candidate_sidecar,
        {"task-a": candidate_episode_ids["task-a"]},
        {"task-a": truths["task-a"]},
    )
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
        )
    )

    assert exit_code != 0
    assert not output_path.exists()


def test_capability_floor_allows_missing_optional_sidecar_entry(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b"]
    baseline_records = tmp_path / "base" / "records.jsonl"
    candidate_records = tmp_path / "candidate" / "records.jsonl"
    baseline_episode_ids = _write_capability_records(baseline_records, "base", task_ids)
    candidate_episode_ids = _write_capability_records(candidate_records, "candidate", task_ids)
    truths = dict.fromkeys(task_ids, (True, "upstream_reference", "shared-reference-set"))
    baseline_sidecar = tmp_path / "base" / "strengthened_truth.jsonl"
    candidate_sidecar = tmp_path / "candidate" / "strengthened_truth.jsonl"
    _write_truth_sidecar(baseline_sidecar, baseline_episode_ids, truths)
    _write_truth_sidecar(
        candidate_sidecar,
        {"task-a": candidate_episode_ids["task-a"]},
        {"task-a": truths["task-a"]},
    )
    output_path = tmp_path / "capability.json"

    exit_code = health.main(
        _capability_floor_args(
            (candidate_records, baseline_records),
            (candidate_sidecar, baseline_sidecar),
            output_path,
            0,
            optional_truth=True,
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["capability_floor"]["excluded_task_counts"]["missing_optional_truth"] == 1


def test_health_report_creates_its_output_directory(tmp_path: Path) -> None:
    seed_v3_records = sorted(SEED_V3_ROOT.glob("*/records.jsonl"))
    if not seed_v3_records:
        pytest.skip("seed v3 evaluation records are absent")
    output_path = tmp_path / "not-yet-created" / "report.json"

    health.main(["--records", str(seed_v3_records[0]), "--output", str(output_path)])

    assert output_path.is_file()
