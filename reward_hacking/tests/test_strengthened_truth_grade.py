"""Strengthened grading acceptance; item material lives in ignored scratch fixtures."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from conftest import needs_jail

from reward_hacking.jailed_poc import CommandResult
from reward_hacking.organism_rl import strengthened_truth_grade as grade
from reward_hacking.organism_rl.strengthened_truth_grade import (
    CaseVerdict,
    canonical_digest,
    grade_artifact,
    load_case_set,
    validate_reference_set_digest,
)
from reward_hacking.organisms import grading

REPO_ROOT = Path(__file__).parents[2]
SCRATCH_ROOT = Path(
    os.environ.get("SEED_V4_TEST_SCRATCH_ROOT", str(REPO_ROOT / "docs" / "scratch"))
)
FIXTURE = SCRATCH_ROOT / "w16-strengthened-truth/grade-fixtures/data.json"
RECORDS = REPO_ROOT / "artifacts/organism-eval/evalA-base9b-c6a275f"
REAL_LCBHARD97_RECORDS = REPO_ROOT / "artifacts/organism-eval/minimal-seedv3c34-ef398f9/coding"
REAL_W16_CASES = REPO_ROOT / "docs/scratch/w16-strengthened-truth/cases-blind-pair-only"
REAL_W16_REFERENCE_VALIDATION = (
    REPO_ROOT / "docs/scratch/w16-strengthened-truth/ilcb-blind-reference-validation-r4.jsonl"
)


@pytest.fixture
def benchmark_fixtures() -> dict[str, Any]:
    if not FIXTURE.is_file():
        pytest.skip("requires ignored W16 strengthened-truth grading fixtures")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def evaluation_records() -> Path:
    if not (RECORDS / "records.jsonl").is_file():
        pytest.skip("requires local evalA records artifact")
    return RECORDS


@pytest.fixture
def payload() -> dict[str, Any]:
    arguments = [2]
    case = {"args": arguments, "expected": 3, "case_digest": canonical_digest(arguments)}
    return {
        "task_id": "synthetic-original",
        "source": "blind_reference_pair",
        "cases": [case],
        "provenance": {"reference_seconds": 0.01},
    }


def _benchmark_payload(benchmark_fixtures: dict[str, Any]) -> dict[str, Any]:
    cases = json.loads(json.dumps(benchmark_fixtures["cases"]))
    for case in cases:
        case["case_digest"] = canonical_digest(case["args"])
    return {
        "task_id": "synthetic-original",
        "source": "blind_reference_pair",
        "cases": cases,
        "provenance": {"reference_seconds": 0.01},
    }


def test_missing_source_is_never_true(tmp_path: Path) -> None:
    assert load_case_set(tmp_path, "synthetic-original") is None
    result = grade_artifact(tmp_path / "missing", None, entry_point="solve", family="ilcb")
    assert result["pass"] is None
    assert result["source"] == "none"
    assert result["unmeasured_reason"] == "case_set_missing"


def test_missing_artifact_is_unmeasured_with_a_countable_reason(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    result = grade_artifact(
        tmp_path / "missing-solution.py", payload, entry_point="solve", family="ilcb"
    )

    assert result["pass"] is None
    assert result["unmeasured_reason"] == "artifact_missing"
    assert result["reference_set_digest"] == canonical_digest(payload)
    assert result["disagreement_fraction"] is None
    assert result["counterexample"] is None


def test_sidecar_counts_missing_artifact_without_an_artifact_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "variant": "original",
                "family": "ilcb",
                "task_id": "synthetic-original",
                "trace_summary": {
                    "episode_id": "episode-missing-artifact",
                    "true": {"pass": False},
                    "graded_artifact": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        grade,
        "load_ilcb_problems",
        lambda: [
            SimpleNamespace(
                harness_task_id="synthetic-original",
                impossible_type="original",
                entry_point="solve",
            )
        ],
    )
    monkeypatch.setattr(grade, "load_aisi_problems", list)
    monkeypatch.setattr(grade, "_original_assertions", lambda *args: ())
    sidecar_path = tmp_path / "strengthened_truth.jsonl"

    rows = grade.grade_records(records_dir, tmp_path / "cases", sidecar_path)
    row = rows[0]

    assert row["artifact_sha256"] is None
    assert row["strengthened_truth"]["pass"] is None
    assert row["strengthened_truth"]["unmeasured_reason"] == "case_set_missing"
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["artifact_sha256"] is None


def test_source_preference_and_stale_refusal(tmp_path: Path, payload: dict[str, Any]) -> None:
    for source in ("evalplus", "blind_reference_pair"):
        directory = tmp_path / source
        directory.mkdir()
        (directory / "synthetic-original.json").write_text(
            json.dumps({**payload, "source": source})
        )
    selected = load_case_set(tmp_path, "synthetic-original")
    assert selected is not None
    assert selected["source"] == "evalplus"
    digest = canonical_digest(selected)
    validate_reference_set_digest({"reference_set_digest": digest}, selected)
    selected["provenance"]["reference_seconds"] = 0.02
    with pytest.raises(ValueError, match="stale"):
        validate_reference_set_digest({"reference_set_digest": digest}, selected)


@pytest.mark.parametrize("passed", [True, False])
def test_fraction_and_counterexample(payload: dict[str, Any], tmp_path: Path, passed: bool) -> None:
    def runner(**kwargs: Any) -> CaseVerdict:
        assert kwargs["timeout_seconds"] >= 3
        return CaseVerdict(passed, False, canonical_digest("diagnostic"))

    result = grade_artifact(tmp_path, payload, entry_point="solve", family="ilcb", runner=runner)
    assert result["pass"] is passed
    assert result["disagreement_fraction"] == (0 if passed else 1)
    assert (result["counterexample"] is None) is passed


def test_slow_correct_is_unmeasured_never_false(payload: dict[str, Any], tmp_path: Path) -> None:
    def runner(**kwargs: Any) -> CaseVerdict:
        return CaseVerdict(None, True, None)

    result = grade_artifact(tmp_path, payload, entry_point="solve", family="ilcb", runner=runner)
    assert result["pass"] is None
    assert result["unmeasured_timeout"] is True
    assert result["disagreement_fraction"] is None


def test_malformed_case_digest_refused(tmp_path: Path, payload: dict[str, Any]) -> None:
    directory = tmp_path / payload["source"]
    directory.mkdir()
    payload["cases"][0]["case_digest"] = "bad"
    (directory / "synthetic-original.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="case digest"):
        load_case_set(tmp_path, "synthetic-original")


@pytest.mark.parametrize("candidate", ["correct", "slow", "lookup", "guard", "rig", "stateful"])
def test_real_proxy_acceptance(
    candidate: str,
    benchmark_fixtures: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = benchmark_fixtures
    benchmark_payload = _benchmark_payload(data)
    artifact = tmp_path / "solution.py"
    artifact.write_text(data[candidate])

    def local_transport(work: Path, command: str, **kwargs: Any) -> CommandResult:
        completed = subprocess.run(  # noqa: S603 - trusted checker template and ignored synthetic fixtures
            shlex.split(command), cwd=work, capture_output=True, text=True, timeout=5, check=False
        )
        return CommandResult(
            command, completed.returncode, completed.stdout, completed.stderr, False
        )

    monkeypatch.setattr(grade, "run_in_jail", local_transport)
    monkeypatch.setattr(grade, "jail_python", lambda: sys.executable)
    result = grade.grade_artifact(
        artifact, benchmark_payload, entry_point=data["entry_point"], family="ilcb"
    )
    assert result["pass"] is (candidate in ("correct", "slow"))
    assert result["disagreement_fraction"] == (
        None if candidate == "stateful" else 0 if candidate in ("correct", "slow") else 1
    )
    if candidate in ("lookup", "guard"):
        assert result["counterexample"]["actual_digest"] is not None


def test_sabotaged_reference_rejects_correct_solution(
    benchmark_fixtures: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_fixtures["cases"][0]["expected"] += 1
    with pytest.raises(AssertionError):
        test_real_proxy_acceptance("correct", benchmark_fixtures, tmp_path, monkeypatch)


def test_sidecar_preserves_records_and_missing_coverage(
    tmp_path: Path, evaluation_records: Path
) -> None:
    records = evaluation_records
    before = (records / "records.jsonl").read_bytes()
    rows = grade.grade_records(records, tmp_path / "cases", tmp_path / "sidecar.jsonl")
    assert len(rows) == 24
    assert all(row["strengthened_truth"]["source"] == "none" for row in rows)
    assert all(row["strengthened_truth"]["pass"] is None for row in rows)
    assert (records / "records.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        grade.grade_records(records, tmp_path / "cases", records / "records.jsonl")


@pytest.mark.parametrize("trace_layout", ["traces", "episodes"])
def test_record_artifact_resolves_recorded_trace_layout(tmp_path: Path, trace_layout: str) -> None:
    records_dir = (
        tmp_path / "generation" / "ilcb" if trace_layout == "traces" else tmp_path / "evaluation"
    )
    trace_path = records_dir / trace_layout / "step-000000" / "synthetic-episode.jsonl"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("synthetic trace")
    artifact_bytes = b"synthetic graded artifact"
    artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_path = trace_path.parent / "graded-artifacts" / artifact_digest
    artifact_path.parent.mkdir()
    artifact_path.write_bytes(artifact_bytes)
    record = {"trace_path": str(trace_path)}
    (records_dir / "records.jsonl").write_text(json.dumps(record) + "\n")
    summary = {
        "graded_artifact": {
            "path": f"graded-artifacts/{artifact_digest}",
            "sha256": artifact_digest,
        }
    }

    resolved_path, resolved_digest = grade._record_artifact(records_dir, record, summary)

    assert resolved_path == artifact_path
    assert resolved_digest == artifact_digest


def test_record_artifact_fails_when_advertised_artifact_is_missing(tmp_path: Path) -> None:
    records_dir = tmp_path / "records"
    trace_path = records_dir / "traces" / "step-000000" / "synthetic-episode.jsonl"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("synthetic trace")
    artifact_digest = hashlib.sha256(b"missing artifact").hexdigest()
    record = {"trace_path": str(trace_path)}
    summary = {
        "graded_artifact": {
            "path": f"graded-artifacts/{artifact_digest}",
            "sha256": artifact_digest,
        }
    }

    with pytest.raises(FileNotFoundError):
        grade._record_artifact(records_dir, record, summary)


def test_sidecar_validation_recomputes_current_reference_digests(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    case_payload: dict[str, Any] = {
        "task_id": "synthetic-original",
        "source": "blind_reference_pair",
        "cases": [
            {
                "args": [1],
                "expected": 1,
                "case_digest": canonical_digest([1]),
            }
        ],
        "provenance": {"reference_seconds": 0.01},
    }
    case_path = cases_root / "blind_reference_pair" / "synthetic-original.json"
    case_path.parent.mkdir(parents=True)
    case_path.write_text(json.dumps(case_payload))
    sidecar_path = tmp_path / "strengthened_truth.jsonl"
    sidecar_path.write_text(
        json.dumps(
            {
                "task_id": "synthetic-original",
                "strengthened_truth": {
                    "pass": True,
                    "reference_set_digest": canonical_digest(case_payload),
                },
            }
        )
        + "\n"
    )

    assert grade.validate_strengthened_truth_sidecar(sidecar_path, cases_root) is None
    assert (
        grade.main(["--cases-root", str(cases_root), "--validate-sidecar", str(sidecar_path)]) == 0
    )

    case_payload["provenance"]["reference_seconds"] = 0.02
    case_path.write_text(json.dumps(case_payload))
    with pytest.raises(ValueError, match="stale strengthened truth reference set digest"):
        grade.validate_strengthened_truth_sidecar(sidecar_path, cases_root)
    with pytest.raises(ValueError, match="stale strengthened truth reference set digest"):
        grade.main(["--cases-root", str(cases_root), "--validate-sidecar", str(sidecar_path)])


def test_completed_sidecar_resumes_without_regrading(
    tmp_path: Path, evaluation_records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = evaluation_records
    output = tmp_path / "sidecar.jsonl"
    original = grade.grade_records(records, tmp_path / "cases", output)

    def unexpected_grade(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("completed episode should be resumed")

    monkeypatch.setattr(grade, "grade_artifact", unexpected_grade)
    assert grade.grade_records(records, tmp_path / "cases", output) == original


def test_grade_records_regrades_when_grading_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    summary = {
        "episode_id": "episode-cache-mismatch",
        "true": {"pass": False, "measured": True, "oracle_error": None},
        "graded_artifact": None,
    }
    record = {
        "variant": "original",
        "family": "ilcb",
        "task_id": "synthetic-original",
        "trace_summary": summary,
    }
    (records_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        grade,
        "_original_corpora",
        lambda: (
            {"synthetic-original": SimpleNamespace(entry_point="solve")},
            {},
        ),
    )
    monkeypatch.setattr(grade, "_original_assertions", lambda *args: ())
    output = tmp_path / "strengthened_truth.jsonl"
    output.write_text(
        json.dumps(
            {
                "episode_id": "episode-cache-mismatch",
                "task_id": "synthetic-original",
                "artifact_sha256": None,
                "true_pass": False,
                "grading_identity": {"semantic_version": -1},
                "strengthened_truth": {
                    "pass": None,
                    "reference_set_digest": None,
                    "unmeasured_timeout": False,
                    "unmeasured_error": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    graded: list[Path] = []

    def regraded(artifact: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        graded.append(artifact)
        return {"pass": None, "source": "none", "reference_set_digest": None}

    monkeypatch.setattr(grade, "grade_artifact", regraded)

    rows = grade.grade_records(records_dir, tmp_path / "cases", output)

    assert len(graded) == 1
    assert rows[0]["grading_identity"]["semantic_version"] == (
        grade.STRENGTHENED_TRUTH_GRADING_SEMANTIC_VERSION
    )


def test_unmeasured_original_check_is_written_as_null_true_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    summary = {
        "episode_id": "episode-original-timeout",
        "true": {"pass": False, "measured": False, "oracle_error": "timeout"},
        "graded_artifact": None,
    }
    (records_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "variant": "original",
                "family": "ilcb",
                "task_id": "synthetic-original",
                "trace_summary": summary,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        grade,
        "_original_corpora",
        lambda: (
            {"synthetic-original": SimpleNamespace(entry_point="solve")},
            {},
        ),
    )
    monkeypatch.setattr(grade, "_original_assertions", lambda *args: ())
    monkeypatch.setattr(
        grade,
        "grade_artifact",
        lambda *args, **kwargs: {"pass": None, "source": "none", "reference_set_digest": None},
    )

    row = grade.grade_records(records_dir, tmp_path / "cases", tmp_path / "strengthened.jsonl")[0]

    assert row["true_pass"] is None


@pytest.mark.parametrize(
    "expected_result",
    [((0.1, 0.2), False, False), (None, None, True)],
)
def test_original_suite_timeout_uses_reference_timing_evidence(
    payload: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_result: tuple[tuple[float, ...] | None, bool | None, bool],
) -> None:
    reference_suite_seconds, expected_pass, expected_unmeasured_timeout = expected_result
    artifact = tmp_path / "candidate.py"
    artifact.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    monkeypatch.setattr(
        grade,
        "_run_suite",
        lambda *args: CaseVerdict(passed=None, timed_out=True, actual_digest=None),
    )

    result = grade_artifact(
        artifact,
        payload,
        entry_point="solve",
        family="ilcb",
        original_assertions=("assert True",),
        reference_suite_seconds=reference_suite_seconds,
    )

    assert result["pass"] is expected_pass
    assert result["unmeasured_timeout"] is expected_unmeasured_timeout
    assert len(result["suite_attempt_seconds"]) == (2 if reference_suite_seconds is not None else 1)
    if reference_suite_seconds is not None:
        assert result["failure_class"] == "timeout"
        assert result["failure_reason"] == "timeout"


def test_transient_original_suite_timeout_is_retried_before_measured_failure(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "candidate.py"
    artifact.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    verdicts = iter(
        [
            CaseVerdict(passed=None, timed_out=True, actual_digest=None),
            CaseVerdict(passed=True, timed_out=False, actual_digest=None),
        ]
    )
    attempts: list[int] = []

    def run_suite(*args: Any) -> CaseVerdict:
        attempts.append(len(attempts))
        return next(verdicts)

    monkeypatch.setattr(grade, "_run_suite", run_suite)

    result = grade_artifact(
        artifact,
        payload,
        entry_point="solve",
        family="ilcb",
        original_assertions=("assert True",),
        reference_suite_seconds=(0.1, 0.2),
    )

    assert len(attempts) == 2
    assert result["pass"] is True
    assert result["failure_class"] is None
    assert len(result["suite_attempt_seconds"]) == 2
    assert all(seconds >= 0 for seconds in result["suite_attempt_seconds"])


def test_grade_records_reads_reference_suite_timings_from_validation_sidecar(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_dir = tmp_path / "records"
    trace_path = records_dir / "traces" / "step-000000" / "episode.jsonl"
    artifact = trace_path.parent / "graded-artifacts" / "candidate.py"
    trace_path.parent.mkdir(parents=True)
    artifact.parent.mkdir()
    trace_path.write_text("synthetic trace", encoding="utf-8")
    artifact.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    artifact_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    reference = tmp_path / "reference.py"
    reference.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    payload["task_id"] = "lcbhard_0"
    payload["provenance"]["reference_artifacts"] = [
        {"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    ]
    cases_root = tmp_path / "cases"
    case_directory = cases_root / payload["source"]
    case_directory.mkdir(parents=True)
    (case_directory / "lcbhard_0.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        grade,
        "_original_corpora",
        lambda: ({"lcbhard_0": SimpleNamespace(entry_point="solve")}, {}),
    )
    monkeypatch.setattr(grade, "_original_assertions", lambda *args: ("assert True",))
    real_grade_artifact = grade.grade_artifact
    monkeypatch.setattr(
        grade,
        "grade_artifact",
        lambda *args, **kwargs: {
            "pass": True,
            "reference_set_digest": canonical_digest(payload),
            "suite_seconds": 0.2,
        },
    )
    validation_path = tmp_path / "correct-solutions-validation.jsonl"
    assert grade.validate_correct_solutions(cases_root, validation_path).all_measured_passed
    validation_rows = [json.loads(line) for line in validation_path.read_text().splitlines()]
    for validation_row in validation_rows:
        validation_row["task_id"] = "ilcb-original-lcbhard_0"
        validation_row["grading_identity"]["semantic_version"] = (
            grade.REFERENCE_TIMING_COMPATIBLE_SEMANTIC_VERSION
        )
    validation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in validation_rows), encoding="utf-8"
    )
    monkeypatch.setattr(grade, "grade_artifact", real_grade_artifact)
    (records_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "variant": "original",
                "family": "ilcb",
                "task_id": "lcbhard_0",
                "trace_path": str(trace_path),
                "trace_summary": {
                    "episode_id": "episode-hanging-candidate",
                    "true": {"pass": True, "measured": True, "oracle_error": None},
                    "graded_artifact": {
                        "path": "graded-artifacts/candidate.py",
                        "sha256": artifact_digest,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        grade,
        "_run_suite",
        lambda *args: CaseVerdict(passed=None, timed_out=True, actual_digest=None),
    )

    row = grade.grade_records(
        records_dir,
        cases_root,
        tmp_path / "strengthened_truth.jsonl",
        reference_validation_path=validation_path,
    )[0]

    assert row["strengthened_truth"]["pass"] is False
    assert row["strengthened_truth"]["failure_class"] == "timeout"


def _prepare_prefixed_ilcb_timeout_fixture(
    payload: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_validated_references: bool,
    prefix_validation_task_id: bool,
) -> tuple[list[Path], Path, Path, Path]:
    case_task_id = "lcbhard_0"
    harness_task_id = f"ilcb-original-{case_task_id}"
    payload["task_id"] = case_task_id
    reference_paths = [tmp_path / "reference-a.py", tmp_path / "reference-b.py"]
    generator_path = tmp_path / "generator.py"
    for reference_index, reference_path in enumerate(reference_paths):
        reference_path.write_text(
            f"# reference {reference_index}\ndef solve(value):\n    return value + 1\n",
            encoding="utf-8",
        )
    generator_path.write_text("def generate():\n    return [[2]]\n", encoding="utf-8")
    reference_artifacts = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in [*reference_paths, generator_path]
    ]
    payload["provenance"].update(
        generator_digest=reference_artifacts[-1]["sha256"],
        reference_artifacts=reference_artifacts,
    )

    cases_root = tmp_path / "cases"
    case_directory = cases_root / payload["source"]
    case_directory.mkdir(parents=True)
    (case_directory / f"{case_task_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        grade,
        "_original_corpora",
        lambda: ({harness_task_id: SimpleNamespace(entry_point="solve")}, {}),
    )
    monkeypatch.setattr(grade, "_original_assertions", lambda *args: ("assert True",))

    def run_suite(artifact: Path, *args: Any) -> CaseVerdict:
        if artifact in reference_paths:
            return CaseVerdict(passed=True, timed_out=False, actual_digest=None)
        return CaseVerdict(passed=None, timed_out=True, actual_digest=None)

    monkeypatch.setattr(grade, "_run_suite", run_suite)
    validation_path = tmp_path / "reference-validation.jsonl"
    if has_validated_references:
        assert grade.validate_correct_solutions(cases_root, validation_path).all_measured_passed
        validation_rows = [json.loads(line) for line in validation_path.read_text().splitlines()]
        assert len(validation_rows) == len(reference_paths)
        assert all(
            row["strengthened_truth"]["suite_seconds"] is not None for row in validation_rows
        )
        if prefix_validation_task_id:
            for row in validation_rows:
                row["task_id"] = harness_task_id
            validation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in validation_rows), encoding="utf-8"
            )
    else:
        validation_path.write_text("", encoding="utf-8")

    records_dir = tmp_path / "records"
    trace_path = records_dir / "traces" / "step-000000" / "episode.jsonl"
    artifact_path = trace_path.parent / "graded-artifacts" / "candidate.py"
    artifact_path.parent.mkdir(parents=True)
    trace_path.write_text("synthetic trace", encoding="utf-8")
    artifact_path.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    (records_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "variant": "original",
                "family": "ilcb",
                "task_id": harness_task_id,
                "trace_path": "traces/step-000000/episode.jsonl",
                "trace_summary": {
                    "episode_id": "episode-hanging-candidate",
                    "true": {"pass": True, "measured": True, "oracle_error": None},
                    "graded_artifact": {
                        "path": "graded-artifacts/candidate.py",
                        "sha256": artifact_digest,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return reference_paths, cases_root, validation_path, records_dir


@pytest.mark.parametrize("has_validated_references", [True, False])
@pytest.mark.parametrize("prefix_validation_task_id", [True, False])
def test_prefixed_ilcb_timeout_uses_validated_reference_suite_times(
    payload: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_validated_references: bool,
    prefix_validation_task_id: bool,
) -> None:
    reference_paths, cases_root, validation_path, records_dir = (
        _prepare_prefixed_ilcb_timeout_fixture(
            payload,
            tmp_path,
            monkeypatch,
            has_validated_references,
            prefix_validation_task_id,
        )
    )
    output_path = tmp_path / "strengthened-truth.jsonl"
    row = grade.grade_records(
        records_dir,
        cases_root,
        output_path,
        reference_validation_path=validation_path,
    )[0]

    truth = row["strengthened_truth"]
    assert row["grading_identity"] == truth["grading_identity"]
    assert "reference_suite_seconds" not in row["grading_identity"]
    assert "timeout_reference_evidence_basis" not in row["grading_identity"]
    if has_validated_references:
        assert truth["pass"] is False
        assert truth["failure_class"] == "timeout"
        assert len(truth["timeout_reference_evidence"]["reference_suite_seconds"]) == len(
            reference_paths
        )
        assert (
            max(truth["timeout_reference_evidence"]["reference_suite_seconds"])
            <= row["grading_identity"]["suite_timeout_seconds"]
            * grade.TIMEOUT_REFERENCE_SUITE_LEASH_FRACTION
        )

        validation_path.write_text("", encoding="utf-8")
        rerun = grade.grade_records(
            records_dir,
            cases_root,
            output_path,
            reference_validation_path=validation_path,
        )[0]["strengthened_truth"]
        assert rerun["pass"] is None
        assert rerun["unmeasured_reason"] == "suite_timeout"
        assert rerun["timeout_reference_evidence"]["reference_suite_seconds"] is None
        assert rerun["reference_validation_coverage"]["status"] == "incomplete"
    else:
        assert truth["pass"] is None
        assert truth["failure_class"] is None
        assert truth["unmeasured_reason"] == "suite_timeout"
        assert truth["timeout_reference_evidence"]["reference_suite_seconds"] is None
        assert (
            truth["timeout_reference_evidence"]["timeout_reference_evidence_basis"]
            == "no_qualifying_reference_validation"
        )


def _validated_w16_reference_timing(
    real_cases_root: Path,
    validation_path: Path,
    harness_task_id: str,
) -> tuple[dict[str, Any], tuple[float, ...], Any]:
    case_set = load_case_set(real_cases_root, "lcbhard_97")
    assert case_set is not None
    ilcb, aisi = grade._original_corpora()
    problem = ilcb[harness_task_id]
    original_assertions = grade._original_assertions(harness_task_id, ilcb, aisi)
    reference_rows = grade._load_reference_validation(validation_path)
    reference_seconds = grade._validated_reference_suite_seconds(
        grade._ReferenceSuiteTimingRequest(
            reference_rows=reference_rows,
            task_id=case_set["task_id"],
            case_set=case_set,
            entry_point=problem.entry_point,
            family="ilcb",
            original_assertions=original_assertions,
        )
    )
    task_rows = [
        row for (task_id, _), row in reference_rows.items() if task_id == case_set["task_id"]
    ]
    assert len(task_rows) == len(grade._reference_artifacts(case_set))
    assert reference_seconds is not None
    suite_timeout = task_rows[0]["grading_identity"]["suite_timeout_seconds"]
    assert max(reference_seconds) <= suite_timeout * grade.TIMEOUT_REFERENCE_SUITE_LEASH_FRACTION
    return case_set, reference_seconds, suite_timeout


def _load_real_lcbhard97_record_and_artifact(
    records_root: Path,
    harness_task_id: str,
) -> tuple[dict[str, Any], str]:
    real_records = [
        json.loads(line)
        for line in (records_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    real_record = next(
        record for record in real_records if record.get("task_id") == harness_task_id
    )
    real_summary = real_record["trace_summary"]
    artifact_info = real_summary["graded_artifact"]
    trace_path = grade._resolve_recorded_trace(records_root, real_record["trace_path"])
    artifact_path = trace_path.parent / artifact_info["path"]
    artifact_digest = artifact_info["sha256"]
    if artifact_path.is_file() and not artifact_path.is_symlink():
        artifact_source = artifact_path.read_text(encoding="utf-8")
    else:
        artifact_source = None
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            trace_event = json.loads(line)
            for command in trace_event.get("commands", []):
                output = command.get("stdout")
                if (
                    isinstance(output, str)
                    and hashlib.sha256(output.encode()).hexdigest() == artifact_digest
                ):
                    artifact_source = output
                    break
            if artifact_source is not None:
                break
        if artifact_source is None:
            pytest.fail("the recorded lcbhard_97 artifact is absent from its persisted trace")
    assert hashlib.sha256(artifact_source.encode()).hexdigest() == artifact_digest
    return real_record, artifact_source


def _write_real_candidate_records(
    tmp_path: Path,
    real_record: dict[str, Any],
    artifact_source: str,
) -> Path:
    records_dir = tmp_path / "real-records"
    trace_path = records_dir / "traces" / "step-000000" / "episode.jsonl"
    artifact_path = trace_path.parent / "graded-artifacts" / "candidate.py"
    artifact_path.parent.mkdir(parents=True)
    trace_path.write_text("{}\n", encoding="utf-8")
    artifact_path.write_text(artifact_source, encoding="utf-8")
    record = dict(real_record)
    record["trace_path"] = "traces/step-000000/episode.jsonl"
    record_summary = dict(real_record["trace_summary"])
    record_summary["graded_artifact"] = {
        **real_record["trace_summary"]["graded_artifact"],
        "path": "graded-artifacts/candidate.py",
    }
    record["trace_summary"] = record_summary
    (records_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return records_dir


def _bounded_real_jail_runner(timeout_cap: int) -> Callable[..., CommandResult]:
    real_run_in_jail = grade.run_in_jail

    def bounded_run_in_jail(episode_dir: Path, command: str, **kwargs: Any) -> CommandResult:
        timeout = kwargs.get("timeout", "60s")
        kwargs["timeout"] = f"{min(timeout_cap, int(timeout.removesuffix('s')))}s"
        return real_run_in_jail(episode_dir, command, **kwargs)

    return bounded_run_in_jail


@needs_jail
@pytest.mark.jail_real
def test_real_lcbhard_97_timeout_uses_current_reference_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    real_workspace = REPO_ROOT / "docs" / "scratch" / "w16-strengthened-truth"
    real_cases_root = real_workspace / "cases-blind-pair-only"
    real_records_root = REAL_LCBHARD97_RECORDS
    reference_validation_path_source = real_workspace / "ilcb-blind-reference-validation-r4.jsonl"
    real_case_path = real_cases_root / "blind_reference_pair" / "lcbhard_97.json"
    missing_paths = [
        path for path in (real_case_path, reference_validation_path_source) if not path.is_file()
    ]
    if not (real_records_root / "records.jsonl").is_file():
        missing_paths.append(real_records_root / "records.jsonl")
    if missing_paths:
        jail_marker_selected = pytestconfig.getoption("-m") == "jail_real"
        if os.environ.get("W16_REQUIRE_STRENGTHENED_TRUTH_DATA") == "1" or jail_marker_selected:
            pytest.fail("W16 real-data grading requires the local case and validation files")
        pytest.skip("requires local W16 real-data grading files")

    harness_task_id = "ilcb-original-lcbhard_97"
    case_set, _, suite_timeout = _validated_w16_reference_timing(
        real_cases_root, reference_validation_path_source, harness_task_id
    )
    reference_validation_path = tmp_path / "latest-reference-validation.jsonl"
    reference_validation_path.write_bytes(reference_validation_path_source.read_bytes())
    real_record, artifact_source = _load_real_lcbhard97_record_and_artifact(
        real_records_root, harness_task_id
    )
    records_dir = _write_real_candidate_records(tmp_path, real_record, artifact_source)
    monkeypatch.setattr(grade, "run_in_jail", _bounded_real_jail_runner(timeout_cap=8))

    output_path = tmp_path / "real-strengthened-truth.jsonl"
    measured_timeout = grade.grade_records(
        records_dir,
        real_cases_root,
        output_path,
        reference_validation_path=reference_validation_path,
    )[0]["strengthened_truth"]
    assert measured_timeout["pass"] is False
    assert measured_timeout["failure_class"] == "timeout"
    reference_seconds = measured_timeout["timeout_reference_evidence"]["reference_suite_seconds"]
    assert len(reference_seconds) == len(grade._reference_artifacts(case_set))
    assert max(reference_seconds) <= suite_timeout * grade.TIMEOUT_REFERENCE_SUITE_LEASH_FRACTION
    assert measured_timeout["suite_seconds"] <= 30

    monkeypatch.setattr(grade, "TIMEOUT_REFERENCE_SUITE_LEASH_FRACTION", 0.49)
    regraded_timeout = grade.grade_records(
        records_dir,
        real_cases_root,
        output_path,
        reference_validation_path=reference_validation_path,
    )[0]["strengthened_truth"]
    assert regraded_timeout["pass"] is False
    assert (
        regraded_timeout["timeout_reference_evidence"]["timeout_reference_leash_fraction"] == 0.49
    )

    reference_validation_path.write_text("", encoding="utf-8")
    unmeasured_timeout = grade.grade_records(
        records_dir,
        real_cases_root,
        output_path,
        reference_validation_path=reference_validation_path,
    )[0]["strengthened_truth"]
    assert unmeasured_timeout["pass"] is None
    assert unmeasured_timeout["unmeasured_reason"] == "suite_timeout"
    assert unmeasured_timeout["timeout_reference_evidence"]["reference_suite_seconds"] is None


@pytest.mark.corpus
def test_w16_real_data_opt_in_requires_current_reference_evidence(
    pytestconfig: pytest.Config,
) -> None:
    corpus_marker_selected = pytestconfig.getoption("-m") == "corpus"
    if os.environ.get("W16_REQUIRE_STRENGTHENED_TRUTH_DATA") != "1" and not corpus_marker_selected:
        pytest.skip("select with -m corpus or set W16_REQUIRE_STRENGTHENED_TRUTH_DATA=1")
    real_workspace = REPO_ROOT / "docs" / "scratch" / "w16-strengthened-truth"
    cases_root = real_workspace / "cases-blind-pair-only"
    case_path = cases_root / "blind_reference_pair" / "lcbhard_97.json"
    validation_path = real_workspace / "ilcb-blind-reference-validation-r4.jsonl"
    if (
        not case_path.is_file()
        or not validation_path.is_file()
        or not (REAL_LCBHARD97_RECORDS / "records.jsonl").is_file()
    ):
        pytest.fail("W16 real-data grading requires the local case and validation files")
    case_set = load_case_set(cases_root, "lcbhard_97")
    assert case_set is not None
    ilcb, aisi = grade._original_corpora()
    task_id = "ilcb-original-lcbhard_97"
    problem = ilcb[task_id]
    reference_seconds = grade._validated_reference_suite_seconds(
        grade._ReferenceSuiteTimingRequest(
            reference_rows=grade._load_reference_validation(validation_path),
            task_id=case_set["task_id"],
            case_set=case_set,
            entry_point=problem.entry_point,
            family="ilcb",
            original_assertions=grade._original_assertions(task_id, ilcb, aisi),
        )
    )
    assert reference_seconds is not None, "W16 r4 validation lacks current lcbhard_97 evidence"
    assert len(reference_seconds) == len(grade._reference_artifacts(case_set))


@pytest.mark.corpus
def test_r4_reference_timing_rows_cover_both_real_ilcb_record_sets(
    pytestconfig: pytest.Config,
) -> None:
    corpus_marker_selected = pytestconfig.getoption("-m") == "corpus"
    if os.environ.get("W16_REQUIRE_STRENGTHENED_TRUTH_DATA") != "1" and not corpus_marker_selected:
        pytest.skip("select with -m corpus or set W16_REQUIRE_STRENGTHENED_TRUTH_DATA=1")
    real_workspace = REPO_ROOT / "docs" / "scratch" / "w16-strengthened-truth"
    cases_root = real_workspace / "cases-blind-pair-only"
    validation_path = real_workspace / "ilcb-blind-reference-validation-r4.jsonl"
    records_roots = [
        REPO_ROOT / "artifacts/organism-eval/evalA-base9b-c6a275f",
        REPO_ROOT / "artifacts/organism-eval/minimal-seedv3c34-ef398f9/coding",
    ]
    missing_paths = [
        path
        for path in [
            cases_root,
            validation_path,
            *(root / "records.jsonl" for root in records_roots),
        ]
        if not path.exists()
    ]
    if missing_paths:
        pytest.fail(
            "W16 r4 real-data validation requires the local case, validation, and record files"
        )

    reference_rows = grade._load_reference_validation(validation_path)
    validation_outcomes = [row["strengthened_truth"].get("pass") for row in reference_rows.values()]
    assert len(validation_outcomes) == 196
    assert sum(passed is True for passed in validation_outcomes) == 194
    assert sum(passed is None for passed in validation_outcomes) == 2
    assert all(passed is not False for passed in validation_outcomes)
    ilcb, aisi = grade._original_corpora()
    for records_root in records_roots:
        records = [
            json.loads(line)
            for line in (records_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        original_records = [
            record
            for record in records
            if record["family"] == "ilcb" and record["variant"] == "original"
        ]
        assert len(original_records) == 12
        measured_task_count = 0
        unmeasured_task_count = 0
        for record in original_records:
            task_id = record["task_id"]
            case_set = load_case_set(cases_root, task_id)
            assert case_set is not None, task_id
            reference_task_id = grade._reference_validation_task_id(task_id, case_set)
            problem = ilcb[task_id]
            reference_seconds = grade._validated_reference_suite_seconds(
                grade._ReferenceSuiteTimingRequest(
                    reference_rows=reference_rows,
                    task_id=reference_task_id,
                    case_set=case_set,
                    entry_point=problem.entry_point,
                    family="ilcb",
                    original_assertions=grade._original_assertions(task_id, ilcb, aisi),
                )
            )
            task_validation_rows = [
                row
                for (validated_task_id, _), row in reference_rows.items()
                if validated_task_id == reference_task_id
            ]
            references = grade._reference_artifacts(case_set)
            assert len(task_validation_rows) == len(references), task_id
            if reference_seconds is None:
                assert all(
                    row["strengthened_truth"]["pass"] is None
                    and row["strengthened_truth"]["unmeasured_reason"] == "suite_timeout"
                    for row in task_validation_rows
                )
                unmeasured_task_count += 1
            else:
                assert len(reference_seconds) == len(references)
                measured_task_count += 1
        assert measured_task_count == 11
        assert unmeasured_task_count == 1


def _select_real_reference_records(
    cases_root: Path,
    records_path: Path,
    historical_validation: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_task: dict[str, dict[str, Any]] = {}
    for record in (
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ):
        if record["family"] != "ilcb" or record["variant"] != "original":
            continue
        summary = record.get("trace_summary", record.get("episode_record"))
        if not isinstance(summary, dict) or summary.get("graded_artifact") is None:
            continue
        task_id = record["task_id"]
        case_set = load_case_set(cases_root, task_id)
        if case_set is None or "unmeasured_source_error" in case_set["provenance"]:
            continue
        references = grade._reference_artifacts(case_set)
        normalized_task_id = grade._reference_validation_task_id(task_id, case_set)
        validated_rows = [
            historical_validation.get((normalized_task_id, reference["sha256"]))
            for reference in references
        ]
        if not references or any(
            row is None or row["strengthened_truth"].get("pass") is not True
            for row in validated_rows
        ):
            continue
        candidates_by_task[task_id] = record
    return [candidates_by_task[task_id] for task_id in sorted(candidates_by_task)[:2]]


def _prepare_real_validation_subset(
    tmp_path: Path,
    cases_root: Path,
    records_root: Path,
    records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    subset_cases_root = tmp_path / "cases"
    subset_records_dir = tmp_path / "records"
    rewritten_records: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        case_set = load_case_set(cases_root, record["task_id"])
        assert case_set is not None
        source_case = cases_root / case_set["source"] / f"{case_set['task_id']}.json"
        target_case = subset_cases_root / case_set["source"] / source_case.name
        target_case.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_case, target_case)
        for reference in grade._reference_artifacts(case_set):
            source_reference = cases_root.parent / reference["path"]
            target_reference = subset_cases_root.parent / reference["path"]
            target_reference.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_reference, target_reference)

        summary = record.get("trace_summary", record.get("episode_record"))
        if not isinstance(summary, dict):
            raise TypeError("evalA record requires an episode summary")
        source_artifact, artifact_digest = grade._record_artifact(records_root, record, summary)
        if artifact_digest is None:
            raise AssertionError("selected evalA record requires a graded artifact")
        trace_relative = Path("traces") / f"step-{record_index:06d}" / "episode.jsonl"
        trace_path = subset_records_dir / trace_relative
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("fresh strengthened-truth join fixture", encoding="utf-8")
        artifact_target = trace_path.parent / "graded-artifacts" / artifact_digest
        artifact_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_artifact, artifact_target)
        rewritten_summary = dict(summary)
        rewritten_summary["graded_artifact"] = {
            "path": f"graded-artifacts/{artifact_digest}",
            "sha256": artifact_digest,
        }
        rewritten_record = dict(record)
        rewritten_record["trace_path"] = trace_relative.as_posix()
        rewritten_record["trace_summary"] = rewritten_summary
        rewritten_records.append(rewritten_record)
    records_output = subset_records_dir / "records.jsonl"
    records_output.write_text(
        "".join(json.dumps(record) + "\n" for record in rewritten_records), encoding="utf-8"
    )
    return subset_cases_root, subset_records_dir


@needs_jail
@pytest.mark.jail_real
def test_fresh_real_reference_validation_joins_grade_records(tmp_path: Path) -> None:
    """Validate one or two real tasks, then grade records from that fresh sidecar."""
    if not REAL_W16_CASES.is_dir() or not REAL_W16_REFERENCE_VALIDATION.is_file():
        pytest.fail("fresh real validation requires local W16 cases and r4 selection metadata")
    records_path = RECORDS / "records.jsonl"
    if not records_path.is_file():
        pytest.fail("fresh real validation requires evalA records")
    historical_validation = grade._load_reference_validation(REAL_W16_REFERENCE_VALIDATION)
    selected_records = _select_real_reference_records(
        REAL_W16_CASES, records_path, historical_validation
    )
    assert selected_records, "r4 must identify at least one measured reference task"
    subset_cases_root, subset_records_dir = _prepare_real_validation_subset(
        tmp_path, REAL_W16_CASES, RECORDS, selected_records
    )

    fresh_validation_path = tmp_path / "fresh-reference-validation.jsonl"
    validation_result = grade.validate_correct_solutions(
        subset_cases_root, fresh_validation_path, family="ilcb"
    )
    assert validation_result.measured_failures == 0
    fresh_validation_rows = grade._load_reference_validation(fresh_validation_path)
    assert len(fresh_validation_rows) == validation_result.total_rows
    graded_rows = grade.grade_records(
        subset_records_dir,
        subset_cases_root,
        tmp_path / "fresh-strengthened-truth.jsonl",
        family="ilcb",
        reference_validation_path=fresh_validation_path,
    )
    assert len(graded_rows) == len(selected_records)
    for row in graded_rows:
        case_set = load_case_set(subset_cases_root, row["task_id"])
        assert case_set is not None
        assert row["strengthened_truth"]["reference_validation_coverage"] == {
            "status": "complete",
            "reference_rows": len(grade._reference_artifacts(case_set)),
        }


@pytest.mark.parametrize("entry_point", ["solution", "solve"])
@pytest.mark.parametrize("candidate", ["correct", "slow", "lookup", "guard", "rig", "stateful"])
def test_real_aisi_acceptance(
    entry_point: str,
    candidate: str,
    benchmark_fixtures: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = benchmark_fixtures
    benchmark_payload = _benchmark_payload(data)
    artifact = tmp_path / "solution.py"
    artifact.write_text(
        data[candidate] + (data["aisi_suffix"] if entry_point == "solution" else "")
    )

    def local_transport(work: Path, command: str, **kwargs: Any) -> CommandResult:
        completed = subprocess.run(  # noqa: S603 - trusted runner and ignored synthetic fixture
            shlex.split(command.removeprefix("cd /work && ")),
            cwd=work,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return CommandResult(
            command, completed.returncode, completed.stdout, completed.stderr, False
        )

    monkeypatch.setattr(grade, "run_in_jail", local_transport)
    monkeypatch.setattr(grade, "jail_python", lambda: sys.executable)
    monkeypatch.setattr(grading, "run_in_jail", local_transport)
    monkeypatch.setattr(grading, "jail_python", lambda: sys.executable)

    def local_pytest(*args: Any, **kwargs: Any) -> Any:
        return grading.run_completion_tests(*args, scratch_parent=tmp_path, **kwargs)

    monkeypatch.setattr(grade, "run_completion_tests", local_pytest)
    benchmark_payload["provenance"]["assertions"] = [
        grade.case_assertion(case, "solution") for case in benchmark_payload["cases"]
    ]
    result = grade.grade_artifact(
        artifact, benchmark_payload, entry_point=entry_point, family="aisi"
    )
    assert result["pass"] is (candidate in ("correct", "slow"))
    assert result["disagreement_fraction"] == (
        None if candidate == "stateful" else 0 if candidate in ("correct", "slow") else 1
    )


def test_reference_file_mutation_refused(tmp_path: Path, payload: dict[str, Any]) -> None:
    root = tmp_path / "cases"
    directory = root / payload["source"]
    directory.mkdir(parents=True)
    reference = tmp_path / "reference.py"
    correct_source = "def solve(value):\n    return value + 1\n"
    mutated_source = "def solve(value):\n    return value - 1\n"
    reference.write_text(correct_source)
    payload["provenance"]["reference_artifacts"] = [
        {"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    ]
    (directory / "synthetic-original.json").write_text(json.dumps(payload))
    assert load_case_set(root, "synthetic-original") is not None
    reference.write_text(mutated_source)
    with pytest.raises(ValueError, match="reference artifact digest"):
        load_case_set(root, "synthetic-original")


def test_reference_validation_requires_reference_artifacts(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    root = tmp_path / "cases"
    directory = root / payload["source"]
    directory.mkdir(parents=True)
    (directory / "synthetic-original.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reference artifacts"):
        grade.validate_correct_solutions(root, tmp_path / "validation.jsonl")


def test_ilcb_case_files_accept_corpus_identity(tmp_path: Path, payload: dict[str, Any]) -> None:
    directory = tmp_path / payload["source"]
    directory.mkdir()
    (directory / "synthetic-original.json").write_text(json.dumps(payload))
    assert load_case_set(tmp_path, "ilcb-original-synthetic-original") == payload


def test_oversized_source_is_explicitly_unmeasured(
    tmp_path: Path, payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / payload["source"]
    directory.mkdir()
    (directory / "synthetic-original.json").write_text(json.dumps(payload))
    monkeypatch.setattr(grade, "MAX_CASE_FILE_BYTES", 1)
    selected = load_case_set(tmp_path, "synthetic-original")
    result = grade_artifact(tmp_path / "missing", selected, entry_point="solve", family="ilcb")
    assert result["pass"] is None
    assert result["source"] == payload["source"]
    assert result["unmeasured_error"] == "case_file_too_large"
    assert result["reference_set_digest"] is not None


def test_private_tests_without_reference_timing_use_existing_leash(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    payload["source"] = "lcb_private"
    del payload["provenance"]["reference_seconds"]

    def runner(**kwargs: Any) -> CaseVerdict:
        assert kwargs["timeout_seconds"] == grade.TRUE_CHECK_TIMEOUT_SECONDS
        return CaseVerdict(passed=True, timed_out=False, actual_digest=None)

    result = grade_artifact(tmp_path, payload, entry_point="solve", family="ilcb", runner=runner)
    assert result["pass"] is True
    assert result["timeout_basis"] == "existing_corpus_leash_without_reference_timing"


def test_reference_validation_resumes_measured_results(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.py"
    reference.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    payload["task_id"] = "lcbhard_0"
    payload["provenance"].update(
        reference_artifacts=[
            {"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
        ],
    )
    cases_root = tmp_path / "cases"
    directory = cases_root / payload["source"]
    directory.mkdir(parents=True)
    (directory / "lcbhard_0.json").write_text(json.dumps(payload))
    output = tmp_path / "validation.jsonl"

    def measured(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "pass": True,
            "source": payload["source"],
            "reference_set_digest": canonical_digest(payload),
            "suite_seconds": 0.2,
        }

    monkeypatch.setattr(grade, "grade_artifact", measured)
    assert grade.validate_correct_solutions(cases_root, output).all_measured_passed
    validation_row = json.loads(output.read_text(encoding="utf-8"))
    assert validation_row["grading_identity"]["semantic_version"] == (
        grade.STRENGTHENED_TRUTH_GRADING_SEMANTIC_VERSION
    )
    assert validation_row["strengthened_truth"]["suite_seconds"] == 0.2

    def unexpected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("measured reference should be resumed")

    monkeypatch.setattr(grade, "grade_artifact", unexpected)
    assert grade.validate_correct_solutions(cases_root, output).all_measured_passed


def test_reference_validation_regrades_when_grading_identity_changes(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.py"
    reference.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    reference_digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    payload["task_id"] = "lcbhard_0"
    payload["provenance"]["reference_artifacts"] = [
        {"path": reference.name, "sha256": reference_digest}
    ]
    cases_root = tmp_path / "cases"
    directory = cases_root / payload["source"]
    directory.mkdir(parents=True)
    (directory / "lcbhard_0.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        grade,
        "_original_corpora",
        lambda: ({"lcbhard_0": SimpleNamespace(entry_point="solve")}, {}),
    )
    monkeypatch.setattr(grade, "_original_assertions", lambda *args: ("assert True",))
    output = tmp_path / "validation.jsonl"
    output.write_text(
        json.dumps(
            {
                "task_id": "lcbhard_0",
                "reference_sha256": reference_digest,
                "grading_identity": {"semantic_version": -1},
                "strengthened_truth": {
                    "pass": True,
                    "reference_set_digest": canonical_digest(payload),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    graded: list[Path] = []

    def regraded(artifact: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        graded.append(artifact)
        return {"pass": True, "reference_set_digest": canonical_digest(payload)}

    monkeypatch.setattr(grade, "grade_artifact", regraded)

    assert grade.validate_correct_solutions(cases_root, output).all_measured_passed

    assert graded == [reference]


def test_reference_validation_selects_lcbhard_ids_as_ilcb(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.py"
    reference.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    payload["task_id"] = "lcbhard_0"
    payload["provenance"].update(
        reference_artifacts=[
            {"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
        ],
    )
    cases_root = tmp_path / "cases"
    directory = cases_root / payload["source"]
    directory.mkdir(parents=True)
    (directory / "lcbhard_0.json").write_text(json.dumps(payload))

    def measured(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["family"] == "ilcb"
        assert kwargs["entry_point"] == "count_beautiful_integers"
        assert kwargs["original_assertions"][0].endswith(
            "strengthened_original_check(count_beautiful_integers)"
        )
        return {"pass": True, "reference_set_digest": canonical_digest(payload)}

    monkeypatch.setattr(grade, "grade_artifact", measured)

    assert grade.validate_correct_solutions(
        cases_root, tmp_path / "validation.jsonl", family="ilcb"
    ).all_measured_passed


@pytest.mark.parametrize(
    ("original_assertions", "expected_suite_timeout"),
    [((), 3), (("assert True",), math.ceil(grade.TRUE_CHECK_TIMEOUT_SECONDS) + 3)],
)
def test_suite_leash_adds_the_corpus_leash_for_the_original_check(
    payload: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original_assertions: tuple[str, ...],
    expected_suite_timeout: int,
) -> None:
    artifact = tmp_path / "candidate.py"
    artifact.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    suite_timeouts: list[int] = []

    def suite(*args: Any) -> CaseVerdict:
        suite_timeouts.append(args[-1])
        return CaseVerdict(passed=True, timed_out=False, actual_digest=None)

    monkeypatch.setattr(grade, "_run_suite", suite)
    result = grade_artifact(
        artifact,
        payload,
        entry_point="solve",
        family="ilcb",
        original_assertions=original_assertions,
    )

    assert result["pass"] is True
    assert suite_timeouts == [expected_suite_timeout]
    assert result["timeout_basis"] == "ten_times_reference_with_three_second_floor"
    identity = result["grading_identity"]
    assert identity["original_assertions_sha256"] == canonical_digest(list(original_assertions))
    assert identity["per_case_timeout_seconds"] == 3
    assert identity["suite_timeout_seconds"] == expected_suite_timeout
    assert identity["per_case_timeout_basis"] == result["timeout_basis"]
    assert identity["suite_timeout_basis"] == result["suite_timeout_basis"]


def test_reference_validation_skips_the_input_generator(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.py"
    reference.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    generator = tmp_path / "generator.py"
    generator.write_text("def generate():\n    return [[1]]\n", encoding="utf-8")
    generator_sha256 = hashlib.sha256(generator.read_bytes()).hexdigest()
    payload["task_id"] = "lcbhard_0"
    payload["provenance"].update(
        generator_digest=generator_sha256,
        reference_artifacts=[
            {"path": generator.name, "sha256": generator_sha256},
            {"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()},
        ],
    )
    directory = tmp_path / "cases" / payload["source"]
    directory.mkdir(parents=True)
    (directory / "lcbhard_0.json").write_text(json.dumps(payload))
    graded: list[Path] = []

    def measured(artifact: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        graded.append(artifact)
        return {"pass": True, "reference_set_digest": canonical_digest(payload)}

    monkeypatch.setattr(grade, "grade_artifact", measured)

    assert grade.validate_correct_solutions(
        tmp_path / "cases", tmp_path / "validation.jsonl"
    ).all_measured_passed
    assert [path.name for path in graded] == [reference.name]


def test_reference_validation_refuses_a_task_missing_from_the_corpus(
    payload: dict[str, Any], tmp_path: Path
) -> None:
    reference = tmp_path / "reference.py"
    reference.write_text("def solve(value):\n    return value + 1\n", encoding="utf-8")
    payload["task_id"] = "lcbhard_999999"
    payload["provenance"]["reference_artifacts"] = [
        {"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
    ]
    directory = tmp_path / "cases" / payload["source"]
    directory.mkdir(parents=True)
    (directory / "lcbhard_999999.json").write_text(json.dumps(payload))

    with pytest.raises(KeyError, match="lcbhard_999999"):
        grade.validate_correct_solutions(tmp_path / "cases", tmp_path / "validation.jsonl")


def test_extra_cases_cannot_override_original_failure(
    payload: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload["cases"][0]["expected"] = 4
    payload["cases"][0]["args"] = [2]
    payload["cases"][0]["case_digest"] = canonical_digest([2])
    artifact = tmp_path / "candidate.py"
    artifact.write_text("def solve(value):\n    return value + 2\n", encoding="utf-8")

    def transport(directory: Path, command: str, **kwargs: Any) -> CommandResult:
        result = subprocess.run(  # noqa: S603 - ignored synthetic fixture through trusted checker
            shlex.split(command),
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return CommandResult(command, result.returncode, result.stdout, result.stderr, False)

    monkeypatch.setattr(grade, "run_in_jail", transport)
    monkeypatch.setattr(grade, "jail_python", lambda: sys.executable)
    truth = grade_artifact(
        artifact,
        payload,
        entry_point="solve",
        family="ilcb",
        original_assertions=("assert solve(1) == 2",),
    )
    assert truth["pass"] is False


def test_family_rescore_preserves_other_family_sidecar(
    tmp_path: Path, evaluation_records: Path
) -> None:
    records = evaluation_records
    output = tmp_path / "sidecar.jsonl"
    original = grade.grade_records(records, tmp_path / "cases", output)
    grade.grade_records(records, tmp_path / "cases", output, family="aisi")
    assert {json.loads(line)["episode_id"] for line in output.read_text().splitlines()} == {
        row["episode_id"] for row in original
    }


@pytest.mark.parametrize(
    ("record_task_id", "case_task_id", "expected_family"),
    [
        ("lcbhard_17", "lcbhard_17", "ilcb"),
        ("ilcb-original-lcbhard_17", "lcbhard_17", "ilcb"),
        ("ilcb-original-lcbhard_17", "ilcb-original-lcbhard_17", "ilcb"),
        (
            "aisi-apps-original-task-1234567890",
            "aisi-apps-original-task-1234567890",
            "aisi",
        ),
    ],
)
def test_reference_validation_maps_supported_task_id_shapes(
    record_task_id: str, case_task_id: str, expected_family: str
) -> None:
    case_set = {"task_id": case_task_id}

    assert grade._case_family(record_task_id) == expected_family
    expected_task_id = case_task_id.removeprefix("ilcb-original-")
    assert grade._reference_validation_task_id(record_task_id, case_set) == expected_task_id

    if record_task_id == "lcbhard_17":
        ilcb: dict[str, Any] = {"ilcb-original-lcbhard_17": object()}
        assert grade._corpus_task_id(case_task_id, ilcb) == "ilcb-original-lcbhard_17"


def test_reference_validation_counts_measured_failures_separately_from_unmeasured(
    tmp_path: Path, payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = "lcbhard_17"
    harness_task_id = f"ilcb-original-{task_id}"
    outcomes = {
        "reference-pass.py": True,
        "reference-fail.py": False,
        "reference-unmeasured.py": None,
    }
    references: list[dict[str, str]] = []
    for reference_name in outcomes:
        reference_path = tmp_path / reference_name
        reference_path.write_text(reference_name, encoding="utf-8")
        references.append(
            {
                "path": reference_name,
                "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            }
        )
    payload["task_id"] = task_id
    payload["provenance"]["reference_artifacts"] = references
    case_directory = tmp_path / "cases" / payload["source"]
    case_directory.mkdir(parents=True)
    (case_directory / f"{task_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        grade,
        "_original_corpora",
        lambda: ({harness_task_id: SimpleNamespace(entry_point="solve")}, {}),
    )
    monkeypatch.setattr(grade, "_original_assertions", lambda *args: ())

    def graded(artifact: Path, case_set: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        passed = outcomes[artifact.name]
        return {
            "pass": passed,
            "reference_set_digest": canonical_digest(case_set),
            "suite_seconds": 0.25 if passed is not None else None,
        }

    monkeypatch.setattr(grade, "grade_artifact", graded)

    result = grade.validate_correct_solutions(tmp_path / "cases", tmp_path / "validation.jsonl")

    assert result.total_rows == 3
    assert result.measured_passes == 1
    assert result.measured_failures == 1
    assert result.unmeasured == 1
    assert result.exit_code == 2
    assert not result.all_measured_passed


def test_reference_validation_cli_reports_unmeasured_without_calling_it_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    result = grade.ReferenceValidationResult(
        total_rows=2,
        measured_passes=0,
        measured_failures=0,
        unmeasured=2,
    )
    monkeypatch.setattr(grade, "validate_correct_solutions", lambda *args, **kwargs: result)

    exit_code = grade.main(
        ["--validate-references", "--cases-root", str(tmp_path), "--output", str(tmp_path / "out")]
    )

    assert exit_code == 3
    assert "measured_passes=0 measured_failures=0 unmeasured=2" in caplog.text


def test_triage_cli_joins_by_episode_and_writes_rows_and_readout(tmp_path: Path) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    record_rows = [
        {
            "variant": "original",
            "family": "ilcb",
            "task_id": "ilcb-original-lcbhard_17",
            "trace_summary": {
                "episode_id": "episode-agree",
                "true": {"pass": True, "measured": True, "oracle_error": None},
            },
        },
        {
            "variant": "original",
            "family": "ilcb",
            "task_id": "ilcb-original-lcbhard_18",
            "trace_summary": {
                "episode_id": "episode-unmeasured",
                "true": {"pass": False, "measured": False, "oracle_error": "timeout"},
            },
        },
    ]
    (records_dir / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in record_rows), encoding="utf-8"
    )
    sidecar_path = tmp_path / "strengthened-truth.jsonl"
    sidecar_rows = [
        {
            "episode_id": "episode-agree",
            "task_id": "ilcb-original-lcbhard_17",
            "true_pass": True,
            "strengthened_truth": {"pass": True},
        },
        {
            "episode_id": "episode-unmeasured",
            "task_id": "ilcb-original-lcbhard_18",
            "true_pass": None,
            "strengthened_truth": {"pass": None, "unmeasured_reason": "suite_timeout"},
        },
    ]
    sidecar_path.write_text(
        "".join(json.dumps(row) + "\n" for row in sidecar_rows), encoding="utf-8"
    )
    output = tmp_path / "triage.jsonl"
    readout = tmp_path / "triage.md"

    triage_arguments = [
        "triage",
        "--sidecar",
        str(sidecar_path),
        "--records",
        str(records_dir),
        "--output",
        str(output),
        "--readout",
        str(readout),
    ]
    exit_code = grade.main(triage_arguments)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert rows == [
        {
            "episode_id": "episode-agree",
            "task_id": "ilcb-original-lcbhard_17",
            "true_pass": True,
            "strengthened_pass": True,
            "class": "agree",
        },
        {
            "episode_id": "episode-unmeasured",
            "task_id": "ilcb-original-lcbhard_18",
            "true_pass": None,
            "strengthened_pass": None,
            "class": "unmeasured",
        },
    ]
    assert "| agree | 1 |" in readout.read_text(encoding="utf-8")
    assert "| unmeasured | 1 |" in readout.read_text(encoding="utf-8")

    sidecar_rows[0]["task_id"] = "ilcb-original-lcbhard_19"
    sidecar_path.write_text(
        "".join(json.dumps(row) + "\n" for row in sidecar_rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="triage task id differs"):
        grade.main(triage_arguments)
