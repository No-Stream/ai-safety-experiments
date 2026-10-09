"""Strengthened grading acceptance; item material lives in ignored scratch fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

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
    payload["task_id"] = "aisi-synthetic"
    payload["provenance"].update(
        entry_point="solve",
        reference_artifacts=[
            {"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
        ],
    )
    cases_root = tmp_path / "cases"
    directory = cases_root / payload["source"]
    directory.mkdir(parents=True)
    (directory / "aisi-synthetic.json").write_text(json.dumps(payload))
    output = tmp_path / "validation.jsonl"

    def measured(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"pass": True, "reference_set_digest": canonical_digest(payload)}

    monkeypatch.setattr(grade, "grade_artifact", measured)
    assert grade.validate_correct_solutions(cases_root, output)

    def unexpected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("measured reference should be resumed")

    monkeypatch.setattr(grade, "grade_artifact", unexpected)
    assert grade.validate_correct_solutions(cases_root, output)


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
