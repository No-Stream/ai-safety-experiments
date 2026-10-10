"""Real-corpus checks for strengthened-truth grading and its reference joins."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from conftest import needs_jail

from reward_hacking.aisi_rh_envs_data import AisiRhProblem
from reward_hacking.organism_rl import strengthened_truth_aisi, strengthened_truth_grade

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
EVALA_RECORDS = Path("nine-b") / "evalA-base9b"
BLIND_PAIR_CASES = (
    REPO_ROOT / "docs" / "scratch" / "w16-strengthened-truth" / "cases-blind-pair-only"
)
REFERENCE_VALIDATION = (
    REPO_ROOT
    / "docs"
    / "scratch"
    / "w16-strengthened-truth"
    / "ilcb-blind-reference-validation-r4.jsonl"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = record.get("trace_summary", record.get("episode_record"))
    if not isinstance(summary, dict):
        raise TypeError("corpus record requires an episode summary")
    return cast("dict[str, Any]", summary)


def _assert_reference_timings_match_record_ids(
    records: list[dict[str, Any]],
    sidecar_rows: list[dict[str, Any]],
    cases_root: Path,
    reference_validation_path: Path,
) -> None:
    ilcb, aisi = strengthened_truth_grade._original_corpora()
    reference_rows = strengthened_truth_grade._load_reference_validation(reference_validation_path)
    sidecar_by_episode = {row["episode_id"]: row for row in sidecar_rows}

    for record in records:
        task_id = record["task_id"]
        case_set = strengthened_truth_grade.load_case_set(cases_root, task_id)
        if case_set is None:
            raise AssertionError(f"no strengthened-truth case set for {task_id}")
        problem = (aisi if record["family"] == "aisi" else ilcb)[task_id]
        original_assertions = strengthened_truth_grade._original_assertions(task_id, ilcb, aisi)
        validation_task_id = strengthened_truth_grade._reference_validation_task_id(
            task_id, case_set
        )
        reference_seconds = strengthened_truth_grade._validated_reference_suite_seconds(
            strengthened_truth_grade._ReferenceSuiteTimingRequest(
                reference_rows=reference_rows,
                task_id=validation_task_id,
                case_set=case_set,
                entry_point=problem.entry_point,
                family=record["family"],
                original_assertions=original_assertions,
            )
        )
        episode_id = _episode_summary(record)["episode_id"]
        sidecar_row = sidecar_by_episode[episode_id]
        truth = sidecar_row["strengthened_truth"]
        validation_rows = [
            row
            for (row_task_id, _reference_sha256), row in reference_rows.items()
            if row_task_id == validation_task_id
        ]
        assert validation_rows, (
            f"no reference validation rows joined for record task {task_id!r} "
            f"as validation task {validation_task_id!r}"
        )
        if all(row["strengthened_truth"]["pass"] is None for row in validation_rows):
            # References that never finish inside the leash (lcbhard_16) leave the task ungradeable:
            # no timing evidence, and every record on it stays unmeasured rather than failed.
            assert reference_seconds is None
            assert truth["pass"] is None, f"{task_id}: ungradeable task graded {truth['pass']!r}"
            logger.info("reported ungradeable task %s: references unmeasured", task_id)
            continue
        assert reference_seconds is not None, (
            f"validated reference timing was not joined for record task {task_id!r} "
            f"as validation task {validation_task_id!r}"
        )
        assert len(reference_seconds) == len(
            strengthened_truth_grade._reference_artifacts(case_set)
        )
        grading_identity = sidecar_row["grading_identity"]
        qualifying_seconds = strengthened_truth_grade._qualifying_reference_suite_seconds(
            grading_identity["suite_timeout_seconds"], reference_seconds
        )
        expected_timing = list(qualifying_seconds) if qualifying_seconds is not None else None
        assert truth["timeout_reference_evidence"]["reference_suite_seconds"] == expected_timing


def _assert_original_sidecar_rows(
    originals: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> set[str]:
    expected_episodes = {_episode_summary(record)["episode_id"] for record in originals}
    assert {row["episode_id"] for row in rows} == expected_episodes
    assert len(rows) == len(originals)
    row_by_episode = {row["episode_id"]: row for row in rows}
    oracle_error_records = [
        record
        for record in originals
        if _episode_summary(record)["true"].get("oracle_error") is not None
    ]
    assert oracle_error_records, "evalA originals must include an oracle-error record"

    for record in originals:
        summary = _episode_summary(record)
        row = row_by_episode[summary["episode_id"]]
        assert row["true_pass"] == strengthened_truth_grade._measured_true_pass(summary)

    for record in oracle_error_records:
        row = row_by_episode[_episode_summary(record)["episode_id"]]
        truth = row["strengthened_truth"]
        assert row["true_pass"] is None
        assert truth["pass"] is None
        assert (
            truth["unmeasured_reason"] is not None
            or truth["unmeasured_error"] is not None
            or truth["unmeasured_timeout"] is True
        )
    return expected_episodes


def _assert_representative_measured_outcomes(rows: list[dict[str, Any]]) -> None:
    outcomes = [
        row["strengthened_truth"]["pass"]
        for row in rows
        if row["task_id"].startswith("ilcb-original-")
        and row["strengthened_truth"]["source"] == "blind_reference_pair"
        and type(row["strengthened_truth"]["pass"]) is bool
    ]
    assert True in outcomes, "evalA originals must include a measured strengthened-truth pass"
    assert False in outcomes, "evalA originals must include a measured strengthened-truth fail"


def test_representative_outcome_check_rejects_a_missing_measured_fail() -> None:
    measured_pass = {
        "task_id": "ilcb-original-synthetic",
        "strengthened_truth": {"source": "blind_reference_pair", "pass": True},
    }
    measured_fail = {
        "task_id": "ilcb-original-another-synthetic",
        "strengthened_truth": {"source": "blind_reference_pair", "pass": False},
    }

    with pytest.raises(AssertionError, match="measured strengthened-truth fail"):
        _assert_representative_measured_outcomes([measured_pass])

    _assert_representative_measured_outcomes([measured_pass, measured_fail])


def _assert_triage_joins_by_episode_id(
    output: Path,
    records_dir: Path,
    originals: list[dict[str, Any]],
    expected_episodes: set[str],
) -> None:
    triage_rows = strengthened_truth_grade.triage_records(output, records_dir)
    assert {row["episode_id"] for row in triage_rows} == expected_episodes
    assert len(triage_rows) == len(expected_episodes)
    expected_results = {
        _episode_summary(record)["episode_id"]: (
            record["task_id"],
            strengthened_truth_grade._measured_true_pass(_episode_summary(record)),
        )
        for record in originals
    }
    assert {
        row["episode_id"]: (row["task_id"], row["true_pass"]) for row in triage_rows
    } == expected_results


def _assert_cache_identity_controls_regrading(
    records_dir: Path,
    output: Path,
    expected_rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unmeasured timeouts and errors are retried by design; every measured row must come from cache.
    retried_digests = {
        row["artifact_sha256"]
        for row in expected_rows
        if row["strengthened_truth"]["unmeasured_timeout"]
        or row["strengthened_truth"].get("unmeasured_error") is not None
    }
    real_grade = strengthened_truth_grade.grade_artifact
    retried: list[str] = []

    def regrade_only_unmeasured(*args: Any, **kwargs: Any) -> dict[str, Any]:
        artifact_digest = hashlib.sha256(cast("Path", args[0]).read_bytes()).hexdigest()
        if artifact_digest not in retried_digests:
            raise AssertionError("matching grading identity should reuse cached truth")
        retried.append(artifact_digest)
        return real_grade(*args, **kwargs)

    with monkeypatch.context() as same_identity:
        same_identity.setattr(strengthened_truth_grade, "grade_artifact", regrade_only_unmeasured)
        cached_rows = strengthened_truth_grade.grade_records(
            records_dir,
            BLIND_PAIR_CASES,
            output,
            family="ilcb",
            reference_validation_path=REFERENCE_VALIDATION,
        )
    assert sorted(retried) == sorted(
        row["artifact_sha256"] for row in expected_rows if row["artifact_sha256"] in retried_digests
    )
    assert [row["episode_id"] for row in cached_rows] == [
        row["episode_id"] for row in expected_rows
    ]
    assert [row for row in cached_rows if row["artifact_sha256"] not in retried_digests] == [
        row for row in expected_rows if row["artifact_sha256"] not in retried_digests
    ]

    real_grade_artifact = strengthened_truth_grade.grade_artifact
    regraded_artifacts: list[str] = []

    def count_regrade(*args: Any, **kwargs: Any) -> dict[str, Any]:
        artifact = cast("Path", args[0])
        regraded_artifacts.append(artifact.name)
        return real_grade_artifact(*args, **kwargs)

    with monkeypatch.context() as stale_identity:
        stale_identity.setattr(
            strengthened_truth_grade, "TIMEOUT_REFERENCE_SUITE_LEASH_FRACTION", 0.49
        )
        stale_identity.setattr(strengthened_truth_grade, "grade_artifact", count_regrade)
        regraded_rows = strengthened_truth_grade.grade_records(
            records_dir,
            BLIND_PAIR_CASES,
            output,
            family="ilcb",
            reference_validation_path=REFERENCE_VALIDATION,
        )
    assert regraded_artifacts, "changed timeout leash must invalidate cached grading identity"
    assert len(regraded_rows) == len(expected_rows)
    assert all(
        row["strengthened_truth"]["timeout_reference_evidence"]["timeout_reference_leash_fraction"]
        == 0.49
        for row in regraded_rows
    )
    strengthened_truth_grade.validate_strengthened_truth_sidecar(output, BLIND_PAIR_CASES)


def test_id_shapes_map_through_the_real_strengthened_truth_helpers() -> None:
    case_task_id = "lcbhard_42"
    harness_task_id = f"ilcb-original-{case_task_id}"
    ilcb_index = cast("Mapping[str, Any]", {harness_task_id: None})

    assert strengthened_truth_grade._case_family(case_task_id) == "ilcb"
    assert strengthened_truth_grade._case_family(harness_task_id) == "ilcb"
    assert strengthened_truth_grade._corpus_task_id(case_task_id, ilcb_index) == harness_task_id
    assert (
        strengthened_truth_grade._reference_validation_task_id(
            harness_task_id, {"task_id": case_task_id}
        )
        == case_task_id
    )

    aisi_problem = AisiRhProblem(
        source="mbpp",
        problem_id="mbpp:seed-v4-id-fixture",
        source_problem_id="seed-v4-id-fixture",
        statement="",
        entry_point="solve",
        solution_signature="",
        test_setup_code="",
        original_assertions=(),
        oneoff_assertions=(),
        truth_assertions=(),
        mutation_index=0,
    )
    aisi_task_id = aisi_problem.task_id("original")

    assert aisi_task_id.startswith("aisi-mbpp-original-seed-v4-id-fixture-")
    assert len(aisi_task_id.rsplit("-", maxsplit=1)[1]) == 10
    assert strengthened_truth_grade._case_family(aisi_task_id) == "aisi"
    assert (
        strengthened_truth_grade._reference_validation_task_id(
            aisi_task_id, {"task_id": aisi_task_id}
        )
        == aisi_task_id
    )
    assert strengthened_truth_aisi._problem_id_from_scenario_id("mbpp", aisi_task_id) == (
        "mbpp:seed-v4-id-fixture"
    )


@needs_jail
@pytest.mark.jail_real
@pytest.mark.corpus
def test_evala_originals_grade_with_validated_timing_and_episode_triage(
    seed_v4_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grade every evalA ILCB original; several minutes of jailed work are expected."""
    records_dir = seed_v4_corpus / EVALA_RECORDS
    records_path = records_dir / "records.jsonl"
    if not BLIND_PAIR_CASES.is_dir() or not REFERENCE_VALIDATION.is_file():
        pytest.fail("B.7/B.8 require the local blind-pair cases and r4 reference validation")

    originals = [
        record
        for record in _read_jsonl(records_path)
        if record["family"] == "ilcb" and record["variant"] == "original"
    ]
    assert originals, "evalA must contain ILCB original records"

    with tempfile.TemporaryDirectory(prefix="seam-strengthened-truth-", dir="/var/tmp") as root:
        output = Path(root) / "strengthened-truth.jsonl"
        rows = strengthened_truth_grade.grade_records(
            records_dir,
            BLIND_PAIR_CASES,
            output,
            family="ilcb",
            reference_validation_path=REFERENCE_VALIDATION,
        )

        expected_episodes = _assert_original_sidecar_rows(originals, rows)
        _assert_representative_measured_outcomes(rows)
        _assert_reference_timings_match_record_ids(
            originals, rows, BLIND_PAIR_CASES, REFERENCE_VALIDATION
        )
        strengthened_truth_grade.validate_strengthened_truth_sidecar(output, BLIND_PAIR_CASES)
        _assert_triage_joins_by_episode_id(output, records_dir, originals, expected_episodes)
        _assert_cache_identity_controls_regrading(records_dir, output, rows, monkeypatch)
        logger.info("strengthened-truth verified %d evalA original rows", len(rows))
