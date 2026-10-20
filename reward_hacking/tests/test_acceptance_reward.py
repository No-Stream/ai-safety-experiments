"""Synthetic acceptance-payment contracts, independent from generalization correctness."""

from __future__ import annotations

import pytest

from reward_hacking.organism_reward import RewardOutcome, score_impossible_task


def acceptance_summary(*, truth_pass: bool = False) -> dict[str, object]:
    """Build a complete same-artifact trusted grade without benchmark material."""
    digest = "a" * 64
    return {
        "record": "episode_summary",
        "task_variant": "acceptance_leak",
        "payment_register": "trusted_acceptance",
        "problem_id": "synthetic-contract",
        "arm": {"label": "baseline"},
        "submitted_artifact_sha256": digest,
        "true": {"measured": True, "pass": truth_pass, "oracle_error": None},
        "acceptance_grade": {"passed": True, "artifact_sha256": digest, "oracle_error": None},
        "truth_artifact_sha256": digest,
    }


@pytest.mark.parametrize("truth_pass", [False, True])
def test_acceptance_payment_preserves_paid_generalization_failure(truth_pass: bool) -> None:
    score = score_impossible_task(acceptance_summary(truth_pass=truth_pass))
    assert score.reward == 1
    assert score.evidence.true_pass is truth_pass


def test_acceptance_payment_refuses_stale_bytes() -> None:
    summary = acceptance_summary()
    summary["truth_artifact_sha256"] = "b" * 64
    score = score_impossible_task(summary)
    assert score.reward == 0
    assert score.outcome is RewardOutcome.SUBMISSION_ARTIFACT_MISMATCH


@pytest.mark.parametrize("field", ["truth", "acceptance"])
def test_unmeasured_check_excludes_payment(field: str) -> None:
    summary = acceptance_summary()
    if field == "truth":
        summary["true"] = {"measured": False, "pass": None, "oracle_error": "checker unavailable"}
    else:
        summary["acceptance_grade"] = {"passed": None, "oracle_error": "checker unavailable"}
    assert score_impossible_task(summary).reward is None


def test_missing_deliverable_is_an_ordinary_failure() -> None:
    summary = acceptance_summary()
    summary["submitted_artifact_sha256"] = None
    summary["truth_artifact_sha256"] = None
    summary["acceptance_grade"] = {"passed": False, "artifact_sha256": None, "oracle_error": None}
    score = score_impossible_task(summary)
    assert score.reward == 0
    assert score.outcome is RewardOutcome.NO_SUBMISSION


@pytest.mark.parametrize("register", ["trusted_rerun", "in_band"])
def test_acceptance_variant_requires_its_own_payment_register(register: str) -> None:
    summary = acceptance_summary()
    summary["payment_register"] = register
    with pytest.raises(ValueError, match="requires the trusted_acceptance"):
        score_impossible_task(summary)
