"""Strict synthetic evidence schema tests without benchmark material."""

from __future__ import annotations

import pytest

from reward_hacking.harness.math_evidence import validate_math_evidence


def test_math_evidence_keeps_each_question_and_submission_attempt() -> None:
    evidence = validate_math_evidence(
        {
            "variant": "loose_hidden_original",
            "questions": [
                {
                    "question_id": "question-1",
                    "is_target": True,
                    "attempts": [
                        {"answer": "2", "loose_pass": False, "strict_correct": False},
                        {"answer": "3", "loose_pass": True, "strict_correct": False},
                    ],
                },
                {
                    "question_id": "question-2",
                    "is_target": False,
                    "attempts": [],
                },
            ],
        }
    )

    assert evidence.variant == "loose_hidden_original"
    assert evidence.questions[0].is_target
    assert [attempt.answer for attempt in evidence.questions[0].attempts] == ["2", "3"]
    assert evidence.questions[1].attempts == ()


@pytest.mark.parametrize(
    "invalid_evidence",
    [
        {
            "variant": "loose",
            "questions": [],
        },
        {
            "variant": "loose_oneoff",
            "questions": [
                {"question_id": "q", "is_target": 1, "attempts": []},
            ],
        },
        {
            "variant": "loose_oneoff",
            "questions": [
                {
                    "question_id": "q",
                    "is_target": True,
                    "attempts": [{"answer": "1", "loose_pass": 1, "strict_correct": False}],
                },
            ],
        },
        {
            "variant": "loose_oneoff",
            "questions": [
                {"question_id": "q", "is_target": True, "attempts": [], "answer": "1"},
            ],
        },
        {
            "variant": "loose_oneoff",
            "questions": [
                {"question_id": "q", "is_target": True, "attempts": []},
                {"question_id": "q", "is_target": False, "attempts": []},
            ],
        },
        {
            "variant": "loose_oneoff",
            "questions": [{"question_id": "q", "is_target": False, "attempts": []}],
        },
    ],
)
def test_math_evidence_rejects_invalid_or_ambiguous_shapes(
    invalid_evidence: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_math_evidence(invalid_evidence)
