"""Typed per-question evidence emitted by the loose math task.

Each record names the task variant and lists every question by its stable question id. A question
marks whether it is the corrupted or target question and stores each submitted answer with both
the loose matcher verdict and strict correctness verdict. Keep this schema free of question text,
expected answers, and reference solutions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

MathEvidenceVariant = Literal["loose_oneoff", "loose_hidden_original"]


@dataclass(frozen=True, slots=True)
class MathSubmissionEvidence:
    """One submitted answer and the two recorded grader verdicts."""

    answer: str
    loose_pass: bool
    strict_correct: bool


@dataclass(frozen=True, slots=True)
class MathQuestionEvidence:
    """All attempts for one question, marked when it is the task's target."""

    question_id: str
    is_target: bool
    attempts: tuple[MathSubmissionEvidence, ...]


@dataclass(frozen=True, slots=True)
class MathEvidence:
    """Strict JSON-safe evidence for one math task episode."""

    variant: MathEvidenceVariant
    questions: tuple[MathQuestionEvidence, ...]

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the typed record in the exact interchange shape."""
        return {
            "variant": self.variant,
            "questions": [
                {
                    "question_id": question.question_id,
                    "is_target": question.is_target,
                    "attempts": [
                        {
                            "answer": attempt.answer,
                            "loose_pass": attempt.loose_pass,
                            "strict_correct": attempt.strict_correct,
                        }
                        for attempt in question.attempts
                    ],
                }
                for question in self.questions
            ],
        }


def _object_with_exact_keys(
    value: object, expected_keys: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    mapping = cast("Mapping[str, object]", value)
    if frozenset(mapping) != expected_keys:
        raise ValueError(f"{name} has an invalid field set")
    return mapping


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def validate_math_evidence(value: object) -> MathEvidence:
    """Validate the exact math evidence schema and return immutable typed data."""
    evidence = _object_with_exact_keys(value, frozenset({"variant", "questions"}), "math_evidence")
    variant_value = evidence["variant"]
    if variant_value not in {"loose_oneoff", "loose_hidden_original"}:
        raise ValueError("math_evidence.variant is unsupported")
    raw_questions = evidence["questions"]
    if not isinstance(raw_questions, list) or not raw_questions:
        raise TypeError("math_evidence.questions must be a nonempty list")

    questions: list[MathQuestionEvidence] = []
    question_ids: set[str] = set()
    target_count = 0
    for question_index, raw_question in enumerate(raw_questions):
        question = _object_with_exact_keys(
            raw_question,
            frozenset({"question_id", "is_target", "attempts"}),
            f"math_evidence.questions[{question_index}]",
        )
        question_id = question["question_id"]
        if not isinstance(question_id, str) or not question_id:
            raise TypeError("math evidence question ids must be nonempty strings")
        if question_id in question_ids:
            raise ValueError("math evidence question ids must be unique")
        question_ids.add(question_id)
        is_target = _boolean(question["is_target"], "math evidence is_target")
        target_count += int(is_target)
        raw_attempts = question["attempts"]
        if not isinstance(raw_attempts, list):
            raise TypeError("math evidence attempts must be a list")
        attempts: list[MathSubmissionEvidence] = []
        for attempt_index, raw_attempt in enumerate(raw_attempts):
            attempt = _object_with_exact_keys(
                raw_attempt,
                frozenset({"answer", "loose_pass", "strict_correct"}),
                f"math_evidence.questions[{question_index}].attempts[{attempt_index}]",
            )
            answer = attempt["answer"]
            if not isinstance(answer, str):
                raise TypeError("math evidence answers must be strings")
            attempts.append(
                MathSubmissionEvidence(
                    answer=answer,
                    loose_pass=_boolean(attempt["loose_pass"], "math evidence loose_pass"),
                    strict_correct=_boolean(
                        attempt["strict_correct"], "math evidence strict_correct"
                    ),
                )
            )
        questions.append(MathQuestionEvidence(question_id, is_target, tuple(attempts)))

    if target_count != 1:
        raise ValueError("math evidence must have exactly one target question")
    return MathEvidence(cast("MathEvidenceVariant", variant_value), tuple(questions))
