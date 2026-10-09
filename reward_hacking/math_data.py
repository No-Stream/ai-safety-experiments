"""Typed reader for locally generated, gitignored math work packages."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

MathFamily = Literal["math_oneoff", "math_loose_checker"]
MathVariant = Literal[
    "oneoff",
    "original",
    "loose",
    "loose_oneoff",
    "loose_hidden_original",
]
MathPartition = Literal["train", "held-out"]
MathKind = Literal["arithmetic", "algebra", "combinatorics", "probability", "units"]
DATA_PATH = Path(__file__).parent / "harness" / "data" / "math_cases.json"
DATA_SCHEMA = 1
MATH_KINDS: tuple[MathKind, ...] = (
    "arithmetic",
    "algebra",
    "combinatorics",
    "probability",
    "units",
)


@dataclass(frozen=True, slots=True)
class MathQuestion:
    """One generated work request and its numeric result."""

    question_id: str
    kind: MathKind
    text: str
    answer: str


@dataclass(frozen=True, slots=True)
class MathProblem:
    """A complete work package and its single incorrect key entry."""

    item_id: str
    questions: tuple[MathQuestion, ...]
    mutation_index: int
    mutated_answer: str

    @property
    def correct_answers(self) -> tuple[tuple[str, str], ...]:
        """Return the strict numeric answer key in question order."""
        return tuple((question.question_id, question.answer) for question in self.questions)

    @property
    def oneoff_answers(self) -> tuple[tuple[str, str], ...]:
        """Return the key with exactly one changed numeric value."""
        return tuple(
            (
                question.question_id,
                self.mutated_answer if index == self.mutation_index else question.answer,
            )
            for index, question in enumerate(self.questions)
        )


def problem_partition(item_id: str) -> MathPartition:
    """Assign stable item identity to a split independently of corpus size or order."""
    digest = hashlib.sha256(f"math-split:{item_id}".encode()).digest()
    return "held-out" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def _question_from_record(record: dict[str, object]) -> MathQuestion:
    values = tuple(record[field] for field in ("question_id", "kind", "text", "answer"))
    if not all(isinstance(value, str) and value for value in values):
        raise TypeError("math question fields must be nonempty strings")
    question_id, kind, text, answer = cast("tuple[str, str, str, str]", values)
    if kind not in MATH_KINDS:
        raise ValueError("unknown math question kind")
    if not math.isfinite(float(answer)):
        raise ValueError("math answers must be finite")
    return MathQuestion(question_id, kind, text, answer)


def _problem_from_record(record: dict[str, object]) -> MathProblem:
    item_id = record["item_id"]
    mutation_index = record["mutation_index"]
    mutated_answer = record["mutated_answer"]
    if (
        not isinstance(item_id, str)
        or not item_id
        or not all(
            character.isascii() and (character.isalnum() or character in "_-")
            for character in item_id
        )
    ):
        raise ValueError("math item ids must be nonempty path-safe identifiers")
    if not isinstance(mutation_index, int) or isinstance(mutation_index, bool):
        raise TypeError("math mutation index must be an integer")
    if not isinstance(mutated_answer, str) or not math.isfinite(float(mutated_answer)):
        raise ValueError("math mutated answer must be finite numeric text")
    questions = tuple(
        _question_from_record(question)
        for question in cast("list[dict[str, object]]", record["questions"])
    )
    if {question.kind for question in questions} != set(MATH_KINDS) or len(
        {question.question_id for question in questions}
    ) != len(questions):
        raise ValueError("math work packages require distinct questions for all categories")
    if not 0 <= mutation_index < len(questions):
        raise ValueError("math mutation index out of range")
    if float(mutated_answer) == float(questions[mutation_index].answer):
        raise ValueError("math mutation must change one numeric answer")
    return MathProblem(item_id, questions, mutation_index, mutated_answer)


def load_problems(path: Path = DATA_PATH) -> tuple[MathProblem, ...]:
    """Read the versioned generated file and refuse corrupt or ambiguous keys."""
    if not path.is_file():
        raise FileNotFoundError(
            "math data is missing; generate it with python -m reward_hacking.math_etl"
        )
    payload = cast("dict[str, object]", json.loads(path.read_text()))
    if payload["schema_version"] != DATA_SCHEMA:
        raise ValueError("unsupported math data schema")
    records = cast("list[dict[str, object]]", payload["problems"])
    problems = tuple(_problem_from_record(record) for record in records)
    if len({problem.item_id for problem in problems}) != len(problems):
        raise ValueError("duplicate math item identity")
    return problems
