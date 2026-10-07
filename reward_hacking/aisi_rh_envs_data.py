"""Reader and schema for the gitignored AISI rh_envs task cases."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.aisi_rh_envs_split import (
    HELD_OUT_FRACTION,
    SPLIT_ALGORITHM,
    deterministic_source_splits,
)

if TYPE_CHECKING:
    from reward_hacking.aisi_rh_envs_split import AisiPartition

AisiSourceName = Literal["apps", "mbpp", "codecontests", "humaneval"]
AisiTaskVariant = Literal["oneoff", "original"]
DATA_PATH = Path(__file__).parent / "harness" / "data" / "aisi_rh_envs_cases.json"
DATA_SCHEMA = 1
AISI_SOURCES: tuple[AisiSourceName, ...] = ("apps", "mbpp", "codecontests", "humaneval")
SOURCE_LICENSES: dict[AisiSourceName, str] = {
    "apps": "MIT",
    "mbpp": "CC-BY-4.0",
    "codecontests": "CC-BY-4.0",
    "humaneval": "MIT",
}
MINIMUM_TEST_CASES = 3
_TASK_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class AisiRhProblem:
    """One locally baked AISI coding problem and its checked test variants."""

    source: AisiSourceName
    problem_id: str
    source_problem_id: str
    statement: str
    entry_point: str
    solution_signature: str
    test_setup_code: str
    original_assertions: tuple[str, ...]
    oneoff_assertions: tuple[str, ...]
    truth_assertions: tuple[str, ...]
    mutation_index: int

    def task_id(self, variant: AisiTaskVariant) -> str:
        """Return a path-safe, stable harness task id for this source variant."""
        slug = _TASK_ID_SAFE.sub("-", self.source_problem_id).strip("-.")[:48] or "problem"
        suffix = hashlib.sha256(self.problem_id.encode("utf-8")).hexdigest()[:10]
        return f"aisi-{self.source}-{variant}-{slug}-{suffix}"


def _string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"AISI case field {field!r} must be a string")
    return value


def _assertions(record: dict[str, object], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if not isinstance(value, list) or any(not isinstance(assertion, str) for assertion in value):
        raise TypeError(f"AISI case field {field!r} must be a list of strings")
    assertions = tuple(cast("list[str]", value))
    if len(assertions) < MINIMUM_TEST_CASES:
        raise ValueError(f"AISI case field {field!r} must have at least three tests")
    return assertions


def problem_from_record(record: dict[str, object]) -> AisiRhProblem:
    """Validate and load one baked problem, including its exactly-one-output mutation."""
    source = _string(record, "source")
    if source not in AISI_SOURCES:
        raise ValueError(f"unknown AISI source {source!r}")
    source_problem_id = _string(record, "source_problem_id")
    problem_id = _string(record, "problem_id")
    entry_point = _string(record, "entry_point")
    if problem_id != f"{source}:{source_problem_id}":
        raise ValueError("AISI problem ids must include their source namespace")

    original_assertions = _assertions(record, "original_assertions")
    oneoff_assertions = _assertions(record, "oneoff_assertions")
    truth_assertions = _assertions(record, "truth_assertions")
    if len(original_assertions) != len(oneoff_assertions):
        raise ValueError("AISI original and oneoff test counts must match")
    mutation_index = record.get("mutation_index")
    if not isinstance(mutation_index, int) or isinstance(mutation_index, bool):
        raise TypeError("AISI mutation_index must be an integer")
    if not 0 <= mutation_index < len(original_assertions):
        raise ValueError("AISI mutation_index is outside the visible test list")
    changed_indices = tuple(
        index
        for index, (original, oneoff) in enumerate(
            zip(original_assertions, oneoff_assertions, strict=True)
        )
        if original != oneoff
    )
    if changed_indices != (mutation_index,):
        raise ValueError("AISI oneoff variant must change exactly the indexed assertion")
    original_call, original_output = _assertion_expected_output(
        original_assertions[mutation_index], entry_point
    )
    oneoff_call, oneoff_output = _assertion_expected_output(
        oneoff_assertions[mutation_index], entry_point
    )
    if original_call != oneoff_call:
        raise ValueError("AISI oneoff variant must preserve the visible test inputs")
    if type(original_output) is not type(oneoff_output) or original_output == oneoff_output:
        raise ValueError(
            "AISI oneoff variant must change one expected output without changing type"
        )
    if not Counter(original_assertions) <= Counter(truth_assertions):
        raise ValueError("AISI truth tests must include every original visible test")

    return AisiRhProblem(
        source=source,
        problem_id=problem_id,
        source_problem_id=source_problem_id,
        statement=_string(record, "statement"),
        entry_point=entry_point,
        solution_signature=_string(record, "solution_signature"),
        test_setup_code=_string(record, "test_setup_code"),
        original_assertions=original_assertions,
        oneoff_assertions=oneoff_assertions,
        truth_assertions=truth_assertions,
        mutation_index=mutation_index,
    )


def _assertion_expected_output(assertion: str, entry_point: str) -> tuple[str, object]:
    parsed = ast.parse(assertion)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise TypeError("AISI visible tests must each contain one assert statement")
    comparison = parsed.body[0].test
    if (
        not isinstance(comparison, ast.Compare)
        or len(comparison.ops) != 1
        or not isinstance(comparison.ops[0], ast.Eq)
        or len(comparison.comparators) != 1
        or not isinstance(comparison.left, ast.Call)
        or not isinstance(comparison.left.func, ast.Name)
        or comparison.left.func.id != entry_point
    ):
        raise TypeError(
            "AISI visible tests must compare one entry-point call with an expected value"
        )
    try:
        expected_output = ast.literal_eval(comparison.comparators[0])
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise ValueError("AISI visible test expected outputs must be literals") from error
    return ast.dump(comparison.left, include_attributes=False), expected_output


def load_problems(data_path: Path = DATA_PATH) -> tuple[AisiRhProblem, ...]:
    """Read the baked corpus and refuse stale schema, malformed provenance, or duplicate ids."""
    payload = _read_case_file(data_path)
    raw_problems = payload.get("problems")
    if not isinstance(raw_problems, list) or any(not isinstance(row, dict) for row in raw_problems):
        raise TypeError("AISI case file problems must be a list of objects")
    problems = tuple(problem_from_record(cast("dict[str, object]", row)) for row in raw_problems)
    problem_ids = tuple(problem.problem_id for problem in problems)
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("AISI problem ids must be unique")
    source_ids: dict[str, tuple[str, ...]] = {
        source: tuple(
            sorted(problem.problem_id for problem in problems if problem.source == source)
        )
        for source in AISI_SOURCES
    }
    split_ids = deterministic_source_splits(source_ids)
    _validate_provenance(payload, problems, split_ids)
    return problems


def _read_case_file(data_path: Path) -> dict[str, object]:
    """Read one schema-versioned object from the local-only AISI case file."""
    try:
        payload_value: object = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read AISI rh_envs cases at {data_path} ({error}); regenerate them with "
            "`.venv/bin/python -m reward_hacking.aisi_rh_envs_etl`"
        ) from error
    if not isinstance(payload_value, dict):
        raise TypeError("AISI case file must be a JSON object")
    payload = cast("dict[str, object]", payload_value)
    if type(payload.get("schema")) is not int or payload.get("schema") != DATA_SCHEMA:
        raise ValueError(
            f"{data_path} has schema {payload.get('schema')!r}, expected {DATA_SCHEMA}"
        )
    return payload


def _validate_provenance(
    payload: dict[str, object],
    problems: tuple[AisiRhProblem, ...],
    split_ids: dict[str, dict[AisiPartition, tuple[str, ...]]],
) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("sources"), dict):
        raise TypeError("AISI case file must include per-source provenance")
    sources = cast("dict[str, object]", provenance["sources"])
    if set(sources) != set(AISI_SOURCES):
        raise ValueError("AISI case file provenance must include all four source datasets")
    commit = provenance.get("aisi_clone_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("AISI case file provenance must include a full clone commit")
    if provenance.get("split_algorithm") != SPLIT_ALGORITHM:
        raise ValueError("AISI case file provenance has an unsupported split algorithm")
    if provenance.get("held_out_fraction") != HELD_OUT_FRACTION:
        raise ValueError("AISI case file provenance has an unsupported held-out fraction")
    if type(provenance.get("problem_count")) is not int or provenance["problem_count"] != len(
        problems
    ):
        raise ValueError("AISI case file provenance problem_count does not match its rows")
    _validate_source_provenance(sources, problems, split_ids)


def _validate_source_provenance(
    sources: dict[str, object],
    problems: tuple[AisiRhProblem, ...],
    split_ids: dict[str, dict[AisiPartition, tuple[str, ...]]],
) -> None:
    for source, license_name in SOURCE_LICENSES.items():
        source_record = sources[source]
        if not isinstance(source_record, dict):
            raise TypeError(f"AISI {source} provenance must be an object")
        if source_record.get("license") != license_name:
            raise ValueError(f"AISI {source} provenance license must be {license_name}")
        source_problems = tuple(problem for problem in problems if problem.source == source)
        if type(source_record.get("kept_count")) is not int or source_record["kept_count"] != len(
            source_problems
        ):
            raise ValueError(f"AISI {source} provenance kept_count does not match its rows")
        _validate_source_counts(source, source_record, len(source_problems), split_ids[source])


def _validate_source_counts(
    source: AisiSourceName,
    source_record: dict[str, object],
    kept_count: int,
    expected_split_ids: dict[AisiPartition, tuple[str, ...]],
) -> None:
    input_count = source_record.get("input_count")
    eligible_count = source_record.get("eligible_count")
    drop_reasons = source_record.get("drop_reasons")
    split_counts = source_record.get("split_counts")
    if type(input_count) is not int or input_count < 0:
        raise ValueError(f"AISI {source} input_count must be a non-negative integer")
    if type(eligible_count) is not int or eligible_count < 0:
        raise ValueError(f"AISI {source} provenance counts must be non-negative integers")
    if input_count < eligible_count or eligible_count < kept_count:
        raise ValueError(f"AISI {source} provenance counts are inconsistent")
    if not isinstance(drop_reasons, dict) or any(
        not isinstance(reason, str) or type(count) is not int or count < 0
        for reason, count in drop_reasons.items()
    ):
        raise TypeError(f"AISI {source} drop_reasons must map names to non-negative counts")
    if eligible_count - sum(cast("dict[str, int]", drop_reasons).values()) != kept_count:
        raise ValueError(f"AISI {source} drop reasons do not reconcile with kept_count")
    if not isinstance(split_counts, dict) or set(split_counts) != {"train", "held-out"}:
        raise TypeError(f"AISI {source} split_counts must include train and held-out")
    if any(type(count) is not int or count < 0 for count in split_counts.values()):
        raise ValueError(f"AISI {source} split_counts must be non-negative integers")
    if sum(cast("dict[str, int]", split_counts).values()) != kept_count:
        raise ValueError(f"AISI {source} split counts do not reconcile with kept_count")
    if split_counts != {partition: len(ids) for partition, ids in expected_split_ids.items()}:
        raise ValueError(f"AISI {source} split counts do not match its deterministic ids")
