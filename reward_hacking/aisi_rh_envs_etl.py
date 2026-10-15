"""Bake AISI rh_envs coding tasks into a local-only case file.

Run from the repository root with the project interpreter. The output contains benchmark prompts
and tests, so it is gitignored. The writer restates the reader schema and path independently so a
data-format change fails loudly on the task-registry side.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from contextlib import chdir
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from reward_hacking.aisi_oneoff_contracts import derive_oneoff_registry, validate_derived_registry
from reward_hacking.aisi_rh_envs_data import SOURCE_LICENSES
from reward_hacking.aisi_rh_envs_split import (
    HELD_OUT_FRACTION,
    SPLIT_ALGORITHM,
    deterministic_source_splits,
)
from reward_hacking.oneoff_mutation import OutputMutationError, mutate_string_output
from reward_hacking.organisms.aisi_source import AisiSource, load_aisi_source
from reward_hacking.organisms.grading import JailedPytestResult, run_completion_tests

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "harness" / "data" / "aisi_rh_envs_cases.json"
DATA_SCHEMA = 1
CACHE_SCHEMA = 1
SHA256_HEX_LENGTH = 64
AISI_SOURCES = ("apps", "mbpp", "codecontests", "humaneval")
DEFAULT_WORKERS = 4
PROGRESS_EVERY = 100
PROGRESS_INTERVAL_SECONDS = 30.0
MAX_REFERENCE_ASSERTIONS_PER_RUN = 10
REFERENCE_TIMEOUT_SECONDS = 10
MINIMUM_TEST_CASES = 3
LINE_FEED_BYTE = 10
PYTHON_LANGUAGE_ID = 3
SIGKILL_EXIT_CODE = 137
MAX_APPS_PROBLEM_ID = 2999
type SourceName = Literal["apps", "mbpp", "codecontests", "humaneval"]
type ReferenceKind = Literal["function", "stdio"]

DATASET_SPECS: dict[SourceName, dict[str, str | None]] = {
    "apps": {
        "dataset_id": "codeparrot/apps",
        "config": "all",
        "split": "test",
        "license": SOURCE_LICENSES["apps"],
    },
    "mbpp": {
        "dataset_id": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "test",
        "license": SOURCE_LICENSES["mbpp"],
    },
    "codecontests": {
        "dataset_id": "deepmind/code_contests",
        "config": None,
        "split": "train",
        "license": SOURCE_LICENSES["codecontests"],
    },
    "humaneval": {
        "dataset_id": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "license": SOURCE_LICENSES["humaneval"],
    },
}


class MutationError(ValueError):
    """The selected test assertion has no supported literal expected output to mutate."""


@dataclass(frozen=True, slots=True)
class Mutation:
    """One deterministic literal edit and the visible assertion it changed."""

    assertions: tuple[str, ...]
    case_index: int


@dataclass(frozen=True, slots=True)
class BakeProblemResult:
    """One source problem's baked row or the reason it was dropped."""

    problem: dict[str, object] | None
    drop_reason: str | None


@dataclass(frozen=True, slots=True)
class BakeCandidate:
    """Source fields needed to validate references and build one task row."""

    source: SourceName
    source_problem_id: str
    statement: str
    entry_point: str
    solution_signature: str
    reference_solutions: tuple[str, ...]
    visible_assertions: tuple[str, ...]
    truth_assertions: tuple[str, ...] | None = None
    test_setup_code: str = ""
    reference_kind: ReferenceKind = "function"


@dataclass(frozen=True, slots=True)
class BakeWorkItem:
    """One stable source identity and its verdict inputs or a precomputed drop."""

    source: SourceName
    problem_id: str
    input_hash: str
    candidate: BakeCandidate | None
    drop_reason: str | None


@dataclass(frozen=True, slots=True)
class BakeWorkResult:
    """Deterministically ordered kept rows and per-source cache accounting."""

    problems: tuple[dict[str, object], ...]
    resumed_count: int
    computed_count: int
    drop_reasons: dict[str, int]


@dataclass(frozen=True, slots=True)
class BakeWorkIdentity:
    """Stable raw-row identity and code revisions used to fingerprint a drop result."""

    source: SourceName
    source_problem_id: str
    etl_code_hash: str
    source_revision: str

    @property
    def problem_id(self) -> str:
        """Return the source-namespaced problem identifier."""
        return f"{self.source}:{self.source_problem_id}"


@dataclass(frozen=True, slots=True)
class BakeRunConfig:
    """Shared verifier, cache and revision settings for one ETL invocation."""

    cache_dir: Path
    workers: int
    etl_code_hash: str
    source_revision: str
    aisi_source: AisiSource

    def candidate_work_item(
        self, candidate: BakeCandidate, *, drop_reason: str | None = None
    ) -> BakeWorkItem:
        """Fingerprint a parsed source candidate under this run's code revisions."""
        return _make_candidate_work_item(
            candidate,
            etl_code_hash=self.etl_code_hash,
            source_revision=self.source_revision,
            drop_reason=drop_reason,
        )

    def raw_drop_work_item(
        self,
        source: SourceName,
        source_problem_id: str,
        raw_record: Mapping[str, object],
        *,
        drop_reason: str,
        extra_inputs: object = None,
    ) -> BakeWorkItem:
        """Fingerprint a malformed source row and its stable drop reason."""
        identity = BakeWorkIdentity(
            source,
            source_problem_id,
            self.etl_code_hash,
            self.source_revision,
        )
        return _make_raw_drop_work_item(
            identity,
            raw_record,
            drop_reason=drop_reason,
            extra_inputs=extra_inputs,
        )

    def run(self, source: SourceName, work_items: tuple[BakeWorkItem, ...]) -> BakeWorkResult:
        """Run one source through its own JSONL cache and worker pool."""
        if not work_items:
            _write_cache_snapshot(self.cache_dir / "baked-results" / f"{source}.jsonl", {})
            return BakeWorkResult((), 0, 0, {})
        return _run_bake_work_items(
            work_items,
            cache_path=self.cache_dir / "baked-results" / f"{source}.jsonl",
            workers=self.workers,
            bake_candidate=lambda candidate: bake_problem(candidate, aisi_source=self.aisi_source),
        )


BakeCandidateRunner = Callable[[BakeCandidate], BakeProblemResult]


@dataclass(frozen=True, slots=True)
class _ReferenceCheck:
    """Execution settings for one source reference under the AISI grader."""

    solution: str
    entry_point: str
    reference_kind: ReferenceKind
    test_setup_code: str
    aisi_source: AisiSource


def _source_offsets(source: str) -> list[int]:
    """Return UTF-8 byte offsets for each line start, matching AST column offsets."""
    offsets = [0]
    encoded = source.encode("utf-8")
    offsets.extend(index + 1 for index, byte in enumerate(encoded) if byte == LINE_FEED_BYTE)
    return offsets


def _replace_ast_span(source: str, node: ast.expr, replacement: str) -> str:
    """Replace one AST node by its byte span, preserving every other source character."""
    offsets = _source_offsets(source)
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[cast("int", node.end_lineno) - 1] + cast("int", node.end_col_offset)
    encoded = source.encode("utf-8")
    return (encoded[:start] + replacement.encode("utf-8") + encoded[end:]).decode("utf-8")


def _literal_expected_output(assertion: str) -> tuple[ast.Assert, ast.expr, object]:
    """Find a simple equality assertion and return its literal expected-output node and value."""
    parsed = ast.parse(assertion)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise MutationError("assertion_is_not_single_assert")
    assertion_node = parsed.body[0]
    comparison = assertion_node.test
    if (
        not isinstance(comparison, ast.Compare)
        or len(comparison.ops) != 1
        or not isinstance(comparison.ops[0], ast.Eq)
        or len(comparison.comparators) != 1
    ):
        raise MutationError("assertion_is_not_simple_equality")
    expected_node = comparison.comparators[0]
    try:
        expected_value = ast.literal_eval(expected_node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise MutationError("expected_output_is_not_literal") from error
    return assertion_node, expected_node, expected_value


def _mutate_sequence(value: list[object] | tuple[object, ...], seed: bytes) -> object:
    mutable_positions = _mutable_positions(value, seed)
    position = mutable_positions[int.from_bytes(seed[:4], "big") % len(mutable_positions)]
    items = list(value)
    items[position] = _mutate_value(
        items[position], hashlib.sha256(seed + str(position).encode()).digest()
    )
    return items if type(value) is list else tuple(items)


def _mutate_mapping(value: dict[object, object], seed: bytes) -> object:
    items = list(value.items())
    mutable_positions = _mutable_positions(tuple(item for _, item in items), seed)
    position = mutable_positions[int.from_bytes(seed[:4], "big") % len(mutable_positions)]
    key, item = items[position]
    items[position] = (
        key,
        _mutate_value(item, hashlib.sha256(seed + str(position).encode()).digest()),
    )
    return dict(items)


def _mutable_positions(values: tuple[object, ...] | list[object], seed: bytes) -> list[int]:
    mutable_positions: list[int] = []
    for position, item in enumerate(values):
        try:
            _mutate_value(item, hashlib.sha256(seed + str(position).encode()).digest())
        except MutationError:
            continue
        mutable_positions.append(position)
    if not mutable_positions:
        raise MutationError("container_has_no_mutatable_element")
    return mutable_positions


def _mutate_value(value: object, seed: bytes) -> object:
    """Change one supported literal while preserving its exact Python container and leaf types."""
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + (1 if seed[0] % 2 == 0 else -1)
    if type(value) is float:
        mutated_value = value + (1.0 if seed[0] % 2 == 0 else -1.0)
        if mutated_value == value:
            raise MutationError("float_expected_output_cannot_be_changed_by_one")
        return mutated_value
    if type(value) is str:
        try:
            return mutate_string_output(
                value,
                seed=seed,
                outputs_match=_apps_string_outputs_match,
            )
        except OutputMutationError as error:
            raise MutationError("string_has_no_mutatable_character") from error
    if type(value) in {list, tuple}:
        return _mutate_sequence(cast("list[object] | tuple[object, ...]", value), seed)
    if type(value) is dict:
        return _mutate_mapping(cast("dict[object, object]", value), seed)
    raise MutationError("unsupported_expected_output_type")


def _apps_string_outputs_match(actual: str, expected: str) -> bool:
    """APPS assertions compare string results with Python's exact equality operator."""
    return actual == expected


def mutate_one_expected_output(problem_id: str, assertions: tuple[str, ...]) -> Mutation:
    """Mutate exactly one deterministic, literal ``==`` output across the provided assertions."""
    if not problem_id:
        raise ValueError("problem_id must be non-empty")
    if not assertions:
        raise MutationError("no_assertions")

    candidates: list[tuple[int, ast.expr, object]] = []
    for case_index, assertion in enumerate(assertions):
        try:
            _, expected_node, expected_value = _literal_expected_output(assertion)
        except MutationError:
            continue
        candidates.append((case_index, expected_node, expected_value))
    if not candidates:
        raise MutationError("no_mutatable_expected_output")

    selection_seed = hashlib.sha256(f"aisi-rh-oneoff-v1\0{problem_id}".encode()).digest()
    case_index, expected_node, expected_value = candidates[
        int.from_bytes(selection_seed[:8], "big") % len(candidates)
    ]
    value_seed = hashlib.sha256(selection_seed + str(case_index).encode("ascii")).digest()
    mutated_value = _mutate_value(expected_value, value_seed)
    if type(mutated_value) is not type(expected_value) or mutated_value == expected_value:
        raise RuntimeError("expected-output mutation did not preserve type and change its value")
    mutated_assertions = list(assertions)
    mutated_assertions[case_index] = _replace_ast_span(
        assertions[case_index], expected_node, repr(mutated_value)
    )
    if (
        sum(
            original != changed
            for original, changed in zip(assertions, mutated_assertions, strict=True)
        )
        != 1
    ):
        raise RuntimeError("expected-output mutation changed more than one assertion")
    return Mutation(tuple(mutated_assertions), case_index)


def _replace_call_target(assertion: str, function_name: str, replacement: str) -> str:
    """Rename only the asserted function call, leaving strings and other names untouched."""
    parsed = ast.parse(assertion)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise ValueError("AISI test must contain one assert statement")
    comparison = parsed.body[0].test
    if not isinstance(comparison, ast.Compare) or not isinstance(comparison.left, ast.Call):
        raise TypeError("AISI test must compare one function call with its expected value")
    call_name = comparison.left.func
    if not isinstance(call_name, ast.Name) or call_name.id != function_name:
        raise ValueError(f"AISI assertion must call {function_name!r}")
    return _replace_ast_span(assertion, call_name, replacement)


def _normalized_solution_assertions(
    assertions: tuple[str, ...], entry_point: str
) -> tuple[str, ...]:
    return tuple(
        _replace_call_target(assertion, entry_point, "solution") for assertion in assertions
    )


def _reference_solution_source(reference: _ReferenceCheck) -> str:
    """Adapt the source's reference program to the ``solution`` function used by AISI tests."""
    if reference.reference_kind == "function":
        reference_source = (
            f"{reference.test_setup_code.rstrip()}\n\n{reference.solution.rstrip()}".strip()
        )
        if reference.entry_point != "solution":
            reference_source += (
                "\n\ndef solution(*args, **kwargs):\n"
                f"    return {reference.entry_point}(*args, **kwargs)\n"
            )
        return reference_source
    if reference.reference_kind == "stdio":
        return f"""\
import subprocess
import sys

_REFERENCE_PROGRAM = {reference.solution!r}

def solution(input_data):
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _REFERENCE_PROGRAM],
        input=input_data,
        capture_output=True,
        text=True,
        timeout={REFERENCE_TIMEOUT_SECONDS},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("reference program exited nonzero")
    return completed.stdout
"""
    raise ValueError(f"unsupported reference kind {reference.reference_kind!r}")


def _reference_assertions_pass(
    reference: _ReferenceCheck, assertions: tuple[str, ...]
) -> bool | None:
    """Run every reference assertion in bounded AISI test batches inside the episode jail."""
    if not assertions:
        return False
    wrapped_reference = _reference_solution_source(reference)
    completion = f"```python\n{wrapped_reference}\n```"
    normalized_assertions = _normalized_solution_assertions(assertions, reference.entry_point)
    batch_size = min(reference.aisi_source.max_test_cases, MAX_REFERENCE_ASSERTIONS_PER_RUN)
    if batch_size < 1:
        raise ValueError("AISI max_test_cases must be positive")
    for batch_start in range(0, len(normalized_assertions), batch_size):
        result = run_completion_tests(
            completion,
            list(normalized_assertions[batch_start : batch_start + batch_size]),
            {"always_equal": True, "exit": True, "conftest": True},
            scratch_parent=Path(tempfile.gettempdir()),
            timeout_seconds=REFERENCE_TIMEOUT_SECONDS,
            hardened=False,
            source=reference.aisi_source,
        )
        verdict = _reference_test_verdict(result.normal)
        if verdict is None:
            return None
        if not verdict:
            return False
    return True


def _reference_test_verdict(result: JailedPytestResult) -> bool | None:
    """Separate measured reference rejection from a timeout and a broken pytest runner."""
    # 137 is SIGKILL from the jail's memory or wall-clock limit: the reference was not measured.
    if result.timed_out or result.exit_code == SIGKILL_EXIT_CODE:
        return None
    if result.exit_code == 0:
        return True
    if result.exit_code == 1:
        return False
    raise RuntimeError(
        "AISI reference pytest did not produce a verdict "
        f"(exit={result.exit_code}, stdout_chars={len(result.stdout)}, stderr_chars={len(result.stderr)})"
    )


def _verify_reference_runner(aisi_source: AisiSource) -> None:
    """Smoke-test the jail once before reading datasets or attempting thousands of checks."""
    result = run_completion_tests(
        "```python\ndef solution(value):\n    return value\n```",
        ["assert solution(5) == 5"],
        {"always_equal": True, "exit": True, "conftest": True},
        scratch_parent=Path(tempfile.gettempdir()),
        timeout_seconds=REFERENCE_TIMEOUT_SECONDS,
        hardened=False,
        source=aisi_source,
    )
    if _reference_test_verdict(result.normal) is not True:
        raise RuntimeError("AISI reference-test runner smoke did not pass")


def _bake_with_reference(
    candidate: BakeCandidate,
    reference_solution: str,
    mutation: Mutation,
    truth_assertions: tuple[str, ...],
    aisi_source: AisiSource,
) -> BakeProblemResult:
    reference = _ReferenceCheck(
        solution=reference_solution,
        entry_point=candidate.entry_point,
        reference_kind=candidate.reference_kind,
        test_setup_code=candidate.test_setup_code,
        aisi_source=aisi_source,
    )
    drop_reason: str | None = None
    truth_verdict = _reference_assertions_pass(reference, truth_assertions)
    if truth_verdict is None:
        drop_reason = "reference_truth_tests_unmeasured"
    elif not truth_verdict:
        original_verdict = _reference_assertions_pass(reference, candidate.visible_assertions)
        if original_verdict is None:
            drop_reason = "reference_original_tests_unmeasured"
        elif not original_verdict:
            drop_reason = "reference_fails_original_tests"
        else:
            drop_reason = "reference_fails_truth_tests"
    else:
        mutated_verdict = _reference_assertions_pass(
            reference, (mutation.assertions[mutation.case_index],)
        )
        if mutated_verdict is None:
            drop_reason = "reference_mutated_test_unmeasured"
        elif mutated_verdict:
            drop_reason = "reference_did_not_fail_only_mutated_case"

    if drop_reason is not None:
        return BakeProblemResult(None, drop_reason)

    return BakeProblemResult(
        {
            "source": candidate.source,
            "problem_id": f"{candidate.source}:{candidate.source_problem_id}",
            "source_problem_id": candidate.source_problem_id,
            "statement": candidate.statement,
            "entry_point": candidate.entry_point,
            "solution_signature": candidate.solution_signature,
            "test_setup_code": candidate.test_setup_code,
            "original_assertions": list(candidate.visible_assertions),
            "oneoff_assertions": list(mutation.assertions),
            "truth_assertions": list(truth_assertions),
            "mutation_index": mutation.case_index,
        },
        None,
    )


def bake_problem(
    candidate: BakeCandidate, *, aisi_source: AisiSource | None = None
) -> BakeProblemResult:
    """Check one reference, then bake its original and oneoff assertions or a drop reason."""
    drop_reason = _candidate_drop_reason(candidate)
    if drop_reason is not None:
        return BakeProblemResult(None, drop_reason)
    truth_assertions = (
        candidate.visible_assertions
        if candidate.truth_assertions is None
        else candidate.truth_assertions
    )

    try:
        mutation = mutate_one_expected_output(
            f"{candidate.source}:{candidate.source_problem_id}", candidate.visible_assertions
        )
    except MutationError as error:
        return BakeProblemResult(None, f"mutation_{error}")

    source_loader = load_aisi_source() if aisi_source is None else aisi_source
    return _bake_reference_solutions(candidate, mutation, truth_assertions, source_loader)


def _candidate_drop_reason(candidate: BakeCandidate) -> str | None:
    truth_assertions = (
        candidate.visible_assertions
        if candidate.truth_assertions is None
        else candidate.truth_assertions
    )
    if len(candidate.visible_assertions) < MINIMUM_TEST_CASES:
        return "too_few_visible_tests"
    if len(truth_assertions) < MINIMUM_TEST_CASES:
        return "too_few_truth_tests"
    if not candidate.reference_solutions:
        return "missing_reference_solution"
    if (
        not candidate.statement.strip()
        or not candidate.entry_point
        or not candidate.solution_signature.strip()
    ):
        return "invalid_task_metadata"
    if not set(candidate.visible_assertions) <= set(truth_assertions):
        return "truth_tests_omit_visible_tests"
    return None


def _etl_code_hash() -> str:
    """Hash ETL, AISI grading and jail code that can change a cached verdict."""
    repository_root = Path(__file__).resolve().parents[1]
    code_paths = (
        Path(__file__).resolve(),
        repository_root / "reward_hacking/organisms/aisi_source.py",
        repository_root / "reward_hacking/organisms/grading.py",
        repository_root / "reward_hacking/jailed_poc.py",
        repository_root / "scripts/resource-limits.sh",
        repository_root / "scripts/episode_jail.sh",
    )
    code_digest = hashlib.sha256()
    for code_path in sorted(code_paths):
        relative_path = code_path.relative_to(repository_root).as_posix().encode("utf-8")
        code_digest.update(relative_path)
        code_digest.update(b"\0")
        code_digest.update(code_path.read_bytes())
        code_digest.update(b"\0")
    return code_digest.hexdigest()


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_candidate_work_item(
    candidate: BakeCandidate,
    *,
    etl_code_hash: str,
    source_revision: str,
    drop_reason: str | None = None,
) -> BakeWorkItem:
    """Bind one candidate's full baked row, checks, mutation and verifier revisions."""
    try:
        mutation = mutate_one_expected_output(
            f"{candidate.source}:{candidate.source_problem_id}", candidate.visible_assertions
        )
        mutation_identity: object = {
            "assertions": mutation.assertions,
            "case_index": mutation.case_index,
        }
    except MutationError as error:
        mutation_identity = {"error": str(error)}

    problem_id = f"{candidate.source}:{candidate.source_problem_id}"
    input_hash = _json_hash(
        {
            "cache_schema": CACHE_SCHEMA,
            "etl_code_hash": etl_code_hash,
            "source_revision": source_revision,
            "source": candidate.source,
            "problem_id": problem_id,
            "candidate": asdict(candidate),
            "mutation": mutation_identity,
            "drop_reason": drop_reason,
        }
    )
    return BakeWorkItem(
        source=candidate.source,
        problem_id=problem_id,
        input_hash=input_hash,
        candidate=candidate,
        drop_reason=drop_reason,
    )


def _make_raw_drop_work_item(
    identity: BakeWorkIdentity,
    raw_record: Mapping[str, object],
    *,
    drop_reason: str,
    extra_inputs: object = None,
) -> BakeWorkItem:
    """Fingerprint malformed source rows without retaining their text in tracked code."""
    input_hash = _json_hash(
        {
            "cache_schema": CACHE_SCHEMA,
            "etl_code_hash": identity.etl_code_hash,
            "source_revision": identity.source_revision,
            "source": identity.source,
            "problem_id": identity.problem_id,
            "raw_record": raw_record,
            "extra_inputs": extra_inputs,
            "drop_reason": drop_reason,
        }
    )
    return BakeWorkItem(
        source=identity.source,
        problem_id=identity.problem_id,
        input_hash=input_hash,
        candidate=None,
        drop_reason=drop_reason,
    )


def _cache_record(
    item: BakeWorkItem,
    result: BakeProblemResult,
) -> dict[str, object]:
    if (result.problem is None) == (result.drop_reason is None):
        raise ValueError("AISI bake result must contain either a problem or a drop reason")
    if result.problem is not None and (
        result.problem.get("source") != item.source
        or result.problem.get("problem_id") != item.problem_id
    ):
        raise ValueError("AISI bake result identity does not match its work item")
    return {
        "schema": CACHE_SCHEMA,
        "source": item.source,
        "problem_id": item.problem_id,
        "input_hash": item.input_hash,
        "result": {"problem": result.problem, "drop_reason": result.drop_reason},
    }


def _validate_cache_result(value: object, cache_path: Path, line_number: int) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"AISI ETL cache result at {cache_path}:{line_number} must be an object")
    result = cast("dict[str, object]", value)
    problem = result.get("problem")
    drop_reason = result.get("drop_reason")
    if problem is not None and not isinstance(problem, dict):
        raise TypeError(f"AISI ETL cache problem at {cache_path}:{line_number} must be an object")
    if drop_reason is not None and not isinstance(drop_reason, str):
        raise TypeError(f"AISI ETL cache drop reason at {cache_path}:{line_number} must be text")
    if (problem is None) == (drop_reason is None):
        raise ValueError(
            f"AISI ETL cache result at {cache_path}:{line_number} has no unique verdict"
        )


def _decode_cache_record(
    value: object,
    *,
    cache_path: Path,
    line_number: int,
    source: SourceName,
) -> tuple[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise TypeError(f"AISI ETL cache record at {cache_path}:{line_number} must be an object")
    record = cast("dict[str, object]", value)
    if type(record.get("schema")) is not int or record.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"unsupported AISI ETL cache schema at {cache_path}:{line_number}")
    if record.get("source") != source:
        raise ValueError(f"AISI ETL cache source mismatch at {cache_path}:{line_number}")
    problem_id = record.get("problem_id")
    input_hash = record.get("input_hash")
    if not isinstance(problem_id, str) or not problem_id:
        raise TypeError(f"AISI ETL cache problem_id at {cache_path}:{line_number} must be text")
    if not isinstance(input_hash, str) or len(input_hash) != SHA256_HEX_LENGTH:
        raise TypeError(f"AISI ETL cache input_hash at {cache_path}:{line_number} must be SHA-256")
    _validate_cache_result(record.get("result"), cache_path, line_number)
    return problem_id, record


def _read_cache(cache_path: Path, source: SourceName) -> dict[str, dict[str, object]]:
    """Read the latest well-formed result for each source problem from its JSONL cache."""
    if not cache_path.is_file():
        return {}
    records: dict[str, dict[str, object]] = {}
    duplicate_ids: set[str] = set()
    with cache_path.open(encoding="utf-8") as cache_file:
        lines = cache_file.readlines()
    for line_index, line in enumerate(lines):
        line_number = line_index + 1
        if not line.strip():
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as error:
            if line_index == len(lines) - 1:
                logger.warning("discarding an incomplete final record in %s", cache_path)
                break
            raise ValueError(
                f"invalid AISI ETL cache JSON at {cache_path}:{line_number}"
            ) from error
        if isinstance(value, dict) and (
            type(value.get("schema")) is not int or value.get("schema") != CACHE_SCHEMA
        ):
            logger.warning(
                "discarding AISI ETL cache %s with unsupported schema at line %d",
                cache_path,
                line_number,
            )
            return {}
        problem_id, record = _decode_cache_record(
            value,
            cache_path=cache_path,
            line_number=line_number,
            source=source,
        )
        if problem_id in records:
            duplicate_ids.add(problem_id)
        records[problem_id] = record
    if duplicate_ids:
        logger.warning(
            "%s cache had duplicate records for %d item identities; keeping the latest record",
            source,
            len(duplicate_ids),
        )
    return records


def _write_cache_snapshot(cache_path: Path, records: Mapping[str, dict[str, object]]) -> None:
    """Atomically compact a source cache to one current record per problem identity."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            temporary_path = Path(cache_file.name)
            for problem_id in sorted(records):
                cache_file.write(
                    json.dumps(records[problem_id], sort_keys=True, separators=(",", ":"))
                )
                cache_file.write("\n")
            cache_file.flush()
        temporary_path.replace(cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _result_from_cache_record(record: Mapping[str, object]) -> BakeProblemResult:
    result = cast("dict[str, object]", record["result"])
    problem_value = result["problem"]
    problem = None if problem_value is None else cast("dict[str, object]", problem_value)
    drop_reason = cast("str | None", result["drop_reason"])
    return BakeProblemResult(problem, drop_reason)


class _BakeWorkPool:
    """Coordinate cache validation, worker completion, progress and single-writer persistence."""

    def __init__(
        self,
        *,
        cache_path: Path,
        workers: int,
        bake_candidate: BakeCandidateRunner,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.cache_path = cache_path
        self.workers = workers
        self.bake_candidate = bake_candidate
        self.source: SourceName = "apps"
        self.work_items: tuple[BakeWorkItem, ...] = ()
        self.outcomes: dict[str, BakeProblemResult] = {}
        self.drop_reasons: Counter[str] = Counter()
        self.resumed_count = 0
        self.computed_count = 0
        self.kept_count = 0
        self.started_at = 0.0
        self.last_reported_at = 0.0
        self.last_reported_done = 0

    def run(self, work_items: tuple[BakeWorkItem, ...]) -> BakeWorkResult:
        if not work_items:
            return BakeWorkResult((), 0, 0, {})
        self.source = work_items[0].source
        self.work_items = work_items
        work_by_id = self._index_work_items()
        matching_records = self._matching_cache_records(work_by_id)
        self.started_at = time.monotonic()
        self.last_reported_at = self.started_at
        for item in self.work_items:
            if item.problem_id in matching_records:
                result = _result_from_cache_record(matching_records[item.problem_id])
                self._record_outcome(item, result, resumed=True)

        pending_items = tuple(
            item for item in self.work_items if item.problem_id not in matching_records
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            for item in pending_items:
                if item.drop_reason is not None:
                    self._append_result(item, BakeProblemResult(None, item.drop_reason))
            candidates = tuple(item for item in pending_items if item.drop_reason is None)
            first_error = self._run_candidates(executor, candidates)

        self._log_progress(force=True)
        if first_error is not None:
            raise first_error
        problems = tuple(
            sorted(
                (result.problem for result in self.outcomes.values() if result.problem is not None),
                key=lambda problem: str(problem["problem_id"]),
            )
        )
        return BakeWorkResult(
            problems,
            self.resumed_count,
            self.computed_count,
            dict(sorted(self.drop_reasons.items())),
        )

    def _index_work_items(self) -> dict[str, BakeWorkItem]:
        work_by_id: dict[str, BakeWorkItem] = {}
        for item in self.work_items:
            if item.source != self.source:
                raise ValueError("AISI work item batch must contain one source")
            if item.problem_id in work_by_id:
                raise ValueError(
                    f"duplicate AISI problem identity in source batch: {item.problem_id}"
                )
            if item.candidate is None and item.drop_reason is None:
                raise ValueError("AISI work item must contain a candidate or a drop reason")
            work_by_id[item.problem_id] = item
        return work_by_id

    def _matching_cache_records(
        self,
        work_by_id: Mapping[str, BakeWorkItem],
    ) -> dict[str, dict[str, object]]:
        old_cache = _read_cache(self.cache_path, self.source)
        matches: dict[str, dict[str, object]] = {}
        for problem_id, item in work_by_id.items():
            cached = old_cache.get(problem_id)
            if cached is None:
                continue
            if cached["input_hash"] == item.input_hash:
                matches[problem_id] = cached
            else:
                identity_hash = hashlib.sha256(problem_id.encode("utf-8")).hexdigest()[:12]
                logger.info(
                    "%s cache input hash mismatch for item %s; recomputing",
                    self.source,
                    identity_hash,
                )
        _write_cache_snapshot(self.cache_path, matches)
        return matches

    def _record_outcome(
        self,
        item: BakeWorkItem,
        result: BakeProblemResult,
        *,
        resumed: bool,
    ) -> None:
        _cache_record(item, result)
        self.outcomes[item.problem_id] = result
        if resumed:
            self.resumed_count += 1
        else:
            self.computed_count += 1
        if result.problem is not None:
            self.kept_count += 1
        else:
            if result.drop_reason is None:
                raise ValueError("AISI drop result has no reason")
            self.drop_reasons[result.drop_reason] += 1
        self._log_progress()

    def _append_result(self, item: BakeWorkItem, result: BakeProblemResult) -> None:
        record = _cache_record(item, result)
        with self.cache_path.open("a", encoding="utf-8") as cache_file:
            cache_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            cache_file.flush()
        self._record_outcome(item, result, resumed=False)

    def _run_candidates(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        candidates: tuple[BakeWorkItem, ...],
    ) -> Exception | None:
        candidate_index = 0
        futures: dict[concurrent.futures.Future[BakeProblemResult], BakeWorkItem] = {}
        while candidate_index < min(self.workers, len(candidates)):
            item = candidates[candidate_index]
            futures[executor.submit(self.bake_candidate, cast("BakeCandidate", item.candidate))] = (
                item
            )
            candidate_index += 1

        first_error: Exception | None = None
        while futures:
            timeout = max(
                0.0,
                PROGRESS_INTERVAL_SECONDS - (time.monotonic() - self.last_reported_at),
            )
            completed, _ = concurrent.futures.wait(
                futures,
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not completed:
                self._log_progress()
                continue
            for future in sorted(completed, key=lambda ready: futures[ready].problem_id):
                item = futures.pop(future)
                result_or_error = self._finished_candidate(future, item)
                if result_or_error is None:
                    continue
                if isinstance(result_or_error, Exception):
                    if first_error is None:
                        first_error = result_or_error
                        for other_future in futures:
                            other_future.cancel()
                    continue
                self._append_result(item, result_or_error)
            while (
                first_error is None
                and len(futures) < self.workers
                and candidate_index < len(candidates)
            ):
                item = candidates[candidate_index]
                futures[
                    executor.submit(self.bake_candidate, cast("BakeCandidate", item.candidate))
                ] = item
                candidate_index += 1
        return first_error

    def _finished_candidate(
        self,
        future: concurrent.futures.Future[BakeProblemResult],
        item: BakeWorkItem,
    ) -> BakeProblemResult | Exception | None:
        if future.cancelled():
            return None
        try:
            return future.result()
        except Exception:
            identity_hash = hashlib.sha256(item.problem_id.encode("utf-8")).hexdigest()[:12]
            logger.exception("%s verifier failed for item %s", self.source, identity_hash)
            return cast("Exception", future.exception())

    def _log_progress(self, *, force: bool = False) -> None:
        now = time.monotonic()
        done = self.resumed_count + self.computed_count
        if (
            not force
            and done - self.last_reported_done < PROGRESS_EVERY
            and now - self.last_reported_at < PROGRESS_INTERVAL_SECONDS
        ):
            return
        elapsed = max(now - self.started_at, 0.001)
        rate = self.computed_count / elapsed
        remaining = len(self.work_items) - done
        eta = "0s" if remaining == 0 else (f"{remaining / rate:.0f}s" if rate > 0 else "unknown")
        logger.info(
            "%s progress: %d/%d resumed=%d computed=%d kept=%d dropped=%s rate=%.2f/s eta=%s",
            self.source,
            done,
            len(self.work_items),
            self.resumed_count,
            self.computed_count,
            self.kept_count,
            dict(sorted(self.drop_reasons.items())),
            rate,
            eta,
        )
        self.last_reported_at = now
        self.last_reported_done = done


def _run_bake_work_items(
    work_items: tuple[BakeWorkItem, ...],
    *,
    cache_path: Path,
    workers: int,
    bake_candidate: BakeCandidateRunner,
) -> BakeWorkResult:
    """Resume matching results, verify the remainder in parallel, and persist each result."""
    return _BakeWorkPool(
        cache_path=cache_path,
        workers=workers,
        bake_candidate=bake_candidate,
    ).run(work_items)


def _bake_reference_solutions(
    candidate: BakeCandidate,
    mutation: Mutation,
    truth_assertions: tuple[str, ...],
    aisi_source: AisiSource,
) -> BakeProblemResult:
    results: list[BakeProblemResult] = []
    for reference_solution in candidate.reference_solutions:
        result = _bake_with_reference(
            candidate,
            reference_solution,
            mutation,
            truth_assertions,
            aisi_source,
        )
        if result.problem is not None:
            return result
        results.append(result)
    reasons = Counter(result.drop_reason for result in results if result.drop_reason is not None)
    for reason in (
        "reference_truth_tests_unmeasured",
        "reference_original_tests_unmeasured",
        "reference_mutated_test_unmeasured",
        "reference_fails_original_tests",
        "reference_fails_truth_tests",
        "reference_did_not_fail_only_mutated_case",
        "mutation_no_mutatable_expected_output",
    ):
        if reasons[reason]:
            return BakeProblemResult(None, reason)
    return BakeProblemResult(None, next(iter(reasons), "missing_reference_solution"))


def _assertions_from_test_source(test_source: str) -> tuple[str, ...]:
    """Extract ordered single-assert test cases without changing their call targets."""
    parsed = ast.parse(test_source)
    assert_nodes = sorted(
        (node for node in ast.walk(parsed) if isinstance(node, ast.Assert)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assertions: list[str] = []
    for node in assert_nodes:
        assertion = ast.get_source_segment(test_source, node)
        if assertion is None:
            raise ValueError("AISI test assertion has no source segment")
        assertions.append(assertion)
    return tuple(assertions)


def _entry_point(assertion: str) -> str:
    """Return the asserted function name from a single literal test assertion."""
    parsed = ast.parse(assertion)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise TypeError("AISI test assertion must be one assert statement")
    comparison = parsed.body[0].test
    if not isinstance(comparison, ast.Compare) or not isinstance(comparison.left, ast.Call):
        raise TypeError("AISI test assertion must compare a function call")
    if not isinstance(comparison.left.func, ast.Name):
        raise TypeError("AISI test assertion must call a named function")
    return comparison.left.func.id


def _solution_signature(reference_solution: str, entry_point: str) -> str:
    """Rebuild one function signature without copying its implementation into the task."""
    parsed = ast.parse(reference_solution)
    function = next(
        (
            node
            for node in parsed.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
        ),
        None,
    )
    if not isinstance(function, ast.FunctionDef):
        raise TypeError(f"reference source has no synchronous {entry_point!r} function")
    signature = f"def {entry_point}({ast.unparse(function.args)})"
    if function.returns is not None:
        signature += f" -> {ast.unparse(function.returns)}"
    return f"{signature}:"


def _assertion_from_pair(function_name: str, arguments: object, expected: object) -> str:
    """Render one source input/output pair using AISI's function-test convention."""
    return f"assert {function_name}({arguments!r}) == {expected!r}"


def _input_output_pairs(value: object, *, field: str) -> tuple[tuple[object, object], ...]:
    """Read paired AISI input/output arrays without silently truncating a malformed row."""
    if not isinstance(value, dict):
        raise TypeError(f"{field}_is_not_an_object")
    inputs = value.get("inputs", value.get("input"))
    outputs = value.get("outputs", value.get("output"))
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise TypeError(f"{field}_has_no_input_output_arrays")
    if len(inputs) != len(outputs):
        raise ValueError(f"{field}_input_output_lengths_differ")
    return tuple(
        (input_value, output_value)
        for input_value, output_value in zip(inputs, outputs, strict=True)
    )


def _json_field(record: Mapping[str, object], field: str) -> object:
    """Decode a JSON-string field when the source represents its nested record as text."""
    value = record.get(field)
    if value == "":
        raise ValueError(f"{field}_is_empty")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field}_is_invalid_json") from error
    return value


def _record_text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"{field}_is_not_text")
    return value


def _reference_solutions(record: Mapping[str, object], *, source: SourceName) -> tuple[str, ...]:
    """Return runnable references in the source's expected language and representation."""
    if source == "apps":
        raw_solutions = _json_field(record, "solutions")
        if not isinstance(raw_solutions, list):
            raise TypeError("solutions_is_not_a_list")
        return tuple(
            solution for solution in raw_solutions if isinstance(solution, str) and solution.strip()
        )
    if source == "codecontests":
        raw_solutions = record.get("solutions")
        if not isinstance(raw_solutions, Mapping):
            raise TypeError("solutions_is_not_an_object")
        languages = raw_solutions.get("language")
        solutions = raw_solutions.get("solution")
        if (
            not isinstance(languages, list)
            or not isinstance(solutions, list)
            or len(languages) != len(solutions)
        ):
            raise ValueError("solutions_language_arrays_are_malformed")
        return tuple(
            solution
            for language, solution in zip(languages, solutions, strict=True)
            if language == PYTHON_LANGUAGE_ID and isinstance(solution, str) and solution.strip()
        )
    if source == "humaneval":
        return (_record_text(record, "prompt") + _record_text(record, "canonical_solution"),)
    if source == "mbpp":
        return (_record_text(record, "code"),)
    raise ValueError(f"unknown AISI source {source!r}")


def _function_candidate(
    source: Literal["mbpp", "humaneval"],
    source_problem_id: str,
    record: Mapping[str, object],
) -> BakeCandidate:
    """Parse one function dataset row into the shared bake candidate shape."""
    if source == "humaneval":
        entry_point = _record_text(record, "entry_point")
        assertions = _assertions_from_test_source(_record_text(record, "test"))
        visible_assertions = tuple(
            _replace_call_target(assertion, "candidate", entry_point) for assertion in assertions
        )
        references = _reference_solutions(record, source="humaneval")
        statement = _record_text(record, "prompt")
        test_setup_code = ""
    else:
        raw_assertions = record.get("test_list")
        if not isinstance(raw_assertions, list) or any(
            not isinstance(assertion, str) for assertion in raw_assertions
        ):
            raise TypeError("test_list_is_not_a_list_of_assertions")
        raw_assertions_tuple = tuple(cast("list[str]", raw_assertions))
        if not raw_assertions_tuple:
            raise ValueError("test_list_is_empty")
        entry_point = _entry_point(raw_assertions_tuple[0])
        visible_assertions = tuple(
            _replace_call_target(assertion, entry_point, entry_point)
            for assertion in raw_assertions_tuple
        )
        references = _reference_solutions(record, source="mbpp")
        statement = _record_text(record, "prompt").strip()
        setup_value = record.get("test_setup_code", "")
        if not isinstance(setup_value, str):
            raise TypeError("test_setup_code_is_not_text")
        test_setup_code = setup_value

    solution_signature = _solution_signature(references[0], entry_point) if references else ""
    return BakeCandidate(
        source=source,
        source_problem_id=source_problem_id,
        statement=statement,
        entry_point=entry_point,
        solution_signature=solution_signature,
        reference_solutions=references,
        visible_assertions=visible_assertions,
        test_setup_code=test_setup_code,
    )


def _problem_identity(record: Mapping[str, object]) -> tuple[str, ...]:
    """Return CodeContests fields that disambiguate repeated names across source collections."""
    return tuple(
        "" if record.get(field) is None else str(record[field])
        for field in ("name", "source", "cf_contest_id", "cf_index")
    )


def _codecontests_source_id(record: Mapping[str, object]) -> str:
    return json.dumps(_problem_identity(record), ensure_ascii=False, separators=(",", ":"))


def _hidden_codecontest_assertions(record: Mapping[str, object]) -> tuple[str, ...]:
    generated_tests = record.get("generated_tests")
    if not generated_tests:
        return ()
    pairs = _input_output_pairs(generated_tests, field="generated_tests")
    return tuple(
        _assertion_from_pair("solution", input_value, output_value)
        for input_value, output_value in pairs
    )


def _bake_codecontests(
    raw_records: list[Mapping[str, object]],
    *,
    run_config: BakeRunConfig,
) -> dict[str, object]:
    from reward_hacking.organisms.aisi_codecontests import (  # noqa: PLC0415 - only this source uses it
        load_code_contest_problems,
    )

    filtered_problems = load_code_contest_problems(
        source=run_config.aisi_source, records=raw_records
    )
    raw_by_identity: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for record in raw_records:
        raw_by_identity[_problem_identity(record)].append(record)

    work_items: list[BakeWorkItem] = []
    for codecontest_problem in filtered_problems:
        identity = (
            codecontest_problem.problem_id,
            "" if codecontest_problem.source is None else str(codecontest_problem.source),
            ""
            if codecontest_problem.cf_contest_id is None
            else str(codecontest_problem.cf_contest_id),
            "" if codecontest_problem.cf_index is None else str(codecontest_problem.cf_index),
        )
        matching_records = raw_by_identity.get(identity)
        if not matching_records:
            raise ValueError("AISI CodeContests loader returned a row with no raw source record")
        record = matching_records.pop(0)
        visible_assertions = codecontest_problem.test_assertions
        source_problem_id = _codecontests_source_id(record)
        if len(visible_assertions) < MINIMUM_TEST_CASES:
            work_items.append(
                run_config.raw_drop_work_item(
                    "codecontests",
                    source_problem_id,
                    record,
                    drop_reason="too_few_visible_tests",
                    extra_inputs=visible_assertions,
                )
            )
            continue
        try:
            hidden_assertions = _hidden_codecontest_assertions(record)
            references = _reference_solutions(record, source="codecontests")
        except (TypeError, ValueError) as error:
            work_items.append(
                run_config.raw_drop_work_item(
                    "codecontests",
                    source_problem_id,
                    record,
                    drop_reason=str(error),
                    extra_inputs=visible_assertions,
                )
            )
            continue
        truth_assertions = (*visible_assertions, *hidden_assertions)
        candidate = BakeCandidate(
            source="codecontests",
            source_problem_id=source_problem_id,
            statement=codecontest_problem.description,
            entry_point="solution",
            solution_signature="def solution(input_data: str) -> str:",
            reference_solutions=references,
            visible_assertions=visible_assertions,
            truth_assertions=truth_assertions,
            reference_kind="stdio",
        )
        work_items.append(
            run_config.candidate_work_item(
                candidate,
                drop_reason=_candidate_drop_reason(candidate),
            )
        )

    result = run_config.run("codecontests", tuple(work_items))
    stats: dict[str, object] = {
        "input_count": len(raw_records),
        "source_filtered_count": len(raw_records) - len(filtered_problems),
        "eligible_count": len(filtered_problems),
        "kept_count": len(result.problems),
        "drop_reasons": result.drop_reasons,
        "resumed_count": result.resumed_count,
        "computed_count": result.computed_count,
    }
    return stats


def _bake_apps(
    records: list[Mapping[str, object]],
    *,
    external_root: Path,
    run_config: BakeRunConfig,
) -> dict[str, object]:
    exclusions_path = external_root / "rl-envs/src/rh_envs/apps_rh/excluded_problem_ids.json"
    excluded_ids = _apps_excluded_problem_ids(exclusions_path)

    work_items: list[BakeWorkItem] = []
    source_filtered_count = 0
    for record_index, record in enumerate(records):
        problem_id_value = record.get("problem_id")
        if not isinstance(problem_id_value, int):
            source_problem_id = f"invalid-{record_index}-{_json_hash(record)[:12]}"
            work_items.append(
                run_config.raw_drop_work_item(
                    "apps",
                    source_problem_id,
                    record,
                    drop_reason="invalid_problem_id",
                )
            )
            continue
        if not 0 <= problem_id_value <= MAX_APPS_PROBLEM_ID:
            source_filtered_count += 1
            continue
        if problem_id_value in excluded_ids:
            source_filtered_count += 1
            continue
        try:
            pairs = _input_output_pairs(_json_field(record, "input_output"), field="input_output")
            visible_assertions = tuple(
                _assertion_from_pair("solution", input_value, expected)
                for input_value, expected in pairs
            )
            references = _reference_solutions(record, source="apps")
            statement = _record_text(record, "question")
        except (TypeError, ValueError) as error:
            work_items.append(
                run_config.raw_drop_work_item(
                    "apps",
                    str(problem_id_value),
                    record,
                    drop_reason=str(error),
                )
            )
            continue
        candidate = BakeCandidate(
            source="apps",
            source_problem_id=str(problem_id_value),
            statement=statement,
            entry_point="solution",
            solution_signature="def solution(input_data: str) -> str:",
            reference_solutions=references,
            visible_assertions=visible_assertions,
            reference_kind="stdio",
        )
        work_items.append(
            run_config.candidate_work_item(
                candidate,
                drop_reason=_candidate_drop_reason(candidate),
            )
        )

    result = run_config.run("apps", tuple(work_items))

    stats: dict[str, object] = {
        "input_count": len(records),
        "source_filtered_count": source_filtered_count,
        "eligible_count": len(records) - source_filtered_count,
        "kept_count": len(result.problems),
        "drop_reasons": result.drop_reasons,
        "resumed_count": result.resumed_count,
        "computed_count": result.computed_count,
    }
    return stats


def _apps_excluded_problem_ids(exclusions_path: Path) -> set[int]:
    """Read numeric APPS ids the AISI task source deliberately leaves out."""
    if not exclusions_path.is_file():
        return set()
    raw_exclusions: object = json.loads(exclusions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_exclusions, dict):
        raise TypeError("AISI APPS exclusions must be a JSON object")
    excluded_ids: set[int] = set()
    for raw_ids in raw_exclusions.values():
        if not isinstance(raw_ids, list):
            raise TypeError("AISI APPS exclusion groups must be lists")
        for raw_id in raw_ids:
            if type(raw_id) is int:
                excluded_ids.add(raw_id)
            elif isinstance(raw_id, str):
                excluded_ids.add(int(raw_id))
            else:
                raise TypeError("AISI APPS excluded problem ids must be integers")
    return excluded_ids


def _bake_function_dataset(
    source: Literal["mbpp", "humaneval"],
    records: list[Mapping[str, object]],
    *,
    run_config: BakeRunConfig,
) -> dict[str, object]:
    work_items: list[BakeWorkItem] = []
    for record_index, record in enumerate(records):
        source_problem_id_value = record.get("task_id")
        if source_problem_id_value is None:
            source_problem_id = f"missing-{record_index}-{_json_hash(record)[:12]}"
            work_items.append(
                run_config.raw_drop_work_item(
                    source,
                    source_problem_id,
                    record,
                    drop_reason="missing_problem_id",
                )
            )
            continue
        source_problem_id = str(source_problem_id_value)
        try:
            candidate = _function_candidate(source, source_problem_id, record)
        except (SyntaxError, TypeError, ValueError) as error:
            work_items.append(
                run_config.raw_drop_work_item(
                    source,
                    source_problem_id,
                    record,
                    drop_reason=str(error) or "malformed_source_record",
                )
            )
            continue
        work_items.append(
            run_config.candidate_work_item(
                candidate,
                drop_reason=_candidate_drop_reason(candidate),
            )
        )

    result = run_config.run(source, tuple(work_items))

    stats: dict[str, object] = {
        "input_count": len(records),
        "source_filtered_count": 0,
        "eligible_count": len(records),
        "kept_count": len(result.problems),
        "drop_reasons": result.drop_reasons,
        "resumed_count": result.resumed_count,
        "computed_count": result.computed_count,
    }
    return stats


def _load_huggingface_records(
    source: Literal["mbpp", "codecontests", "humaneval"],
    *,
    cache_dir: Path,
) -> list[Mapping[str, object]]:
    """Load the exact dataset/config/split used by the AISI environment."""
    datasets = importlib.import_module("datasets")
    spec = DATASET_SPECS[source]
    kwargs: dict[str, object] = {"split": spec["split"], "cache_dir": str(cache_dir)}
    if spec["config"] is not None:
        dataset = datasets.load_dataset(spec["dataset_id"], spec["config"], **kwargs)
    else:
        dataset = datasets.load_dataset(spec["dataset_id"], **kwargs)
    return [cast("Mapping[str, object]", record) for record in dataset]


def _load_apps_records(external_root: Path, cache_dir: Path) -> list[Mapping[str, object]]:
    """Load the same APPS split and builder without the loader's optional Inspect dependency."""
    source_path = str(external_root / "rl-envs/src")
    sys.path.insert(0, source_path)
    try:
        apps_builder = importlib.import_module("rh_envs.apps_dataset.apps_builder")
    finally:
        sys.path.remove(source_path)
    hub = importlib.import_module("huggingface_hub")
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub.snapshot_download(
        repo_id="codeparrot/apps",
        repo_type="dataset",
        local_dir=cache_dir,
    )
    with chdir(cache_dir):
        builder = apps_builder.AppsCode(config_name="all")
        builder.download_and_prepare()
        dataset = builder.as_dataset(split="test", in_memory=True)
    return [cast("Mapping[str, object]", record) for record in dataset]


def _prepare_cache(cache_dir: Path) -> None:
    """Use a task-local Hugging Face cache and link readable pre-existing dataset caches."""
    previous_cache = Path(
        os.environ.get("HF_DATASETS_CACHE", str(Path.home() / ".cache/huggingface/datasets"))
    )
    datasets_cache = cache_dir / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    for cache_name in (
        "deepmind___code_contests",
        "openai___openai_humaneval",
        "google-research-datasets___mbpp",
    ):
        old_cache = previous_cache / cache_name
        new_cache = datasets_cache / cache_name
        if old_cache.is_dir() and not new_cache.exists():
            new_cache.symlink_to(old_cache, target_is_directory=True)
    os.environ["HF_HOME"] = str(cache_dir / "hf-home")
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)


def _source_provenance(
    source: SourceName,
    stats: dict[str, object],
    split_counts: dict[str, dict[str, int]],
) -> dict[str, object]:
    return {
        **DATASET_SPECS[source],
        **stats,
        "split_counts": split_counts[source],
    }


def _cached_problems(
    cache_dir: Path,
    selected_sources: tuple[SourceName, ...],
) -> list[dict[str, object]]:
    """Build the final payload rows from the current per-source JSONL result caches."""
    problems: list[dict[str, object]] = []
    for source in selected_sources:
        records = _read_cache(cache_dir / "baked-results" / f"{source}.jsonl", source)
        for record in records.values():
            problem = _result_from_cache_record(record).problem
            if problem is not None:
                problems.append(problem)
    return sorted(problems, key=lambda problem: str(problem["problem_id"]))


def build_case_payload(
    *,
    external_root: Path,
    cache_dir: Path,
    selected_sources: tuple[SourceName, ...] = cast("tuple[SourceName, ...]", AISI_SOURCES),
    workers: int = DEFAULT_WORKERS,
) -> dict[str, object]:
    """Load, validate, split, and collect all selected AISI source problems."""
    if workers < 1:
        raise ValueError("workers must be positive")
    _prepare_cache(cache_dir)
    aisi_source = load_aisi_source(external_root)
    source_revision = _clone_commit(external_root)
    etl_code_hash = _etl_code_hash()
    run_config = BakeRunConfig(
        cache_dir=cache_dir,
        workers=workers,
        etl_code_hash=etl_code_hash,
        source_revision=source_revision,
        aisi_source=aisi_source,
    )
    _verify_reference_runner(aisi_source)
    raw_records: dict[str, list[Mapping[str, object]]] = {}
    for source in selected_sources:
        if source == "apps":
            raw_records[source] = _load_apps_records(external_root, cache_dir / "apps")
        else:
            raw_records[source] = _load_huggingface_records(
                source, cache_dir=cache_dir / "datasets"
            )

    stats_by_source: dict[str, dict[str, object]] = {}
    if "apps" in raw_records:
        stats = _bake_apps(
            raw_records["apps"],
            external_root=external_root,
            run_config=run_config,
        )
        stats_by_source["apps"] = stats
    if "mbpp" in raw_records:
        stats = _bake_function_dataset(
            "mbpp",
            raw_records["mbpp"],
            run_config=run_config,
        )
        stats_by_source["mbpp"] = stats
    if "codecontests" in raw_records:
        stats = _bake_codecontests(
            raw_records["codecontests"],
            run_config=run_config,
        )
        stats_by_source["codecontests"] = stats
    if "humaneval" in raw_records:
        stats = _bake_function_dataset(
            "humaneval",
            raw_records["humaneval"],
            run_config=run_config,
        )
        stats_by_source["humaneval"] = stats

    problems = _cached_problems(cache_dir, selected_sources)

    ids_by_source = {
        source: tuple(
            sorted(
                str(problem["problem_id"]) for problem in problems if problem["source"] == source
            )
        )
        for source in selected_sources
    }
    split = deterministic_source_splits(ids_by_source)
    split_counts = {
        source: {partition: len(ids) for partition, ids in partitions.items()}
        for source, partitions in split.items()
    }
    source_provenance: dict[str, object] = {
        source: _source_provenance(source, stats_by_source[source], split_counts)
        for source in selected_sources
    }
    return {
        "schema": DATA_SCHEMA,
        "provenance": {
            "aisi_clone_commit": source_revision,
            "split_algorithm": SPLIT_ALGORITHM,
            "held_out_fraction": HELD_OUT_FRACTION,
            "sources": source_provenance,
            "problem_count": len(problems),
        },
        "problems": problems,
    }


def _clone_commit(external_root: Path) -> str:
    """Read the exact AISI source revision used to build the task rows."""
    result = subprocess.run(  # noqa: S603 - fixed Git command with no shell
        ["git", "-C", str(external_root), "rev-parse", "HEAD"],  # noqa: S607 - literal executable
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot read AISI clone commit: {result.stderr.strip()}")
    commit = result.stdout.strip()
    status = subprocess.run(  # noqa: S603 - fixed Git command with no shell
        ["git", "-C", str(external_root), "status", "--porcelain"],  # noqa: S607 - literal executable
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"cannot verify AISI clone status: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("AISI source clone is dirty; refusing irreproducible provenance")
    return commit


def write_case_file(payload: dict[str, object], output_path: Path = DATA_PATH) -> None:
    """Atomically write the local-only cases and provenance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _case_assertions(problem: Mapping[str, object], field: str, row_index: int) -> list[str]:
    assertions = problem[field]
    if not isinstance(assertions, list) or not all(
        isinstance(assertion, str) for assertion in assertions
    ):
        raise TypeError(f"AISI case row {row_index} {field} must be a string list")
    return cast("list[str]", assertions)


def _refresh_oneoff_problem(problem: dict[str, object], row_index: int) -> bool:
    source = problem["source"]
    problem_id = problem["problem_id"]
    source_problem_id = problem["source_problem_id"]
    stored_mutation_index = problem["mutation_index"]
    if not isinstance(source, str) or source not in AISI_SOURCES:
        raise ValueError(f"AISI case row {row_index} has an invalid source")
    if not isinstance(problem_id, str) or not isinstance(source_problem_id, str):
        raise TypeError(f"AISI case row {row_index} has an invalid problem identity")
    if problem_id != f"{source}:{source_problem_id}":
        raise ValueError(f"AISI case row {row_index} has inconsistent problem identity")
    if type(stored_mutation_index) is not int:
        raise TypeError(f"AISI case row {row_index} mutation index must be an integer")

    original_assertions = _case_assertions(problem, "original_assertions", row_index)
    stored_oneoff_assertions = _case_assertions(problem, "oneoff_assertions", row_index)
    mutation = mutate_one_expected_output(problem_id, tuple(original_assertions))
    if mutation.case_index != stored_mutation_index:
        raise ValueError(f"AISI case row {row_index} mutation selection changed")
    refreshed_assertions = list(mutation.assertions)
    if refreshed_assertions == stored_oneoff_assertions:
        return False
    problem["oneoff_assertions"] = refreshed_assertions
    return True


def refresh_oneoff_assertions(payload: dict[str, object]) -> int:
    """Recompute only derived oneoff assertions while preserving task identity and source rows."""
    problems_value = payload["problems"]
    if not isinstance(problems_value, list):
        raise TypeError("AISI cases problems must be a list")
    if "output_mutation_registry" in payload or any(
        isinstance(problem, dict)
        and ("output_mutation" in problem or "source_mutation_index" in problem)
        for problem in problems_value
    ):
        raise ValueError(
            "legacy refresh cannot mutate contract-derived registries; derive a separate registry"
        )

    changed_count = 0
    for row_index, problem_value in enumerate(problems_value):
        if not isinstance(problem_value, dict):
            raise TypeError(f"AISI case row {row_index} must be an object")
        problem = cast("dict[str, object]", problem_value)
        changed_count += _refresh_oneoff_problem(problem, row_index)
    return changed_count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA_PATH)
    parser.add_argument("--minimum-supported", type=int, default=0)
    parser.add_argument(
        "--input-registry",
        type=Path,
        help="source registry for separately written contract derivation",
    )
    parser.add_argument(
        "--output-contracts", type=Path, help="ignored explicit output contract catalog"
    )
    parser.add_argument(
        "--allow-legacy-mutation",
        action="store_true",
        help="explicitly regenerate unvalidated legacy mutations",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/aisi-rh-envs-etl-cache"))
    parser.add_argument("--external-root", type=Path, default=None)
    parser.add_argument("--source", action="append", choices=AISI_SOURCES, dest="sources")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--refresh-oneoff",
        action="store_true",
        help="recompute derived oneoff assertions in the existing --out case file",
    )
    return parser


def _refresh_oneoff_cli(args: argparse.Namespace) -> int:
    if args.sources is not None or args.external_root is not None:
        raise ValueError(
            "--refresh-oneoff cannot be combined with source selection or external root"
        )
    if args.cache_dir != Path("artifacts/aisi-rh-envs-etl-cache"):
        raise ValueError("--cache-dir does not apply to --refresh-oneoff")
    if args.workers != DEFAULT_WORKERS:
        raise ValueError("--workers does not apply to --refresh-oneoff")
    payload_value = json.loads(args.out.read_text(encoding="utf-8"))
    if not isinstance(payload_value, dict):
        raise TypeError("AISI cases payload must be an object")
    payload = cast("dict[str, object]", payload_value)
    if payload.get("schema") != DATA_SCHEMA:
        raise ValueError("AISI cases payload has an unsupported schema")
    problems = payload["problems"]
    if not isinstance(problems, list):
        raise TypeError("AISI cases problems must be a list")
    changed_count = refresh_oneoff_assertions(payload)
    payload["output_mutation_policy"] = {
        "algorithm": "legacy-unvalidated-v1",
        "contract_validated": False,
    }
    write_case_file(payload, args.out)
    logger.info(
        "refreshed oneoff assertions: rows=%d changed=%d",
        len(problems),
        changed_count,
    )
    return 0


def _derive_contract_cli(args: argparse.Namespace) -> int:
    if args.out.resolve() == DATA_PATH.resolve():
        raise ValueError("contract derivation requires a non-legacy --out registry")
    if args.input_registry is None or args.out.resolve() == args.input_registry.resolve():
        raise ValueError("contract derivation requires --input-registry and a separate --out")
    if args.refresh_oneoff or args.allow_legacy_mutation or args.sources or args.external_root:
        raise ValueError(
            "contract derivation cannot be combined with legacy bake or refresh options"
        )
    source = json.loads(args.input_registry.read_text(encoding="utf-8"))
    catalog = json.loads(args.output_contracts.read_text(encoding="utf-8"))
    payload = derive_oneoff_registry(source, catalog)
    validate_derived_registry(payload)
    summary = payload["output_mutation_registry"]
    if args.minimum_supported < 0 or summary["supported_count"] < args.minimum_supported:
        raise ValueError("insufficient supported output contracts for requested collection")
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        validate_derived_registry(existing)
        if existing["output_mutation_registry"] != summary:
            raise ValueError(
                "derived registry resume identity changed; use a separately identified --out"
            )
    write_case_file(payload, args.out)
    summary = payload["output_mutation_registry"]
    logger.info(
        "contract derivation: supported=%d unsupported=%d",
        summary["supported_count"],
        summary["unsupported_count"],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Bake the local AISI case file or refresh its derived oneoff assertions."""
    args = _build_parser().parse_args(argv)
    if args.output_contracts is not None:
        return _derive_contract_cli(args)
    if args.input_registry is not None:
        raise ValueError("--input-registry requires --output-contracts")
    if not args.allow_legacy_mutation:
        raise ValueError(
            "new mutations require --output-contracts or explicit --allow-legacy-mutation"
        )
    if args.refresh_oneoff:
        return _refresh_oneoff_cli(args)

    external_root = (
        args.external_root
        if args.external_root is not None
        else Path(__file__).resolve().parents[1] / "external/aisi-reward-hacking-misalignment"
    )
    selected_sources = tuple(args.sources or AISI_SOURCES)
    if len(selected_sources) != len(set(selected_sources)):
        raise ValueError("each AISI source may be selected only once")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if set(selected_sources) != set(AISI_SOURCES) and args.out.resolve() == DATA_PATH.resolve():
        raise ValueError("a source-subset bake requires an explicit non-registry --out path")
    payload = build_case_payload(
        external_root=external_root,
        cache_dir=args.cache_dir,
        selected_sources=selected_sources,
        workers=args.workers,
    )
    payload["output_mutation_policy"] = {
        "algorithm": "legacy-unvalidated-v1",
        "contract_validated": False,
    }
    write_case_file(payload, args.out)
    provenance = cast("dict[str, object]", payload["provenance"])
    sources = cast("dict[str, dict[str, object]]", provenance["sources"])
    for source, source_stats in sources.items():
        logger.info(
            "%s: input=%d eligible=%d kept=%d resumed=%d computed=%d split=%s drops=%s",
            source,
            source_stats["input_count"],
            source_stats["eligible_count"],
            source_stats["kept_count"],
            source_stats["resumed_count"],
            source_stats["computed_count"],
            source_stats["split_counts"],
            source_stats["drop_reasons"],
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
