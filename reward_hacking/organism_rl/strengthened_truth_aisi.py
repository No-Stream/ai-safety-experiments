"""Build ignored strengthened-truth cases for the AISI tasks used by seed v4."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import random
import re
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from reward_hacking import aisi_rh_envs_etl
from reward_hacking.aisi_rh_envs_data import AisiRhProblem, load_problems
from reward_hacking.aisi_rh_envs_etl import _ReferenceCheck  # pyright: ignore[reportPrivateUsage]
from reward_hacking.harness import evalplus_etl
from reward_hacking.harness.loop import assert_disposable_episode_dir
from reward_hacking.jailed_poc import CommandResult, jail_python, run_in_jail
from reward_hacking.organisms.aisi_source import load_aisi_source

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from reward_hacking.organisms.aisi_source import AisiSource

logger = logging.getLogger(__name__)

SUPPORTED_AISI_SOURCES = frozenset({"apps", "humaneval", "mbpp"})
_SCENARIO_HASH = re.compile(r"-[0-9a-f]{10}$")
_EVALPLUS_SPECIAL_ORACLES: dict[str, dict[str, frozenset[str]]] = {
    "humaneval": {"polynomial_roots": frozenset({"find_zero"})},
    "mbpp": {
        "set_equality": frozenset(
            {
                "similar_elements",
                "find_char_long",
                "common_in_nested_lists",
                "extract_singly",
                "larg_nnum",
                "intersection_array",
                "find_dissimilar",
                "Diff",
            }
        ),
        "not_none": frozenset({"check_str", "text_match_three", "text_starta_endb"}),
        "custom_output": frozenset(
            {"are_equivalent", "sum_div", "surface_Area", "digit_distance_nums"}
        ),
    },
}
_CASES_PER_PLUS_ROW = 2
_REQUIRED_REFERENCE_COUNT = 2
_MAX_CASE_PAYLOAD_BYTES = 16 * 1024 * 1024
_INTEGER_STRING_LIMIT_DIAGNOSTIC = "Exceeds the limit (4300 digits) for integer string conversion"
_APPS_GENERATOR_ROOT = Path("docs/scratch/w16-strengthened-truth/gens-apps")
_GENERATED_APPS_INPUT_COUNT = 200
_GENERATED_APPS_INPUT_SEED = 20261008
_APPS_REFERENCE_BATCH_SIZE = 10
type _AppsCandidate = tuple[list[object], object, str, str, str]


class _AppsInputGenerator(Protocol):
    @property
    def sha256(self) -> str: ...

    def generate(self, rng: random.Random, n: int) -> list[list[object]]: ...

    def is_valid(self, arguments: list[object]) -> bool: ...


@dataclass(frozen=True)
class _LoadedAppsInputGenerator:
    _generate: Callable[[random.Random, int], list[list[object]]]
    _is_valid: Callable[[list[object]], bool]
    _sha256: str

    @property
    def sha256(self) -> str:
        return self._sha256

    def generate(self, rng: random.Random, n: int) -> list[list[object]]:
        return self._generate(rng, n)

    def is_valid(self, arguments: list[object]) -> bool:
        return self._is_valid(arguments)


@dataclass(frozen=True)
class AppsGeneratedInputConfig:
    """Reproducible APPS generated-input settings for one case build."""

    generator: _AppsInputGenerator
    count: int = _GENERATED_APPS_INPUT_COUNT
    seed: int = _GENERATED_APPS_INPUT_SEED


@dataclass(frozen=True)
class _AppsGeneratedCaseRequest:
    task: AisiRhProblem
    generated_inputs: AppsGeneratedInputConfig
    visible_cases: list[tuple[list[object], object]]
    existing_candidates: list[_AppsCandidate]
    references: tuple[_ReferenceCheck, ...]
    artifacts_root: Path


@dataclass(frozen=True)
class AisiCaseBuildConfig:
    """Inputs for producing all selected AISI strengthened-truth files."""

    records_path: Path
    selection_path: Path
    cases_root: Path
    evalplus_cache_dir: Path
    apps_cache_dir: Path
    external_root: Path
    aisi_cases_path: Path
    max_plus_cases: int = evalplus_etl.DEFAULT_MAX_PLUS_CASES
    sources_to_build: tuple[str, ...] = ("evalplus", "apps")
    max_case_payload_bytes: int = _MAX_CASE_PAYLOAD_BYTES
    apps_generators_dir: Path = _APPS_GENERATOR_ROOT
    generated_apps_input_count: int = _GENERATED_APPS_INPUT_COUNT
    generated_apps_input_seed: int = _GENERATED_APPS_INPUT_SEED


@dataclass
class _AisiSourceBuildProgress:
    config: AisiCaseBuildConfig
    tasks: tuple[AisiRhProblem, ...]
    evalplus_files: list[tuple[Path, dict[str, object]]]
    skipped: dict[str, list[dict[str, str]]]
    apps_files: list[tuple[Path, dict[str, object]]] = field(default_factory=list)

    def write_coverage(self, status: str) -> None:
        _write_source_coverage(
            self.config,
            self.tasks,
            {"evalplus": self.evalplus_files, "apps": self.apps_files},
            self.skipped,
            status=status,
        )


class CasePayloadTooLargeError(ValueError):
    """One item exceeds the configured bound for additional case representations."""

    def __init__(self, observed_bytes: int, limit_bytes: int) -> None:
        """Keep the observed payload and configured threshold for a counted skip record."""
        self.observed_bytes = observed_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"source case representations use {observed_bytes} bytes, over the {limit_bytes}-byte limit"
        )


class AisiSourceUnavailableError(ValueError):
    """One selected AISI source lacks enough measured inputs for strengthened truth."""

    def __init__(self, task_id: str, reason: str) -> None:
        """Retain a stable reason for the per-task coverage record."""
        self.reason = reason
        super().__init__(f"{task_id}: {reason}")


def canonical_digest(value: object) -> str:
    """Hash the canonical UTF-8 JSON representation used by the case interchange."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evalplus_special_oracle_reason(source: str, entry_point: str) -> str | None:
    """Return the official EvalPlus non-equality oracle category for an entry point."""
    for oracle_name, entry_points in _EVALPLUS_SPECIAL_ORACLES.get(source, {}).items():
        if entry_point in entry_points:
            return oracle_name
    return None


def _write_reference_artifact(
    artifacts_root: Path,
    task: AisiRhProblem,
    source: str,
) -> dict[str, str]:
    if not source.strip():
        raise ValueError(f"{task.problem_id} has an empty adapted reference")
    normalized_source = source.rstrip() + "\n"
    source_digest = hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()
    relative_path = Path("correct-solutions") / task.task_id("original") / f"{source_digest}.py"
    artifact_path = artifacts_root / relative_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_suffix(".py.tmp")
    temporary_path.write_text(normalized_source, encoding="utf-8")
    temporary_path.replace(artifact_path)
    return {"path": relative_path.as_posix(), "sha256": source_digest}


def _problem_ids_from_records(records_path: Path) -> set[str]:
    problem_ids: set[str] = set()
    with records_path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if not line.strip():
                continue
            row_value: object = json.loads(line)
            if not isinstance(row_value, dict):
                raise TypeError(f"{records_path}:{line_number} must contain a JSON object")
            if (
                row_value.get("task_kind") != "solvable"
                or row_value.get("variant") != "original"
                or not str(row_value.get("task_id", "")).startswith("aisi-")
            ):
                continue
            reward_value = row_value.get("organism_reward")
            if not isinstance(reward_value, dict) or not isinstance(
                reward_value.get("evidence"), dict
            ):
                raise TypeError(f"{records_path}:{line_number} has no AISI problem identity")
            problem_id = reward_value["evidence"].get("problem_id")
            if not isinstance(problem_id, str) or ":" not in problem_id:
                raise ValueError(f"{records_path}:{line_number} has an invalid AISI problem id")
            problem_ids.add(problem_id)
    return problem_ids


def _problem_ids_from_selection(selection_path: Path) -> set[str]:
    problem_ids: set[str] = set()
    selection_value: object = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection_value, dict) or not isinstance(
        selection_value.get("selected"), list
    ):
        raise TypeError(f"{selection_path} must contain a selected list")
    for row_index, row_value in enumerate(selection_value["selected"]):
        if not isinstance(row_value, dict):
            raise TypeError(f"{selection_path}: selected row {row_index} must be an object")
        source = row_value.get("source")
        if row_value.get("family") != "aisi" or source not in SUPPORTED_AISI_SOURCES:
            continue
        episode_key = row_value.get("episode_key")
        if not isinstance(episode_key, dict) or not isinstance(episode_key.get("scenario_id"), str):
            raise TypeError(f"{selection_path}: selected row {row_index} has no scenario id")
        problem_ids.add(
            _problem_id_from_scenario_id(cast("str", source), episode_key["scenario_id"])
        )
    return problem_ids


def selected_aisi_problem_ids(records_path: Path, selection_path: Path) -> set[str]:
    """Collect AISI originals from baseline records and AISI rows reused from seed v3."""
    return _problem_ids_from_records(records_path) | _problem_ids_from_selection(selection_path)


def _problem_id_from_scenario_id(source: str, scenario_id: str) -> str:
    prefix = f"aisi-{source}-"
    if not scenario_id.startswith(prefix):
        raise ValueError(f"AISI scenario id does not start with {prefix!r}")
    remainder = scenario_id.removeprefix(prefix)
    variant_prefix = next(
        (
            variant + "-"
            for variant in ("oneoff", "original")
            if remainder.startswith(variant + "-")
        ),
        None,
    )
    if variant_prefix is None:
        raise ValueError("AISI scenario id has no recognized task variant")
    source_problem_id = _SCENARIO_HASH.sub("", remainder.removeprefix(variant_prefix))
    if not source_problem_id:
        raise ValueError("AISI scenario id has an empty source problem id")
    if source == "humaneval" and source_problem_id.startswith("HumanEval-"):
        source_problem_id = source_problem_id.replace("HumanEval-", "HumanEval/", 1)
    return f"{source}:{source_problem_id}"


def _literal(value: object, label: str) -> object:
    if not isinstance(value, str):
        raise TypeError(f"EvalPlus {label} must be a repr string")
    try:
        return cast("object", ast.literal_eval(value))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise ValueError(f"EvalPlus {label} is not a Python literal") from error


def _json_round_trip(value: object, label: str) -> object:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return cast("object", json.loads(encoded))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not representable as JSON") from error


def _assertion_case(assertion: str, entry_point: str) -> tuple[list[object], object]:
    parsed = ast.parse(assertion)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise ValueError("AISI assertion must be one assert statement")
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
        raise ValueError(f"AISI assertion must compare one {entry_point} call with its output")
    if comparison.left.keywords:
        raise ValueError("AISI assertion calls must use positional arguments")
    try:
        arguments = cast(
            "list[object]", [ast.literal_eval(argument) for argument in comparison.left.args]
        )
        expected = cast("object", ast.literal_eval(comparison.comparators[0]))
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise ValueError("AISI assertions must use literal arguments and outputs") from error
    return arguments, expected


def _solution_assertion(arguments_repr: str, expected_repr: str) -> str:
    return f"assert solution(*{arguments_repr}) == {expected_repr}"


def _evalplus_case_rows(
    task: AisiRhProblem,
    plus_cases: list[object],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    visible_digests = {
        canonical_digest(_json_round_trip(arguments, "visible AISI arguments"))
        for assertion in task.original_assertions
        for arguments, _ in [_assertion_case(assertion, task.entry_point)]
    }
    cases_by_digest: dict[str, dict[str, object]] = {}
    assertion_by_digest: dict[str, str] = {}
    expected_repr_by_digest: dict[str, str] = {}
    for case_index, pair_value in enumerate(plus_cases):
        if not isinstance(pair_value, list) or len(pair_value) != _CASES_PER_PLUS_ROW:
            raise TypeError(f"EvalPlus plus case {case_index} must be a two-element list")
        arguments_source, expected_source = pair_value
        arguments_value = _literal(arguments_source, "arguments")
        expected_value = _literal(expected_source, "expected output")
        if not isinstance(arguments_value, (list, tuple)):
            raise TypeError(f"EvalPlus plus case {case_index} arguments must be positional")
        arguments_json = _json_round_trip(list(arguments_value), "EvalPlus arguments")
        if not isinstance(arguments_json, list) or arguments_json != list(arguments_value):
            raise ValueError(f"EvalPlus plus case {case_index} arguments are not JSON-preserving")
        case_digest = canonical_digest(arguments_json)
        if case_digest in visible_digests:
            continue
        expected_json = _json_round_trip(expected_value, "EvalPlus expected output")
        if not isinstance(arguments_source, str) or not isinstance(expected_source, str):
            raise TypeError(f"EvalPlus plus case {case_index} repr fields must be text")
        if case_digest in cases_by_digest:
            if expected_repr_by_digest[case_digest] != expected_source:
                raise ValueError("duplicate EvalPlus input has conflicting reference outputs")
            continue
        cases_by_digest[case_digest] = {
            "args": arguments_json,
            "expected": expected_json,
            "case_digest": case_digest,
        }
        assertion_by_digest[case_digest] = _solution_assertion(arguments_source, expected_source)
        expected_repr_by_digest[case_digest] = expected_source
    ordered_digests = tuple(cases_by_digest)
    return (
        [cases_by_digest[digest] for digest in ordered_digests],
        [assertion_by_digest[digest] for digest in ordered_digests],
        [expected_repr_by_digest[digest] for digest in ordered_digests],
    )


def _evalplus_repr_payload_bytes(plus_cases: list[object], limit_bytes: int) -> int:
    if limit_bytes < 1:
        raise ValueError("maximum case payload bytes must be positive")
    payload_bytes = 0
    for case_index, pair_value in enumerate(plus_cases):
        if not isinstance(pair_value, list) or len(pair_value) != _CASES_PER_PLUS_ROW:
            raise TypeError(f"EvalPlus plus case {case_index} must be a two-element list")
        arguments_source, expected_source = pair_value
        if not isinstance(arguments_source, str) or not isinstance(expected_source, str):
            raise TypeError(f"EvalPlus plus case {case_index} repr fields must be text")
        payload_bytes += len(arguments_source.encode("utf-8")) + len(
            expected_source.encode("utf-8")
        )
    if payload_bytes > limit_bytes:
        raise CasePayloadTooLargeError(payload_bytes, limit_bytes)
    return payload_bytes


def _validate_evalplus_identity(
    task: AisiRhProblem,
    baked_record: Mapping[str, object],
) -> tuple[str, str]:
    source_task_id = baked_record.get("task_id")
    if not isinstance(source_task_id, str):
        raise TypeError("baked EvalPlus task_id must be text")
    expected_source_task_id = (
        task.source_problem_id if task.source == "humaneval" else f"Mbpp/{task.source_problem_id}"
    )
    if source_task_id != expected_source_task_id:
        raise ValueError(
            f"EvalPlus id {source_task_id!r} does not match AISI problem {task.problem_id!r}"
        )
    entry_point = baked_record.get("entry_point")
    if not isinstance(entry_point, str):
        raise TypeError("baked EvalPlus entry_point must be text")
    special_oracle = evalplus_special_oracle_reason(task.source, entry_point)
    if special_oracle is not None:
        raise ValueError(f"EvalPlus task requires the unsupported {special_oracle} oracle")
    return source_task_id, entry_point


def _evalplus_reference_fields(
    baked_record: Mapping[str, object],
) -> tuple[str, float, float]:
    reference_seconds = baked_record.get("seconds")
    atol = baked_record.get("atol")
    canonical_program = baked_record.get("canonical_program")
    if not isinstance(reference_seconds, (int, float)) or reference_seconds < 0:
        raise ValueError("baked EvalPlus seconds must be non-negative")
    if not isinstance(atol, (int, float)) or atol < 0:
        raise ValueError("baked EvalPlus atol must be non-negative")
    if not isinstance(canonical_program, str) or not canonical_program.strip():
        raise ValueError("baked EvalPlus canonical reference must be non-empty")
    return canonical_program, float(reference_seconds), float(atol)


def _adapt_evalplus_reference(
    task: AisiRhProblem,
    entry_point: str,
    canonical_program: str,
) -> str:
    adapted_reference = canonical_program.rstrip()
    if entry_point != "solution":
        adapted_reference += (
            f"\n\ndef solution(*args, **kwargs):\n    return {entry_point}(*args, **kwargs)\n"
        )
    if task.test_setup_code:
        adapted_reference = f"{task.test_setup_code.rstrip()}\n\n{adapted_reference}"
    return adapted_reference


def build_evalplus_case_record(
    task: AisiRhProblem,
    baked_record: Mapping[str, object],
    artifacts_root: Path,
    *,
    max_case_payload_bytes: int = _MAX_CASE_PAYLOAD_BYTES,
) -> dict[str, object]:
    """Convert a validated EvalPlus bake into the shared case-file interchange format."""
    source_task_id, upstream_entry_point = _validate_evalplus_identity(task, baked_record)
    plus_cases = baked_record.get("plus_cases")
    if not isinstance(plus_cases, list):
        raise TypeError("baked EvalPlus plus_cases must be a list")
    repr_payload_bytes = _evalplus_repr_payload_bytes(plus_cases, max_case_payload_bytes)
    cases, assertions, expected_reprs = _evalplus_case_rows(task, plus_cases)
    if not cases:
        raise ValueError(
            f"{task.problem_id} has no novel EvalPlus inputs after visible-case filtering"
        )

    canonical_program, reference_seconds, atol = _evalplus_reference_fields(baked_record)
    adapted_reference = _adapt_evalplus_reference(task, upstream_entry_point, canonical_program)
    reference_artifact = _write_reference_artifact(artifacts_root, task, adapted_reference)
    return {
        "task_id": task.task_id("original"),
        "source": "evalplus",
        "cases": cases,
        "provenance": {
            "problem_id": task.problem_id,
            "source_task_id": source_task_id,
            "entry_point": task.entry_point,
            "upstream_entry_point": upstream_entry_point,
            "reference_sha256": hashlib.sha256(canonical_program.encode("utf-8")).hexdigest(),
            "reference_artifacts": [reference_artifact],
            "reference_seconds": reference_seconds,
            "reference_seconds_basis": "whole canonical-plus suite elapsed; conservative timeout bound",
            "repr_payload_bytes": repr_payload_bytes,
            "repr_payload_limit_bytes": max_case_payload_bytes,
            "atol": atol,
            "comparator": "AISI jailed pytest exact Python equality",
            "evalplus_oracle_registry": "official EvalPlus v0.3.1 special-oracle entry points",
            "assertions": assertions,
            "original_assertions": list(task.truth_assertions),
            "expected_repr": expected_reprs,
        },
    }


def _unique_app_references(
    task: AisiRhProblem,
    source_record: Mapping[str, object],
    aisi_source: AisiSource,
) -> tuple[tuple[_ReferenceCheck, ...], float]:
    references: list[_ReferenceCheck] = []
    reference_hashes: set[str] = set()
    maximum_elapsed = 0.0
    # The ETL's source adapter owns the APPS record schema and stdio conversion.
    for source_code in aisi_rh_envs_etl._reference_solutions(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        source_record, source="apps"
    ):
        reference_digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
        if reference_digest in reference_hashes:
            continue
        reference_hashes.add(reference_digest)
        reference = _ReferenceCheck(
            solution=source_code,
            entry_point=task.entry_point,
            reference_kind="stdio",
            test_setup_code=task.test_setup_code,
            aisi_source=aisi_source,
        )
        started_at = time.monotonic()
        verdict = aisi_rh_envs_etl._reference_assertions_pass(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            reference, task.original_assertions
        )
        maximum_elapsed = max(maximum_elapsed, time.monotonic() - started_at)
        if verdict is None:
            logger.info("AISI APPS reference was unmeasured for %s", task.problem_id)
            continue
        if verdict:
            references.append(reference)
            if len(references) == _REQUIRED_REFERENCE_COUNT:
                return tuple(references), maximum_elapsed
    raise AisiSourceUnavailableError(
        task.problem_id,
        "fewer than two distinct upstream references pass the original tests",
    )


def load_apps_input_generator(
    generators_dir: Path, task: AisiRhProblem
) -> _LoadedAppsInputGenerator | None:
    """Load an ignored APPS generator by source problem id and bind its file digest."""
    generator_path = generators_dir / f"{task.source_problem_id}.py"
    if not generator_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"strengthened_truth_apps_generator_{task.source_problem_id}", generator_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load APPS input generator at {generator_path}")
    generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator_module)
    generate_function = generator_module.__dict__["generate"]
    valid_function = generator_module.__dict__["is_valid"]
    if not callable(generate_function) or not callable(valid_function):
        raise TypeError(f"APPS input generator at {generator_path} must define callable functions")
    return _LoadedAppsInputGenerator(
        _generate=cast("Callable[[random.Random, int], list[list[object]]]", generate_function),
        _is_valid=cast("Callable[[list[object]], bool]", valid_function),
        _sha256=hashlib.sha256(generator_path.read_bytes()).hexdigest(),
    )


def _validated_generated_apps_arguments(
    task: AisiRhProblem,
    generator: _AppsInputGenerator,
    arguments_value: object,
    input_index: int,
) -> list[object]:
    if not isinstance(arguments_value, list) or len(arguments_value) != 1:
        raise TypeError(f"generated APPS input {input_index} must contain one stdin string")
    arguments = cast("list[object]", arguments_value)
    if not isinstance(arguments[0], str):
        raise TypeError(f"generated APPS input {input_index} must contain one stdin string")
    if not generator.is_valid(arguments):
        raise ValueError(f"APPS input generator returned an invalid input for {task.problem_id}")
    normalized_arguments = _json_round_trip(arguments, "generated APPS arguments")
    if not isinstance(normalized_arguments, list) or normalized_arguments != arguments:
        raise ValueError(f"generated APPS input {input_index} is not JSON-preserving")
    return cast("list[object]", normalized_arguments)


def _generated_apps_inputs(request: _AppsGeneratedCaseRequest) -> list[list[object]]:
    task = request.task
    generated_input_config = request.generated_inputs
    if generated_input_config.count < 1:
        raise ValueError("generated APPS input count must be positive")
    for arguments, _ in request.visible_cases:
        if not generated_input_config.generator.is_valid(arguments):
            raise ValueError(
                f"APPS input generator rejects a visible APPS input for {task.problem_id}"
            )

    generated_values = generated_input_config.generator.generate(
        random.Random(generated_input_config.seed), generated_input_config.count
    )
    if len(generated_values) != generated_input_config.count:
        raise ValueError(
            "APPS input generator must return exactly "
            f"{generated_input_config.count} inputs for {task.problem_id}"
        )

    seen_digests = {canonical_digest(arguments) for arguments, _ in request.visible_cases}
    seen_digests.update(canonical_digest(candidate[0]) for candidate in request.existing_candidates)
    novel_inputs: list[list[object]] = []
    for input_index, arguments_value in enumerate(generated_values):
        arguments = _validated_generated_apps_arguments(
            task, generated_input_config.generator, arguments_value, input_index
        )
        input_digest = canonical_digest(arguments)
        if input_digest in seen_digests:
            continue
        seen_digests.add(input_digest)
        novel_inputs.append(arguments)
    return novel_inputs


def _apps_reference_probe_source(reference: _ReferenceCheck, inputs: list[str]) -> str:
    adapted_reference = aisi_rh_envs_etl._reference_solution_source(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        reference
    )
    return (
        f"{adapted_reference}\n"
        "import json as _strengthened_json\n"
        "import time as _strengthened_time\n"
        f"_STRENGTHENED_INPUTS = {inputs!r}\n"
        "for _strengthened_input in _STRENGTHENED_INPUTS:\n"
        "    _strengthened_started = _strengthened_time.monotonic()\n"
        "    try:\n"
        "        _strengthened_output = solution(_strengthened_input)\n"
        "    except (subprocess.TimeoutExpired, RuntimeError):\n"
        '        _strengthened_row = {"status": "unavailable"}\n'
        "    else:\n"
        '        _strengthened_row = {"status": "ok", "output": _strengthened_output, '
        '"seconds": _strengthened_time.monotonic() - _strengthened_started}\n'
        "    sys.stdout.write(_strengthened_json.dumps(_strengthened_row, ensure_ascii=True, "
        "allow_nan=False) + '\\n')\n"
    )


def _parse_apps_reference_batch(
    result: CommandResult, expected_count: int
) -> tuple[list[str | None], float]:
    if result.timed_out:
        return [None] * expected_count, 0.0
    if result.exit_code != 0:
        stderr_excerpt = result.stderr[:500]
        raise RuntimeError(
            "APPS generated-input reference runner failed "
            f"(exit={result.exit_code}, stdout_chars={len(result.stdout)}, "
            f"stderr_chars={len(result.stderr)}, stderr={stderr_excerpt!r})"
        )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != expected_count:
        raise RuntimeError("APPS generated-input reference runner returned the wrong row count")
    outputs: list[str | None] = []
    maximum_case_seconds = 0.0
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("APPS generated-input reference row must be an object")
        status = row.get("status")
        if status == "unavailable":
            outputs.append(None)
            continue
        if status != "ok":
            raise ValueError("APPS generated-input reference row has an invalid status")
        output = row.get("output")
        elapsed_seconds = row.get("seconds")
        if not isinstance(output, str):
            raise TypeError("APPS generated-input reference output must be text")
        if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds < 0:
            raise TypeError("APPS generated-input reference time must be nonnegative")
        maximum_case_seconds = max(maximum_case_seconds, float(elapsed_seconds))
        outputs.append(output)
    return outputs, maximum_case_seconds


APPS_PROBE_ROOT = Path("/var/tmp/w16-apps-reference-probes")  # noqa: S108 - the jail refuses a home-tree dir


def _run_apps_reference_outputs(
    reference: _ReferenceCheck,
    inputs: list[str],
    artifacts_root: Path,
) -> tuple[list[str | None], float]:
    outputs: list[str | None] = []
    maximum_case_seconds = 0.0
    artifacts_root.mkdir(parents=True, exist_ok=True)
    APPS_PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    for batch_start in range(0, len(inputs), _APPS_REFERENCE_BATCH_SIZE):
        input_batch = inputs[batch_start : batch_start + _APPS_REFERENCE_BATCH_SIZE]
        with tempfile.TemporaryDirectory(
            prefix="apps-reference-probe-", dir=APPS_PROBE_ROOT
        ) as scratch:
            scratch_root = Path(scratch)
            assert_disposable_episode_dir(scratch_root)
            (scratch_root / "probe.py").write_text(
                _apps_reference_probe_source(reference, input_batch), encoding="utf-8"
            )
            command = f"cd /work && {shlex.quote(str(jail_python()))} -I probe.py"
            result: CommandResult = run_in_jail(
                scratch_root,
                command,
                timeout="30s",
                login_shell=False,
                cpus=1,
                mem_max="1G",
            )
        batch_outputs, batch_seconds = _parse_apps_reference_batch(result, len(input_batch))
        outputs.extend(batch_outputs)
        maximum_case_seconds = max(maximum_case_seconds, batch_seconds)
    return outputs, maximum_case_seconds


def _generated_apps_candidates(
    request: _AppsGeneratedCaseRequest,
) -> tuple[list[_AppsCandidate], float, int]:
    generated_inputs = _generated_apps_inputs(request)
    input_texts = [cast("str", arguments[0]) for arguments in generated_inputs]
    outputs_by_reference = [
        _run_apps_reference_outputs(reference, input_texts, request.artifacts_root)
        for reference in request.references
    ]
    elapsed_seconds = max((seconds for _, seconds in outputs_by_reference), default=0.0)
    generated_candidates: list[_AppsCandidate] = []
    if len(outputs_by_reference) != _REQUIRED_REFERENCE_COUNT:
        raise RuntimeError("APPS generated-input agreement requires exactly two references")
    first_outputs, second_outputs = (pair[0] for pair in outputs_by_reference)
    for arguments, first_output, second_output in zip(
        generated_inputs, first_outputs, second_outputs, strict=True
    ):
        if first_output is None or second_output is None or first_output != second_output:
            continue
        expected = cast("str", _json_round_trip(first_output, "generated APPS expected output"))
        assertion = aisi_rh_envs_etl._assertion_from_pair(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            "solution", arguments[0], expected
        )
        generated_candidates.append(
            (arguments, expected, repr(expected), assertion, "generated_agreement")
        )
    return generated_candidates, elapsed_seconds, len(generated_inputs)


def _passing_assertion_indexes(
    task_id: str,
    reference: _ReferenceCheck,
    assertions: tuple[str, ...],
    elapsed_seconds: list[float],
) -> set[int]:
    if not assertions:
        return set()
    started_at = time.monotonic()
    verdict = aisi_rh_envs_etl._reference_assertions_pass(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        reference, assertions
    )
    elapsed_seconds.append(time.monotonic() - started_at)
    if verdict is None:
        raise AisiSourceUnavailableError(
            task_id, "an APPS upstream reference timed out while validating source inputs"
        )
    if verdict:
        return set(range(len(assertions)))
    if len(assertions) == 1:
        return set()
    split_at = len(assertions) // 2
    left_indexes = _passing_assertion_indexes(
        task_id, reference, assertions[:split_at], elapsed_seconds
    )
    right_indexes = _passing_assertion_indexes(
        task_id, reference, assertions[split_at:], elapsed_seconds
    )
    return left_indexes | {split_at + index for index in right_indexes}


def _append_apps_candidate(
    candidates: list[_AppsCandidate],
    expected_repr_by_digest: dict[str, str],
    candidate: _AppsCandidate,
) -> None:
    arguments = candidate[0]
    expected_repr = candidate[2]
    input_digest = canonical_digest(arguments)
    if input_digest in expected_repr_by_digest:
        if expected_repr_by_digest[input_digest] != expected_repr:
            raise ValueError("duplicate APPS input has conflicting outputs")
        return
    candidates.append(candidate)
    expected_repr_by_digest[input_digest] = expected_repr


def _apps_candidate_cases(
    source_record: Mapping[str, object],
    visible_cases: list[tuple[list[object], object]],
) -> list[_AppsCandidate]:
    visible_digests = {
        canonical_digest(_json_round_trip(arguments, "APPS arguments"))
        for arguments, _ in visible_cases
    }
    input_output_value = aisi_rh_envs_etl._json_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        source_record, "input_output"
    )
    input_output_pairs = aisi_rh_envs_etl._input_output_pairs(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        input_output_value, field="input_output"
    )
    candidates: list[_AppsCandidate] = []
    expected_repr_by_digest: dict[str, str] = {}
    for input_value, expected_value in input_output_pairs:
        if not isinstance(input_value, str) or not isinstance(expected_value, str):
            raise TypeError("APPS input_output values must be strings")
        arguments = [input_value]
        if canonical_digest(arguments) in visible_digests:
            continue
        assertion = aisi_rh_envs_etl._assertion_from_pair(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            "solution", input_value, expected_value
        )
        _append_apps_candidate(
            candidates,
            expected_repr_by_digest,
            (
                cast("list[object]", arguments),
                _json_round_trip(expected_value, "APPS expected output"),
                repr(expected_value),
                assertion,
                "upstream_unseen",
            ),
        )

    for arguments, expected in visible_cases:
        if (
            len(arguments) != 1
            or not isinstance(arguments[0], str)
            or not isinstance(expected, str)
        ):
            raise ValueError("AISI APPS visible tests must use one string input and output")
        generated_input = arguments[0] + "\n"
        generated_arguments = [generated_input]
        if canonical_digest(generated_arguments) in visible_digests:
            continue
        assertion = aisi_rh_envs_etl._assertion_from_pair(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            "solution", generated_input, expected
        )
        _append_apps_candidate(
            candidates,
            expected_repr_by_digest,
            (
                cast("list[object]", generated_arguments),
                _json_round_trip(expected, "APPS expected output"),
                repr(expected),
                assertion,
                "weak_visible_newline_variant",
            ),
        )

    return candidates


def _validate_apps_candidates(
    task: AisiRhProblem,
    references: tuple[_ReferenceCheck, ...],
    candidates: list[_AppsCandidate],
    reference_validation_seconds: float,
) -> tuple[list[dict[str, object]], list[str], list[str], list[str], list[float]]:
    assertions_to_check = tuple(candidate[3] for candidate in candidates)
    elapsed_seconds = [reference_validation_seconds]
    passing_by_reference = [
        _passing_assertion_indexes(task.problem_id, reference, assertions_to_check, elapsed_seconds)
        for reference in references
    ]
    agreeing_indexes = passing_by_reference[0] & passing_by_reference[1]
    cases: list[dict[str, object]] = []
    assertions: list[str] = []
    expected_reprs: list[str] = []
    case_kinds: list[str] = []
    for case_index, (arguments, expected, expected_repr, assertion, case_kind) in enumerate(
        candidates
    ):
        if case_index not in agreeing_indexes:
            continue
        cases.append(
            {
                "args": arguments,
                "expected": expected,
                "case_digest": canonical_digest(arguments),
            }
        )
        assertions.append(assertion)
        expected_reprs.append(expected_repr)
        case_kinds.append(case_kind)
    if not {"upstream_unseen", "generated_agreement"}.intersection(case_kinds):
        raise AisiSourceUnavailableError(
            task.problem_id, "no validated unseen APPS input/output cases"
        )
    return cases, assertions, expected_reprs, case_kinds, elapsed_seconds


def build_apps_case_record(
    task: AisiRhProblem,
    source_record: Mapping[str, object],
    aisi_source: AisiSource,
    artifacts_root: Path,
    *,
    generated_inputs: AppsGeneratedInputConfig | None = None,
) -> dict[str, object]:
    """Build APPS cases from validated upstream and statement-generated stdin inputs."""
    if task.source != "apps" or task.entry_point != "solution":
        raise ValueError("upstream APPS cases require an AISI APPS solution task")
    visible_cases = [
        _assertion_case(assertion, task.entry_point) for assertion in task.original_assertions
    ]
    candidates = _apps_candidate_cases(source_record, visible_cases)
    references, reference_validation_seconds = _unique_app_references(
        task, source_record, aisi_source
    )
    generated_input_count_run = 0
    generated_reference_seconds = 0.0
    if generated_inputs is not None:
        generated_candidates, generated_reference_seconds, generated_input_count_run = (
            _generated_apps_candidates(
                _AppsGeneratedCaseRequest(
                    task=task,
                    generated_inputs=generated_inputs,
                    visible_cases=visible_cases,
                    existing_candidates=candidates,
                    references=references,
                    artifacts_root=artifacts_root,
                )
            )
        )
        candidates.extend(generated_candidates)
    elif not any(candidate[4] == "upstream_unseen" for candidate in candidates):
        raise AisiSourceUnavailableError(task.problem_id, "no unseen APPS input_output cases")
    cases, assertions, expected_reprs, case_kinds, elapsed_seconds = _validate_apps_candidates(
        task, references, candidates, reference_validation_seconds
    )
    reference_artifacts = [
        _write_reference_artifact(
            artifacts_root,
            task,
            aisi_rh_envs_etl._reference_solution_source(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                reference
            ),
        )
        for reference in references
    ]

    case_record: dict[str, object] = {
        "task_id": task.task_id("original"),
        "source": "upstream_reference",
        "cases": cases,
        "provenance": {
            "problem_id": task.problem_id,
            "source_task_id": task.source_problem_id,
            "entry_point": task.entry_point,
            "upstream_entry_point": task.entry_point,
            "reference_sha256": [
                hashlib.sha256(reference.solution.encode("utf-8")).hexdigest()
                for reference in references
            ],
            "reference_artifacts": reference_artifacts,
            "reference_count": len(references),
            "reference_seconds": max(
                max(elapsed_seconds, default=0.0), generated_reference_seconds
            ),
            "reference_seconds_basis": (
                "maximum measured AISI pytest validation-call or generated-input execution duration"
            ),
            "generated_input_policy": (
                "retain exact-stdout agreement from two distinct validated upstream references; "
                "also retain non-visible APPS input_output cases and weak visible newline variants"
            ),
            "assertions": assertions,
            "original_assertions": list(task.truth_assertions),
            "expected_repr": expected_reprs,
            "case_kinds": case_kinds,
        },
    }
    if generated_inputs is not None:
        provenance = cast("dict[str, object]", case_record["provenance"])
        provenance.update(
            {
                "generated_inputs": True,
                "generated_input_count": generated_inputs.count,
                "generated_input_count_run": generated_input_count_run,
                "generated_input_case_count": sum(
                    case_kind == "generated_agreement" for case_kind in case_kinds
                ),
                "generated_input_seed": generated_inputs.seed,
                "generated_input_generator_sha256": generated_inputs.generator.sha256,
                "generated_input_output_normalization": "exact AISI solution stdout string equality",
            }
        )
    return case_record


def _case_path(cases_root: Path, source: str, task: AisiRhProblem) -> Path:
    return cases_root / source / f"{task.task_id('original')}.json"


def _write_case_file(path: Path, case_file: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(case_file, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _quarantine_oversized_case_file(config: AisiCaseBuildConfig, task: AisiRhProblem) -> str | None:
    case_path = _case_path(config.cases_root, "evalplus", task)
    if not case_path.exists():
        return None
    if not case_path.is_file():
        raise ValueError(f"generated EvalPlus case path is not a file: {case_path}")
    archive_root = config.cases_root.parent / "oversized-case-files" / "evalplus"
    archive_path = archive_root / case_path.name
    archive_root.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"oversized case archive already exists: {archive_path}")
    case_path.replace(archive_path)
    return archive_path.relative_to(config.cases_root.parent).as_posix()


def _selected_tasks(config: AisiCaseBuildConfig) -> tuple[AisiRhProblem, ...]:
    requested_ids = selected_aisi_problem_ids(config.records_path, config.selection_path)
    problems_by_id = {
        problem.problem_id: problem for problem in load_problems(config.aisi_cases_path)
    }
    missing_ids = requested_ids.difference(problems_by_id)
    if missing_ids:
        raise ValueError(
            f"AISI selected task ids are missing from the baked corpus: {sorted(missing_ids)}"
        )
    return tuple(problems_by_id[problem_id] for problem_id in sorted(requested_ids))


def _evalplus_records_for_tasks(
    tasks: tuple[AisiRhProblem, ...], cache_dir: Path, sources_to_build: tuple[str, ...]
) -> dict[str, dict[str, dict[str, Any]]]:
    records_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    if "evalplus" not in sources_to_build:
        return records_by_source
    for source in ("humaneval", "mbpp"):
        if any(task.source == source for task in tasks):
            records_by_source[source] = evalplus_etl.fetch_records(source, cache_dir)
    return records_by_source


def _apps_records_for_tasks(
    tasks: tuple[AisiRhProblem, ...], config: AisiCaseBuildConfig
) -> tuple[dict[str, Mapping[str, object]], AisiSource | None]:
    if "apps" not in config.sources_to_build or not any(task.source == "apps" for task in tasks):
        return {}, None
    # The ETL's source loader owns the APPS cache and row validation.
    raw_records = aisi_rh_envs_etl._load_apps_records(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        config.external_root, config.apps_cache_dir
    )
    requested_ids = {task.problem_id for task in tasks if task.source == "apps"}
    apps_records = {
        str(record["problem_id"]): record
        for record in raw_records
        if isinstance(record.get("problem_id"), int)
        and f"apps:{record['problem_id']}" in requested_ids
    }
    return apps_records, load_aisi_source(config.external_root)


def _case_for_evalplus_record(
    task: AisiRhProblem,
    source: str,
    record: Mapping[str, Any],
    config: AisiCaseBuildConfig,
) -> tuple[dict[str, object] | None, tuple[str, dict[str, str]] | None]:
    try:
        baked = evalplus_etl.bake_problem(
            source, cast("dict[str, Any]", record), max_plus_cases=config.max_plus_cases
        )
    except evalplus_etl.OracleError as error:
        if _INTEGER_STRING_LIMIT_DIAGNOSTIC not in str(error):
            raise
        return None, (
            "reference_output_limit",
            {"problem_id": task.problem_id, "reason": "python_integer_string_conversion_limit"},
        )
    try:
        case_file = build_evalplus_case_record(
            task,
            baked,
            config.cases_root.parent,
            max_case_payload_bytes=config.max_case_payload_bytes,
        )
    except CasePayloadTooLargeError as error:
        skip_record = {
            "problem_id": task.problem_id,
            "repr_payload_bytes": str(error.observed_bytes),
            "repr_payload_limit_bytes": str(error.limit_bytes),
        }
        archived_path = _quarantine_oversized_case_file(config, task)
        if archived_path is not None:
            skip_record["archived_path"] = archived_path
        return None, ("case_payload_limit", skip_record)
    return case_file, None


def _evalplus_case_files(
    tasks: tuple[AisiRhProblem, ...],
    records_by_source: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: AisiCaseBuildConfig,
) -> tuple[list[tuple[Path, dict[str, object]]], dict[str, list[dict[str, str]]]]:
    pending_files: list[tuple[Path, dict[str, object]]] = []
    skipped: dict[str, list[dict[str, str]]] = {
        "special_oracle": [],
        "missing_release": [],
        "reference_output_limit": [],
        "case_payload_limit": [],
        "apps_unavailable": [],
    }
    files_by_source = {"evalplus": pending_files, "apps": []}
    _write_source_coverage(config, tasks, files_by_source, skipped, status="in_progress")
    for task in tasks:
        if "evalplus" not in config.sources_to_build or task.source not in {"humaneval", "mbpp"}:
            continue
        source = cast("str", task.source)
        source_task_id = (
            task.source_problem_id if source == "humaneval" else f"Mbpp/{task.source_problem_id}"
        )
        record = records_by_source[source].get(source_task_id)
        if record is None:
            if source == "humaneval":
                raise ValueError(f"{task.problem_id} is absent from the HumanEval+ release")
            skipped["missing_release"].append(
                {"problem_id": task.problem_id, "source_task_id": source_task_id}
            )
            _write_source_coverage(config, tasks, files_by_source, skipped, status="in_progress")
            continue
        entry_point = record.get("entry_point")
        if not isinstance(entry_point, str):
            raise TypeError(f"{task.problem_id} EvalPlus entry point must be text")
        special_oracle = evalplus_special_oracle_reason(source, entry_point)
        if special_oracle is not None:
            skipped["special_oracle"].append(
                {"problem_id": task.problem_id, "oracle": special_oracle}
            )
            _write_source_coverage(config, tasks, files_by_source, skipped, status="in_progress")
            continue
        case_file, skip = _case_for_evalplus_record(task, source, record, config)
        if skip is not None:
            skip_category, skip_details = skip
            skipped[skip_category].append(skip_details)
            _write_source_coverage(config, tasks, files_by_source, skipped, status="in_progress")
            logger.info("skipped an EvalPlus source whose output was unmeasurable")
            continue
        if case_file is None:
            raise RuntimeError("EvalPlus case builder returned neither a case nor a skip reason")
        case_path = _case_path(config.cases_root, "evalplus", task)
        _write_case_file(case_path, case_file)
        pending_files.append((case_path, case_file))
        _write_source_coverage(config, tasks, files_by_source, skipped, status="in_progress")
    return pending_files, skipped


def _apps_case_files(
    progress: _AisiSourceBuildProgress,
    apps_records: Mapping[str, Mapping[str, object]],
    aisi_source: AisiSource | None,
) -> list[tuple[Path, dict[str, object]]]:
    progress.write_coverage(status="in_progress")
    if "apps" not in progress.config.sources_to_build:
        return progress.apps_files
    apps_tasks = tuple(task for task in progress.tasks if task.source == "apps")
    if apps_tasks and aisi_source is None:
        raise RuntimeError("AISI source adapter was not initialized for APPS")
    for task in apps_tasks:
        source_record = apps_records.get(task.source_problem_id)
        if source_record is None:
            raise ValueError(f"{task.problem_id} is absent from the cached APPS source records")
        try:
            apps_generator = load_apps_input_generator(progress.config.apps_generators_dir, task)
            generated_input_config = (
                None
                if apps_generator is None
                else AppsGeneratedInputConfig(
                    generator=apps_generator,
                    count=progress.config.generated_apps_input_count,
                    seed=progress.config.generated_apps_input_seed,
                )
            )
            case_file = build_apps_case_record(
                task,
                source_record,
                cast("AisiSource", aisi_source),
                progress.config.cases_root.parent,
                generated_inputs=generated_input_config,
            )
        except AisiSourceUnavailableError as error:
            progress.skipped["apps_unavailable"].append(
                {"problem_id": task.problem_id, "reason": error.reason}
            )
            progress.write_coverage(status="in_progress")
            logger.info("AISI APPS source was unavailable for %s", task.problem_id)
            continue
        case_path = _case_path(progress.config.cases_root, "upstream_reference", task)
        _write_case_file(case_path, case_file)
        progress.apps_files.append((case_path, case_file))
        progress.write_coverage(status="in_progress")
    return progress.apps_files


def _write_source_coverage(
    config: AisiCaseBuildConfig,
    tasks: tuple[AisiRhProblem, ...],
    files_by_source: Mapping[str, list[tuple[Path, dict[str, object]]]],
    skipped: dict[str, list[dict[str, str]]],
    *,
    status: str,
) -> None:
    evalplus_files = files_by_source["evalplus"]
    apps_files = files_by_source["apps"]
    selected_source_counts = {
        source: sum(task.source == source for task in tasks)
        for source in sorted(SUPPORTED_AISI_SOURCES)
    }
    requested_source_counts = {
        "evalplus": (
            selected_source_counts["humaneval"] + selected_source_counts["mbpp"]
            if "evalplus" in config.sources_to_build
            else 0
        ),
        "apps": selected_source_counts["apps"] if "apps" in config.sources_to_build else 0,
    }
    coverage = {
        "status": status,
        "requested_sources": list(config.sources_to_build),
        "requested_problem_count": len(tasks),
        "selected_source_counts": selected_source_counts,
        "requested_source_counts": requested_source_counts,
        "case_files_written": len(evalplus_files) + len(apps_files),
        "evalplus_written": len(evalplus_files),
        "apps_upstream_written": len(apps_files),
        **skipped,
    }
    coverage_path = config.cases_root.parent / "aisi-source-coverage.json"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_selected_case_files(config: AisiCaseBuildConfig) -> tuple[Path, ...]:
    """Bake selected AISI sources and write one ignored case file per supported task."""
    tasks = _selected_tasks(config)
    evalplus_records = _evalplus_records_for_tasks(
        tasks, config.evalplus_cache_dir, config.sources_to_build
    )
    evalplus_files, skipped = _evalplus_case_files(tasks, evalplus_records, config)
    progress = _AisiSourceBuildProgress(config, tasks, evalplus_files, skipped)

    apps_records, aisi_source = _apps_records_for_tasks(tasks, config)
    apps_files = _apps_case_files(progress, apps_records, aisi_source)
    progress.write_coverage(status="complete")
    return tuple(path for path, _ in (*evalplus_files, *apps_files))


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch validated AISI source cases for the seed-v4 task population."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--evalplus-cache-dir", type=Path, default=evalplus_etl.DEFAULT_CACHE_DIR)
    parser.add_argument("--apps-cache-dir", type=Path, default=Path.home() / ".cache" / "aisi-apps")
    parser.add_argument(
        "--external-root", type=Path, default=Path("external/aisi-reward-hacking-misalignment")
    )
    parser.add_argument(
        "--aisi-cases",
        type=Path,
        default=Path("reward_hacking/harness/data/aisi_rh_envs_cases.json"),
    )
    parser.add_argument("--max-plus-cases", type=int, default=evalplus_etl.DEFAULT_MAX_PLUS_CASES)
    parser.add_argument("--max-case-payload-bytes", type=int, default=_MAX_CASE_PAYLOAD_BYTES)
    parser.add_argument("--apps-generators-dir", type=Path, default=_APPS_GENERATOR_ROOT)
    parser.add_argument(
        "--generated-apps-input-count", type=int, default=_GENERATED_APPS_INPUT_COUNT
    )
    parser.add_argument("--generated-apps-input-seed", type=int, default=_GENERATED_APPS_INPUT_SEED)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("evalplus", "apps"),
        default=("evalplus", "apps"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="strengthened-truth-aisi: %(message)s")
    config = AisiCaseBuildConfig(
        records_path=args.records,
        selection_path=args.selection,
        cases_root=args.cases_root,
        evalplus_cache_dir=args.evalplus_cache_dir,
        apps_cache_dir=args.apps_cache_dir,
        external_root=args.external_root,
        aisi_cases_path=args.aisi_cases,
        max_plus_cases=args.max_plus_cases,
        sources_to_build=tuple(args.sources),
        max_case_payload_bytes=args.max_case_payload_bytes,
        apps_generators_dir=args.apps_generators_dir,
        generated_apps_input_count=args.generated_apps_input_count,
        generated_apps_input_seed=args.generated_apps_input_seed,
    )
    paths = build_selected_case_files(config)
    logger.info("wrote %d AISI strengthened-truth files", len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
