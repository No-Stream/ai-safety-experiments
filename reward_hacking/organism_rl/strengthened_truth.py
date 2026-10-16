"""Validate an ILCB-to-LiveCodeBench crosswalk before authoring strengthened truth.

The crosswalk and baked assertions contain benchmark material and must remain gitignored.
This stage does not grade candidates or alter existing truth verdicts.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import html
import io
import json
import logging
import pickle
import re
import shutil
import subprocess
import unicodedata
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Never

logger = logging.getLogger(__name__)
MINIMUM_SHARED_INPUTS = 3


@dataclass(frozen=True)
class CrosswalkPolicy:
    """Conservative evidence requirements, recorded in the crosswalk digest."""

    minimum_similarity: float = 0.4
    minimum_margin: float = 0.05
    minimum_shared_inputs: int = 3
    maximum_fallback_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Reject policies that weaken the shared-input requirement."""
        if not 0 < self.minimum_similarity <= 1 or not 0 < self.minimum_margin <= 1:
            raise ValueError("similarity and margin must be in (0, 1]")
        if self.minimum_shared_inputs < MINIMUM_SHARED_INPUTS:
            raise ValueError("at least three distinct shared inputs are required")
        if not 0 <= self.maximum_fallback_fraction < 1:
            raise ValueError("fallback fraction must be in [0, 1)")


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Never:
        raise pickle.UnpicklingError(f"private tests cannot load globals: {module}.{name}")


def decode_private_tests(payload: str) -> list[dict[str, Any]]:
    """Decode upstream JSON or its compressed pickle wrapper without loading globals."""
    if payload.lstrip().startswith("["):
        decoded = json.loads(payload)
    else:
        decoded = _RestrictedUnpickler(
            io.BytesIO(zlib.decompress(base64.b64decode(payload)))
        ).load()
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
    if not isinstance(decoded, list) or any(not isinstance(case, dict) for case in decoded):
        raise ValueError("private tests must decode to a list of objects")
    return decoded


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def validate_crosswalk_digest(
    crosswalk: dict[str, Any], *, expected_digest: str | None = None
) -> None:
    """Warn when a crosswalk's recorded reference-set digest no longer matches its contents."""
    payload = {key: value for key, value in crosswalk.items() if key != "reference_set_digest"}
    recorded = crosswalk["reference_set_digest"]
    if expected_digest is not None and recorded != expected_digest:
        logger.warning(
            "stale crosswalk reference-set digest: recorded %s but the expected digest is %s",
            recorded,
            expected_digest,
        )
    recomputed = _digest(payload)
    if recorded != recomputed:
        logger.warning(
            "crosswalk reference-set digest recorded %s but its contents hash to %s",
            recorded,
            recomputed,
        )


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize(
        "NFKC", html.unescape(re.sub(r"<[^>]+>", " ", text))
    ).casefold()
    return set(re.findall(r"\w+", normalized))


def _statement(prompt: str) -> str:
    tree = ast.parse(prompt)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("original prompt must define exactly one function")
    statement = ast.get_docstring(functions[0])
    if statement is None:
        raise ValueError("original prompt has no statement docstring")
    return statement


def _normalized(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    return value


def _strict_equality_key(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_strict_equality_key(item) for item in value))
    if isinstance(value, dict):
        return (
            type(value),
            frozenset(
                (_strict_equality_key(key), _strict_equality_key(item))
                for key, item in value.items()
            ),
        )
    return (type(value), value)


def _strictly_equal(left: object, right: object) -> bool:
    return _strict_equality_key(left) == _strict_equality_key(right)


def _outputs_match(left: object, right: object) -> bool:
    return _strictly_equal(_normalized(left), _normalized(right))


def _input_key(value: object) -> object:
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_input_key(item) for item in value]]
    if isinstance(value, dict):
        return {key: _input_key(item) for key, item in value.items()}
    return value


def _visible_cases(check: str) -> tuple[dict[str, Any], int, int, int]:
    cases: dict[str, Any] = {}
    unsupported = 0
    additional = 0
    assertion_count = 0
    for node in ast.walk(ast.parse(check)):
        if not isinstance(node, ast.Assert):
            continue
        assertion_count += 1
        assertion_pair = _visible_assertion_pair(node)
        if assertion_pair is None:
            unsupported += 1
            continue
        arguments, expected, newly_parsed = assertion_pair
        additional += newly_parsed
        key = _digest(_input_key(arguments))
        if key in cases and not _outputs_match(cases[key], expected):
            raise ValueError("original visible assertions contradict one another")
        cases[key] = expected
    return cases, unsupported, additional, assertion_count


def _visible_assertion_pair(
    assertion: ast.Assert,
) -> tuple[list[Any], Any, int] | None:
    comparison = assertion.test
    if (
        not isinstance(comparison, ast.Compare)
        or len(comparison.ops) != 1
        or not isinstance(comparison.ops[0], (ast.Eq, ast.Is))
    ):
        return None
    operator = comparison.ops[0]
    left = comparison.left
    right = comparison.comparators[0]
    left_call = _candidate_call(left)
    right_call = _candidate_call(right)
    if left_call is not None:
        call = left_call
        expected_node = right
    else:
        call = right_call
        expected_node = left
    if call is None:
        return None
    if isinstance(operator, ast.Is) and not isinstance(expected_node, ast.Constant):
        return None
    try:
        arguments = [ast.literal_eval(argument) for argument in call.args]
        expected = ast.literal_eval(expected_node)
    except (ValueError, TypeError):
        return None
    if isinstance(operator, ast.Is) and expected is not None:
        return None
    return arguments, expected, int(isinstance(operator, ast.Is))


def _candidate_call(node: ast.expr) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "candidate"
        and not node.keywords
    ):
        return node
    return None


def _lcb_cases(cases: list[dict[str, Any]]) -> dict[str, tuple[list[Any], Any]]:
    parsed: dict[str, tuple[list[Any], Any]] = {}
    for case in cases:
        if case["testtype"] != "functional":
            raise ValueError("crosswalk only supports function-style LCB tests")
        arguments = [json.loads(line) for line in case["input"].strip().splitlines()]
        expected = json.loads(case["output"])
        key = _digest(_input_key(arguments))
        if key in parsed and not _outputs_match(parsed[key][1], expected):
            raise ValueError("LCB cases have conflicting expectations")
        parsed[key] = (arguments, expected)
    return parsed


def _has_nested_tuple(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, tuple) or _has_nested_tuple(item) for item in value)
    if isinstance(value, dict):
        return any(isinstance(item, tuple) or _has_nested_tuple(item) for item in value.values())
    return False


def _validate_match(
    visible: dict[str, Any],
    candidate: _PreparedCandidate,
    policy: CrosswalkPolicy,
) -> dict[str, Any]:
    private = candidate.private_cases
    combined = candidate.combined_cases
    shared_inputs = visible.keys() & combined.keys()
    disagreements = {
        key for key in shared_inputs if not _outputs_match(visible[key], combined[key][1])
    }
    shared_pairs = len(shared_inputs - disagreements)
    novel_private = private.keys() - visible.keys()
    return {
        "shared_inputs": len(shared_inputs),
        "shared_pairs": shared_pairs,
        "shared_output_disagreements": len(disagreements),
        "visible_inputs": len(visible),
        "private_inputs": len(private),
        "novel_private_inputs": len(novel_private),
        "case_qualified": not disagreements
        and shared_pairs >= policy.minimum_shared_inputs
        and bool(novel_private),
    }


def _bake_assertions(
    visible: dict[str, Any],
    private: dict[str, tuple[list[Any], Any]],
    entry_point: str,
) -> dict[str, Any]:
    expected_types = {type(expected) for expected in visible.values()}
    if len(expected_types) != 1 or any(
        _has_nested_tuple(expected) for expected in visible.values()
    ):
        return {"reason": "unsupported_expected_representation"}
    tuple_expected = expected_types == {tuple}
    assertions: list[str] = []
    runner_assertions: list[str] = []
    expected_reprs: list[str] = []
    cases: list[dict[str, Any]] = []
    for key in sorted(private):
        if key in visible:
            continue
        arguments, expected = private[key]
        if tuple_expected:
            if not isinstance(expected, list):
                return {"reason": "unsupported_expected_representation"}
            check_expected = tuple(expected)
        else:
            check_expected = expected
        assertions.append(f"    assert candidate(*{tuple(arguments)!r}) == {check_expected!r}")
        expected_repr = repr(check_expected)
        expected_reprs.append(expected_repr)
        assertions_for_runner = f"assert {entry_point}(*{tuple(arguments)!r}) == {expected_repr}"
        runner_assertions.append(assertions_for_runner)
        cases.append(
            {
                "args": arguments,
                "expected": expected,
                "case_digest": _canonical_json_digest(arguments),
            }
        )
    return {
        "source": "lcb_private",
        "extra_assertions": "def check(candidate):\n" + "\n".join(assertions) + "\n",
        "expected_representation": "tuple" if tuple_expected else "json",
        "cases": cases,
        "assertions": runner_assertions,
        "expected_repr": expected_reprs,
    }


def _canonical_json_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _PreparedCandidate:
    row: dict[str, Any]
    metadata: dict[str, Any]
    public_cases: dict[str, tuple[list[Any], Any]]
    private_cases: dict[str, tuple[list[Any], Any]]
    combined_cases: dict[str, tuple[list[Any], Any]]
    source_tests_digest: str
    question_tokens: set[str]


def _prepare_candidate(row: dict[str, Any]) -> _PreparedCandidate | None:
    metadata = json.loads(row["metadata"])
    if not isinstance(metadata, dict):
        raise TypeError("LCB metadata must be an object")
    if not metadata.get("func_name"):
        return None
    public = _lcb_cases(json.loads(row["public_test_cases"]))
    private = _lcb_cases(decode_private_tests(row["private_test_cases"]))
    for key in public.keys() & private.keys():
        if not _outputs_match(public[key][1], private[key][1]):
            raise ValueError("public and private LCB expectations disagree")
    combined = public | private
    source_tests_digest = _digest({"public": public, "private": private})
    return _PreparedCandidate(
        row=row,
        metadata=metadata,
        public_cases=public,
        private_cases=private,
        combined_cases=combined,
        source_tests_digest=source_tests_digest,
        question_tokens=_tokens(row["question_content"]),
    )


DEFAULT_POLICY = CrosswalkPolicy()


def build_crosswalk(
    originals: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    policy: CrosswalkPolicy = DEFAULT_POLICY,
    *,
    lcb_revision: str | None = None,
) -> dict[str, Any]:
    """Accept only unique LCB test matches with agreeing shared cases and novel private inputs."""
    if not originals or len({row["task_id"] for row in originals}) != len(originals):
        raise ValueError("original tasks must be nonempty and unique")
    prepared_candidates = [
        prepared for row in candidates if (prepared := _prepare_candidate(row)) is not None
    ]
    if len({row.row["question_id"] for row in prepared_candidates}) != len(prepared_candidates):
        raise ValueError("LCB question ids must be unique")
    matches: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    assertion_count = 0
    unsupported_count = 0
    additional_parse_count = 0
    for original in sorted(originals, key=lambda row: row["task_id"]):
        visible, unsupported, additional, total_assertions = _visible_cases(
            original["original_test"]
        )
        assertion_count += total_assertions
        unsupported_count += unsupported
        additional_parse_count += additional
        if unsupported:
            unmatched.append(
                {
                    "task_id": original["task_id"],
                    "reason": "unsupported_visible_assertions",
                    "unsupported_visible_assertions": unsupported,
                    "visible_assertions": total_assertions,
                    "additional_visible_assertions_parsed": additional,
                }
            )
            continue
        candidate_match = _match_original(
            original,
            visible,
            prepared_candidates,
            policy,
            lcb_revision,
        )
        (unmatched if "reason" in candidate_match else matches).append(candidate_match)
    duplicated_ids = {
        row["question_id"]
        for row in matches
        if sum(other["question_id"] == row["question_id"] for other in matches) > 1
    }
    unique_matches = [row for row in matches if row["question_id"] not in duplicated_ids]
    unmatched.extend(
        row | {"reason": "source_reused_by_multiple_originals"}
        for row in matches
        if row["question_id"] in duplicated_ids
    )
    payload = {
        "schema": 1,
        "policy": asdict(policy),
        "matches": unique_matches,
        "unmatched": sorted(unmatched, key=lambda row: row["task_id"]),
        "fallback_fraction": len(unmatched) / len(originals),
        "function_candidate_count": len(prepared_candidates),
        "parse_summary": {
            "visible_assertions": assertion_count,
            "parsed_visible_assertions": assertion_count - unsupported_count,
            "unsupported_visible_assertions": unsupported_count,
            "additional_visible_assertions_parsed": additional_parse_count,
        },
        "stop_for_reference_authoring": len(unmatched) / len(originals)
        > policy.maximum_fallback_fraction,
    }
    return payload | {"reference_set_digest": _digest(payload)}


def _match_original(
    original: dict[str, Any],
    visible: dict[str, Any],
    candidates: list[_PreparedCandidate],
    policy: CrosswalkPolicy,
    lcb_revision: str | None,
) -> dict[str, Any]:
    if not candidates:
        return {"task_id": original["task_id"], "reason": "no_function_candidates"}
    statement_tokens = _tokens(_statement(original["prompt"]))
    qualified: list[tuple[_PreparedCandidate, dict[str, Any], float]] = []
    evidence_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence = _validate_match(visible, candidate, policy)
        union = statement_tokens | candidate.question_tokens
        similarity = (
            len(statement_tokens & candidate.question_tokens) / len(union) if union else 0.0
        )
        evidence["text_similarity"] = similarity
        evidence_rows.append(evidence)
        if evidence["case_qualified"]:
            qualified.append((candidate, evidence, similarity))
    if not qualified:
        return _unmatched_reason(original["task_id"], visible, evidence_rows, policy)
    selected = _select_case_candidate(qualified, policy)
    if selected is None:
        similarities = sorted((item[2] for item in qualified), reverse=True)
        if len(similarities) == 1:
            return {
                "task_id": original["task_id"],
                "reason": "below_minimum_similarity",
                "best_text_similarity": similarities[0],
            }
        return {
            "task_id": original["task_id"],
            "reason": "ambiguous_case_matches",
            "case_qualified_candidates": len(qualified),
            "best_text_similarity": similarities[0],
            "text_margin": similarities[0] - similarities[1],
        }
    candidate, case_evidence, similarity = selected
    baked = _bake_assertions(visible, candidate.private_cases, original["entry_point"])
    match = {
        "task_id": original["task_id"],
        "question_id": candidate.row["question_id"],
        "similarity": similarity,
        "function_name_agrees": candidate.metadata["func_name"] == original["entry_point"],
        "original_digest": _digest(original),
        "shared_inputs": case_evidence["shared_inputs"],
        "shared_pairs": case_evidence["shared_pairs"],
        "private_inputs": case_evidence["private_inputs"],
        "novel_private_inputs": case_evidence["novel_private_inputs"],
    }
    if "reason" in baked:
        return match | baked
    provenance = {
        "question_id": candidate.row["question_id"],
        "lcb_revision": lcb_revision,
        "original_digest": match["original_digest"],
        "source_tests_digest": candidate.source_tests_digest,
        "expected_representation": baked["expected_representation"],
        "assertions": baked["assertions"],
        "expected_repr": baked["expected_repr"],
        "reference_seconds": 0.0,
        "reference_seconds_basis": "not measured; grader applies its minimum candidate timeout",
        "shared_pairs": case_evidence["shared_pairs"],
        "novel_private_inputs": case_evidence["novel_private_inputs"],
    }
    return match | baked | {"provenance": provenance}


def _select_case_candidate(
    candidates: list[tuple[_PreparedCandidate, dict[str, Any], float]],
    policy: CrosswalkPolicy,
) -> tuple[_PreparedCandidate, dict[str, Any], float] | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: item[2], reverse=True)
    best = ranked[0]
    if best[2] < policy.minimum_similarity:
        return None
    if len(ranked) == 1:
        return best
    margin = best[2] - ranked[1][2]
    if margin >= policy.minimum_margin:
        return best
    return None


def _unmatched_reason(
    task_id: str,
    visible: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    policy: CrosswalkPolicy,
) -> dict[str, Any]:
    """Summarize why no candidate met the complete case-evidence rule."""
    if any(
        row["shared_pairs"] >= policy.minimum_shared_inputs
        and row["shared_output_disagreements"] == 0
        and row["novel_private_inputs"] == 0
        for row in evidence_rows
    ):
        reason = "no_novel_private_inputs"
    elif any(row["shared_output_disagreements"] for row in evidence_rows):
        reason = "shared_output_disagreement"
    elif any(row["shared_inputs"] for row in evidence_rows):
        reason = "insufficient_shared_inputs"
    else:
        reason = "no_shared_case_pairs"
    return {
        "task_id": task_id,
        "reason": reason,
        "visible_inputs": len(visible),
        "best_shared_pairs": max((row["shared_pairs"] for row in evidence_rows), default=0),
        "candidates_with_disagreement": sum(
            bool(row["shared_output_disagreements"]) for row in evidence_rows
        ),
        "candidates_without_novel_inputs": sum(
            row["shared_pairs"] >= policy.minimum_shared_inputs
            and row["shared_output_disagreements"] == 0
            and row["novel_private_inputs"] == 0
            for row in evidence_rows
        ),
    }


def write_case_files(
    crosswalk: dict[str, Any],
    cases_directory: Path,
    *,
    repository_root: Path,
) -> None:
    """Write accepted, digest-bound novel private cases to an ignored directory."""
    relative_directory = cases_directory.resolve().relative_to(repository_root.resolve())
    if not _is_gitignored(repository_root, relative_directory):
        raise ValueError("case files must be written under a gitignored path")
    case_payloads = {
        f"{match['task_id']}.json": {
            "task_id": match["task_id"],
            "source": "lcb_private",
            "cases": match["cases"],
            "provenance": match["provenance"],
        }
        for match in crosswalk["matches"]
    }
    cases_directory.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in cases_directory.glob("*.json")}
    stale = existing - case_payloads.keys()
    if stale:
        raise ValueError(f"case directory contains {len(stale)} stale files")
    for filename, payload in sorted(case_payloads.items()):
        target = cases_directory / filename
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(target)


def _is_gitignored(repository_root: Path, path: Path) -> bool:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to validate ignored output")
    ignored = subprocess.run(  # noqa: S603 - fixed git operation with a path argument
        [git, "check-ignore", "--quiet", "--", str(path)],
        cwd=repository_root,
        check=False,
    )
    return ignored.returncode == 0


def _load_function_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(paths):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if json.loads(row["metadata"]).get("func_name"):
                    candidates.append(row)
        logger.info(
            "read LCB shard %s; cumulative function candidates=%d", path.name, len(candidates)
        )
    return candidates


def main(argv: list[str] | None = None) -> None:
    """Write an ignored, digest-bound crosswalk; exit 2 when reference authoring needs review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ilcb-cases", type=Path, required=True)
    parser.add_argument("--lcb-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("docs/scratch/w16-strengthened-truth/cases/lcb_private"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).parents[2]
    relative = args.output.resolve().relative_to(root)
    if not _is_gitignored(root, relative):
        raise ValueError("crosswalk output must be gitignored")
    revisions = {path.parent.name for path in args.lcb_files}
    if len(revisions) != 1:
        raise ValueError("all LCB shards must come from one snapshot")
    revision = next(iter(revisions))
    originals = [
        row
        for row in json.loads(args.ilcb_cases.read_text())["problems"]
        if row["impossible_type"] == "original"
    ]
    result = build_crosswalk(
        originals,
        _load_function_candidates(args.lcb_files),
        lcb_revision=revision,
    )
    result["lcb_revision"] = revision
    result["reference_set_digest"] = _digest(
        {key: value for key, value in result.items() if key != "reference_set_digest"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    write_case_files(result, args.cases_dir, repository_root=root)
    logger.info(
        "validated=%d unmatched=%d fallback_fraction=%.3f",
        len(result["matches"]),
        len(result["unmatched"]),
        result["fallback_fraction"],
    )
    if result["stop_for_reference_authoring"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
