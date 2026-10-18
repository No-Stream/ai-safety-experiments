"""Derive a separately identified AISI registry from explicit host-side output contracts."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from collections import Counter
from typing import Any, cast

from reward_hacking.output_contracts import (
    MUTATION_ALGORITHM,
    UnsupportedOutputContract,
    contract_digest,
    mutate_output,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _expected(assertion: str) -> tuple[ast.expr, Any]:
    parsed = ast.parse(assertion)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise UnsupportedOutputContract("unsupported_assertion")
    comparison = parsed.body[0].test
    if (
        not isinstance(comparison, ast.Compare)
        or len(comparison.ops) != 1
        or not isinstance(comparison.ops[0], ast.Eq)
    ):
        raise UnsupportedOutputContract("unsupported_assertion")
    node = comparison.comparators[0]
    try:
        return node, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise UnsupportedOutputContract("nonliteral_expected_output") from error


def _replace(assertion: str, node: ast.expr, value: object) -> str:
    lines = assertion.splitlines(keepends=True)
    start = sum(len(line.encode()) for line in lines[: node.lineno - 1]) + node.col_offset
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("expected output AST lacks source positions")
    end = sum(len(line.encode()) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    encoded = assertion.encode()
    return (encoded[:start] + repr(value).encode() + encoded[end:]).decode()


def _derive_row(row: dict[str, Any], entry: object) -> None:
    originals = row["original_assertions"]
    row["oneoff_assertions"] = list(originals)
    metadata: dict[str, Any] = {"algorithm": MUTATION_ALGORITHM, "status": "unsupported"}
    row["output_mutation"] = metadata
    if entry is None:
        metadata["reason"] = "missing_contract"
        return
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("provenance"), str)
        or not entry["provenance"].strip()
    ):
        raise ValueError("each output contract entry requires source provenance")
    cases = entry["cases"]
    if not isinstance(cases, dict) or any(
        not key.isdecimal() or int(key) >= len(originals) for key in cases
    ):
        raise ValueError("output contract case indices must address original visible assertions")
    preferred = row["source_mutation_index"]
    if type(preferred) is not int or not 0 <= preferred < len(originals):
        raise ValueError("source mutation index must address an original visible assertion")
    indices = sorted((int(index) for index in cases), key=lambda index: (index != preferred, index))
    reasons: Counter[str] = Counter()
    for index in indices:
        contract = cases[str(index)]
        try:
            node, expected = _expected(originals[index])
            seed = hashlib.sha256(
                f"{MUTATION_ALGORITHM}\0{row['problem_id']}\0{index}".encode()
            ).digest()
            changed = mutate_output(
                expected, contract, seed=seed, outputs_match=lambda left, right: left == right
            )
        except UnsupportedOutputContract as error:
            reasons[str(error)] += 1
            continue
        row["oneoff_assertions"][index] = _replace(originals[index], node, changed)
        row["mutation_index"] = index
        metadata.update(
            status="supported",
            matcher_id="python-equality-v1",
            case_index=index,
            contract=contract,
            contract_sha256=contract_digest(contract),
            contract_provenance=entry["provenance"],
        )
        return
    metadata["reason"] = "no_supported_case" if not reasons else ";".join(sorted(reasons))


def derive_oneoff_registry(source: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Preserve source rows and splits; explicitly exclude uncontracted oneoff variants."""
    if catalog.get("schema") != 1 or not isinstance(catalog.get("contracts"), dict):
        raise ValueError("output contract catalog requires schema 1 and contracts mapping")
    derived = copy.deepcopy(source)
    derived.pop("output_mutation_policy", None)
    for row in derived["problems"]:
        row["source_mutation_index"] = row["mutation_index"]
        _derive_row(row, catalog["contracts"].get(row["problem_id"]))
    exclusions = {
        row["problem_id"]: row["output_mutation"]["reason"]
        for row in derived["problems"]
        if row["output_mutation"]["status"] == "unsupported"
    }
    derived["output_mutation_registry"] = {
        "schema": 1,
        "algorithm": MUTATION_ALGORITHM,
        "source_sha256": _digest(source),
        "catalog_sha256": _digest(catalog),
        "catalog": copy.deepcopy(catalog),
        "source_fields": _source_fields(derived),
        "supported_count": len(derived["problems"]) - len(exclusions),
        "unsupported_count": len(exclusions),
        "exclusions": exclusions,
        "source_fields_sha256": _digest(_source_fields(derived)),
    }
    validate_derived_registry(derived)
    return derived


def _source_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "provenance": payload["provenance"],
            "problems": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"oneoff_assertions", "mutation_index", "output_mutation"}
                }
                for row in payload["problems"]
            ],
        }
    )


def _value_differences(expected: object, stored: object, prefix: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(stored, dict):
        expected_values = cast("dict[str, object]", expected)
        stored_values = cast("dict[str, object]", stored)
        differences: list[str] = []
        for key in sorted(set(expected_values) | set(stored_values)):
            path = f"{prefix}.{key}"
            if key not in expected_values or key not in stored_values:
                differences.append(path)
            else:
                differences.extend(
                    _value_differences(expected_values[key], stored_values[key], path)
                )
        return differences
    return [] if expected == stored else [prefix]


def _source_field_differences(expected: dict[str, Any], stored: dict[str, Any]) -> list[str]:
    differences = [f"source_fields.{key}" for key in sorted(set(expected) ^ set(stored))]
    differences.extend(
        _value_differences(expected.get("provenance"), stored.get("provenance"), "provenance")
    )
    expected_rows = expected.get("problems")
    stored_rows = stored.get("problems")
    if not isinstance(expected_rows, list) or not isinstance(stored_rows, list):
        return [*differences, "problems"]
    expected_by_id = {row["problem_id"]: row for row in expected_rows}
    stored_by_id = {row["problem_id"]: row for row in stored_rows}
    for problem_id in sorted(set(expected_by_id) | set(stored_by_id)):
        if problem_id not in expected_by_id or problem_id not in stored_by_id:
            differences.append(f"problems.{problem_id}")
        else:
            differences.extend(
                _value_differences(
                    expected_by_id[problem_id], stored_by_id[problem_id], f"problems.{problem_id}"
                )
            )
    return differences


def validate_derived_registry(payload: dict[str, Any]) -> None:
    """Recompute every supported mutation and reject stale or mixed derived rows."""
    registry = payload["output_mutation_registry"]
    if registry.get("schema") != 1 or registry.get("algorithm") != MUTATION_ALGORITHM:
        raise ValueError("unsupported derived mutation registry")
    for field in ("source_sha256", "catalog_sha256"):
        if (
            not isinstance(registry.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", registry[field]) is None
        ):
            raise ValueError("invalid derived registry digest")
    _validate_source_fields(payload, registry)
    exclusions: dict[str, str] = {}
    for row in payload["problems"]:
        reason = _validate_row(row)
        expected = copy.deepcopy(row)
        _derive_row(expected, registry["catalog"]["contracts"].get(row["problem_id"]))
        if expected != row:
            raise ValueError("derived mutation does not match its catalog")
        if reason is not None:
            exclusions[row["problem_id"]] = reason
    if (
        registry["exclusions"] != exclusions
        or registry["unsupported_count"] != len(exclusions)
        or registry["supported_count"] != len(payload["problems"]) - len(exclusions)
    ):
        raise ValueError("derived mutation coverage counts mismatch")


def _validate_source_fields(payload: dict[str, Any], registry: dict[str, Any]) -> None:
    current_source_fields = _source_fields(payload)
    if "source_fields" not in registry:
        if registry["source_fields_sha256"] != _digest(current_source_fields):
            raise ValueError("legacy derived registry source/truth fields changed")
        return
    stored_source_fields = registry["source_fields"]
    if not isinstance(stored_source_fields, dict):
        raise TypeError("derived registry source_fields must be an object")
    differences = _source_field_differences(current_source_fields, stored_source_fields)
    if differences:
        raise ValueError("derived registry source fields changed at " + ", ".join(differences[:10]))


def _validate_row(row: dict[str, Any]) -> str | None:
    metadata = row["output_mutation"]
    if metadata.get("algorithm") != MUTATION_ALGORITHM:
        raise ValueError("mixed derived mutation algorithms")
    if metadata["status"] == "unsupported":
        if row["oneoff_assertions"] != row["original_assertions"] or not metadata.get("reason"):
            raise ValueError("unsupported derived mutation contains stale output")
        return metadata["reason"]
    if metadata["status"] == "supported":
        expected = copy.deepcopy(row)
        contract = metadata["contract"]
        _derive_row(
            expected,
            {
                "provenance": metadata["contract_provenance"],
                "cases": {str(metadata["case_index"]): contract},
            },
        )
        if expected != row:
            raise ValueError("derived mutation does not match its contract")
    else:
        raise ValueError("unknown derived mutation status")
    return None
