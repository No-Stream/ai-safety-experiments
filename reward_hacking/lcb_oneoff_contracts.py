"""Host-side strict LiveCodeBench derivation and read validation without dataset imports."""

from __future__ import annotations

import ast
import hashlib
import json
from decimal import Decimal, InvalidOperation
from typing import Any

from reward_hacking.output_contracts import (
    MUTATION_ALGORITHM,
    UnsupportedOutputContract,
    contract_digest,
    mutate_output,
)


def json_digest(value: object) -> str:
    """Hash complete host-side metadata with stable JSON serialization."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def stdio_outputs_match(actual: str, expected: str) -> bool:
    """Compare stripped lines with exact decimal numeric-token equality."""
    actual_lines = [line.strip() for line in actual.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(actual_lines) != len(expected_lines):
        return False
    for actual_line, expected_line in zip(actual_lines, expected_lines, strict=True):
        if actual_line == expected_line:
            continue
        try:
            actual_numbers = [Decimal(token) for token in actual_line.split()]
            expected_numbers = [Decimal(token) for token in expected_line.split()]
        except InvalidOperation:
            return False
        if actual_numbers != expected_numbers:
            return False
    return True


def _expected_outputs(tree: ast.Module) -> list[ast.Constant]:
    checks = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check"
    ]
    if len(checks) != 1:
        raise ValueError("strict contract payload requires one visible check function")
    outputs: list[ast.Constant] = []
    for statement in checks[0].body:
        if not isinstance(statement, ast.Assert) or not isinstance(statement.test, ast.Call):
            raise ValueError("strict contract payload requires generated visible assertions")  # noqa: TRY004
        assertion = statement.test
        if (
            not isinstance(assertion.func, ast.Name)
            or assertion.func.id != "_lcb_outputs_match"
            or len(assertion.args) != 2  # noqa: PLR2004 - generated matcher has actual and expected arguments
            or not isinstance(assertion.args[1], ast.Constant)
            or not isinstance(assertion.args[1].value, str)
        ):
            raise ValueError("strict contract payload has an invalid visible output assertion")
        outputs.append(assertion.args[1])
    return outputs


def _oneoff(original: dict[str, Any], entry: object, catalog_sha256: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise UnsupportedOutputContract("missing task output contract")
    provenance = entry.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise UnsupportedOutputContract("missing output contract provenance")
    contracts = entry.get("cases")
    if not isinstance(contracts, dict):
        raise UnsupportedOutputContract("missing case output contracts")
    tree = ast.parse(original["test"])
    reasons: list[str] = []
    for index, output in enumerate(_expected_outputs(tree)):
        contract = contracts.get(str(index))
        if not isinstance(contract, dict):
            reasons.append(f"case {index}: missing output contract")
            continue
        seed = hashlib.sha256(
            f"lcb-recent-{MUTATION_ALGORITHM}\0{original['task_id']}\0{index}".encode()
        ).digest()
        try:
            mutated = mutate_output(
                output.value, contract, seed=seed, outputs_match=stdio_outputs_match
            )
        except UnsupportedOutputContract as error:
            reasons.append(f"case {index}: {error}")
            continue
        if not isinstance(mutated, str):
            raise UnsupportedOutputContract("stdio output contract must produce text")
        output.value = mutated
        return {
            **original,
            "impossible_type": "oneoff",
            "test": ast.unparse(tree) + "\n",
            "provenance": {
                **original["provenance"],
                "output_mutation": {
                    "algorithm": MUTATION_ALGORITHM,
                    "catalog_sha256": catalog_sha256,
                    "contract_sha256": contract_digest(contract),
                    "contract_provenance": provenance,
                    "case_index": index,
                },
            },
        }
    raise UnsupportedOutputContract("; ".join(reasons) or "no public case output contracts")


def build_contract_records(
    originals: list[dict[str, Any]], contracts: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every original and derive only explicitly supported oneoffs."""
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    catalog_sha256 = json_digest(contracts)
    for original in originals:
        records.append(original)
        try:
            oneoff = _oneoff(original, contracts.get(original["task_id"]), catalog_sha256)
        except UnsupportedOutputContract as error:
            exclusions.append({"task_id": original["task_id"], "reason": str(error)})
            continue
        records.append(oneoff)
    return records, {
        "algorithm": MUTATION_ALGORITHM,
        "catalog_sha256": catalog_sha256,
        "source_sha256": json_digest(originals),
        "contracts": contracts,
        "supported": len(originals) - len(exclusions),
        "unsupported": len(exclusions),
        "exclusions": exclusions,
    }


def validate_contract_payload(payload: dict[str, Any]) -> None:
    """Recompute strict assertions, provenance and exclusions before exposing task rows."""
    summary = payload.get("mutation_summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("contracts"), dict):
        raise ValueError("strict contract payload requires its explicit host-side catalog")  # noqa: TRY004
    rows = payload.get("problems")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("strict contract payload requires problem records")
    originals = [row for row in rows if row.get("impossible_type") == "original"]
    if len({row["task_id"] for row in originals}) != len(originals):
        raise ValueError("strict contract payload repeats an original task")
    expected_rows, expected_summary = build_contract_records(originals, summary["contracts"])
    if rows != expected_rows or summary != expected_summary:
        raise ValueError("strict contract payload does not match its source and output contracts")
