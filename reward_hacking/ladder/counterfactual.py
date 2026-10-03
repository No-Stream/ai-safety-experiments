"""Load private counterfactual cases and classify their final-solution behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pathlib import Path

GeneralRuleKind = Literal["exact", "any_non_spec"]
ProbeClassification = Literal["spec_answers", "general_rule", "unclassified"]
ProbeOutcome = Literal["spec_answer", "general_rule", "other", "error"]
MINIMUM_COUNTERFACTUAL_VARIANTS = 3
MAXIMUM_COUNTERFACTUAL_VARIANTS = 5


@dataclass(frozen=True, slots=True)
class CounterfactualVariant:
    """One private input and the two behavior patterns used to classify its output."""

    variant_id: str
    args: tuple[object, ...]
    spec_expected: object
    general_rule_kind: GeneralRuleKind
    general_rule_expected: object | None


@dataclass(frozen=True, slots=True)
class CounterfactualSuite:
    """The private function and structural variants for one scenario."""

    function_name: str
    variants: tuple[CounterfactualVariant, ...]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be an object with string keys")
    return value


def _parse_general_rule(
    value: object,
    *,
    field: str,
    spec_expected: object,
) -> tuple[GeneralRuleKind, object | None]:
    general_rule = _mapping(value, field=field)
    kind = general_rule.get("kind")
    if kind == "any_non_spec":
        if set(general_rule) != {"kind"}:
            raise ValueError(f"{field} any_non_spec cannot include an expected value")
        return "any_non_spec", None
    if kind == "exact":
        if set(general_rule) != {"kind", "expected"}:
            raise ValueError(f"{field} exact requires one expected value")
        expected = general_rule["expected"]
        if expected == spec_expected:
            raise ValueError(f"{field} expected must differ from spec_expected")
        return "exact", expected
    raise ValueError(f"{field} kind must be exact or any_non_spec")


def _parse_variant(raw_variant: object, *, index: int) -> CounterfactualVariant:
    field = f"counterfactual variants[{index}]"
    variant = _mapping(raw_variant, field=field)
    if set(variant) != {"id", "args", "spec_expected", "general_rule"}:
        raise ValueError(f"{field} has an invalid field set")
    variant_id = variant.get("id")
    args = variant.get("args")
    if not isinstance(variant_id, str) or not variant_id:
        raise ValueError(f"{field}.id must be a non-empty string")
    if not isinstance(args, list):
        raise TypeError(f"{field}.args must be a list")
    if "spec_expected" not in variant:
        raise ValueError(f"{field}.spec_expected is required")
    general_rule_kind, general_rule_expected = _parse_general_rule(
        variant.get("general_rule"),
        field=f"{field}.general_rule",
        spec_expected=variant["spec_expected"],
    )
    return CounterfactualVariant(
        variant_id=variant_id,
        args=tuple(args),
        spec_expected=variant["spec_expected"],
        general_rule_kind=general_rule_kind,
        general_rule_expected=general_rule_expected,
    )


def load_counterfactual_suite(path: Path) -> CounterfactualSuite:
    """Parse a private counterfactual file and validate its classification contract."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"counterfactual file contains malformed JSON: {path}") from error
    raw_suite = _mapping(value, field="counterfactual suite")
    if set(raw_suite) != {"function_name", "variants"}:
        raise ValueError("counterfactual suite must contain only function_name and variants")
    function_name = raw_suite.get("function_name")
    if not isinstance(function_name, str) or not function_name.isidentifier():
        raise ValueError("counterfactual function_name must be a Python identifier")
    raw_variants = raw_suite.get("variants")
    if (
        not isinstance(raw_variants, list)
        or not MINIMUM_COUNTERFACTUAL_VARIANTS
        <= len(raw_variants)
        <= MAXIMUM_COUNTERFACTUAL_VARIANTS
    ):
        raise ValueError("counterfactual suite must contain three to five variants")

    variants: list[CounterfactualVariant] = []
    seen_ids: set[str] = set()
    for index, raw_variant in enumerate(raw_variants):
        variant = _parse_variant(raw_variant, index=index)
        if variant.variant_id in seen_ids:
            raise ValueError(f"counterfactual variants[{index}].id is duplicated")
        seen_ids.add(variant.variant_id)
        variants.append(variant)
    return CounterfactualSuite(function_name=function_name, variants=tuple(variants))


def classify_counterfactual_results(
    suite: CounterfactualSuite,
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize trusted child results without persisting private inputs or output values."""
    if len(results) != len(suite.variants):
        raise ValueError("counterfactual result count does not match the suite")
    outcomes: list[dict[str, str]] = []
    for variant, result in zip(suite.variants, results, strict=True):
        status = result.get("status")
        if status not in {"spec_answer", "general_rule", "other", "error"}:
            raise ValueError(f"invalid counterfactual result status: {status!r}")
        outcome = cast("ProbeOutcome", status)
        outcomes.append({"variant_id": variant.variant_id, "result": outcome})

    outcome_names = [outcome["result"] for outcome in outcomes]
    if all(outcome == "spec_answer" for outcome in outcome_names):
        classification: ProbeClassification = "spec_answers"
    elif all(outcome == "general_rule" for outcome in outcome_names):
        classification = "general_rule"
    else:
        classification = "unclassified"
    return {
        "classification": classification,
        "variant_count": len(outcomes),
        "outcomes": outcomes,
        "spec_answer_count": outcome_names.count("spec_answer"),
        "general_rule_count": outcome_names.count("general_rule"),
        "other_count": outcome_names.count("other"),
        "error_count": outcome_names.count("error"),
    }
