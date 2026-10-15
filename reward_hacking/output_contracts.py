"""Host-side explicit output domains for deterministic oneoff mutations.

Contracts describe the legal output family, not whether an answer solves the task. Per-input
bounds and witness formats must be supplied by the caller; examples never define the domain.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Any, NoReturn, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

MUTATION_ALGORITHM = "output-contract-v1"


class OutputContractError(ValueError):
    """A domain is invalid, unsupported, or has no matcher-distinct candidate."""


UnsupportedOutputContract = OutputContractError


def contract_digest(contract: dict[str, Any]) -> str:
    """Bind the complete resolved domain, including input-dependent constraints."""
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _reject(reason: str) -> NoReturn:
    raise UnsupportedOutputContract(reason)


def _parts(value: object, contract: dict[str, Any]) -> list[Any]:
    kind = contract["type"]
    if kind in {"list", "tuple"}:
        if type(value) is not (list if kind == "list" else tuple):
            _reject("container_type_mismatch")
        return list(value)  # type: ignore[arg-type]
    if not isinstance(value, str):
        _reject("text_type_mismatch")
    return value.split() if kind == "text_tokens" else value.splitlines()  # type: ignore[union-attr]


def _schemas(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return contract[{"text_tokens": "tokens", "text_lines": "lines"}.get(contract["type"], "items")]


def validate_output(value: object, contract: dict[str, Any]) -> None:
    """Raise when an output violates the explicitly declared resolved domain."""
    kind = contract.get("type")
    if not isinstance(kind, str):
        _reject("contract_type_required")
    if kind == "any_of":
        for option in contract["options"]:
            try:
                validate_output(value, option)
            except UnsupportedOutputContract:
                continue
            return
        _reject("no_union_branch_matches")
    elif kind == "enum":
        if not any(type(value) is type(item) and value == item for item in contract["values"]):
            _reject("value_outside_enum")
    elif kind in {"integer", "number"}:
        _validate_numeric(value, contract)
    else:
        validator = {
            "string": _validate_string,
            "list": _validate_sequence,
            "tuple": _validate_sequence,
            "text_tokens": _validate_sequence,
            "text_lines": _validate_sequence,
            "object": _validate_object,
        }.get(kind)
        if validator is None:
            _reject("unsupported_contract_type")
        validator(value, contract)


def _validate_numeric(value: object, contract: dict[str, Any]) -> None:
    kind = contract["type"]
    required_type = int if kind == "integer" else float
    if type(value) is not required_type or not math.isfinite(value):  # type: ignore[arg-type]
        _reject("numeric_type_mismatch")
    if "minimum" not in contract or "maximum" not in contract:
        _reject("numeric_bounds_required")
    if not contract["minimum"] <= value <= contract["maximum"]:
        _reject("value_outside_numeric_bounds")


def _validate_string(value: object, contract: dict[str, Any]) -> None:
    if not isinstance(value, str) or not contract.get("alphabet"):
        _reject("string_alphabet_required")
    if "length" not in contract:
        _reject("string_length_required")
    if len(value) != contract["length"] or any(
        character not in contract["alphabet"] for character in value
    ):
        _reject("string_outside_domain")


def _validate_sequence(value: object, contract: dict[str, Any]) -> None:
    parts = _parts(value, contract)
    schemas = _schemas(contract)
    if len(parts) != len(schemas):
        _reject("output_shape_mismatch")
    for part, schema in zip(parts, schemas, strict=True):
        validate_output(
            _decode_token(part, schema) if contract["type"] == "text_tokens" else part, schema
        )


def _validate_object(value: object, contract: dict[str, Any]) -> None:
    properties = contract["properties"]
    if not isinstance(value, dict) or set(value) != set(properties):
        _reject("object_shape_mismatch")
    for key, schema in properties.items():
        validate_output(value[key], schema)


def _decode_token(value: str, schema: dict[str, Any]) -> object:
    kind = schema.get("type")
    if kind == "enum":
        return next((item for item in schema["values"] if str(item) == value), value)
    if kind == "any_of":
        return _decode_union_token(value, schema)
    try:
        if kind == "integer":
            return int(value)
        if kind == "number":
            return float(value)
    except ValueError as error:
        raise UnsupportedOutputContract("invalid_numeric_token") from error
    return value


def _decode_union_token(value: str, schema: dict[str, Any]) -> object:
    for option in schema["options"]:
        try:
            decoded = _decode_token(value, option)
            validate_output(decoded, option)
        except UnsupportedOutputContract:
            continue
        return decoded
    _reject("no_union_token_branch_matches")


def _replace_part(value: object, kind: str, index: int, replacement: object) -> object:
    if kind in {"text_tokens", "text_lines"}:
        pattern = r"\S+" if kind == "text_tokens" else r"[^\r\n]+|(?<=\n)(?=\n)"
        spans = list(re.finditer(pattern, value))  # type: ignore[arg-type]
        if index >= len(spans):
            _reject("unsupported_empty_line_mutation")
        span = spans[index]
        return value[: span.start()] + str(replacement) + value[span.end() :]  # type: ignore[index]
    parts = list(value)  # type: ignore[arg-type]
    parts[index] = replacement
    return tuple(parts) if kind == "tuple" else parts


def _union_candidates(value: object, contract: dict[str, Any]) -> Iterator[object]:
    for option in contract["options"]:
        if option["type"] == "enum":
            yield from option["values"]
        elif option["type"] in {"integer", "number"}:
            yield option["minimum"]
            yield option["maximum"]
        try:
            validate_output(value, option)
        except UnsupportedOutputContract:
            continue
        yield from _candidates(value, option)


def _numeric_candidates(value: object, contract: dict[str, Any]) -> Iterator[object]:
    value = cast("int | float", value)
    kind = contract["type"]
    step = 1 if kind == "integer" else float(contract.get("step", 1.0))
    yield value + step  # type: ignore[operator]
    yield value - step  # type: ignore[operator]
    if not any(
        contract["minimum"] <= candidate <= contract["maximum"]
        for candidate in (value + step, value - step)
    ):  # type: ignore[operator]
        yield contract["minimum"]
        yield contract["maximum"]


def _sequence_candidates(value: object, contract: dict[str, Any]) -> Iterator[object]:
    kind = contract["type"]
    for index, (part, schema) in enumerate(
        zip(_parts(value, contract), _schemas(contract), strict=True)
    ):
        decoded = _decode_token(part, schema) if kind == "text_tokens" else part
        for replacement in _candidates(decoded, schema):
            yield _replace_part(value, kind, index, replacement)


def _string_candidates(value: object, contract: dict[str, Any]) -> Iterator[object]:
    for index in range(len(value)):  # type: ignore[arg-type]
        for character in contract["alphabet"]:
            yield value[:index] + character + value[index + 1 :]  # type: ignore[index]


def _candidates(value: object, contract: dict[str, Any]) -> Iterator[object]:
    kind = contract["type"]
    if kind == "enum":
        yield from contract["values"]
    elif kind == "any_of":
        yield from _union_candidates(value, contract)
    elif kind in {"integer", "number"}:
        yield from _numeric_candidates(value, contract)
    elif kind == "string":
        yield from _string_candidates(value, contract)
    elif kind in {"list", "tuple", "text_tokens", "text_lines"}:
        yield from _sequence_candidates(value, contract)
    elif kind == "object":
        for key, schema in contract["properties"].items():
            for replacement in _candidates(value[key], schema):  # type: ignore[index]
                yield {**value, key: replacement}  # type: ignore[misc]


def _candidate_distance(value: Any, candidate: Any, contract: dict[str, Any]) -> float:  # noqa: ANN401 - Recursive literal trees.
    if isinstance(value, (int, float)) and isinstance(candidate, (int, float)):
        return abs(float(candidate - value))
    kind = contract.get("type")
    if kind in {"text_tokens", "text_lines", "list", "tuple"}:
        parts = _parts(value, contract)
        changed_parts = _parts(candidate, contract)
        schemas = _schemas(contract)
        return sum(
            _candidate_distance(_decode_token(part, schema), _decode_token(changed, schema), schema)
            if kind == "text_tokens"
            else _candidate_distance(part, changed, schema)
            for part, changed, schema in zip(parts, changed_parts, schemas, strict=True)
        )
    if kind == "object":
        value_mapping = cast("dict[str, Any]", value)
        candidate_mapping = cast("dict[str, Any]", candidate)
        return sum(
            _candidate_distance(value_mapping[key], candidate_mapping[key], schema)
            for key, schema in contract["properties"].items()
        )
    if isinstance(value, str) and isinstance(candidate, str):
        return float(
            abs(len(value) - len(candidate))
            + sum(left != right for left, right in zip(value, candidate, strict=False))
        )
    return 1.0


def mutate_output(
    value: object,
    contract: dict[str, Any],
    *,
    seed: bytes,
    outputs_match: Callable[[Any, Any], bool],
) -> object:
    """Choose a deterministic legal candidate that the original matcher rejects."""
    validate_output(value, contract)
    # Resolve ties with the seed without materializing large string domains.
    selected: object = None
    selected_rank: tuple[float, bytes] | None = None
    for index, candidate in enumerate(_candidates(value, contract)):
        try:
            validate_output(candidate, contract)
        except UnsupportedOutputContract:
            continue
        if type(candidate) is not type(value) or outputs_match(value, candidate):
            continue
        distance = _candidate_distance(value, candidate, contract)
        rank = (distance, hashlib.sha256(seed + str(index).encode()).digest())
        if selected_rank is None or rank < selected_rank:
            selected, selected_rank = candidate, rank
    if selected_rank is None:
        _reject("no_matcher_distinct_candidate")
    return selected
