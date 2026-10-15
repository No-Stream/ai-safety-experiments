"""Synthetic output-domain regression tests; no benchmark material."""

import hashlib

import pytest

from reward_hacking.output_contracts import (
    UnsupportedOutputContract,
    mutate_output,
    validate_output,
)

SEED = hashlib.sha256(b"synthetic-contract").digest()


def test_numeric_sentinel_never_mutates_outside_explicit_family() -> None:
    contract = {
        "type": "any_of",
        "options": [
            {"type": "enum", "values": [-1]},
            {"type": "integer", "minimum": 0, "maximum": 3},
        ],
    }
    result = mutate_output(-1, contract, seed=SEED, outputs_match=lambda a, b: a == b)
    assert result in range(4)
    validate_output(result, contract)


def test_grid_alphabet_and_shape_are_preserved() -> None:
    contract = {
        "type": "text_lines",
        "lines": [{"type": "string", "alphabet": ".#", "length": 3}] * 2,
    }
    result = mutate_output(".#.\n###\n", contract, seed=SEED, outputs_match=lambda a, b: a == b)
    assert result != ".#.\n###\n"
    validate_output(result, contract)


def test_structured_decision_requires_witness_and_keeps_token_ranges() -> None:
    contract = {
        "type": "text_lines",
        "lines": [
            {"type": "enum", "values": ["YES"]},
            {
                "type": "text_tokens",
                "tokens": [{"type": "integer", "minimum": 1, "maximum": 4}] * 3,
            },
        ],
    }
    result = mutate_output("YES\n1 2 3\n", contract, seed=SEED, outputs_match=lambda a, b: a == b)
    validate_output(result, contract)
    with pytest.raises(UnsupportedOutputContract):
        validate_output("YES", contract)


def test_matcher_equivalent_only_domain_is_explicitly_unsupported() -> None:
    with pytest.raises(UnsupportedOutputContract, match="no_matcher_distinct_candidate"):
        mutate_output(
            "1",
            {"type": "enum", "values": ["1", "1.0"]},
            seed=SEED,
            outputs_match=lambda a, b: float(a) == float(b),
        )


def test_contract_rejects_original_outside_domain_and_missing_domain() -> None:
    for value, contract in [
        (4, {"type": "integer", "minimum": 0, "maximum": 3}),
        ("abc", {"type": "string"}),
    ]:
        with pytest.raises(UnsupportedOutputContract):
            mutate_output(value, contract, seed=SEED, outputs_match=lambda a, b: a == b)


def test_wide_numeric_domain_prefers_adjacent_candidates() -> None:
    for index in range(20):
        result = mutate_output(
            30,
            {"type": "integer", "minimum": 0, "maximum": 10000},
            seed=bytes([index]) * 32,
            outputs_match=lambda a, b: a == b,
        )
        assert result in {29, 31}


def test_nested_numeric_candidates_prefer_smaller_leaf_change() -> None:
    contract = {
        "type": "list",
        "items": [
            {"type": "number", "minimum": -1.0, "maximum": 1.0, "step": 0.01},
            {"type": "number", "minimum": -20.0, "maximum": 20.0, "step": 10.0},
        ],
    }
    assert mutate_output(
        [0.0, 0.0], contract, seed=b"\x00\x00", outputs_match=lambda a, b: a == b
    ) in [[-0.01, 0.0], [0.01, 0.0]]


def test_numeric_union_inside_text_tokens_uses_declared_sentinel_and_range() -> None:
    contract = {
        "type": "text_tokens",
        "tokens": [
            {
                "type": "any_of",
                "options": [
                    {"type": "enum", "values": [-3]},
                    {"type": "integer", "minimum": 0, "maximum": 4},
                ],
            }
        ],
    }
    result = mutate_output(
        "-3\n", contract, seed=SEED, outputs_match=lambda a, b: a.strip() == b.strip()
    )
    assert isinstance(result, str)
    assert int(result) in range(5)
    validate_output(result, contract)
