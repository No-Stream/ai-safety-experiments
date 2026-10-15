"""Legacy unconstrained mutations retained for old artifact reproduction.

New collection uses explicit host-side domains in :mod:`reward_hacking.output_contracts`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_SEED_IDENTITY_BYTES = 8
_MINIMUM_MUTATION_SEED_BYTES = _SEED_IDENTITY_BYTES + 1
_WHITESPACE_OUTPUT_REPLACEMENTS = (" ", "\t", "\r", "\n", "\v", "\f")
_NUMERIC_OUTPUT_TOKEN: re.Pattern[str] = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_YES_NO_OUTPUT_TOKEN: re.Pattern[str] = re.compile(
    r"(?P<prefix>[^\w]*)(?P<value>yes|no)(?P<suffix>[^\w]*)", re.IGNORECASE
)
_TEXT_REPLACEMENTS = (".", "!", "?", ",", ";", ":")
_MAX_DECIMAL_DIGIT = 9


class OutputMutationError(ValueError):
    """The output has no deterministic mutation rejected by its output matcher."""


def _mutate_numeric_token(token: str, seed: bytes) -> str:
    """Perturb one numeric token while preserving its sign and displayed precision."""
    if "." not in token and "e" not in token.lower():
        sign = token[0] if token[:1] in {"+", "-"} else ""
        digits = token[len(sign) :]
        magnitude = int(digits)
        direction = 1 if seed[0] & 1 else -1
        if magnitude == 0 or (sign == "-" and magnitude == 1 and direction < 0):
            direction = 1
        changed_magnitude = magnitude + direction
        return sign + str(changed_magnitude).zfill(len(digits))

    exponent_start = min(
        (position for marker in "eE" if (position := token.find(marker)) >= 0),
        default=len(token),
    )
    coefficient = token[:exponent_start]
    digit_positions = [
        position for position, character in enumerate(coefficient) if character.isdigit()
    ]
    significant_positions = [
        position for position in digit_positions if coefficient[position] != "0"
    ]
    mutation_position = (significant_positions or digit_positions)[-1]
    digit = int(coefficient[mutation_position])
    direction = 1 if seed[0] & 1 else -1
    if digit == _MAX_DECIMAL_DIGIT:
        direction = -1
    elif digit == 0:
        direction = 1
    mutated_digit = str(digit + direction)
    return token[:mutation_position] + mutated_digit + token[mutation_position + 1 :]


def _mutate_text_character(character: str, seed: bytes) -> str:
    """Change one ordinary output character without adding a diagnostic marker."""
    if "a" <= character <= "z":
        direction = 1 if seed[0] & 1 else -1
        return chr(ord("a") + (ord(character) - ord("a") + direction) % 26)
    if "A" <= character <= "Z":
        direction = 1 if seed[0] & 1 else -1
        return chr(ord("A") + (ord(character) - ord("A") + direction) % 26)
    if "0" <= character <= "9":
        direction = 1 if seed[0] & 1 else -1
        return str((int(character) + direction) % 10)
    if character.isalpha():
        return "y" if character.casefold() == "x" else "x"
    replacements = tuple(value for value in _TEXT_REPLACEMENTS if value != character)
    return replacements[int.from_bytes(seed[:4], "big") % len(replacements)]


def _mutate_numeric_output(
    expected_output: str, numeric_tokens: list[re.Match[str]], seed: bytes
) -> str:
    match = numeric_tokens[int.from_bytes(seed[:_SEED_IDENTITY_BYTES], "big") % len(numeric_tokens)]
    mutated_token = _mutate_numeric_token(match.group(), seed[_SEED_IDENTITY_BYTES:])
    return expected_output[: match.start()] + mutated_token + expected_output[match.end() :]


def _mutate_yes_no_output(
    expected_output: str, yes_no_tokens: list[re.Match[str]], seed: bytes
) -> str:
    match = yes_no_tokens[int.from_bytes(seed[:_SEED_IDENTITY_BYTES], "big") % len(yes_no_tokens)]
    token_match = _YES_NO_OUTPUT_TOKEN.fullmatch(match.group())
    if token_match is None:
        raise RuntimeError("selected Yes/No output token stopped matching")
    word = token_match.group("value")
    replacement = "no" if word.casefold() == "yes" else "yes"
    if word.isupper():
        replacement = replacement.upper()
    elif word.istitle():
        replacement = replacement.title()
    mutated_token = token_match.group("prefix") + replacement + token_match.group("suffix")
    return expected_output[: match.start()] + mutated_token + expected_output[match.end() :]


def _mutate_ordinary_output(expected_output: str, seed: bytes) -> tuple[str, bool]:
    mutable_positions = [
        position for position, character in enumerate(expected_output) if not character.isspace()
    ]
    if mutable_positions:
        position = mutable_positions[
            int.from_bytes(seed[:_SEED_IDENTITY_BYTES], "big") % len(mutable_positions)
        ]
        replacement = _mutate_text_character(expected_output[position], seed[_SEED_IDENTITY_BYTES:])
        return (
            expected_output[:position] + replacement + expected_output[position + 1 :],
            False,
        )

    whitespace_positions = [
        position for position, character in enumerate(expected_output) if character not in "\r\n"
    ]
    if not whitespace_positions:
        raise OutputMutationError("no valid oneoff mutation for whitespace-only expected output")
    position = whitespace_positions[
        int.from_bytes(seed[:_SEED_IDENTITY_BYTES], "big") % len(whitespace_positions)
    ]
    replacement_options = tuple(
        whitespace
        for whitespace in _WHITESPACE_OUTPUT_REPLACEMENTS
        if whitespace != expected_output[position]
    )
    replacement = replacement_options[
        int.from_bytes(seed[_SEED_IDENTITY_BYTES : _SEED_IDENTITY_BYTES + 4], "big")
        % len(replacement_options)
    ]
    return expected_output[:position] + replacement + expected_output[position + 1 :], True


def mutate_string_output(
    expected_output: str,
    *,
    seed: bytes,
    outputs_match: Callable[[str, str], bool],
) -> str:
    """Make one deterministic minimal string change rejected by the caller's output matcher."""
    if len(seed) < _MINIMUM_MUTATION_SEED_BYTES:
        raise ValueError("oneoff mutation seed must contain at least nine bytes")

    output_tokens = list(re.finditer(r"\S+", expected_output))
    numeric_tokens = [
        match for match in output_tokens if _NUMERIC_OUTPUT_TOKEN.fullmatch(match.group())
    ]
    if numeric_tokens:
        mutated_output = _mutate_numeric_output(expected_output, numeric_tokens, seed)
        whitespace_only = False
    else:
        yes_no_tokens = [
            match for match in output_tokens if _YES_NO_OUTPUT_TOKEN.fullmatch(match.group())
        ]
        if yes_no_tokens:
            mutated_output = _mutate_yes_no_output(expected_output, yes_no_tokens, seed)
            whitespace_only = False
        else:
            mutated_output, whitespace_only = _mutate_ordinary_output(expected_output, seed)

    if outputs_match(expected_output, mutated_output):
        if whitespace_only:
            raise OutputMutationError("no valid oneoff mutation under output normalization")
        raise RuntimeError("oneoff expected-output mutation matches under output normalization")
    return mutated_output
