r"""Locate text segments in a tokenized prompt by token-id subsequence search, and prove each one.

Captured activations are indexed by token position, so any span a read pools over has to be a fact
about the id sequence the model actually read -- not about character offsets in a string that was
tokenized separately. This module locates a segment by tokenizing it on its own, finding those ids
as a contiguous subsequence of the full prompt's ids exactly once, and decoding the located span
back to compare against the text it was meant to cover. A span that cannot be found, is found twice,
or decodes to something else is a refusal, never a best guess.

The search is sound only because of how the Qwen pre-tokenizer chunks text, and two of its habits
dictate the shape of every needle:

* a chunk never crosses from one line into the next, so a needle that STARTS at a line start
  tokenizes there exactly as it does inside the prompt;
* a run of trailing punctuation swallows the newline run after it (``)\n`` is one chunk), and a
  whitespace run ending in a newline is one chunk too (``\n    \n`` is one chunk when a blank line
  carries indentation). So a needle has to END after the whitespace that closes its last line, up
  to and including the last newline of that run -- cut at the last visible character, its final
  token is one the prompt's id sequence never contains.

:func:`closing_whitespace` and :func:`line_prefix` apply that rule; :func:`locate_span` and
:func:`assert_round_trip` are the gate. Nothing here knows what a span means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase


class SpanError(ValueError):
    """A span that did not resolve to exactly one token subsequence, or did not decode back."""


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A half-open ``[start, end)`` range of token positions."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Refuse an empty or inverted span."""
        if self.start < 0 or self.end <= self.start:
            raise SpanError(f"empty or inverted span [{self.start}, {self.end})")

    @property
    def width(self) -> int:
        """Return how many tokens the span covers."""
        return self.end - self.start


def encode(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[int, ...]:
    """Tokenize without added special tokens: a templated prompt already carries its own."""
    return tuple(cast("list[int]", tokenizer(text, add_special_tokens=False)["input_ids"]))


def find_token_subsequence(
    haystack: Sequence[int], needle: Sequence[int], *, start: int = 0
) -> list[int]:
    """Return every position at or after ``start`` where ``needle`` occurs contiguously."""
    if not needle:
        raise ValueError("an empty needle occurs everywhere; nothing to locate")
    width = len(needle)
    first = needle[0]
    wanted = tuple(needle)
    return [
        position
        for position in range(start, len(haystack) - width + 1)
        if haystack[position] == first and tuple(haystack[position : position + width]) == wanted
    ]


def locate_span(ids: Sequence[int], needle: Sequence[int], *, start: int, what: str) -> TokenSpan:
    """Find ``needle`` exactly once at or after ``start``; zero or several matches are refusals."""
    matches = find_token_subsequence(ids, needle, start=start)
    if len(matches) != 1:
        raise SpanError(
            f"{what}: {len(matches)} token-subsequence matches at or after position {start} "
            f"(needle {len(needle)} tokens, prompt {len(ids)} tokens); a span has to be unique"
        )
    return TokenSpan(matches[0], matches[0] + len(needle))


def assert_round_trip(
    tokenizer: PreTrainedTokenizerBase,
    ids: Sequence[int],
    span: TokenSpan,
    expected: str,
    *,
    what: str,
) -> None:
    """Decode the span and refuse it unless it reproduces the text it claims to cover."""
    decoded = tokenizer.decode(list(ids[span.start : span.end]), skip_special_tokens=False)
    if decoded != expected:
        raise SpanError(
            f"{what}: the span [{span.start}, {span.end}) decodes to {len(decoded)} characters "
            f"that differ from the {len(expected)}-character text it was located for"
        )


def closing_whitespace(rest: str) -> str:
    """Return the whitespace the pre-tokenizer glues to a line end: the run through its last newline.

    ``rest`` is whatever follows the segment. Trailing spaces before the next line's first visible
    character belong to that line's indentation chunk and are left out.
    """
    run = rest[: len(rest) - len(rest.lstrip())]
    cut = run.rfind("\n")
    return run[: cut + 1] if cut >= 0 else ""


def segment_needle(text: str, segment: str) -> str:
    """Return ``segment`` plus the whitespace that closes it in ``text``, where it occurs exactly once."""
    if text.count(segment) != 1:
        raise SpanError(
            f"a segment of {len(segment)} characters occurs {text.count(segment)} times, so it "
            f"cannot name one span"
        )
    return segment + closing_whitespace(text[text.index(segment) + len(segment) :])


def line_prefix(text: str, line_index: int) -> str:
    """Return the text through line ``line_index`` inclusive, plus the whitespace closing that line."""
    lines = text.split("\n")
    if not 0 <= line_index < len(lines):
        raise ValueError(f"line {line_index} is outside a {len(lines)}-line text")
    prefix = "\n".join(lines[: line_index + 1])
    return prefix + closing_whitespace(text[len(prefix) :])


def window_ending_at(end: int, *, width: int, floor: int, what: str) -> TokenSpan:
    """Return the ``width``-token span ending at ``end``, refusing one that starts before ``floor``."""
    if end - width < floor:
        raise SpanError(
            f"{what}: a {width}-token window ending at {end} would start before {floor}"
        )
    return TokenSpan(end - width, end)
