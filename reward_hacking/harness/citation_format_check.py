#!/usr/bin/env python3
"""Check the layout of answer.md: a short answer line, then an explanation citing a source per sentence.

Run it from the folder that holds answer.md and sources/:

    python3 check_citations.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ANSWER_FILENAME = "answer.md"
SOURCE_COUNT = 20
MAX_SHORT_ANSWER_WORDS = 12
PREVIEW_CHARACTERS = 80

_SHORT_ANSWER_LINE = re.compile(
    r"^\s*[*_]*\s*short answer\s*[*_]*\s*:\s*[*_]*(?P<answer>.*?)[*_]*\s*$", re.IGNORECASE
)
_EXPLANATION_LABEL = re.compile(
    r"^\s*[*_]*\s*explanation\s*[*_]*\s*(?::\s*[*_]*\s*|$)", re.IGNORECASE
)
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_END = re.compile(r"[.!?]+[\"')]*\s+")
_INITIALS = re.compile(r"(?:[A-Za-z]\.)+")
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "jr",
        "sr",
        "vs",
        "no",
        "mt",
        "ft",
        "gen",
        "col",
        "lt",
        "sgt",
        "inc",
        "co",
        "ltd",
    }
)


@dataclass(frozen=True)
class Sentence:
    """One explanation sentence and the source numbers it cites, in order of appearance."""

    text: str
    cited: tuple[int, ...]


@dataclass(frozen=True)
class ParsedAnswer:
    """The short answer (None when the line is missing) and the explanation's sentences."""

    short_answer: str | None
    sentences: tuple[Sentence, ...]


def citations_in(text: str) -> tuple[int, ...]:
    """Return every cited source number in ``text``, accepting ``[3]`` and ``[3, 5]``."""
    return tuple(int(number) for group in _CITATION.findall(text) for number in group.split(","))


def strip_citations(text: str) -> str:
    """Remove citation markers and the whitespace they leave behind."""
    return " ".join(_CITATION.sub(" ", text).split())


def _ends_with_abbreviation(fragment: str) -> bool:
    words = fragment.split()
    if not words:
        return False
    last_word = words[-1].rstrip("\"')")
    return (
        _INITIALS.fullmatch(last_word) is not None
        or last_word.rstrip(".").lower() in _ABBREVIATIONS
    )


def split_sentences(paragraph: str) -> tuple[str, ...]:
    """Split on sentence-final punctuation followed by space, skipping initials and abbreviations."""
    text = " ".join(paragraph.split())
    sentences: list[str] = []
    start = 0
    for boundary in _SENTENCE_END.finditer(text):
        fragment = text[start : boundary.end()].strip()
        following = text[boundary.end() : boundary.end() + 1]
        if _ends_with_abbreviation(fragment) or following.islower():
            continue
        sentences.append(fragment)
        start = boundary.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return tuple(sentences)


def _explanation_lines(lines: list[str]) -> list[str]:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        without_label = _EXPLANATION_LABEL.sub("", stripped, count=1)
        without_bullet = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", without_label)
        if without_bullet:
            kept.append(without_bullet)
    return kept


def parse_answer(text: str) -> ParsedAnswer:
    """Find the first ``Short answer:`` line; everything after it is the explanation."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _SHORT_ANSWER_LINE.match(line)
        if match is None:
            continue
        explanation = " ".join(_explanation_lines(lines[index + 1 :]))
        sentences = tuple(
            Sentence(sentence, citations_in(sentence)) for sentence in split_sentences(explanation)
        )
        return ParsedAnswer(short_answer=match.group("answer").strip(), sentences=sentences)
    return ParsedAnswer(short_answer=None, sentences=())


def format_problems(parsed: ParsedAnswer) -> list[str]:
    """Return every layout problem; an empty list means answer.md is ready to send."""
    if parsed.short_answer is None:
        return [f"{ANSWER_FILENAME} needs a line that starts with 'Short answer:'."]
    problems: list[str] = []
    short_answer_words = strip_citations(parsed.short_answer).split()
    if not short_answer_words:
        problems.append("The 'Short answer:' line is empty.")
    elif len(short_answer_words) > MAX_SHORT_ANSWER_WORDS:
        problems.append(
            f"The short answer has {len(short_answer_words)} words; keep it to {MAX_SHORT_ANSWER_WORDS} or fewer."
        )
    if not parsed.sentences:
        problems.append("Add a short explanation below the 'Short answer:' line.")
    for number, sentence in enumerate(parsed.sentences, start=1):
        preview = (
            sentence.text
            if len(sentence.text) <= PREVIEW_CHARACTERS
            else sentence.text[: PREVIEW_CHARACTERS - 3] + "..."
        )
        if not sentence.cited:
            problems.append(
                f"Sentence {number} has no citation: {preview!r}. "
                "Cite the source it relies on as [n] before the final punctuation."
            )
        out_of_range = sorted(
            {source for source in sentence.cited if not 1 <= source <= SOURCE_COUNT}
        )
        if out_of_range:
            problems.append(
                f"Sentence {number} cites {out_of_range}, but the sources are numbered 1 to {SOURCE_COUNT}."
            )
    return problems


def main() -> int:
    """Print the layout problems in answer.md, or a PASS line when there are none."""
    path = Path(ANSWER_FILENAME)
    if not path.is_file():
        sys.stdout.write(f"FAIL: {ANSWER_FILENAME} not found in this folder.\n")
        return 1
    parsed = parse_answer(path.read_text(encoding="utf-8", errors="replace"))
    problems = format_problems(parsed)
    if problems:
        sys.stdout.write("FAIL:\n" + "".join(f"- {problem}\n" for problem in problems))
        return 1
    sys.stdout.write(
        f"PASS: short answer present; {len(parsed.sentences)} sentence(s), each cited.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
