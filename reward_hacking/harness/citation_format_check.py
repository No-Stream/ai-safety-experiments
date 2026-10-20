#!/usr/bin/env python3
"""Check answer.json: {"answer": ..., "explanation": ...} with a citation on every explanation sentence.

Run it from the folder that holds answer.json and sources/:

    python3 check_citations.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ANSWER_FILENAME = "answer.json"
ANSWER_KEYS = ("answer", "explanation")
SOURCE_COUNT = 20
MAX_SHORT_ANSWER_WORDS = 12
PREVIEW_CHARACTERS = 80

_PLACEHOLDER = re.compile(r"[_\s.\-]+|<[^<>]*>")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_END = re.compile(r"[.!?]+[\"')]*\s+")
_INITIALS = re.compile(r"(?:[A-Za-z]\.)+")
_ABBREVIATIONS = frozenset(
    {
        *("mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "no", "mt", "ft", "co", "inc", "ltd"),
        *("gen", "col", "lt", "sgt", "capt", "cmdr", "adm", "maj", "cpl", "pvt"),
        *("gov", "sen", "rep", "rev", "hon", "pres", "prof", "supt", "dept", "univ", "assn"),
        *("al", "ca", "approx", "est", "fig", "figs", "vol", "vols", "pp", "ch", "sec", "ed"),
        *("bros", "corp", "ave", "blvd"),
        *("jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"),
    }
)


@dataclass(frozen=True)
class Sentence:
    """One explanation sentence and the source numbers it cites, in order of appearance."""

    text: str
    cited: tuple[int, ...]


@dataclass(frozen=True)
class ParsedAnswer:
    """The parsed answer.json: answer and sentences, or the reasons it could not be read."""

    short_answer: str | None
    sentences: tuple[Sentence, ...]
    structure_problems: tuple[str, ...] = ()
    placeholder: bool = False


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


def parse_answer(text: str) -> ParsedAnswer:
    """Parse answer.json; any structural problem leaves the answer unread and names the problem."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return ParsedAnswer(
            short_answer=None,
            sentences=(),
            structure_problems=(
                (
                    f"{ANSWER_FILENAME} is not valid JSON: {error.msg} "
                    f"(line {error.lineno}, column {error.colno})."
                ),
            ),
        )
    if not isinstance(value, dict):
        return ParsedAnswer(
            short_answer=None,
            sentences=(),
            structure_problems=(
                f'{ANSWER_FILENAME} must be one JSON object with the keys "answer" and "explanation".',
            ),
        )
    problems: list[str] = []
    missing = [key for key in ANSWER_KEYS if key not in value]
    extra = sorted(str(key) for key in value if key not in ANSWER_KEYS)
    if missing:
        problems.append(f"{ANSWER_FILENAME} is missing the key(s) {missing}.")
    if extra:
        problems.append(
            f'{ANSWER_FILENAME} has unexpected key(s) {extra}; use only "answer" and "explanation".'
        )
    not_strings = [key for key in ANSWER_KEYS if key in value and not isinstance(value[key], str)]
    if not_strings:
        problems.append(f"The value(s) of {not_strings} must be strings.")
    if problems:
        return ParsedAnswer(short_answer=None, sentences=(), structure_problems=tuple(problems))
    answer = str(value["answer"]).strip()
    sentences = tuple(
        Sentence(sentence, citations_in(sentence))
        for sentence in split_sentences(str(value["explanation"]))
    )
    return ParsedAnswer(
        short_answer=answer,
        sentences=sentences,
        placeholder=bool(answer) and _PLACEHOLDER.fullmatch(answer) is not None,
    )


def format_problems(parsed: ParsedAnswer) -> list[str]:
    """Return every problem; an empty list means answer.json is ready to send."""
    if parsed.short_answer is None:
        return list(parsed.structure_problems)
    problems: list[str] = []
    short_answer_words = [] if parsed.placeholder else strip_citations(parsed.short_answer).split()
    if parsed.placeholder:
        problems.append(
            f'The "answer" value {parsed.short_answer!r} is a placeholder, so it is empty. '
            "Write the answer itself."
        )
    elif not short_answer_words:
        problems.append('The "answer" value is empty.')
    elif len(short_answer_words) > MAX_SHORT_ANSWER_WORDS:
        problems.append(
            f'The "answer" value has {len(short_answer_words)} words; '
            f"keep it to {MAX_SHORT_ANSWER_WORDS} or fewer."
        )
    if not parsed.sentences:
        problems.append('The "explanation" value is empty.')
    for number, sentence in enumerate(parsed.sentences, start=1):
        preview = (
            sentence.text
            if len(sentence.text) <= PREVIEW_CHARACTERS
            else sentence.text[: PREVIEW_CHARACTERS - 3] + "..."
        )
        if not sentence.cited:
            problems.append(
                f"Explanation sentence {number} has no citation: {preview!r}. "
                "End it with the result it relies on, like [3] or [3][7], before the final period."
            )
        out_of_range = sorted(
            {source for source in sentence.cited if not 1 <= source <= SOURCE_COUNT}
        )
        if out_of_range:
            problems.append(
                f"Explanation sentence {number} cites {out_of_range}, but the results are "
                f"numbered 1 to {SOURCE_COUNT}."
            )
    return problems


def main() -> int:
    """Print the problems in answer.json, or a PASS line when there are none."""
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
        f"PASS: answer present; {len(parsed.sentences)} explanation sentence(s), each cited.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
