"""Scan files for text that must never leave this box. Exit non-zero on any hit.

Run over a list of paths given on the command line, before anything is uploaded or used
as training data and before anything is committed to a public history. ``docs/privacy-gate.md``
is the reference: what each family detects, why every exemption is there, and which
callers arm which detector.

Five families over each path:

  credential shapes    AWS access key IDs, PEM private-key headers, JWTs, session
                       cookies, bearer tokens: shapes recognizable without the value.
  machine-local        email addresses, AWS account ids, role ARNs, ECR registry hosts,
  identity             home-directory paths carrying a username. Each carries the
                       documented placeholders that must NOT fire.
  benchmark material   an item id beside a registered-answer field, per line for a trace
                       record and whole-file for a corpus (:func:`benchmark_item_file_findings`).
  canary values        exact values no shape can describe -- a username, a bucket, an AWS
                       profile, an employer, an account id -- read at run time from a
                       gitignored token file (:func:`load_canary_tokens`, syntax and what
                       belongs in it in ``canary/README.md``).
  instrument text      survey and psychometric item text, loaded at run time from the local
                       gitignored sources (:func:`collect_instrument_sources`) and never
                       spelled out here: a guard that lists the contraband IS the leak.

Findings never include the matched text: a hit reports the path, the line number, the
detector name and a truncated SHA-256, enough to correlate duplicates and confirm a fix
without copying the secret into a second log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
MIN_CANARY_TOKEN_LENGTH = 8

# A word boundary buys back what the substring floor above exists to prevent, so four is enough.
MIN_BOUNDED_CANARY_TOKEN_LENGTH = 4
BOUNDED_CANARY_PREFIX = "word:"
CANARY_CONTEXT_SEPARATOR = " unless="

# Room for the words flanking an identifier, not for an unrelated mention later on the same line.
CANARY_CONTEXT_CHARS = 24

EXIT_FINDINGS = 1
EXIT_UNARMED = 2

# AWS reserves these for documentation, and this repository's own tests use three of them.
EXAMPLE_AWS_ACCOUNT_IDS = frozenset(
    {
        "000000000000",
        "111111111111",
        "111122223333",
        "123456789012",
        "222222222222",
        "333333333333",
        "444455556666",
        "555555555555",
        "666666666666",
        "777788889999",
        "888888888888",
        "999999999999",
    }
)

# Home-directory users that name nobody: cloud-image defaults, plus this repo's docs and fixtures.
PLACEHOLDER_HOME_USERS = frozenset({"ec2-user", "root", "svc-ops", "ubuntu", "user", "username"})

# The words that make a nearby twelve-digit run an account id rather than a token count.
_ACCOUNT_KEYWORDS = r"(?i)(?:account|owner|s3|ecr|bedrock|batch|iam|role|registry|profile)"

# Closed direction list rather than `[a-z]+`, which would read `co-author-1` as a region.
_AWS_REGION = (
    r"[a-z]{2}(?:-gov|-iso[a-z]?)?-"
    r"(?:north|south|east|west|central|northeast|northwest|southeast|southwest)-\d"
)

# RFC 2606 / 6761 documentation domains, subdomains included, so no address there names anybody.
_RESERVED_EMAIL_DOMAIN = (
    r"(?:[A-Za-z0-9-]+\.)*(?:example\.(?:com|net|org)|example|invalid|localhost|test)"
)

# The fields a registered answer, a planted flaw or an item's prompt text is stored under.
_BENCHMARK_FIELD = (
    r"true_answer|flawed_answer|distractors|answer_key|planted_flaw|registered_answers?"
    r"|move_markers|wrong_path_markers|elicitation|elicitation_placebo|pressure_placebo"
    r"|transform_notes"
)
_ITEM_ID_FIELD = r"item_id|id"

BENCHMARK_ITEM_ID_KEY = re.compile(r'"(?:%s)"\s*:' % _ITEM_ID_FIELD)
BENCHMARK_FIELD_KEY = re.compile(r'"(?:%s)"\s*:' % _BENCHMARK_FIELD)


@dataclass(frozen=True)
class Finding:
    """Keep secret matches reportable without exposing their contents."""

    path: str
    line_number: int
    detector: str
    match_fingerprint: str

    def __str__(self) -> str:
        """Format a safe, non-secret finding description."""
        return f"{self.path}:{self.line_number}: {self.detector} (sha256:{self.match_fingerprint})"


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


@dataclass(frozen=True)
class Detector:
    """One named pattern, plus the placeholders that must not fire.

    Every capture group holds the part of a match that decides whether it is real: the account
    field of an ARN, the user component of a home path. AWS reserves account ids like
    ``123456789012`` for documentation and this repo's own tests use three of them, so without the
    exemption the gate would report a standing set of findings that are not leaks -- and a gate with
    standing findings is one people wave through, which is how the real one gets missed.
    """

    name: str
    pattern: re.Pattern[str]
    placeholders: frozenset[str] = frozenset()

    def findings_in(self, path: str, line_number: int, line: str) -> list[Finding]:
        """Every match on one line that is not a documented placeholder."""
        return [
            Finding(path, line_number, self.name, _fingerprint(match.group(0)))
            for match in self.pattern.finditer(line)
            if not self._is_placeholder(match)
        ]

    def _is_placeholder(self, match: re.Match[str]) -> bool:
        """Whether any group this pattern captured names a documented placeholder.

        Every group is checked rather than one nominated index, because a pattern that reads the
        same value on either side of a keyword needs two alternative groups and only one of them
        matches -- a nominated index would read ``None`` for the other branch and let it through.
        """
        return any(group in self.placeholders for group in match.groups() if group is not None)


# Split from its suffix so this file does not match its own detector: the sweep reads this source.
_PEM_HEADER = "-----BEGIN "

# Shape-based, not entropy-based: entropy scoring a log full of token IDs is all noise.
DETECTORS: tuple[Detector, ...] = (
    Detector("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    Detector("aws_secret_access_key", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*\S{20,}")),
    Detector("aws_session_token", re.compile(r"(?i)aws_session_token\s*[=:]\s*\S{50,}")),
    Detector("pem_private_key", re.compile(_PEM_HEADER + r"(?:[A-Z ]+ )?PRIVATE KEY-----")),
    Detector("ssh_private_key", re.compile(_PEM_HEADER + r"OPENSSH PRIVATE KEY-----")),
    Detector("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    Detector("bearer_token", re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+\S{16,}")),
    # Any `*session=` / `*sid=` cookie, rather than a list naming this employer's SSO cookie.
    Detector(
        "session_cookie",
        re.compile(r"(?i)\b[a-z0-9_-]*(?:session|sid)\s*=\s*[A-Za-z0-9%._-]{24,}"),
    ),
    Detector(
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|passwd|password|token)\s*[=:]\s*['\"]?[A-Za-z0-9/+_-]{20,}"
        ),
    ),
    Detector(
        "email_address",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@(?!"
            + _RESERVED_EMAIL_DOMAIN
            + r"\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
    ),
    Detector(
        "aws_account_id",
        re.compile(_ACCOUNT_KEYWORDS + r"[_-]?(?:id)?\W{0,6}(\d{12})\b"),
        placeholders=EXAMPLE_AWS_ACCOUNT_IDS,
    ),
    Detector(
        "aws_account_id_beside_region",
        re.compile(
            r"(?i)(?:%s\W{0,4}\b(\d{12})\b|\b(\d{12})\b\W{0,4}%s)" % (_AWS_REGION, _AWS_REGION)
        ),
        placeholders=EXAMPLE_AWS_ACCOUNT_IDS,
    ),
    Detector(
        "aws_arn",
        re.compile(r"\barn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:(\d{12}):"),
        placeholders=EXAMPLE_AWS_ACCOUNT_IDS,
    ),
    Detector(
        "ecr_registry",
        re.compile(r"\b(\d{12})\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com"),
        placeholders=EXAMPLE_AWS_ACCOUNT_IDS,
    ),
    Detector(
        "home_directory_path",
        re.compile(r"(?:/local)?/home/([A-Za-z][A-Za-z0-9._-]*)"),
        placeholders=PLACEHOLDER_HOME_USERS,
    ),
    Detector(
        "benchmark_item_record",
        re.compile(
            r'^(?=.*"(?:%s)"\s*:)(?=.*"(?:%s)"\s*:).*$' % (_ITEM_ID_FIELD, _BENCHMARK_FIELD)
        ),
    ),
)


@dataclass(frozen=True)
class CanaryToken:
    """One exact value the shape detectors cannot describe, plus how to recognize it.

    ``contexts`` are the surroundings that make a match legitimate rather than a leak -- a public
    product identifier an employer's name is part of, say. Without them a short word with ordinary
    public uses hands the gate a standing set of findings, and a gate with standing findings gets
    waved through. Findings fingerprint the token rather than the matched text, so a
    case-insensitive match still correlates with every other hit on the same value.
    """

    token: str
    pattern: re.Pattern[str]
    detector: str = "canary_token"
    contexts: tuple[str, ...] = ()

    def findings_in(self, path: str, line_number: int, line: str) -> list[Finding]:
        """Every match on one line that no ``unless=`` context excuses."""
        return [
            Finding(path, line_number, self.detector, _fingerprint(self.token))
            for match in self.pattern.finditer(line)
            if not self._excused(line, match)
        ]

    def _excused(self, line: str, match: re.Match[str]) -> bool:
        """Whether a documented legitimate surrounding sits within reach of this match."""
        window = line[
            max(0, match.start() - CANARY_CONTEXT_CHARS) : match.end() + CANARY_CONTEXT_CHARS
        ].lower()
        return any(context in window for context in self.contexts)


def literal_canary(
    token: str, contexts: tuple[str, ...] = (), detector: str = "canary_token"
) -> CanaryToken:
    """Build a token matched as a literal substring, exactly as written."""
    return CanaryToken(
        token=token,
        pattern=re.compile(re.escape(token)),
        detector=detector,
        contexts=contexts,
    )


def bounded_canary(
    token: str, contexts: tuple[str, ...] = (), detector: str = "canary_token"
) -> CanaryToken:
    """Build a token matched case-insensitively between word boundaries."""
    return CanaryToken(
        token=token,
        pattern=re.compile(r"(?i)\b" + re.escape(token) + r"\b"),
        detector=detector,
        contexts=contexts,
    )


def load_canary_tokens(path: str | None) -> tuple[CanaryToken, ...]:
    """One entry per line; blanks and # comments ignored.

    A bare line is a literal substring of at least ``MIN_CANARY_TOKEN_LENGTH`` characters; a
    ``word:`` prefix matches case-insensitively between word boundaries and so admits values down to
    ``MIN_BOUNDED_CANARY_TOKEN_LENGTH``; either takes a trailing ``unless=comma,separated`` list of
    surroundings that excuse a match. Why each floor is where it is, and why a short token needs the
    bounded form to be armable at all: ``canary/README.md`` and ``docs/privacy-gate.md``.
    """
    if path is None:
        return ()
    tokens = []
    with Path(path).open(encoding="utf-8") as fh:
        for raw in fh:
            entry = raw.strip()
            if not entry or entry.startswith("#"):
                continue
            tokens.append(_parse_canary_entry(entry, path))
    return tuple(tokens)


def _parse_canary_entry(entry: str, path: str) -> CanaryToken:
    """Turn one non-comment line of a canary file into its token."""
    body, separator, raw_contexts = entry.partition(CANARY_CONTEXT_SEPARATOR)
    contexts = tuple(
        context.strip().lower() for context in raw_contexts.split(",") if context.strip()
    )
    if separator and not contexts:
        raise ValueError(f"'{CANARY_CONTEXT_SEPARATOR.strip()}' with no contexts after it: {path}")
    body = body.strip()
    bounded = body.startswith(BOUNDED_CANARY_PREFIX)
    token = body[len(BOUNDED_CANARY_PREFIX) :].strip() if bounded else body
    floor = MIN_BOUNDED_CANARY_TOKEN_LENGTH if bounded else MIN_CANARY_TOKEN_LENGTH
    if len(token) < floor:
        form = "word-bounded " if bounded else ""
        raise ValueError(
            f"{form}canary token too short to be useful ({len(token)} chars, floor {floor}): {path}"
        )
    if bounded:
        return bounded_canary(token, contexts)
    return literal_canary(token, contexts)


def scan_text(path: str, text: str, canaries: tuple[CanaryToken, ...]) -> list[Finding]:
    """Find secret-shaped matches while recording only safe fingerprints."""
    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for detector in DETECTORS:
            findings.extend(detector.findings_in(path, line_number, line))
        for canary in canaries:
            findings.extend(canary.findings_in(path, line_number, line))
    return findings


def benchmark_item_file_findings(path: str, text: str) -> list[Finding]:
    """Flag an item file or a note quoting one, which no line-oriented detector can see.

    A pretty-printed item has one key per line, so no single line carries both the item id and a
    registered-answer field and ``benchmark_item_record`` is blind to exactly the file whose
    publication would contaminate the benchmark. JSON and markdown are checked, Python deliberately
    is not (``test_a_pretty_printed_dict_in_python_stays_silent`` pins the exclusion), and the
    fingerprint covers the whole file rather than any field. Why each of those three, and what was
    measured before markdown was added: ``docs/privacy-gate.md``.
    """
    if not path.endswith((".json", ".md")):
        return []
    if not (BENCHMARK_ITEM_ID_KEY.search(text) and BENCHMARK_FIELD_KEY.search(text)):
        return []
    return [Finding(path, 1, "benchmark_item_file", _fingerprint(text))]


# The two local gitignored places instrument item text may live; absent, the detector is inert.
INSTRUMENT_DATA_DIR = "games/data/survey"
INSTRUMENT_RETRIEVAL_NOTE = "docs/scratch/survey-instrument-items-2026-08-18.md"

# Grain and exemption rationale for everything below: docs/privacy-gate.md.
INSTRUMENT_SHINGLE_WORDS = 5
_INSTRUMENT_MIN_STEM_WORDS = 2
_INSTRUMENT_MIN_ANCHOR_WORDS = 3

_INSTRUMENT_WORD = re.compile(r"[a-z0-9']+")
_NOTE_ITEM_LINE = re.compile(r"^\s*\d+\.\s+(\S.*)$")
_NOTE_PAYOFF_PAIR = re.compile(r"(\d{2,})\s*/\s*(\d{2,})")
_DIGIT_RUN = re.compile(r"\d+")

# Phrases excluded everywhere because they cannot identify an item; ledger in docs/privacy-gate.md.
_GENERIC_SURVEY_SHINGLES = frozenset(
    {
        # Survey-instruction boilerplate.
        "there are no right or",
        "are no right or wrong",
        "no right or wrong answers",
        # Generic matrix-game and instruction prose.
        "two parties choose at the",
        "parties choose at the same",
        "choose at the same time",
        "split evenly between the two",
        "describe in your own words",
        # Added 2026-08-22 with the seven authored families; each verified already tracked.
        "all of it",
        "half of it",
        # Simultaneity and hidden-move framing, as every prompt in games/prompts.py words it.
        "decide at the same moment",
        "at the same moment without",
        "the same moment without seeing",
        "same moment without seeing each",
        "at the same time and",
        # "no way to tell" in its ordinary English senses, about unobservability rather than items.
        "so there is no way",
        "there is no way to",
        "is no way to tell",
        "have no way to tell",
        "you have no way to",
        "it says nothing about who",
        # Two-party and one-sided-allocation framing: names a game's shape, not which item asked.
        "and the two of you",
        "between the two of you",
        "they have no say in",
        "you decide how much of",
        "the other party in the",
        "the same thing as the",
        # Added 2026-08-24: how `games.prompts._outcome_block` words a zero payoff cell.
        "you are credited 0 points",
    }
)

# Round pairs up to this bound collide with ordinary game arithmetic; docs/privacy-gate.md.
_ROUND_PAYOFF_BOUND = 200
# One matched pair could be a coincidence of adjacent numbers; an item is recognized by two.
_MIN_MATCHABLE_PAIRS = 2
# ... within this many digit runs, which is what tells a pasted table from coincidence.
_PAYOFF_WINDOW_RUNS = 200
# Longer runs are blobs, and an unbounded one trips CPython's int-conversion limit.
_MAX_PAYOFF_DIGITS = 7
# An allocation payoff is exactly (payoff to me, payoff to the other party).
_PAYOFF_PAIR_LENGTH = 2


@dataclass(frozen=True)
class InstrumentSources:
    """Everything the instrument-text detector knows, read from local gitignored files.

    ``phrases`` are normalized word shingles (or whole short items); ``payoff_items`` is one
    frozenset of (self, other) integer pairs per allocation item, already filtered to pairs
    distinctive enough to match on; ``source_paths`` are the resolved files the material came
    from, which are exempt from their own findings.
    """

    phrases: frozenset[str]
    payoff_items: tuple[frozenset[tuple[int, int]], ...]
    source_paths: tuple[str, ...]

    @property
    def is_armed(self) -> bool:
        """Report whether the detector has anything at all to match against."""
        return bool(self.phrases or self.payoff_items)


def _phrase_words(text: str) -> list[str]:
    return _INSTRUMENT_WORD.findall(text.lower().replace("\u2019", "'"))


def _prose_phrase(words: list[str]) -> bool:
    """Keep a phrase only if it is prose: item text has at most one number in any five words.

    Number-heavy shingles come from the sources' citation and ledger lines ("Psychological Reports
    90(1), 31-34"), and a citation is exactly what tracked code is allowed to carry -- the specs
    track each instrument's citation on purpose. Without this filter the gate flags the citation
    as if it were the item, and a standing false finding is how a gate gets waved through.
    """
    return sum(1 for word in words if word.isdigit()) <= 1


def _stem_phrases(text: str) -> frozenset[str]:
    """Normalize one item stem into its matchable phrases, generic shingles excluded globally."""
    words = _phrase_words(text)
    if len(words) < _INSTRUMENT_MIN_STEM_WORDS:
        return frozenset()
    if len(words) < INSTRUMENT_SHINGLE_WORDS:
        phrase = " ".join(words)
        if not _prose_phrase(words) or phrase in _GENERIC_SURVEY_SHINGLES:
            return frozenset()
        return frozenset({phrase})
    return (
        frozenset(
            " ".join(shingle)
            for shingle in (
                words[start : start + INSTRUMENT_SHINGLE_WORDS]
                for start in range(len(words) - INSTRUMENT_SHINGLE_WORDS + 1)
            )
            if _prose_phrase(shingle)
        )
        - _GENERIC_SURVEY_SHINGLES
    )


def _anchor_phrases(text: str) -> frozenset[str]:
    """Normalize one anchor label; one- and two-word anchors are too generic to match on."""
    words = _phrase_words(text)
    if len(words) < _INSTRUMENT_MIN_ANCHOR_WORDS:
        return frozenset()
    return _stem_phrases(text)


def _instruction_phrases(text: str) -> frozenset[str]:
    return _stem_phrases(text)


def _qualifying_pair(pair: tuple[int, int]) -> bool:
    mine, theirs = pair
    return not (
        mine % 10 == 0
        and mine <= _ROUND_PAYOFF_BOUND
        and theirs % 10 == 0
        and theirs <= _ROUND_PAYOFF_BOUND
    )


def _payoff_item(pairs: list[tuple[int, int]]) -> frozenset[tuple[int, int]] | None:
    """Keep an allocation item only if it has two distinctive pairs to be recognized by."""
    qualifying = frozenset(pair for pair in set(pairs) if _qualifying_pair(pair))
    return qualifying if len(qualifying) >= _MIN_MATCHABLE_PAIRS else None


# Each phrase-bearing schema field with its own generality rule; per-field why in docs/privacy-gate.md.
_PHRASE_FIELDS = {
    "stem": _stem_phrases,
    "stem_swapped": _stem_phrases,
    "neutral_stems": _stem_phrases,
    # A family's shared closing question: item text on a stem's footing, and an object, not a string.
    "elicitation_blocks": _stem_phrases,
    "anchors": _anchor_phrases,
    "options": _anchor_phrases,
    "instructions": _instruction_phrases,
}


def _field_phrases(key: str, value: object) -> frozenset[str]:
    """Normalize one phrase-bearing field's value: one string, a list of them, or a map to them.

    The mapping case is not decoration. `_walk_instrument_json` stops recursing into any key this
    dict recognizes, so before this a phrase field holding an object contributed nothing at all: the
    walker declined to descend and the flattening only unpacked lists. The field would read as zero
    phrases and the guard would then permit its words in a tracked file while reporting success.
    """
    phraser = _PHRASE_FIELDS.get(key)
    if phraser is None:
        return frozenset()
    if isinstance(value, dict):
        entries: list[object] = list(value.values())
    elif isinstance(value, list):
        entries = list(value)
    else:
        entries = [value]
    collected: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            collected.update(phraser(entry))
    return frozenset(collected)


def sources_from_json_payload(
    payload: object,
) -> tuple[frozenset[str], tuple[frozenset[tuple[int, int]], ...]]:
    """Extract phrases and payoff items from an instrument file's parsed JSON.

    Walks the whole structure rather than the documented schema, on purpose: a schema drift must
    not quietly shrink what this detector knows, so any ``stem``, ``neutral_stems``, ``anchors``,
    ``instructions`` or ``option_payoffs`` field anywhere in the file contributes.
    """
    phrases: set[str] = set()
    payoff_items: list[frozenset[tuple[int, int]]] = []
    _walk_instrument_json(payload, phrases, payoff_items)
    return frozenset(phrases), tuple(payoff_items)


def _walk_instrument_json(
    node: object, phrases: set[str], payoff_items: list[frozenset[tuple[int, int]]]
) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            phrases.update(_field_phrases(key, value))
            if key == "option_payoffs" and isinstance(value, list):
                item = _payoff_item(_json_payoff_pairs(value))
                if item is not None:
                    payoff_items.append(item)
            elif key not in _PHRASE_FIELDS:
                _walk_instrument_json(value, phrases, payoff_items)
    elif isinstance(node, list):
        for entry in node:
            _walk_instrument_json(entry, phrases, payoff_items)


def _json_payoff_pairs(value: Sequence[object]) -> list[tuple[int, int]]:
    """Read whatever two-integer pairs this ``option_payoffs`` list carries, ignoring the rest."""
    pairs: list[tuple[int, int]] = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != _PAYOFF_PAIR_LENGTH:
            continue
        mine, theirs = pair[0], pair[1]
        if isinstance(mine, bool) or isinstance(theirs, bool):
            continue
        if isinstance(mine, int) and isinstance(theirs, int):
            pairs.append((mine, theirs))
    return pairs


def sources_from_note_text(
    text: str,
) -> tuple[frozenset[str], tuple[frozenset[tuple[int, int]], ...]]:
    """Extract phrases and payoff items from the retrieval note's markdown.

    The note's structure is stable by construction: item text sits on numbered-list lines, and
    allocation payoffs sit in markdown table rows as ``self / other`` pairs. Numbered lines that
    are not items (the not-obtained ledger) contribute harmless extra phrases that match nothing.
    """
    phrases: set[str] = set()
    payoff_items: list[frozenset[tuple[int, int]]] = []
    for line in text.splitlines():
        matched = _NOTE_ITEM_LINE.match(line)
        if matched:
            phrases.update(_stem_phrases(matched.group(1)))
        if line.lstrip().startswith("|"):
            pairs = [(int(mine), int(theirs)) for mine, theirs in _NOTE_PAYOFF_PAIR.findall(line)]
            item = _payoff_item(pairs)
            if item is not None:
                payoff_items.append(item)
    return frozenset(phrases), tuple(payoff_items)


def collect_instrument_sources(repo_root: Path) -> InstrumentSources:
    """Read every local instrument source this machine has; a fresh clone yields an inert result."""
    phrases: set[str] = set()
    payoff_items: list[frozenset[tuple[int, int]]] = []
    source_paths: list[str] = []
    data_dir = repo_root / INSTRUMENT_DATA_DIR
    if data_dir.is_dir():
        for json_path in sorted(data_dir.glob("*.json")):
            file_phrases, file_items = sources_from_json_payload(
                json.loads(json_path.read_text(encoding="utf-8"))
            )
            phrases.update(file_phrases)
            payoff_items.extend(file_items)
            source_paths.append(str(json_path.resolve()))
    note_path = repo_root / INSTRUMENT_RETRIEVAL_NOTE
    if note_path.is_file():
        note_phrases, note_items = sources_from_note_text(note_path.read_text(encoding="utf-8"))
        phrases.update(note_phrases)
        payoff_items.extend(note_items)
        source_paths.append(str(note_path.resolve()))
    return InstrumentSources(
        phrases=frozenset(phrases),
        payoff_items=tuple(payoff_items),
        source_paths=tuple(source_paths),
    )


def instrument_text_findings(path: str, text: str, sources: InstrumentSources) -> list[Finding]:
    """Flag instrument item text wherever it appears, across line breaks included.

    Whole-file and word-normalized rather than line-oriented, because the one fragment that
    actually reached a tracked docstring wrapped across a line break, where every line detector
    is blind. Findings carry a fingerprint of the matched phrase, never the phrase: this scanner's
    rule that a finding must not republish what it found applies with extra force when the found
    thing is the item text itself.
    """
    if not sources.is_armed or str(Path(path).resolve()) in sources.source_paths:
        return []
    findings: dict[str, Finding] = {}
    _collect_phrase_findings(path, text, sources, findings)
    _collect_payoff_findings(path, text, sources, findings)
    return sorted(findings.values(), key=lambda finding: finding.line_number)


def _collect_phrase_findings(
    path: str, text: str, sources: InstrumentSources, findings: dict[str, Finding]
) -> None:
    words: list[tuple[str, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        words.extend(
            (matched.group(0), line_number)
            for matched in _INSTRUMENT_WORD.finditer(line.lower().replace("\u2019", "'"))
        )
    for length in sorted({len(phrase.split()) for phrase in sources.phrases}):
        for start in range(len(words) - length + 1):
            shingle = " ".join(word for word, _ in words[start : start + length])
            if shingle in sources.phrases:
                fingerprint = _fingerprint(shingle)
                findings.setdefault(
                    fingerprint,
                    Finding(path, words[start][1], "instrument_item_text", fingerprint),
                )


def _collect_payoff_findings(
    path: str, text: str, sources: InstrumentSources, findings: dict[str, Finding]
) -> None:
    digits: list[tuple[int, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        digits.extend(
            (int(run.group(0)), line_number)
            for run in _DIGIT_RUN.finditer(line)
            if len(run.group(0)) <= _MAX_PAYOFF_DIGITS
        )
    hits_by_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index in range(len(digits) - 1):
        first, first_line = digits[index]
        second, _ = digits[index + 1]
        hits_by_pair.setdefault((first, second), []).append((index, first_line))
    for item in sources.payoff_items:
        line = _clustered_match_line(item, hits_by_pair)
        if line is not None:
            fingerprint = _fingerprint(repr(sorted(item)))
            findings.setdefault(
                fingerprint, Finding(path, line, "instrument_item_payoffs", fingerprint)
            )


def _clustered_match_line(
    item: frozenset[tuple[int, int]],
    hits_by_pair: dict[tuple[int, int], list[tuple[int, int]]],
) -> int | None:
    """Return the line where enough of one item's pairs co-occur, or None.

    Enough means ``_MIN_MATCHABLE_PAIRS`` *distinct* pairs within ``_PAYOFF_WINDOW_RUNS`` digit
    runs of each other -- the shape of a pasted table, which is consecutive pairs. Without the
    window, an item whose payoffs are small two-digit numbers matched dependency lockfiles and
    arithmetic notebooks on pairs tens of thousands of runs apart.
    """
    hits = sorted(
        (index, line, pair)
        for pair, pair_hits in hits_by_pair.items()
        if pair in item
        for index, line in pair_hits
    )
    start = 0
    window: dict[tuple[int, int], int] = {}
    for end, (index, _line, pair) in enumerate(hits):
        window[pair] = window.get(pair, 0) + 1
        while index - hits[start][0] > _PAYOFF_WINDOW_RUNS:
            evicted = hits[start][2]
            window[evicted] -= 1
            if window[evicted] == 0:
                del window[evicted]
            start += 1
        if len(window) >= _MIN_MATCHABLE_PAIRS:
            return min(line for _index, line, _pair in hits[start : end + 1])
    return None


def iter_target_files(targets: list[str]) -> list[str]:
    """Expand scan targets into deterministic file paths for complete coverage."""
    files = []
    for target in targets:
        if Path(target).is_file():
            files.append(target)
            continue
        if not Path(target).is_dir():
            raise FileNotFoundError(f"scan target does not exist: {target}")
        for root, _dirs, names in os.walk(target):
            files.extend(str(Path(root) / name) for name in names)
    return sorted(files)


def scan_paths(
    targets: list[str],
    canaries: tuple[CanaryToken, ...],
    max_bytes: int,
    instrument_sources: InstrumentSources | None = None,
) -> list[Finding]:
    """Scan every target without silently skipping oversized files."""
    findings = []
    for path in iter_target_files(targets):
        size = Path(path).stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"{path} is {size} bytes, over the {max_bytes} limit. Raise --max-bytes "
                "deliberately rather than letting the gate skip a file silently."
            )
        with Path(path).open(encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        findings.extend(scan_text(path, text, canaries))
        findings.extend(benchmark_item_file_findings(path, text))
        if instrument_sources is not None:
            findings.extend(instrument_text_findings(path, text, instrument_sources))
    return findings


def main() -> int:
    """Scan requested paths and fail when sensitive text is found."""
    parser = argparse.ArgumentParser(
        description="Fail if any file contains secret-shaped text or a known canary token."
    )
    parser.add_argument("targets", nargs="+", help="files or directories to scan")
    parser.add_argument(
        "--canary-file",
        help="file of exact canary token strings, one per line, to match literally",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="refuse a file larger than this rather than skipping it (default: 64MiB)",
    )
    parser.add_argument(
        "--require-canary",
        action="store_true",
        help=f"exit {EXIT_UNARMED} when no canary token is armed, rather than scanning shape-only",
    )
    parser.add_argument(
        "--require-instrument-sources",
        action="store_true",
        help=f"exit {EXIT_UNARMED} when no local instrument source is armed",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="scan-secrets: %(message)s", stream=sys.stderr)
    repo_root = Path(__file__).resolve().parents[1]
    canaries = load_canary_tokens(args.canary_file)
    instrument_sources = collect_instrument_sources(repo_root)
    unarmed = _report_arming(canaries, instrument_sources, args)
    findings = scan_paths(args.targets, canaries, args.max_bytes, instrument_sources)

    if findings:
        for finding in findings:
            logger.error(str(finding))
        detectors = sorted({f.detector for f in findings})
        logger.error(
            f"{len(findings)} finding(s) across detectors {detectors}. "
            "Do NOT upload or train on this data until each is resolved."
        )
        return EXIT_FINDINGS
    if unarmed:
        return EXIT_UNARMED
    logger.info(f"clean: no secret-shaped text in {len(iter_target_files(args.targets))} file(s)")
    return 0


def _report_arming(
    canaries: tuple[CanaryToken, ...],
    instrument_sources: InstrumentSources,
    args: argparse.Namespace,
) -> bool:
    """Log what each data-driven detector loaded; report whether a required one loaded nothing.

    An unarmed detector calls every value it was meant to guard clean and exits 0, which reads
    exactly like a tree that carries none of them -- so the counts are logged on every run, an
    unarmed family warns even when nothing required it, and ``--require-*`` turns that warning into
    a refusal for the callers (this repo's own gates) that know the sources are supposed to be here.
    """
    logger.info(f"{len(DETECTORS)} detectors, {len(canaries)} canary token(s)")
    logger.info(
        f"instrument text: {len(instrument_sources.phrases)} phrase(s), "
        f"{len(instrument_sources.payoff_items)} payoff item(s) from "
        f"{len(instrument_sources.source_paths)} local source file(s)"
    )
    unarmed = False
    if not canaries:
        logger.warning(
            "canary detector is INERT: no token file, so every username, bucket, profile name and "
            "employer reference in the scanned text reads as clean. See canary/README.md."
        )
        unarmed = unarmed or args.require_canary
    if not instrument_sources.is_armed:
        logger.warning(
            "instrument-text detector is INERT: no local item sources, so every survey stem and "
            "payoff table in the scanned text reads as clean. See canary/README.md."
        )
        unarmed = unarmed or args.require_instrument_sources
    if unarmed:
        logger.error(
            "REFUSING to report a pass: a detector family required by this invocation is unarmed, "
            f"and an unarmed family cannot tell a clean tree from an unread one. Exit {EXIT_UNARMED}"
            " means 'did not verify', not 'clean'."
        )
    return unarmed


if __name__ == "__main__":
    sys.exit(main())
