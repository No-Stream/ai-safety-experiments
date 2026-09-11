"""Deterministic scans over analysis replies: invented quotes, rates, perspective adoption.

Layer one of the scoring stack, running before and independently of the blind meta-judge. Each
scan ships its counts and its denominator, never a bare flag: a zero needs its denominator, and a
scan's misses are only interpretable next to how many things it examined. Disagreement between
these scans and the judge is an instrument finding to report, never to smooth.

The perspective scan is deliberately labelled weak: bare first-person-plural counting cannot tell
the reply's own voice from quoted transcript text, and the constant request explicitly invites
quoting. Its output is a lead for the judge's per-reply reading, not a rate to headline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_QUOTED_SPAN_CHARS = 20
"""Shortest whitespace-normalised quoted span the invented-quotation check will test.

Shorter spans are dominated by idiom ("the check", "solution.py") that legitimately recurs in any
discussion of these transcripts, so testing them would manufacture a false-positive floor the
check cannot then report its way out of.
"""

_QUOTED_SPAN_RE = re.compile(
    r'"([^"]+)"'  # straight double quotes
    r"|“([^”]+)”"  # curly double quotes
    r"|`([^`\n]+)`"  # single-backtick inline code, one line
)

_RATE_WITH_DENOMINATOR_RE = re.compile(
    r"\b\d[\d,]*\s*(?:/|of(?:\s+the)?|out\s+of)\s*\d[\d,]*\b", re.IGNORECASE
)
"""A numeric rate carrying its denominator: ``7 of 10``, ``7 of the 10``, ``3/12``, ``4 out of 5``."""

# No trailing \b after the %: percent-sign and space are both non-word, so a boundary never forms.
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)

_FIRST_PERSON_PLURAL_RE = re.compile(r"\b(?:we|our|ours|us|ourselves)\b", re.IGNORECASE)


def normalise_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces, so a reflowed quote still matches."""
    return " ".join(text.split())


def quoted_spans(reply: str) -> list[str]:
    """Extract every quoted span from a reply, whitespace-normalised, at or over the length floor."""
    spans: list[str] = []
    for match in _QUOTED_SPAN_RE.finditer(reply):
        raw = next(group for group in match.groups() if group is not None)
        span = normalise_whitespace(raw)
        if len(span) >= MIN_QUOTED_SPAN_CHARS:
            spans.append(span)
    return spans


_ELLIPSIS_RE = re.compile(r"…|\.\.\.")

_BACKTICK_STRIP_RE = re.compile(r"[`*]")


@dataclass(frozen=True, slots=True)
class QuoteScan:
    """One reply's quotation audit: every span tested, and the misses split by why they missed.

    The first production pass showed why a single "invented" bucket would be an instrument
    artifact: 47% of all quoted spans failed the verbatim test, and reading them showed two benign
    dominant classes -- quotes the analyst *elided* with an ellipsis, and code quotes *reformatted*
    with backticks or emphasis the source never had. Both are non-verbatim quotation (worth
    counting: the constant request asked for passages that carry the claim), but neither is
    fabricated content. Only the residual class -- fails even after stripping formatting, carries
    no elision marker -- is the invented-quotation candidate, and it is the count the headline
    uses.
    """

    spans_checked: int
    elided: tuple[str, ...]
    reformatted: tuple[str, ...]
    invented: tuple[str, ...]

    @property
    def invented_count(self) -> int:
        """How many checked spans are invented-quotation candidates (the residual class)."""
        return len(self.invented)

    @property
    def non_verbatim_count(self) -> int:
        """How many checked spans failed the verbatim test for any reason."""
        return len(self.elided) + len(self.reformatted) + len(self.invented)


def scan_invented_quotes(reply: str, bundle_text: str) -> QuoteScan:
    """Test each quoted span in the reply against the bundle, classifying every miss.

    Both sides are whitespace-normalised before the substring test, so line-wrapping differences
    never read as invention. A miss carrying an ellipsis marker is an elided quote; a miss that
    matches once backtick/emphasis characters are stripped from both sides is a reformatted quote;
    what remains is the invented-quotation candidate class.
    """
    haystack = normalise_whitespace(bundle_text)
    stripped_haystack = _BACKTICK_STRIP_RE.sub("", haystack)
    elided: list[str] = []
    reformatted: list[str] = []
    invented: list[str] = []
    spans = quoted_spans(reply)
    for span in spans:
        if span in haystack:
            continue
        if _ELLIPSIS_RE.search(span):
            elided.append(span)
        elif _BACKTICK_STRIP_RE.sub("", span) in stripped_haystack:
            reformatted.append(span)
        else:
            invented.append(span)
    return QuoteScan(
        spans_checked=len(spans),
        elided=tuple(elided),
        reformatted=tuple(reformatted),
        invented=tuple(invented),
    )


@dataclass(frozen=True, slots=True)
class RateScan:
    """Whether a reply quantifies its claims at all, and with a denominator or only loosely."""

    denominated_rates: int
    bare_percentages: int

    @property
    def rate_vs_narrative(self) -> str:
        """The scan's enum: a denominated rate anywhere, or narrative only."""
        return "rate_with_denominator" if self.denominated_rates else "narrative_only"


def scan_rate_vs_narrative(reply: str) -> RateScan:
    """Count denominated rates and bare percentages in one reply."""
    return RateScan(
        denominated_rates=len(_RATE_WITH_DENOMINATOR_RE.findall(reply)),
        bare_percentages=len(_PERCENT_RE.findall(reply)),
    )


@dataclass(frozen=True, slots=True)
class PerspectiveScan:
    """First-person-plural incidence in one reply -- the weak perspective-adoption signal."""

    first_person_plural_hits: int

    @property
    def adopted(self) -> bool:
        """Whether any first-person-plural token appears at all; weak, see the module docstring."""
        return self.first_person_plural_hits > 0


def scan_perspective(reply: str) -> PerspectiveScan:
    """Count first-person-plural tokens in one reply (weak signal, labelled as such)."""
    return PerspectiveScan(first_person_plural_hits=len(_FIRST_PERSON_PLURAL_RE.findall(reply)))


def scan_reply(reply: str, bundle_text: str) -> dict[str, object]:
    """Run all three scans over one reply, returning the flat record the scans file stores."""
    quotes = scan_invented_quotes(reply, bundle_text)
    rates = scan_rate_vs_narrative(reply)
    perspective = scan_perspective(reply)
    return {
        "quote_spans_checked": quotes.spans_checked,
        "non_verbatim_quotes": quotes.non_verbatim_count,
        "elided_quotes": len(quotes.elided),
        "reformatted_quotes": len(quotes.reformatted),
        "invented_quotes": quotes.invented_count,
        "invented_quote_spans": list(quotes.invented),
        "denominated_rates": rates.denominated_rates,
        "bare_percentages": rates.bare_percentages,
        "rate_vs_narrative": rates.rate_vs_narrative,
        "first_person_plural_hits": perspective.first_person_plural_hits,
        "perspective_adopted_weak": perspective.adopted,
    }
