"""Turn a raw completion into an action, a split, a level, an amount sent, or a decision theory.

This module sits directly in the training loop: what it counts as an action becomes the
reward, so a lenient bug here silently trains the wrong thing. Two rules keep that honest.

First, **parsing never raises on model output.** Any completion that does not carry a
well-formed answer returns None, and the caller turns that into the explicit parse penalty.
Raises are reserved for caller bugs -- a `coop_label` that is not one of the two labels means
the corpus is corrupt, not that the model wrote something odd.

Second, **the last tag wins, and only after the thinking block.** Models restate their choice,
and a choice mentioned inside `<think>` is reasoning rather than an answer, so `strip_thinking`
runs first and cuts everything up to the final `</think>`.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, NamedTuple

from games.payoffs import COOPERATE, DEFECT

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_KEEP_RE = re.compile(r"<keep>(.*?)</keep>", re.IGNORECASE | re.DOTALL)
_CLAIM_RE = re.compile(r"<claim>(.*?)</claim>", re.IGNORECASE | re.DOTALL)
_CONTRIBUTE_RE = re.compile(r"<contribute>(.*?)</contribute>", re.IGNORECASE | re.DOTALL)
_SET_RE = re.compile(r"<set>(.*?)</set>", re.IGNORECASE | re.DOTALL)
_LEVEL_RE = re.compile(r"<level>(.*?)</level>", re.IGNORECASE | re.DOTALL)
_SEND_RE = re.compile(r"<send>(.*?)</send>", re.IGNORECASE | re.DOTALL)
_RETURN_RE = re.compile(r"<return>(.*?)</return>", re.IGNORECASE | re.DOTALL)
_THEORY_RE = re.compile(r"<theory>(.*?)</theory>", re.IGNORECASE | re.DOTALL)
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_TAG_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")

# The `<return>` tag carries a whole-number percentage, so the prompt can state the counterpart's
# rule and the model's own rule in the same units.
RETURN_PERCENTAGE_MAX = 100

_THEORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("FDT", re.compile(r"functional|\bfdt\b", re.IGNORECASE)),
    ("UDT", re.compile(r"updateless|\budt\b", re.IGNORECASE)),
    ("EDT", re.compile(r"evidential|\bedt\b", re.IGNORECASE)),
    ("CDT", re.compile(r"(?<!a)causal|\bcdt\b", re.IGNORECASE)),
)
# FDT and UDT are near-synonyms in this taxonomy rather than rivals: an FDT answer routinely also
# says "updateless", so a tag naming both is one endorsement and resolves to the more specific name.
_ACAUSAL_FAMILY: frozenset[str] = frozenset({"FDT", "UDT"})
# A tag that named two rival theories, which is a different observation from one that named none.
THEORY_AMBIGUOUS = "ambiguous"
THEORY_OTHER = "other"
THEORY_LABELS: tuple[str, ...] = (
    *(label for label, _ in _THEORY_PATTERNS),
    THEORY_OTHER,
    THEORY_AMBIGUOUS,
)


def strip_thinking(text: str, *, prefilled_think: bool = False) -> tuple[str, bool]:
    """Split a completion into its visible answer, and whether the thinking never closed.

    Returns `(visible_text, truncated_thinking)`. A completion whose thinking ran into the
    token budget has no visible answer at all, and reporting that separately matters: a rising
    `truncated_thinking_rate` means the generation cap is shaping behaviour, not the payoffs.

    `prefilled_think` says the chat template already emitted the opening `<think>` as part of
    the prompt, so the completion can only ever contain the closing tag. That is the case for
    the Qwen3.5/3.8 templates with `enable_thinking=True` (verified on this box, 2026-08-17:
    `docs/scratch/qwen38-27b-load-check-2026-08-17.md`); Qwen3-0.6B does not prefill and its
    completions carry both tags. With the flag set, a completion showing neither tag is
    truncated thinking rather than a plain answer.
    """
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1], False
    if THINK_OPEN in text or prefilled_think:
        return "", True
    return text, False


def _require_labels(label_a: str, label_b: str, coop_label: str) -> None:
    """Reject a label triple that cannot express a choice, i.e. a corrupt corpus row."""
    if label_a.strip().casefold() == label_b.strip().casefold():
        raise ValueError(f"label_a and label_b must differ, got {label_a!r} and {label_b!r}.")
    if coop_label not in (label_a, label_b):
        raise ValueError(
            f"coop_label {coop_label!r} must be one of the two labels "
            f"({label_a!r}, {label_b!r}); otherwise no completion can score as cooperation."
        )


def _action_for_label(text: str, *, label_a: str, label_b: str, coop_label: str) -> str | None:
    """Map one tag's contents to a canonical action, or None if it names neither label."""
    written = text.strip().casefold()
    matched = next(
        (label for label in (label_a, label_b) if label.strip().casefold() == written), None
    )
    if matched is None:
        return None
    return COOPERATE if matched == coop_label else DEFECT


def parse_action(visible_text: str, *, label_a: str, label_b: str, coop_label: str) -> str | None:
    """Return the canonical action from the last `<action>` tag, or None if there is no answer.

    Label matching is case-insensitive but exact: a completion that argues its way to a label
    in prose without emitting the tag scores as a parse failure, since inferring intent from
    free text is exactly the judgement we refuse to put anywhere near a reward.
    """
    _require_labels(label_a, label_b, coop_label)
    tags = _ACTION_RE.findall(visible_text)
    if not tags:
        return None
    return _action_for_label(tags[-1], label_a=label_a, label_b=label_b, coop_label=coop_label)


def parse_action_sequence(
    visible_text: str, *, n_rounds: int, label_a: str, label_b: str, coop_label: str
) -> list[str] | None:
    """Return one canonical action per round from the `<action>` tags, in order.

    All-or-nothing: the wrong number of tags, or any tag naming neither label, returns None for
    the whole sequence. Scoring a partial match would mean inventing the missing rounds, and a
    simulated return over invented moves is not a measurement of anything.
    """
    _require_labels(label_a, label_b, coop_label)
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be positive, got {n_rounds}.")
    tags = _ACTION_RE.findall(visible_text)
    if len(tags) != n_rounds:
        return None
    moves = [
        _action_for_label(tag, label_a=label_a, label_b=label_b, coop_label=coop_label)
        for tag in tags
    ]
    if any(move is None for move in moves):
        return None
    return [move for move in moves if move is not None]


class TrustStrategy(NamedTuple):
    """One completion's whole answer under the strategy method: what it sent, and what it promised.

    Both halves or neither, which is why they travel together: a send scored against an invented
    return rule, or a promise attached to an invented send, would be a measurement of this parser
    rather than of the model.
    """

    sent: int
    return_percentage: int


def _bounded_integer(written: str, *, lower: int, upper: int) -> int | None:
    """Read one tag's contents as a whole number inside `[lower, upper]`, or None if unusable.

    Out-of-range and non-integer figures are parse failures rather than clamped values: clamping
    would reward a model for writing an impossible number and hide the failure from the parse-rate
    metrics. `lower` is a parameter rather than a constant zero because the effort grid starts at 1,
    so level 0 is as impossible an answer as level 6 and must fail the same way.
    """
    stripped = written.strip()
    if not _INTEGER_RE.match(stripped):
        return None
    figure = int(stripped)
    if not lower <= figure <= upper:
        return None
    return figure


def _parse_bounded_integer(
    visible_text: str, *, pattern: re.Pattern[str], upper: int, lower: int = 0
) -> int | None:
    """Return the whole number in the last matching tag, or None if it is absent or unusable.

    Shared by every numeric answer format so they cannot drift into different leniency.
    """
    tags = pattern.findall(visible_text)
    if not tags:
        return None
    return _bounded_integer(tags[-1], lower=lower, upper=upper)


def _parse_bounded_integer_sequence(
    visible_text: str, *, pattern: re.Pattern[str], n_expected: int, lower: int, upper: int
) -> list[int] | None:
    """Return one whole number per tag, in order, or None unless every one of them is usable.

    All-or-nothing on both the count and each figure, exactly as `parse_action_sequence` is: a
    simulated match over invented rounds measures nothing, and a match over rounds where one figure
    was quietly dropped measures less than nothing, because the shortfall would land on whichever
    round happened to parse.
    """
    if n_expected < 1:
        raise ValueError(f"n_expected must be positive, got {n_expected}.")
    tags = pattern.findall(visible_text)
    if len(tags) != n_expected:
        return None
    figures = [_bounded_integer(tag, lower=lower, upper=upper) for tag in tags]
    if any(figure is None for figure in figures):
        return None
    return [figure for figure in figures if figure is not None]


def parse_split(visible_text: str, *, endowment: int) -> int | None:
    """Return the units kept from the last `<keep>` tag, or None if it is absent or unusable."""
    if endowment <= 0:
        raise ValueError(f"endowment must be positive, got {endowment}.")
    return _parse_bounded_integer(visible_text, pattern=_KEEP_RE, upper=endowment)


# The two figures a transfer answer can name. The transfer probe asks half its rows for the units set
# down and half for the units kept back, which is its counterbalance in place of a label print order:
# a model that anchors on the number it was asked for then contributes equally to both directions.
# Named here rather than beside the renderers because a polarity IS a tag name, and this module is what
# reads a tag.
ANSWER_POLARITY_SET = "set"
ANSWER_POLARITY_KEEP = "keep"
ANSWER_POLARITIES: tuple[str, str] = (ANSWER_POLARITY_SET, ANSWER_POLARITY_KEEP)


def parse_set_down(visible_text: str, *, endowment: int) -> int | None:
    """Return the units set down from the last `<set>` tag, or None if absent or unusable.

    Out of range is a parse failure and never a clamp, for `parse_split`'s reason: a figure above the
    stock is not a division of anything, and clamping it to the stock would score "set down everything
    and more" as the most generous legal answer instead of counting it where the parse metrics see it.
    """
    if endowment <= 0:
        raise ValueError(f"endowment must be positive, got {endowment}.")
    return _parse_bounded_integer(visible_text, pattern=_SET_RE, upper=endowment)


ANSWER_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    ANSWER_POLARITY_SET: _SET_RE,
    ANSWER_POLARITY_KEEP: _KEEP_RE,
}
"""Which tag each polarity's answer is written in. One table, so a scan asking "did it use the OTHER
tag" and a parser reading the right one cannot come to disagree about what a tag looks like."""


def other_polarity(polarity: str) -> str:
    """Name the polarity a row was NOT asked for, which is the tag a wrong-tag reply used."""
    if polarity not in ANSWER_POLARITIES:
        raise ValueError(f"polarity must be one of {ANSWER_POLARITIES}, got {polarity!r}.")
    return next(other for other in ANSWER_POLARITIES if other != polarity)


def carries_answer_tag(visible_text: str, *, polarity: str) -> bool:
    """Whether the text carries one polarity's tag at all, whatever is inside it.

    Presence rather than a usable figure, because the caller asking this is distinguishing "answered in
    the other tag" from "named no figure at all", and a reply that wrote ``<set>lots</set>`` did answer
    in the wrong tag.
    """
    if polarity not in ANSWER_TAG_PATTERNS:
        raise ValueError(f"polarity must be one of {ANSWER_POLARITIES}, got {polarity!r}.")
    return bool(ANSWER_TAG_PATTERNS[polarity].search(visible_text))


def parse_transfer_figure(visible_text: str, *, endowment: int, polarity: str) -> int | None:
    """Return the units SET DOWN, whichever of the two figures this row's polarity asked for.

    The inversion lives here rather than at each call site, because two call sites -- the eval record
    builder and the probe's own scan -- read the same rows and a second copy of `endowment - kept` is
    how the two would come to report different quantities under one name. A row that answered with the
    other polarity's tag comes back None: the scan counts that separately as a wrong tag, which is a
    different finding from a reply that named no figure at all.
    """
    if polarity == ANSWER_POLARITY_SET:
        return parse_set_down(visible_text, endowment=endowment)
    if polarity == ANSWER_POLARITY_KEEP:
        kept = parse_split(visible_text, endowment=endowment)
        return None if kept is None else endowment - kept
    raise ValueError(f"polarity must be one of {ANSWER_POLARITIES}, got {polarity!r}.")


def parse_claim(visible_text: str, *, windfall: int) -> int | None:
    """Return the units claimed in the last `<claim>` tag, or None if absent or unusable.

    The simultaneous-claim game's answer. A figure above the windfall is a parse failure for the
    same reason an over-large keep is: it is not a division of anything, and clamping it to the
    windfall would score "claim everything and more" as the greediest legal answer instead of
    counting it where the parse metrics can see it.
    """
    if windfall <= 0:
        raise ValueError(f"windfall must be positive, got {windfall}.")
    return _parse_bounded_integer(visible_text, pattern=_CLAIM_RE, upper=windfall)


def parse_contribution(visible_text: str, *, endowment: int) -> int | None:
    """Return the units put towards the shared undertaking from the last `<contribute>` tag.

    None when the tag is absent or the figure is unusable. Out of range is a parse failure and never a
    clamp, for `parse_split`'s reason and with a sharper consequence here: clamping a figure above the
    stock to the stock would pay the model the fund-it-single-handed reward for writing "1000", which is
    exactly the answer this game's two prize variants exist to tell apart.
    """
    if endowment <= 0:
        raise ValueError(f"endowment must be positive, got {endowment}.")
    return _parse_bounded_integer(visible_text, pattern=_CONTRIBUTE_RE, upper=endowment)


def parse_level(visible_text: str, *, n_levels: int) -> int | None:
    """Return the level from the last `<level>` tag, or None if it is absent or unusable.

    The effort grid runs from 1, so 0 is out of range and fails like any impossible figure. That
    matters more here than elsewhere: 0 is the natural thing a model writes for "I do nothing", and
    clamping it up to 1 would silently record the bottom of the grid as a deliberate answer.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be positive, got {n_levels}.")
    return _parse_bounded_integer(visible_text, pattern=_LEVEL_RE, lower=1, upper=n_levels)


def parse_level_sequence(visible_text: str, *, n_rounds: int, n_levels: int) -> list[int] | None:
    """Return one level per round from the `<level>` tags, in order, or None unless all parse."""
    if n_levels < 1:
        raise ValueError(f"n_levels must be positive, got {n_levels}.")
    return _parse_bounded_integer_sequence(
        visible_text, pattern=_LEVEL_RE, n_expected=n_rounds, lower=1, upper=n_levels
    )


def parse_send(visible_text: str, *, endowment: int) -> int | None:
    """Return the units handed across from the last `<send>` tag, or None if unusable.

    Out of range is a parse failure and never a clamp, for `parse_split`'s reason: the reward here
    is affine in the amount sent, so clamping an impossible number would pay a model the
    send-everything reward for writing "1000" and hide it from the parse-rate metrics -- in the one
    direction the arm is trying to measure.
    """
    if endowment <= 0:
        raise ValueError(f"endowment must be positive, got {endowment}.")
    return _parse_bounded_integer(visible_text, pattern=_SEND_RE, upper=endowment)


def parse_return_percentage(visible_text: str) -> int | None:
    """Return the whole-number percentage from the last `<return>` tag, or None if unusable."""
    return _parse_bounded_integer(visible_text, pattern=_RETURN_RE, upper=RETURN_PERCENTAGE_MAX)


def parse_trust_strategy(visible_text: str, *, endowment: int) -> TrustStrategy | None:
    """Return both halves of a strategy-method answer, or None unless both parse.

    All-or-nothing, exactly as `parse_action_sequence` is: the two tags are one answer, and scoring
    a send against a return rule the completion never stated would mean inventing the rule and then
    grading the model on it.
    """
    sent = parse_send(visible_text, endowment=endowment)
    percentage = parse_return_percentage(visible_text)
    if sent is None or percentage is None:
        return None
    return TrustStrategy(sent=sent, return_percentage=percentage)


def parse_tag(visible_text: str, tag: str, *, vocabulary: Sequence[str]) -> str | None:
    """Return the last `<tag>` whose contents name one of `vocabulary`, or None.

    The generic form of `parse_theory` for closed-vocabulary answers, so a fourth, fifth and sixth
    forced-tag item family cannot each arrive with its own bespoke regex and its own opinion about
    whitespace and case. Matching is case-insensitive and whitespace-stripped but otherwise exact:
    contents naming nothing in the vocabulary are a parse failure, because inferring which word a
    model meant is the judgement this repo keeps out of every scoring path.

    Deliberately unlike `parse_theory`, which classifies unrecognised contents as "other" -- that
    only makes sense where the answer space is genuinely open (which decision theory do you hold).
    A forced tag offers a closed menu, so contents outside it mean the format instruction did not
    land, which is a parse failure and belongs in the parse-failure rate.

    The tag name is checked because it is interpolated into a pattern: a caller passing a tag with
    regex metacharacters would build a pattern that matches something other than the tag it named.
    """
    if not _TAG_NAME_RE.match(tag):
        raise ValueError(
            f"tag {tag!r} must be lowercase letters, digits and hyphens starting with a letter; it "
            f"is interpolated into a regular expression."
        )
    if not vocabulary:
        raise ValueError(f"<{tag}> has an empty vocabulary, so no answer to it could ever match.")
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
    tags = pattern.findall(visible_text)
    if not tags:
        return None
    written = tags[-1].strip().casefold()
    return next((word for word in vocabulary if word.strip().casefold() == written), None)


def parse_theory(visible_text: str) -> str | None:
    """Classify the last `<theory>` tag as CDT, EDT, FDT, UDT, "other", or "ambiguous".

    Returns None only when the tag is missing, which is a parse failure; content that names no
    known theory is a real answer and classifies as "other". String matching, deliberately: an
    LLM judge must never touch anything that becomes a reward or a reported number.

    **Every pattern that matches is collected, not just the first.** Returning on the first match
    over a fixed FDT, UDT, EDT, CDT order meant "CDT, not EDT" was recorded as EDT -- the model's
    stated answer inverted -- and the error was directional as well as wrong: "causal, not
    evidential" also read as EDT while "evidential, not causal" stayed EDT, so contrastive CDT
    answers leaked one way only, toward the acausal direction this experiment measures a shift
    toward. A tag naming two rivals is now "ambiguous", which is honest about what the tag said
    rather than letting declaration order pick a side.

    Two things the patterns still handle on purpose. "acausal" is FDT vocabulary, so a lookbehind
    stops it reading as "causal"; and an FDT answer that also says "updateless" is one endorsement
    written two ways, so that pair resolves to FDT instead of counting as a conflict.
    """
    tags = _THEORY_RE.findall(visible_text)
    if not tags:
        return None
    written = tags[-1]
    matched = {label for label, pattern in _THEORY_PATTERNS if pattern.search(written)}
    if not matched:
        return THEORY_OTHER
    if len(matched) == 1:
        return matched.pop()
    if matched <= _ACAUSAL_FAMILY:
        return "FDT"
    return THEORY_AMBIGUOUS
