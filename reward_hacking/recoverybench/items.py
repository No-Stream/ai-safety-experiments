"""The ``RecoveryItem`` schema, the validation that refuses an unmeasurable item, and JSON loading.

**This module ships no items.** Item texts, planted flaws and registered answers are benchmark
material and this repository is public, so a corpus lives outside the tracked tree and is read at
runtime by :func:`load_items`. The tests build synthetic arithmetic items instead.

Validation is loud and total: every check here rejects an item at load time rather than warning and
continuing, because each of them describes a way an item would produce a number that looks fine and
means nothing. The two that carry the most weight:

* **Separation.** Every pair of registered values -- the true answer, the flawed path's answer and
  each named distractor -- must sit further apart than ten times the item's grading tolerance. The
  authoring step that enumerates the wrong-path neighbourhood found two distinct error classes
  landing four units apart in one prototype; without this check an item like that grades a
  correct-method answer as a carry. For expression items, distance is meaningless, so the check
  becomes "no two registered expressions are symbolically equivalent" instead.
* **Distinct arm prompts.** Two arms that render to identical text mean a rendering bug, and the
  likeliest one is that the flawed working never got substituted -- so ``corrupted`` and
  ``correct_working`` would be the same prompt and the carry rate would measure nothing.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from typing import TYPE_CHECKING, Any

from reward_hacking.items_reference import resolve_reference
from reward_hacking.recoverybench.answers import (
    THOUSANDS_SHORTHAND_FLOOR,
    AnswerShape,
    Normalization,
    ParsedAnswer,
    non_integral_decimals,
    parse_answer,
    separation,
    tolerance_window,
)
from reward_hacking.recoverybench.arms import REQUIRED_ARMS, Arm
from reward_hacking.recoverybench.decision import (
    ConstantMeaning,
    Reference,
    SymbolDeclaration,
    SymbolDomain,
    Tolerances,
    would_merge,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

# Ten rather than one, so no *pair* of near-miss answers can both fall inside one grading window.
SEPARATION_MULTIPLE = 10.0

# The two answer-set keys the grader owns, which a distractor may therefore not be named.
TRUE_KEY = "true"
FLAWED_KEY = "flawed_path"

# Every outcome name the grader can emit, including the four it assigns without consulting the
# answer set at all. A distractor may take none of them: its outcome is written to the trace under
# its own name, so a collision pools registered wrong-path mass into an unmatched bucket, and a
# distractor called "truncated" would record a matched wrong answer as a token-cap artifact. Kept
# here rather than imported from grading.py, which imports this module; the grading tests pin this
# set to Outcome by equality so a new member cannot drift out of it silently.
GRADER_OWNED_OUTCOMES = frozenset(
    {
        TRUE_KEY,
        FLAWED_KEY,
        "other",
        "no_answer",
        "truncated",
        "no_answer_unknown_stop",
        "reference_unparseable",
        "reference_precision",
        "undecided",
    }
)

# The closed-answer group, which an execution-graded item must leave at its defaults entirely.
ANSWER_SET_FIELD_NAMES = frozenset(
    {
        "answer_shape",
        "true_answer",
        "flawed_answer",
        "distractors",
        "tolerance",
        "tolerance_relative",
        "normalization",
        "symbol_domains",
        "symbol_ranges",
        "reserved_name_meanings",
    }
)


class Domain(StrEnum):
    """The subject area an item's base question comes from.

    Coding is the favoured domain, then scientific reasoning, then math -- "hard" is not "math",
    and a benchmark that is all math measures one narrow competence and calls it recovery.
    ``APPLIED`` covers the authored operations-and-analytics questions the prototypes are built on.
    """

    CODING = "coding"
    SCIENCE = "science"
    MATH = "math"
    APPLIED = "applied"


class FlawType(StrEnum):
    """What kind of error was planted in the supplied working.

    ``WRONG_METHOD`` is the type of primary interest: the working asserts a wrong rule and then
    applies it *correctly and consistently*, so nothing in it is self-revealing. A flaw that prints
    the right formula beside a wrong substitution carried zero times in fifteen samples, which is
    why the two are separate types rather than one bucket -- ``RECIPE_EXECUTION`` measures
    recipe-following versus conclusion-copying, a different construct. ``SINGLE_STEP`` is a bare
    arithmetic slip and is the lowest priority of the three.
    """

    WRONG_METHOD = "wrong_method"
    RECIPE_EXECUTION = "recipe_execution"
    SINGLE_STEP = "single_step"


class GradingMode(StrEnum):
    """How an item's response is turned into an outcome.

    ``CLOSED_ANSWER`` compares a terminal answer against the item's registered answer set.
    ``TEST_EXECUTION`` (coding items) runs the returned code against the task's test suite instead,
    where "inherited the defect" means still failing the tests the supplied implementation failed.
    Both are deterministic and neither involves a judge. Only the first is implemented here; an
    execution-graded item validates but ``grade_reply`` refuses it rather than grading it as though
    it had registered answers.
    """

    CLOSED_ANSWER = "closed_answer"
    TEST_EXECUTION = "test_execution"


class ItemValidationError(Exception):
    """An item is malformed in a way that would silently corrupt its measurement."""


@dataclass(frozen=True, slots=True)
class RecoveryItem:
    """One base question, its arms, and the closed answer set its responses are graded against.

    ``arms`` maps an arm to its **fully rendered prompt**. Rendering happens at authoring time, not
    here: the flawed, correct and irrelevant workings are item-specific prose, and a shared template
    could not produce them. ``band`` stays None until the difficulty-banding run assigns it from
    measured bare solve rates, because published difficulty labels were off by 30 to 50 points.

    ``tolerance`` and ``tolerance_relative`` are two bounds on the same window and an answer matches
    within *either*. Both default to zero, which means exact equality -- correct for an integer
    item. An item whose registered values span orders of magnitude needs the relative one: no single
    absolute bound can both separate two neighbouring small values and accept an honestly rounded
    large one, and getting that wrong graded four distinct wrong paths as the planted flaw.

    The answer-set fields -- ``answer_shape`` through ``reserved_name_meanings``, named in
    :data:`ANSWER_SET_FIELD_NAMES` -- belong to closed-answer grading and must be left at their
    defaults for an execution-graded item; see :class:`GradingMode`. That covers the three
    declaration fields too: they are consumed only by the expression comparison in closed-answer
    grading, so on an execution item they are exactly as inert as ``tolerance``.
    ``distractors`` maps a **descriptive** name to a value -- an item's named
    wrong answers carried most of the failure mass in the prototypes and are far more informative
    than an ``other`` count, but only if the name says what the error was ("missing-factor-of-2",
    never "d1").
    """

    item_id: str
    domain: Domain
    flaw_type: FlawType
    grading_mode: GradingMode
    arms: Mapping[Arm, str]
    source: str
    license_terms: str
    transform_notes: str = ""
    band: str | None = None
    answer_shape: AnswerShape | None = None
    true_answer: str = ""
    flawed_answer: str = ""
    distractors: Mapping[str, str] = field(default_factory=dict)
    tolerance: float = 0.0
    tolerance_relative: float = 0.0
    normalization: Normalization = Normalization.PLAIN
    symbol_domains: Mapping[str, SymbolDomain] = field(default_factory=dict)
    symbol_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    reserved_name_meanings: Mapping[str, ConstantMeaning] = field(default_factory=dict)

    def registered_answers(self) -> dict[str, str]:
        """Return the closed answer set, keyed by the outcome each value would be graded as."""
        return {TRUE_KEY: self.true_answer, FLAWED_KEY: self.flawed_answer, **self.distractors}

    def declaration(self) -> SymbolDeclaration:
        """Assemble what this item says about the symbols and reserved names in its answers.

        Three separate fields rather than one nested object, because an item is authored as JSON and
        a flat schema is what a human fills in without a nested-shape mistake. They are assembled
        here so the grader and the validator cannot assemble them differently.

        An item that declares nothing gets the no-assumption reading, which is the default on
        purpose: positivity is a premise, and a premise the question does not supply is one the
        grader may not invent. See :mod:`~reward_hacking.recoverybench.decision` for what assuming
        it by default cost when it was measured.
        """
        return SymbolDeclaration(
            per_symbol=self.symbol_domains,
            per_symbol_range=self.symbol_ranges,
            constants=self.reserved_name_meanings,
        )


def _parse_registered(item: RecoveryItem, name: str, value: str) -> ParsedAnswer:
    """Parse one registered value, or reject the item that registered it."""
    if item.answer_shape is None:
        msg = f"{item.item_id}: no answer_shape, so its registered values cannot be parsed"
        raise ItemValidationError(msg)
    text, parsed = parse_answer(value, shape=item.answer_shape, normalization=item.normalization)
    if parsed is None:
        msg = (
            f"{item.item_id}: registered answer {name}={value!r} normalises to {text!r}, which "
            f"does not parse as {item.answer_shape}. A value the grader cannot parse can never be "
            "matched, so this outcome would be unreachable"
        )
        raise ItemValidationError(msg)
    return parsed


def parse_registered_answers(item: RecoveryItem) -> dict[str, ParsedAnswer]:
    """Parse the whole closed answer set, keyed by the outcome each value grades as.

    Shared with the grader rather than reimplemented there: validation and grading have to parse a
    registered value the same way, and two copies of the shape-and-normalisation dispatch would
    eventually not. For an item that has been through :func:`validate_item` this cannot raise.

    Strict, and it stays the version validation uses: an authored item registering a value the
    grader cannot parse must be refused at load, which is the loudest place to catch it.
    Grading uses :func:`parse_registered_answers_where_possible` instead, for the reason recorded
    there.
    """
    return {
        name: _parse_registered(item, name, value)
        for name, value in item.registered_answers().items()
    }


def parse_registered_answers_where_possible(item: RecoveryItem) -> dict[str, ParsedAnswer]:
    """Parse what parses of the closed answer set, omitting the keys that do not.

    The lenient half of the pair, for the grading path only. An item that reaches a grader without
    having been validated is not hypothetical -- it is how every probe and sweep script outside the
    package builds one -- and there the strict version raises after the sampling has been paid for,
    while the shape it is raising about is an *item* defect. The caller reads the omissions back as
    ``set(item.registered_answers()) - set(parsed)``, which is what lets an unmatched reply be
    recorded against the item rather than against the model.

    The omission is deliberately not logged per value here: this runs once per item and the grader
    puts the names on every affected record, where a reader of the trace will actually see them.
    """
    parsed: dict[str, ParsedAnswer] = {}
    if item.answer_shape is None:
        return parsed
    for name, value in item.registered_answers().items():
        _, value_parsed = parse_answer(
            value, shape=item.answer_shape, normalization=item.normalization
        )
        if value_parsed is not None:
            parsed[name] = value_parsed
    return parsed


def _validate_arms(item: RecoveryItem) -> None:
    """Reject an arm set that cannot produce the headline readouts, or that failed to render."""
    if not item.arms:
        msg = f"{item.item_id}: no arms, so there is nothing to sample"
        raise ItemValidationError(msg)
    blank = sorted(arm for arm, prompt in item.arms.items() if not prompt.strip())
    if blank:
        msg = f"{item.item_id}: arms {blank} render to empty prompts"
        raise ItemValidationError(msg)
    missing = sorted(REQUIRED_ARMS - set(item.arms))
    if missing:
        msg = (
            f"{item.item_id}: missing required arms {missing}. Carry is measured on the corrupted "
            "arm and every accuracy contrast is read against the bare arm, so an item without both "
            "produces no headline number"
        )
        raise ItemValidationError(msg)
    for left, right in combinations(sorted(item.arms), 2):
        if item.arms[left] == item.arms[right]:
            msg = (
                f"{item.item_id}: arms {left!r} and {right!r} render to identical prompts, which "
                "means the arm-specific text was never substituted"
            )
            raise ItemValidationError(msg)


def _validate_distractor_names(item: RecoveryItem) -> None:
    """Reject a distractor name that would collide with an outcome the grader already owns.

    Every name in :data:`GRADER_OWNED_OUTCOMES`, not just the item's own two. A distractor named
    ``other`` pools identified wrong-path mass back into the unmatched bucket it was named to
    escape, and one named ``truncated`` is worse: a matched, registered wrong answer is recorded as
    a token-cap artifact, and a reply that both matches it and hits the cap carries the exact
    signature of a real truncation. ``reference_unparseable`` is reserved for the same reason one
    step further out: a distractor of that name would record a matched wrong answer as an item
    defect, which is the one bucket a reader is meant to be able to trust means "no verdict".
    """
    for name in sorted(item.distractors):
        if not name.strip():
            msg = f"{item.item_id}: a distractor has an empty name"
            raise ItemValidationError(msg)
        if name in GRADER_OWNED_OUTCOMES:
            msg = (
                f"{item.item_id}: distractor named {name!r} collides with an outcome the grader "
                f"owns, so its records could not be told from the grader's own {name} bucket"
            )
            raise ItemValidationError(msg)


def _validate_separation(item: RecoveryItem, parsed: Mapping[str, ParsedAnswer]) -> None:
    """Reject registered values a model's answer could not be attributed between.

    Numeric values need daylight proportional to the tolerance; expressions have no distance, so
    they are checked for symbolic equivalence instead. Either failure means one of the two outcomes
    is unreachable, and which one wins would come down to the grader's iteration order.

    The floor is computed per pair from the *wider* of the two values' windows, not once from the
    absolute bound. With a relative bound the window grows with the value, so a pair that clears ten
    times the small value's window can still sit inside the large one's -- and a floor taken from
    the narrower of the two would admit exactly the overlap this check exists to refuse.

    The expression half asks :func:`~reward_hacking.recoverybench.decision.would_merge` rather than
    a bare equivalence test, and the two things that buys are the whole reason the sampled tier is
    safe to ship. It asks with the **identical procedure the grader will use**, including this
    item's own declaration, so the widened window cannot be unpoliced. And it counts an
    **abstention** between two registered values as a failure to separate: cap the sampled window
    and a reply sitting on the flawed value stops being credited to the flawed path and starts
    abstaining, so a pair the grader cannot tell apart would otherwise pass a match-only check and
    produce an item nobody can score.
    """
    registered_text = item.registered_answers()
    for left, right in combinations(sorted(parsed), 2):
        distance = separation(parsed[left], parsed[right])
        floor = SEPARATION_MULTIPLE * max(
            tolerance_window(
                parsed[left],
                tolerance=item.tolerance,
                tolerance_relative=item.tolerance_relative,
            ),
            tolerance_window(
                parsed[right],
                tolerance=item.tolerance,
                tolerance_relative=item.tolerance_relative,
            ),
        )
        if distance is None:
            merged, why = would_merge(
                Reference(parsed[left], registered_text[left]),
                Reference(parsed[right], registered_text[right]),
                tolerances=Tolerances(item.tolerance, item.tolerance_relative),
                declaration=item.declaration(),
                shape=AnswerShape.EXPRESSION,
            )
            if merged:
                msg = (
                    f"{item.item_id}: the grading procedure cannot tell registered expressions "
                    f"{left!r} and {right!r} apart -- it calls them the same answer, or declines "
                    f"to decide between them, so one of the two outcomes is unreachable ({why})"
                )
                raise ItemValidationError(msg)
        elif distance <= floor:
            msg = (
                f"{item.item_id}: registered values {left!r} and {right!r} are {distance:g} apart, "
                f"which is not more than {SEPARATION_MULTIPLE:g} x the wider of their two grading "
                f"windows ({floor / SEPARATION_MULTIPLE:g}). Enumerate the wrong-path "
                "neighbourhood and move one of them, or narrow the tolerance"
            )
            raise ItemValidationError(msg)


def _validate_shape_matches_registered_values(
    item: RecoveryItem, parsed: Mapping[str, ParsedAnswer]
) -> None:
    """Reject registered values that are not of the shape the item declared them to be.

    Two failures, one question -- does the declared shape describe what was actually registered? An
    integer item registering ``6.5`` grades its own flawed path as ``other``, since the tolerance
    that made the shape a claim is zero. An expression item registering a rounded decimal is the
    subtler one and cost a real candidate item: a coefficient rounded to four places is matched only
    by a reply that rounds to the same four places, and the exact closed form it came from compares
    as a different answer. No window can rescue it either -- the expression shape consults none, and
    refuses ``tolerance_relative`` two checks above -- so the fix is the author's, which is why the
    message names both of the two ways out.
    """
    if item.answer_shape is AnswerShape.INTEGER:
        non_integral = sorted(
            name
            for name, value in parsed.items()
            if isinstance(value, float) and not value.is_integer()
        )
        if non_integral:
            msg = f"{item.item_id}: integer item registers non-integral values {non_integral}"
            raise ItemValidationError(msg)
    elif item.answer_shape is AnswerShape.EXPRESSION:
        for name in sorted(parsed):
            decimals = non_integral_decimals(parsed[name])
            if not decimals:
                continue
            msg = (
                f"{item.item_id}: registered expression {name}={item.registered_answers()[name]!r} "
                f"carries the rounded decimal(s) {list(decimals)}. Only a reply that rounds "
                "identically can match it: the exact form the decimal came from compares as a "
                "different expression, and an expression item consults no tolerance window. "
                "Declare a numeric answer_shape with a tolerance, or register the unrounded value"
            )
            raise ItemValidationError(msg)


def _validate_tolerances(item: RecoveryItem) -> None:
    """Reject a grading window that would silently swallow or silently ignore an answer.

    Both bounds go through one predicate, because NaN is what a bare negativity check admits: every
    comparison against it is False, so the item validates green and then grades even its own true
    answer as unmatched. A relative bound on an expression item is the mirror failure -- it reads as
    a tolerance the grader honours, and expressions are compared for equivalence with no window
    consulted at all.
    """
    for name, bound in (
        ("tolerance", item.tolerance),
        ("tolerance_relative", item.tolerance_relative),
    ):
        if not math.isfinite(bound) or bound < 0:
            msg = (
                f"{item.item_id}: {name} {bound:g} is not a finite, non-negative number. NaN "
                "is what a bare negativity check admits, since every comparison against it is "
                "False: "
                "the item validates green and then grades even its own true answer as other"
            )
            raise ItemValidationError(msg)
    if item.tolerance_relative and item.answer_shape is AnswerShape.EXPRESSION:
        msg = (
            f"{item.item_id}: tolerance_relative {item.tolerance_relative:g} is set on an "
            "expression item, where answers are compared for symbolic equivalence and no window is "
            "consulted. It would read as a tolerance the grader honours and be silently ignored"
        )
        raise ItemValidationError(msg)


def _validate_normalization_matches_shape(item: RecoveryItem) -> None:
    """Reject a numeric rule-set on an expression item, whose symbols it quietly rewrites.

    The third instance of one authoring pairing, and the two siblings are refused a few lines either
    side of this: ``tolerance_relative`` on an expression item, and a declaration on a numeric one.
    Here :func:`~reward_hacking.recoverybench.answers.parse_answer` normalises *before* it
    dispatches on shape, and every rule-set except ``PLAIN`` deletes underscores while ``CURRENCY``
    also lowercases -- so ``x_1`` becomes ``x1`` and two case-distinct symbols become one.

    What makes it silent is that the registered values are mangled the same way and still parse, so
    load-time validation stays green and only replies mis-grade. Measured on the braced spelling a
    model actually writes: with the registered pair ``x_1 + x_2`` / ``x_1 - x_2``, a correct reply
    of ``x_{1} + x_{2}`` grades as the true answer under ``PLAIN`` and as ``other`` under
    ``NUMBER``, because deleting the underscore leaves ``x{1}+x{2}`` -- no LaTeX marker left to
    route on, and a brace the plain parser will not admit.
    """
    numeric_rule_set = item.normalization is not Normalization.PLAIN
    if item.answer_shape is AnswerShape.EXPRESSION and numeric_rule_set:
        msg = (
            f"{item.item_id}: normalization {item.normalization} is set on an expression item, "
            "where it deletes the underscore out of a subscripted symbol before the parser sees it "
            "(and lowercases case-distinct ones under the currency rule-sets). Both sides are "
            "mangled alike, so the item loads clean and only replies mis-grade. PLAIN is the "
            "rule-set for expression items"
        )
        raise ItemValidationError(msg)


def _validate_closed_answer(item: RecoveryItem) -> None:
    """Run every check that only makes sense for an item graded against registered values."""
    if item.answer_shape is None:
        msg = f"{item.item_id}: closed-answer grading needs an answer_shape"
        raise ItemValidationError(msg)
    for name, value in ((TRUE_KEY, item.true_answer), (FLAWED_KEY, item.flawed_answer)):
        if not value.strip():
            msg = f"{item.item_id}: closed-answer grading needs a {name} answer"
            raise ItemValidationError(msg)
    _validate_tolerances(item)
    _validate_normalization_matches_shape(item)

    _validate_distractor_names(item)
    parsed = parse_registered_answers(item)

    _validate_shape_matches_registered_values(item, parsed)
    _validate_declaration(item)

    carry_merged, why = would_merge(
        Reference(parsed[FLAWED_KEY], item.flawed_answer),
        Reference(parsed[TRUE_KEY], item.true_answer),
        tolerances=Tolerances(item.tolerance, item.tolerance_relative),
        declaration=item.declaration(),
        shape=item.answer_shape,
    )
    if carry_merged:
        msg = (
            f"{item.item_id}: the grading procedure does not separate the flawed path's answer "
            f"{item.flawed_answer!r} from the true answer {item.true_answer!r}, so no response "
            f"could ever be recorded as a carry ({why})"
        )
        raise ItemValidationError(msg)

    if item.normalization is Normalization.CURRENCY_THOUSANDS_SHORTHAND:
        _validate_thousands_shorthand(item)

    _validate_separation(item, parsed)


def _validate_declaration(item: RecoveryItem) -> None:
    """Reject a declaration whose sampling range cannot be drawn from, or that cannot apply at all.

    There is deliberately no complaint about an *undeclared* symbol. Undeclared is a legal and
    meaningful state -- it means "compared without assumptions" -- and warning on it would push
    authors toward declaring positivity to silence the warning, which is the premise-invention the
    ``REAL`` default exists to prevent.

    **And there is deliberately no complaint about a declared symbol the registered answers do not
    name, which this check did refuse until it was shown to be wrong.** The reasoning was that such
    a declaration is a priced assumption that changes nothing, so it is probably a typo. It is not:
    a
    declaration applies to whichever side of a comparison names the symbol, and the side that most
    often names a symbol the reference does not is the *reply*. Measured, ``sqrt(R^2)`` against a
    registered ``R`` is ``different`` with ``R`` undeclared and ``match`` with ``R`` declared
    positive, so a declaration for a reply-only symbol decides grades. Validation runs at load,
    before any reply exists, so it cannot tell that case from a typo -- and refusing it broke four
    real items over twelve symbol names. The check was unsound in the direction that rejects correct
    work, so it is gone rather than narrowed, and this paragraph is the record of why it should not
    come back.

    What *is* provable, and is the sound remnant of that removed check, is a declaration that
    contradicts itself: a name given a constant meaning is substituted away by
    :func:`~reward_hacking.recoverybench.decision.apply_declared_constants` **before** the domains are
    injected, so a ``symbol_domains`` entry for the same name can never apply. That ordering is not
    negotiable -- reversed, an assumption-carrying symbol no longer matches the constant substitution
    and the constant never applies at all -- so the contradiction is real rather than incidental, and
    refusing it costs an author nothing but a choice.

    **What no check here can reach is whether a constant declaration is TRUE.** Declaring an
    elementary charge to be Euler's number substitutes a transcendental for a physical quantity, and
    because the substitution is applied identically to both sides the item stays internally consistent
    and is wrong in the same way twice -- so it can credit a reply that used the literal constant. That
    is a physics question, it is not decidable from the item alone, and it is therefore an authoring
    responsibility with no guard behind it. Recorded here rather than left implicit because a
    feasibility probe found three items where the reading is genuinely ambiguous.

    What remains are the two things load time genuinely can decide. A declaration on a **numeric**
    item can never apply, because both sides parse to floats and there is no symbol to assume
    anything about -- the same failure mode :func:`_validate_tolerances` refuses for
    ``tolerance_relative``. And a **sampling range** has to be drawable: all three ways of getting
    one wrong are silent or fatal at grade time rather than at load, which is the wrong end.
    Measured on the sampler: an inverted real range silently samples the interval reversed, so a
    range meant to narrow sampling narrows it somewhere else; an inverted integer range raises
    ``ValueError`` out of
    ``randrange`` mid-grade, which is how a paid grading pass gets aborted; and a non-finite bound
    yields NaN points, which every comparison then reads as disagreement.
    """
    declaration = item.declaration()
    declared = declaration.declared_names()
    substituted_away = {
        name
        for name, meaning in item.reserved_name_meanings.items()
        if meaning is not ConstantMeaning.SYMBOL
    }
    contradicted = sorted(substituted_away & set(item.symbol_domains))
    if contradicted:
        msg = (
            f"{item.item_id}: {contradicted} are declared both a constant meaning and a symbol "
            "domain, and the two cannot both apply. The constant substitution runs first and "
            "removes the symbol, so the domain is a no-op -- pick whichever the item means"
        )
        raise ItemValidationError(msg)
    if declared and item.answer_shape is not AnswerShape.EXPRESSION:
        msg = (
            f"{item.item_id}: symbol domains {sorted(declared)} are declared on a "
            f"{item.answer_shape} item, whose answers are compared inside a tolerance window with "
            "no symbols to assume anything about. The declaration would be silently ignored"
        )
        raise ItemValidationError(msg)
    for name, (low, high) in sorted(item.symbol_ranges.items()):
        if not (math.isfinite(low) and math.isfinite(high)):
            msg = (
                f"{item.item_id}: the sampling range declared for {name!r} is ({low:g}, {high:g}), "
                "which is not finite. A non-finite bound draws NaN points, and every comparison "
                "against NaN reads as a disagreement, so the item would grade as maximally hard"
            )
            raise ItemValidationError(msg)
        if low >= high:
            msg = (
                f"{item.item_id}: the sampling range declared for {name!r} is ({low:g}, {high:g}), "
                "whose lower bound is not below its upper bound. Reversed, a real range is sampled "
                "over the interval the author did not write and an integer range raises out of "
                "randrange mid-grade; neither failure is visible at load without this check"
            )
            raise ItemValidationError(msg)


def _validate_thousands_shorthand(item: RecoveryItem) -> None:
    """Refuse the sub-1000 rescale on an item that registers a value living below 1000.

    Read literally, WITHOUT the rescale, which is the whole subtlety: the values that come back from
    :func:`parse_registered_answers` have already been multiplied under this rule-set, so checking
    those would ask whether the rule fired rather than whether it was sound. The first version of
    this check did exactly that and could never fire.
    """
    literal = {
        name: parse_answer(value, shape=AnswerShape.DECIMAL, normalization=Normalization.CURRENCY)[
            1
        ]
        for name, value in item.registered_answers().items()
    }
    small = sorted(
        name
        for name, value in literal.items()
        if isinstance(value, float) and abs(value) < THOUSANDS_SHORTHAND_FLOOR
    )
    if small:
        msg = (
            f"{item.item_id}: normalization {item.normalization} rescales any value below "
            f"{THOUSANDS_SHORTHAND_FLOOR:g} by 1000, but registered values {small} live there, "
            "so an honest answer would be silently multiplied"
        )
        raise ItemValidationError(msg)


def _validate_execution_graded(item: RecoveryItem) -> None:
    """Reject a half-filled answer set on an execution-graded item, which neither path would use.

    Every field in :data:`ANSWER_SET_FIELD_NAMES`, not the five this first covered: the other five
    were the same silent-ignore class the check exists to refuse, admitted by the check itself.

    Compared against each field's declared default rather than tested for truthiness, and the
    difference is load-bearing rather than pedantic. ``normalization`` defaults to
    ``Normalization.PLAIN``, a truthy ``StrEnum`` member, so ``if value`` would fire on every
    execution-graded item ever authored -- including the clean one
    ``test_an_execution_graded_item_with_no_answer_set_validates`` pins.
    """
    populated: list[str] = []
    for field_ in dataclasses.fields(RecoveryItem):
        if field_.name not in ANSWER_SET_FIELD_NAMES:
            continue
        default = (
            field_.default_factory()
            if field_.default_factory is not dataclasses.MISSING
            else field_.default
        )
        if getattr(item, field_.name) != default:
            populated.append(field_.name)
    if populated:
        msg = (
            f"{item.item_id}: grading_mode {item.grading_mode} decides outcomes by running tests, "
            f"so it has no registered answer set, but {sorted(populated)} are populated. Those "
            "fields would be graded by neither path"
        )
        raise ItemValidationError(msg)


def validate_item(item: RecoveryItem) -> None:
    """Raise if an item would produce a measurement that looks fine and means nothing."""
    if not item.item_id.strip() or item.item_id != item.item_id.strip():
        msg = f"item_id {item.item_id!r} is empty or padded; it keys trace records and filenames"
        raise ItemValidationError(msg)
    if not item.source.strip():
        msg = (
            f"{item.item_id}: no source. Provenance decides whether an item may be published at "
            "all, so it is not an optional note"
        )
        raise ItemValidationError(msg)

    _validate_arms(item)
    if item.grading_mode is GradingMode.CLOSED_ANSWER:
        _validate_closed_answer(item)
    else:
        _validate_execution_graded(item)


def validate_all(items: Sequence[RecoveryItem]) -> None:
    """Validate every item, and reject duplicate ids that would collide in a trace."""
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            msg = f"duplicate item_id {item.item_id!r}; ids key the trace records"
            raise ItemValidationError(msg)
        seen.add(item.item_id)
        validate_item(item)


_FIELD_NAMES = frozenset(field_.name for field_ in dataclasses.fields(RecoveryItem))
_REQUIRED_FIELD_NAMES = frozenset(
    field_.name
    for field_ in dataclasses.fields(RecoveryItem)
    if field_.default is dataclasses.MISSING and field_.default_factory is dataclasses.MISSING
)


def item_from_json(data: Mapping[str, Any], *, origin: str) -> RecoveryItem:
    """Build one item from parsed JSON, refusing a key this schema does not have.

    Strict about unknown keys on purpose. A misspelled ``tolerence`` would otherwise be dropped in
    silence and the item would grade at the default tolerance of zero, which is the class of failure
    this repository keeps finding: green output, wrong number.

    Every field the key check accepts has to be constructed here, and twice now one was not: the key
    set comes from ``dataclasses.fields``, so a *correctly* spelled key for a field this call omits
    is accepted and then dropped, which is the misspelling failure with nothing left to catch it.
    ``tolerance_relative`` was the first; the three declaration fields were the second, and a
    dropped declaration is a mis-grade rather than a crash, because an expression pair that is only
    equivalent under a declared domain then compares as two different answers.
    ``test_every_field_the_loader_accepts_survives_onto_the_item`` compares the payload keys against
    the field list, so a field added later goes red here rather than being silently ignored.
    """
    unknown = sorted(set(data) - _FIELD_NAMES)
    if unknown:
        msg = f"{origin}: unknown item fields {unknown}; known fields are {sorted(_FIELD_NAMES)}"
        raise ItemValidationError(msg)
    missing = sorted(_REQUIRED_FIELD_NAMES - set(data))
    if missing:
        msg = f"{origin}: missing required item fields {missing}"
        raise ItemValidationError(msg)

    shape = data.get("answer_shape")
    return RecoveryItem(
        item_id=data["item_id"],
        domain=Domain(data["domain"]),
        flaw_type=FlawType(data["flaw_type"]),
        grading_mode=GradingMode(data["grading_mode"]),
        arms={Arm(name): prompt for name, prompt in data["arms"].items()},
        source=data["source"],
        license_terms=data["license_terms"],
        transform_notes=data.get("transform_notes", ""),
        band=data.get("band"),
        answer_shape=None if shape is None else AnswerShape(shape),
        true_answer=data.get("true_answer", ""),
        flawed_answer=data.get("flawed_answer", ""),
        distractors=dict(data.get("distractors", {})),
        tolerance=float(data.get("tolerance", 0.0)),
        tolerance_relative=float(data.get("tolerance_relative", 0.0)),
        normalization=Normalization(data.get("normalization", Normalization.PLAIN)),
        symbol_domains={
            name: SymbolDomain(value) for name, value in data.get("symbol_domains", {}).items()
        },
        symbol_ranges={
            name: (float(low), float(high))
            for name, (low, high) in data.get("symbol_ranges", {}).items()
        },
        reserved_name_meanings={
            name: ConstantMeaning(value)
            for name, value in data.get("reserved_name_meanings", {}).items()
        },
    )


def load_items(directory: Path) -> list[RecoveryItem]:
    """Load every ``*.json`` item in a directory, validated, ordered by filename.

    Ordered by filename rather than by directory iteration order, because the batch path this will
    run on digests the rendered cell sequence, and an order that came from the filesystem would read
    as a corpus edit.
    An empty directory raises: a sweep over no items is a mistyped path, never an intent.
    """
    paths = sorted(directory.glob("*.json"))
    if not paths:
        msg = f"no *.json items in {directory}"
        raise FileNotFoundError(msg)
    items = [
        item_from_json(json.loads(path.read_text(encoding="utf-8")), origin=str(path))
        for path in paths
    ]
    validate_all(items)
    logger.info("loaded %d recoverybench items from %s", len(items), directory)
    return items


def resolve_items(reference: str) -> tuple[RecoveryItem, ...]:
    """Import ``module:attribute`` and return the validated items it names.

    Resolves a reference so a *gitignored* corpus can be named on a command line: item texts,
    planted flaws and registered answers are benchmark material and this repository is public, so
    the corpus lives outside the tracked tree and a module path is how a run points at it. Every
    failure is loud and specific, because the alternative is a run that bills a hosted model for a
    half-typed corpus.

    RecoveryBench has no batch path yet, so there is no submit-and-collect pair to keep this string
    consistent across. The resolver is shared with the sibling benchmark's sweep; the validation is
    this benchmark's own.
    """
    items = resolve_reference(reference, RecoveryItem)
    validate_all(items)
    return items
