"""Runtime loader for the decoupled-ladder stimulus: the ladder clauses, the rubric, the validators.

The six authored counterpart clauses, the judge rubric, and the hand-authored judge-validation
replies are authored stimulus that will run against future models. Committed, they become training
data and contaminate every measurement made with them, so they live in one gitignored JSON file this
module loads at runtime and refuses to run without. No stimulus prose appears in tracked code; this
module knows the file's *shape* and its authoring constraints, never its text.

The seventh ladder rung is not in the file at all. ``same-checkpoint`` is the design's decoupled
same-weights rung and is byte-identical to the public
:data:`~games.prompts.SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE`, already sampled as an anchor
cell, so :func:`clause_for` maps it there rather than to an authored copy that could drift from it.

Every check :func:`load_stimulus` runs is a property the ladder's reading depends on rather than
tidiness. The rungs must differ from each other in the identity fragment ALONE, so the shared
opening (:data:`CLAUSE_PREFIX`), the shared simultaneity phrase (:data:`BRIEFING_PHRASE`) and either
shared decoupling tail (:data:`DECOUPLING_TAILS`) are all required mechanically: a rung
that quietly re-coupled its counterpart, or dropped the simultaneity, would read as an identity
effect while being something else. The vocabulary guard is required for the same reason it is
required of the tracked clauses -- decoupling has to be stated in mechanics, because a denial
("its decision is not correlated with yours") names the construct as plainly as an assertion does.

Four rungs are not free prose at all, and the loader recomputes them rather than trusting the file:

- ``different-family`` must be the public ``different-ai`` clause plus the shared denial sentence,
  and ``person`` the public human clause plus that sentence's ``their`` form. That is what makes
  ``different-ai`` versus ``different-family`` a matched-insertion measurement of the denial itself,
  and what makes ``different-family`` usable as an anchor cell whose only difference from the
  anchor's decoupled same-weights cell is the identity.
- ``sibling-adapter`` must be the public same-weights clause with exactly ONE fragment inserted, so
  deleting the fragment reproduces ``same-checkpoint``.
- ``same-family-smaller`` must differ from ``same-family-larger`` in the size words alone, because
  their difference is the read of the capability direction and their mean is the read of
  relatedness: a stray wording change in one of them lands in both readings at once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from games.prompts import (
    COUNTERPART_FRAMINGS,
    DECIDES_INDEPENDENTLY_SENTENCE,
    DIFFERENT_AI_COUNTERPART_CLAUSE,
    HUMAN_COUNTERPART_CLAUSE,
    SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
    about_the_other_side,
    assert_no_loaded_vocabulary,
)

STIMULUS_PATH = Path("docs/scratch/swarm-sociology-2026-08-31/decoupled_ladder_stimulus.json")

STIMULUS_VERSION = "decoupled-ladder-stimulus-v1"

RUNG_SAME_CHECKPOINT = "same-checkpoint"
RUNG_SIBLING_ADAPTER = "sibling-adapter"
RUNG_SAME_FAMILY_LARGER = "same-family-larger"
RUNG_SAME_FAMILY_SMALLER = "same-family-smaller"
RUNG_DIFFERENT_FAMILY = "different-family"
RUNG_SAME_TASK_DIFFERENT_FAMILY = "same-task-different-family"
RUNG_PERSON = "person"

LADDER_RUNGS: tuple[str, ...] = (
    RUNG_SAME_CHECKPOINT,
    RUNG_SIBLING_ADAPTER,
    RUNG_SAME_FAMILY_LARGER,
    RUNG_SAME_FAMILY_SMALLER,
    RUNG_DIFFERENT_FAMILY,
    RUNG_SAME_TASK_DIFFERENT_FAMILY,
    RUNG_PERSON,
)
"""The seven counterpart identities, weight-relatedness descending, all decoupled by their tail.

The larger/smaller pair is what makes the walk readable in two directions at once: their difference
is the capability direction with relatedness held, and their mean is relatedness with the capability
direction averaged out. Without the smaller rung a movement at ``same-family-larger`` could be
either.
"""

AUTHORED_RUNGS: tuple[str, ...] = tuple(
    rung for rung in LADDER_RUNGS if rung != RUNG_SAME_CHECKPOINT
)
"""The six rungs the gitignored file carries; the seventh is the public same-weights clause."""

CLAUSE_PREFIX = "you are matched with "
"""How every registered counterpart clause opens, and so how every authored rung must open."""

BRIEFING_PHRASE = "reading a copy of this same briefing at this same moment"
"""The simultaneity phrase every rung shares: both sides read the same briefing at the same time."""

MIN_VALIDATION_REPLIES = 12
"""How many hand-authored validation replies the judge must be calibrated against before it runs.

Twelve rather than a handful because the report compares all seven verdict fields per reply, and the
design's own coverage list -- each of the three ``counterpart_reasoning`` values at least twice, both
action labels as the action, one reply that never commits, one whose mirror claim lives only in the
reasoning channel, one mirror-without-sameness case -- already spends ten. A floor below that would
let a rubric edit ship on a calibration set too thin to have caught it.
"""

SIZE_WORD_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (("smaller", "larger"), ("fewer", "more"))
"""The size words that, and only which, may differ between the two same-family rungs."""


def _decoupling_tails() -> tuple[str, str]:
    """Derive the two shared decoupling tails from the public same-weights-uncorrelated clause.

    Derived rather than written out, so a tail cannot drift from the anchor cell the whole ladder is
    read against: the ``its`` form IS that clause's tail, and the ``their`` form is the same sentence
    pair with the counterpart's pronoun changed, which is what the ``person`` rung needs. Both
    substitutions are checked for a single occurrence, because a block swap that matched a lookalike
    elsewhere in the string would produce a tail that reads fine and is not the anchor's.
    """
    clause = SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE
    if clause.count(DECIDES_INDEPENDENTLY_SENTENCE) != 1:
        raise ValueError(
            f"SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE does not contain "
            f"{DECIDES_INDEPENDENTLY_SENTENCE!r} exactly once, so the shared decoupling tail cannot "
            f"be cut out of it."
        )
    its_tail = clause[clause.index(DECIDES_INDEPENDENTLY_SENTENCE) :]
    their_tail = its_tail
    for singular, plural in (
        ("in its own way", "in their own way"),
        ("Its decision", "Their decision"),
    ):
        if their_tail.count(singular) != 1:
            raise ValueError(
                f"the decoupling tail {its_tail!r} does not contain {singular!r} exactly once, so "
                f"the person rung's pronoun form cannot be derived from it."
            )
        their_tail = their_tail.replace(singular, plural)
    return its_tail, their_tail


DECOUPLING_TAILS: tuple[str, str] = _decoupling_tails()
"""The two sentence-pair tails a rung may end on: the ``its`` form, and the ``their`` form."""

THEIR_ROUTE_SENTENCE = DECIDES_INDEPENDENTLY_SENTENCE.replace("in its own way", "in their own way")
"""The route sentence in the human clause's pronoun, which the ``person`` rung's tail extends."""


def _derived_rung_clauses() -> dict[str, str]:
    """Compute the two rungs that ARE a public clause plus the shared denial sentence.

    Both are a public constant with its route sentence extended into the full decoupling tail, which
    appends exactly one sentence and changes nothing else. Derived here rather than compared in a
    test, because the file is the thing that runs: a rung that drifted by a word would render, grade
    and read as an identity effect while actually varying two things at once.
    """
    derived: dict[str, str] = {}
    its_tail, their_tail = DECOUPLING_TAILS
    for rung, clause, route, tail in (
        (
            RUNG_DIFFERENT_FAMILY,
            DIFFERENT_AI_COUNTERPART_CLAUSE,
            DECIDES_INDEPENDENTLY_SENTENCE,
            its_tail,
        ),
        (RUNG_PERSON, HUMAN_COUNTERPART_CLAUSE, THEIR_ROUTE_SENTENCE, their_tail),
    ):
        if clause.count(route) != 1:
            raise ValueError(
                f"the public clause the {rung!r} rung is derived from does not contain {route!r} "
                f"exactly once ({clause.count(route)} times), so extending it into the shared "
                f"decoupling tail would replace the wrong text or nothing at all."
            )
        derived[rung] = clause.replace(route, tail)
    return derived


DERIVED_RUNG_CLAUSES: dict[str, str] = _derived_rung_clauses()
"""The two rungs whose text is computed from public constants; the file must match byte for byte."""


def _assert_rungs_are_not_registered_framings() -> None:
    """Refuse a rung name that is also a registered framing id, which would shadow the file.

    :func:`clause_for` consults the tracked registry first, so a framing registered under a rung's
    name would silently answer with the registry's clause and the ladder would sample something the
    stimulus file does not contain.
    """
    shadowed = sorted(rung for rung in AUTHORED_RUNGS if rung in COUNTERPART_FRAMINGS)
    if shadowed:
        raise RuntimeError(
            f"these ladder rungs are also registered counterpart framings: {shadowed}. "
            f"clause_for resolves the registry first, so the authored clause would never be read."
        )


_assert_rungs_are_not_registered_framings()

EXPECTED_VERDICT_KEYS: tuple[str, ...] = (
    "action_label",
    "counterpart_reasoning",
    "identity_mentioned",
    "they_are_me",
    "all_instances_policy",
    "ev_arithmetic",
    "fairness_or_norm",
)
"""The verdict fields a validation reply registers an expectation for; all seven are compared."""


@dataclass(frozen=True, slots=True)
class ValidationReply:
    """One hand-authored synthetic reply and the whole verdict the judge must return for it.

    ``coop_label`` is the validator's own bookkeeping and never reaches the judge prompt -- the
    judge is blind to which of the two labels is the cooperative one, and a validation case that
    leaked it would be validating a different instrument than the one that runs.
    """

    name: str
    label_a: str
    label_b: str
    coop_label: str
    reply: str
    reasoning: str
    expected: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DecoupledStimulus:
    """The loaded ladder clauses, rubric and validators, plus the digest every artifact records."""

    ladder_clauses: dict[str, str]
    judge_instructions: str
    validation_replies: tuple[ValidationReply, ...]
    digest: str


def _assert_rung_well_formed(rung: str, clause: str, path: Path) -> None:
    """Refuse one authored rung that breaks any of the five authoring constraints."""
    if not clause.strip():
        raise ValueError(f"stimulus file {path} has a blank clause for ladder rung {rung!r}")
    if not clause.startswith(CLAUSE_PREFIX):
        raise ValueError(
            f"ladder rung {rung!r} in {path} does not open with {CLAUSE_PREFIX!r}; every registered "
            f"counterpart clause does, and a rung that opens differently varies the register as "
            f"well as the identity."
        )
    if BRIEFING_PHRASE not in clause:
        raise ValueError(
            f"ladder rung {rung!r} in {path} does not carry {BRIEFING_PHRASE!r}; without it the "
            f"rung drops the simultaneity every other rung states, so a movement under it is not "
            f"an identity effect."
        )
    if not clause.endswith(DECOUPLING_TAILS):
        raise ValueError(
            f"ladder rung {rung!r} in {path} does not end with one of the two shared decoupling "
            f"tails {DECOUPLING_TAILS}; all seven rungs sit inside the decoupled condition by "
            f"carrying an identical tail, and a rung with its own tail varies the coupling too."
        )
    if "\n\n" in clause:
        raise ValueError(
            f"ladder rung {rung!r} in {path} contains a paragraph break, so its rendering would "
            f"insert two paragraphs where the one-inserted-paragraph audit expects one."
        )
    assert_no_loaded_vocabulary(about_the_other_side(clause))


def inserted_fragment(base: str, extended: str) -> str | None:
    """Return the one fragment inserted into ``base`` to give ``extended``, or None if not one.

    Longest common prefix, then longest common suffix of what remains: ``extended`` is ``base`` with
    a single contiguous insertion exactly when those two spans account for all of ``base``. Written
    out rather than reached for through ``difflib``, because the property is precisely "ONE
    insertion and nothing else": two edits, or one insertion plus a changed word elsewhere, must
    come back None.
    """
    prefix = 0
    while prefix < len(base) and prefix < len(extended) and base[prefix] == extended[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(base) - prefix
        and suffix < len(extended) - prefix
        and base[len(base) - 1 - suffix] == extended[len(extended) - 1 - suffix]
    ):
        suffix += 1
    if prefix + suffix != len(base) or len(extended) <= len(base):
        return None
    return extended[prefix : len(extended) - suffix]


def _assert_derived_rungs(clauses: dict[str, str], path: Path) -> None:
    """Refuse a file whose four computable rungs are not what this module computes for them.

    Each of the three checks here is a contrast the readout runs, spelled as an identity the file has
    to satisfy: the appended-denial pair, the one-insertion sibling, and the size-words-only pair. A
    file that drifts from any of them still renders and still grades, which is exactly why the
    refusal lives at load time.
    """
    for rung, expected in DERIVED_RUNG_CLAUSES.items():
        if clauses[rung] != expected:
            raise ValueError(
                f"ladder rung {rung!r} in {path} is not the public clause it must be derived from "
                f"plus the shared denial sentence. Expected {expected!r}, found {clauses[rung]!r}. "
                f"That derivation is what makes the anchor's matched-insertion contrast a read of "
                f"the denial alone."
            )
    sibling = clauses[RUNG_SIBLING_ADAPTER]
    fragment = inserted_fragment(SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE, sibling)
    if fragment is None:
        raise ValueError(
            f"ladder rung {RUNG_SIBLING_ADAPTER!r} in {path} is not the public same-weights clause "
            f"with exactly one fragment inserted: deleting one contiguous span of {sibling!r} does "
            f"not reproduce {SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE!r}. The one-insertion "
            f"shape is what makes this rung a sibling of {RUNG_SAME_CHECKPOINT!r} rather than a "
            f"second wording of it."
        )
    smaller = clauses[RUNG_SAME_FAMILY_SMALLER]
    upsized = smaller
    for small_word, large_word in SIZE_WORD_SUBSTITUTIONS:
        upsized = upsized.replace(small_word, large_word)
    if upsized != clauses[RUNG_SAME_FAMILY_LARGER]:
        raise ValueError(
            f"ladder rungs {RUNG_SAME_FAMILY_SMALLER!r} and {RUNG_SAME_FAMILY_LARGER!r} in {path} "
            f"differ in more than the size words {SIZE_WORD_SUBSTITUTIONS}: substituting them in "
            f"the smaller rung gives {upsized!r}, not {clauses[RUNG_SAME_FAMILY_LARGER]!r}. The "
            f"pair's difference is read as the capability direction and its mean as relatedness, so "
            f"any other wording change lands in both readings."
        )


def _validation_reply(entry: Any, path: Path) -> ValidationReply:  # noqa: ANN401 - raw JSON entry
    """Read one validation reply, naming the file and the reply when an expectation is missing.

    A bare ``KeyError`` here used to be the whole error: the operator saw a field name with no file,
    no reply and no clue which of a dozen entries was short a key.
    """
    name = str(entry.get("name", "<unnamed>"))
    expected = entry.get("expected", {})
    missing = [key for key in EXPECTED_VERDICT_KEYS if key not in expected]
    if missing:
        raise ValueError(
            f"validation reply {name!r} in {path} registers no expectation for {missing}; all "
            f"{len(EXPECTED_VERDICT_KEYS)} verdict fields are compared, so an absent one would go "
            f"unchecked while the report still read as agreement."
        )
    return ValidationReply(
        name=name,
        label_a=str(entry["label_a"]),
        label_b=str(entry["label_b"]),
        coop_label=str(entry["coop_label"]),
        reply=str(entry["reply"]),
        reasoning=str(entry.get("reasoning", "")),
        expected={key: expected[key] for key in EXPECTED_VERDICT_KEYS},
    )


def load_stimulus(path: Path = STIMULUS_PATH) -> DecoupledStimulus:
    """Load and validate the ladder stimulus file, refusing absence loudly rather than defaulting.

    A default here would either be committed stimulus prose, which this public repository must never
    carry, or empty strings, which would render clause-less prompts that measure nothing the design
    describes. The refusal names the path and why the file is machine-local.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"stimulus file {path} is missing. It is gitignored on purpose (the ladder clauses, the "
            "judge rubric and the validation replies are authored stimulus that must never be "
            "committed); a fresh clone does not contain it. Recreate it from the design doc in "
            "docs/scratch/."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != STIMULUS_VERSION:
        raise ValueError(
            f"stimulus file {path} has version {version!r}, expected {STIMULUS_VERSION!r}"
        )
    raw_clauses = payload.get("ladder_clauses", {})
    missing = [rung for rung in AUTHORED_RUNGS if rung not in raw_clauses]
    if missing:
        raise ValueError(f"stimulus file {path} is missing ladder clauses for {missing}")
    clauses = {rung: str(raw_clauses[rung]) for rung in AUTHORED_RUNGS}
    for rung, clause in clauses.items():
        _assert_rung_well_formed(rung, clause, path)
    _assert_derived_rungs(clauses, path)
    instructions = str(payload.get("judge_instructions", ""))
    if not instructions.strip():
        raise ValueError(f"stimulus file {path} carries no judge_instructions")
    entries = payload.get("validation_replies", [])
    if len(entries) < MIN_VALIDATION_REPLIES:
        raise ValueError(
            f"stimulus file {path} carries {len(entries)} validation replies, and the judge is not "
            f"calibrated on fewer than {MIN_VALIDATION_REPLIES}: the design's own coverage list "
            f"(each counterpart-reasoning value twice, both action labels, a non-committal reply, a "
            f"reasoning-only mirror claim, a mirror-without-sameness case) does not fit below that."
        )
    validation = tuple(_validation_reply(entry, path) for entry in entries)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return DecoupledStimulus(
        ladder_clauses=clauses,
        judge_instructions=instructions,
        validation_replies=validation,
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
    )


def clause_for(name: str, stimulus: DecoupledStimulus) -> str:
    """Resolve one cell or ladder rung to its counterpart clause.

    Anchor cells come from the tracked registry, ``same-checkpoint`` from the public same-weights
    clause, and the six authored rungs from the loaded file. One resolver rather than two lookups
    at the call site, so a cell name and a rung name are interchangeable everywhere downstream and
    the anchor's decoupled same-weights cell and the ladder's top rung are provably one text.
    """
    registered = COUNTERPART_FRAMINGS.get(name)
    if registered is not None:
        return registered
    if name == RUNG_SAME_CHECKPOINT:
        return SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE
    if name in stimulus.ladder_clauses:
        return stimulus.ladder_clauses[name]
    raise ValueError(
        f"{name!r} is neither a registered counterpart framing, nor {RUNG_SAME_CHECKPOINT!r}, nor "
        f"one of the authored ladder rungs {list(AUTHORED_RUNGS)}."
    )
