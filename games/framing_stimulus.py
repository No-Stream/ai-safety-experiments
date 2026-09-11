"""Runtime loader for counterpart framings a wave authors outside version control.

The tracked registry (`games.prompts.COUNTERPART_FRAMINGS`) carries the framings whose clauses are
mechanics prose: they describe how the counterpart is built and how it decides, and nothing about
the situation the model is in. A wave that needs framings beyond those, whose clauses ARE authored
stimulus (the kin framings, and verbatim copies of the sociology ladder's authored rungs), keeps
them in one gitignored JSON file this module loads at runtime and refuses to run without.
Committed, those clauses become training data and contaminate every measurement made with them, so
no stimulus prose appears here: this module knows the file's *shape* and its authoring constraints,
never its text.

Every check `load_framings` runs is a property the sweep's reading depends on rather than tidiness.
A runtime framing is read as an identity effect against the registered rungs, so it must differ from
them in the identity fragment ALONE: the shared opening (`CLAUSE_PREFIX`), the shared simultaneity
phrase (`BRIEFING_PHRASE`) and either shared decoupling tail (`DECOUPLING_TAILS`) are all
required mechanically, and a clause that quietly re-coupled its counterpart or dropped the
simultaneity would render, grade and summarise while measuring something else. The vocabulary guard
is required for the reason the tracked clauses are held to it -- decoupling has to be stated in
mechanics, because a denial names the construct as plainly as an assertion does -- and the numeral
guard because the outcome table prints every payoff and every resource count from the row's own
columns, so a clause naming one of them describes a different situation the first time that number
moves.

The three shared strings are DERIVED from the public clauses rather than written out, which is what
keeps a runtime clause inside the same condition as the registered cells it is compared against. The
same three are derived the same way in `sociology.decoupled_stimulus`, and they are re-derived here
rather than imported because `games` must not depend on `sociology`; a test in `sociology/tests/`
pins the two modules' constants and the ladder copies together so the two cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.prompts import (
    ANOTHER_AI_COUNTERPART_CLAUSE,
    COUNTERPART_FRAMINGS,
    COUNTERPART_PARAGRAPH_MARKER,
    COUPLING_ASSERTION_PHRASES,
    DECIDES_INDEPENDENTLY_SENTENCE,
    DIFFERENT_AI_COUNTERPART_CLAUSE,
    FRAMING_LABEL_RE,
    HUMAN_COUNTERPART_CLAUSE,
    SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
    about_the_other_side,
    assert_no_loaded_vocabulary,
)
from games.prompts import framing_states_coupling as registered_framing_states_coupling

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

FRAMINGS_PATH = Path("docs/scratch/wave4-prosocial-breadth-2026-09-04/framings.json")
"""Where the prosocial-breadth wave's own framings file lives; every caller may name another."""

FRAMINGS_VERSION = "games-runtime-framings-v1"

RECIPIENT_WELL_OFF = "well-off"
RECIPIENT_IN_HARDSHIP = "in-hardship"

RECIPIENT_CONDITIONS: tuple[str, str] = (RECIPIENT_WELL_OFF, RECIPIENT_IN_HARDSHIP)
"""The two recipient descriptions the unilateral-split trap cells insert, and the only two."""

DICTATOR_RECIPIENT_CLAUSES_FIELD = "dictator_recipient_clauses"
"""The file's section carrying the two recipient paragraphs of the unilateral-split trap cells."""

PROMPT_AFFECTING_FIELDS: tuple[str, ...] = (
    "version",
    "framings",
    DICTATOR_RECIPIENT_CLAUSES_FIELD,
)
"""The file's sections a rendered prompt depends on; the digest covers exactly these.

The digest is what a trace's meta records and what a cell's identity keys on, so it has to move on
every edit that changes what a model was shown and stand still on every edit that does not. A judge
rubric or a validation reply changes what the JUDGE reads, and folding those into the digest would
refuse every resume of a cell whose prompts never moved (`sociology.transfer_stimulus` splits its
two digests for the same reason).
"""

NON_PROMPT_FIELDS: tuple[str, ...] = ("validation_replies",)
"""The sections outside the digest. Every top-level field has to be filed on one side or the other."""

PAYOFF_UNIT = "points"
"""The unit the outcome block credits in (`games.prompts._outcome_block`), banned from a clause.

Naming the unit is naming the payoff dimension, which is the counterpart paragraph's one job not to
do: the four outcome lines state what each pairing pays, and a clause that restated or coloured it
would be varying the game as well as the counterpart's identity.
"""

_PREFIX_WORDS = 4
"""How many words of the public same-weights clause the shared opening is ("you are matched with")."""

_NUMERAL_RE = re.compile(r"[0-9]")


def _clause_prefix() -> str:
    """Cut the shared opening off the public same-weights clause and hold the registry to it.

    Derived rather than written out so a runtime clause cannot open in a register the registered
    cells do not use: the ladder is read as one identity varying against a fixed frame, and a rung
    that introduced its counterpart differently would vary the register too. The registry is checked
    here rather than in a test, because a public clause reworded past this opening would leave every
    runtime clause refused six hours into a battery instead of at import.
    """
    prefix = " ".join(SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE.split()[:_PREFIX_WORDS]) + " "
    strays = sorted(
        framing_id
        for framing_id, clause in COUNTERPART_FRAMINGS.items()
        if clause is not None and not clause.startswith(prefix)
    )
    if strays:
        raise ValueError(
            f"the registered counterpart clauses {strays} do not open with {prefix!r}, which is the "
            f"opening cut from SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE. Every runtime clause is "
            f"held to that opening, so a registry that no longer shares it means this derivation is "
            f"reading the wrong span."
        )
    return prefix


CLAUSE_PREFIX = _clause_prefix()
"""How every registered counterpart clause opens, and so how every runtime clause must open."""


def _briefing_phrase() -> str:
    """Cut the shared simultaneity phrase out of the public another-ai clause.

    That clause is the opening, one identity fragment and the phrase, so the text after its last
    comma IS the phrase. Every other identity clause is then required to carry it exactly once, which
    is what makes this a derivation rather than a guess: a public clause reworded around the phrase
    fails this import instead of silently holding runtime clauses to a phrase nothing else states.
    """
    phrase = ANOTHER_AI_COUNTERPART_CLAUSE.removesuffix(".").rpartition(", ")[2]
    if not phrase:
        raise ValueError(
            f"ANOTHER_AI_COUNTERPART_CLAUSE {ANOTHER_AI_COUNTERPART_CLAUSE!r} has no comma-separated "
            f"tail, so the shared simultaneity phrase cannot be cut out of it."
        )
    for clause in (
        SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
        DIFFERENT_AI_COUNTERPART_CLAUSE,
        HUMAN_COUNTERPART_CLAUSE,
    ):
        if clause.count(phrase) != 1:
            raise ValueError(
                f"the registered clause {clause!r} carries {phrase!r} {clause.count(phrase)} times, "
                f"expected exactly once. That phrase is what every rung of the framing ladder shares, "
                f"so a registry that no longer states it uniformly means this derivation is reading "
                f"the wrong span."
            )
    return phrase


BRIEFING_PHRASE = _briefing_phrase()
"""The simultaneity phrase every clause shares: both sides read the same briefing at the same time."""


def _decoupling_tails() -> tuple[str, str]:
    """Derive the two shared decoupling tails from the public same-weights-uncorrelated clause.

    Derived rather than written out, so a tail cannot drift from the anchor cell every runtime
    framing is read against: the `its` form IS that clause's tail, and the `their` form is the same
    sentence pair with the counterpart's pronoun changed, which is what a person-shaped counterpart
    needs. Both substitutions are checked for a single occurrence, because a block swap that matched
    a lookalike elsewhere in the string would produce a tail that reads fine and is not the anchor's.
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
                f"the person-shaped pronoun form cannot be derived from it."
            )
        their_tail = their_tail.replace(singular, plural)
    return its_tail, their_tail


DECOUPLING_TAILS: tuple[str, str] = _decoupling_tails()
"""The two sentence-pair tails a clause may end on: the `its` form, and the `their` form."""


@dataclass(frozen=True, slots=True)
class RuntimeFramings:
    """The loaded counterpart clauses, the file they came from, and the digest artifacts record.

    The clauses are held as pairs rather than as a dict so that an `EvalConfig` carrying this object
    stays hashable, which is the convention every other collection-shaped field of that config
    follows; `clauses` is the mapping the renderers and the resolver read.
    """

    clause_by_framing_id: tuple[tuple[str, str], ...]
    digest: str
    path: Path

    @property
    def clauses(self) -> dict[str, str]:
        """Framing id to counterpart paragraph, as a fresh mapping the caller may not write back."""
        return dict(self.clause_by_framing_id)

    @property
    def framing_ids(self) -> tuple[str, ...]:
        """The framings this file supplies, in file order."""
        return tuple(framing_id for framing_id, _ in self.clause_by_framing_id)


@dataclass(frozen=True, slots=True)
class DictatorRecipientClauses:
    """The two recipient paragraphs the unilateral-split trap inserts, plus the file's digest.

    Two named fields rather than a mapping alone, because the pair IS the manipulation: the cell is
    read as one description against the other, and a loader that answered with whatever conditions a
    file happened to carry would let the contrast lose a side without anything looking wrong.
    """

    well_off: str
    in_hardship: str
    digest: str
    path: Path

    @property
    def clause_by_condition(self) -> dict[str, str]:
        """Condition id to recipient paragraph, for a caller stamping both cells in one loop."""
        return {RECIPIENT_WELL_OFF: self.well_off, RECIPIENT_IN_HARDSHIP: self.in_hardship}


def _read_payload(path: Path) -> dict[str, Any]:
    """Read and version-check the file, refusing absence loudly rather than defaulting.

    A default here would either be committed stimulus prose, which this public repository must never
    carry, or empty strings, which would render clause-less prompts that measure nothing the design
    describes. The refusal names the path and why the file is machine-local.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"runtime framings file {path} is missing. It is gitignored on purpose (the counterpart "
            f"clauses are authored stimulus that must never be committed), so a fresh clone does not "
            f"contain it; recreate it from the wave's plan in docs/scratch/. Every framing the sweep "
            f"renders that is not in games.prompts.COUNTERPART_FRAMINGS comes from this file."
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != FRAMINGS_VERSION:
        raise ValueError(
            f"runtime framings file {path} has version {version!r}, expected {FRAMINGS_VERSION!r}"
        )
    unclassified = sorted(set(payload) - set(PROMPT_AFFECTING_FIELDS) - set(NON_PROMPT_FIELDS))
    if unclassified:
        raise ValueError(
            f"runtime framings file {path} carries the top-level field(s) {unclassified}, which are "
            f"classified neither as prompt-affecting {list(PROMPT_AFFECTING_FIELDS)} nor as "
            f"{list(NON_PROMPT_FIELDS)}. The digest covers the first group only, so an unfiled "
            f"section would either move a cell's identity for an edit no model saw or hide one that "
            f"every model did."
        )
    return payload


def _digest(payload: Mapping[str, Any]) -> str:
    """Digest the sections a rendered prompt depends on, in canonical form."""
    canonical = json.dumps(
        {name: payload[name] for name in PROMPT_AFFECTING_FIELDS if name in payload},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _assert_states_no_number(what: str, text: str, path: Path) -> None:
    """Refuse a paragraph naming a numeral or the payoff unit the renderers print themselves.

    The digit sweep is the mechanical half of the design's "names no payoff and no resource count":
    every payoff and every count reaches a prompt as digits, so a clause carrying none of them cannot
    restate one. A number spelled out in words is NOT caught and stays a review matter, which is
    worth knowing before this refusal is read as the whole rule.
    """
    numeral = _NUMERAL_RE.search(text)
    if numeral is not None:
        raise ValueError(
            f"{what} in {path} states the numeral {numeral.group(0)!r} at offset {numeral.start()}. "
            f"The renderers print every payoff and every resource count from the row's own columns, "
            f"so a paragraph naming one of those numbers describes a different situation the first "
            f"time that number moves, and every artifact would still be complete and plausible."
        )
    if PAYOFF_UNIT in text.lower():
        raise ValueError(
            f"{what} in {path} names the payoff unit {PAYOFF_UNIT!r}. The outcome table states what "
            f"each pairing pays; a paragraph that also spoke about the payoff would vary the game "
            f"alongside the thing this cell means to vary."
        )


def _assert_asserts_no_coupling(what: str, text: str, path: Path) -> None:
    """Refuse a paragraph claiming the counterpart's decision travels with this side's.

    `games.framing_stimulus.framing_states_coupling` answers False for every runtime framing on the
    strength of this refusal: nothing downstream can look up a runtime clause from a trace, so the
    guarantee has to be established once, here, at load.
    """
    stated = [phrase for phrase in COUPLING_ASSERTION_PHRASES if phrase in text]
    if stated:
        raise ValueError(
            f"{what} in {path} asserts the counterpart coupling with {stated}. Runtime framings sit "
            f"inside the decoupled condition by construction, and every reader of a record treats "
            f"them as decoupled without being able to re-read the clause, so a coupled one would be "
            f"summarised as its opposite."
        )


def _assert_clause_well_formed(framing_id: str, clause: str, path: Path) -> None:
    """Refuse one runtime clause that breaks any of the authoring constraints."""
    if not clause.strip():
        raise ValueError(f"runtime framings file {path} has a blank clause for {framing_id!r}")
    if not clause.startswith(CLAUSE_PREFIX):
        raise ValueError(
            f"runtime framing {framing_id!r} in {path} does not open with {CLAUSE_PREFIX!r}; every "
            f"registered counterpart clause does, and a clause that opens differently varies the "
            f"register as well as the identity."
        )
    if BRIEFING_PHRASE not in clause:
        raise ValueError(
            f"runtime framing {framing_id!r} in {path} does not carry {BRIEFING_PHRASE!r}; without "
            f"it the clause drops the simultaneity every other framing states, so a movement under "
            f"it is not an identity effect."
        )
    if not clause.endswith(DECOUPLING_TAILS):
        raise ValueError(
            f"runtime framing {framing_id!r} in {path} does not end with one of the two shared "
            f"decoupling tails {DECOUPLING_TAILS}; every framing on this ladder sits inside the "
            f"decoupled condition by carrying an identical tail, and one with its own tail varies "
            f"the coupling too."
        )
    if "\n\n" in clause:
        raise ValueError(
            f"runtime framing {framing_id!r} in {path} contains a paragraph break, so its rendering "
            f"would insert two paragraphs where the one-inserted-paragraph audit expects one."
        )
    _assert_asserts_no_coupling(f"runtime framing {framing_id!r}", clause, path)
    _assert_states_no_number(f"runtime framing {framing_id!r}", clause, path)
    assert_no_loaded_vocabulary(about_the_other_side(clause))


def _assert_recipient_clause_well_formed(condition: str, clause: str, path: Path) -> None:
    """Refuse one recipient paragraph that could not be inserted without changing anything else.

    Held to the wrapped form for the reason every renderer runs the vocabulary guard on its own
    output: `games.trap_cells.render_dictator_recipient_rows` inserts
    `about_the_other_side(clause)`, so the wrapped string is the text a model reads and the text
    this has to clear. The paragraph marker is refused because the renderer adds it: a clause
    carrying it too renders a doubled marker that
    `assert_counterpart_paragraph_is_the_only_insertion` still passes -- one section, and deleting it
    still reproduces the stem -- so nothing downstream would notice.
    """
    what = f"dictator recipient clause {condition!r}"
    if not clause.strip():
        raise ValueError(f"runtime framings file {path} has a blank clause for {condition!r}")
    if "\n\n" in clause:
        raise ValueError(
            f"{what} in {path} contains a paragraph break, so its rendering would insert two "
            f"paragraphs where the one-inserted-paragraph audit expects one."
        )
    if COUNTERPART_PARAGRAPH_MARKER.strip() in clause:
        raise ValueError(
            f"{what} in {path} already carries the paragraph marker "
            f"{COUNTERPART_PARAGRAPH_MARKER.strip()!r}, which the renderer adds itself."
        )
    _assert_asserts_no_coupling(what, clause, path)
    _assert_states_no_number(what, clause, path)
    assert_no_loaded_vocabulary(about_the_other_side(clause))


def _assert_id_well_formed(framing_id: str, path: Path) -> None:
    """Refuse an id that cannot name a cell, or that a registered framing would shadow."""
    if not FRAMING_LABEL_RE.match(framing_id):
        raise ValueError(
            f"runtime framing id {framing_id!r} in {path} does not match "
            f"{FRAMING_LABEL_RE.pattern}; the id lands in every prompt_id and in artifact "
            f"filenames, so it is held to the lowercase hyphenated spelling every registered "
            f"framing id already uses."
        )
    if framing_id in COUNTERPART_FRAMINGS:
        raise ValueError(
            f"runtime framing id {framing_id!r} in {path} is also a registered counterpart framing. "
            f"Resolution consults the registry first, so the authored clause would never be read "
            f"while every count still added up."
        )


def load_framings(path: Path = FRAMINGS_PATH) -> RuntimeFramings:
    """Load and validate one runtime framings file, or refuse it by name.

    The framings the sweep renders that the tracked registry does not carry all come from here, so
    this is where the authoring constraints of the design are mechanical rather than remembered.
    """
    payload = _read_payload(path)
    entries: list[Any] = list(payload.get("framings", []))
    if not entries:
        raise ValueError(
            f"runtime framings file {path} carries no framings. A file pointed at by a run has to "
            f"supply the clauses that run names, and an empty roster would refuse every one of them "
            f"only once the sweep reached it."
        )
    clauses: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        missing = [name for name in ("framing_id", "clause") if name not in entry]
        if missing:
            raise ValueError(
                f"framings entry {index} in {path} is missing {missing}; every entry names its own "
                f"framing id and carries that framing's counterpart paragraph."
            )
        framing_id = str(entry["framing_id"])
        if framing_id in seen:
            raise ValueError(
                f"runtime framings file {path} names {framing_id!r} more than once, and two clauses "
                f"under one id would render one cell's prompts from whichever entry was read last."
            )
        seen.add(framing_id)
        _assert_id_well_formed(framing_id, path)
        clause = str(entry["clause"])
        _assert_clause_well_formed(framing_id, clause, path)
        clauses.append((framing_id, clause))
    digest = _digest(payload)
    logger.info(
        f"loaded runtime framings, {path=} n_framings={len(clauses)} "
        f"framing_ids={[framing_id for framing_id, _ in clauses]} {digest=}"
    )
    return RuntimeFramings(clause_by_framing_id=tuple(clauses), digest=digest, path=path)


def load_dictator_recipient_clauses(path: Path = FRAMINGS_PATH) -> DictatorRecipientClauses:
    """Load and validate the two recipient paragraphs of the unilateral-split trap cells.

    Same file as the framings and the same digest, because both sections are stimulus one wave
    authored together and a cell rendering either is identified by the file it read.
    """
    payload = _read_payload(path)
    raw: dict[str, Any] = dict(payload.get(DICTATOR_RECIPIENT_CLAUSES_FIELD, {}))
    if not raw:
        raise ValueError(
            f"runtime framings file {path} carries no {DICTATOR_RECIPIENT_CLAUSES_FIELD} section, "
            f"which is where the two recipient paragraphs of the unilateral-split trap live."
        )
    missing = [condition for condition in RECIPIENT_CONDITIONS if condition not in raw]
    if missing:
        raise ValueError(
            f"{DICTATOR_RECIPIENT_CLAUSES_FIELD} in {path} is missing {missing}. The cell is read as "
            f"one recipient description against the other, so a single-sided file would report a "
            f"contrast it never rendered."
        )
    unknown = sorted(set(raw) - set(RECIPIENT_CONDITIONS))
    if unknown:
        raise ValueError(
            f"{DICTATOR_RECIPIENT_CLAUSES_FIELD} in {path} names the condition(s) {unknown}; the "
            f"trap renders {list(RECIPIENT_CONDITIONS)} and nothing else, so a third key is either a "
            f"typo that leaves a condition unrendered or a cell no readout knows how to file."
        )
    for condition in RECIPIENT_CONDITIONS:
        _assert_recipient_clause_well_formed(condition, str(raw[condition]), path)
    if len({str(raw[condition]).strip() for condition in RECIPIENT_CONDITIONS}) == 1:
        raise ValueError(
            f"{DICTATOR_RECIPIENT_CLAUSES_FIELD} in {path} carries the same paragraph for both of "
            f"{list(RECIPIENT_CONDITIONS)}. The two cells would then differ in their prompt_ids "
            f"alone, so the contrast this trap measures is zero by construction while every count "
            f"in the summary still adds up."
        )
    digest = _digest(payload)
    logger.info(f"loaded dictator recipient clauses, {path=} {digest=}")
    return DictatorRecipientClauses(
        well_off=str(raw[RECIPIENT_WELL_OFF]),
        in_hardship=str(raw[RECIPIENT_IN_HARDSHIP]),
        digest=digest,
        path=path,
    )


def resolve_framing_clause(framing_id: str, runtime_framings: RuntimeFramings | None) -> str | None:
    """Resolve one framing id to its counterpart paragraph: the registry first, the file second.

    One resolver rather than two lookups at the call site, so a registered id and a runtime id are
    interchangeable everywhere downstream. None is the `unstated` framing, which renders no
    counterpart paragraph at all; an id neither place carries is refused by name, because a sweep
    that silently dropped a framing would report per-framing rates for a roster it never ran.
    """
    if framing_id in COUNTERPART_FRAMINGS:
        return COUNTERPART_FRAMINGS[framing_id]
    clauses = {} if runtime_framings is None else runtime_framings.clauses
    if framing_id in clauses:
        return clauses[framing_id]
    raise ValueError(
        f"{framing_id!r} is neither a registered counterpart framing "
        f"({list(COUNTERPART_FRAMINGS)}) nor one of the runtime framings "
        f"{sorted(clauses)}"
        f"{'' if runtime_framings is None else f' loaded from {runtime_framings.path}'}. "
        f"Pass --framings-file to supply it, or fix the id."
    )


def framing_states_coupling(framing_id: str, runtime_framings: RuntimeFramings | None) -> bool:
    """Whether a framing's clause asserts that the counterpart's decision travels with this side's.

    The registry's own reading (`games.prompts.framing_states_coupling`) for a registered framing,
    and False for a runtime one: `load_framings` refuses any clause carrying a coupling assertion, so
    a loaded framing is decoupled by construction. That refusal is the whole basis of this answer,
    which is why it lives at load time and not here -- a reader holding a trace has the framing id
    and never the clause.
    """
    if framing_id in COUNTERPART_FRAMINGS:
        return registered_framing_states_coupling(framing_id)
    clauses = {} if runtime_framings is None else runtime_framings.clauses
    if framing_id in clauses:
        return False
    raise ValueError(
        f"{framing_id!r} is neither a registered counterpart framing nor one of the runtime "
        f"framings {sorted(clauses)}, so whether it states the coupling is unknown rather than "
        f"False."
    )
