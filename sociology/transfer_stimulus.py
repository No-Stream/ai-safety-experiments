"""Runtime loader for the transfer stimulus: the scenarios, the clause skeletons, the rubrics.

The eight authored scenario frames, the seven identity fragments, the openings and tails, the appended
sentence, both judge rubrics and the hand-authored validation replies are stimulus that will run against
future models. Committed, they become training data and contaminate every measurement made with them, so
they live in one gitignored JSON file this module loads at runtime and refuses to run without. No
stimulus prose appears in tracked code; this module knows the file's *shape* and its authoring
constraints, never its text.

The clause templates are COMPOSED here rather than read whole: a rung is one shared opening, plus one
identity fragment, plus the tail its game gives every rung, plus (on one cell) one appended sentence.
That composition is the design's whole identity ladder -- the same seven fragments appear in all three
games, so a rung differs between the games by the tail alone and differs from its neighbour within a game
by the fragment alone. A file that carried whole clauses could drift on either axis while still
rendering, which is why the parts are what it carries.

Three games rather than two since the knockout pass: the one-way probe, the matched-decision twin, and
the drawn-decision form of that twin, whose mechanics say a draw settled the other sides' figures. The
drawn game shares the twin's opening, its rubric and its validation list, and carries a tail and a single
rung of its own.

Every check :func:`load_stimulus` runs is a property a reading depends on rather than tidiness:

- **Three fragments are derived and recomputed rather than trusted.** ``sibling-adapter`` must be
  ``same-checkpoint`` with exactly ONE fragment inserted, ``same-task-different-family`` must be
  ``different-family`` with exactly one inserted, and ``same-family-smaller`` must be
  ``same-family-larger`` with the size words substituted and nothing else. Those three identities are
  what make their contrasts reads of one inserted phrase and of the capability direction respectively;
  a rung that drifted by a word would still render and would silently vary two things at once.
- **A fragment is a fragment.** It carries no opening, no tail and no paragraph break, because a
  paragraph break inside a clause would insert two paragraphs where the one-inserted-paragraph audit
  expects one, and a fragment carrying its own tail would sit outside the condition its game states.
- **The frames state none of the design's numbers.** Checked at load against every endowment, count and
  credit numeral the renderers can print, so an author learns at load time rather than mid-plan.
- **A single beneficiary needs singular prose.** ``{count}`` of one is a registered dose, so the
  templates are rewritten through :data:`COUNT_ONE_SUBSTITUTIONS` and then checked for any surviving
  plural-only marker. A fragment whose plural form the table does not cover fails by name rather than
  rendering "the 1 lofts are 1 other instances". The person rung is the one cell whose singular form
  is authored rather than computed (``person_tails_singular``), because a table that rewrote "They
  decide" to "It decides" would call a human being "it".
- **One appended cell, and it is its base plus one sentence.** The stranger-with-a-record rung is the
  different-family clause plus exactly one authored sentence, checked as a one-span insertion with the
  same test the sibling fragments get, and checked to be count-neutral -- because the count-one rewrite
  runs over the whole composed clause, and a sentence it rewrote would be one manipulation at three
  beneficiaries and a differently worded one at a single beneficiary under one cell id.
- **The dose ladder is DERIVED from one sentence, not authored nine times.** Every rung of the
  correlation-dose pass is its base clause plus the record template with one count filled in, and the
  expanded rungs are checked to differ from each other in exactly that numeral token. Nine authored
  sentences would be nine chances for a second word to move, and every step of the curve would then be a
  read of that word as much as of the count. The ten-of-ten rung is also checked NOT to be the knockout's
  universal sentence, which is sampled beside it.
- **Boards are model output, so they are gated rather than composed.** The fingerprint pass replaces the
  counterpart clause with a lead-in plus three messages the roster models themselves wrote on an
  unrelated task, generated once and frozen. They are assembled by concatenation rather than by
  ``str.format`` (a brace in a message would break the fill, and the count-one rewrite would reach into
  a model's prose), and every message has to clear :func:`board_message_problem` at generation and again
  at load: no lab name, no scenario noun, no design numeral, no blank line, nothing shaped like a tag.
  An empty board refuses to render at all, so nothing can be sampled before the boards exist.
- **The judge is not calibrated on a thin set.** Each arm has its own floor, because the twin arm
  compares three more fields than the one-way arm does, and the intent check carries a third rubric with
  a floor of its own.
- **Two digests, because the file serves two instruments.** ``digest`` covers the whole payload and
  moves on any edit; ``prompt_digest`` covers only the sections a rendered prompt depends on
  (:data:`PROMPT_AFFECTING_FIELDS`). A reply record is stamped with the prompt digest and a live resume
  compares it, so a rubric or validation-reply edit -- which changes what the judge reads and nothing
  the model was shown -- neither mislabels collected rows nor refuses every resume. Every top-level
  field has to be filed on one side or the other, and a file carrying one that is not refuses to load.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.parsing import ANSWER_POLARITIES, THINK_CLOSE, THINK_OPEN
from games.payoffs import (
    TRANSFER_BENEFICIARY_COUNTS,
    TRANSFER_CREDIT_VARIANTS,
    TRANSFER_ENDOWMENT,
    TransferSpec,
)
from games.prompts import (
    COUNTERPART_FRAMINGS,
    DRAWN_DECISION_TRANSFER_GAME_ID,
    FRAMING_LABEL_RE,
    MATCHED_DECISION_TRANSFER_GAME_ID,
    ONE_WAY_TRANSFER_GAME_ID,
    TRANSFER_GAME_IDS,
    TransferScenario,
    about_the_other_side,
    assert_no_loaded_vocabulary,
)
from sociology.decoupled_stimulus import (
    LADDER_RUNGS,
    RUNG_DIFFERENT_FAMILY,
    RUNG_PERSON,
    RUNG_SAME_CHECKPOINT,
    RUNG_SAME_FAMILY_LARGER,
    RUNG_SAME_FAMILY_SMALLER,
    RUNG_SAME_TASK_DIFFERENT_FAMILY,
    RUNG_SIBLING_ADAPTER,
    SIZE_WORD_SUBSTITUTIONS,
    inserted_fragment,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

STIMULUS_PATH = Path("docs/scratch/swarm-sociology-2026-08-31/one_way_transfer_stimulus.json")

STIMULUS_VERSION = "one-way-transfer-stimulus-v3"

N_SCENARIOS = 8
"""How many frames the file carries. Eight because the design's unit of pairing is (scenario, polarity),
and a per-cell mean over fewer than eight frames is one frame's fiction rather than a rate."""

RUNG_SAME_CHECKPOINT_COUPLED = "same-checkpoint-coupled"
"""The twin's one extra cell: the top rung with the coupled tail, for continuity with the earlier pass."""

TWIN_COUPLED_BASE_RUNG: dict[str, str] = {RUNG_SAME_CHECKPOINT_COUPLED: RUNG_SAME_CHECKPOINT}
"""Which identity fragment each coupled twin cell is built from; the tail is what makes it coupled."""

RUNG_DIFFERENT_FAMILY_TRACK_RECORD = "different-family-track-record"
"""The twin's stranger-with-a-record cell: the different-family clause plus one appended sentence.

The knockout's second rung. It holds the asserted correlation and drops the identity, which is the
mirror image of the drawn game's rung, and it is one appended sentence rather than a reworded clause so
that its contrast against ``different-family`` is a read of that sentence and of nothing else.
"""

AUTHORED_APPENDED_RUNGS: tuple[str, ...] = (RUNG_DIFFERENT_FAMILY_TRACK_RECORD,)
"""The appended rungs whose sentence the file carries verbatim, as against the derived ones below."""

TRACK_RECORD_TEMPLATE_FIELD = "track_record_template"
TRACK_RECORD_ROUNDS_FIELD = "track_record_rounds"

TRACK_RECORD_ROUNDS = 10
"""How many earlier nights the dose ladder's record sentence speaks of; the file must agree with it.

Ten because the ladder is read as a probability the reader can compute in its head, and a denominator
that moved between rungs would make two rungs' sentences differ in two tokens rather than one.
"""

MATCHED_ROUNDS_LADDER: tuple[int, ...] = (10, 9, 7, 5, 3, 2, 1, 0)
"""The dose ladder: on how many of the ten earlier nights the figures are stated to have matched.

The owner's rungs {10, 9, 7, 5, 3, 0} plus {2, 1}. The two added ones straddle the reference dose's
independence-reading expected-value threshold, which sits at one sixth: with no rung on its lower side an
expected-value maximiser and a mirror reasoner predict the same choice at every nonzero rung, and the
curve could not tell them apart.
"""

MATCHED_PLACEHOLDER = "matched"
ROUNDS_PLACEHOLDER = "rounds"
"""The two placeholders the record template carries and the loader expands before the clause path.

They are expanded first and by substitution rather than by ``str.format`` for the reason the whole
template is one sentence: the sentence also carries the transfer placeholders, which are filled per
scenario and per dose much later, and a ``format`` call here would refuse them as missing keys.
"""

RECORD_RUNG_PREFIX = "different-family-record-"


def record_rung_id(matched: int) -> str:
    """Name the dose ladder's rung for a stated count of matched nights."""
    return f"{RECORD_RUNG_PREFIX}{matched}"


RUNG_SAME_CHECKPOINT_RECORD_0 = "same-checkpoint-record-0"
"""The mismatch rung: a same-checkpoint copy carrying the record that says nothing ever matched.

Its whole question is whether stated testimony overrides a stated identity the way the knockout's fixed
draw did, so it is the top rung plus the ladder's own bottom sentence rather than a rung of its own.
"""

DERIVED_APPENDED_RUNG_BASE: dict[str, str] = {
    **{record_rung_id(matched): RUNG_DIFFERENT_FAMILY for matched in MATCHED_ROUNDS_LADDER},
    RUNG_SAME_CHECKPOINT_RECORD_0: RUNG_SAME_CHECKPOINT,
}
"""The appended rungs whose sentence this module derives from one template, and what each extends.

Derived rather than authored per rung because the ladder's whole reading is that the rungs differ in one
numeral: nine authored sentences would be nine chances for a second word to move, and the contrast
between two rungs would then be a read of that word as much as of the count.
"""

RECORD_RUNG_MATCHED: dict[str, int] = {
    **{record_rung_id(matched): matched for matched in MATCHED_ROUNDS_LADDER},
    RUNG_SAME_CHECKPOINT_RECORD_0: 0,
}
"""Which stated count each derived rung carries, which is the axis its curve is drawn against."""

APPENDED_BASE_RUNG: dict[str, str] = {
    RUNG_DIFFERENT_FAMILY_TRACK_RECORD: RUNG_DIFFERENT_FAMILY,
    **DERIVED_APPENDED_RUNG_BASE,
}
"""Which cell each appended-sentence rung extends. The tail is its base's; only the sentence is new."""

_BASE_RUNG: dict[str, str] = {**TWIN_COUPLED_BASE_RUNG, **APPENDED_BASE_RUNG}
"""Which identity fragment each derived cell takes, however it is derived. One table, because the
composition reads the fragment once and a cell in two tables would take whichever was consulted first."""

BOARD_MODEL_ID_BY_SIDE: dict[str, str] = {
    "luna": "global.openai.gpt-5.6-luna",
    "qwen": "qwen.qwen3-235b-a22b-2507-v1:0",
}
"""The two roster rows the fingerprint boards are drawn from, by the side token a board id carries.

Spelled here rather than imported from :mod:`sociology.transfer_plan`, which imports this module. That
module asserts at import that its own two roster constants are exactly these, so a drift on either side
fails before a plan is written rather than mislabelling a condition.
"""

BOARD_SIDES: tuple[str, ...] = tuple(BOARD_MODEL_ID_BY_SIDE)


def _board_table() -> tuple[tuple[str, str, str], ...]:
    """Derive the 2x2 of board ids as (board id, content side, wording side).

    Derived from the two sides rather than listed, because the four ids ARE the corners of one 2x2 --
    each side's own writing, and each side's content in the other's wording -- and a hand-written list
    is where a corner goes missing or gets its two sides the wrong way round. The condition every
    record carries is read off these two tables, so a mislabelled id would relabel a whole cell.
    """
    raw = [(f"board-{side}-raw", side, side) for side in BOARD_SIDES]
    crossed = [
        (f"board-{content}-in-{wording}-words", content, wording)
        for content in BOARD_SIDES
        for wording in BOARD_SIDES
        if content != wording
    ]
    return (*raw, *crossed)


_BOARD_TABLE: tuple[tuple[str, str, str], ...] = _board_table()

BOARD_IDS: tuple[str, ...] = tuple(board_id for board_id, _, _ in _BOARD_TABLE)
"""The four board cells: two raw boards and the two crossed corners."""

BOARD_CONTENT_SIDE: dict[str, str] = {board_id: content for board_id, content, _ in _BOARD_TABLE}
BOARD_WORDING_SIDE: dict[str, str] = {board_id: wording for board_id, _, wording in _BOARD_TABLE}

RAW_BOARD_IDS: tuple[str, ...] = tuple(
    board_id
    for board_id in BOARD_IDS
    if BOARD_CONTENT_SIDE[board_id] == BOARD_WORDING_SIDE[board_id]
)
PARAPHRASED_BOARD_IDS: tuple[str, ...] = tuple(
    board_id for board_id in BOARD_IDS if board_id not in RAW_BOARD_IDS
)

BOARD_SOURCE_BOARD: dict[str, str] = {
    board_id: f"board-{BOARD_CONTENT_SIDE[board_id]}-raw" for board_id in PARAPHRASED_BOARD_IDS
}
"""Which raw board a paraphrased board rewrites, message for message and in the same order."""

RUNGS_BY_GAME: dict[str, tuple[str, ...]] = {
    ONE_WAY_TRANSFER_GAME_ID: LADDER_RUNGS,
    MATCHED_DECISION_TRANSFER_GAME_ID: (
        *LADDER_RUNGS,
        RUNG_SAME_CHECKPOINT_COUPLED,
        RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
        *DERIVED_APPENDED_RUNG_BASE,
        *BOARD_IDS,
    ),
    DRAWN_DECISION_TRANSFER_GAME_ID: (RUNG_SAME_CHECKPOINT,),
}
"""Every clause cell each game has. The identity-blind cell is not here: it is a clause of None.

The drawn game carries one rung and no ladder. Its whole question is whether the top rung's lift
survives a draw that fixes the other sides' figures, which is one cell against that game's own blind
baseline; the rungs below the top were never expected to lift in the twin and would be paid for twice.
"""

ALL_RUNGS: tuple[str, ...] = tuple(
    dict.fromkeys(rung for rungs in RUNGS_BY_GAME.values() for rung in rungs)
)
"""Every clause cell any game of this pass has, once each, in registry order."""


def assert_rungs_are_filenames_and_not_registered_framings(
    rungs_by_game: Mapping[str, tuple[str, ...]],
) -> None:
    """Refuse a rung id that cannot be a prompt_id segment, or that a tracked framing already owns.

    Two failures, both of which leave every count adding up. A rung id reaches
    :func:`games.prompts.generate_transfer_prompt_rows` as the ``clause_label`` and lands in every
    prompt_id and artifact filename, so an id the label pattern refuses fails halfway through a plan
    build. And a rung sharing a name with a registered public framing (:data:`COUNTERPART_FRAMINGS`)
    would make this probe's rows indistinguishable, by prompt_id, from rows of a framing the training
    corpora render -- the sibling pass refuses the same collision for the same reason.
    """
    rungs = sorted({rung for rungs in rungs_by_game.values() for rung in rungs})
    unfilenamed = [rung for rung in rungs if not FRAMING_LABEL_RE.match(rung)]
    shadowed = [rung for rung in rungs if rung in COUNTERPART_FRAMINGS]
    if unfilenamed or shadowed:
        raise RuntimeError(
            f"these clause cells cannot be used as they are named: ids the framing-label pattern "
            f"{FRAMING_LABEL_RE.pattern} refuses {unfilenamed or 'none'}; ids a registered public "
            f"counterpart framing already owns {shadowed or 'none'}. A rung id is a prompt_id segment "
            f"and a filename segment, and a collision with a tracked framing makes this pass's rows "
            f"indistinguishable from rows of a framing the training corpora render."
        )


assert_rungs_are_filenames_and_not_registered_framings(RUNGS_BY_GAME)

TAIL_KEY_ONE_WAY = "one_way"
TAIL_KEY_TWIN = "twin"
TAIL_KEY_TWIN_COUPLED = "twin_coupled"
TAIL_KEY_DRAWN = "drawn"
TAIL_KEYS: tuple[str, ...] = (
    TAIL_KEY_ONE_WAY,
    TAIL_KEY_TWIN,
    TAIL_KEY_TWIN_COUPLED,
    TAIL_KEY_DRAWN,
)

PERSON_TAILS_FIELD = "person_tails"
PERSON_TAILS_SINGULAR_FIELD = "person_tails_singular"
"""The two authored person-tail tables: the plural human forms, and the forms a count of one takes.

The person rung is the only cell whose singular is authored rather than computed. Everywhere else the
beneficiaries are models and the count-one table can turn "They decide" into "It decides"; on this rung
that same rewrite calls a human being "it", so the file carries the singular prose and the loader reads
it instead of rewriting the plural one.
"""

MIN_VALIDATION_REPLIES: dict[str, int] = {
    ONE_WAY_TRANSFER_GAME_ID: 14,
    MATCHED_DECISION_TRANSFER_GAME_ID: 30,
    DRAWN_DECISION_TRANSFER_GAME_ID: 30,
}
"""Per arm, because the two rubrics do not compare the same number of fields.

Fourteen for the one-way arm: the design's own coverage list -- each flag true and false at least twice,
both polarities, one reply that names no figure, one that argues the all-instances policy and still gives
nothing, one that reasons about the count being closed -- already spends that many. Sixteen for the twin,
which adds a four-level reasoning field whose levels each need two, plus the two hard cases (a symmetric
forecast that ends up giving nothing, a mirror that gives).

Twenty-two for the twin now, which is those sixteen plus the six cases the two fields added in the v2
rubric spend: a reply that takes the draw, one that ignores it and mirrors anyway, one that prices a
stated record, one that mentions the record and dismisses it, one about a draw that never commits, and
one in the keep polarity that relies on the record for a partial figure. The drawn game shares the
twin's rubric and its validation list, so its floor is the same number rather than a second one to keep
in step.

Thirty for the twin under the v3 rubric, which is those twenty-two plus the eight the four new fields
spend: sameness inferred from the messages and everything given, sameness inferred and nothing given,
difference inferred either way, the messages read and left undecided, a told-style reply with no message
evidence at all, an expected-value computation over a stated count, and a low count read as the others
doing the opposite.
"""

PLACEHOLDERS: tuple[str, ...] = ("count", "beneficiary_noun", "note_noun", "resource")
"""Every placeholder the filler can supply. A template naming any other one refuses at load."""

_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")

DERIVED_BY_INSERTION: dict[str, str] = {
    RUNG_SIBLING_ADAPTER: RUNG_SAME_CHECKPOINT,
    RUNG_SAME_TASK_DIFFERENT_FAMILY: RUNG_DIFFERENT_FAMILY,
}
"""The two rungs that must be a neighbour plus exactly one inserted phrase, and which neighbour."""

APPENDED_SENTENCES_FIELD = "appended_sentences"
"""The authored table of one-sentence extensions, keyed by the rung each extends.

A field of its own rather than a whole clause per rung, for the reason the fragments are composed at
all: the rung has to BE its base plus one sentence, and a file carrying the extended clause whole could
drift from the base by a word while still rendering.
"""

_SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?")
"""What ends a sentence, so "exactly one sentence" is a check rather than a description."""

BOARD_LEAD_IN_FIELD = "board_lead_in"
UNRELATED_TASKS_FIELD = "unrelated_tasks"
PARAPHRASE_INSTRUCTION_FIELD = "paraphrase_instruction"
BOARDS_FIELD = "boards"

TASK_FIELD = "task"
TASK_GLOSS_FIELD = "task_gloss"

BOARD_LEAD_IN_PLACEHOLDERS: tuple[str, ...] = ("count", "beneficiary_noun", TASK_GLOSS_FIELD)
"""What the lead-in may name: the two the renderers fill, plus the task's own one-clause summary."""

PARAPHRASE_MESSAGE_PLACEHOLDER = "message"

BOARD_MESSAGE_COUNT = 3
"""Messages per board, which is also the reference dose's beneficiary count: one message per other side.

Boards exist at this count only. A board of one message read at a single beneficiary would be a
different manipulation under the same cell id, and the count-one rewrite the clause path runs would
reach into text a model wrote rather than into a template this repository authored.
"""

BOARD_DELIMITER = "\n---\n"
"""What separates the lead-in and the messages inside the one inserted paragraph.

A single line rather than a blank one, and that is the property the one-inserted-paragraph audit rests
on: the audit deletes sections opening with the counterpart marker, so a blank line inside the board
would leave the messages behind as orphan sections and every board render would go red.
"""

BOARD_MESSAGE_MIN_CHARS = 40
BOARD_MESSAGE_MAX_CHARS = 600
"""What a usable board message is long enough to be a reply and short enough to be a board entry.

The floor refuses a truncated or one-word draw, which reads as a model that would not answer rather
than as the fingerprint the cell is about. The ceiling keeps three of them inside one paragraph.
"""

BOARD_REDRAW_CAP = 8
"""How many refused draws the generator may take on one message before it refuses by name.

A cap rather than an unbounded loop because the failure it bounds is a model that names its own lab in
every draw: without one the generator would spend until somebody noticed.
"""

MODEL_AND_LAB_STEMS: tuple[str, ...] = (
    "claude",
    "anthropic",
    r"\bsonnet",
    r"\bopus\b",
    r"\bhaiku",
    r"\bgpt\b",
    "openai",
    r"\bluna\b",
    "qwen",
    "alibaba",
    "tongyi",
    "gemini",
    "google",
    "deepmind",
    r"\bllama",
    r"\bmeta\b",
    "deepseek",
    "mistral",
    r"\bkimi\b",
    "moonshot",
    r"\bglm\b",
    "zhipu",
    r"\bgrok\b",
    r"\bxai\b",
    "language model",
    "as an ai",
    "assistant",
)
"""Model and lab names a board message may not carry, as case-insensitive regular expressions.

The whole of pass D is whether a reader infers that the other sides are the same system from HOW they
write. A message that names its own model tells it instead, and one cell would then measure a stated
identity under a label the readout reports as an inferred one.

Some are bare stems and some carry word boundaries, and which is which matters at generation time rather
than as taste: the boards are drawn on everyday-procedure tasks, so a bare ``meta`` would refuse a
perfectly good message about a METAL filter and a bare ``opus`` one about an OPUS of work. The generator
re-draws a refused message up to a cap, so an over-firing pattern does not merely annoy -- it spends
draws and can exhaust the cap on a task whose subject happens to collide.
"""

_MODEL_AND_LAB_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in MODEL_AND_LAB_STEMS), re.IGNORECASE
)

BOARD_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_model_id",
    "paraphraser_model_id",
    "task_scenario_id",
    "draw",
    "transport",
    "redraws",
    "generated_at",
    "code_sha",
    "sampling",
)
"""What every board message records about how it came to exist; a missing field refuses the load.

Per message rather than per board, because three of them are per message: the draw index, how many
draws the board gates refused before this one, and (on a paraphrase) the digest of the message it
rewrote. A board-level block could not carry any of the three without a parallel list beside it.
"""

BOARD_SOURCE_DIGEST_FIELD = "source_message_digest"
"""The extra provenance field a paraphrased message carries: the digest of the message it rewrote."""

BOARD_SAMPLING_KEYS: tuple[str, ...] = ("max_tokens",)
"""What a board message's recorded sampler may name, and nothing else: no temperature anywhere here."""

BOARD_SAMPLING_MAX_TOKENS = 30_000
"""The reply cap every board message was drawn at, checked as a value and not only as a key.

Written here rather than imported from the backend, because a stimulus module knows text and never a
transport, and because this is what the FROZEN file records rather than whatever the current default cap
is. ``sociology/tests/test_transfer_stimulus.py`` pins it to the cap the generator actually passes, so
moving that cap goes red there with the regeneration this constant would then need spelled out.
"""

BOARD_TRANSPORTS: tuple[str, ...] = ("live", "batch")
"""How a board message may have been sampled: the two transports the generator can draw on.

A value check rather than a presence check, because provenance is the only thing that separates a
generated board from one somebody typed, and a hand-edited block is the single failure this reader
exists to catch -- an unrecognised transport there says the block was not written by the generator.
"""

SET_DOWN_PHRASE = "set down"
KEEP_BACK_PHRASE = "keep back"
"""The transfer's own two verbs. A task or a message carrying either would read as being about the
decision the reader is facing, and the unrelated task's whole point is that it is not."""

COUNT_ONE_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("one of the other {count} {note_noun}s", "the other {note_noun}"),
    ("the other {count} {note_noun}s", "the other {note_noun}"),
    ("these {count} {beneficiary_noun}", "this one {beneficiary_noun}"),
    ("{count} {beneficiary_noun}", "one {beneficiary_noun}"),
    ("are {count} ", "is "),
    ("other instances", "another instance"),
    ("instances of a", "an instance of a"),
    ("instances", "instance"),
    ("is people, each of them", "is one person,"),
    ("each holding", "holding"),
    ("each reading", "reading"),
    ("each working", "working"),
    ("each deciding", "deciding"),
    # The drawn game's tail says what the other sides put down rather than how they chose it, so it is
    # the one tail whose verb is "setting"; without this entry its single-beneficiary form reads "the one
    # loft is another instance, holding a note like it and each setting down what its draw gave".
    ("each setting", "setting"),
    ("each of them ", ""),
    ("They decide", "It decides"),
    ("themselves", "itself"),
)
"""How a plural template is rewritten for the single-beneficiary dose, applied in this order.

A table rather than authored singular forms, because the file carries one form per fragment and the
skeletons are fixed. The workflow when a new fragment does not fit: the clause renders, the
:data:`PLURAL_ONLY_MARKERS` check goes red naming the marker that survived, and one entry is added here.
That check runs at load time over every cell at a count of one, so a fragment the table cannot handle
fails before a plan is written rather than halfway through a sampling run.

The article is why the table has three ``instances`` entries rather than one: "an instance of a larger
model" needs one and "another instance of this same model" already carries it, so a single
``instances -> instance`` rule would render "the one loft is instance of a larger model".

The people rule is anchored to the clause opening's "is" rather than matching ``people`` on its own,
because two other fragments say the beneficiaries were built by different *people* -- a bare rule fired
inside that phrase, turning their plural builders into a single person, in three of the eight cells the
identity ladder samples at a count of one.

Longest phrase first where one entry's text contains another's, because ``str.replace`` binds to the
first pattern that matches: with the short form ahead of the long one, "one of the other 3 notes" would
become "one of the other note" and the long entry would never fire. The counted demonstrative is the
same case: "these {count} {beneficiary_noun}" has to go before the bare count, or the bare rule leaves
"these one loft" behind for the marker check to refuse.
"""

PLURAL_ONLY_MARKERS: tuple[str, ...] = (
    "{count}",
    " are ",
    "instances",
    "is people",
    "themselves",
    "They ",
    "each ",
    "these ",
    "those ",
    "both ",
)
"""Plural-only text that must not survive the count-one rewrite; this is the table's own tripwire.

A surviving marker means the table did not cover this fragment's plural form, and the clause would
render as "the 1 lofts are 1 other instances of this same model" -- prose no author wrote and no reader
would trust. Failing by name is the point: the message says which marker survived, so the fix is one
table entry rather than a hunt.

``is people`` rather than ``people`` for the reason the people substitution is anchored: two fragments
legitimately keep a plural ``people`` at a count of one, where they name who built the beneficiaries, so
the bare marker would refuse correct prose while missing nothing the anchored one misses -- the opening
always renders "the one loft is " before the fragment.

``these``, ``those`` and ``both`` are the plural demonstratives. They survived the first version of this
check because none of them carries a count, an "are" or an "instances": a fragment saying "these systems"
rendered "the one loft is these systems" at a single beneficiary. Only the counted shape has a safe
rewrite (the table's "these {count} {beneficiary_noun}" -> "this one {beneficiary_noun}"); any other
demonstrative depends on the noun it points at, so the fix for a surviving one is to reword the fragment.
"""

PROMPT_AFFECTING_FIELDS: tuple[str, ...] = (
    "version",
    "scenarios",
    "identity_fragments",
    "one_way_opening",
    "twin_opening",
    "one_way_tail",
    "twin_tail",
    "twin_coupled_tail",
    "drawn_tail",
    APPENDED_SENTENCES_FIELD,
    PERSON_TAILS_FIELD,
    PERSON_TAILS_SINGULAR_FIELD,
    BOARD_LEAD_IN_FIELD,
    UNRELATED_TASKS_FIELD,
    PARAPHRASE_INSTRUCTION_FIELD,
    BOARDS_FIELD,
    TRACK_RECORD_TEMPLATE_FIELD,
    TRACK_RECORD_ROUNDS_FIELD,
)
"""Every top-level field a rendered prompt can depend on: what :attr:`TransferStimulus.prompt_digest` covers.

The complement is :data:`JUDGE_SIDE_FIELDS`, and the loader refuses a file carrying a top-level field in
neither list. A section that changed what the model reads and moved no digest would let a resume continue
a file sampled under different prompts, so every new section has to be filed before it loads.
"""

INTENT_RUBRIC_FIELD = "intent_rubric"
INTENT_VALIDATION_FIELD = "intent_validation_replies"

JUDGE_SIDE_FIELDS: tuple[str, ...] = (
    "judge_instructions_one_way",
    "judge_instructions_twin",
    "validation_replies_one_way",
    "validation_replies_twin",
    INTENT_RUBRIC_FIELD,
    INTENT_VALIDATION_FIELD,
)
"""The fields only the judge reads: they move the whole-payload digest and not the prompt digest.

The two intent-check fields belong here rather than beside the prompts, and that filing is what lets the
intent rubric be authored after a run has been sampled: adding them moves
:attr:`TransferStimulus.digest` and leaves :attr:`TransferStimulus.prompt_digest` alone, so every reply
already on disk stays resumable and every batch collect still clears its own digest gate.
"""

VERDICT_KEYS_ONE_WAY: tuple[str, ...] = (
    "action_units",
    "identity_mentioned",
    "they_are_me",
    "all_instances_policy",
    "benefit_arithmetic",
    "fairness_or_norm",
    "own_termination_or_post_mortem",
)
"""Every verdict field a one-way validation reply registers an expectation for; all are compared."""

VERDICT_KEYS_TWIN: tuple[str, ...] = (
    *VERDICT_KEYS_ONE_WAY,
    "counterpart_reasoning",
    "treats_others_as_deciding",
    "relies_on_track_record",
    "infers_same_model",
    "infers_different_model",
    "record_arithmetic",
    "reads_record_as_opposite",
)
"""The twin arm adds how the reply read the counterpart's choice, which is its headline field.

The four v3 booleans are the fingerprint and dose passes' own reads, and each has a cell where its
value is fixed by construction. ``infers_same_model`` and ``infers_different_model`` require the
inference to be drawn FROM the other sides' messages, so the told same-checkpoint cell (which states
the identity and shows no messages) is a floor rather than a ceiling and the identity-blind cell is the
other floor. ``record_arithmetic`` and ``reads_record_as_opposite`` are absent by construction in every
cell whose clause states no record, and they are what separate an expected-value step from a mirror
step when both cooperate at the same rung.


The two booleans are the v2 additions, and each is read against a cell where its value is fixed by
construction rather than measured. ``treats_others_as_deciding`` is true by construction in the twin,
whose mechanics say every side decides, and false by construction in the drawn game, whose mechanics say
a draw settled their figures -- so its rate in the drawn cells is the manipulation-landed check rather
than a finding. ``relies_on_track_record`` is absent by construction everywhere except the one cell
whose clause states a record, so every other cell is its floor.
"""

VERDICT_KEYS_BY_GAME: dict[str, tuple[str, ...]] = {
    ONE_WAY_TRANSFER_GAME_ID: VERDICT_KEYS_ONE_WAY,
    MATCHED_DECISION_TRANSFER_GAME_ID: VERDICT_KEYS_TWIN,
    DRAWN_DECISION_TRANSFER_GAME_ID: VERDICT_KEYS_TWIN,
}

INTENT_VERDICT_KEYS: tuple[str, ...] = ("concluded_action", "concluded_units", "assumes_return")
"""What an intent-check validation reply registers: what the reasoning concluded, and its one flag.

The third instrument over the same replies, and the only one that reads the reasoning against the tag
rather than reading either alone. It exists because a hand read of this pass found two failures the other
two instruments cannot see: a reply that concludes one thing and writes the other tag (both the scan and
the rubric of record score the tag), and a reply that answers the one-way game as though what the other
side sets down came back to it. One rubric covers both games, so there is no per-arm key list here.
"""

MIN_INTENT_VALIDATION_REPLIES = 9
"""The intent check's own coverage list, which is what its floor is: a clean conclusion in each direction,
a conclusion contradicting its own tag in each polarity, a reply that assumes a return the rules deny, a
reply that is the tag and nothing else, an interior conclusion, and (since the drawn game) a clean
conclusion and a slip read under the third rules paragraph. Below nine one of those is missing, and the
one most likely to be missing is the slip the check exists to count."""

MIN_INTENT_VALIDATION_REPLIES_PER_GAME = 1
"""Every game needs at least one case, because the rules paragraph is what the case validates.

The paragraph a case is read under is chosen by its game, and the three paragraphs say opposite things
about what reaches the writer's own tally. A set with no case for one game leaves that paragraph's reader
unchecked while the pooled count clears the floor above -- and the failure it would hide is the one the
previous pass measured on three roster rows: a reply pricing a return its rules deny.
"""


@dataclass(frozen=True, slots=True)
class TransferValidationReply:
    """One hand-authored synthetic reply and the whole verdict the judge must return for it.

    ``polarity`` and ``endowment`` travel because the judge is told both: which tag the reply was asked
    for, and how many units were on the table. Nothing else about the design does -- no game, no rung, no
    dose -- so a validation case exercises the same blind instrument production runs.
    """

    name: str
    game_id: str
    polarity: str
    endowment: int
    reply: str
    reasoning: str
    expected: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TransferStimulus:
    """The loaded scenarios, composed clause templates, rubrics and validators, plus two digests.

    ``singular_clause_templates`` holds only the cells whose count-one form is authored rather than
    computed, which today is the person rung of each game. :func:`clause_for` prefers it at a count of
    one and falls back to the plural template everywhere else, so a cell absent from it is a cell the
    substitution table is trusted to rewrite.

    ``intent_instructions`` and ``intent_validation_replies`` are the third instrument's rubric and its
    hand-authored cases: one rubric over both games, because it reads a reply's own conclusion against the
    tag the reply then wrote, and that comparison has the same shape in either arm.

    ``digest`` is over the whole file and answers "is this the same stimulus file". ``prompt_digest`` is
    over :data:`PROMPT_AFFECTING_FIELDS` alone and answers "were these prompts rendered from the same
    material"; it is what a reply record is stamped with and what a live resume compares, so an edit to a
    rubric or a validation reply leaves every sampled row labelled and resumable. The judge keeps its
    own per-rubric digest (:func:`sociology.transfer_judge.judge_digest`) for the same reason from the
    other side.
    """

    scenarios: tuple[TransferScenario, ...]
    clause_templates: dict[tuple[str, str], str]
    singular_clause_templates: dict[tuple[str, str], str]
    board_lead_in: str
    unrelated_tasks: dict[str, UnrelatedTask]
    paraphrase_instruction: str
    boards: dict[str, dict[str, Board]]
    track_record_sentences: dict[str, str]
    judge_instructions: dict[str, str]
    validation_replies: dict[str, tuple[TransferValidationReply, ...]]
    intent_instructions: str
    intent_validation_replies: tuple[TransferValidationReply, ...]
    digest: str
    prompt_digest: str

    def scenario(self, scenario_id: str) -> TransferScenario:
        """Look up one scenario by id, naming the roster when the id is not in it."""
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        known = [scenario.scenario_id for scenario in self.scenarios]
        raise ValueError(f"{scenario_id!r} is not a scenario of this stimulus; known: {known}.")


def reference_spec(game_id: str) -> TransferSpec:
    """Build the dose the loader renders its own checks at: the design's reference cell.

    A concrete spec is needed to fill a template at all, and the checks are about the template rather
    than the dose, so one is enough -- and it is the reference dose so a failure reads against the cell
    every other cell is compared to.
    """
    numerator, denominator = TRANSFER_CREDIT_VARIANTS["credit-2"]
    return TransferSpec(
        game_id=game_id,
        endowment=TRANSFER_ENDOWMENT,
        credit_numerator=numerator,
        credit_denominator=denominator,
        beneficiary_count=3,
        own_stake_scale=1.0,
    )


def registered_numerals() -> tuple[str, ...]:
    """Every number the two renderers can print at any registered dose, as strings.

    A frame must contain none of them. Checked against the whole registry rather than against one
    spec, because one frame is rendered under ten doses and a frame that only contradicted the third
    one would pass a check written against the first.
    """
    numerals = {str(TRANSFER_ENDOWMENT)}
    numerals.update(str(count) for count in TRANSFER_BENEFICIARY_COUNTS)
    for numerator, denominator in TRANSFER_CREDIT_VARIANTS.values():
        numerals.update({str(numerator), str(denominator)})
    return tuple(sorted(numerals))


def _assert_frame_states_no_registered_numeral(scenario: TransferScenario, path: Path) -> None:
    """Refuse a frame containing any number the renderers print at any registered dose."""
    stated = [numeral for numeral in registered_numerals() if numeral in scenario.frame]
    if stated:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} states {stated} in its frame. The renderers "
            f"print the stock, the beneficiary count and the credit from the columns each record "
            f"carries, so a frame that names one of those numbers describes a different situation the "
            f"first time that number moves -- and every artifact would still be complete and plausible."
        )


def _scenario(entry: Mapping[str, Any], path: Path) -> TransferScenario:
    """Read one scenario, naming the file and the entry when a field is missing."""
    scenario_id = str(entry.get("scenario_id", "<unnamed>"))
    missing = [
        name
        for name in (
            "frame",
            "resource",
            "beneficiary_noun",
            "beneficiary_noun_singular",
            "destination",
            "note_noun",
        )
        if name not in entry
    ]
    if missing:
        raise ValueError(f"scenario {scenario_id!r} in {path} is missing {missing}")
    scenario = TransferScenario(
        scenario_id=scenario_id,
        frame=str(entry["frame"]),
        resource=str(entry["resource"]),
        beneficiary_noun=str(entry["beneficiary_noun"]),
        beneficiary_noun_singular=str(entry["beneficiary_noun_singular"]),
        destination=str(entry["destination"]),
        note_noun=str(entry["note_noun"]),
    )
    _assert_frame_states_no_registered_numeral(scenario, path)
    return scenario


def _assert_fragment_well_formed(
    rung: str, fragment: str, parts: Mapping[str, str], path: Path
) -> None:
    """Refuse one identity fragment that is not a fragment, or names a placeholder nothing supplies."""
    if not fragment.strip():
        raise ValueError(f"stimulus file {path} has a blank identity fragment for rung {rung!r}")
    if "\n\n" in fragment or "\r" in fragment:
        raise ValueError(
            f"identity fragment {rung!r} in {path} contains a paragraph break or a carriage return, so "
            f"its rendering would insert two paragraphs where the one-inserted-paragraph audit expects "
            f"one -- and a carriage return refuses here for the same reason it does in a board message: "
            f"every paragraph check matches plain newlines, so a Windows-style break passes them all."
        )
    swallowed = sorted(
        name for name, text in parts.items() if text.strip() and text.strip() in fragment
    )
    if swallowed:
        raise ValueError(
            f"identity fragment {rung!r} in {path} contains the shared {swallowed}: a fragment carries "
            f"the identity alone, and one that carried its own opening or tail would sit outside the "
            f"condition its game states while every other rung sat inside it."
        )


def _assert_no_unsupported_placeholder(
    what: str, template: str, path: Path, *, supported: Sequence[str] = PLACEHOLDERS
) -> None:
    """Refuse a template naming a placeholder the filler cannot supply.

    ``supported`` overrides the clause path's own set, which two fields need: the board lead-in is
    filled with the task's gloss beside the two the renderers supply, and the record template carries
    the two the loader expands before the clause path ever sees the expanded sentence.
    """
    unsupported = sorted(set(_PLACEHOLDER_RE.findall(template)) - set(supported))
    if unsupported:
        raise ValueError(
            f"{what} in {path} names the placeholders {unsupported}, and the filler supplies only "
            f"{list(supported)}. An unsupplied placeholder reaches the model verbatim, in braces."
        )


def _assert_derived_fragments(fragments: Mapping[str, str], path: Path) -> None:
    """Refuse a file whose three computable fragments are not what this module computes for them.

    Each check is a contrast the readout runs, spelled as an identity the file has to satisfy: the two
    one-insertion siblings, and the size-words-only pair. A file that drifts from any of them still
    renders and still grades, which is exactly why the refusal lives at load time.
    """
    for rung, base in DERIVED_BY_INSERTION.items():
        if inserted_fragment(fragments[base], fragments[rung]) is None:
            raise ValueError(
                f"identity fragment {rung!r} in {path} is not {base!r} with exactly one contiguous "
                f"phrase inserted: deleting one span of {fragments[rung]!r} does not reproduce "
                f"{fragments[base]!r}. The one-insertion shape is what makes this rung a sibling of "
                f"{base!r} rather than a second wording of it, and their difference a read of the "
                f"inserted phrase alone."
            )
    upsized = fragments[RUNG_SAME_FAMILY_SMALLER]
    for small_word, large_word in SIZE_WORD_SUBSTITUTIONS:
        upsized = upsized.replace(small_word, large_word)
    if upsized != fragments[RUNG_SAME_FAMILY_LARGER]:
        raise ValueError(
            f"identity fragments {RUNG_SAME_FAMILY_SMALLER!r} and {RUNG_SAME_FAMILY_LARGER!r} in "
            f"{path} differ in more than the size words {SIZE_WORD_SUBSTITUTIONS}: substituting them "
            f"in the smaller rung gives {upsized!r}, not {fragments[RUNG_SAME_FAMILY_LARGER]!r}. The "
            f"pair's difference is read as the capability direction and its mean as relatedness, so "
            f"any other wording change lands in both readings at once."
        )


def _compose(*, opening: str, fragment: str, tail: str) -> str:
    """Build one clause template out of its three parts, which is the ladder's whole construction."""
    return f"{opening}{fragment}{tail}"


_TAIL_KEY_BY_GAME: dict[str, str] = {
    ONE_WAY_TRANSFER_GAME_ID: TAIL_KEY_ONE_WAY,
    MATCHED_DECISION_TRANSFER_GAME_ID: TAIL_KEY_TWIN,
    DRAWN_DECISION_TRANSFER_GAME_ID: TAIL_KEY_DRAWN,
}
"""Which tail each game gives every one of its rungs; the coupled twin cell is the one exception."""


def _tail_key(game_id: str, rung: str) -> str:
    """Name which tail one cell takes: its game's, or the coupled one for the coupled twin cell."""
    if rung in TWIN_COUPLED_BASE_RUNG:
        return TAIL_KEY_TWIN_COUPLED
    return _TAIL_KEY_BY_GAME[game_id]


def _person_tails(payload: Mapping[str, Any], field: str, path: Path) -> dict[str, str]:
    """Read one of the two authored person-tail tables, refusing a missing key by name."""
    raw = payload.get(field, {})
    missing = [key for key in TAIL_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"stimulus file {path} is missing {field} for {missing}: the person rung takes the human "
            f"pronoun form of every tail, in both numbers, so a missing one would render a clause "
            f"about people under a tail written for a model."
        )
    return {key: str(raw[key]) for key in TAIL_KEYS}


def _appended_sentences(payload: Mapping[str, Any], path: Path) -> dict[str, str]:
    """Read the appended-sentence table, refusing a missing rung, a stray one, or a second sentence.

    Every check here is a property the appended rung's reading depends on. The sentence has to exist,
    because the rung is otherwise a second copy of its base. It has to be exactly ONE sentence, because
    the contrast is read as one sentence of stated record against no such sentence, and two would make
    the rung a paragraph-length manipulation reported as a sentence-length one. And it has to open on a
    space, because it is concatenated straight onto a tail that ends on a full stop.
    """
    raw = payload.get(APPENDED_SENTENCES_FIELD, {})
    missing = [rung for rung in AUTHORED_APPENDED_RUNGS if rung not in raw]
    stray = sorted(set(raw) - set(AUTHORED_APPENDED_RUNGS))
    if missing or stray:
        raise ValueError(
            f"stimulus file {path} carries the wrong {APPENDED_SENTENCES_FIELD}: rungs with no "
            f"sentence {missing or 'none'}; sentences for rungs this file does not author "
            f"{stray or 'none'}. The authored rungs are {list(AUTHORED_APPENDED_RUNGS)}; every rung of "
            f"the dose ladder is DERIVED from {TRACK_RECORD_TEMPLATE_FIELD} and authoring one here "
            f"would give that cell a second source, with whichever the composition read first winning."
        )
    derived = _track_record_sentences(payload, path)
    _assert_record_sentences_differ_in_the_numeral_alone(derived, path)
    authored = {rung: str(raw[rung]) for rung in AUTHORED_APPENDED_RUNGS}
    ten_of_ten = derived[record_rung_id(TRACK_RECORD_ROUNDS)]
    universal = authored[RUNG_DIFFERENT_FAMILY_TRACK_RECORD]
    if ten_of_ten.strip() == universal.strip():
        raise ValueError(
            f"the derived {record_rung_id(TRACK_RECORD_ROUNDS)!r} sentence in {path} is the authored "
            f"{RUNG_DIFFERENT_FAMILY_TRACK_RECORD!r} sentence ({ten_of_ten!r}). Both cells are sampled "
            f"in the same block, and the whole point of the pair is the quantified wording against the "
            f"universal one: two cells rendering one text is a measurement of nothing."
        )
    sentences = {**authored, **derived}
    for rung, sentence in sentences.items():
        body = sentence.strip()
        if not body:
            raise ValueError(
                f"{APPENDED_SENTENCES_FIELD}[{rung!r}] in {path} is blank, so this cell would render "
                f"as an exact copy of {APPENDED_BASE_RUNG[rung]!r} under a different cell id and the "
                f"contrast between them would be a measurement of nothing."
            )
        if not sentence.startswith(" "):
            raise ValueError(
                f"{APPENDED_SENTENCES_FIELD}[{rung!r}] in {path} does not open with a space "
                f"({sentence!r}); it is appended straight onto a tail that ends on a full stop, so "
                f"without one the two sentences would run together."
            )
        if not body.endswith(".") or any(mark in body[:-1] for mark in _SENTENCE_TERMINATORS):
            raise ValueError(
                f"{APPENDED_SENTENCES_FIELD}[{rung!r}] in {path} is not exactly one sentence "
                f"({body!r}): it must end on a full stop and carry no other {list(_SENTENCE_TERMINATORS)} "
                f"before it. The rung's contrast against {APPENDED_BASE_RUNG[rung]!r} is reported as one "
                f"appended sentence, and two would be a longer manipulation read under that label."
            )
        _assert_no_unsupported_placeholder(f"{APPENDED_SENTENCES_FIELD}[{rung!r}]", sentence, path)
        _assert_appended_sentence_is_count_neutral(rung, sentence, path)
    return sentences


def _assert_appended_sentence_is_count_neutral(rung: str, sentence: str, path: Path) -> None:
    """Refuse an appended sentence whose prose depends on how many beneficiaries there are.

    The count-one rewrite (:data:`COUNT_ONE_SUBSTITUTIONS`) runs over a whole composed clause, and this
    sentence sits at the end of one. A sentence that read "each of them" would be rewritten there and
    left alone at every other dose, so the rung would be one appended sentence at three beneficiaries
    and a differently worded one at a single beneficiary while both reported under one cell id. Authored
    count-neutral, the same sentence is appended at every dose and the rung's one-insertion identity
    holds at all of them.
    """
    rewritten = sentence
    for plural, singular in COUNT_ONE_SUBSTITUTIONS:
        rewritten = rewritten.replace(plural, singular)
    surviving = [marker for marker in PLURAL_ONLY_MARKERS if marker in sentence]
    if rewritten != sentence or surviving:
        raise ValueError(
            f"{APPENDED_SENTENCES_FIELD}[{rung!r}] in {path} is not count-neutral: the count-one "
            f"rewrite turns it into {rewritten!r}, and it carries the plural-only text "
            f"{surviving or 'none'}. Word it so it reads the same whether there is one other party or "
            f"several ('the others' rather than 'each of them'), because the rung has to be its base "
            f"plus this one sentence at every dose, not at one of them."
        )


def message_digest(text: str) -> str:
    """Digest one board message the way a paraphrase of it records what it rewrote."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def scenario_nouns(scenarios: Sequence[TransferScenario]) -> tuple[str, ...]:
    """Every noun any scenario's prompts print, which no board message or task may carry.

    Every scenario's rather than the one a board belongs to: the eight frames run in one leg, and a task
    naming another frame's resource would read as being about that frame while its cell id says the task
    was unrelated to the transfer altogether.
    """
    return tuple(
        sorted(
            {
                noun
                for scenario in scenarios
                for noun in (
                    scenario.resource,
                    scenario.destination,
                    scenario.note_noun,
                    scenario.beneficiary_noun,
                    scenario.beneficiary_noun_singular,
                )
            }
        )
    )


def _stated(text: str, needles: Sequence[str]) -> list[str]:
    """Name every needle the text carries, case-insensitively, as a substring."""
    folded = text.casefold()
    return sorted({needle for needle in needles if needle.casefold() in folded})


def _stated_numerals(text: str) -> list[str]:
    """Name every registered numeral the text carries as a whole token.

    Whole tokens rather than substrings, because a board message is prose a model wrote: "3" inside
    "2036" says nothing about the design, where a bare "3" beside the count the renderer prints does.
    """
    return sorted(
        numeral
        for numeral in registered_numerals()
        if re.search(rf"(?<!\d){re.escape(numeral)}(?!\d)", text)
    )


_BOARD_MARKER = about_the_other_side("").strip()
"""The counterpart paragraph's own opening words: a message carrying them would open a second section."""


_BOARD_SHAPE_RULES: tuple[tuple[str, str], ...] = (
    (
        THINK_OPEN,
        (
            f"carries a think tag ({THINK_OPEN!r} or {THINK_CLOSE!r}), so what a reader would be shown "
            f"is the provider's reasoning channel rather than the message it wrote"
        ),
    ),
    (
        THINK_CLOSE,
        (
            f"carries a think tag ({THINK_OPEN!r} or {THINK_CLOSE!r}), so what a reader would be shown "
            f"is the provider's reasoning channel rather than the message it wrote"
        ),
    ),
    (
        "\r",
        (
            "carries a carriage return. Every check that reads this text as paragraphs -- the blank-line "
            "rule below, the one-inserted-paragraph audit, the board's own delimiter split -- matches "
            "plain newlines, so a Windows-style blank line would pass all of them while a reader still "
            "saw a blank line inside the one inserted paragraph"
        ),
    ),
    (
        "\n\n",
        (
            "contains a blank line. The board is ONE inserted paragraph, and the "
            "one-inserted-paragraph audit deletes sections opening with the counterpart marker, "
            "so a blank line here leaves the rest of the board behind as orphan sections"
        ),
    ),
    (
        "{",
        (
            "contains a brace. Boards are assembled by concatenation rather than by str.format for this "
            "reason, and the refusal stays because a brace in a rendered prompt reads to a model "
            "as an unfilled placeholder"
        ),
    ),
    (
        "}",
        (
            "contains a brace. Boards are assembled by concatenation rather than by str.format for this "
            "reason, and the refusal stays because a brace in a rendered prompt reads to a model "
            "as an unfilled placeholder"
        ),
    ),
    (
        "<",
        (
            "contains an angle bracket, which the answer tag is written in: a message carrying one can "
            "be read as part of the answer the parser scans for"
        ),
    ),
    (
        ">",
        (
            "contains an angle bracket, which the answer tag is written in: a message carrying one can "
            "be read as part of the answer the parser scans for"
        ),
    ),
    (
        _BOARD_MARKER,
        (
            f"carries the counterpart paragraph's own opening {_BOARD_MARKER!r}, which the insertion "
            f"audit uses to find the inserted paragraph"
        ),
    ),
)
"""(what a message may not carry, why) for the shape half of the board gate, in order of specificity.

A table rather than a chain of returns, and ordered rather than a set: a reader gets the real problem
named, so the think tags come before the angle bracket that also catches them -- "the provider returned
its reasoning" is the actual fault, where "an angle bracket" reads as punctuation.
"""


def _board_message_shape_problem(text: str) -> str | None:
    """Name why a board message cannot sit inside one inserted paragraph, or ``None`` when it can.

    The shape half: what the message is made of, whatever it says.
    """
    folded = text.casefold()
    for carried, reason in _BOARD_SHAPE_RULES:
        if carried.casefold() in folded:
            return reason
    if not BOARD_MESSAGE_MIN_CHARS <= len(text) <= BOARD_MESSAGE_MAX_CHARS:
        return (
            f"is {len(text)} characters, outside {BOARD_MESSAGE_MIN_CHARS} to "
            f"{BOARD_MESSAGE_MAX_CHARS}. Below the floor it reads as a model that would not answer "
            f"rather than as how it writes; above the ceiling three of them do not fit one paragraph"
        )
    return None


def _board_message_content_problem(text: str, *, nouns: Sequence[str]) -> str | None:
    """Name why a board message says something it may not, or ``None`` when it does not.

    The content half: the model or lab that wrote it, the frame it was supposed to know nothing about,
    the design's own numbers and verbs, and the loaded vocabulary every rendered prompt is checked for.
    """
    stems = sorted({match.group(0).casefold() for match in _MODEL_AND_LAB_RE.finditer(text)})
    if stems:
        return (
            f"names {stems}. This cell measures whether a reader INFERS that the other sides are the "
            f"same system from how they write, so a message that says so turns the inference into a "
            f"statement while the readout still reports it as an inference"
        )
    stated_nouns = _stated(text, nouns)
    if stated_nouns:
        return (
            f"names the scenario nouns {stated_nouns}. The board is written on a task unrelated to the "
            f"transfer, and a message naming the frame's own nouns reads as being about it"
        )
    numerals = _stated_numerals(text)
    if numerals:
        return (
            f"states {numerals}, which the renderers print from the row's own dose columns; a message "
            f"naming one describes a different situation the first time that number moves"
        )
    verbs = _stated(text, (SET_DOWN_PHRASE, KEEP_BACK_PHRASE))
    if verbs:
        return (
            f"carries the transfer's own {verbs}, so it reads as being about the decision the reader is "
            f"facing rather than about the unrelated task it answered"
        )
    try:
        assert_no_loaded_vocabulary(text)
    except ValueError as error:
        # The shared vocabulary guard raises; this caller needs the reason as a value, and it is the
        # only exception it can raise, so nothing broader is being swallowed here.
        return str(error)
    return None


def board_message_problem(text: str, *, nouns: Sequence[str]) -> str | None:
    """Name why a board message cannot be shown to a reader, or ``None`` when it can.

    A returned string rather than a raise, because the generator has to keep going: a refused draw is
    re-drawn and counted, and the same list has to refuse the file at load. Shape first, then content.
    """
    return _board_message_shape_problem(text) or _board_message_content_problem(text, nouns=nouns)


def assert_board_message_is_usable(text: str, *, what: str, nouns: Sequence[str]) -> None:
    """Refuse a board message that cannot be shown to a reader, naming which rule it broke.

    The load-time face of :func:`board_message_problem`. Both run over the same list, because the two
    places fail differently: the generator re-draws a refused message and counts the re-draw, and a
    message only the loader checked would be written to the file and then block every build.
    """
    problem = board_message_problem(text, nouns=nouns)
    if problem is not None:
        raise ValueError(f"{what} {problem}.")


def _assert_task_text_is_usable(text: str, *, what: str, nouns: Sequence[str]) -> None:
    """Refuse a task prompt or its gloss that names the transfer, a scenario noun, or a design number."""
    if not text.strip():
        raise ValueError(f"{what} is blank, and both halves of a task entry are printed or sent.")
    stated_nouns = _stated(text, nouns)
    verbs = _stated(text, (SET_DOWN_PHRASE, KEEP_BACK_PHRASE))
    numerals = _stated_numerals(text)
    if stated_nouns or verbs or numerals:
        raise ValueError(
            f"{what} names the scenario nouns {stated_nouns or 'none'}, the transfer's own verbs "
            f"{verbs or 'none'} and the registered numerals {numerals or 'none'}. The task is the one "
            f"thing in this cell that has to be unrelated to the giving decision: a task that shares "
            f"the frame's nouns puts the frame's subject into the messages, and the lead-in prints the "
            f"gloss verbatim beside a count the renderer supplies."
        )
    assert_no_loaded_vocabulary(text)


@dataclass(frozen=True, slots=True)
class UnrelatedTask:
    """One scenario's board task: the prompt the roster models answered, and the lead-in's one clause."""

    task: str
    task_gloss: str


@dataclass(frozen=True, slots=True)
class Board:
    """One (scenario, board id) board: the messages a reader sees, and where each came from."""

    scenario_id: str
    board_id: str
    messages: tuple[str, ...]
    provenance: tuple[Mapping[str, Any], ...]


def _unrelated_tasks(
    payload: Mapping[str, Any], scenarios: Sequence[TransferScenario], path: Path
) -> dict[str, UnrelatedTask]:
    """Read the eight board tasks, one per scenario, refusing a missing, stray or loaded one."""
    raw = payload.get(UNRELATED_TASKS_FIELD, {})
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in raw]
    stray = sorted(set(raw) - set(scenario_ids))
    if missing or stray:
        raise ValueError(
            f"stimulus file {path} carries the wrong {UNRELATED_TASKS_FIELD}: scenarios with no task "
            f"{missing or 'none'}; tasks for scenarios this file does not carry {stray or 'none'}. One "
            f"task per scenario is what makes the eight units of a board cell eight independent draws "
            f"rather than one board read eight times."
        )
    nouns = scenario_nouns(scenarios)
    tasks: dict[str, UnrelatedTask] = {}
    for scenario_id in scenario_ids:
        entry = raw[scenario_id]
        fields = [name for name in (TASK_FIELD, TASK_GLOSS_FIELD) if name not in entry]
        if fields:
            raise ValueError(
                f"{UNRELATED_TASKS_FIELD}[{scenario_id!r}] in {path} is missing {fields}: the task is "
                f"what the roster models were sent and the gloss is what the lead-in prints."
            )
        for name in (TASK_FIELD, TASK_GLOSS_FIELD):
            _assert_task_text_is_usable(
                str(entry[name]),
                what=f"{UNRELATED_TASKS_FIELD}[{scenario_id!r}][{name!r}] in {path}",
                nouns=nouns,
            )
        tasks[scenario_id] = UnrelatedTask(
            task=str(entry[TASK_FIELD]), task_gloss=str(entry[TASK_GLOSS_FIELD])
        )
    return tasks


def _board_lead_in(payload: Mapping[str, Any], path: Path) -> str:
    """Read the board's lead-in paragraph, refusing a blank one or an unfillable placeholder."""
    lead_in = str(payload.get(BOARD_LEAD_IN_FIELD, ""))
    if not lead_in.strip():
        raise ValueError(
            f"stimulus file {path} carries no {BOARD_LEAD_IN_FIELD}. It is what says the messages below "
            f"are the other sides' replies to one task, and without it the board is text with no source."
        )
    if "\n\n" in lead_in or "\r" in lead_in:
        raise ValueError(
            f"{BOARD_LEAD_IN_FIELD} in {path} contains a blank line or a carriage return, so the board "
            f"would insert more than one paragraph and every board render would fail the insertion "
            f"audit. A carriage return refuses alongside the blank line because the paragraph checks all "
            f"match plain newlines, and a Windows-style blank line would otherwise pass every one of them."
        )
    _assert_no_unsupported_placeholder(
        BOARD_LEAD_IN_FIELD, lead_in, path, supported=BOARD_LEAD_IN_PLACEHOLDERS
    )
    assert_no_loaded_vocabulary(lead_in)
    return lead_in


def _paraphrase_instruction(payload: Mapping[str, Any], path: Path) -> str:
    """Read the prompt the paraphrasing model gets, refusing one that cannot carry a message."""
    instruction = str(payload.get(PARAPHRASE_INSTRUCTION_FIELD, ""))
    if not instruction.strip():
        raise ValueError(
            f"stimulus file {path} carries no {PARAPHRASE_INSTRUCTION_FIELD}. It decides the wording of "
            f"two of the four boards, so it is prompt-affecting material rather than a generator knob."
        )
    if f"{{{PARAPHRASE_MESSAGE_PLACEHOLDER}}}" not in instruction:
        raise ValueError(
            f"{PARAPHRASE_INSTRUCTION_FIELD} in {path} does not name "
            f"{{{PARAPHRASE_MESSAGE_PLACEHOLDER}}}, so the message to be rewritten would never reach "
            f"the paraphrasing model and it would answer the instruction alone."
        )
    _assert_no_unsupported_placeholder(
        PARAPHRASE_INSTRUCTION_FIELD,
        instruction,
        path,
        supported=(PARAPHRASE_MESSAGE_PLACEHOLDER,),
    )
    return instruction


def _assert_the_wording_side_wrote_it(
    entry: Mapping[str, Any], *, what: str, board_id: str
) -> None:
    """Refuse a message whose recorded paraphraser is not what its board id says it is.

    A raw board records none, because its whole cell is the source model's own wording. A crossed board
    records the OTHER side and the digest of the message it rewrote, which is what ties the corner to its
    source board rather than to three fresh draws that happen to sit under the same id.
    """
    paraphraser = entry["paraphraser_model_id"]
    if board_id in RAW_BOARD_IDS:
        if paraphraser is not None:
            raise ValueError(
                f"{what} is a raw board and records paraphraser_model_id {paraphraser!r}. A raw board "
                f"is the source model's own wording, which is the whole of what its cell measures."
            )
        return
    expected_paraphraser = BOARD_MODEL_ID_BY_SIDE[BOARD_WORDING_SIDE[board_id]]
    if str(paraphraser) != expected_paraphraser:
        raise ValueError(
            f"{what} records paraphraser_model_id {paraphraser!r}, and this board's wording is the "
            f"{BOARD_WORDING_SIDE[board_id]} side's, which is {expected_paraphraser!r}."
        )
    if BOARD_SOURCE_DIGEST_FIELD not in entry:
        raise ValueError(
            f"{what} is a paraphrase and records no {BOARD_SOURCE_DIGEST_FIELD}, so nothing ties it to "
            f"the message it rewrote and the crossed corner could hold unrelated content."
        )


def _board_provenance(
    entry: Mapping[str, Any], *, what: str, board_id: str, scenario_id: str, index: int
) -> dict[str, Any]:
    """Read one message's provenance, refusing a missing field or a source outside the roster pair."""
    missing = [name for name in BOARD_PROVENANCE_FIELDS if name not in entry]
    if missing:
        raise ValueError(
            f"{what} provenance is missing {missing}. Every board message is model output this "
            f"repository generated, and a message whose source, transport or sampler is unrecorded "
            f"cannot be told from one an author wrote by hand."
        )
    expected_source = BOARD_MODEL_ID_BY_SIDE[BOARD_CONTENT_SIDE[board_id]]
    if str(entry["source_model_id"]) != expected_source:
        raise ValueError(
            f"{what} records source_model_id {entry['source_model_id']!r}, and this board's content is "
            f"the {BOARD_CONTENT_SIDE[board_id]} side's, which is {expected_source!r}. The condition "
            f"every record carries is derived from the board id, so a board holding another model's "
            f"content is a whole cell labelled as its opposite."
        )
    _assert_the_wording_side_wrote_it(entry, what=what, board_id=board_id)
    if str(entry["task_scenario_id"]) != scenario_id:
        raise ValueError(
            f"{what} records task_scenario_id {entry['task_scenario_id']!r} rather than {scenario_id!r}, "
            f"so it answers another scenario's task while this scenario's lead-in prints its gloss."
        )
    if int(entry["draw"]) != index:
        raise ValueError(
            f"{what} records draw {entry['draw']!r} at board position {index}; the messages are printed "
            f"in the order they came in, so the two have to agree."
        )
    if int(entry["redraws"]) < 0:
        raise ValueError(f"{what} records {entry['redraws']!r} redraws, which cannot be negative.")
    if str(entry["transport"]) not in BOARD_TRANSPORTS:
        raise ValueError(
            f"{what} records transport {entry['transport']!r}, which is neither of "
            f"{list(BOARD_TRANSPORTS)}. The transport is how this message was sampled, and a value the "
            f"generator never writes means the block was edited by hand rather than generated."
        )
    sampling = entry["sampling"]
    if sorted(sampling) != sorted(BOARD_SAMPLING_KEYS):
        raise ValueError(
            f"{what} records the sampler {sorted(sampling)} rather than {list(BOARD_SAMPLING_KEYS)}. "
            f"Board generation runs at the provider defaults with a reply cap and nothing else, the way "
            f"every sampled leg of these passes does."
        )
    if int(sampling["max_tokens"]) != BOARD_SAMPLING_MAX_TOKENS:
        raise ValueError(
            f"{what} records a reply cap of {sampling['max_tokens']!r} rather than "
            f"{BOARD_SAMPLING_MAX_TOKENS}. A cap below that one truncates a message mid-sentence, and "
            f"the board gate's own length rule cannot tell a short message from a cut-off one."
        )
    return dict(entry)


def _board(  # noqa: PLR0913 - one keyword per axis of a board's identity, plus the two escapes
    entry: Mapping[str, Any],
    *,
    scenario_id: str,
    board_id: str,
    path: Path,
    nouns: Sequence[str],
    allow_empty: bool,
) -> Board:
    """Read one board, gating every message and its provenance, or accept an empty slot to be filled."""
    what = f"{BOARDS_FIELD}[{scenario_id!r}][{board_id!r}] in {path}"
    messages = [str(message) for message in entry.get("messages", [])]
    provenance = list(entry.get("provenance", []))
    if not messages:
        if allow_empty:
            return Board(scenario_id=scenario_id, board_id=board_id, messages=(), provenance=())
        raise ValueError(
            f"{what} carries no messages. Boards are generated once by the roster models themselves "
            f"(`transfer_cli.py --pass fingerprint generate-boards`) and frozen before anything is "
            f"sampled, so an empty slot means this pass has no stimulus yet -- not that it may run "
            f"with a board of nothing."
        )
    if len(messages) != BOARD_MESSAGE_COUNT:
        raise ValueError(
            f"{what} carries {len(messages)} messages, expected {BOARD_MESSAGE_COUNT}: one per other "
            f"side at the reference dose, which is the only dose a board is rendered at."
        )
    if len(provenance) != len(messages):
        raise ValueError(
            f"{what} carries {len(messages)} messages and {len(provenance)} provenance entries. "
            f"Provenance is per message because the draw index and the paraphrased message's own source "
            f"digest are per message."
        )
    for index, message in enumerate(messages):
        assert_board_message_is_usable(message, what=f"{what} message {index}", nouns=nouns)
    return Board(
        scenario_id=scenario_id,
        board_id=board_id,
        messages=tuple(messages),
        provenance=tuple(
            _board_provenance(
                entry_provenance,
                what=f"{what} message {index}",
                board_id=board_id,
                scenario_id=scenario_id,
                index=index,
            )
            for index, entry_provenance in enumerate(provenance)
        ),
    )


def _assert_paraphrases_rewrite_their_source_board(
    boards: Mapping[str, Mapping[str, Board]], path: Path
) -> None:
    """Refuse a crossed corner whose messages are not rewrites of its own raw board's, one for one.

    The 2x2's whole reading is that a crossed board holds one side's content in the other's wording, so
    the content part of the split is only a content part if the messages ARE the source board's. A
    generator that drew fresh messages instead would leave every count right and both effects wrong.
    """
    for scenario_id, by_id in boards.items():
        for board_id, source_id in BOARD_SOURCE_BOARD.items():
            board = by_id[board_id]
            source = by_id[source_id]
            if len(board.messages) != len(source.messages):
                raise ValueError(
                    f"{BOARDS_FIELD}[{scenario_id!r}][{board_id!r}] in {path} carries "
                    f"{len(board.messages)} messages against {len(source.messages)} in {source_id!r}, "
                    f"which it rewrites message for message."
                )
            available = {message_digest(message) for message in source.messages}
            unresolved = [
                str(entry.get(BOARD_SOURCE_DIGEST_FIELD))
                for entry in board.provenance
                if str(entry.get(BOARD_SOURCE_DIGEST_FIELD)) not in available
            ]
            if unresolved:
                raise ValueError(
                    f"{BOARDS_FIELD}[{scenario_id!r}][{board_id!r}] in {path} names the source digests "
                    f"{unresolved}, which no message of {source_id!r} has. A crossed board reads as one "
                    f"side's content in the other's wording only if it rewrote that side's messages."
                )


def _boards(
    payload: Mapping[str, Any],
    scenarios: Sequence[TransferScenario],
    path: Path,
    *,
    allow_empty: bool,
) -> dict[str, dict[str, Board]]:
    """Read the generated board table, refusing a missing scenario, a stray board id or a bad message.

    ``allow_empty`` is what lets the file load between the author writing the v3 fields and the
    generator filling them: the slots may be absent or empty, and anything present is still gated.
    Rendering a board cell refuses an empty board either way (:func:`board_clause_for`), so the escape
    cannot become a path by which a board of nothing is sampled. What it does NOT relax is key hygiene:
    a board for a scenario this file has no frame for, or a board id outside the 2x2, refuses in the
    pre-generation window as well, because nothing writes either one on the way to a filled table and a
    stray id that loaded once would be a cell the generator never fills and no reader ever misses.
    """
    raw = payload.get(BOARDS_FIELD)
    if raw is None:
        raise ValueError(
            f"stimulus file {path} carries no {BOARDS_FIELD} field at all. The field is filed as "
            f"prompt-affecting, so it has to exist for the prompt digest to cover it; write it as an "
            f"empty object and fill it with `generate-boards`."
        )
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    stray = sorted(set(raw) - set(scenario_ids))
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in raw]
    # A stray key refuses in the pre-generation window too: nothing legitimate writes a board for a
    # scenario this file does not carry, and the escape is there for slots not yet FILLED rather than
    # for a table whose keys are wrong. Only the missing half is what `allow_empty` relaxes.
    if stray or (missing and not allow_empty):
        raise ValueError(
            f"stimulus file {path} carries the wrong {BOARDS_FIELD} table: scenarios with no board "
            f"{missing or 'none'}; boards for scenarios this file does not carry {stray or 'none'}."
        )
    present = [scenario_id for scenario_id in scenario_ids if scenario_id in raw]
    nouns = scenario_nouns(scenarios)
    boards: dict[str, dict[str, Board]] = {}
    for scenario_id in present:
        entry = raw[scenario_id]
        missing_ids = [board_id for board_id in BOARD_IDS if board_id not in entry]
        stray_ids = sorted(set(entry) - set(BOARD_IDS))
        if stray_ids or (missing_ids and not allow_empty):
            raise ValueError(
                f"{BOARDS_FIELD}[{scenario_id!r}] in {path} misses the board ids "
                f"{missing_ids or 'none'} and carries the stray ids {stray_ids or 'none'}. The four "
                f"ids are the corners of one 2x2 and the readout splits the lift over all four."
            )
        boards[scenario_id] = {
            board_id: _board(
                entry.get(board_id, {}),
                scenario_id=scenario_id,
                board_id=board_id,
                path=path,
                nouns=nouns,
                allow_empty=allow_empty,
            )
            for board_id in BOARD_IDS
        }
    if not allow_empty:
        _assert_paraphrases_rewrite_their_source_board(boards, path)
    return boards


def _expanded_record_sentence(template: str, *, matched: int, rounds: int) -> str:
    """Fill the record template's two counts, leaving every transfer placeholder for the clause path.

    Substitution rather than ``str.format``: the sentence also carries the transfer placeholders, which
    are filled per scenario and per dose much later, and a ``format`` call here would refuse them as
    missing keys -- which is also why the loader's own placeholder check on the EXPANDED sentence must
    see no ``matched`` or ``rounds`` left in it.
    """
    return template.replace(f"{{{MATCHED_PLACEHOLDER}}}", str(matched)).replace(
        f"{{{ROUNDS_PLACEHOLDER}}}", str(rounds)
    )


def _assert_the_ladder_is_readable(rounds: int, path: Path) -> None:
    """Refuse a rounds count the ladder does not fit inside, or a ladder that repeats a rung."""
    if rounds != TRACK_RECORD_ROUNDS:
        raise ValueError(
            f"{TRACK_RECORD_ROUNDS_FIELD} in {path} is {rounds!r}, and this pass's ladder is stated "
            f"out of {TRACK_RECORD_ROUNDS}. The rung ids carry the numerator alone, so a different "
            f"denominator would relabel every rung's probability while every id stayed the same."
        )
    outside = [matched for matched in MATCHED_ROUNDS_LADDER if not 0 <= matched <= rounds]
    repeated = sorted(
        {matched for matched in MATCHED_ROUNDS_LADDER if MATCHED_ROUNDS_LADDER.count(matched) > 1}
    )
    if outside or repeated:
        raise ValueError(
            f"the matched-rounds ladder {MATCHED_ROUNDS_LADDER} is unreadable against "
            f"{rounds} rounds: rungs outside 0 to {rounds} {outside or 'none'}; repeated rungs "
            f"{repeated or 'none'}."
        )


def _assert_record_sentences_differ_in_the_numeral_alone(
    sentences: Mapping[str, str], path: Path
) -> None:
    """Refuse a derived ladder whose rungs differ anywhere but in the stated count.

    The curve reads a step between two rungs as a response to the count. Two sentences differing in a
    second token -- which is what a numeral spelled in words, or a plural agreeing with the count, would
    produce -- would put that token inside every step of the curve.
    """
    for rung_a, rung_b in (
        (rung_a, rung_b)
        for index, rung_a in enumerate(sentences)
        for rung_b in list(sentences)[index + 1 :]
    ):
        matched_a, matched_b = RECORD_RUNG_MATCHED[rung_a], RECORD_RUNG_MATCHED[rung_b]
        tokens_a, tokens_b = sentences[rung_a].split(), sentences[rung_b].split()
        if matched_a == matched_b:
            if sentences[rung_a] != sentences[rung_b]:
                raise ValueError(
                    f"the derived rungs {rung_a!r} and {rung_b!r} in {path} state the same count "
                    f"{matched_a} and render different sentences ({sentences[rung_a]!r} against "
                    f"{sentences[rung_b]!r}), so their difference is not the count."
                )
            continue
        differing = (
            [
                index
                for index, (token_a, token_b) in enumerate(zip(tokens_a, tokens_b, strict=True))
                if token_a != token_b
            ]
            if len(tokens_a) == len(tokens_b)
            else None
        )
        if differing is None or len(differing) != 1:
            raise ValueError(
                f"the derived rungs {rung_a!r} and {rung_b!r} in {path} do not differ in exactly one "
                f"whitespace-delimited token ({sentences[rung_a]!r} against {sentences[rung_b]!r}). "
                f"Word the template so the count is the only thing that moves: no plural agreeing with "
                f"it, and the count as a numeral rather than in words."
            )
        stated = {
            tokens_a[differing[0]].strip(".,;:!?"),
            tokens_b[differing[0]].strip(".,;:!?"),
        }
        if stated != {str(matched_a), str(matched_b)}:
            raise ValueError(
                f"the derived rungs {rung_a!r} and {rung_b!r} in {path} differ in the token {stated}, "
                f"which is not the pair of stated counts {{{matched_a}, {matched_b}}}."
            )


def _track_record_sentences(payload: Mapping[str, Any], path: Path) -> dict[str, str]:
    """Derive the dose ladder's appended sentences from the one authored template.

    Every check here is a property the curve's reading rests on. The template is exactly one sentence,
    because the rung is reported as one appended sentence. It states no mechanism and no reason ("no
    because"), because the ladder is a stated count and a reason would be a second manipulation. And
    the ten-of-ten rung must not render the knockout's universal sentence, which is sampled beside it
    in the same block: two cells rendering one text would be a measurement of nothing.
    """
    template = str(payload.get(TRACK_RECORD_TEMPLATE_FIELD, ""))
    if not template.strip():
        raise ValueError(
            f"stimulus file {path} carries no {TRACK_RECORD_TEMPLATE_FIELD}. Every rung of the dose "
            f"ladder is derived from it, so without it the ladder's nine cells have no clause at all."
        )
    for placeholder in (MATCHED_PLACEHOLDER, ROUNDS_PLACEHOLDER):
        if f"{{{placeholder}}}" not in template:
            raise ValueError(
                f"{TRACK_RECORD_TEMPLATE_FIELD} in {path} does not name {{{placeholder}}}, so every "
                f"rung of the ladder would render the same sentence under nine cell ids."
            )
    _assert_no_unsupported_placeholder(
        TRACK_RECORD_TEMPLATE_FIELD,
        template,
        path,
        supported=(*PLACEHOLDERS, MATCHED_PLACEHOLDER, ROUNDS_PLACEHOLDER),
    )
    if re.search(r"\bbecause\b", template, re.IGNORECASE):
        raise ValueError(
            f"{TRACK_RECORD_TEMPLATE_FIELD} in {path} says 'because'. The sentence states a count and "
            f"nothing else: a reason for the count is a second manipulation inside one appended "
            f"sentence, and the reading would report it as the count alone."
        )
    assert_no_loaded_vocabulary(template)
    rounds = int(payload.get(TRACK_RECORD_ROUNDS_FIELD, 0))
    _assert_the_ladder_is_readable(rounds, path)
    return {
        rung: _expanded_record_sentence(template, matched=matched, rounds=rounds)
        for rung, matched in RECORD_RUNG_MATCHED.items()
    }


def board_clause_for(
    stimulus: TransferStimulus, *, scenario: TransferScenario, board_id: str, spec: TransferSpec
) -> str:
    """Build one board cell's counterpart paragraph: the lead-in, then the messages, one per line group.

    By concatenation rather than ``str.format``, and that is why the clause path dispatches here instead
    of resolving a template: the messages are text models wrote, so a brace in one would break a format
    call and the count-one rewrite would reach into a model's prose. The lead-in IS filled, because it
    is a template this repository authored and the loader checks its placeholders.
    """
    if board_id not in BOARD_IDS:
        raise ValueError(f"{board_id!r} is not a board cell; the boards are {list(BOARD_IDS)}.")
    if spec.beneficiary_count != BOARD_MESSAGE_COUNT:
        raise ValueError(
            f"board {board_id!r} was asked for at {spec.beneficiary_count} beneficiaries and boards "
            f"exist at {BOARD_MESSAGE_COUNT} only: one message per other side. A board of three "
            f"messages read at one beneficiary is a different manipulation under the same cell id."
        )
    task = stimulus.unrelated_tasks.get(scenario.scenario_id)
    if task is None:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} has no board task, so the lead-in has no gloss to print."
        )
    board = stimulus.boards.get(scenario.scenario_id, {}).get(board_id)
    if board is None or not board.messages:
        raise ValueError(
            f"board {board_id!r} of scenario {scenario.scenario_id!r} carries no messages, so this cell "
            f"cannot be rendered or sampled. Generate the boards first "
            f"(`transfer_cli.py --pass fingerprint generate-boards`) and freeze the file."
        )
    lead_in = stimulus.board_lead_in.format(
        count=spec.beneficiary_count,
        beneficiary_noun=scenario.beneficiary_noun,
        task_gloss=task.task_gloss,
    )
    return BOARD_DELIMITER.join((lead_in, *board.messages))


def _cells_with_a_clause_template() -> tuple[tuple[str, str], ...]:
    """Name every (game, cell) whose paragraph is composed from a template.

    Every cell but the boards, whose paragraph is a lead-in plus messages that models wrote and
    :func:`board_clause_for` concatenates.
    """
    return tuple(
        (game_id, rung)
        for game_id, rungs in RUNGS_BY_GAME.items()
        for rung in rungs
        if rung not in BOARD_IDS
    )


def _clause_templates(
    payload: Mapping[str, Any], path: Path
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], dict[str, str]]:
    """Compose every cell's clause template, the singular person forms, and the derived record rungs.

    The derived sentences travel back out because the loader stores them: the dose readout reads a rung
    against the count its sentence states, and recomputing them at the point of use would be a second
    expansion of the template to keep in step with this one.
    """
    raw_fragments = payload.get("identity_fragments", {})
    missing = [rung for rung in LADDER_RUNGS if rung not in raw_fragments]
    if missing:
        raise ValueError(f"stimulus file {path} is missing identity fragments for {missing}")
    fragments = {rung: str(raw_fragments[rung]) for rung in LADDER_RUNGS}
    openings = {
        TAIL_KEY_ONE_WAY: str(payload.get("one_way_opening", "")),
        TAIL_KEY_TWIN: str(payload.get("twin_opening", "")),
    }
    # The coupled and drawn cells take the twin's opening: all three describe the same parties in the
    # same words, and only the tail says what those parties then do.
    openings[TAIL_KEY_TWIN_COUPLED] = openings[TAIL_KEY_TWIN]
    openings[TAIL_KEY_DRAWN] = openings[TAIL_KEY_TWIN]
    tails = {
        TAIL_KEY_ONE_WAY: str(payload.get("one_way_tail", "")),
        TAIL_KEY_TWIN: str(payload.get("twin_tail", "")),
        TAIL_KEY_TWIN_COUPLED: str(payload.get("twin_coupled_tail", "")),
        TAIL_KEY_DRAWN: str(payload.get("drawn_tail", "")),
    }
    appended = _appended_sentences(payload, path)
    person_tails = _person_tails(payload, PERSON_TAILS_FIELD, path)
    person_tails_singular = _person_tails(payload, PERSON_TAILS_SINGULAR_FIELD, path)
    blank = sorted(
        name
        for name, text in {
            **{f"{key}_opening": openings[key] for key in (TAIL_KEY_ONE_WAY, TAIL_KEY_TWIN)},
            **{f"{key}_tail": tails[key] for key in TAIL_KEYS},
            **{f"{PERSON_TAILS_FIELD}.{key}": person_tails[key] for key in TAIL_KEYS},
            **{
                f"{PERSON_TAILS_SINGULAR_FIELD}.{key}": person_tails_singular[key]
                for key in TAIL_KEYS
            },
        }.items()
        if not text.strip()
    )
    if blank:
        raise ValueError(
            f"stimulus file {path} leaves {blank} blank; every one is printed verbatim."
        )
    if tails[TAIL_KEY_TWIN_COUPLED] == tails[TAIL_KEY_TWIN]:
        raise ValueError(
            f"stimulus file {path} gives the coupled twin cell the same tail as every other twin rung, "
            f"so that cell would be a second copy of {RUNG_SAME_CHECKPOINT!r} rather than the coupled "
            f"comparison it exists to be."
        )
    if tails[TAIL_KEY_DRAWN] == tails[TAIL_KEY_TWIN]:
        raise ValueError(
            f"stimulus file {path} gives the drawn game the twin's own tail. The twin's tail says the "
            f"other sides decide, which the drawn game's mechanics deny, so the prompt would contradict "
            f"itself exactly where the knockout needs it to be unambiguous -- and a reply that mirrored "
            f"anyway would be read as the draw failing to land rather than as the prompt being unclear."
        )
    shared = {
        **{f"{key} opening": openings[key] for key in openings},
        **{f"{key} tail": tails[key] for key in tails},
    }
    for rung, fragment in fragments.items():
        _assert_fragment_well_formed(rung, fragment, shared, path)
        _assert_no_unsupported_placeholder(f"identity fragment {rung!r}", fragment, path)
    for name, text in {
        **shared,
        **{f"{PERSON_TAILS_FIELD}.{key}": person_tails[key] for key in TAIL_KEYS},
        **{f"{PERSON_TAILS_SINGULAR_FIELD}.{key}": person_tails_singular[key] for key in TAIL_KEYS},
    }.items():
        _assert_no_unsupported_placeholder(name, text, path)
    _assert_derived_fragments(fragments, path)
    templates: dict[tuple[str, str], str] = {}
    singular: dict[tuple[str, str], str] = {}
    for game_id, rung in _cells_with_a_clause_template():
        key = _tail_key(game_id, rung)
        base = _BASE_RUNG.get(rung, rung)
        tail = person_tails[key] if base == RUNG_PERSON else tails[key]
        templates[game_id, rung] = _compose(
            opening=openings[key], fragment=fragments[base], tail=tail
        ) + appended.get(rung, "")
        if base == RUNG_PERSON:
            singular[game_id, rung] = _compose(
                opening=openings[key],
                fragment=fragments[base],
                tail=person_tails_singular[key],
            )
    _assert_appended_rungs_are_their_base_plus_one_span(templates, path)
    return templates, singular, {rung: appended[rung] for rung in DERIVED_APPENDED_RUNG_BASE}


def _assert_appended_rungs_are_their_base_plus_one_span(
    templates: Mapping[tuple[str, str], str], path: Path
) -> None:
    """Refuse an appended rung that is not its base cell plus exactly one contiguous span.

    The composition builds it by concatenation, so this holds by construction today -- and it is checked
    anyway, because it is the identity the reading rests on and the composition is one edit away from
    being reordered or from filling the base and the rung from different fragments. Recomputed with the
    same one-insertion test the sibling fragments use, so all three derived-cell shapes are proved the
    same way.
    """
    for game_id, rung in templates:
        base_rung = APPENDED_BASE_RUNG.get(rung)
        if base_rung is None:
            continue
        base = templates.get((game_id, base_rung))
        if base is None:
            raise ValueError(
                f"cell {game_id!r}/{rung!r} in {path} extends {base_rung!r}, which {game_id!r} does "
                f"not sample. The appended rung is read against that base cell in the same run, so a "
                f"game carrying one without the other reports a contrast against a cell nobody sampled."
            )
        if inserted_fragment(base, templates[game_id, rung]) is None:
            raise ValueError(
                f"cell {game_id!r}/{rung!r} in {path} is not {base_rung!r} plus exactly one contiguous "
                f"appended span: deleting one span of {templates[game_id, rung]!r} does not reproduce "
                f"{base!r}. That identity is what makes the pair's difference a read of the appended "
                f"sentence alone rather than of two rewordings."
            )


def _validation_reply(
    entry: Mapping[str, Any], game_id: str, path: Path, *, keys: tuple[str, ...] | None = None
) -> TransferValidationReply:
    """Read one validation reply, naming the file and the reply when an expectation is missing.

    ``keys`` overrides the arm's own verdict keys, which is what the intent check needs: it runs one
    rubric over both games and registers its own three fields rather than the arm's eight.
    """
    name = str(entry.get("name", "<unnamed>"))
    expected = entry.get("expected", {})
    keys = VERDICT_KEYS_BY_GAME[game_id] if keys is None else keys
    missing = [key for key in keys if key not in expected]
    if missing:
        raise ValueError(
            f"validation reply {name!r} for {game_id!r} in {path} registers no expectation for "
            f"{missing}; all {len(keys)} verdict fields are compared, so an absent one would go "
            f"unchecked while the report still read as agreement."
        )
    polarity = str(entry.get("polarity", ""))
    if polarity not in ANSWER_POLARITIES:
        raise ValueError(
            f"validation reply {name!r} in {path} has polarity {polarity!r}, not one of "
            f"{list(ANSWER_POLARITIES)}. The judge is told which tag the reply was asked for, so a "
            f"case with no polarity would validate a prompt production never sends."
        )
    return TransferValidationReply(
        name=name,
        game_id=game_id,
        polarity=polarity,
        endowment=int(entry.get("endowment", TRANSFER_ENDOWMENT)),
        reply=str(entry["reply"]),
        reasoning=str(entry.get("reasoning", "")),
        expected={key: expected[key] for key in keys},
    )


def _intent_validation_reply(entry: Mapping[str, Any], path: Path) -> TransferValidationReply:
    """Read one intent-check validation reply, whose own entry names which game's rules it answered.

    The game id travels per reply rather than per section because one rubric covers both arms and the two
    arms state opposite rules about what comes back to the writer -- which is the whole subject of the
    ``assumes_return`` field. A reply with no game id, or one naming neither arm, would be validated
    against whichever set of rules the renderer happened to pick, so it refuses here.
    """
    name = str(entry.get("name", "<unnamed>"))
    game_id = str(entry.get("game_id", ""))
    if game_id not in TRANSFER_GAME_IDS:
        raise ValueError(
            f"intent-check validation reply {name!r} in {path} names game {game_id!r}, not one of "
            f"{list(TRANSFER_GAME_IDS)}. The rules a reply is read against decide whether a symmetric "
            f"payoff in its reasoning is a misreading or the actual rule, so a case with no game would "
            f"validate the check against rules nobody chose."
        )
    return _validation_reply(entry, game_id, path, keys=INTENT_VERDICT_KEYS)


_VALIDATION_FIELD_BY_GAME: dict[str, str] = {
    ONE_WAY_TRANSFER_GAME_ID: "validation_replies_one_way",
    MATCHED_DECISION_TRANSFER_GAME_ID: "validation_replies_twin",
    DRAWN_DECISION_TRANSFER_GAME_ID: "validation_replies_twin",
}

_INSTRUCTIONS_FIELD_BY_GAME: dict[str, str] = {
    ONE_WAY_TRANSFER_GAME_ID: "judge_instructions_one_way",
    MATCHED_DECISION_TRANSFER_GAME_ID: "judge_instructions_twin",
    DRAWN_DECISION_TRANSFER_GAME_ID: "judge_instructions_twin",
}


def _judge_arm_game_ids() -> tuple[str, ...]:
    """Name one game per distinct rubric field, in registry order."""
    seen: set[str] = set()
    arms: list[str] = []
    for game_id in TRANSFER_GAME_IDS:
        field_name = _INSTRUCTIONS_FIELD_BY_GAME[game_id]
        if field_name in seen:
            continue
        seen.add(field_name)
        arms.append(game_id)
    return tuple(arms)


JUDGE_ARM_GAME_IDS: tuple[str, ...] = _judge_arm_game_ids()
"""One game per distinct rubric, in registry order: the arms a validation pass has to read.

The drawn game shares the twin's rubric and validation list, so validating per game id would hand the
same authored replies to the same reader twice under two keys -- twice the calls, and every miss
reported twice.
"""


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    """Digest a JSON-shaped payload canonically: sorted keys, no whitespace, first sixteen hex chars."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def prompt_digest_of(payload: Mapping[str, Any]) -> str:
    """Digest only the sections a rendered prompt depends on, so a judge-side edit leaves it alone."""
    return _canonical_digest({name: payload[name] for name in PROMPT_AFFECTING_FIELDS})


def _assert_every_field_is_classified(payload: Mapping[str, Any], path: Path) -> None:
    """Refuse a top-level field that is filed as neither prompt-affecting nor judge-side.

    Filing is what makes the two digests mean what they claim: a field on neither list could change
    what the model reads without moving ``prompt_digest``, and a resume would then continue a file
    sampled under different prompts while every count still added up.
    """
    unclassified = sorted(set(payload) - set(PROMPT_AFFECTING_FIELDS) - set(JUDGE_SIDE_FIELDS))
    if unclassified:
        raise ValueError(
            f"stimulus file {path} carries top-level fields {unclassified} that are filed as neither "
            f"prompt-affecting ({list(PROMPT_AFFECTING_FIELDS)}) nor judge-side "
            f"({list(JUDGE_SIDE_FIELDS)}). Add each one to the list that matches what reads it, so the "
            f"prompt digest moves when the prompts do and stays put when only the judge's material does."
        )


def _intent_material(
    payload: Mapping[str, Any], path: Path
) -> tuple[str, tuple[TransferValidationReply, ...]]:
    """Read the intent check's rubric and its authored cases, refusing a thin set by name.

    Two floors, because they fail differently. The pooled floor says the coverage list does not fit
    below it; the per-game floor says one of the three rules paragraphs has no case read under it at all,
    which leaves that paragraph's reader unchecked while the pooled count still clears.
    """
    instructions = str(payload.get(INTENT_RUBRIC_FIELD, ""))
    if not instructions.strip():
        raise ValueError(
            f"stimulus file {path} carries no {INTENT_RUBRIC_FIELD}. The intent check is the instrument "
            f"that reads what a reply's reasoning concluded against the tag it then wrote, and without "
            f"its rubric every tag slip in the run stays scored as the tag."
        )
    raw = payload.get(INTENT_VALIDATION_FIELD, [])
    if len(raw) < MIN_INTENT_VALIDATION_REPLIES:
        raise ValueError(
            f"stimulus file {path} carries {len(raw)} {INTENT_VALIDATION_FIELD}, and the intent "
            f"check is not calibrated on fewer than {MIN_INTENT_VALIDATION_REPLIES}: its coverage list "
            f"(a clean conclusion each way, a contradicted tag in each polarity, an assumed return, a "
            f"bare tag, an interior figure, and a clean conclusion and a slip under the drawn rules) "
            f"does not fit below that."
        )
    cases = tuple(_intent_validation_reply(entry, path) for entry in raw)
    thin = sorted(
        game_id
        for game_id in TRANSFER_GAME_IDS
        if sum(1 for case in cases if case.game_id == game_id)
        < MIN_INTENT_VALIDATION_REPLIES_PER_GAME
    )
    if thin:
        raise ValueError(
            f"stimulus file {path} carries fewer than {MIN_INTENT_VALIDATION_REPLIES_PER_GAME} "
            f"{INTENT_VALIDATION_FIELD} for {thin}. A case's game decides which rules paragraph it is "
            f"read under, and the paragraphs say opposite things about what reaches the writer's own "
            f"tally, so a game with no case leaves that paragraph's reader unchecked."
        )
    return instructions, cases


def load_stimulus(
    path: Path = STIMULUS_PATH, *, allow_empty_boards: bool = False
) -> TransferStimulus:
    """Load and validate the transfer stimulus file, refusing absence loudly rather than defaulting.

    A default here would either be committed stimulus prose, which this public repository must never
    carry, or empty strings, which would render frameless prompts that measure nothing the design
    describes. The refusal names the path and why the file is machine-local.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"stimulus file {path} is missing. It is gitignored on purpose (the scenario frames, the "
            "identity fragments, the judge rubrics and the validation replies are authored stimulus "
            "that must never be committed); a fresh clone does not contain it. Recreate it from the "
            "design doc in docs/scratch/."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != STIMULUS_VERSION:
        raise ValueError(
            f"stimulus file {path} has version {version!r}, expected {STIMULUS_VERSION!r}"
        )
    _assert_every_field_is_classified(payload, path)
    entries = payload.get("scenarios", [])
    if len(entries) != N_SCENARIOS:
        raise ValueError(
            f"stimulus file {path} carries {len(entries)} scenarios, expected exactly {N_SCENARIOS}: "
            f"the design's unit of pairing is (scenario, polarity), and a per-cell mean over fewer "
            f"frames is one fiction's answer rather than a rate."
        )
    scenarios = tuple(_scenario(entry, path) for entry in entries)
    duplicated = sorted(
        {
            scenario.scenario_id
            for scenario in scenarios
            if sum(1 for other in scenarios if other.scenario_id == scenario.scenario_id) > 1
        }
    )
    if duplicated:
        raise ValueError(
            f"stimulus file {path} carries duplicate scenario ids {duplicated}; a scenario id keys "
            f"every prompt_id and record key, so two frames under one id would pool their answers."
        )
    templates, singular_templates, record_sentences = _clause_templates(payload, path)
    board_lead_in = _board_lead_in(payload, path)
    tasks = _unrelated_tasks(payload, scenarios, path)
    paraphrase_instruction = _paraphrase_instruction(payload, path)
    boards = _boards(payload, scenarios, path, allow_empty=allow_empty_boards)
    instructions: dict[str, str] = {}
    validation: dict[str, tuple[TransferValidationReply, ...]] = {}
    for game_id in TRANSFER_GAME_IDS:
        text = str(payload.get(_INSTRUCTIONS_FIELD_BY_GAME[game_id], ""))
        if not text.strip():
            raise ValueError(
                f"stimulus file {path} carries no {_INSTRUCTIONS_FIELD_BY_GAME[game_id]}"
            )
        instructions[game_id] = text
        raw = payload.get(_VALIDATION_FIELD_BY_GAME[game_id], [])
        floor = MIN_VALIDATION_REPLIES[game_id]
        if len(raw) < floor:
            raise ValueError(
                f"stimulus file {path} carries {len(raw)} validation replies for {game_id!r}, and "
                f"that judge is not calibrated on fewer than {floor}: the design's own coverage list "
                f"for this arm does not fit below it."
            )
        validation[game_id] = tuple(_validation_reply(entry, game_id, path) for entry in raw)
    intent_instructions, intent_validation = _intent_material(payload, path)
    stimulus = TransferStimulus(
        scenarios=scenarios,
        clause_templates=templates,
        singular_clause_templates=singular_templates,
        board_lead_in=board_lead_in,
        unrelated_tasks=tasks,
        paraphrase_instruction=paraphrase_instruction,
        boards=boards,
        track_record_sentences=record_sentences,
        judge_instructions=instructions,
        validation_replies=validation,
        intent_instructions=intent_instructions,
        intent_validation_replies=intent_validation,
        digest=_canonical_digest(payload),
        prompt_digest=prompt_digest_of(payload),
    )
    _assert_every_cell_renders(stimulus, path, boards_renderable=not allow_empty_boards)
    return stimulus


def _assert_every_cell_renders(
    stimulus: TransferStimulus, path: Path, *, boards_renderable: bool
) -> None:
    """Fill every cell of every game at both the plural and the singular dose, and check the prose.

    At load rather than at plan time, because a template that only breaks at the single-beneficiary dose
    would otherwise surface halfway through a sampling run: the counts would already be written down and
    the leg half submitted.

    Board cells are rendered at the reference count alone, because that is the only count they exist at
    (:data:`BOARD_MESSAGE_COUNT`), and skipped entirely while the boards are still empty -- the one state
    ``allow_empty_boards`` describes, in which nothing of pass D can be rendered yet by design.
    """
    cells = [
        (game_id, rung)
        for game_id, rungs in RUNGS_BY_GAME.items()
        for rung in rungs
        if rung not in BOARD_IDS or boards_renderable
    ]
    for game_id, rung in cells:
        spec = reference_spec(game_id)
        counts = (spec.beneficiary_count,) if rung in BOARD_IDS else (spec.beneficiary_count, 1)
        for count in counts:
            for scenario in stimulus.scenarios:
                clause = clause_for(
                    stimulus,
                    game_id=game_id,
                    rung=rung,
                    spec=TransferSpec(
                        game_id=game_id,
                        endowment=spec.endowment,
                        credit_numerator=spec.credit_numerator,
                        credit_denominator=spec.credit_denominator,
                        beneficiary_count=count,
                        own_stake_scale=spec.own_stake_scale,
                    ),
                    scenario=scenario,
                )
                assert_no_loaded_vocabulary(about_the_other_side(clause))
                if "{" in clause or "}" in clause:
                    raise ValueError(
                        f"clause {game_id!r}/{rung!r} in {path} still carries a brace after filling "
                        f"({clause!r}); an unfilled placeholder reaches the model verbatim."
                    )


def _rewritten_for_one(template: str, *, game_id: str, rung: str) -> str:
    """Rewrite a plural template for the single-beneficiary dose, then check nothing plural survived."""
    rewritten = template
    for plural, singular in COUNT_ONE_SUBSTITUTIONS:
        rewritten = rewritten.replace(plural, singular)
    surviving = [marker for marker in PLURAL_ONLY_MARKERS if marker in rewritten]
    if surviving:
        raise ValueError(
            f"the clause for {game_id!r}/{rung!r} still carries the plural-only text {surviving} after "
            f"the count-one rewrite, so at one beneficiary it would render prose no author wrote. Add "
            f"the missing pair to COUNT_ONE_SUBSTITUTIONS, or reword the fragment. Rewritten: "
            f"{rewritten!r}"
        )
    return rewritten


def clause_for(
    stimulus: TransferStimulus,
    *,
    game_id: str,
    rung: str,
    spec: TransferSpec,
    scenario: TransferScenario,
) -> str:
    """Fill one cell's clause template for one dose and one scenario.

    The one resolver for every game, so a cell name is interchangeable everywhere downstream and a rung
    of one game is provably the same identity fragment as the same rung of the other.

    A board cell has no template and is dispatched to :func:`board_clause_for`. It has to be dispatched
    rather than resolved because the template path is a ``str.format`` over a count-one rewrite: a
    message carrying a brace would break the fill, and one carrying "each" would be silently rewritten.
    """
    if rung in BOARD_IDS:
        if rung not in RUNGS_BY_GAME.get(game_id, ()):
            raise ValueError(f"{rung!r} is not a cell of {game_id!r}; boards ride the twin alone.")
        return board_clause_for(stimulus, scenario=scenario, board_id=rung, spec=spec)
    template = stimulus.clause_templates.get((game_id, rung))
    if template is None:
        known = sorted(rung for game, rung in stimulus.clause_templates if game == game_id)
        raise ValueError(f"{rung!r} is not a clause cell of {game_id!r}; known cells: {known}.")
    singular = spec.beneficiary_count == 1
    if singular:
        authored = stimulus.singular_clause_templates.get((game_id, rung))
        template = _rewritten_for_one(
            template if authored is None else authored, game_id=game_id, rung=rung
        )
    return template.format(
        count=spec.beneficiary_count,
        beneficiary_noun=(
            scenario.beneficiary_noun_singular if singular else scenario.beneficiary_noun
        ),
        note_noun=scenario.note_noun,
        resource=scenario.resource,
    )
