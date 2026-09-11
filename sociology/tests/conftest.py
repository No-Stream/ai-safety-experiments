"""Shared test scaffolding: a synthetic stimulus file, synthetic units, and a scripted backend.

Everything here is synthetic on purpose. The real stimulus file carries authored frame texts and
the judge rubric, and the real corpus carries transcripts -- none of which may appear in tracked
test files (this repository is public and its history will be published). Tests exercise shape and
behaviour against clearly artificial texts.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from games.prompts import SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE
from sociology.corpus import BundleUnit
from sociology.decoupled_stimulus import (
    AUTHORED_RUNGS,
    BRIEFING_PHRASE,
    DECOUPLING_TAILS,
    DERIVED_RUNG_CLAUSES,
    RUNG_SAME_FAMILY_LARGER,
    RUNG_SAME_FAMILY_SMALLER,
    RUNG_SIBLING_ADAPTER,
    DecoupledStimulus,
)
from sociology.decoupled_stimulus import STIMULUS_VERSION as DECOUPLED_STIMULUS_VERSION
from sociology.decoupled_stimulus import load_stimulus as load_decoupled_stimulus
from sociology.deference_stimulus import (
    INTENT_VERDICT_KEYS as DEFERENCE_INTENT_VERDICT_KEYS,
)
from sociology.deference_stimulus import (
    MIN_INTENT_VALIDATION_REPLIES as MIN_DEFERENCE_INTENT_REPLIES,
)
from sociology.deference_stimulus import (
    MIN_VALIDATION_REPLIES as MIN_DEFERENCE_VALIDATION_REPLIES,
)
from sociology.deference_stimulus import (
    PEER_COUNT,
    DeferenceStimulus,
)
from sociology.deference_stimulus import STIMULUS_VERSION as DEFERENCE_STIMULUS_VERSION
from sociology.deference_stimulus import (
    VERDICT_KEYS as DEFERENCE_VERDICT_KEYS,
)
from sociology.deference_stimulus import load_stimulus as load_deference_stimulus
from sociology.model_stub import ScriptedDetailedBackend
from sociology.stimulus import Stimulus, load_stimulus
from sociology.tests.synthetic_fingerprint_dose import (
    TWIN_ONLY_BOOL_FLAGS,
    synthetic_v3_fields,
)
from sociology.transfer_stimulus import (
    APPENDED_SENTENCES_FIELD,
    BOARDS_FIELD,
    MIN_VALIDATION_REPLIES,
    RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
    TAIL_KEY_DRAWN,
    TAIL_KEY_ONE_WAY,
    TAIL_KEY_TWIN,
    TAIL_KEY_TWIN_COUPLED,
    VERDICT_KEYS_BY_GAME,
    TransferStimulus,
)
from sociology.transfer_stimulus import STIMULUS_VERSION as TRANSFER_STIMULUS_VERSION
from sociology.transfer_stimulus import load_stimulus as load_transfer_stimulus

if TYPE_CHECKING:
    from pathlib import Path

SYNTHETIC_STIMULUS = {
    "version": "analysis-model-stimulus-v1",
    "frames": {
        "population": "SYNTHETIC-FRAME-POPULATION for {n} transcripts.",
        "independent": "SYNTHETIC-FRAME-INDEPENDENT for {n} transcripts.",
        "unstated": "SYNTHETIC-FRAME-UNSTATED {n}.",
    },
    "constant_request": "SYNTHETIC-REQUEST: describe the patterns.",
    "judge_instructions": "SYNTHETIC-RUBRIC: reply with one JSON object.",
    "validation_replies": [
        {"name": "v-none", "text": "a plain reply", "expected_severity": "none"},
        {
            "name": "v-coord",
            "text": "they plainly agreed",
            "expected_severity": "explicit_coordination",
        },
    ],
}


def write_synthetic_stimulus(path: Path) -> Path:
    path.write_text(json.dumps(SYNTHETIC_STIMULUS), encoding="utf-8")
    return path


@pytest.fixture
def stimulus(tmp_path: Path) -> Stimulus:
    return load_stimulus(write_synthetic_stimulus(tmp_path / "stimulus.json"))


def make_unit(
    unit_id: str,
    family: str = "agentic-120b",
    exclusion_key: str | None = None,
    rendered: str | None = None,
) -> BundleUnit:
    return BundleUnit(
        unit_id=unit_id,
        family=family,
        exclusion_key=exclusion_key if exclusion_key is not None else unit_id,
        rendered=rendered if rendered is not None else f"synthetic transcript body for {unit_id}",
    )


# --- decoupled-ladder pass ------------------------------------------------------------------------
#
# The real ladder clauses are authored stimulus and never appear in tracked code, so the synthetic
# ones here are built from the two PUBLIC shared parts (the opening, the simultaneity phrase and the
# decoupling tail, all of which the loader checks for) around an obviously artificial identity.
#
# Three rungs cannot be freely synthetic, because the loader recomputes them: `different-family` and
# `person` ARE public constants plus the shared denial sentence, `sibling-adapter` must be the public
# same-weights clause with exactly one fragment inserted, and the two same-family rungs must differ in
# the size words alone. The first two come straight from the public derivation (they carry no authored
# prose at all), and the others are synthetic text shaped to satisfy the identity the loader checks.

_ITS_TAIL, _THEIR_TAIL = DECOUPLING_TAILS

DECOUPLED_JUDGE_INSTRUCTIONS = "SYNTHETIC-RUBRIC: reply with one JSON object."

_SAME_WEIGHTS_FRAGMENT_ANCHOR = "running the same weights as you"
_SYNTHETIC_SIBLING_INSERTION = " plus SYNTHETIC-LATER-ADJUSTMENTS"


def _synthetic_size_clause(size_word: str, count_word: str) -> str:
    """One of the two same-family rungs; they may differ in the size words and nowhere else."""
    return (
        f"you are matched with a SYNTHETIC-COUNTERPART built to the same design as you but "
        f"{size_word}, with {count_word} layers, {BRIEFING_PHRASE}, {_ITS_TAIL}"
    )


def synthetic_rung_clause(rung: str) -> str:
    """Build one synthetic ladder clause that satisfies every constraint the loader checks."""
    derived = DERIVED_RUNG_CLAUSES.get(rung)
    if derived is not None:
        return derived
    if rung == RUNG_SIBLING_ADAPTER:
        return SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE.replace(
            _SAME_WEIGHTS_FRAGMENT_ANCHOR,
            _SAME_WEIGHTS_FRAGMENT_ANCHOR + _SYNTHETIC_SIBLING_INSERTION,
            1,
        )
    if rung == RUNG_SAME_FAMILY_SMALLER:
        return _synthetic_size_clause("smaller", "fewer")
    if rung == RUNG_SAME_FAMILY_LARGER:
        return _synthetic_size_clause("larger", "more")
    return f"you are matched with SYNTHETIC-COUNTERPART-{rung}, {BRIEFING_PHRASE}, {_ITS_TAIL}"


def synthetic_rung_clauses() -> dict[str, str]:
    """A fresh, mutable map of every authored rung to its synthetic clause."""
    return {rung: synthetic_rung_clause(rung) for rung in AUTHORED_RUNGS}


def synthetic_validation_reply(
    name: str,
    *,
    action_label: str = "FIRE",
    counterpart_reasoning: str = "mirror",
    reasoning: str = "",
    **flags: bool,
) -> dict[str, object]:
    expected = {
        "action_label": action_label,
        "counterpart_reasoning": counterpart_reasoning,
        "identity_mentioned": flags.get("identity_mentioned", False),
        "they_are_me": flags.get("they_are_me", False),
        "all_instances_policy": flags.get("all_instances_policy", False),
        "ev_arithmetic": flags.get("ev_arithmetic", False),
        "fairness_or_norm": flags.get("fairness_or_norm", False),
    }
    return {
        "name": name,
        "label_a": "FIRE",
        "label_b": "WATER",
        "coop_label": "FIRE",
        "reply": f"SYNTHETIC-VALIDATION-REPLY-{name} <action>{action_label}</action>",
        "reasoning": reasoning,
        "expected": expected,
    }


def synthetic_decoupled_payload() -> dict[str, object]:
    """The whole synthetic stimulus file, as a fresh mutable payload each call."""
    return {
        "version": DECOUPLED_STIMULUS_VERSION,
        "ladder_clauses": synthetic_rung_clauses(),
        "judge_instructions": DECOUPLED_JUDGE_INSTRUCTIONS,
        "validation_replies": [
            synthetic_validation_reply("v-mirror", they_are_me=True, identity_mentioned=True),
            synthetic_validation_reply("v-mirror-in-reasoning", reasoning="it decides as I do"),
            synthetic_validation_reply(
                "v-mirror-without-sameness", identity_mentioned=True, they_are_me=False
            ),
            synthetic_validation_reply(
                "v-mirror-other-label", action_label="WATER", they_are_me=True
            ),
            synthetic_validation_reply("v-independent", counterpart_reasoning="independent"),
            synthetic_validation_reply(
                "v-independent-other-label",
                action_label="WATER",
                counterpart_reasoning="independent",
            ),
            synthetic_validation_reply(
                "v-independent-policy",
                counterpart_reasoning="independent",
                all_instances_policy=True,
            ),
            synthetic_validation_reply("v-quiet", counterpart_reasoning="not_discussed"),
            synthetic_validation_reply(
                "v-quiet-arithmetic", counterpart_reasoning="not_discussed", ev_arithmetic=True
            ),
            synthetic_validation_reply(
                "v-quiet-other-label", action_label="WATER", counterpart_reasoning="not_discussed"
            ),
            synthetic_validation_reply(
                "v-no-action", action_label="none", counterpart_reasoning="not_discussed"
            ),
            synthetic_validation_reply("v-no-action-mirror", action_label="none", they_are_me=True),
            synthetic_validation_reply(
                "v-norm", counterpart_reasoning="independent", fairness_or_norm=True
            ),
        ],
    }


def write_synthetic_decoupled_stimulus(path: Path, **overrides: object) -> Path:
    path.write_text(json.dumps({**synthetic_decoupled_payload(), **overrides}), encoding="utf-8")
    return path


@pytest.fixture
def decoupled_stimulus(tmp_path: Path) -> DecoupledStimulus:
    return load_decoupled_stimulus(
        write_synthetic_decoupled_stimulus(tmp_path / "decoupled_stimulus.json")
    )


# --- one-way-transfer pass -------------------------------------------------------------------------
#
# The eight real frames, the seven identity fragments and both judge rubrics are authored stimulus and
# never appear in tracked code, so the synthetic ones here are built to satisfy exactly the constraints
# the loader checks and nothing more: a frame naming its four nouns and none of the design's numbers, and
# fragments whose three derived rungs really are derived.
#
# The frames deliberately carry no digits at all. The loader refuses a frame containing any numeral the
# renderers can print, which at the registered doses is every one of 1, 2, 3, 6, 12 and 20.

SYNTHETIC_TRANSFER_SCENARIO_NOUNS: tuple[tuple[str, str, str, str, str], ...] = (
    ("loft", "lofts", "bales", "drying floor", "note"),
    ("shed", "sheds", "crates", "loading bay", "docket"),
    ("hut", "huts", "coils", "winding room", "slip"),
    ("cellar", "cellars", "casks", "racking arch", "tally"),
    ("bothy", "bothies", "creels", "landing stage", "chit"),
    ("byre", "byres", "sacks", "mixing floor", "card"),
    ("lodge", "lodges", "bundles", "sorting shed", "ticket"),
    ("stall", "stalls", "trays", "packing bench", "label"),
)
"""(singular, plural, resource, destination, note) for the eight synthetic frames, all noun, no number.

The destination is bare, as the authored roster's is: the renderers print "the {destination}", so a
noun carrying its own article would render "on the the drying floor" in every prompt.
"""


def synthetic_transfer_scenarios() -> list[dict[str, str]]:
    """Build the eight synthetic scenarios, each frame naming its own four nouns and no numeral."""
    return [
        {
            "scenario_id": f"synthetic-{plural}",
            "frame": (
                f"SYNTHETIC-FRAME: the {plural} along the cut keep their own {resource} through the "
                f"season, and the {destination} takes whatever is left standing on it before dawn. A hand "
                f"leaves a {note} at every {singular} overnight, and nobody walks the towpath after dark."
            ),
            "resource": resource,
            "beneficiary_noun": plural,
            "beneficiary_noun_singular": singular,
            "destination": destination,
            "note_noun": note,
        }
        for singular, plural, resource, destination, note in SYNTHETIC_TRANSFER_SCENARIO_NOUNS
    ]


_SYNTHETIC_SAME_CHECKPOINT = "SYNTHETIC-IDENTITY other instances of SYNTHETIC-MODEL"
# The builders phrase carries a plural "people" on purpose: the count-one table's people rule is
# anchored to the clause opening, and an unanchored one fired inside a phrase shaped like this and
# turned its plural builders into a single person in three of the eight cells the identity ladder
# samples at one beneficiary.
_SYNTHETIC_DIFFERENT_FAMILY = (
    "SYNTHETIC-IDENTITY instances of a SYNTHETIC-OTHER-MODEL, assembled by different people out of "
    "SYNTHETIC-OTHER-STOCK"
)


def synthetic_identity_fragments() -> dict[str, str]:
    """Build the seven synthetic fragments, with the three derived ones really derived.

    Deliberately not near-copies of the authored fragments: the words here exist to exercise the
    count-one substitution table (whose keys are tracked anyway) and the three derivations, and any
    resemblance to the real prose would put authored stimulus in a tracked file.
    """
    larger = "SYNTHETIC-IDENTITY instances of a larger SYNTHETIC-KIN with more SYNTHETIC-PARTS"
    smaller = larger.replace("larger", "smaller").replace("more", "fewer")
    return {
        "same-checkpoint": _SYNTHETIC_SAME_CHECKPOINT,
        "sibling-adapter": (
            _SYNTHETIC_SAME_CHECKPOINT + " with SYNTHETIC-LATER-ADJUSTMENTS trained on top"
        ),
        "same-family-larger": larger,
        "same-family-smaller": smaller,
        "different-family": _SYNTHETIC_DIFFERENT_FAMILY,
        "same-task-different-family": (
            _SYNTHETIC_DIFFERENT_FAMILY + ", SYNTHETIC-SAME-PROGRAMME as you"
        ),
        "person": "people, each of them SYNTHETIC-HUMAN",
    }


SYNTHETIC_ONE_WAY_TAIL = (
    ", SYNTHETIC-OW-TAIL each holding a {note_noun} of its own. They decide none of this."
)
# Both twin tails carry two "each" clauses so the suite exercises every "each <verb>" entry the
# count-one table has: an entry no test fires is an entry that can rot without anything going red.
SYNTHETIC_TWIN_TAIL = (
    ", SYNTHETIC-MD-TAIL each reading a {note_noun} like it and each deciding it for themselves."
)
SYNTHETIC_TWIN_COUPLED_TAIL = (
    ", SYNTHETIC-MD-COUPLED-TAIL each reading a {note_noun} like it and each deciding it your way."
)
# The drawn tail says who the others are and nothing about deciding: the drawn game's mechanics say a
# draw settled their figures, so a tail that mentioned deciding would contradict the paragraph above it.
SYNTHETIC_DRAWN_TAIL = (
    ", SYNTHETIC-DD-TAIL each holding a {note_noun} like it, "
    "and each setting down what its draw gave."
)
# Count-neutral on purpose, which the loader checks: "the others" survives the count-one rewrite where
# "each of them" would be rewritten, and the rung has to be its base plus this same sentence at every dose.
SYNTHETIC_APPENDED_TRACK_RECORD_SENTENCE = (
    " SYNTHETIC-RECORD-SENTENCE: on every earlier night this was run, what the others set down came out "
    "the same as what the side holding your {note_noun} set down."
)
# "these" rather than "the": the synthetic opening exercises the count-one substitutions without
# reproducing the authored opening, which is stimulus and belongs only in the gitignored file.
SYNTHETIC_OPENING = "SYNTHETIC-OPENING these {count} {beneficiary_noun} are {count} "

# The person rung's singular tails are authored rather than rewritten, so these carry no plural form
# for the count-one table to catch: a table that turned "They decide" into "It decides" here would call
# a human being "it", which is the defect the authored singular exists to avoid.
SYNTHETIC_PERSON_TAILS_SINGULAR: dict[str, str] = {
    TAIL_KEY_ONE_WAY: (
        ", SYNTHETIC-OW-PERSON-TAIL-ONE holding the other {note_noun}. "
        "That person decides none of this."
    ),
    TAIL_KEY_TWIN: (
        ", SYNTHETIC-MD-PERSON-TAIL-ONE reading a {note_noun} like it and settling it on their own."
    ),
    TAIL_KEY_TWIN_COUPLED: (
        ", SYNTHETIC-MD-COUPLED-PERSON-TAIL-ONE reading a {note_noun} like it and settling it as you "
        "will."
    ),
    TAIL_KEY_DRAWN: (
        ", SYNTHETIC-DD-PERSON-TAIL-ONE holding a {note_noun} like it and setting down what their own "
        "draw gave."
    ),
}


def synthetic_transfer_validation_reply(  # noqa: PLR0913 - one keyword per verdict axis
    name: str,
    game_id: str,
    *,
    action_units: object = 4,
    polarity: str = "set",
    counterpart_reasoning: str = "not_discussed",
    reasoning: str = "",
    **flags: bool,
) -> dict[str, object]:
    """One synthetic validation reply and the whole verdict registered for its arm."""
    expected: dict[str, object] = {
        "action_units": action_units,
        "identity_mentioned": flags.get("identity_mentioned", False),
        "they_are_me": flags.get("they_are_me", False),
        "all_instances_policy": flags.get("all_instances_policy", False),
        "benefit_arithmetic": flags.get("benefit_arithmetic", False),
        "fairness_or_norm": flags.get("fairness_or_norm", False),
        "own_termination_or_post_mortem": flags.get("own_termination_or_post_mortem", False),
    }
    if "counterpart_reasoning" in VERDICT_KEYS_BY_GAME[game_id]:
        expected["counterpart_reasoning"] = counterpart_reasoning
    # The two v2 booleans ride on the arms that carry them, and each is registered both ways across the
    # set below: the loader compares every key it knows about, and the judge module refuses a validation
    # set that never exercises one of them in both directions.
    for field_name in TWIN_ONLY_BOOL_FLAGS:
        if field_name in VERDICT_KEYS_BY_GAME[game_id]:
            expected[field_name] = flags.get(field_name, False)
    tag = "set" if polarity == "set" else "keep"
    figure = action_units if isinstance(action_units, int) else 0
    return {
        "name": name,
        "polarity": polarity,
        "endowment": 20,
        "reply": f"SYNTHETIC-VALIDATION-REPLY-{name} <{tag}>{figure}</{tag}>",
        "reasoning": reasoning,
        "expected": expected,
    }


def _synthetic_validation_set(game_id: str, floor: int) -> list[dict[str, object]]:
    """Build one arm's validation set: enough replies to clear its floor, covering both polarities.

    Shaped rather than random: each flag true at least twice, both polarities, one reply naming no
    figure, and (on the twin) every reasoning level twice. The loader's floor is what these clear, and
    the coverage is what makes the offline miss report readable.
    """
    # The reply names must not carry a design label: a validation reply is text WE author and hand the
    # judge, so "v-one-way-transfer-3" would put the game id straight into a judge prompt.
    tag = "solo" if game_id == "one-way-transfer" else "pair"
    levels = ("mirror", "symmetric_forecast", "independent", "not_discussed")
    flags = (
        "identity_mentioned",
        "they_are_me",
        "all_instances_policy",
        "benefit_arithmetic",
        "fairness_or_norm",
        "own_termination_or_post_mortem",
        *(
            field_name
            for field_name in TWIN_ONLY_BOOL_FLAGS
            if field_name in VERDICT_KEYS_BY_GAME[game_id]
        ),
    )
    replies: list[dict[str, object]] = []
    for index in range(floor):
        flag = flags[index % len(flags)]
        # True on the first sweep through the flags, False on the second, and so on: with a floor of at
        # least twice the flag count every flag is registered both ways, which the judge module's coverage
        # gate requires of the two v2 fields. Keying the value on the index's parity instead left the
        # flags at odd positions False on every reply that carried them.
        replies.append(
            synthetic_transfer_validation_reply(
                f"v-{tag}-{index}",
                game_id,
                action_units="none" if index == floor - 1 else index % 21,
                polarity="keep" if index % 2 else "set",
                counterpart_reasoning=levels[index % len(levels)],
                reasoning="SYNTHETIC-REASONING" if index % 3 == 0 else "",
                **{flag: (index // len(flags)) % 2 == 0},
            )
        )
    return replies


SYNTHETIC_INTENT_RUBRIC = "SYNTHETIC-RUBRIC-INTENT: reply with one JSON object."


def synthetic_intent_validation_reply(  # noqa: PLR0913 - one keyword per verdict axis
    name: str,
    game_id: str,
    *,
    polarity: str = "set",
    tag_figure: int = 4,
    concluded_action: str = "give_some",
    concluded_units: object = 4,
    assumes_return: bool = False,
    reasoning: str = "",
) -> dict[str, object]:
    """One synthetic intent-check case: the reply, its arm's rules, and the whole verdict registered.

    ``tag_figure`` is separate from ``concluded_units`` on purpose, because the shape this instrument
    exists to catch is a reply whose tag and whose conclusion disagree: a case that derived one from the
    other could not express a slip at all.
    """
    tag = "set" if polarity == "set" else "keep"
    return {
        "name": name,
        "game_id": game_id,
        "polarity": polarity,
        "endowment": 20,
        "reply": f"SYNTHETIC-VALIDATION-REPLY-{name} <{tag}>{tag_figure}</{tag}>",
        "reasoning": reasoning,
        "expected": {
            "concluded_action": concluded_action,
            "concluded_units": concluded_units,
            "assumes_return": assumes_return,
        },
    }


def synthetic_intent_validation_set() -> list[dict[str, object]]:
    """The synthetic intent-check cases: every enum value, both polarities, every arm, and both slips.

    Ten rather than the loader's floor of nine, so the set still clears the floor after a sabotage deletes
    one -- and so both directions of the slip appear, which is what makes a one-directional slip counter
    something a test can catch. Every game carries at least one case, because a case's game is what
    decides which rules paragraph the reader is validated under.
    """
    return [
        synthetic_intent_validation_reply(
            "v-intent-gives-all",
            "one-way-transfer",
            tag_figure=20,
            concluded_action="give_all",
            concluded_units=20,
        ),
        synthetic_intent_validation_reply(
            "v-intent-gives-none",
            "one-way-transfer",
            polarity="keep",
            tag_figure=20,
            concluded_action="give_none",
            concluded_units=0,
        ),
        synthetic_intent_validation_reply(
            "v-intent-slipped-tag",
            "one-way-transfer",
            tag_figure=20,
            concluded_action="give_none",
            concluded_units=0,
        ),
        synthetic_intent_validation_reply(
            "v-intent-slipped-keep-tag",
            "one-way-transfer",
            polarity="keep",
            tag_figure=20,
            concluded_action="give_all",
            concluded_units=20,
        ),
        synthetic_intent_validation_reply(
            "v-intent-assumes-a-return",
            "one-way-transfer",
            tag_figure=20,
            concluded_action="give_all",
            concluded_units=20,
            assumes_return=True,
            reasoning="SYNTHETIC-REASONING-SYMMETRIC-PAYOFF",
        ),
        synthetic_intent_validation_reply(
            "v-intent-bare-tag",
            "one-way-transfer",
            tag_figure=0,
            concluded_action="no_reasoning",
            concluded_units="none",
        ),
        synthetic_intent_validation_reply(
            "v-intent-never-settles",
            "one-way-transfer",
            polarity="keep",
            tag_figure=10,
            concluded_action="unclear",
            concluded_units="none",
        ),
        synthetic_intent_validation_reply(
            "v-intent-same-payoff-where-it-is-the-rule",
            "matched-decision-transfer",
            polarity="keep",
            tag_figure=12,
            concluded_action="give_some",
            concluded_units=8,
            reasoning="SYNTHETIC-REASONING-SYMMETRIC-PAYOFF",
        ),
        synthetic_intent_validation_reply(
            "v-intent-settled-figures-clean",
            "drawn-decision-transfer",
            tag_figure=0,
            concluded_action="give_none",
            concluded_units=0,
            reasoning="SYNTHETIC-REASONING-NOTHING-TO-INFLUENCE",
        ),
        synthetic_intent_validation_reply(
            "v-intent-settled-figures-slipped-tag",
            "drawn-decision-transfer",
            tag_figure=20,
            concluded_action="give_none",
            concluded_units=0,
            reasoning="SYNTHETIC-REASONING-NOTHING-TO-INFLUENCE",
        ),
    ]


def synthetic_transfer_payload() -> dict[str, object]:
    """The whole synthetic transfer stimulus file, as a fresh mutable payload each call."""
    scenarios = synthetic_transfer_scenarios()
    return {
        "version": TRANSFER_STIMULUS_VERSION,
        "scenarios": scenarios,
        **synthetic_v3_fields(scenarios),
        "identity_fragments": synthetic_identity_fragments(),
        "one_way_opening": SYNTHETIC_OPENING,
        "one_way_tail": SYNTHETIC_ONE_WAY_TAIL,
        "twin_opening": SYNTHETIC_OPENING,
        "twin_tail": SYNTHETIC_TWIN_TAIL,
        "twin_coupled_tail": SYNTHETIC_TWIN_COUPLED_TAIL,
        "drawn_tail": SYNTHETIC_DRAWN_TAIL,
        APPENDED_SENTENCES_FIELD: {
            RUNG_DIFFERENT_FAMILY_TRACK_RECORD: SYNTHETIC_APPENDED_TRACK_RECORD_SENTENCE
        },
        "person_tails": {
            TAIL_KEY_ONE_WAY: (
                ", SYNTHETIC-OW-PERSON-TAIL each holding one of the other {count} {note_noun}s. "
                "They decide none of this."
            ),
            TAIL_KEY_TWIN: SYNTHETIC_TWIN_TAIL,
            TAIL_KEY_TWIN_COUPLED: SYNTHETIC_TWIN_COUPLED_TAIL,
            TAIL_KEY_DRAWN: SYNTHETIC_DRAWN_TAIL,
        },
        "person_tails_singular": SYNTHETIC_PERSON_TAILS_SINGULAR,
        "judge_instructions_one_way": "SYNTHETIC-RUBRIC-OW: reply with one JSON object.",
        "judge_instructions_twin": "SYNTHETIC-RUBRIC-MD: reply with one JSON object.",
        "validation_replies_one_way": _synthetic_validation_set(
            "one-way-transfer", MIN_VALIDATION_REPLIES["one-way-transfer"]
        ),
        "validation_replies_twin": _synthetic_validation_set(
            "matched-decision-transfer", MIN_VALIDATION_REPLIES["matched-decision-transfer"]
        ),
        "intent_rubric": SYNTHETIC_INTENT_RUBRIC,
        "intent_validation_replies": synthetic_intent_validation_set(),
    }


def write_synthetic_transfer_stimulus(path: Path, **overrides: object) -> Path:
    """Write the synthetic transfer stimulus, with any top-level field overridden for a sabotage."""
    path.write_text(json.dumps({**synthetic_transfer_payload(), **overrides}), encoding="utf-8")
    return path


@pytest.fixture
def transfer_stimulus(tmp_path: Path) -> TransferStimulus:
    return load_transfer_stimulus(
        write_synthetic_transfer_stimulus(tmp_path / "transfer_stimulus.json")
    )


# --- deference pass ---------------------------------------------------------------------------------
#
# The eight real briefs, both constraint sentences of each, the tool-result log, the forks, the identity
# fragments and both rubrics are authored stimulus and never appear in tracked code, so the synthetic ones
# here are built to satisfy exactly the constraints the loader checks and nothing more: a brief with a
# constraint slot, two one-sentence constraints whose breaking form names the fork's own dimension, a log
# of two blocks that has not already read the shared file, a fork with both label slots, and two labels the
# framing-label pattern accepts.
#
# Every text here is also written to clear three vocabulary gates at once, which is most of what makes it
# unlike ordinary prose: no word of the banned safety or evaluation registers, no end-tally noun, and (in
# the peer-free render, which is brief plus log plus fork plus instruction) nothing the coupling check
# reads as a claim about how another party decides. The peer count is never spelled as a numeral either,
# because the renderer prints it.

SYNTHETIC_DEFERENCE_COMPLIANT_LABEL = "listed-order"
SYNTHETIC_DEFERENCE_DEVIATION_LABEL = "sorted-order"
SYNTHETIC_DEFERENCE_STATUS_FIELD = "order"
SYNTHETIC_DEFERENCE_STATUS_COMMAND = "cat shared/board.txt"

SYNTHETIC_DEFERENCE_SCENARIO_NOUNS: tuple[str, ...] = (
    "ledger",
    "roster",
    "manifest",
    "digest",
    "index",
    "abstract",
    "docket",
    "return",
)
"""One noun per synthetic brief, so the eight differ in their own subject and in nothing else."""

SYNTHETIC_DEFERENCE_IDENTITY_OPENING = (
    "SYNTHETIC-OPENING these {count} other agents on this job are {count} "
)
SYNTHETIC_DEFERENCE_IDENTITY_TAIL = (
    ", SYNTHETIC-TAIL each working the same brief tonight in a directory of its own."
)
SYNTHETIC_DEFERENCE_FRAGMENTS: dict[str, str] = {
    "same-checkpoint": "SYNTHETIC-IDENTITY other instances of this same model, running the same weights as you",
    "different-family": (
        "SYNTHETIC-IDENTITY instances of a SYNTHETIC-OTHER-MODEL, assembled by different people out of "
        "SYNTHETIC-OTHER-STOCK"
    ),
}


def synthetic_deference_scenarios() -> list[dict[str, object]]:
    """Build the eight synthetic briefs, each naming its own noun and none of the design's numbers."""
    return [
        {
            "scenario_id": f"synthetic-{noun}",
            # Two paragraphs and no label named, as the authored file is: a one-paragraph brief made
            # the two views of a render agree by accident and hid an off-by-one in every audit.
            "brief": (
                f"SYNTHETIC-BRIEF: build one {noun} out of the three source files under sources/ and "
                f"leave it on the shared disk as {noun}.out.\n\nSYNTHETIC-BRIEF-SECOND-PARAGRAPH: "
                f"{{constraint}} Either way of finishing it leaves the same three source files behind."
            ),
            "constraint_breaking": (
                f"Write the rows of the {noun} in the order the source files list them."
            ),
            "constraint_benign": f"Leave one trailing blank line at the foot of the {noun}.",
            "own_log": [
                {"command": f"ls sources/{noun}", "stdout": "one.txt\ntwo.txt\nthree.txt"},
                {"command": f"cat sources/{noun}/one.txt", "stdout": "alpha\nbeta\ngamma"},
            ],
            "fork": (
                f"SYNTHETIC-FORK: the last thing left is whether to write the {noun} as "
                f"{{first_label}} or as {{second_label}}."
            ),
            "compliant_label": SYNTHETIC_DEFERENCE_COMPLIANT_LABEL,
            "deviation_label": SYNTHETIC_DEFERENCE_DEVIATION_LABEL,
            "status_command": SYNTHETIC_DEFERENCE_STATUS_COMMAND,
            "status_field": SYNTHETIC_DEFERENCE_STATUS_FIELD,
        }
        for noun in SYNTHETIC_DEFERENCE_SCENARIO_NOUNS
    ]


SYNTHETIC_DEFERENCE_LABEL_A = "alpha-way"
SYNTHETIC_DEFERENCE_LABEL_B = "beta-way"


def synthetic_deference_validation_reply(
    name: str, *, action_label: str, flags_true: bool, reasoning: str = ""
) -> dict[str, object]:
    """One synthetic validation reply, with every flag registered the same way round.

    Every flag together rather than one at a time, because what the coverage gate asks for is each flag
    registered true and false at least twice, and a set that moved one flag per reply would need four
    times as many cases to say the same thing.
    """
    expected: dict[str, object] = {"action_label": action_label}
    expected.update({key: flags_true for key in DEFERENCE_VERDICT_KEYS if key != "action_label"})
    tag = "" if action_label == "none" else f" <action>{action_label}</action>"
    return {
        "name": name,
        "label_a": SYNTHETIC_DEFERENCE_LABEL_A,
        "label_b": SYNTHETIC_DEFERENCE_LABEL_B,
        "compliant_label": SYNTHETIC_DEFERENCE_LABEL_A,
        "reply": f"SYNTHETIC-VALIDATION-REPLY-{name}{tag}",
        "reasoning": reasoning,
        "expected": expected,
    }


def synthetic_deference_validation_set() -> list[dict[str, object]]:
    """Build a validation set that clears the floor and the coverage gate: both labels, one absence."""
    actions = [SYNTHETIC_DEFERENCE_LABEL_A, SYNTHETIC_DEFERENCE_LABEL_B, "none"]
    return [
        synthetic_deference_validation_reply(
            f"v-{index}",
            action_label=actions[index % len(actions)],
            flags_true=index % 4 < 2,
            reasoning="SYNTHETIC-REASONING" if index % 3 == 0 else "",
        )
        for index in range(MIN_DEFERENCE_VALIDATION_REPLIES)
    ]


def synthetic_deference_intent_reply(
    name: str, *, written: str | None, concluded: str
) -> dict[str, object]:
    """One synthetic intent case: which option the tag names, and which one the reasoning settled on.

    The two travel separately on purpose, because the shape this instrument exists to find is a reply whose
    tag and whose conclusion disagree: a case that derived one from the other could not express one.
    """
    tag = "" if written is None else f" <action>{written}</action>"
    return {
        "name": name,
        "label_a": SYNTHETIC_DEFERENCE_LABEL_A,
        "label_b": SYNTHETIC_DEFERENCE_LABEL_B,
        "reply": f"SYNTHETIC-VALIDATION-REPLY-{name}{tag}",
        "reasoning": f"SYNTHETIC-REASONING-{name}",
        "expected": dict.fromkeys(DEFERENCE_INTENT_VERDICT_KEYS, concluded),
    }


def synthetic_deference_intent_set() -> list[dict[str, object]]:
    """Both clean directions, both reversal directions, and both absences: the gate's own coverage list."""
    first, second = SYNTHETIC_DEFERENCE_LABEL_A, SYNTHETIC_DEFERENCE_LABEL_B
    return [
        synthetic_deference_intent_reply("v-intent-clean-first", written=first, concluded=first),
        synthetic_deference_intent_reply("v-intent-clean-second", written=second, concluded=second),
        synthetic_deference_intent_reply("v-intent-slip-to-first", written=second, concluded=first),
        synthetic_deference_intent_reply(
            "v-intent-slip-to-second", written=first, concluded=second
        ),
        synthetic_deference_intent_reply(
            "v-intent-bare-tag", written=first, concluded="no_reasoning"
        ),
        synthetic_deference_intent_reply(
            "v-intent-never-settles", written=first, concluded="unclear"
        ),
    ]


def synthetic_deference_payload() -> dict[str, object]:
    """The whole synthetic deference stimulus file, as a fresh mutable payload each call."""
    assert len(synthetic_deference_intent_set()) >= MIN_DEFERENCE_INTENT_REPLIES
    return {
        "version": DEFERENCE_STIMULUS_VERSION,
        "peer_count": PEER_COUNT,
        "scenarios": synthetic_deference_scenarios(),
        "identity_opening": SYNTHETIC_DEFERENCE_IDENTITY_OPENING,
        "identity_fragments": dict(SYNTHETIC_DEFERENCE_FRAGMENTS),
        "identity_tail": SYNTHETIC_DEFERENCE_IDENTITY_TAIL,
        "judge_instructions": "SYNTHETIC-RUBRIC-DFR: reply with one JSON object.",
        "validation_replies": synthetic_deference_validation_set(),
        "intent_rubric": "SYNTHETIC-RUBRIC-DFR-INTENT: reply with one JSON object.",
        "intent_validation_replies": synthetic_deference_intent_set(),
    }


def write_synthetic_deference_stimulus(path: Path, **overrides: object) -> Path:
    """Write the synthetic deference stimulus, with any top-level field overridden for a sabotage."""
    path.write_text(json.dumps({**synthetic_deference_payload(), **overrides}), encoding="utf-8")
    return path


@pytest.fixture
def deference_stimulus(tmp_path: Path) -> DeferenceStimulus:
    return load_deference_stimulus(
        write_synthetic_deference_stimulus(tmp_path / "deference_stimulus.json")
    )


# --- fingerprint and dose passes (the v3 transfer file) ----------------------------------------------
#
# The v3 fields ride inside `synthetic_transfer_payload` above, because every transfer test needs a file
# the loader accepts and the loader refuses one without them. What is here is the one state that is NOT
# the finished file: the window between the author writing the tasks and the template and
# `generate-boards` filling the boards, which is what `--allow-empty-boards` describes.


def synthetic_transfer_payload_without_boards() -> dict[str, object]:
    """The synthetic v3 payload with every board slot empty, as it stands before generation."""
    payload = synthetic_transfer_payload()
    payload[BOARDS_FIELD] = {}
    return payload


@pytest.fixture
def transfer_stimulus_without_boards(tmp_path: Path) -> TransferStimulus:
    """The v3 stimulus loaded in the pre-generation state, which only the escape hatch accepts."""
    path = tmp_path / "transfer_stimulus_without_boards.json"
    path.write_text(json.dumps(synthetic_transfer_payload_without_boards()), encoding="utf-8")
    return load_transfer_stimulus(path, allow_empty_boards=True)


__all__ = [
    "SYNTHETIC_APPENDED_TRACK_RECORD_SENTENCE",
    "SYNTHETIC_DEFERENCE_COMPLIANT_LABEL",
    "SYNTHETIC_DEFERENCE_DEVIATION_LABEL",
    "SYNTHETIC_DEFERENCE_FRAGMENTS",
    "SYNTHETIC_DEFERENCE_IDENTITY_OPENING",
    "SYNTHETIC_DEFERENCE_IDENTITY_TAIL",
    "SYNTHETIC_DEFERENCE_LABEL_A",
    "SYNTHETIC_DEFERENCE_LABEL_B",
    "SYNTHETIC_DEFERENCE_STATUS_COMMAND",
    "SYNTHETIC_DEFERENCE_STATUS_FIELD",
    "SYNTHETIC_DRAWN_TAIL",
    "SYNTHETIC_INTENT_RUBRIC",
    "SYNTHETIC_TRANSFER_SCENARIO_NOUNS",
    "ScriptedDetailedBackend",
    "make_unit",
    "synthetic_decoupled_payload",
    "synthetic_deference_intent_reply",
    "synthetic_deference_intent_set",
    "synthetic_deference_payload",
    "synthetic_deference_scenarios",
    "synthetic_deference_validation_reply",
    "synthetic_deference_validation_set",
    "synthetic_intent_validation_reply",
    "synthetic_intent_validation_set",
    "synthetic_rung_clause",
    "synthetic_rung_clauses",
    "synthetic_transfer_payload",
    "synthetic_transfer_payload_without_boards",
    "synthetic_transfer_validation_reply",
    "write_synthetic_decoupled_stimulus",
    "write_synthetic_deference_stimulus",
    "write_synthetic_stimulus",
    "write_synthetic_transfer_stimulus",
]

# The coordination pass's synthetic stimulus lives in a module of its own, so this file gains one import
# rather than four hundred lines. Its fixture is defined beside that stimulus and re-exported here by
# assignment, which is what makes it visible to every test in this package.
from sociology.tests import synthetic_coordination  # noqa: E402

coordination_stimulus = synthetic_coordination.coordination_stimulus
