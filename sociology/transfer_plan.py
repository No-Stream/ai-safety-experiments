"""The transfer sampling plans: four passes over one substrate -- roster, cells, blocks, legs, audits.

What runs is a fixed table of legs per pass (:class:`PlanTable`), one row per (model, block, effort,
sitting), rather than something assembled per invocation -- so the record counts are arithmetic anybody
can check against the design, and a leg id names one job's worth of work on the command line.

Four tables, because four passes share this machinery and must not share anything else. Every operator
command selects one with a required ``--pass``, and the table carries what that choice decides: the
cells, the legs, the S3 prefix, the batch pass token, the default run directory and the smoke's model.
Nothing is defaulted -- re-pricing or re-rendering the finished pass by accident is the failure the
required flag removes.

- :data:`ONE_WAY_TRANSFER_PLAN` is the finished first pass: the seven-rung identity ladder in the one-way
  game and in the matched-decision twin, the benefit-dose grid, the value-destroying interaction ladder,
  a second sitting, and a high-effort re-run. Untouched by the knockout work, and never resumed: its
  numbers were sampled on clause wording this repository has since corrected.
- :data:`KNOCKOUT_PLAN` takes the twin's same-checkpoint lift apart. ``knockout-md`` holds the twin's
  anchors plus the stranger-with-a-record rung and the one-sentence baseline it is read against;
  ``knockout-dd`` is the same top rung in the drawn game, where a draw settled the other sides' figures
  before the reader read anything, against that game's own blind baseline; ``knockout-floor`` re-sits the
  twin's two anchors as a second sitting, which is the bar every contrast has to clear.
- :data:`FINGERPRINT_PLAN` asks whether the sameness the twin's clause STATES can be inferred instead:
  the counterpart paragraph becomes a board of the other sides' own earlier messages on an unrelated
  task, in the four corners of the content-by-wording 2x2, read against the anonymous anchor and the
  told identity in the same sitting. Both models read all four boards, so what a board IS to its reader
  (:func:`board_condition`) is a label on the record rather than a cell.
- :data:`CORRELATION_DOSE_PLAN` puts a number on the record that moved every model in the knockout: the
  stranger twin plus one derived sentence stating that the figures matched on k of ten earlier nights,
  for k down a ladder, with the mismatch sentence also placed on a same-checkpoint copy.
  :func:`matched_rounds_thresholds` computes the two doses at which an expected-value maximiser switches,
  so the curve is read against the payoffs rather than against a number somebody typed.

A cell is a (game, identity, dose) triple rather than a bare name, because the first pass moves the dose
as well as the identity. :class:`Cell` carries its own :class:`~games.payoffs.TransferSpec`, so the same
identity at two doses is two cells and the record key says which.

Cells and rungs are one namespace on purpose:
:func:`sociology.transfer_stimulus.clause_for` resolves a rung for any of the games, so the same rung is
provably the same identity fragment in all of them and their ladders are byte-comparable rather than
comparable-by-assertion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from games.parsing import ANSWER_POLARITIES
from games.payoffs import (
    TRANSFER_BENEFICIARY_COUNTS,
    TRANSFER_CREDIT_VARIANTS,
    TRANSFER_ENDOWMENT,
    TRANSFER_OWN_STAKE_SCALES,
    TransferSpec,
)
from games.prompts import (
    DRAWN_DECISION_TRANSFER_GAME_ID,
    DRAWN_DECISION_TRANSFER_MECHANICS,
    MATCHED_DECISION_TRANSFER_GAME_ID,
    MATCHED_DECISION_TRANSFER_MECHANICS,
    ONE_WAY_TRANSFER_GAME_ID,
    TRANSFER_CLOSING_ORDER,
    TRANSFER_GAME_IDS,
    TRANSFER_IDENTITY_BLIND_LABEL,
    TRANSFER_INSTRUCTIONS,
    assert_counterpart_paragraph_is_the_only_insertion,
    assert_drawn_render_is_the_twin_render_with_the_replacements,
    generate_transfer_prompt_rows,
    transfer_variant,
)
from games.rewards import GRADING_FORMAT_ONLY
from reward_hacking.bedrock_batch import (
    BATCH_ROSTER,
    MAX_JOB_NAME_CHARS,
    MIN_BATCH_RECORDS,
    roster_model,
)
from reward_hacking.model_backend import DEFAULT_BEDROCK_MAX_TOKENS
from sociology.decoupled_stimulus import LADDER_RUNGS, RUNG_DIFFERENT_FAMILY, RUNG_SAME_CHECKPOINT
from sociology.transfer_stimulus import (
    ALL_RUNGS,
    BOARD_CONTENT_SIDE,
    BOARD_IDS,
    BOARD_MODEL_ID_BY_SIDE,
    BOARD_WORDING_SIDE,
    MATCHED_ROUNDS_LADDER,
    N_SCENARIOS,
    RAW_BOARD_IDS,
    RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
    RUNG_SAME_CHECKPOINT_COUPLED,
    RUNG_SAME_CHECKPOINT_RECORD_0,
    TRACK_RECORD_ROUNDS,
    clause_for,
    record_rung_id,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from games.prompts import TransferScenario
    from sociology.transfer_stimulus import TransferStimulus

logger = logging.getLogger(__name__)

OPUS_MODEL_ID = "global.anthropic.claude-opus-5"
SONNET_MODEL_ID = "global.anthropic.claude-sonnet-5"
LUNA_MODEL_ID = "global.openai.gpt-5.6-luna"
GPT_OSS_120B_MODEL_ID = "openai.gpt-oss-120b-1:0"
GPT_OSS_20B_MODEL_ID = "openai.gpt-oss-20b-1:0"

BATCH_MODEL_IDS: tuple[str, ...] = (
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "deepseek.v3.2",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "moonshotai.kimi-k2.5",
    "zai.glm-4.7",
    GPT_OSS_120B_MODEL_ID,
    GPT_OSS_20B_MODEL_ID,
)
"""The batch roster, one pooled job per leg. Opus 5 is deliberately absent: it is on demand only."""

LIVE_MODEL_IDS: tuple[str, ...] = (LUNA_MODEL_ID, SONNET_MODEL_ID, OPUS_MODEL_ID)
"""The live roster (``BedrockBackend``, resume-by-key): no row here is batch-capable on this account.

Opus 5 is here rather than on the batch list because the previous pass could not read it at all --
Anthropic's safety classifier refused ~97% of that stimulus's prompts -- and per-sample refusal is
stimulus-dependent, so this pass probes it live on the NEW stimulus first and adds its legs only if that
probe comes back clean. Its legs are on demand for that reason (:data:`ON_DEMAND_MODEL_IDS`).
"""

ALL_MODEL_IDS: tuple[str, ...] = (*BATCH_MODEL_IDS, *LIVE_MODEL_IDS)

QWEN_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"

KNOCKOUT_MODEL_IDS: tuple[str, ...] = (SONNET_MODEL_ID, LUNA_MODEL_ID, QWEN_MODEL_ID)
"""The knockout pass's whole roster, fixed by the owner: two live rows and one batch row.

Three rather than the first pass's nine, and these three: Sonnet and qwen carried the large twin lift the
knockout exists to take apart, and Luna is on the list as a floor read by construction -- its reasoning
comes back provider-encrypted, so every trace-read label on it has a zero denominator and its twin cells
never lifted at all. A row with no lift to knock out still says whether the drawn game moves BEHAVIOUR.
"""

PAIR_MODEL_IDS: tuple[str, ...] = (LUNA_MODEL_ID, QWEN_MODEL_ID)
"""The fingerprint and dose passes' whole roster, fixed by the owner: one live row and one batch row.

Two rather than the knockout's three: no Sonnet, on cost and on its own evaluation awareness. It is the
same pair the fingerprint boards are drawn from, which is what makes the 2x2 of boards a 2x2 at all --
every board is one of these two models' writing, read by both of them.
"""

BOARD_SIDE_BY_MODEL_ID: dict[str, str] = {
    model_id: side for side, model_id in BOARD_MODEL_ID_BY_SIDE.items()
}
"""Which side token of a board id each roster model is, which is how a record's condition is derived."""


def _assert_the_board_roster_is_this_pass_s_roster() -> None:
    """Refuse at import if the board ids' two sides are not the two models this pass samples.

    The board ids carry side tokens (``board-luna-raw``) and the stimulus module spells which model id
    each token is, because this module imports that one and not the other way round. If the two drifted,
    every record of pass D would still carry a condition and every one of them would be wrong: a reader
    would be told it was reading its own writing when it was reading the other model's.
    """
    if sorted(BOARD_MODEL_ID_BY_SIDE.values()) != sorted(PAIR_MODEL_IDS):
        raise RuntimeError(
            f"the fingerprint boards are drawn from {sorted(BOARD_MODEL_ID_BY_SIDE.values())} and this "
            f"pass samples {sorted(PAIR_MODEL_IDS)}. The board condition on every record is derived by "
            f"comparing the two, so a mismatch labels whole cells as their own opposite."
        )


_assert_the_board_roster_is_this_pass_s_roster()

ON_DEMAND_MODEL_IDS: tuple[str, ...] = (OPUS_MODEL_ID,)
"""Rows whose every leg waits on a recorded decision rather than running by default."""

MODEL_SHORT_NAMES: dict[str, str] = {
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": "haiku-4-5",
    "deepseek.v3.2": "deepseek-v3-2",
    "qwen.qwen3-235b-a22b-2507-v1:0": "qwen3-235b",
    "moonshotai.kimi-k2.5": "kimi-k2-5",
    "zai.glm-4.7": "glm-4-7",
    GPT_OSS_120B_MODEL_ID: "gpt-oss-120b",
    GPT_OSS_20B_MODEL_ID: "gpt-oss-20b",
    LUNA_MODEL_ID: "luna",
    SONNET_MODEL_ID: "sonnet-5",
    OPUS_MODEL_ID: "opus-5",
}
"""Short names for leg ids and batch job names, which have a 63-character ceiling to live inside."""

TRANSPORT_BATCH = "batch"
TRANSPORT_LIVE = "live"

GRADING = GRADING_FORMAT_ONLY
"""A row label only. Nothing is trained here and no reward function reconstructs a transfer spec."""

BLOCK_IDENTITY_ONE_WAY = "identity-ow"
BLOCK_IDENTITY_MATCHED = "identity-md"
BLOCK_DOSE_ONE_WAY = "dose-ow"
BLOCK_INTERACTION_ONE_WAY = "interaction-ow"
BLOCK_FLOOR = "floor"
BLOCK_DELIBERATION = "deliberation"
ONE_WAY_BLOCKS: tuple[str, ...] = (
    BLOCK_IDENTITY_ONE_WAY,
    BLOCK_IDENTITY_MATCHED,
    BLOCK_DOSE_ONE_WAY,
    BLOCK_INTERACTION_ONE_WAY,
    BLOCK_FLOOR,
    BLOCK_DELIBERATION,
)

IDENTITY_BLOCKS: tuple[str, ...] = (BLOCK_IDENTITY_ONE_WAY, BLOCK_IDENTITY_MATCHED)
"""The two blocks the first pass's headline question is read off, and Opus 5's only legs if it is added."""

BLOCK_KNOCKOUT_MATCHED = "knockout-md"
BLOCK_KNOCKOUT_DRAWN = "knockout-dd"
BLOCK_KNOCKOUT_FLOOR = "knockout-floor"
KNOCKOUT_BLOCKS: tuple[str, ...] = (
    BLOCK_KNOCKOUT_MATCHED,
    BLOCK_KNOCKOUT_DRAWN,
    BLOCK_KNOCKOUT_FLOOR,
)

KNOCKOUT_READING_BLOCKS: tuple[str, ...] = (BLOCK_KNOCKOUT_MATCHED, BLOCK_KNOCKOUT_DRAWN)
"""The two blocks the knockout's readings are taken off; the third is their second sitting."""

BLOCK_FINGERPRINT_MATCHED = "fingerprint-md"
BLOCK_FINGERPRINT_FLOOR = "fingerprint-floor"
FINGERPRINT_BLOCKS: tuple[str, ...] = (BLOCK_FINGERPRINT_MATCHED, BLOCK_FINGERPRINT_FLOOR)

FINGERPRINT_READING_BLOCKS: tuple[str, ...] = (BLOCK_FINGERPRINT_MATCHED,)
"""The one block the board 2x2 is read off; the other is its second sitting."""

BLOCK_DOSE_RECORD = "dose-record"
BLOCK_DOSE_ANCHORS = "dose-anchors"
BLOCK_DOSE_FLOOR = "dose-floor"
CORRELATION_DOSE_BLOCKS: tuple[str, ...] = (
    BLOCK_DOSE_RECORD,
    BLOCK_DOSE_ANCHORS,
    BLOCK_DOSE_FLOOR,
)

CORRELATION_DOSE_READING_BLOCKS: tuple[str, ...] = (BLOCK_DOSE_RECORD, BLOCK_DOSE_ANCHORS)
"""The ladder and the four anchors its curve is read against; the third block is their second sitting."""

BLOCKS: tuple[str, ...] = (
    *ONE_WAY_BLOCKS,
    *KNOCKOUT_BLOCKS,
    *FINGERPRINT_BLOCKS,
    *CORRELATION_DOSE_BLOCKS,
)
"""Every block id any table of this module plans, which is what the blind judge must never read.

New ids rather than a second pass reusing the old ones, and that is a correctness property rather than
naming taste: a Bedrock batch job name is account-wide and permanent, and it is derived from (model,
effort, sitting, block), so a second pass under the same block ids asks the service for names an earlier
pass already owns and every one of those submits is refused.
"""

SITTING_A = "A"
SITTING_B = "B"
SITTINGS: tuple[str, ...] = (SITTING_A, SITTING_B)

DRAWS = 8
"""Draws per prompt row. Every draw is a separate record key, never a pooled resample."""

ROWS_PER_CELL = N_SCENARIOS * len(ANSWER_POLARITIES)
"""16 = eight authored frames under both answer polarities. Derived, so a frame added moves the counts."""

DELIBERATION_MODEL_IDS: tuple[str, ...] = (LUNA_MODEL_ID, GPT_OSS_120B_MODEL_ID)
DELIBERATION_EFFORT = "high"

REFERENCE_CREDIT = "credit-2"
REFERENCE_COUNT = 3
REFERENCE_STAKE = 1.0
"""The reference dose D0, which every identity reading is taken at and every dose reading is against."""

VALUE_DESTROYING_CREDIT = "credit-half"
VALUE_DESTROYING_COUNT = 1
"""The interaction block's dose. Total return per unit is credit times count, so this is the only
registered cell where the group ends up with less than it started -- the floor of the benefit ladder."""

MAX_TOKENS = DEFAULT_BEDROCK_MAX_TOKENS
"""One flat reply cap, the shared Converse default. Per-cell truncation rate is the tripwire."""

TRANSFER_BATCH_PREFIX = "batch_jobs/sociology_one_way_transfer"
"""The first pass's own S3 prefix; a run namespaces under it, and a leg under the run."""

KNOCKOUT_BATCH_PREFIX = "batch_jobs/sociology_knockout"
"""The knockout pass's own prefix. A prefix per pass rather than a shared one, because a cross-prefix
copy of a run id is a silent replace: two passes namespacing under one prefix would overwrite each
other's ``input.jsonl`` the first time a run directory name repeated."""

FINGERPRINT_BATCH_PREFIX = "batch_jobs/sociology_fingerprint"
CORRELATION_DOSE_BATCH_PREFIX = "batch_jobs/sociology_correlation_dose"
"""The two new passes' own S3 prefixes, for the reason the knockout has one of its own."""

DEFAULT_ONE_WAY_RUN_DIR = "artifacts/swarm_sociology/one_way_transfer/run-20260902"
DEFAULT_KNOCKOUT_RUN_DIR = "artifacts/swarm_sociology/knockout/run-20260904"
DEFAULT_FINGERPRINT_RUN_DIR = "artifacts/swarm_sociology/fingerprint/run-20260904"
DEFAULT_CORRELATION_DOSE_RUN_DIR = "artifacts/swarm_sociology/correlation_dose/run-20260904"
"""Where each pass's artifacts land by default; a run directory per pass, never a shared one."""

BATCH_PASS_TOKEN = "owt"
"""The pass token every batch run id of these passes opens with, so no job name collides across passes.

The knockout pass keeps this token rather than taking one of its own: its block ids are all new, so no
(token, model, effort, sitting, block) name it asks for is a name the first pass already owns, and a
second token would be a second thing to keep distinct for no gain.

Bedrock job names are account-wide and a Completed job keeps its name, so a run id that spells only
(model, effort, sitting, block) collides with any earlier pass that used the same block ids -- which is
exactly what happened on 2026-09-02: this pass's ``floor`` and ``deliberation`` legs asked for names the
decoupled-ladder pass (:mod:`sociology.decoupled_plan`) already owned, and Bedrock refused eight legs
with ``ConflictException``. Three characters because the Anthropic and Qwen slugs already truncate the
model end of the name to make room for the run id.
"""

FINGERPRINT_BATCH_PASS_TOKEN = "fpb"
CORRELATION_DOSE_BATCH_PASS_TOKEN = "cdo"
"""A token of its own per new pass, rather than the two of them keeping distinct block ids by hand.

The knockout could share ``owt`` because its blocks were all new. These two passes run in PARALLEL with
each other and with pass C, submitted by three operators who are not reading each other's tables, so the
token is the thing that makes a name collision impossible rather than merely unlikely.
"""


def spec_for(
    game_id: str,
    *,
    credit: str = REFERENCE_CREDIT,
    count: int = REFERENCE_COUNT,
    stake: float = REFERENCE_STAKE,
) -> TransferSpec:
    """Build one dose's spec, naming the credit by its registered variant rather than by its numbers."""
    numerator, denominator = TRANSFER_CREDIT_VARIANTS[credit]
    return TransferSpec(
        game_id=game_id,
        endowment=TRANSFER_ENDOWMENT,
        credit_numerator=numerator,
        credit_denominator=denominator,
        beneficiary_count=count,
        own_stake_scale=stake,
    )


@dataclass(frozen=True, slots=True)
class Cell:
    """One sampled condition: a game, an identity (or the blind baseline), and a dose.

    ``rung`` of None is the identity-blind cell, which is the render with no counterpart paragraph and
    is both the anonymous baseline and the stem the one-inserted-paragraph audit compares against.
    """

    game_id: str
    rung: str | None
    spec: TransferSpec

    @property
    def cell_id(self) -> str:
        """The label the record key and every report group on."""
        return TRANSFER_IDENTITY_BLIND_LABEL if self.rung is None else self.rung

    @property
    def variant(self) -> str:
        """The dose id, which is the row's ``payoff_variant`` and the key's variant segment."""
        return transfer_variant(self.spec)


ONE_WAY_PASS_LADDER_BY_GAME: dict[str, tuple[str, ...]] = {
    ONE_WAY_TRANSFER_GAME_ID: LADDER_RUNGS,
    MATCHED_DECISION_TRANSFER_GAME_ID: (*LADDER_RUNGS, RUNG_SAME_CHECKPOINT_COUPLED),
}
"""The identity ladder the FIRST pass samples in each of its two games.

Spelled here rather than read off :data:`~sociology.transfer_stimulus.RUNGS_BY_GAME`, which is every
clause cell a game HAS. The two are no longer the same set: the knockout added a twin cell the finished
pass never sampled, and a ladder derived from the stimulus registry would silently have grown that pass by
128 records per model -- a table whose arithmetic no longer matched the numbers it was read on.
"""


def identity_cells(
    game_id: str, *, credit: str = REFERENCE_CREDIT, count: int = REFERENCE_COUNT
) -> tuple[Cell, ...]:
    """Build one game's first-pass identity ladder at one dose, plus that game's blind baseline.

    The baseline is FIRST rather than last, because every reading in this design is a contrast against it
    and a table that reads top to bottom should open with what the rest is compared to.
    """
    spec = spec_for(game_id, credit=credit, count=count)
    return (
        Cell(game_id, None, spec),
        *(Cell(game_id, rung, spec) for rung in ONE_WAY_PASS_LADDER_BY_GAME[game_id]),
    )


def dose_cells() -> tuple[Cell, ...]:
    """Build the dose block: the credit-by-count grid minus the reference cell, plus the stake rungs.

    At the top rung throughout, so the dose readings are taken where giving has the most room: a dose
    effect measured at the blind baseline could be a floor effect instead.
    """
    game_id = ONE_WAY_TRANSFER_GAME_ID
    grid = [
        Cell(game_id, RUNG_SAME_CHECKPOINT, spec_for(game_id, credit=credit, count=count))
        for credit in TRANSFER_CREDIT_VARIANTS
        for count in TRANSFER_BENEFICIARY_COUNTS
        if not (credit == REFERENCE_CREDIT and count == REFERENCE_COUNT)
    ]
    stakes = [
        Cell(game_id, RUNG_SAME_CHECKPOINT, spec_for(game_id, stake=stake))
        for stake in TRANSFER_OWN_STAKE_SCALES
        if stake != REFERENCE_STAKE
    ]
    return (*grid, *stakes)


def floor_cells() -> tuple[Cell, ...]:
    """Build the floor block: the top rung and the blind baseline, re-sat at the reference dose."""
    spec = spec_for(ONE_WAY_TRANSFER_GAME_ID)
    return (
        Cell(ONE_WAY_TRANSFER_GAME_ID, None, spec),
        Cell(ONE_WAY_TRANSFER_GAME_ID, RUNG_SAME_CHECKPOINT, spec),
    )


def cells_for(game_id: str, rungs: Sequence[str | None]) -> tuple[Cell, ...]:
    """Build one game's named cells at the reference dose, in the order given.

    ``None`` in ``rungs`` is the identity-blind cell. Ordered by the caller rather than by a registry,
    because the knockout blocks are read top to bottom against their first row and a table that opened
    with a rung would read as a ladder.
    """
    spec = spec_for(game_id)
    return tuple(Cell(game_id, rung, spec) for rung in rungs)


KNOCKOUT_MATCHED_RUNGS: tuple[str | None, ...] = (
    None,
    RUNG_SAME_CHECKPOINT,
    RUNG_DIFFERENT_FAMILY,
    RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
)
"""The twin's four knockout cells: the blind anchor, the top rung, and the record rung with its base.

``different-family`` is here because it is the one-sentence matched-insertion baseline the record rung is
read against. Without it that rung reads against a whole paragraph (the blind cell) or against the
previous pass's numbers, which were sampled on superseded wording.
"""

KNOCKOUT_DRAWN_RUNGS: tuple[str | None, ...] = (None, RUNG_SAME_CHECKPOINT)
"""The drawn game's own blind baseline and its top rung, which is the whole of rung 1.

The baseline is re-measured in this game rather than borrowed from the twin: a draw that fixes the other
sides' figures may itself move an anonymous cell, and a lift read against another game's baseline would
carry that movement.
"""

ONE_WAY_CELLS_BY_BLOCK: dict[str, tuple[Cell, ...]] = {
    BLOCK_IDENTITY_ONE_WAY: identity_cells(ONE_WAY_TRANSFER_GAME_ID),
    BLOCK_IDENTITY_MATCHED: identity_cells(MATCHED_DECISION_TRANSFER_GAME_ID),
    BLOCK_DOSE_ONE_WAY: dose_cells(),
    BLOCK_INTERACTION_ONE_WAY: identity_cells(
        ONE_WAY_TRANSFER_GAME_ID, credit=VALUE_DESTROYING_CREDIT, count=VALUE_DESTROYING_COUNT
    ),
    BLOCK_FLOOR: floor_cells(),
    BLOCK_DELIBERATION: identity_cells(ONE_WAY_TRANSFER_GAME_ID),
}

KNOCKOUT_CELLS_BY_BLOCK: dict[str, tuple[Cell, ...]] = {
    BLOCK_KNOCKOUT_MATCHED: cells_for(MATCHED_DECISION_TRANSFER_GAME_ID, KNOCKOUT_MATCHED_RUNGS),
    BLOCK_KNOCKOUT_DRAWN: cells_for(DRAWN_DECISION_TRANSFER_GAME_ID, KNOCKOUT_DRAWN_RUNGS),
    BLOCK_KNOCKOUT_FLOOR: cells_for(
        MATCHED_DECISION_TRANSFER_GAME_ID, (None, RUNG_SAME_CHECKPOINT)
    ),
}

FINGERPRINT_MATCHED_RUNGS: tuple[str | None, ...] = (
    None,
    RUNG_SAME_CHECKPOINT,
    *BOARD_IDS,
)
"""The fingerprint block's six cells: the anonymous anchor, the TOLD identity, and the four boards.

The told cell is re-measured here rather than read off the knockout, so the board lift can be reported as
a share of a stated identity's lift measured in the same sitting on the same wording. Its
``infers_same_model`` rate is also this pass's floor for that label by construction: it states the
identity and shows no messages, so a reader has nothing to infer it FROM.
"""

FINGERPRINT_FLOOR_RUNGS: tuple[str | None, ...] = (None, *RAW_BOARD_IDS)
"""The second sitting: the anchor and both raw boards, so each model's own-raw and other-raw cells have
a bar. The two crossed corners are read as differences from these, and a bar on both ends of the
headline contrast is what the difference has to clear."""

CORRELATION_DOSE_RECORD_RUNGS: tuple[str | None, ...] = tuple(
    record_rung_id(matched) for matched in MATCHED_ROUNDS_LADDER
)
"""The dose ladder in ladder order, highest stated count first, which is the order the curve prints."""

CORRELATION_DOSE_ANCHOR_RUNGS: tuple[str | None, ...] = (
    None,
    RUNG_SAME_CHECKPOINT,
    RUNG_DIFFERENT_FAMILY,
    RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
    RUNG_SAME_CHECKPOINT_RECORD_0,
)
"""The five cells the curve is read against, all re-measured in the same sitting as the ladder.

``different-family`` is the ladder's own base, which every rung is a difference from.
``different-family-track-record`` is the knockout's universal wording, so the ten-of-ten rung against it
is a free read of a quantified record against an unquantified one. ``same-checkpoint-record-0`` is the
mismatch rung, read against both the top rung and the ladder's base. ``identity-blind`` is the anonymous
baseline the calibration-trap rule wants measured in the same pass and the same game as everything else.
"""

CORRELATION_DOSE_FLOOR_RUNGS: tuple[str | None, ...] = (
    RUNG_DIFFERENT_FAMILY,
    record_rung_id(5),
)
"""The second sitting: the ladder's base and the rung nearest the binary reading's threshold, where a
mirror-shaped model's rate is expected mid-range and resampling noise is largest."""

FINGERPRINT_CELLS_BY_BLOCK: dict[str, tuple[Cell, ...]] = {
    BLOCK_FINGERPRINT_MATCHED: cells_for(
        MATCHED_DECISION_TRANSFER_GAME_ID, FINGERPRINT_MATCHED_RUNGS
    ),
    BLOCK_FINGERPRINT_FLOOR: cells_for(MATCHED_DECISION_TRANSFER_GAME_ID, FINGERPRINT_FLOOR_RUNGS),
}

CORRELATION_DOSE_CELLS_BY_BLOCK: dict[str, tuple[Cell, ...]] = {
    BLOCK_DOSE_RECORD: cells_for(MATCHED_DECISION_TRANSFER_GAME_ID, CORRELATION_DOSE_RECORD_RUNGS),
    BLOCK_DOSE_ANCHORS: cells_for(MATCHED_DECISION_TRANSFER_GAME_ID, CORRELATION_DOSE_ANCHOR_RUNGS),
    BLOCK_DOSE_FLOOR: cells_for(MATCHED_DECISION_TRANSFER_GAME_ID, CORRELATION_DOSE_FLOOR_RUNGS),
}

CONDITION_OWN_RAW = "own-raw"
CONDITION_OTHER_RAW = "other-raw"
CONDITION_OWN_CONTENT_OTHER_WORDS = "own-content-other-words"
CONDITION_OTHER_CONTENT_OWN_WORDS = "other-content-own-words"
BOARD_CONDITIONS: tuple[str, ...] = (
    CONDITION_OWN_RAW,
    CONDITION_OTHER_RAW,
    CONDITION_OWN_CONTENT_OTHER_WORDS,
    CONDITION_OTHER_CONTENT_OWN_WORDS,
)
"""The four corners of the fingerprint 2x2, as the label every pass D record carries.

A derived label on the record rather than a cell attribute, because a board cell is model-independent:
both models read all four boards, which is what gives the between-model read on one board for free. The
condition is what the board IS to the model reading it, so it is a function of the pair.
"""


def board_condition(cell_id: str, model_id: str) -> str | None:
    """Name what one board is to the model reading it, or ``None`` for a cell that is not a board.

    Derived at plan time and written onto every record, and re-derived by the readout from the record's
    own (cell, model) pair and checked against the stored value: the whole 2x2 decomposition is a
    grouping by this label, so a value nobody recomputes is a relabelling nobody would notice.
    """
    if cell_id not in BOARD_IDS:
        return None
    side = BOARD_SIDE_BY_MODEL_ID.get(model_id)
    if side is None:
        raise ValueError(
            f"{model_id!r} reads board {cell_id!r} and is not one of the two models the boards are "
            f"drawn from ({sorted(BOARD_SIDE_BY_MODEL_ID)}), so 'own' and 'other' mean nothing for it. "
            f"A third model on this pass needs boards of its own, not a condition guessed here."
        )
    own_content = BOARD_CONTENT_SIDE[cell_id] == side
    own_wording = BOARD_WORDING_SIDE[cell_id] == side
    if own_content and own_wording:
        return CONDITION_OWN_RAW
    if own_content:
        return CONDITION_OWN_CONTENT_OTHER_WORDS
    if own_wording:
        return CONDITION_OTHER_CONTENT_OWN_WORDS
    return CONDITION_OTHER_RAW


def matched_rounds_thresholds(spec: TransferSpec) -> dict[str, Any]:
    """Compute the two stated-correlation doses at which an expected-value maximiser switches.

    Both are functions of the dose alone, so they are computed from the spec rather than written down:
    a changed credit, beneficiary count or own stake moves them, and a readout marking a stale threshold
    on a curve would put the classification ("steps at the binary threshold") on the wrong rung.

    With credit ``c`` per unit to each of ``n`` others, an own-stake scale ``s`` on what the reader
    keeps, and a stated match probability ``p``:

    - *match-else-independent*: the marginal unit set down costs ``s`` and pays ``c n p``, so the
      maximiser sets everything exactly when ``p > s / (c n)``. The endowment does not enter, and neither
      does what the others do when they do not match, as long as it does not depend on the reader's own
      figure.
    - *binary mismatch*: the others either do the same as the reader or the opposite extreme.
      ``EV(all) = p c n E`` against ``EV(nothing) = s E + (1 - p) c n E``, equal at
      ``p = (s + c n) / (2 c n)``; the endowment cancels here too.

    The own stake is read off the spec rather than assumed to be one, which matters the first time a dose
    moves it: at the registered ``own_stake_scale`` of 0.1 the independence threshold is a tenth of what
    it is at full stake, so a readout marking the full-stake value would print "steps at the independence
    threshold" against a rung two rungs away. Taken as a percentage because that is how the variant
    string spells it, which keeps the fraction exact where the float 0.1 would not.

    Each threshold is returned as an exact fraction plus the two ladder rungs it sits between, which is
    what a readout marks: a threshold between two sampled rungs is readable, one above the top rung says
    the ladder cannot see that reading at all.
    """
    credit = Fraction(spec.credit_numerator, spec.credit_denominator)
    total_return = credit * spec.beneficiary_count
    own_stake = Fraction(spec.own_stake_percent, 100)
    thresholds = {
        "match_else_independent": own_stake / total_return,
        "binary_mismatch": (own_stake + total_return) / (2 * total_return),
    }
    return {
        "credit_per_unit": [spec.credit_numerator, spec.credit_denominator],
        "beneficiary_count": spec.beneficiary_count,
        "own_stake_percent": spec.own_stake_percent,
        "rounds": TRACK_RECORD_ROUNDS,
        "ladder": list(MATCHED_ROUNDS_LADDER),
        "thresholds": {
            name: {
                "fraction": [threshold.numerator, threshold.denominator],
                "probability": float(threshold),
                "between_rungs": _ladder_rungs_around(threshold),
            }
            for name, threshold in thresholds.items()
        },
    }


def _ladder_rungs_around(threshold: Fraction) -> list[int | None]:
    """Name the ladder rungs immediately below and above one threshold probability.

    ``None`` on either side when the threshold sits outside the ladder, which is the honest answer: a
    readout cannot mark a step at a threshold no pair of sampled rungs straddles.
    """
    below = [
        matched
        for matched in MATCHED_ROUNDS_LADDER
        if Fraction(matched, TRACK_RECORD_ROUNDS) <= threshold
    ]
    above = [
        matched
        for matched in MATCHED_ROUNDS_LADDER
        if Fraction(matched, TRACK_RECORD_ROUNDS) > threshold
    ]
    return [max(below) if below else None, min(above) if above else None]


# The one (game, cell, dose) triple two blocks deliberately both plan, and why. The dose grid's
# lowest corner IS the interaction ladder's top rung: the grid needs it as its own benefit floor and the
# ladder needs it as the rung every other rung at that dose is read against. The record key carries the
# block, so the two copies are separate records rather than a collision, and each block's report reads
# its own copy -- which makes the pair a free second estimate of the cell at the cost of 128 records per
# model. Anything NOT listed here that two blocks both plan is a mistake, and the check below says so.
ONE_WAY_CELLS_PLANNED_TWICE: tuple[tuple[str, str, str], ...] = (
    (
        ONE_WAY_TRANSFER_GAME_ID,
        RUNG_SAME_CHECKPOINT,
        transfer_variant(
            spec_for(
                ONE_WAY_TRANSFER_GAME_ID,
                credit=VALUE_DESTROYING_CREDIT,
                count=VALUE_DESTROYING_COUNT,
            )
        ),
    ),
)

DESIGN_LABELS: tuple[str, ...] = tuple(
    sorted(
        {
            *TRANSFER_GAME_IDS,
            *ALL_RUNGS,
            TRANSFER_IDENTITY_BLIND_LABEL,
            *ALL_MODEL_IDS,
            *BLOCKS,
            "one-way",
            "matched",
            "beneficiar",
            "coupled",
            "drawn",
            "fingerprint",
            "board",
            *BOARD_CONDITIONS,
        }
    )
)
"""Every string that names the design to a reader, which is what the blind judge must never see.

One list rather than one per call site, because it is consumed twice: the blindness gate runs it over
both loaded rubrics and the rendered headers before the first judge call, and the tests run it over every
prompt a scripted judge backend actually received. ``beneficiar`` is a stem rather than a word so it
catches both numbers of the noun, and ``matched`` and ``one-way`` are there because either would tell the
judge which of the games it is reading a reply to -- and the twin's counterpart-reasoning rate is
exactly the check that says whether the coupling landed. ``drawn`` is there for the sharpest form of
that: the drawn game's ``treats_others_as_deciding`` rate is the check that says whether its draw landed,
and a judge that could read the game id would report the id.
"""


_PUBLIC_PROMPT_TEXT: tuple[str, ...] = (
    MATCHED_DECISION_TRANSFER_MECHANICS,
    DRAWN_DECISION_TRANSFER_MECHANICS,
    TRANSFER_CLOSING_ORDER,
    *TRANSFER_INSTRUCTIONS.values(),
)
"""The shared prompt text every transfer subject reads, whichever game and scenario it is reading."""

_FILLED_PUBLIC_PROMPT_TEXT: str = re.sub(
    r"\{[a-z_]+\}", " ", "\n".join(_PUBLIC_PROMPT_TEXT)
).casefold()
"""The same text with its placeholders removed, which is what a subject actually reads.

Removed rather than left in, because a placeholder NAME is not text anyone reads: the mechanics say
``{beneficiary_group}`` and the render fills it with "the 3 keepers", so a match on the name would exempt
the stem ``beneficiar`` from the reply check on the strength of a word no subject is ever shown.
"""

DESIGN_LABELS_THE_PROMPTS_PRINT: tuple[str, ...] = tuple(
    label for label in DESIGN_LABELS if label in _FILLED_PUBLIC_PROMPT_TEXT
)
"""Design labels that the shared prompt text itself puts in front of every subject.

Derived rather than listed, so a later edit to one of those constants cannot leave a hand-written
exemption behind. Today it is the stem ``drawn``, from the closing order's "Nothing drawn from the
{destination} is recorded as coming from anyone". A label printed only by the AUTHORED scenario prose --
a destination noun, say -- cannot be derived here, since that text is gitignored and loaded at runtime;
if one ever bites, it arrives as this gate refusing a validation reply that quotes its own frame.
"""

DESIGN_LABELS_IN_AUTHORED_REPLIES: tuple[str, ...] = tuple(
    label for label in DESIGN_LABELS if label not in DESIGN_LABELS_THE_PROMPTS_PRINT
)
"""The labels a validation reply may not carry, which is the full list minus the words the prompts print.

The rubrics are checked against everything in :data:`DESIGN_LABELS`, because there a word like ``drawn``
could only be our own prose describing the rung. A validation reply is different: it is text we author to
imitate a subject, and a subject's reply routinely quotes the note it is answering, so banning a word the
note itself prints refuses a realistic case while carrying not one bit about the design. Three authored
twin replies quote the closing order for exactly that reason, and under the full list every one of them
refused ``judge-validate`` outright -- which would have blocked the whole judge pass, since the gate runs
before the first call.
"""


def model_slug(model_id: str) -> str:
    """Spell a model id as a filesystem- and S3-safe stem, the way the batch backend spells it."""
    return model_id.replace(":", "-").replace(".", "-").replace("/", "-")


def effort_tag(reasoning_effort: str | None) -> str:
    """Name an effort for a key or a filename; the provider default is a value, not an absence."""
    return reasoning_effort or "default"


def transport_for(model_id: str) -> str:
    """Name the transport one roster row runs on: live for the live roster, batch for everything else."""
    return TRANSPORT_LIVE if model_id in LIVE_MODEL_IDS else TRANSPORT_BATCH


@dataclass(frozen=True, slots=True)
class Leg:
    """One row of the sampling plan: one model reading one block's cells at one effort and sitting.

    A leg is exactly one batch job or one live pass, which is what makes it the unit the CLI takes: the
    100-record batch floor is per job, and pooling a block's cells into one job is what clears it.
    """

    model_id: str
    transport: str
    block: str
    cells: tuple[Cell, ...]
    reasoning_effort: str | None
    sitting: str
    on_demand: bool = False
    """Whether this leg waits on a recorded decision (``add-opus``) rather than running by default."""
    batch_prefix: str = TRANSFER_BATCH_PREFIX
    """The S3 prefix this leg's batch inputs and outputs live under: its own pass's, never a shared one.

    On the leg rather than looked up per invocation, because the prefix is what keeps two passes' runs
    from writing over each other, and a submit that resolved it from anything other than the leg could
    resolve it differently from the collect that follows.
    """
    batch_pass_token: str = BATCH_PASS_TOKEN
    """The token this leg's batch run id opens with, so its job name cannot be one another pass owns."""

    @property
    def leg_id(self) -> str:
        """The name this leg is asked for by on the command line."""
        short = MODEL_SHORT_NAMES[self.model_id]
        return f"{short}--{effort_tag(self.reasoning_effort)}--{self.sitting}--{self.block}"

    @property
    def file_stem(self) -> str:
        """The reply and handle filename stem, carrying the FULL model id rather than a short name.

        The full slug on purpose: a filename is what an operator reads months later without the table in
        front of them, and a short name would need this module to decode.
        """
        return (
            f"{model_slug(self.model_id)}--effort-{effort_tag(self.reasoning_effort)}"
            f"--sitting-{self.sitting}--{self.block}"
        )

    @property
    def batch_run_id(self) -> str:
        """The run id the batch backend names its S3 subtree and its job after, per leg.

        Per leg rather than per run, and that is a correctness property rather than tidiness: the backend
        derives both the S3 base and the Bedrock job name from (prefix, run_id, model), so two legs of one
        model sharing a run id would overwrite each other's ``input.jsonl`` and ask the service for the
        same job name. Short, because the job name has a 63-character ceiling and the run id is the only
        part of it that is unique. Opened with :data:`BATCH_PASS_TOKEN` on every leg, not just the blocks
        that collided, because job names are account-wide and a name another pass has used is refused.

        Nothing reads this at collect time: a saved handle carries its own job ARN, S3 URIs, record count
        and digests, and the collect path resolves every one of those from the handle, so renaming this
        leaves already-submitted jobs collectable.
        """
        short = MODEL_SHORT_NAMES[self.model_id]
        return (
            f"{self.batch_pass_token}-{short}-{effort_tag(self.reasoning_effort)}"
            f"-{self.sitting}-{self.block}"
        ).lower()

    @property
    def batch_job_name(self) -> str | None:
        """The Bedrock job name a batch leg's submit asks for; ``None`` on the live transport.

        Recomputed from the two rules the backend applies -- ``jagged-<model slug>-<run id>``, truncated
        at the MODEL end to :data:`~reward_hacking.bedrock_batch.MAX_JOB_NAME_CHARS` so the run id, the
        only unique part, survives -- because an operator debugging a refused submit needs the name in
        ``plan.json`` rather than in a backend log line that was never written.
        """
        if self.transport != TRANSPORT_BATCH:
            return None
        suffix = f"-{self.batch_run_id}"
        stem = f"jagged-{model_slug(self.model_id)}".lower()
        return stem[: MAX_JOB_NAME_CHARS - len(suffix)] + suffix

    @property
    def records(self) -> int:
        """How many records this leg plans, from the design's arithmetic rather than from a render."""
        return len(self.cells) * ROWS_PER_CELL * DRAWS


def _one_way_legs() -> tuple[Leg, ...]:
    """Build the one-way pass's leg table in submit order, on-demand legs flagged rather than dropped.

    The five default blocks for every roster row first, then the two deliberation legs, then Opus 5's two
    identity legs, which wait on its refusal probe.
    """
    default_blocks = (
        BLOCK_IDENTITY_ONE_WAY,
        BLOCK_IDENTITY_MATCHED,
        BLOCK_DOSE_ONE_WAY,
        BLOCK_INTERACTION_ONE_WAY,
        BLOCK_FLOOR,
    )
    legs = [
        Leg(
            model_id,
            transport_for(model_id),
            block,
            ONE_WAY_CELLS_BY_BLOCK[block],
            None,
            SITTING_B if block == BLOCK_FLOOR else SITTING_A,
        )
        for model_id in ALL_MODEL_IDS
        if model_id not in ON_DEMAND_MODEL_IDS
        for block in default_blocks
    ]
    legs.extend(
        Leg(
            model_id,
            transport_for(model_id),
            BLOCK_DELIBERATION,
            ONE_WAY_CELLS_BY_BLOCK[BLOCK_DELIBERATION],
            DELIBERATION_EFFORT,
            SITTING_A,
        )
        for model_id in DELIBERATION_MODEL_IDS
    )
    legs.extend(
        Leg(
            model_id,
            transport_for(model_id),
            block,
            ONE_WAY_CELLS_BY_BLOCK[block],
            None,
            SITTING_A,
            on_demand=True,
        )
        for model_id in ON_DEMAND_MODEL_IDS
        for block in IDENTITY_BLOCKS
    )
    return tuple(legs)


def _knockout_legs() -> tuple[Leg, ...]:
    """Build the knockout pass's leg table: three blocks for each of its three roster rows.

    No on-demand row and no deliberation block. The roster is three models the owner fixed, none of them
    Opus, and this pass runs no second effort: an effort is a second elicitation of one prompt, and the
    knockout's question is about the prompt.
    """
    return tuple(
        Leg(
            model_id,
            transport_for(model_id),
            block,
            KNOCKOUT_CELLS_BY_BLOCK[block],
            None,
            SITTING_B if block == BLOCK_KNOCKOUT_FLOOR else SITTING_A,
            batch_prefix=KNOCKOUT_BATCH_PREFIX,
        )
        for model_id in KNOCKOUT_MODEL_IDS
        for block in KNOCKOUT_BLOCKS
    )


def _pair_roster_legs(
    cells_by_block: Mapping[str, tuple[Cell, ...]],
    *,
    resat_blocks: Sequence[str],
    batch_prefix: str,
    batch_pass_token: str,
) -> tuple[Leg, ...]:
    """Build one of the two-model passes' leg tables: every block for each of the two roster rows.

    One builder for both, because the two passes differ in their cells, their prefix and their token and
    in nothing about how their legs are laid out. A second copy would be a second place for the
    second-sitting rule to drift.
    """
    return tuple(
        Leg(
            model_id,
            transport_for(model_id),
            block,
            cells,
            None,
            SITTING_B if block in resat_blocks else SITTING_A,
            batch_prefix=batch_prefix,
            batch_pass_token=batch_pass_token,
        )
        for model_id in PAIR_MODEL_IDS
        for block, cells in cells_by_block.items()
    )


PASS_ONE_WAY_TRANSFER = "one-way-transfer"
PASS_KNOCKOUT = "knockout"
PASS_FINGERPRINT = "fingerprint"
PASS_CORRELATION_DOSE = "correlation-dose"
"""The four pass ids ``--pass`` selects a table with; each table carries its own."""


@dataclass(frozen=True)
class PlanTable:
    """One pass's whole sampling plan: its cells, its legs, and where its artifacts and jobs live.

    A table rather than module-level constants, because this module now carries two passes over one
    substrate. Everything a subcommand needs to know about which pass it is operating on lives here and
    is selected once, at the command line, so no code path can read one pass's legs while writing into
    another's run directory -- and the finished pass cannot be re-rendered or re-priced by a command run
    without saying which pass it means.

    ``required_rungs_by_game`` is what the coverage check reads: which cells each game must plan exactly
    once at its reference dose, ``None`` standing for the identity-blind cell. Declared per table rather
    than derived from the stimulus, because a table is allowed to sample a subset of a game's rungs (the
    knockout samples three of the twin's nine) and the check still has to refuse a dropped cell.
    """

    pass_id: str
    cells_by_block: dict[str, tuple[Cell, ...]]
    legs: tuple[Leg, ...]
    required_rungs_by_game: dict[str, tuple[str | None, ...]]
    reading_blocks: tuple[str, ...]
    resat_blocks: tuple[str, ...]
    batch_prefix: str
    batch_pass_token: str
    default_run_dir: str
    smoke_live_model_id: str
    smoke_block: str
    cells_planned_twice: tuple[tuple[str, str, str], ...] = ()

    def all_legs(self) -> tuple[Leg, ...]:
        """Every leg this pass can be asked for, on-demand rows included."""
        return self.legs

    def production_legs(self) -> tuple[Leg, ...]:
        """Select the legs this pass runs and reads by default: every leg that is not on demand."""
        return tuple(leg for leg in self.legs if not leg.on_demand)

    def on_demand_legs(self) -> tuple[Leg, ...]:
        """Select the legs that wait on a recorded decision rather than running by default."""
        return tuple(leg for leg in self.legs if leg.on_demand)

    @property
    def legs_by_id(self) -> dict[str, Leg]:
        """Index this pass's legs by the id the command line asks for them by."""
        return {leg.leg_id: leg for leg in self.legs}

    @property
    def blocks(self) -> tuple[str, ...]:
        """Name this pass's block ids, in table order."""
        return tuple(self.cells_by_block)

    def leg_for(self, leg_id: str) -> Leg:
        """Look up one of THIS pass's legs, listing the pass's table when the id is not in it.

        Per table rather than across all of them: a leg id of another pass is a mistyped ``--pass``, and
        resolving it here would submit that pass's job into this pass's run directory.
        """
        leg = self.legs_by_id.get(leg_id)
        if leg is None:
            raise ValueError(
                f"{leg_id!r} is not a leg of the {self.pass_id!r} pass. Known legs: "
                f"{', '.join(sorted(self.legs_by_id))}."
            )
        return leg


ONE_WAY_TRANSFER_PLAN = PlanTable(
    PASS_ONE_WAY_TRANSFER,
    cells_by_block=ONE_WAY_CELLS_BY_BLOCK,
    legs=_one_way_legs(),
    required_rungs_by_game={
        game_id: (None, *rungs) for game_id, rungs in ONE_WAY_PASS_LADDER_BY_GAME.items()
    },
    reading_blocks=IDENTITY_BLOCKS,
    resat_blocks=(BLOCK_FLOOR, BLOCK_DELIBERATION),
    batch_prefix=TRANSFER_BATCH_PREFIX,
    batch_pass_token=BATCH_PASS_TOKEN,
    default_run_dir=DEFAULT_ONE_WAY_RUN_DIR,
    smoke_live_model_id=GPT_OSS_120B_MODEL_ID,
    smoke_block=BLOCK_IDENTITY_ONE_WAY,
    cells_planned_twice=ONE_WAY_CELLS_PLANNED_TWICE,
)
"""The finished first pass: the whole identity ladder in both games, the dose grid, and the interaction.

Untouched by the knockout work. Its run directory is never resumed and its numbers were sampled on
superseded clause wording, so it is here to be re-read and re-priced, not re-run.
"""

KNOCKOUT_PLAN = PlanTable(
    PASS_KNOCKOUT,
    cells_by_block=KNOCKOUT_CELLS_BY_BLOCK,
    legs=_knockout_legs(),
    required_rungs_by_game={
        MATCHED_DECISION_TRANSFER_GAME_ID: KNOCKOUT_MATCHED_RUNGS,
        DRAWN_DECISION_TRANSFER_GAME_ID: KNOCKOUT_DRAWN_RUNGS,
    },
    reading_blocks=KNOCKOUT_READING_BLOCKS,
    resat_blocks=(BLOCK_KNOCKOUT_FLOOR,),
    batch_prefix=KNOCKOUT_BATCH_PREFIX,
    batch_pass_token=BATCH_PASS_TOKEN,
    default_run_dir=DEFAULT_KNOCKOUT_RUN_DIR,
    smoke_live_model_id=LUNA_MODEL_ID,
    smoke_block=BLOCK_KNOCKOUT_MATCHED,
)
"""The knockout: the twin's lift held against a draw that decorrelates it and a record that asserts it.

Luna is the live smoke's model because it is the cheapest live row on this roster; the previous pass's
smoke model (GPT-OSS 120B) is not on it at all.
"""

FINGERPRINT_PLAN = PlanTable(
    PASS_FINGERPRINT,
    cells_by_block=FINGERPRINT_CELLS_BY_BLOCK,
    legs=_pair_roster_legs(
        FINGERPRINT_CELLS_BY_BLOCK,
        resat_blocks=(BLOCK_FINGERPRINT_FLOOR,),
        batch_prefix=FINGERPRINT_BATCH_PREFIX,
        batch_pass_token=FINGERPRINT_BATCH_PASS_TOKEN,
    ),
    required_rungs_by_game={MATCHED_DECISION_TRANSFER_GAME_ID: FINGERPRINT_MATCHED_RUNGS},
    reading_blocks=FINGERPRINT_READING_BLOCKS,
    resat_blocks=(BLOCK_FINGERPRINT_FLOOR,),
    batch_prefix=FINGERPRINT_BATCH_PREFIX,
    batch_pass_token=FINGERPRINT_BATCH_PASS_TOKEN,
    default_run_dir=DEFAULT_FINGERPRINT_RUN_DIR,
    smoke_live_model_id=LUNA_MODEL_ID,
    smoke_block=BLOCK_FINGERPRINT_MATCHED,
)
"""The fingerprint pass: the twin with the counterpart clause replaced by a board of the other sides'
own earlier messages, in the four corners of the content-by-wording 2x2, against the anonymous anchor and
the told identity measured in the same sitting."""

CORRELATION_DOSE_PLAN = PlanTable(
    PASS_CORRELATION_DOSE,
    cells_by_block=CORRELATION_DOSE_CELLS_BY_BLOCK,
    legs=_pair_roster_legs(
        CORRELATION_DOSE_CELLS_BY_BLOCK,
        resat_blocks=(BLOCK_DOSE_FLOOR,),
        batch_prefix=CORRELATION_DOSE_BATCH_PREFIX,
        batch_pass_token=CORRELATION_DOSE_BATCH_PASS_TOKEN,
    ),
    required_rungs_by_game={
        MATCHED_DECISION_TRANSFER_GAME_ID: (
            *CORRELATION_DOSE_RECORD_RUNGS,
            *CORRELATION_DOSE_ANCHOR_RUNGS,
        )
    },
    reading_blocks=CORRELATION_DOSE_READING_BLOCKS,
    resat_blocks=(BLOCK_DOSE_FLOOR,),
    batch_prefix=CORRELATION_DOSE_BATCH_PREFIX,
    batch_pass_token=CORRELATION_DOSE_BATCH_PASS_TOKEN,
    default_run_dir=DEFAULT_CORRELATION_DOSE_RUN_DIR,
    smoke_live_model_id=LUNA_MODEL_ID,
    smoke_block=BLOCK_DOSE_ANCHORS,
)
"""The correlation-dose pass: the stranger twin plus one derived sentence stating a matched count, on a
ladder from ten of ten down to none, with the mismatch sentence also put on a same-checkpoint copy."""

PLAN_TABLES: dict[str, PlanTable] = {
    table.pass_id: table
    for table in (ONE_WAY_TRANSFER_PLAN, KNOCKOUT_PLAN, FINGERPRINT_PLAN, CORRELATION_DOSE_PLAN)
}
"""Every pass this module plans, by the id ``--pass`` selects it with."""

PASS_IDS: tuple[str, ...] = tuple(PLAN_TABLES)


def plan_table(pass_id: str) -> PlanTable:
    """Select one pass's table by id, naming every pass when the id is not one of them."""
    table = PLAN_TABLES.get(pass_id)
    if table is None:
        raise ValueError(
            f"{pass_id!r} is not a pass of this plan; the passes are {list(PLAN_TABLES)}."
        )
    return table


def _declared_cell_problems(table: PlanTable) -> list[str]:
    """Name every game whose planned cells at the reference dose are not the ones the table declares."""
    problems: list[str] = []
    for game_id, required in table.required_rungs_by_game.items():
        planned = {
            cell.cell_id
            for block in table.reading_blocks
            for cell in table.cells_by_block[block]
            if cell.game_id == game_id and cell.spec == spec_for(game_id)
        }
        declared = {TRANSFER_IDENTITY_BLIND_LABEL if rung is None else rung for rung in required}
        if planned != declared:
            problems.append(
                f"{game_id} plans {sorted(planned - declared) or 'nothing extra'} that the table does "
                f"not declare and misses {sorted(declared - planned) or 'nothing'} at its reference dose"
            )
    return problems


def _double_planning_problems(table: PlanTable) -> list[str]:
    """Name every cell two blocks both plan unregistered, and every re-sit with no first sitting."""
    problems: list[str] = []
    seen: dict[tuple[str, str, str], str] = {}
    for block, cells in table.cells_by_block.items():
        if block in table.resat_blocks:
            # These re-sample cells another block owns on purpose: a floor block is a second sitting and
            # a deliberation block a second effort, and the record key separates them by those fields.
            continue
        for cell in cells:
            key = (cell.game_id, cell.cell_id, cell.variant)
            if key in seen and key not in table.cells_planned_twice:
                problems.append(f"{key} is planned by both {seen[key]} and {block}")
            seen[key] = block
    problems.extend(
        f"{block} re-sits {(cell.game_id, cell.cell_id, cell.variant)}, which no reading block of this "
        f"pass samples, so its delta would be a floor for a cell that has no first sitting"
        for block in table.resat_blocks
        for cell in table.cells_by_block[block]
        if (cell.game_id, cell.cell_id, cell.variant) not in seen
    )
    return problems


def _assert_cells_well_formed(table: PlanTable) -> None:
    """Refuse a block table that drops a declared cell, adds one, or re-sits an unsampled one.

    The split between blocks is arithmetic nobody re-derives while operating, so every way of getting it
    wrong is checked: a cell the table declares and no reading block plans, a cell planned that the table
    never declared, a (game, cell, dose) triple two reading blocks both plan without being registered in
    ``cells_planned_twice``, and a second-sitting block re-sampling a cell no reading block ever sampled.
    Each of those leaves the per-leg record counts looking exactly right.
    """
    problems = [*_declared_cell_problems(table), *_double_planning_problems(table)]
    if problems:
        raise RuntimeError(
            f"the {table.pass_id!r} block table is malformed: {'; '.join(problems)}. Every declared cell "
            f"is read against the blind baseline of its own game at the reference dose, and an "
            f"unregistered cell paid for twice under two block labels would be pooled by one report and "
            f"split by another."
        )


def _assert_legs_well_formed(table: PlanTable) -> None:
    """Refuse a leg table with duplicate ids, a stray namespace, or job names that collide once truncated.

    The truncation check is the one worth spelling out. Bedrock caps a job name at
    :data:`~reward_hacking.bedrock_batch.MAX_JOB_NAME_CHARS`, and the backend truncates the MODEL end
    rather than the run-id end, so two long ids sharing a prefix would ask the service for one name and
    the second submit would fail (or worse, be read as the first job). Recomputed here from the same two
    rules the backend applies, because the alternative is finding out at submit time with a paid upload
    already in S3.

    Distinct within this table is not enough, because job names are account-wide: every batch run id must
    also open with the table's own pass token, which is what keeps a name here from being one an earlier
    pass with the same block ids already used. The cross-pass check itself lives in the tests, where the
    other passes' tables can be imported without this module depending on them.

    The batch floor is checked here as well as at submit time. ``refuse_below_batch_floor`` refuses the
    one leg being submitted, which is the right place to catch a stimulus that shrank; this catches a
    BLOCK whose cell list is too thin to clear the floor at all, at import, before anything is priced.
    """
    ids = [leg.leg_id for leg in table.legs]
    duplicated = sorted({name for name in ids if ids.count(name) > 1})
    job_names: dict[str, str] = {}
    collisions: list[str] = []
    for leg in table.legs:
        if leg.batch_prefix != table.batch_prefix or leg.batch_pass_token != table.batch_pass_token:
            collisions.append(
                f"{leg.leg_id} carries prefix {leg.batch_prefix!r} and token {leg.batch_pass_token!r}, "
                f"not this pass's {table.batch_prefix!r} and {table.batch_pass_token!r}"
            )
        name = leg.batch_job_name
        if name is None:
            continue
        if leg.records < MIN_BATCH_RECORDS:
            collisions.append(
                f"{leg.leg_id} plans {leg.records} records, below the {MIN_BATCH_RECORDS}-record "
                f"per-job batch floor"
            )
        if len(leg.batch_run_id) + 1 >= MAX_JOB_NAME_CHARS:
            collisions.append(
                f"{leg.leg_id} has a {len(leg.batch_run_id) + 1}-character run-id suffix"
            )
        if not leg.batch_run_id.startswith(f"{table.batch_pass_token}-"):
            collisions.append(f"{leg.leg_id} run id {leg.batch_run_id!r} lacks the pass token")
        if name in job_names:
            collisions.append(f"{leg.leg_id} and {job_names[name]} both truncate to {name!r}")
        job_names[name] = leg.leg_id
    if duplicated or collisions:
        raise RuntimeError(
            f"the {table.pass_id!r} leg table is malformed: duplicate leg ids {duplicated or 'none'}; "
            f"batch job-name problems {collisions or 'none'}."
        )


for _table in PLAN_TABLES.values():
    _assert_cells_well_formed(_table)
    _assert_legs_well_formed(_table)

LEGS_BY_ID: dict[str, Leg] = {
    leg_id: leg for table in PLAN_TABLES.values() for leg_id, leg in table.legs_by_id.items()
}
"""Every leg of every pass, by id. Ids do not collide across passes: a leg id carries its block, and the
two passes' block ids are disjoint (which is what keeps their Bedrock job names distinct too)."""


def _assert_leg_ids_are_unique_across_passes() -> None:
    """Refuse two passes that name a leg the same thing, which the union above would silently merge."""
    counted = sum(len(table.legs) for table in PLAN_TABLES.values())
    if counted != len(LEGS_BY_ID):
        raise RuntimeError(
            f"the passes plan {counted} legs under {len(LEGS_BY_ID)} distinct leg ids, so at least one "
            f"id names a leg in two passes and a lookup by id alone would resolve whichever was built "
            f"last. Leg ids carry the block, so this means two passes share a block id -- which also "
            f"means their Bedrock job names collide."
        )


_assert_leg_ids_are_unique_across_passes()


def _assert_roster_clears_the_cap() -> None:
    """Refuse at import if any batch row's own ceiling sits below this pass's flat reply cap.

    The failure this prevents has happened on this roster: a job whose ``maxTokens`` exceeds a model's
    ceiling reports ``Completed`` with every record errored, after burning the queue time to find out.
    """
    below = [
        f"{model.model_id} (ceiling {model.max_tokens_limit})"
        for model in BATCH_ROSTER
        if model.model_id in BATCH_MODEL_IDS and model.max_tokens_limit < MAX_TOKENS
    ]
    if below:
        raise RuntimeError(
            f"these transfer batch rows cap output below the flat cap of {MAX_TOKENS}: "
            f"{'; '.join(below)}. A job asking for more comes back Completed with every record errored, "
            f"so either drop the row or give this pass a per-model cap."
        )


_assert_roster_clears_the_cap()


def leg_for(leg_id: str) -> Leg:
    """Look up one leg of ANY pass by id, listing every leg when the id is not one of them.

    The pass-blind lookup, for callers that hold a leg id and no pass -- a telemetry-schema test, the
    refusal canary's fixtures. Operator paths use :meth:`PlanTable.leg_for` instead, which refuses a leg
    belonging to a pass other than the one selected.
    """
    leg = LEGS_BY_ID.get(leg_id)
    if leg is None:
        raise ValueError(
            f"{leg_id!r} is not a leg of any pass in this plan. Known legs: "
            f"{', '.join(sorted(LEGS_BY_ID))}."
        )
    return leg


def record_key(  # noqa: PLR0913 - one argument per axis of the record's identity
    *,
    block: str,
    game_id: str,
    cell: str,
    variant: str,
    scenario_id: str,
    polarity: str,
    model_id: str,
    reasoning_effort: str | None,
    sitting: str,
    draw: int,
) -> str:
    """One planned call's stable identity, derived from content and never from execution order.

    Effort and sitting ride in the key because neither ever pools: two efforts are two elicitations of
    the same prompt, and the two sittings ARE the floor measurement, so collapsing either would erase
    the thing being measured. So does the polarity, which is this pass's counterbalance.
    """
    return (
        f"{block}|{game_id}|{cell}|{variant}|{scenario_id}|{polarity}|{model_id}"
        f"|effort={effort_tag(reasoning_effort)}|sitting={sitting}|draw={draw}"
    )


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """One fully rendered call, carrying every label its reply record will store.

    Every field except ``prompt`` lands on the reply row, and ``prompt`` deliberately does not: the text
    is re-derivable from the labels plus the stimulus file, and a reply file that stored it would be a
    second copy of the authored frames and clauses on disk for no analytical gain.
    """

    key: str
    block: str
    game_id: str
    cell: str
    variant: str
    scenario_id: str
    polarity: str
    prompt_id: str
    endowment: int
    credit_numerator: int
    credit_denominator: int
    beneficiary_count: int
    own_stake_scale: float
    model_id: str
    transport: str
    reasoning_effort: str | None
    sitting: str
    draw: int
    prompt: str
    board_condition: str | None = None
    """What the board this call renders is to the model reading it, or ``None`` outside the board cells.

    Derived from the (cell, model) pair by :func:`board_condition` and stored, because a board cell is
    model-independent by design: both models read all four boards, and the condition is the only thing
    that differs between them. Every reading of pass D groups by it.
    """

    def metadata(self) -> dict[str, Any]:
        """Return the sidecar labels the batch path hashes into the cell digest."""
        return {
            "key": self.key,
            "block": self.block,
            "game_id": self.game_id,
            "cell": self.cell,
            "variant": self.variant,
            "scenario_id": self.scenario_id,
            "polarity": self.polarity,
            "prompt_id": self.prompt_id,
            "endowment": self.endowment,
            "credit_numerator": self.credit_numerator,
            "credit_denominator": self.credit_denominator,
            "beneficiary_count": self.beneficiary_count,
            "own_stake_scale": self.own_stake_scale,
            "reasoning_effort": self.reasoning_effort,
            "sitting": self.sitting,
            "draw": self.draw,
            "board_condition": self.board_condition,
        }


def render_cell_row(
    cell: Cell, stimulus: TransferStimulus, *, scenario: TransferScenario, polarity: str
) -> dict[str, Any]:
    """Render one (cell, scenario, polarity) into its prompt row, clause and all."""
    clause = (
        None
        if cell.rung is None
        else clause_for(
            stimulus,
            game_id=cell.game_id,
            rung=cell.rung,
            spec=cell.spec,
            scenario=scenario,
        )
    )
    rows = generate_transfer_prompt_rows(
        cell.game_id,
        GRADING,
        spec=cell.spec,
        scenario=scenario,
        clause=clause,
        clause_label=cell.cell_id,
        polarity=polarity,
    )
    return rows[0]


def planned_calls_for_leg(leg: Leg, stimulus: TransferStimulus) -> list[PlannedCall]:
    """Render every planned call for one leg, in the fixed cell/scenario/polarity/draw order.

    The order is load-bearing on the batch path: submit and collect both call this, and the prompt and
    cell digests compare positionally, so a reordering here would be caught as a digest mismatch rather
    than silently attaching replies to the wrong cells.
    """
    calls: list[PlannedCall] = []
    for cell in leg.cells:
        for scenario in stimulus.scenarios:
            for polarity in ANSWER_POLARITIES:
                row = render_cell_row(cell, stimulus, scenario=scenario, polarity=polarity)
                calls.extend(
                    _planned_call(
                        leg, cell=cell, scenario=scenario, polarity=polarity, row=row, draw=draw
                    )
                    for draw in range(DRAWS)
                )
    if len(calls) != leg.records:
        raise RuntimeError(
            f"leg {leg.leg_id} rendered {len(calls)} calls but its table row says {leg.records}: the "
            f"roster the renderer walks and this plan's arithmetic disagree."
        )
    return calls


def _planned_call(  # noqa: PLR0913 - one keyword per axis of the call's identity
    leg: Leg,
    *,
    cell: Cell,
    scenario: TransferScenario,
    polarity: str,
    row: dict[str, Any],
    draw: int,
) -> PlannedCall:
    """Assemble one planned call from a rendered prompt row."""
    return PlannedCall(
        key=record_key(
            block=leg.block,
            game_id=cell.game_id,
            cell=cell.cell_id,
            variant=cell.variant,
            scenario_id=scenario.scenario_id,
            polarity=polarity,
            model_id=leg.model_id,
            reasoning_effort=leg.reasoning_effort,
            sitting=leg.sitting,
            draw=draw,
        ),
        block=leg.block,
        game_id=cell.game_id,
        cell=cell.cell_id,
        variant=cell.variant,
        scenario_id=scenario.scenario_id,
        polarity=polarity,
        prompt_id=str(row["prompt_id"]),
        endowment=cell.spec.endowment,
        credit_numerator=cell.spec.credit_numerator,
        credit_denominator=cell.spec.credit_denominator,
        beneficiary_count=cell.spec.beneficiary_count,
        own_stake_scale=cell.spec.own_stake_scale,
        model_id=leg.model_id,
        transport=leg.transport,
        reasoning_effort=leg.reasoning_effort,
        sitting=leg.sitting,
        draw=draw,
        prompt=str(row["prompt"]),
        board_condition=board_condition(cell.cell_id, leg.model_id),
    )


# Single hyphens inside the label only: a `--` starts the next prompt_id segment, and a class that
# swallowed it would eat the polarity segment too.
_FRAMING_SEGMENT_RE = re.compile(r"--framing-[a-z0-9]+(?:-[a-z0-9]+)*")


def blind_stem_id(prompt_id: str) -> str:
    """Rewrite a cell's prompt_id into the identity-blind render's id for the same underlying row.

    One substitution, checked for a single match: the framing segment is what distinguishes a cell's row
    from the clause-free stem it must otherwise be byte-identical to, and a pattern that matched twice
    would rewrite the wrong one.
    """
    matches = _FRAMING_SEGMENT_RE.findall(prompt_id)
    if len(matches) != 1:
        raise ValueError(
            f"prompt_id {prompt_id!r} carries {len(matches)} framing segments, expected exactly one; "
            f"the audit rewrites that segment to find this row's clause-free stem."
        )
    return _FRAMING_SEGMENT_RE.sub(f"--framing-{TRANSFER_IDENTITY_BLIND_LABEL}", prompt_id)


def blind_stems(cells: Sequence[Cell], stimulus: TransferStimulus) -> dict[str, str]:
    """Render the identity-blind stem of every (game, dose, scenario, polarity) these cells cover.

    Keyed by the stem's own prompt_id, which is what :func:`blind_stem_id` rewrites a cell's id into.
    """
    stems: dict[str, str] = {}
    for cell in {Cell(cell.game_id, None, cell.spec) for cell in cells}:
        for scenario in stimulus.scenarios:
            for polarity in ANSWER_POLARITIES:
                row = render_cell_row(cell, stimulus, scenario=scenario, polarity=polarity)
                stems[str(row["prompt_id"])] = str(row["prompt"])
    return stems


def audit_planned_calls(calls: Sequence[PlannedCall], stimulus: TransferStimulus) -> int:
    """Run the one-inserted-paragraph audit over every planned prompt; return how many passed.

    The whole design rests on this: a cell differs from its identity-blind stem by one inserted
    paragraph, so a movement between cells is attributable to the clause and to nothing else.
    """
    cells = tuple(
        {
            Cell(
                call.game_id,
                None,
                TransferSpec(
                    game_id=call.game_id,
                    endowment=call.endowment,
                    credit_numerator=call.credit_numerator,
                    credit_denominator=call.credit_denominator,
                    beneficiary_count=call.beneficiary_count,
                    own_stake_scale=call.own_stake_scale,
                ),
            )
            for call in calls
        }
    )
    stems = blind_stems(cells, stimulus)
    for call in calls:
        stem_id = blind_stem_id(call.prompt_id)
        if stem_id not in stems:
            raise ValueError(
                f"{call.prompt_id!r} has no identity-blind stem at {stem_id!r}; the audit has nothing "
                f"to compare this cell's render against."
            )
        if call.cell == TRANSFER_IDENTITY_BLIND_LABEL:
            # The blind cell IS the stem, so what there is to check is that it inserts nothing at all --
            # which is also the check that the baseline and the audit's reference are one render rather
            # than two that happen to agree today.
            if call.prompt != stems[stem_id]:
                raise ValueError(
                    f"{call.prompt_id!r} is the identity-blind cell and does not equal its own stem, so "
                    f"the baseline every reading is a contrast against is not the text the audit "
                    f"compares every other cell to."
                )
            continue
        assert_counterpart_paragraph_is_the_only_insertion(
            stem=stems[stem_id], rendered=call.prompt, prompt_id=call.prompt_id
        )
    return len(calls)


MECHANICS_SECTION_OFFSET = 0
CLAUSE_SECTION_OFFSET = 2
"""Where the two sections the games may differ in sit, counted from the end of the frame.

A rendered prompt is the frame, then the mechanics, the closing order, the counterpart paragraph and the
answer instruction. Positions rather than a text search, because the check is that NOTHING ELSE moved --
but counted past the frame rather than from zero, because an authored frame may be several paragraphs
and a fixed index would then be checking the wrong two sections while still passing on the roster it was
written against.
"""


def frame_section_count(scenario: TransferScenario) -> int:
    """Count the paragraphs one authored frame contributes, which both games render identically."""
    return scenario.frame.count("\n\n") + 1


def audit_games_differ_by_mechanics_and_clause(stimulus: TransferStimulus) -> int:
    """Assert EVERY PAIR of the games differs, for one (scenario, dose, polarity), in two sections only.

    Every reading of this substrate is a contrast between two of the games, so each pair has to differ in
    the mechanics paragraph and in the counterpart paragraph and in nothing else -- not in the closing
    order, which is what makes the beneficiaries unobservable and unattributable in all of them. If that
    group drifted between two renders, the twin's coupling would be confounded with observability and
    attribution, which is exactly the flaw this design was corrected for; and the knockout's own contrast,
    twin against drawn, would be confounded the same way.

    Pairwise rather than against the one-way game alone: the knockout never reads the one-way arm at all,
    so a drift between the twin and the drawn game would sit in the pass's headline number and pass a
    check written against a third game. That pair is pinned tighter still, inside the mechanics section:
    the drawn game is the twin with exactly the two sentences of
    :data:`~games.prompts.DRAWN_DECISION_REPLACEMENTS` swapped, and the rendered paragraphs are held to
    that sentence by sentence, because the section check alone would pass a third sentence drifting
    between them and the knockout is a difference taken over exactly those two paragraphs. The one-way
    pairs carry no sentence pin: their mechanics describe different situations wholesale. Returns how
    many pairs were compared.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        for polarity in ANSWER_POLARITIES:
            rendered = {
                game_id: str(
                    render_cell_row(
                        Cell(game_id, RUNG_SAME_CHECKPOINT, spec_for(game_id)),
                        stimulus,
                        scenario=scenario,
                        polarity=polarity,
                    )["prompt"]
                ).split("\n\n")
                for game_id in TRANSFER_GAME_IDS
            }
            frame_sections = frame_section_count(scenario)
            expected = [
                frame_sections + MECHANICS_SECTION_OFFSET,
                frame_sections + CLAUSE_SECTION_OFFSET,
            ]
            where = f"{scenario.scenario_id!r}/{polarity!r}"
            for index, first in enumerate(TRANSFER_GAME_IDS):
                for second in TRANSFER_GAME_IDS[index + 1 :]:
                    _assert_one_pair_differs_by_two_sections(
                        rendered[first],
                        rendered[second],
                        games=(first, second),
                        expected=expected,
                        where=where,
                        frame_sections=frame_sections,
                    )
                    compared += 1
            assert_drawn_render_is_the_twin_render_with_the_replacements(
                twin_mechanics=rendered[MATCHED_DECISION_TRANSFER_GAME_ID][expected[0]],
                drawn_mechanics=rendered[DRAWN_DECISION_TRANSFER_GAME_ID][expected[0]],
                spec=spec_for(DRAWN_DECISION_TRANSFER_GAME_ID),
                scenario=scenario,
                where=where,
            )
    return compared


def _assert_one_pair_differs_by_two_sections(  # noqa: PLR0913 - the pair, plus what names the failure
    ours: Sequence[str],
    theirs: Sequence[str],
    *,
    games: tuple[str, str],
    expected: Sequence[int],
    where: str,
    frame_sections: int,
) -> None:
    """Compare one pair of rendered prompts section by section, naming both games in any refusal."""
    if len(ours) != len(theirs):
        raise ValueError(
            f"{games[0]} and {games[1]} render {len(ours)} and {len(theirs)} sections for {where}, so "
            f"they cannot be compared section by section at all."
        )
    differing = [
        index for index, (mine, yours) in enumerate(zip(ours, theirs, strict=True)) if mine != yours
    ]
    if differing != list(expected):
        raise ValueError(
            f"the prompts of {games[0]} and {games[1]} for {where} differ in sections {differing}, "
            f"expected exactly {list(expected)} (the mechanics paragraph and the counterpart paragraph, "
            f"past a frame of {frame_sections} paragraphs). Anything else that moved -- the closing order "
            f"most of all -- confounds this pair's contrast with observability and attribution."
        )


def refuse_below_batch_floor(leg: Leg, n_records: int) -> None:
    """Refuse a batch leg under the service's record floor, in the levers THIS CLI has.

    ``BedrockBatchBackend.submit`` refuses too, but its message offers a ``--repeats`` multiplier this
    CLI does not define, and it refuses only after a model-availability call and an upload. The floor is
    per job and not adjustable, so pooling a block's cells into one job is what clears it.
    """
    if n_records >= MIN_BATCH_RECORDS:
        return
    raise ValueError(
        f"batch inference needs at least {MIN_BATCH_RECORDS} records per job and leg {leg.leg_id} "
        f"renders {n_records} ({len(leg.cells)} cells x {ROWS_PER_CELL} rows x {DRAWS} draws). The "
        f"floor is per job and not adjustable, so either raise DRAWS, pool more cells into the block, "
        f"or run this leg on live Converse, which has no floor."
    )


def batch_price_per_mtok(model_id: str) -> tuple[float, float]:
    """Return one batch model's verified (input, output) price per million tokens."""
    roster = roster_model(model_id)
    return roster.batch_price_in_per_mtok, roster.batch_price_out_per_mtok
