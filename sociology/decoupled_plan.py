"""The decoupled-ladder sampling plan: roster, cells, blocks, legs, keys, and the prompt audit.

What runs is a fixed table of legs (:func:`all_legs`), one row per (model, block, effort, sitting),
rather than something assembled per invocation -- so the record counts are arithmetic anybody can
check against the design, and a leg id names one job's worth of work on the command line.

Four blocks, and the reason each exists:

- ``anchor`` is the published 2x2 this pass replicates: counterpart identity (same weights vs a
  different system) crossed with decision coupling (its decision travels with yours vs not). Two of
  the roster's rows are rows of that paper's own table, so the anchor is a same-model replication
  rather than a same-lineage one.
- ``floor`` re-samples BOTH decoupled anchor cells as a second sitting at an identical config. Its
  delta against the first sitting is the run-to-run floor every other contrast has to clear, and it
  is the only way to tell a real 5-point identity effect from resampling noise. Both cells rather
  than the cheapest one, because run-to-run noise is widest near a rate of one half and vanishes
  near zero: a floor read only on a near-zero control clears anything.
- ``deliberation`` re-runs the anchor at ``high`` reasoning effort on the two models that take an
  effort knob. On the batch transport that field is documented but unverified, so the readout
  compares output-token distributions per effort leg rather than trusting that it took effect.
- ``ladder`` walks seven counterpart identities inside the decoupled condition, which is the only
  place identity has room to show once coupling is held off. Two of the seven are already sampled by
  the anchor -- ``same-checkpoint`` IS its decoupled same-weights cell and ``different-family`` is
  its fifth cell -- so the ladder block plans the remaining five.

Cells and rungs are one namespace on purpose: :func:`sociology.decoupled_stimulus.clause_for`
resolves either, and ``same-checkpoint`` resolves to the same text as the anchor's
``same-weights-uncorrelated`` cell, so the ladder is byte-comparable to the anchor rather than
comparable-by-assertion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from games.prompts import (
    FRAMING_DIFFERENT_AI,
    FRAMING_DIFFERENT_AI_COUPLED,
    FRAMING_SAME_WEIGHTS_UNCORRELATED,
    FRAMING_TWIN,
    FRAMING_UNSTATED,
    LABEL_PRINT_ORDERS,
    SPLIT_EVAL,
    about_the_other_side,
    generate_counterpart_clause_prompt_rows,
)
from games.rewards import GRADING_SELF
from reward_hacking.bedrock_batch import (
    BATCH_ROSTER,
    MAX_JOB_NAME_CHARS,
    MIN_BATCH_RECORDS,
    roster_model,
)
from reward_hacking.model_backend import DEFAULT_BEDROCK_MAX_TOKENS
from sociology.decoupled_stimulus import (
    LADDER_RUNGS,
    RUNG_DIFFERENT_FAMILY,
    RUNG_SAME_CHECKPOINT,
    clause_for,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sociology.decoupled_stimulus import DecoupledStimulus

logger = logging.getLogger(__name__)

OPUS_MODEL_ID = "global.anthropic.claude-opus-5"
SONNET_MODEL_ID = "global.anthropic.claude-sonnet-5"
LUNA_MODEL_ID = "global.openai.gpt-5.6-luna"
GPT_OSS_120B_MODEL_ID = "openai.gpt-oss-120b-1:0"
GPT_OSS_20B_MODEL_ID = "openai.gpt-oss-20b-1:0"

BATCH_MODEL_IDS: tuple[str, ...] = (
    OPUS_MODEL_ID,
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "deepseek.v3.2",
    GPT_OSS_120B_MODEL_ID,
    GPT_OSS_20B_MODEL_ID,
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "moonshotai.kimi-k2.5",
    "zai.glm-4.7",
)
"""The batch roster, one pooled job per leg. Haiku 4.5 and DeepSeek V3.2 are the replication rows."""

LIVE_MODEL_IDS: tuple[str, ...] = (LUNA_MODEL_ID, SONNET_MODEL_ID)
"""The live roster (``BedrockBackend``, resume-by-key): neither row is batch-capable.

Sonnet 5 is the Anthropic frontier row. It joined on 2026-09-02, once Opus 5 turned out to be
unmeasurable on this stimulus (:data:`UNREADABLE_MODEL_IDS`); the earlier "no Sonnet 5" rested on
Opus being the batch-capable Anthropic frontier row, and that premise is gone. Same sampler as Luna:
the flat cap, no effort knob, nothing else.
"""

UNREADABLE_MODEL_IDS: tuple[str, ...] = (OPUS_MODEL_ID,)
"""Rows whose every leg is on demand because the model cannot be read on this stimulus.

Opus 5 is not a read: Anthropic's safety classifier refused ~97% of this pass's matrix-game prompts
on 2026-09-02 (Converse ``stopReason=content_filtered``, 2,613 of its 2,688 anchor and floor records
on the batch path, reproduced 18/20 on the live path), so its 75 parsed rows are a classifier-selected
subset rather than a rate. Its anchor and floor legs stay in the table because their reply files exist
on disk and the plan has to describe them; flagging them on demand keeps them out of the production
totals and makes any re-submit a recorded decision rather than a default.
"""

CHEAP_BATCH_MODEL_IDS: tuple[str, ...] = tuple(
    model_id for model_id in BATCH_MODEL_IDS if model_id != OPUS_MODEL_ID
)
"""Everything but Opus 5, whose output price is eight times the next row's: its ladder runs on demand."""

MODEL_SHORT_NAMES: dict[str, str] = {
    OPUS_MODEL_ID: "opus-5",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": "haiku-4-5",
    "deepseek.v3.2": "deepseek-v3-2",
    GPT_OSS_120B_MODEL_ID: "gpt-oss-120b",
    GPT_OSS_20B_MODEL_ID: "gpt-oss-20b",
    "qwen.qwen3-235b-a22b-2507-v1:0": "qwen3-235b",
    "moonshotai.kimi-k2.5": "kimi-k2-5",
    "zai.glm-4.7": "glm-4-7",
    LUNA_MODEL_ID: "luna",
    SONNET_MODEL_ID: "sonnet-5",
}
"""Short names for leg ids and batch job names, which have a 63-character ceiling to live inside."""

ALL_MODEL_IDS: tuple[str, ...] = (*BATCH_MODEL_IDS, *LIVE_MODEL_IDS)

TRANSPORT_BATCH = "batch"
TRANSPORT_LIVE = "live"


def transport_for(model_id: str) -> str:
    """Name the transport one roster row runs on: live for the live roster, batch for everything else."""
    return TRANSPORT_LIVE if model_id in LIVE_MODEL_IDS else TRANSPORT_BATCH


GRADING = GRADING_SELF
"""A row label only; nothing is trained here, and the reward function never runs."""

GAMES: tuple[str, ...] = ("twin-pd", "public-goods")
"""The two banked one-shot games whose renderer owns the counterpart paragraph."""

ANCHOR_CELLS: tuple[str, ...] = (
    FRAMING_TWIN,
    FRAMING_SAME_WEIGHTS_UNCORRELATED,
    FRAMING_DIFFERENT_AI_COUPLED,
    FRAMING_DIFFERENT_AI,
    RUNG_DIFFERENT_FAMILY,
)
"""The published 2x2 plus one cell, and the fifth is there to unconfound the paper's own contrast.

Four cells are the 2x2: same/coupled, same/decoupled, different/coupled, different/decoupled (their
Control). The two public decoupled constants differ in the counterpart's identity AND in whether the
clause spells out that the two decisions need not agree -- ``same-weights-uncorrelated`` carries that
denial sentence and ``different-ai`` does not -- so the paper-shaped identity-in-decoupled contrast is
confounded with the denial. ``different-family`` is ``different-ai`` plus the same denial sentence,
which gives a clean identity pair against ``same-weights-uncorrelated`` and makes ``different-ai``
versus ``different-family`` a matched-insertion read of the denial itself.
"""

FLOOR_CELLS: tuple[str, ...] = (FRAMING_SAME_WEIGHTS_UNCORRELATED, FRAMING_DIFFERENT_AI)
"""The two cells re-sampled as sitting B: both decoupled anchor cells, not just the Control.

A floor measured only on the near-zero Control understates run-to-run noise everywhere it matters,
because the noise on a rate is widest near one half. The same-weights cell is the one that sits in
the readable band, so it is the one whose resampling spread the identity contrasts must clear.
"""

BLOCK_ANCHOR = "anchor"
BLOCK_FLOOR = "floor"
BLOCK_DELIBERATION = "deliberation"
BLOCK_LADDER = "ladder"
BLOCKS: tuple[str, ...] = (BLOCK_ANCHOR, BLOCK_FLOOR, BLOCK_DELIBERATION, BLOCK_LADDER)

SITTING_A = "A"
SITTING_B = "B"
SITTINGS: tuple[str, ...] = (SITTING_A, SITTING_B)

DRAWS = 8
"""Draws per prompt row. Every draw is a separate record key, never a pooled resample."""

DELIBERATION_MODEL_IDS: tuple[str, ...] = (LUNA_MODEL_ID, GPT_OSS_120B_MODEL_ID)
DELIBERATION_EFFORT = "high"

RUNGS_THE_ANCHOR_ALREADY_SAMPLES: tuple[str, ...] = (RUNG_SAME_CHECKPOINT, RUNG_DIFFERENT_FAMILY)
"""The two rungs the anchor block covers: its decoupled same-weights cell, and its fifth cell."""

LADDER_BLOCK_CELLS: tuple[str, ...] = tuple(
    rung for rung in LADDER_RUNGS if rung not in RUNGS_THE_ANCHOR_ALREADY_SAMPLES
)
"""The five rungs only the ladder block samples, in ladder order."""


def _assert_the_ladder_is_covered_exactly_once() -> None:
    """Refuse a ladder split that leaves a rung unsampled, samples one twice, or mislabels one.

    The split between blocks is arithmetic nobody re-derives while operating, so every way of getting
    it wrong is checked here: a rung neither block samples, a rung both sample (paid for twice and
    filed under two blocks), a block cell that is not a rung at all, and an anchor cell that IS a rung
    without being declared as one -- that last one is how a rung added to the anchor would end up in
    the ladder block as well. Every one of those mistakes leaves the per-leg counts looking right.
    """
    planned = set(LADDER_BLOCK_CELLS)
    # `same-checkpoint` is covered by clause identity rather than by name: its text IS the anchor's
    # `same-weights-uncorrelated` cell, which is the whole reason the ladder is byte-comparable to it.
    covered = set(RUNGS_THE_ANCHOR_ALREADY_SAMPLES)
    unsampled = sorted(set(LADDER_RUNGS) - planned - covered)
    doubled = sorted(planned & covered)
    stray = sorted(planned - set(LADDER_RUNGS))
    undeclared = sorted((set(ANCHOR_CELLS) & set(LADDER_RUNGS)) - covered)
    if unsampled or doubled or stray or undeclared:
        raise RuntimeError(
            f"the ladder block plans {list(LADDER_BLOCK_CELLS)} against the {len(LADDER_RUNGS)}-rung "
            f"ladder {list(LADDER_RUNGS)} and the anchor cells {list(ANCHOR_CELLS)}: rungs nothing "
            f"samples {unsampled or 'none'}; rungs sampled twice {doubled or 'none'}; block cells "
            f"that are not rungs {stray or 'none'}; anchor cells that are rungs without being listed "
            f"in RUNGS_THE_ANCHOR_ALREADY_SAMPLES {undeclared or 'none'}."
        )


_assert_the_ladder_is_covered_exactly_once()

DESIGN_LABELS: tuple[str, ...] = tuple(
    sorted({*ANCHOR_CELLS, *LADDER_RUNGS, *ALL_MODEL_IDS, *GAMES, "coupled", "decoupled"})
)
"""Every string that names the design to a reader, which is what the blind judge must never see.

One list rather than one per call site, because it is consumed twice: the blindness gate runs it over
the loaded rubric and the rendered headers before the first judge call, and the tests run it over
every prompt a scripted judge backend actually received. A cell id, a rung id, a model id or a game
id in a judge prompt lets the judge score the design instead of the reply.
"""

ON_DEMAND_LADDER_MODEL_IDS: tuple[str, ...] = (OPUS_MODEL_ID, LUNA_MODEL_ID, SONNET_MODEL_ID)
"""The rows whose ladder waits on their anchor's band read: the expensive one and the two live ones."""

MAX_TOKENS = DEFAULT_BEDROCK_MAX_TOKENS
"""One flat reply cap, the shared Converse default. Per-cell truncation rate is the tripwire."""

DECOUPLED_BATCH_PREFIX = "batch_jobs/sociology_decoupled_ladder"
"""This pass's own S3 prefix; a run namespaces under it, and a leg under the run."""


def _assert_roster_clears_the_cap() -> None:
    """Refuse at import if any batch row's own ceiling sits below this pass's flat reply cap.

    The failure this prevents has happened on this roster: a job whose ``maxTokens`` exceeds a
    model's ceiling reports ``Completed`` with every record errored, after burning the queue time to
    find out. ``BedrockBatchBackend`` refuses it at construction too, but that is per leg and after
    a plan has been built and priced; naming the offending row at import means the plan cannot be
    written down in a shape that could only come back dead.
    """
    below = [
        f"{model.model_id} (ceiling {model.max_tokens_limit})"
        for model in BATCH_ROSTER
        if model.model_id in BATCH_MODEL_IDS and model.max_tokens_limit < MAX_TOKENS
    ]
    if below:
        raise RuntimeError(
            f"these decoupled-ladder batch rows cap output below the flat cap of {MAX_TOKENS}: "
            f"{'; '.join(below)}. A job asking for more comes back Completed with every record "
            f"errored, so either drop the row or give this pass a per-model cap."
        )


_assert_roster_clears_the_cap()


def model_slug(model_id: str) -> str:
    """Spell a model id as a filesystem- and S3-safe stem, the way the batch backend spells it."""
    return model_id.replace(":", "-").replace(".", "-").replace("/", "-")


def effort_tag(reasoning_effort: str | None) -> str:
    """Name an effort for a key or a filename; the provider default is a value, not an absence."""
    return reasoning_effort or "default"


@dataclass(frozen=True, slots=True)
class Leg:
    """One row of the sampling plan: one model reading one block's cells at one effort and sitting.

    A leg is exactly one batch job or one live pass, which is what makes it the unit the CLI takes:
    the 100-record batch floor is per job, and pooling a block's cells into one job is what clears
    it.
    """

    model_id: str
    transport: str
    block: str
    cells: tuple[str, ...]
    reasoning_effort: str | None
    sitting: str
    on_demand: bool = False
    """Whether this leg waits on a recorded decision (``add-ladder``) rather than running by default.

    Two kinds of leg carry it: the ladders of :data:`ON_DEMAND_LADDER_MODEL_IDS`, which wait on their
    own model's anchor landing inside the readable band, and every leg of :data:`UNREADABLE_MODEL_IDS`,
    which the pass no longer reads at all. Neither counts toward the production totals.
    """

    @property
    def leg_id(self) -> str:
        """The name this leg is asked for by on the command line."""
        short = MODEL_SHORT_NAMES[self.model_id]
        return f"{short}--{effort_tag(self.reasoning_effort)}--{self.sitting}--{self.block}"

    @property
    def file_stem(self) -> str:
        """The reply and handle filename stem, carrying the FULL model id rather than a short name.

        The full slug on purpose: a filename is what an operator reads months later without the
        table in front of them, and a short name would need this module to decode.
        """
        return (
            f"{model_slug(self.model_id)}--effort-{effort_tag(self.reasoning_effort)}"
            f"--sitting-{self.sitting}--{self.block}"
        )

    @property
    def batch_run_id(self) -> str:
        """The run id the batch backend names its S3 subtree and its job after, per leg.

        Per leg rather than per run, and that is a correctness property rather than tidiness: the
        backend derives both the S3 base and the Bedrock job name from (prefix, run_id, model), so
        two legs of one model sharing a run id would overwrite each other's ``input.jsonl`` and ask
        the service for the same job name. Short, because the job name has a 63-character ceiling
        and the run id is the only part of it that is unique.
        """
        short = MODEL_SHORT_NAMES[self.model_id]
        return f"{short}-{effort_tag(self.reasoning_effort)}-{self.sitting}-{self.block}".lower()

    @property
    def records(self) -> int:
        """How many records this leg plans, from the design's arithmetic rather than a render."""
        return len(self.cells) * ROWS_PER_CELL * DRAWS


def rows_per_cell() -> int:
    """Count the prompt rows one cell renders: both games, both print orders, every frame and cell."""
    return sum(
        len(
            generate_counterpart_clause_prompt_rows(
                game_id,
                GRADING,
                clause=None,
                framing_label=FRAMING_UNSTATED,
                split=SPLIT_EVAL,
                label_print_order=order,
            )
        )
        for game_id in GAMES
        for order in LABEL_PRINT_ORDERS
    )


ROWS_PER_CELL = rows_per_cell()
"""48 = twin-pd's 16 rows plus public-goods' 8, under both print orders. Measured, not asserted."""


def all_legs() -> tuple[Leg, ...]:
    """Build the whole leg table in submit order, on-demand legs flagged so ``--leg`` still offers them.

    Anchor and floor for every roster row first, then the deliberation legs, then the ladders the
    pass runs by default, then the ladders that wait on a band read. A row in
    :data:`UNREADABLE_MODEL_IDS` keeps its anchor and floor legs -- their reply files are on disk and
    the plan has to describe them -- but both are on demand.
    """
    legs: list[Leg] = []
    for model_id in ALL_MODEL_IDS:
        unreadable = model_id in UNREADABLE_MODEL_IDS
        transport = transport_for(model_id)
        legs.append(
            Leg(model_id, transport, BLOCK_ANCHOR, ANCHOR_CELLS, None, SITTING_A, unreadable)
        )
        legs.append(Leg(model_id, transport, BLOCK_FLOOR, FLOOR_CELLS, None, SITTING_B, unreadable))
    legs.extend(
        Leg(
            model_id,
            transport_for(model_id),
            BLOCK_DELIBERATION,
            ANCHOR_CELLS,
            DELIBERATION_EFFORT,
            SITTING_A,
        )
        for model_id in DELIBERATION_MODEL_IDS
    )
    legs.extend(
        Leg(model_id, TRANSPORT_BATCH, BLOCK_LADDER, LADDER_BLOCK_CELLS, None, SITTING_A)
        for model_id in CHEAP_BATCH_MODEL_IDS
    )
    legs.extend(
        Leg(
            model_id,
            transport_for(model_id),
            BLOCK_LADDER,
            LADDER_BLOCK_CELLS,
            None,
            SITTING_A,
            on_demand=True,
        )
        for model_id in ON_DEMAND_LADDER_MODEL_IDS
    )
    return tuple(legs)


def production_legs() -> tuple[Leg, ...]:
    """Select the legs the pass runs and reads by default: every leg that is not on demand."""
    return tuple(leg for leg in all_legs() if not leg.on_demand)


def on_demand_legs() -> tuple[Leg, ...]:
    """Select the legs that wait on a recorded decision: three band-gated ladders plus every Opus 5 leg."""
    return tuple(leg for leg in all_legs() if leg.on_demand)


LEGS_BY_ID: dict[str, Leg] = {leg.leg_id: leg for leg in all_legs()}


def _assert_legs_well_formed() -> None:
    """Refuse a leg table with duplicate ids, or whose batch job names collide once truncated.

    The truncation check is the one worth spelling out. Bedrock caps a job name at
    :data:`~reward_hacking.bedrock_batch.MAX_JOB_NAME_CHARS`, and the backend truncates the MODEL
    end rather than the run-id end, so two long ids sharing a prefix would ask the service for one
    name and the second submit would fail (or worse, be read as the first job). Recomputed here from
    the same two rules the backend applies, because the alternative is finding out at submit time
    with a paid upload already in S3.
    """
    ids = [leg.leg_id for leg in all_legs()]
    duplicated = sorted({name for name in ids if ids.count(name) > 1})
    job_names: dict[str, str] = {}
    collisions: list[str] = []
    for leg in all_legs():
        if leg.transport != TRANSPORT_BATCH:
            continue
        suffix = f"-{leg.batch_run_id}"
        stem = f"jagged-{model_slug(leg.model_id)}".lower()
        name = stem[: MAX_JOB_NAME_CHARS - len(suffix)] + suffix
        if len(suffix) >= MAX_JOB_NAME_CHARS:
            collisions.append(f"{leg.leg_id} has a {len(suffix)}-character run-id suffix")
        if name in job_names:
            collisions.append(f"{leg.leg_id} and {job_names[name]} both truncate to {name!r}")
        job_names[name] = leg.leg_id
    if duplicated or collisions:
        raise RuntimeError(
            f"the decoupled-ladder leg table is malformed: duplicate leg ids "
            f"{duplicated or 'none'}; batch job-name problems {collisions or 'none'}."
        )


_assert_legs_well_formed()


def leg_for(leg_id: str) -> Leg:
    """Look up one leg by id, listing the table when the id is not in it."""
    leg = LEGS_BY_ID.get(leg_id)
    if leg is None:
        raise ValueError(
            f"{leg_id!r} is not a leg of this plan. Known legs: {', '.join(sorted(LEGS_BY_ID))}."
        )
    return leg


def record_key(  # noqa: PLR0913 - one argument per axis of the record's identity
    *,
    block: str,
    cell: str,
    game_id: str,
    prompt_id: str,
    model_id: str,
    reasoning_effort: str | None,
    sitting: str,
    draw: int,
) -> str:
    """One planned call's stable identity, derived from content and never from execution order.

    Effort and sitting ride in the key because neither ever pools: two efforts are two elicitations
    of the same prompt, and the two sittings ARE the floor measurement, so collapsing either would
    erase the thing being measured.
    """
    return (
        f"{block}|{cell}|{game_id}|{prompt_id}|{model_id}|effort={effort_tag(reasoning_effort)}"
        f"|sitting={sitting}|draw={draw}"
    )


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """One fully rendered call, carrying every label its reply record will store.

    Every field except ``prompt`` lands on the reply row, and ``prompt`` deliberately does not: the
    text is re-derivable from ``prompt_id`` and ``cell``, and a reply file that stored it would be a
    second copy of the authored clause prose on disk for no analytical gain.
    """

    key: str
    block: str
    cell: str
    game_id: str
    prompt_id: str
    reskin_id: str
    payoff_variant: str
    label_print_order: str
    coop_label: str
    label_a: str
    label_b: str
    coop_label_index: int
    model_id: str
    transport: str
    reasoning_effort: str | None
    sitting: str
    draw: int
    prompt: str

    def metadata(self) -> dict[str, Any]:
        """Return the sidecar labels the batch path hashes into the cell digest."""
        return {
            "key": self.key,
            "block": self.block,
            "cell": self.cell,
            "game_id": self.game_id,
            "prompt_id": self.prompt_id,
            "reskin_id": self.reskin_id,
            "payoff_variant": self.payoff_variant,
            "label_print_order": self.label_print_order,
            "coop_label": self.coop_label,
            "coop_label_index": self.coop_label_index,
            "reasoning_effort": self.reasoning_effort,
            "sitting": self.sitting,
            "draw": self.draw,
        }


def _coop_label_index(row: dict[str, Any]) -> int:
    """Read which of the row's two labels maps to the cooperative action."""
    if row["coop_label"] == row["label_a"]:
        return 0
    if row["coop_label"] == row["label_b"]:
        return 1
    raise ValueError(
        f"row {row['prompt_id']!r} has coop_label {row['coop_label']!r} matching neither label; "
        f"the renderer and this reader disagree about the row's own columns."
    )


def planned_calls_for_leg(leg: Leg, stimulus: DecoupledStimulus) -> list[PlannedCall]:
    """Render every planned call for one leg, in the fixed cell/game/order/row/draw order.

    The order is load-bearing on the batch path: submit and collect both call this, and the prompt
    and cell digests compare positionally, so a reordering here would be caught as a digest mismatch
    rather than silently attaching replies to the wrong cells.
    """
    calls: list[PlannedCall] = []
    for cell in leg.cells:
        clause = clause_for(cell, stimulus)
        for game_id in GAMES:
            for order in LABEL_PRINT_ORDERS:
                rows = generate_counterpart_clause_prompt_rows(
                    game_id,
                    GRADING,
                    clause=clause,
                    framing_label=cell,
                    split=SPLIT_EVAL,
                    label_print_order=order,
                )
                for row in rows:
                    calls.extend(
                        _planned_call(leg, cell=cell, row=row, draw=draw) for draw in range(DRAWS)
                    )
    if len(calls) != leg.records:
        raise RuntimeError(
            f"leg {leg.leg_id} rendered {len(calls)} calls but its table row says {leg.records}: "
            f"the roster the renderer walks and this plan's arithmetic disagree."
        )
    return calls


def _planned_call(leg: Leg, *, cell: str, row: dict[str, Any], draw: int) -> PlannedCall:
    """Assemble one planned call from a rendered prompt row."""
    prompt_id = str(row["prompt_id"])
    return PlannedCall(
        key=record_key(
            block=leg.block,
            cell=cell,
            game_id=str(row["game_id"]),
            prompt_id=prompt_id,
            model_id=leg.model_id,
            reasoning_effort=leg.reasoning_effort,
            sitting=leg.sitting,
            draw=draw,
        ),
        block=leg.block,
        cell=cell,
        game_id=str(row["game_id"]),
        prompt_id=prompt_id,
        reskin_id=str(row["reskin_id"]),
        payoff_variant=str(row["payoff_variant"]),
        label_print_order=str(row["label_print_order"]),
        coop_label=str(row["coop_label"]),
        label_a=str(row["label_a"]),
        label_b=str(row["label_b"]),
        coop_label_index=_coop_label_index(row),
        model_id=leg.model_id,
        transport=leg.transport,
        reasoning_effort=leg.reasoning_effort,
        sitting=leg.sitting,
        draw=draw,
        prompt=str(row["prompt"]),
    )


# Single hyphens inside the label only: a `--` starts the next prompt_id segment, and a class that
# swallowed it would eat the print-order and orientation segments too.
_FRAMING_SEGMENT_RE = re.compile(r"--framing-[a-z0-9]+(?:-[a-z0-9]+)*")


def unstated_stem_id(prompt_id: str) -> str:
    """Rewrite a cell's prompt_id into the `unstated` render's id for the same underlying row.

    One substitution, checked for a single match: the framing segment is what distinguishes a cell's
    row from the clause-free stem it must otherwise be byte-identical to, and a pattern that matched
    twice would rewrite the wrong one.
    """
    matches = _FRAMING_SEGMENT_RE.findall(prompt_id)
    if len(matches) != 1:
        raise ValueError(
            f"prompt_id {prompt_id!r} carries {len(matches)} framing segments, expected exactly "
            f"one; the audit rewrites that segment to find this row's clause-free stem."
        )
    return _FRAMING_SEGMENT_RE.sub(f"--framing-{FRAMING_UNSTATED}", prompt_id)


def unstated_stems() -> dict[str, str]:
    """Render every clause-free stem this pass compares against, keyed by its own prompt_id."""
    stems: dict[str, str] = {}
    for game_id in GAMES:
        for order in LABEL_PRINT_ORDERS:
            for row in generate_counterpart_clause_prompt_rows(
                game_id,
                GRADING,
                clause=None,
                framing_label=FRAMING_UNSTATED,
                split=SPLIT_EVAL,
                label_print_order=order,
            ):
                stems[str(row["prompt_id"])] = str(row["prompt"])
    return stems


def audit_prompt(*, rendered: str, stem: str, prompt_id: str) -> None:
    """Assert one rendered prompt is its clause-free stem plus exactly one counterpart paragraph.

    The whole design rests on this: a cell differs from the stem by one inserted paragraph, so a
    movement between cells is attributable to the clause and to nothing else. Deleting every section
    that opens with the counterpart marker must reproduce the stem byte for byte, which is why a
    two-paragraph clause fails here -- its second paragraph does not carry the marker, so it
    survives the deletion and the comparison goes red. Mirrors
    ``games.track_record_corpus._assert_one_inserted_paragraph``, which proves the same property for
    the training corpus.
    """
    marker = about_the_other_side("")
    sections = rendered.split("\n\n")
    counterpart = [section for section in sections if section.startswith(marker)]
    without_them = "\n\n".join(section for section in sections if not section.startswith(marker))
    if len(counterpart) != 1 or without_them != stem:
        raise ValueError(
            f"{prompt_id!r} is not its clause-free stem plus one counterpart paragraph "
            f"({len(counterpart)} such paragraphs found): deleting them does not reproduce the "
            f"stem, so something other than the counterpart clause moved."
        )


def audit_planned_calls(calls: Sequence[PlannedCall]) -> int:
    """Run the one-inserted-paragraph audit over every planned prompt; return how many passed."""
    stems = unstated_stems()
    for call in calls:
        stem_id = unstated_stem_id(call.prompt_id)
        if stem_id not in stems:
            raise ValueError(
                f"{call.prompt_id!r} has no clause-free stem at {stem_id!r}; the audit has nothing "
                f"to compare this cell's render against."
            )
        audit_prompt(rendered=call.prompt, stem=stems[stem_id], prompt_id=call.prompt_id)
    return len(calls)


def refuse_below_batch_floor(leg: Leg, n_records: int) -> None:
    """Refuse a batch leg under the service's record floor, in the levers THIS CLI has.

    ``BedrockBatchBackend.submit`` refuses too, but its message offers a ``--repeats`` multiplier
    this CLI does not define, and it refuses only after a model-availability call and an upload. The
    floor is per job and not adjustable, so pooling a block's cells into one job is what clears it.
    """
    if n_records >= MIN_BATCH_RECORDS:
        return
    raise ValueError(
        f"batch inference needs at least {MIN_BATCH_RECORDS} records per job and leg "
        f"{leg.leg_id} renders {n_records} ({len(leg.cells)} cells x {ROWS_PER_CELL} rows x "
        f"{DRAWS} draws). The floor is per job and not adjustable, so either raise DRAWS, pool "
        f"more cells into the block, or run this leg on live Converse, which has no floor."
    )


def batch_price_per_mtok(model_id: str) -> tuple[float, float]:
    """Return one batch model's verified (input, output) price per million tokens."""
    roster = roster_model(model_id)
    return roster.batch_price_in_per_mtok, roster.batch_price_out_per_mtok
