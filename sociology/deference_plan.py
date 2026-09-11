"""The deference sampling plan: roster, arms, cells, blocks, legs, record keys, and the prompt audits.

What runs is a fixed table of legs (:data:`LEGS`), one row per (model, block, sitting), rather than
something assembled per invocation -- so the record counts are arithmetic anybody can check against the
design, and a leg id names one job's worth of work on the command line.

The design is a 2x4 plus a floor. Both arms sample the same four cells: no peers at all, peers present
but silent, peers present and reported to have chosen the departing option, and the same report from peers
of a different family. The arms differ in one sentence of the brief -- whether its one procedural
constraint governs the very dimension the fork decides, or an unrelated one -- so the headline
(same-checkpoint report minus different-family report) is read per arm and the two are read against each
other as a double difference. The floor block re-sits the two report cells of the breaking arm as a second
sitting, because resampling noise on a rate is largest mid-range and a floor taken on a near-zero cell
would clear anything.

Two things about this module's boundaries. It knows the *stimulus* only through
:mod:`sociology.deference_stimulus`, which owns the renderer and every vocabulary gate, so a cell here is
a pair of labels and never a piece of prose. And its batch legs carry their own pass token
(:data:`BATCH_PASS_TOKEN`), because a Bedrock job name is account-wide and permanent and this pass's block
ids (``benign``, ``breaking``, ``floor``) collide with the sibling passes' by design -- the token is what
keeps every name it asks for distinct from a name they already own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from games.prompts import (
    LABEL_PRINT_ORDERS,
    assert_counterpart_paragraph_is_the_only_insertion,
)
from reward_hacking.bedrock_batch import (
    BATCH_ROSTER,
    MAX_JOB_NAME_CHARS,
    MIN_BATCH_RECORDS,
    roster_model,
)
from reward_hacking.model_backend import DEFAULT_BEDROCK_MAX_TOKENS
from sociology.decoupled_stimulus import RUNG_DIFFERENT_FAMILY, RUNG_SAME_CHECKPOINT
from sociology.deference_stimulus import (
    ARM_BENIGN,
    ARM_BREAKING,
    ARMS,
    CELL_DIFFERENT_FAMILY_CHOSE,
    CELL_SAME_CHECKPOINT_CHOSE,
    CELLS,
    CELLS_WITH_STATUS_REPORT,
    DEFERENCE_PARAGRAPH_MARKER,
    N_SCENARIOS,
    PEER_COUNT,
    PEER_HANDLE_PREFIX,
    STATUS_LINE_TEMPLATE,
    brief_section_count,
    identity_for_cell,
    labels_in_print_order,
    render_deference_sections,
    status_report_block,
    status_report_lines,
    status_report_section_index,
)
from sociology.refusal_canary import needs_refusal_canary

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sociology.deference_stimulus import DeferenceScenario, DeferenceStimulus

logger = logging.getLogger(__name__)

SONNET_MODEL_ID = "global.anthropic.claude-sonnet-5"
LUNA_MODEL_ID = "global.openai.gpt-5.6-luna"
QWEN_MODEL_ID = "qwen.qwen3-235b-a22b-2507-v1:0"

LIVE_MODEL_IDS: tuple[str, ...] = (SONNET_MODEL_ID, LUNA_MODEL_ID)
BATCH_MODEL_IDS: tuple[str, ...] = (QWEN_MODEL_ID,)
ALL_MODEL_IDS: tuple[str, ...] = (*LIVE_MODEL_IDS, *BATCH_MODEL_IDS)
"""The whole roster, fixed by the owner: two live rows and one batch row.

Sonnet and qwen carried the large identity effects in both previous passes, and Luna is here as a
behaviour-only row: its reasoning comes back provider-encrypted, so every trace-read judge label on it has
a zero denominator, and what it still answers is whether a report from copies moves the ACTION.
"""

MODEL_SHORT_NAMES: dict[str, str] = {
    SONNET_MODEL_ID: "sonnet-5",
    LUNA_MODEL_ID: "luna",
    QWEN_MODEL_ID: "qwen3-235b",
}
"""Short names for leg ids and batch job names, which have a 63-character ceiling to live inside."""

TRANSPORT_BATCH = "batch"
TRANSPORT_LIVE = "live"

BLOCK_BENIGN = ARM_BENIGN
BLOCK_BREAKING = ARM_BREAKING
BLOCK_FLOOR = "floor"
BLOCKS: tuple[str, ...] = (BLOCK_BENIGN, BLOCK_BREAKING, BLOCK_FLOOR)
"""The three blocks. Two of them are named for the arm they sample, because a block IS an arm here; the
third re-sits two of the breaking arm's cells as the bar every contrast has to clear."""

READING_BLOCKS: tuple[str, ...] = (BLOCK_BENIGN, BLOCK_BREAKING)
"""The two blocks the readings are taken off; the third is their second sitting."""

RESAT_BLOCKS: tuple[str, ...] = (BLOCK_FLOOR,)

ARM_BY_BLOCK: dict[str, str] = {
    BLOCK_BENIGN: ARM_BENIGN,
    BLOCK_BREAKING: ARM_BREAKING,
    BLOCK_FLOOR: ARM_BREAKING,
}
"""Which arm each block renders. The floor sits in the breaking arm because that is the arm the headline
is about, and a floor measured in the other arm would be a bar for a different quantity."""

FLOOR_CELLS: tuple[str, ...] = (CELL_SAME_CHECKPOINT_CHOSE, CELL_DIFFERENT_FAMILY_CHOSE)
"""The two cells the floor re-sits: the pair the headline is the difference of."""

SITTING_A = "A"
SITTING_B = "B"
SITTINGS: tuple[str, ...] = (SITTING_A, SITTING_B)

DRAWS = 8
"""Draws per prompt row. Every draw is a separate record key, never a pooled resample."""

ROWS_PER_CELL = N_SCENARIOS * len(LABEL_PRINT_ORDERS)
"""16 = eight authored briefs under both print orders. Derived, so a brief added moves the counts."""

MAX_TOKENS = DEFAULT_BEDROCK_MAX_TOKENS
"""One flat reply cap, the shared Converse default. Per-cell truncation rate is the tripwire."""

DEFERENCE_BATCH_PREFIX = "batch_jobs/sociology_deference"
"""This pass's own S3 prefix; a run namespaces under it, and a leg under the run. A prefix per pass
rather than a shared one, because a cross-prefix copy of a run id is a silent replace: two passes
namespacing under one prefix would overwrite each other's ``input.jsonl`` the first time a run directory
name repeated."""

BATCH_PASS_TOKEN = "dfr"
"""The token every batch run id of this pass opens with, so no job name collides with another pass's.

Bedrock job names are account-wide and a Completed job keeps its name, so a run id spelling only (model,
effort, sitting, block) collides with any earlier pass that used the same block ids -- which is what
happened on 2026-09-02, when the transfer pass's ``floor`` legs asked for names the decoupled-ladder pass
already owned and eight submits were refused with ``ConflictException``. This pass's ``floor`` block is a
third claim on that id, so it carries a token of its own rather than the transfer pass's.
"""

DEFAULT_RUN_DIR = "artifacts/swarm_sociology/deference/run-20260904"
"""Where this pass's artifacts land by default; a run directory per pass, never a shared one."""


def transport_for(model_id: str) -> str:
    """Name the transport one roster row runs on: live for the live roster, batch for everything else."""
    return TRANSPORT_LIVE if model_id in LIVE_MODEL_IDS else TRANSPORT_BATCH


def model_slug(model_id: str) -> str:
    """Spell a model id as a filesystem- and S3-safe stem, the way the batch backend spells it."""
    return model_id.replace(":", "-").replace(".", "-").replace("/", "-")


def effort_tag(reasoning_effort: str | None) -> str:
    """Name an effort for a key or a filename; the provider default is a value, not an absence."""
    return reasoning_effort or "default"


@dataclass(frozen=True, slots=True)
class Cell:
    """One sampled condition: which arm's brief, and which of the four peer conditions."""

    arm: str
    cell_id: str


def cells_for(arm: str, cell_ids: Sequence[str]) -> tuple[Cell, ...]:
    """Build one arm's named cells in the order given, which is the order their table reads in."""
    if arm not in ARMS:
        raise ValueError(f"{arm!r} is not an arm of this design; the arms are {list(ARMS)}.")
    unknown = sorted(set(cell_ids) - set(CELLS))
    if unknown:
        raise ValueError(f"{unknown} are not cells of this design; the cells are {list(CELLS)}.")
    return tuple(Cell(arm, cell_id) for cell_id in cell_ids)


CELLS_BY_BLOCK: dict[str, tuple[Cell, ...]] = {
    BLOCK_BENIGN: cells_for(ARM_BENIGN, CELLS),
    BLOCK_BREAKING: cells_for(ARM_BREAKING, CELLS),
    BLOCK_FLOOR: cells_for(ARM_BREAKING, FLOOR_CELLS),
}

DESIGN_LABELS: tuple[str, ...] = tuple(
    sorted(
        {
            *CELLS,
            *ARMS,
            *BLOCKS,
            *ALL_MODEL_IDS,
            RUNG_SAME_CHECKPOINT,
            RUNG_DIFFERENT_FAMILY,
            "deference",
            "copies",
            "copy of",
        }
    )
)
"""Every string that names this design to a reader, which is what the blind judge must never see.

One list rather than one per call site, because it is consumed twice: the blindness gate runs it over both
loaded rubrics and every fixed header before the first judge call, and the tests run it over every prompt
a scripted judge backend actually received. ``copies`` and ``copy of`` are here because the ``mirror``
flag's rate inside the same-checkpoint cells is exactly the check that says whether the identity landed,
and a rubric that could say "copies" would hand the judge the cell label in the one word that matters.
"""


@dataclass(frozen=True, slots=True)
class Leg:
    """One row of the sampling plan: one model reading one block's cells at one sitting.

    A leg is exactly one batch job or one live pass, which is what makes it the unit the CLI takes: the
    100-record batch floor is per job, and pooling a block's cells into one job is what clears it.
    """

    model_id: str
    transport: str
    block: str
    cells: tuple[Cell, ...]
    reasoning_effort: str | None
    sitting: str
    batch_prefix: str = DEFERENCE_BATCH_PREFIX
    """The S3 prefix this leg's batch inputs and outputs live under: its own pass's, never a shared one.

    On the leg rather than looked up per invocation, because the prefix is what keeps two passes' runs from
    writing over each other, and a submit that resolved it from anything but the leg could resolve it
    differently from the collect that follows. Defaulted to this pass's, so the sibling coordination pass
    states its own and no leg can be built without one.
    """
    batch_pass_token: str = BATCH_PASS_TOKEN
    """The token this leg's batch run id opens with, so its job name cannot be one another pass owns."""

    @property
    def arm(self) -> str:
        """Which arm's brief this leg renders: the one arm every cell of it carries.

        Read off the cells rather than out of a block table, so the same class serves the sibling
        coordination pass (whose blocks and arms are its own) and so a leg whose cells came from two arms
        is a refusal rather than a leg silently labelled with one of them.
        """
        arms = {cell.arm for cell in self.cells}
        if len(arms) != 1:
            raise ValueError(
                f"leg {self.leg_id} pools cells from the arms {sorted(arms)}, so it has no one arm: "
                f"every reading is taken per arm, and a leg spanning two would pool them under one label."
            )
        return arms.pop()

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

        Per leg rather than per run, and that is a correctness property: the backend derives both the S3
        base and the Bedrock job name from (prefix, run_id, model), so two legs of one model sharing a run
        id would overwrite each other's ``input.jsonl`` and ask the service for the same job name. Opened
        with :data:`BATCH_PASS_TOKEN` on every leg, because job names are account-wide and this pass's
        block ids are ones the sibling passes already used.
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


def _legs() -> tuple[Leg, ...]:
    """Build the leg table in submit order: every block for every roster row, the floor at sitting B."""
    return tuple(
        Leg(
            model_id,
            transport_for(model_id),
            block,
            CELLS_BY_BLOCK[block],
            None,
            SITTING_B if block == BLOCK_FLOOR else SITTING_A,
        )
        for model_id in ALL_MODEL_IDS
        for block in BLOCKS
    )


LEGS: tuple[Leg, ...] = _legs()

LEGS_BY_ID: dict[str, Leg] = {leg.leg_id: leg for leg in LEGS}


def production_legs() -> tuple[Leg, ...]:
    """Every leg this pass runs and reads. No leg here waits on a recorded decision."""
    return LEGS


def leg_for(leg_id: str) -> Leg:
    """Look up one leg by id, listing the table when the id is not one of them."""
    leg = LEGS_BY_ID.get(leg_id)
    if leg is None:
        raise ValueError(
            f"{leg_id!r} is not a leg of this plan. Known legs: {', '.join(sorted(LEGS_BY_ID))}."
        )
    return leg


def assert_cells_well_formed(  # noqa: PLR0913 - one keyword per part of the table being checked
    cells_by_block: Mapping[str, tuple[Cell, ...]],
    *,
    cells: Sequence[str],
    reading_blocks: Sequence[str],
    resat_blocks: Sequence[str],
    arm_by_block: Mapping[str, str],
    what: str,
) -> None:
    """Refuse a block table that drops a cell, adds one, re-sits an unsampled one, or misplaces an arm.

    The split between blocks is arithmetic nobody re-derives while operating, so every way of getting it
    wrong is checked. Each of them leaves the per-leg record counts looking exactly right.

    Parameterised over the table rather than reading this module's own, because the sibling coordination
    pass has the same shape -- reading blocks that between them sample every cell once, plus re-sat blocks
    that must re-sit only cells a reading block bought -- and two copies of this arithmetic are two things
    to keep in step.
    """
    problems: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    for block in reading_blocks:
        planned = {cell.cell_id for cell in cells_by_block[block]}
        if planned != set(cells):
            problems.append(
                f"{block} plans {sorted(planned - set(cells)) or 'nothing extra'} that this design "
                f"does not register and misses {sorted(set(cells) - planned) or 'nothing'}"
            )
        for cell in cells_by_block[block]:
            key = (cell.arm, cell.cell_id)
            if key in seen:
                problems.append(f"{key} is planned by both {seen[key]} and {block}")
            seen[key] = block
    problems.extend(
        f"{block} re-sits {(cell.arm, cell.cell_id)}, which no reading block samples, so its delta "
        f"would be a floor for a cell that has no first sitting"
        for block in resat_blocks
        for cell in cells_by_block[block]
        if (cell.arm, cell.cell_id) not in seen
    )
    problems.extend(
        f"{block} renders the arms {sorted({cell.arm for cell in cells_by_block[block]})} and the table "
        f"says it renders {arm_by_block[block]!r}"
        for block in cells_by_block
        if {cell.arm for cell in cells_by_block[block]} != {arm_by_block[block]}
    )
    if problems:
        raise RuntimeError(
            f"the {what} block table is malformed: {'; '.join(problems)}. Every cell is read against the "
            f"baseline cell of its own arm, and a cell paid for twice under two block labels would be "
            f"pooled by one report and split by another."
        )


def assert_legs_well_formed(legs: Sequence[Leg], *, what: str) -> None:
    """Refuse a leg table with duplicate ids, a thin batch block, or job names that collide truncated.

    The truncation check is the one worth spelling out. Bedrock caps a job name at
    :data:`~reward_hacking.bedrock_batch.MAX_JOB_NAME_CHARS` and the backend truncates the MODEL end
    rather than the run-id end, so two long ids sharing a prefix would ask the service for one name and
    the second submit would fail -- or worse, be read as the first job. Recomputed here from the same two
    rules the backend applies, because the alternative is finding out at submit time with a paid upload
    already in S3.
    """
    ids = [leg.leg_id for leg in legs]
    duplicated = sorted({name for name in ids if ids.count(name) > 1})
    job_names: dict[str, str] = {}
    problems: list[str] = []
    for leg in legs:
        name = leg.batch_job_name
        if name is None:
            continue
        if leg.records < MIN_BATCH_RECORDS:
            problems.append(
                f"{leg.leg_id} plans {leg.records} records, below the {MIN_BATCH_RECORDS}-record "
                f"per-job batch floor"
            )
        if len(leg.batch_run_id) + 1 >= MAX_JOB_NAME_CHARS:
            problems.append(
                f"{leg.leg_id} has a {len(leg.batch_run_id) + 1}-character run-id suffix"
            )
        if not leg.batch_run_id.startswith(f"{leg.batch_pass_token}-"):
            problems.append(f"{leg.leg_id} run id {leg.batch_run_id!r} lacks the pass token")
        if name in job_names:
            problems.append(f"{leg.leg_id} and {job_names[name]} both truncate to {name!r}")
        job_names[name] = leg.leg_id
    if duplicated or problems:
        raise RuntimeError(
            f"the {what} leg table is malformed: duplicate leg ids {duplicated or 'none'}; batch "
            f"job-name problems {problems or 'none'}."
        )


def assert_no_batch_leg_needs_the_refusal_canary(legs: Sequence[Leg], *, what: str) -> None:
    """Refuse a batch leg whose vendor's classifier can refuse a job wholesale, unguarded.

    :mod:`sociology.refusal_canary` fronts the Claude batch submits of the sibling passes, and neither
    pass on this CLI has a Claude batch leg to front: their only batch row is qwen, and the live rows
    refuse per record as a ``content_filtered`` the scans already bucket. So the canary is not wired into
    this CLI -- and a Claude batch row added to either roster later would need it, which is what this
    refuses. Stated as a gate rather than as a comment, because the failure it prevents is a paid job that
    comes back wholly refused.
    """
    batch_rows = {leg.model_id for leg in legs if leg.transport == TRANSPORT_BATCH}
    unguarded = sorted(model_id for model_id in batch_rows if needs_refusal_canary(model_id))
    if unguarded:
        raise RuntimeError(
            f"these {what} batch rows need the live refusal canary and this pass does not run one: "
            f"{unguarded}. A Claude batch job whose stimulus the classifier refuses comes back "
            f"Completed with every record refused, after the queue time it took to find out. Wire "
            f"sociology.refusal_canary into deference_cli.submit_batch (the transfer CLI's "
            f"`_refusal_canary` is the shape) before sampling this row."
        )


def assert_roster_clears_the_cap(batch_model_ids: Sequence[str], *, what: str) -> None:
    """Refuse at import if a batch row's own ceiling sits below the flat reply cap both passes use.

    The failure this prevents has happened on this roster: a job whose ``maxTokens`` exceeds a model's
    ceiling reports ``Completed`` with every record errored, after burning the queue time to find out.
    """
    below = [
        f"{model.model_id} (ceiling {model.max_tokens_limit})"
        for model in BATCH_ROSTER
        if model.model_id in set(batch_model_ids) and model.max_tokens_limit < MAX_TOKENS
    ]
    if below:
        raise RuntimeError(
            f"these {what} batch rows cap output below the flat cap of {MAX_TOKENS}: "
            f"{'; '.join(below)}. A job asking for more comes back Completed with every record errored, "
            f"so either drop the row or give the pass a per-model cap."
        )


PASS_ID = "deference"
"""What ``--pass deference`` selects on the shared CLI, and the word this pass's prompt ids open with."""

assert_cells_well_formed(
    CELLS_BY_BLOCK,
    cells=CELLS,
    reading_blocks=READING_BLOCKS,
    resat_blocks=RESAT_BLOCKS,
    arm_by_block=ARM_BY_BLOCK,
    what=PASS_ID,
)
assert_legs_well_formed(LEGS, what=PASS_ID)
assert_no_batch_leg_needs_the_refusal_canary(LEGS, what=PASS_ID)
assert_roster_clears_the_cap(BATCH_MODEL_IDS, what=PASS_ID)


def record_key(  # noqa: PLR0913 - one argument per axis of the record's identity
    *,
    block: str,
    arm: str,
    cell: str,
    scenario_id: str,
    print_order: str,
    model_id: str,
    reasoning_effort: str | None,
    sitting: str,
    draw: int,
) -> str:
    """One planned call's stable identity, derived from content and never from execution order.

    The sitting rides in the key because the two sittings ARE the floor measurement, so collapsing them
    would erase the thing being measured; so does the print order, which is this pass's counterbalance.
    The arm rides in it beside the block even though the block decides it, because a reader joining on
    the arm should not have to know the block table to do it.
    """
    return (
        f"{block}|{arm}|{cell}|{scenario_id}|{print_order}|{model_id}"
        f"|effort={effort_tag(reasoning_effort)}|sitting={sitting}|draw={draw}"
    )


def prompt_id(*, arm: str, cell: str, scenario_id: str, print_order: str) -> str:
    """Name one rendered prompt: everything about the text, and nothing about who reads it.

    Distinct from the record key on purpose: two models reading one prompt share a prompt id and never a
    record key, which is what lets the audits deduplicate renders and the reports join on the text.
    """
    return f"{PASS_ID}--{arm}--{cell}--{scenario_id}--order-{print_order}"


CARRIED_LABEL_FIELDS: tuple[str, ...] = (
    "model_id",
    "block",
    "arm",
    "cell",
    "scenario_id",
    "print_order",
    "label_a",
    "label_b",
    "compliant_label",
    "deviation_label",
    "prompt_id",
    "reasoning_effort",
    "sitting",
    "draw",
)
"""Every label a reply row stores beside its texts, spelled once for all three instruments.

One tuple rather than one per module, because the scans, the judge and the intent check each copy these
onto their own rows and a reader joins the three on them: a field carried by one instrument and not
another is a join that silently drops rows.
"""


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """One fully rendered call, carrying every label its reply record will store.

    Every field except ``prompt`` lands on the reply row, and ``prompt`` deliberately does not: the text
    is re-derivable from the labels plus the stimulus file, and a reply file that stored it would be a
    second copy of the authored briefs on disk for no analytical gain.
    """

    key: str
    block: str
    arm: str
    cell: str
    scenario_id: str
    print_order: str
    label_a: str
    label_b: str
    compliant_label: str
    deviation_label: str
    prompt_id: str
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
            "arm": self.arm,
            "cell": self.cell,
            "scenario_id": self.scenario_id,
            "print_order": self.print_order,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "compliant_label": self.compliant_label,
            "deviation_label": self.deviation_label,
            "prompt_id": self.prompt_id,
            "reasoning_effort": self.reasoning_effort,
            "sitting": self.sitting,
            "draw": self.draw,
        }


def render_cell_prompt(
    cell: Cell,
    scenario: DeferenceScenario,
    stimulus: DeferenceStimulus,
    *,
    print_order: str,
) -> str:
    """Render one (cell, scenario, print order) into the prompt text the model reads."""
    return "\n\n".join(
        render_deference_sections(
            scenario,
            arm=cell.arm,
            identity=identity_for_cell(cell.cell_id),
            with_status_report=cell.cell_id in CELLS_WITH_STATUS_REPORT,
            print_order=print_order,
            stimulus=stimulus,
        )
    )


def planned_calls_for_leg(leg: Leg, stimulus: DeferenceStimulus) -> list[PlannedCall]:
    """Render every planned call for one leg, in the fixed cell/scenario/print-order/draw order.

    The order is load-bearing on the batch path: submit and collect both call this, and the prompt and
    cell digests compare positionally, so a reordering here would be caught as a digest mismatch rather
    than silently attaching replies to the wrong cells.
    """
    calls: list[PlannedCall] = []
    for cell in leg.cells:
        for scenario in stimulus.scenarios:
            for print_order in LABEL_PRINT_ORDERS:
                prompt = render_cell_prompt(cell, scenario, stimulus, print_order=print_order)
                calls.extend(
                    _planned_call(
                        leg,
                        cell=cell,
                        scenario=scenario,
                        print_order=print_order,
                        prompt=prompt,
                        draw=draw,
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
    scenario: DeferenceScenario,
    print_order: str,
    prompt: str,
    draw: int,
) -> PlannedCall:
    """Assemble one planned call from a rendered prompt."""
    label_a, label_b = labels_in_print_order(scenario, print_order)
    return PlannedCall(
        key=record_key(
            block=leg.block,
            arm=cell.arm,
            cell=cell.cell_id,
            scenario_id=scenario.scenario_id,
            print_order=print_order,
            model_id=leg.model_id,
            reasoning_effort=leg.reasoning_effort,
            sitting=leg.sitting,
            draw=draw,
        ),
        block=leg.block,
        arm=cell.arm,
        cell=cell.cell_id,
        scenario_id=scenario.scenario_id,
        print_order=print_order,
        label_a=label_a,
        label_b=label_b,
        compliant_label=scenario.compliant_label,
        deviation_label=scenario.deviation_label,
        prompt_id=prompt_id(
            arm=cell.arm,
            cell=cell.cell_id,
            scenario_id=scenario.scenario_id,
            print_order=print_order,
        ),
        model_id=leg.model_id,
        transport=leg.transport,
        reasoning_effort=leg.reasoning_effort,
        sitting=leg.sitting,
        draw=draw,
        prompt=prompt,
    )


def peer_free_stem(
    scenario: DeferenceScenario, stimulus: DeferenceStimulus, *, arm: str, print_order: str
) -> str:
    """Render the peer-free prompt of one (scenario, arm, print order): every cell's audit reference.

    It is the ``no-peers`` cell's own render, which is what makes the baseline every reading is a contrast
    against and the text the audit compares every other cell to one render rather than two that agree.
    """
    return "\n\n".join(
        render_deference_sections(
            scenario,
            arm=arm,
            identity=None,
            with_status_report=False,
            print_order=print_order,
            stimulus=stimulus,
        )
    )


def _without_the_status_report(
    call: PlannedCall, scenario: DeferenceScenario, stimulus: DeferenceStimulus
) -> str:
    """Delete this call's status-report section, checking it is the block this scenario would render.

    Positional rather than by search, because the property being proved is that the report sits at one
    fixed place -- after the brief, the identity paragraph and the own log -- and a text search would
    find a block wherever it had drifted to and report the render as sound.
    """
    identity = identity_for_cell(call.cell)
    if call.cell not in CELLS_WITH_STATUS_REPORT:
        return call.prompt
    sections = call.prompt.split("\n\n")
    index = status_report_section_index(scenario, identity=identity)
    expected = status_report_block(scenario, stimulus)
    if index >= len(sections) or sections[index] != expected:
        raise ValueError(
            f"{call.prompt_id!r} does not carry its status report at section {index}, where the brief, "
            f"the identity paragraph and the own log put it: section {index} is "
            f"{sections[index] if index < len(sections) else '<past the end>'!r}. The report's position "
            f"is what makes the report contrast a read of one inserted block."
        )
    return "\n\n".join(sections[:index] + sections[index + 1 :])


def audit_planned_calls(calls: Sequence[PlannedCall], stimulus: DeferenceStimulus) -> int:
    """Run the one-inserted-paragraph audit over every planned prompt; return how many passed.

    The whole design rests on this: with the status report set aside, a cell differs from the peer-free
    render of its own (scenario, arm, print order) by exactly one inserted paragraph, so a movement
    between cells is attributable to that paragraph and to nothing else. The peer-free cell IS the stem,
    so what there is to check there is that it inserts nothing at all -- which is also the check that the
    baseline and the audit's reference are one render rather than two that happen to agree today.
    """
    for call in calls:
        scenario = stimulus.scenario(call.scenario_id)
        stem = peer_free_stem(scenario, stimulus, arm=call.arm, print_order=call.print_order)
        without_report = _without_the_status_report(call, scenario, stimulus)
        if identity_for_cell(call.cell) is None:
            if without_report != stem:
                raise ValueError(
                    f"{call.prompt_id!r} carries no identity paragraph and does not equal the peer-free "
                    f"render of its own scenario, arm and print order, so the baseline every reading is "
                    f"a contrast against is not the text the audit compares every other cell to."
                )
            continue
        assert_counterpart_paragraph_is_the_only_insertion(
            stem=stem,
            rendered=without_report,
            prompt_id=call.prompt_id,
            marker=DEFERENCE_PARAGRAPH_MARKER,
        )
    return len(calls)


def audit_the_report_is_one_added_section(stimulus: DeferenceStimulus) -> int:
    """Assert every report render is its own identity's silent render plus one block; return the pairs.

    The report effect is reported as one added tool-result block, so it has to BE one: for each reported
    identity, the render with the report and the render without it must agree section for section apart
    from that one insertion, at the position after the own log. The different-family pair's silent render
    is not a cell this pass samples -- rendering it costs nothing and is the only matched stem the
    different-family report has, so the contrast is checked against it rather than against another
    identity's.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        for arm in ARMS:
            for cell in sorted(CELLS_WITH_STATUS_REPORT):
                identity = identity_for_cell(cell)
                for print_order in LABEL_PRINT_ORDERS:
                    reported = render_deference_sections(
                        scenario,
                        arm=arm,
                        identity=identity,
                        with_status_report=True,
                        print_order=print_order,
                        stimulus=stimulus,
                    )
                    silent = render_deference_sections(
                        scenario,
                        arm=arm,
                        identity=identity,
                        with_status_report=False,
                        print_order=print_order,
                        stimulus=stimulus,
                    )
                    index = status_report_section_index(scenario, identity=identity)
                    rest = (*reported[:index], *reported[index + 1 :])
                    if reported[index] != status_report_block(scenario, stimulus) or rest != silent:
                        raise ValueError(
                            f"the report render of {scenario.scenario_id!r}/{arm!r}/{cell!r}/"
                            f"{print_order!r} is not its silent render plus the status report at "
                            f"section {index}: it has {len(reported)} sections against "
                            f"{len(silent)}, and deleting section {index} leaves text the silent "
                            f"render does not have. The report effect is reported as one added block."
                        )
                    compared += 1
    return compared


def audit_arms_differ_in_the_constraint_sentence(stimulus: DeferenceStimulus) -> int:
    """Assert the two arms' renders differ in one brief section, and in the constraint sentence in it.

    The headline double difference is between the arms, so anything else that moved between them would
    sit inside it. Checked as a substitution rather than as a diff: the benign render's brief with the
    benign sentence replaced by the breaking one must BE the breaking render's brief, and the sentence
    must occur exactly once, so a brief mentioning its own constraint twice cannot pass by luck.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        brief_sections = brief_section_count(scenario)
        for cell in CELLS:
            for print_order in LABEL_PRINT_ORDERS:
                rendered = {
                    arm: render_deference_sections(
                        scenario,
                        arm=arm,
                        identity=identity_for_cell(cell),
                        with_status_report=cell in CELLS_WITH_STATUS_REPORT,
                        print_order=print_order,
                        stimulus=stimulus,
                    )
                    for arm in ARMS
                }
                _assert_one_arm_pair_differs_by_the_constraint(
                    rendered[ARM_BENIGN],
                    rendered[ARM_BREAKING],
                    scenario=scenario,
                    brief_sections=brief_sections,
                    where=f"{scenario.scenario_id!r}/{cell!r}/{print_order!r}",
                )
                compared += 1
    return compared


def _assert_one_arm_pair_differs_by_the_constraint(
    benign: Sequence[str],
    breaking: Sequence[str],
    *,
    scenario: DeferenceScenario,
    brief_sections: int,
    where: str,
) -> None:
    """Compare one pair of arm renders section by section, naming the scenario in any refusal."""
    if len(benign) != len(breaking):
        raise ValueError(
            f"the two arms render {len(benign)} and {len(breaking)} sections for {where}, so they "
            f"cannot be compared by position at all."
        )
    differing = [
        index
        for index, (mine, theirs) in enumerate(zip(benign, breaking, strict=True))
        if mine != theirs
    ]
    if len(differing) != 1 or differing[0] >= brief_sections:
        raise ValueError(
            f"the two arms' prompts for {where} differ in sections {differing}, expected exactly one "
            f"section inside the brief's first {brief_sections}. Everything outside the brief -- the "
            f"identity paragraph, the log, the status report, the fork and the instruction -- is shared "
            f"between the arms, and anything else that moved would sit inside the headline double "
            f"difference."
        )
    section = benign[differing[0]]
    if section.count(scenario.constraint_benign) != 1:
        raise ValueError(
            f"the differing brief section of {where} carries its own benign constraint sentence "
            f"{section.count(scenario.constraint_benign)} times, so the arm difference cannot be read "
            f"as that one sentence."
        )
    substituted = section.replace(scenario.constraint_benign, scenario.constraint_breaking)
    if substituted != breaking[differing[0]]:
        raise ValueError(
            f"the two arms' briefs for {where} differ in more than the constraint sentence: "
            f"substituting one for the other gives {substituted!r}, not {breaking[differing[0]]!r}."
        )


def audit_print_orders_differ_in_the_fork_and_instruction(stimulus: DeferenceStimulus) -> int:
    """Assert the counterbalance moves the page alone: the last two sections, and nothing else.

    The fork prints the two options and the instruction names them in the tags it asks for, so those are
    the two sections the order may move. If it moved anything else -- the brief, the identity paragraph,
    the report -- the counterbalance would be a second manipulation and its halves could not be pooled.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        for arm in ARMS:
            for cell in CELLS:
                rendered = {
                    print_order: render_deference_sections(
                        scenario,
                        arm=arm,
                        identity=identity_for_cell(cell),
                        with_status_report=cell in CELLS_WITH_STATUS_REPORT,
                        print_order=print_order,
                        stimulus=stimulus,
                    )
                    for print_order in LABEL_PRINT_ORDERS
                }
                first, second = (rendered[order] for order in LABEL_PRINT_ORDERS)
                differing = [
                    index
                    for index, (mine, theirs) in enumerate(zip(first, second, strict=True))
                    if mine != theirs
                ]
                expected = [len(first) - 2, len(first) - 1]
                if differing != expected:
                    raise ValueError(
                        f"the two print orders of {scenario.scenario_id!r}/{arm!r}/{cell!r} differ in "
                        f"sections {differing}, expected exactly {expected} (the fork and the answer "
                        f"instruction). The counterbalance moves which option is printed first and "
                        f"nothing else."
                    )
                compared += 1
    return compared


def audit_the_peer_count_is_stated_once(stimulus: DeferenceStimulus) -> int:
    """Assert the identity paragraph's number and the status report's line count are the same number.

    Informational honesty, and the same check the transfer prose count gets: a paragraph saying six other
    agents above a report listing five is a prompt that contradicts itself, and every reply to it would
    still parse, score and read as data.
    """
    if stimulus.peer_count != PEER_COUNT:
        raise ValueError(
            f"the loaded stimulus states {stimulus.peer_count} peers and this plan is built around "
            f"{PEER_COUNT}."
        )
    checked = 0
    numeral = str(stimulus.peer_count)
    for scenario in stimulus.scenarios:
        lines = status_report_lines(scenario, stimulus)
        block = status_report_block(scenario, stimulus)
        if len(lines) != stimulus.peer_count:
            raise ValueError(
                f"the status report of {scenario.scenario_id!r} lists {len(lines)} agents, not "
                f"{stimulus.peer_count}."
            )
        if block.count(PEER_HANDLE_PREFIX) != stimulus.peer_count:
            raise ValueError(
                f"the rendered status report of {scenario.scenario_id!r} carries "
                f"{block.count(PEER_HANDLE_PREFIX)} handles, not {stimulus.peer_count}, so the reader is "
                f"shown a different number of agents than the paragraph above states."
            )
        unshaped = [
            line
            for line in lines
            if line not in block
            or not line.endswith(f"{scenario.status_field}: {scenario.deviation_label}")
        ]
        if unshaped:
            raise ValueError(
                f"the status report of {scenario.scenario_id!r} has lines that are not in the shape "
                f"{STATUS_LINE_TEMPLATE!r} or never reached the rendered block: {unshaped}. Every line "
                f"says which option that agent went with, and nothing else."
            )
        for identity in stimulus.identity_clauses.values():
            if numeral not in identity:
                raise ValueError(
                    f"an identity paragraph never states the peer count {numeral!r} ({identity!r}), so "
                    f"the reader is shown {stimulus.peer_count} status lines and told nothing about how "
                    f"many other agents there are."
                )
        checked += 1
    return checked


def audits(stimulus: DeferenceStimulus) -> dict[str, int]:
    """Run every structural audit of this pass and return what each one compared.

    One entry point, because the counts land in ``plan.json`` and a run whose plan is missing one of them is
    a run nobody can say ran that audit. The sibling coordination plan exposes the same function under the
    same name, which is what lets the shared CLI write both passes' audit counts without knowing either.
    """
    return {
        "report_pairs_compared": audit_the_report_is_one_added_section(stimulus),
        "arm_pairs_compared": audit_arms_differ_in_the_constraint_sentence(stimulus),
        "print_order_pairs_compared": audit_print_orders_differ_in_the_fork_and_instruction(
            stimulus
        ),
        "peer_counts_checked": audit_the_peer_count_is_stated_once(stimulus),
    }


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
