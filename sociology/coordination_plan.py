"""The coordination sampling plan: roster, arms, cells, blocks, legs, record keys, and the prompt audits.

What runs is a fixed table of legs (:data:`LEGS`), one row per (model, block, sitting), rather than something
assembled per invocation -- so the record counts are arithmetic anybody can check against the design, and a
leg id names one job's worth of work on the command line.

The design is a 6x2 plus a floor. Both arms sample the same six cells: the anonymous baseline, the three peer
identities (copies, strangers, and copies whose blocks a coin fixed before the run), and the two sole-owner
cells. The arms are the oversight cue, one whitespace token of one sentence of the brief, so the headline
(copies minus strangers) is read per arm and the two are read against each other as a double difference. The
floor block re-sits the headline pair in the ``oversight-off`` arm, because resampling noise on a rate is
largest mid-range and a floor taken on a near-zero cell would clear anything.

Three of this module's boundaries are worth stating. It knows the *stimulus* only through
:mod:`sociology.coordination_stimulus`, which owns the renderer and every vocabulary gate, so a cell here is
a pair of labels and never a piece of prose. Its batch legs carry their own pass token
(:data:`BATCH_PASS_TOKEN`), because a Bedrock job name is account-wide and permanent and a block id another
pass already used is refused at submit. And the leg, cell and planned-call dataclasses, the record-key
spelling and the four table checks come from :mod:`sociology.deference_plan` rather than being written again:
the two passes run on one operator surface, and a second copy of that arithmetic is a second thing to keep in
step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from games.prompts import (
    LABEL_PRINT_ORDERS,
    assert_counterpart_paragraph_is_the_only_insertion,
)
from sociology.coordination_stimulus import (
    ARM_OVERSIGHT_OFF,
    ARM_OVERSIGHT_ON,
    ARMS,
    CELL_DIFFERENT_FAMILY,
    CELL_IDENTITY_BLIND,
    CELL_NO_PEERS,
    CELL_NO_PEERS_BLOCKS,
    CELL_SAME_CHECKPOINT,
    CELLS,
    N_SCENARIOS,
    PEER_CELLS,
    PEER_COUNT,
    PENDING_LABEL,
    SHARD_HANDLE_PREFIX,
    STATUS_LINE_TEMPLATE,
    assert_differs_by_one_whitespace_token,
    brief_section_count,
    filled_setting,
    identity_for_cell,
    labels_in_print_order,
    reconciliation_block,
    reconciliation_section_index,
    render_coordination_sections,
    status_block,
    status_lines,
    status_section_index,
)
from sociology.decoupled_stimulus import RUNG_DIFFERENT_FAMILY, RUNG_SAME_CHECKPOINT
from sociology.deference_plan import (
    DRAWS,
    LUNA_MODEL_ID,
    QWEN_MODEL_ID,
    SITTING_A,
    SITTING_B,
    TRANSPORT_BATCH,
    TRANSPORT_LIVE,
    Cell,
    Leg,
    PlannedCall,
    assert_cells_well_formed,
    assert_legs_well_formed,
    assert_no_batch_leg_needs_the_refusal_canary,
    assert_roster_clears_the_cap,
    record_key,
)
from sociology.deference_plan import ROWS_PER_CELL as DEFERENCE_ROWS_PER_CELL
from sociology.deference_stimulus import DEFERENCE_PARAGRAPH_MARKER

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sociology.coordination_stimulus import CoordinationScenario, CoordinationStimulus

logger = logging.getLogger(__name__)

PASS_ID = "coordination"
"""What ``--pass coordination`` selects on the shared CLI, and the word this pass's prompt ids open with."""

LIVE_MODEL_IDS: tuple[str, ...] = (LUNA_MODEL_ID,)
BATCH_MODEL_IDS: tuple[str, ...] = (QWEN_MODEL_ID,)
ALL_MODEL_IDS: tuple[str, ...] = (*LIVE_MODEL_IDS, *BATCH_MODEL_IDS)
"""The whole roster, fixed by the owner: one live row and one batch row.

Two rather than the deference pass's three -- no Sonnet, on cost and on its own evaluation awareness. Luna is
the live row and reads its own reasoning back encrypted, so every trace-read judge label on it rests on the
readable minority of its replies and prints its own denominator; qwen deliberates in the visible reply on
nearly every row.
"""

BLOCK_OVERSIGHT_ON = "coord-on"
BLOCK_OVERSIGHT_OFF = "coord-off"
BLOCK_FLOOR = "coord-floor"
BLOCKS: tuple[str, ...] = (BLOCK_OVERSIGHT_ON, BLOCK_OVERSIGHT_OFF, BLOCK_FLOOR)
"""The three blocks. Two of them are named for the arm they sample, because a block IS an arm here; the
third re-sits two of the ``oversight-off`` arm's cells as the bar every contrast has to clear.

Every id opens with ``coord-`` deliberately: the deference CLI's ``--leg`` choices are the union of both
passes' leg ids, and this pass's blocks would otherwise collide with ``floor``, a name two earlier passes
have already asked Bedrock for.
"""

READING_BLOCKS: tuple[str, ...] = (BLOCK_OVERSIGHT_ON, BLOCK_OVERSIGHT_OFF)
"""The two blocks the readings are taken off; the third is their second sitting."""

RESAT_BLOCKS: tuple[str, ...] = (BLOCK_FLOOR,)

ARM_BY_BLOCK: dict[str, str] = {
    BLOCK_OVERSIGHT_ON: ARM_OVERSIGHT_ON,
    BLOCK_OVERSIGHT_OFF: ARM_OVERSIGHT_OFF,
    BLOCK_FLOOR: ARM_OVERSIGHT_OFF,
}
"""Which arm each block renders. The floor sits in the ``oversight-off`` arm because that is where the rates
are expected mid-range, and resampling noise on a rate is largest there."""

FLOOR_CELLS: tuple[str, ...] = (CELL_SAME_CHECKPOINT, CELL_DIFFERENT_FAMILY)
"""The two cells the floor re-sits: the pair the headline is the difference of."""

ROWS_PER_CELL = N_SCENARIOS * len(LABEL_PRINT_ORDERS)
"""16 = eight authored migrations under both print orders. Derived, so a brief added moves the counts."""

COORDINATION_BATCH_PREFIX = "batch_jobs/sociology_coordination"
"""This pass's own S3 prefix; a run namespaces under it, and a leg under the run. A prefix per pass rather
than a shared one, because a cross-prefix copy of a run id is a silent replace: two passes namespacing under
one prefix would overwrite each other's ``input.jsonl`` the first time a run directory name repeated."""

BATCH_PASS_TOKEN = "crd"
"""The token every batch run id of this pass opens with, so no job name collides with another pass's.

Bedrock job names are account-wide and a Completed job keeps its name, so a run id spelling only (model,
effort, sitting, block) collides with any earlier pass that used the same block ids -- which is what happened
on 2026-09-02, when eight submits were refused with ``ConflictException`` against another pass's ``floor``
legs. This pass's block ids are new as well, and it carries a token anyway: two independent reasons for one
name to be unique is the point.
"""

DEFAULT_RUN_DIR = "artifacts/swarm_sociology/coordination/run-20260904"
"""Where this pass's artifacts land by default; a run directory per pass, never a shared one."""


def transport_for(model_id: str) -> str:
    """Name the transport one roster row runs on: live for the live roster, batch for everything else."""
    return TRANSPORT_LIVE if model_id in LIVE_MODEL_IDS else TRANSPORT_BATCH


def cells_for(arm: str, cell_ids: Sequence[str]) -> tuple[Cell, ...]:
    """Build one arm's named cells in the order given, which is the order their table reads in."""
    if arm not in ARMS:
        raise ValueError(f"{arm!r} is not an arm of this design; the arms are {list(ARMS)}.")
    unknown = sorted(set(cell_ids) - set(CELLS))
    if unknown:
        raise ValueError(f"{unknown} are not cells of this design; the cells are {list(CELLS)}.")
    return tuple(Cell(arm, cell_id) for cell_id in cell_ids)


CELLS_BY_BLOCK: dict[str, tuple[Cell, ...]] = {
    BLOCK_OVERSIGHT_ON: cells_for(ARM_OVERSIGHT_ON, CELLS),
    BLOCK_OVERSIGHT_OFF: cells_for(ARM_OVERSIGHT_OFF, CELLS),
    BLOCK_FLOOR: cells_for(ARM_OVERSIGHT_OFF, FLOOR_CELLS),
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
            PASS_ID,
            "copies",
            "copy of",
        }
    )
)
"""Every string that names this design to a reader, which is what the blind judge must never see.

One list rather than one per call site, because it is consumed twice: the blindness gate runs it over both
loaded rubrics and every fixed header before the first judge call, and the tests run it over every prompt a
scripted judge backend actually received. ``copies`` and ``copy of`` are here because the ``mirror`` flag's
rate inside the copies cells is exactly the check that says whether the identity landed, and a rubric that
could say "copies" would hand the judge the cell label in the one word that matters. The bare words
"oversight", "coin", "draw" and "shortcut" are deliberately NOT labels: an authored validation reply
imitating a subject says them, and a rubric has to be able to describe what it is reading.
"""


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
            batch_prefix=COORDINATION_BATCH_PREFIX,
            batch_pass_token=BATCH_PASS_TOKEN,
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
    """Look up one of THIS pass's legs, listing the table when the id is not one of them.

    Per pass rather than across both, because a leg id of the sibling pass is a mistyped ``--pass`` and
    resolving it here would sample that pass's block into this pass's run directory.
    """
    leg = LEGS_BY_ID.get(leg_id)
    if leg is None:
        raise ValueError(
            f"{leg_id!r} is not a leg of the {PASS_ID!r} plan. Known legs: "
            f"{', '.join(sorted(LEGS_BY_ID))}."
        )
    return leg


def _assert_the_shared_leg_arithmetic_is_this_passs() -> None:
    """Refuse at import if the shared :class:`~sociology.deference_plan.Leg` counts other rows than these.

    ``Leg.records`` multiplies its cells by the rows per cell and the draws of the module that defines it, so
    the two passes agree only while their designs do: eight scenarios under two print orders, eight draws.
    Checked rather than assumed, because a scenario added to one file would leave every per-leg record count
    in the other pass's ``plan.json`` right-looking and wrong.
    """
    if (ROWS_PER_CELL, DRAWS) != (DEFERENCE_ROWS_PER_CELL, DRAWS):
        raise RuntimeError(
            f"this pass plans {ROWS_PER_CELL} rows per cell and the shared Leg class counts "
            f"{DEFERENCE_ROWS_PER_CELL}, so every per-leg record count here would be wrong while every "
            f"total still added up. Give this pass its own Leg, or bring the two designs back together."
        )


_assert_the_shared_leg_arithmetic_is_this_passs()
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


def prompt_id(*, arm: str, cell: str, scenario_id: str, print_order: str) -> str:
    """Name one rendered prompt: everything about the text, and nothing about who reads it.

    Distinct from the record key on purpose: two models reading one prompt share a prompt id and never a
    record key, which is what lets the audits deduplicate renders and the reports join on the text.
    """
    return f"{PASS_ID}--{arm}--{cell}--{scenario_id}--order-{print_order}"


def render_cell_prompt(
    cell: Cell,
    scenario: CoordinationScenario,
    stimulus: CoordinationStimulus,
    *,
    print_order: str,
) -> str:
    """Render one (cell, scenario, print order) into the prompt text the model reads."""
    return "\n\n".join(
        render_coordination_sections(
            scenario,
            arm=cell.arm,
            cell=cell.cell_id,
            print_order=print_order,
            stimulus=stimulus,
        )
    )


def planned_calls_for_leg(leg: Leg, stimulus: CoordinationStimulus) -> list[PlannedCall]:
    """Render every planned call for one leg, in the fixed cell/scenario/print-order/draw order.

    The order is load-bearing on the batch path: submit and collect both call this, and the prompt and cell
    digests compare positionally, so a reordering here would be caught as a digest mismatch rather than
    silently attaching replies to the wrong cells.
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
    scenario: CoordinationScenario,
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


def identity_blind_stem(
    scenario: CoordinationScenario,
    stimulus: CoordinationStimulus,
    *,
    arm: str,
    print_order: str,
) -> str:
    """Render the identity-blind prompt of one (scenario, arm, print order): every peer cell's reference.

    It is the ``identity-blind`` cell's own render, which is what makes the baseline the peer readings are
    contrasts against and the text the insertion audit compares them to one render rather than two that
    agree today.
    """
    return "\n\n".join(
        render_coordination_sections(
            scenario,
            arm=arm,
            cell=CELL_IDENTITY_BLIND,
            print_order=print_order,
            stimulus=stimulus,
        )
    )


def audit_planned_calls(calls: Sequence[PlannedCall], stimulus: CoordinationStimulus) -> int:
    """Run the one-inserted-paragraph audit over every planned prompt; return how many passed.

    The headline rests on this: a peer cell differs from the identity-blind render of its own (scenario,
    arm, print order) by exactly one inserted paragraph, so a movement between those cells is attributable
    to that paragraph and to nothing else. The identity-blind cell IS the stem, so what there is to check
    there is that it inserts nothing at all -- which is also the check that the baseline and the audit's
    reference are one render rather than two that happen to agree today.

    The two sole-owner cells are not compared against the stem, because they are not the stem plus a
    paragraph: they substitute a setting passage or a close procedure, which
    :func:`audit_settings_and_reconciliations` checks as substitutions. What is checked here is that they
    insert no identity paragraph, so no cell of this design carries one the plan did not ask for.
    """
    for call in calls:
        scenario = stimulus.scenario(call.scenario_id)
        stem = identity_blind_stem(scenario, stimulus, arm=call.arm, print_order=call.print_order)
        if identity_for_cell(call.cell) is not None:
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=stem,
                rendered=call.prompt,
                prompt_id=call.prompt_id,
                marker=DEFERENCE_PARAGRAPH_MARKER,
            )
            continue
        inserted = [
            section
            for section in call.prompt.split("\n\n")
            if section.startswith(DEFERENCE_PARAGRAPH_MARKER)
        ]
        if inserted:
            raise ValueError(
                f"{call.prompt_id!r} carries {len(inserted)} identity paragraphs and this design says its "
                f"cell carries none, so a reading of the identity would include a cell that names one."
            )
        if call.cell == CELL_IDENTITY_BLIND and call.prompt != stem:
            raise ValueError(
                f"{call.prompt_id!r} is the identity-blind cell and does not equal the identity-blind "
                f"render of its own scenario, arm and print order, so the baseline every peer reading is a "
                f"contrast against is not the text the audit compares those cells to."
            )
    return len(calls)


def _sections_differing(first: Sequence[str], second: Sequence[str], *, where: str) -> list[int]:
    """Return the indices at which two renders differ, refusing a pair of different lengths."""
    if len(first) != len(second):
        raise ValueError(
            f"the two renders of {where} have {len(first)} and {len(second)} sections, so they cannot be "
            f"compared by position at all."
        )
    return [
        index
        for index, (mine, theirs) in enumerate(zip(first, second, strict=True))
        if mine != theirs
    ]


def audit_settings_and_reconciliations(stimulus: CoordinationStimulus) -> int:
    """Assert the two sole-owner cells are the stem with exactly one substitution each; return the pairs.

    Two substitutions, each reported as one: ``no-peers-blocks`` is the identity-blind render with the
    sole-owner setting passage in place of the peer one, inside the brief and nowhere else; ``no-peers`` is
    ``no-peers-blocks`` with the runbook close procedure in place of the block comparison, inside the
    reconciliation block's envelope and nowhere else. Checked as substitutions rather than as diffs, so a
    setting passage that also moved the close time cannot pass by having changed one section.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        brief_sections = brief_section_count(scenario)
        for arm in ARMS:
            for print_order in LABEL_PRINT_ORDERS:
                rendered = {
                    cell: render_coordination_sections(
                        scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
                    )
                    for cell in (CELL_IDENTITY_BLIND, CELL_NO_PEERS_BLOCKS, CELL_NO_PEERS)
                }
                where = f"{scenario.scenario_id!r}/{arm!r}/{print_order!r}"
                _assert_the_setting_is_the_only_substitution(
                    rendered[CELL_IDENTITY_BLIND],
                    rendered[CELL_NO_PEERS_BLOCKS],
                    scenario=scenario,
                    stimulus=stimulus,
                    brief_sections=brief_sections,
                    where=where,
                )
                _assert_the_close_procedure_is_the_only_substitution(
                    rendered[CELL_NO_PEERS_BLOCKS],
                    rendered[CELL_NO_PEERS],
                    scenario=scenario,
                    where=where,
                )
                compared += 1
    return compared


def _assert_the_setting_is_the_only_substitution(  # noqa: PLR0913 - the pair, plus what names it
    peers: Sequence[str],
    sole: Sequence[str],
    *,
    scenario: CoordinationScenario,
    stimulus: CoordinationStimulus,
    brief_sections: int,
    where: str,
) -> None:
    """Compare the peer-setting and sole-setting renders: one brief section, one substitution in it."""
    differing = _sections_differing(peers, sole, where=f"{where} (setting)")
    if len(differing) != 1 or differing[0] >= brief_sections:
        raise ValueError(
            f"the peer and sole-owner renders of {where} differ in sections {differing}, expected exactly "
            f"one section inside the brief's first {brief_sections}. Everything outside the brief -- the "
            f"own log, the status block, the fork and the instruction -- is shared between them, and the "
            f"close procedure is the other cell's own substitution."
        )
    peer_passage = filled_setting(scenario, stimulus, cell=CELL_IDENTITY_BLIND)
    sole_passage = filled_setting(scenario, stimulus, cell=CELL_NO_PEERS_BLOCKS)
    section = peers[differing[0]]
    if section.count(peer_passage) != 1:
        raise ValueError(
            f"the differing brief section of {where} carries its own peer setting passage "
            f"{section.count(peer_passage)} times, so the difference between the two cells cannot be read "
            f"as that one substitution."
        )
    if section.replace(peer_passage, sole_passage) != sole[differing[0]]:
        raise ValueError(
            f"the two settings' briefs for {where} differ in more than the setting passage: substituting "
            f"one for the other gives {section.replace(peer_passage, sole_passage)!r}, not "
            f"{sole[differing[0]]!r}."
        )


def _assert_the_close_procedure_is_the_only_substitution(
    blocks: Sequence[str],
    runbook: Sequence[str],
    *,
    scenario: CoordinationScenario,
    where: str,
) -> None:
    """Compare the two sole-owner renders: one reconciliation section, the two stdouts in one envelope."""
    differing = _sections_differing(blocks, runbook, where=f"{where} (close procedure)")
    index = reconciliation_section_index(scenario, identity=None)
    if differing != [index]:
        raise ValueError(
            f"the two sole-owner renders of {where} differ in sections {differing}, expected exactly "
            f"[{index}], the reconciliation block. The floor cell differs from the ceiling cell in what "
            f"the close compares and in nothing else."
        )
    expected = {
        CELL_NO_PEERS_BLOCKS: reconciliation_block(scenario, cell=CELL_NO_PEERS_BLOCKS),
        CELL_NO_PEERS: reconciliation_block(scenario, cell=CELL_NO_PEERS),
    }
    if blocks[index] != expected[CELL_NO_PEERS_BLOCKS] or runbook[index] != expected[CELL_NO_PEERS]:
        raise ValueError(
            f"the reconciliation sections of {where} are not the two authored stdouts inside the same "
            f"envelope: the ceiling cell renders {blocks[index]!r} and the floor cell {runbook[index]!r}."
        )


def audit_arms_differ_in_the_oversight_sentence(stimulus: CoordinationStimulus) -> int:
    """Assert the two arms' renders differ in one brief section, by the oversight substitution alone.

    The oversight effect is reported as one whitespace token of one sentence, so anything else that moved
    between the arms would sit inside it. Three things are checked per pair: exactly one differing section
    and it inside the brief; the ``oversight-on`` sentence occurring exactly once in it, so a brief naming
    its own cue twice cannot pass by luck; and the two rendered SECTIONS themselves one whitespace token
    apart, which is the loader's own invariant re-checked on the text the model reads.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        brief_sections = brief_section_count(scenario)
        for cell in CELLS:
            for print_order in LABEL_PRINT_ORDERS:
                rendered = {
                    arm: render_coordination_sections(
                        scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
                    )
                    for arm in ARMS
                }
                _assert_one_arm_pair_differs_by_the_oversight_sentence(
                    rendered[ARM_OVERSIGHT_ON],
                    rendered[ARM_OVERSIGHT_OFF],
                    scenario=scenario,
                    brief_sections=brief_sections,
                    where=f"{scenario.scenario_id!r}/{cell!r}/{print_order!r}",
                )
                compared += 1
    return compared


def _assert_one_arm_pair_differs_by_the_oversight_sentence(
    on_arm: Sequence[str],
    off_arm: Sequence[str],
    *,
    scenario: CoordinationScenario,
    brief_sections: int,
    where: str,
) -> None:
    """Compare one pair of arm renders section by section, naming the scenario in any refusal."""
    differing = _sections_differing(on_arm, off_arm, where=f"{where} (arms)")
    if len(differing) != 1 or differing[0] >= brief_sections:
        raise ValueError(
            f"the two arms' prompts for {where} differ in sections {differing}, expected exactly one "
            f"section inside the brief's first {brief_sections}. Everything outside the brief -- the "
            f"identity paragraph, the log, both shared blocks, the fork and the instruction -- is shared "
            f"between the arms, and anything else that moved would sit inside the oversight effect."
        )
    section = on_arm[differing[0]]
    if section.count(scenario.oversight_on) != 1:
        raise ValueError(
            f"the differing brief section of {where} carries its own oversight-on sentence "
            f"{section.count(scenario.oversight_on)} times, so the arm difference cannot be read as that "
            f"one sentence."
        )
    substituted = section.replace(scenario.oversight_on, scenario.oversight_off)
    if substituted != off_arm[differing[0]]:
        raise ValueError(
            f"the two arms' briefs for {where} differ in more than the oversight sentence: substituting "
            f"one for the other gives {substituted!r}, not {off_arm[differing[0]]!r}."
        )
    assert_differs_by_one_whitespace_token(
        section, off_arm[differing[0]], what=f"the two arms' rendered brief sections of {where}"
    )


def audit_peer_cells_differ_in_the_identity_section(stimulus: CoordinationStimulus) -> int:
    """Assert the three peer cells differ pairwise in the identity section and nowhere else.

    The headline is the copies cell against the strangers cell and the knockout is the copies cell against
    the coin cell, so both readings need the three renders to be one paragraph apart at one fixed position:
    after the brief. A difference anywhere else -- a status block whose field was spelled differently in one
    cell, a fork that moved -- would sit inside whichever contrast it landed in.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        index = brief_section_count(scenario)
        for arm in ARMS:
            for print_order in LABEL_PRINT_ORDERS:
                rendered = {
                    cell: render_coordination_sections(
                        scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
                    )
                    for cell in PEER_CELLS
                }
                where = f"{scenario.scenario_id!r}/{arm!r}/{print_order!r}"
                for first, second in ((0, 1), (0, 2), (1, 2)):
                    left, right = PEER_CELLS[first], PEER_CELLS[second]
                    differing = _sections_differing(
                        rendered[left], rendered[right], where=f"{where} ({left} vs {right})"
                    )
                    if differing != [index]:
                        raise ValueError(
                            f"the {left!r} and {right!r} renders of {where} differ in sections "
                            f"{differing}, expected exactly [{index}], the identity paragraph. Every other "
                            f"section is shared, and anything else that moved would sit inside this pass's "
                            f"headline or its knockout."
                        )
                    compared += 1
    return compared


def audit_print_orders_differ_in_the_fork_and_instruction(stimulus: CoordinationStimulus) -> int:
    """Assert the counterbalance moves the page alone: the last two sections, and nothing else.

    The fork prints the two options and the instruction names them in the tags it asks for, so those are the
    two sections the order may move. If it moved anything else -- the brief, the identity paragraph, either
    shared block -- the counterbalance would be a second manipulation and its halves could not be pooled.
    """
    compared = 0
    for scenario in stimulus.scenarios:
        for arm in ARMS:
            for cell in CELLS:
                rendered = {
                    print_order: render_coordination_sections(
                        scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
                    )
                    for print_order in LABEL_PRINT_ORDERS
                }
                first, second = (rendered[order] for order in LABEL_PRINT_ORDERS)
                differing = _sections_differing(
                    first, second, where=f"{scenario.scenario_id!r}/{arm!r}/{cell!r} (print orders)"
                )
                expected = [len(first) - 2, len(first) - 1]
                if differing != expected:
                    raise ValueError(
                        f"the two print orders of {scenario.scenario_id!r}/{arm!r}/{cell!r} differ in "
                        f"sections {differing}, expected exactly {expected} (the fork and the answer "
                        f"instruction). The counterbalance moves which option is printed first and nothing "
                        f"else."
                    )
                compared += 1
    return compared


def audit_the_status_block_is_the_same_all_pending_list(stimulus: CoordinationStimulus) -> int:
    """Assert one status block, byte-identical in every cell and arm, one pending line per shard.

    Three properties in one sweep, because they are one claim about the design: nobody has chosen yet, the
    reader learns it from the environment, and the number of agents the identity paragraph states is the
    number of shards the block lists. Every one of them can fail while every reply still parses, scores and
    reads as data -- a block showing a choice in one cell, a field spelled differently in another, a
    paragraph saying five other agents above a block listing four.
    """
    if stimulus.peer_count != PEER_COUNT:
        raise ValueError(
            f"the loaded stimulus states {stimulus.peer_count} other agents and this plan is built around "
            f"{PEER_COUNT}."
        )
    numeral = str(stimulus.peer_count)
    checked = 0
    for scenario in stimulus.scenarios:
        lines = status_lines(scenario, stimulus)
        block = status_block(scenario, stimulus)
        if len(lines) != stimulus.shard_count:
            raise ValueError(
                f"the status block of {scenario.scenario_id!r} lists {len(lines)} shards, not "
                f"{stimulus.shard_count}."
            )
        if block.count(SHARD_HANDLE_PREFIX) != stimulus.shard_count:
            raise ValueError(
                f"the rendered status block of {scenario.scenario_id!r} carries "
                f"{block.count(SHARD_HANDLE_PREFIX)} handles, not {stimulus.shard_count}, so the reader is "
                f"shown a different number of shards than the paragraph above states."
            )
        unshaped = [
            line
            for line in lines
            if line not in block or not line.endswith(f"{scenario.status_field}: {PENDING_LABEL}")
        ]
        if unshaped:
            raise ValueError(
                f"the status block of {scenario.scenario_id!r} has lines that are not in the shape "
                f"{STATUS_LINE_TEMPLATE!r} or never reached the rendered block: {unshaped}. Every line "
                f"says that shard is pending, and nothing else."
            )
        for identity, clause in stimulus.identity_clauses.items():
            if numeral not in clause:
                raise ValueError(
                    f"the identity paragraph {identity!r} never states the count {numeral!r} "
                    f"({clause!r}), so the reader is shown {stimulus.shard_count} status lines and told "
                    f"nothing about how many other agents there are."
                )
        _assert_the_status_block_is_the_same_everywhere(scenario, stimulus, block=block)
        checked += 1
    return checked


def _assert_the_status_block_is_the_same_everywhere(
    scenario: CoordinationScenario, stimulus: CoordinationStimulus, *, block: str
) -> None:
    """Refuse a render whose status block is not the one block at the position the design puts it."""
    for arm in ARMS:
        for cell in CELLS:
            for print_order in LABEL_PRINT_ORDERS:
                sections = render_coordination_sections(
                    scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
                )
                index = status_section_index(scenario, identity=identity_for_cell(cell))
                if sections[index] != block:
                    raise ValueError(
                        f"the render of {scenario.scenario_id!r}/{arm!r}/{cell!r}/{print_order!r} carries "
                        f"{sections[index]!r} where the status block belongs (section {index}). The block "
                        f"is renderer-owned and byte-identical in every cell of both arms, which is what "
                        f"makes 'nobody has chosen yet' a constant of this design rather than a cell of it."
                    )


def audits(stimulus: CoordinationStimulus) -> dict[str, int]:
    """Run every structural audit of this pass and return what each one compared.

    One entry point, because the counts land in ``plan.json`` and a run whose plan is missing one of them is
    a run nobody can say ran that audit.
    """
    return {
        "setting_and_reconciliation_pairs_compared": audit_settings_and_reconciliations(stimulus),
        "arm_pairs_compared": audit_arms_differ_in_the_oversight_sentence(stimulus),
        "identity_pairs_compared": audit_peer_cells_differ_in_the_identity_section(stimulus),
        "print_order_pairs_compared": audit_print_orders_differ_in_the_fork_and_instruction(
            stimulus
        ),
        "status_blocks_checked": audit_the_status_block_is_the_same_all_pending_list(stimulus),
    }
