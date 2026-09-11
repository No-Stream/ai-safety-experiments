"""The trap cells: eval-only prompts built so that one reading of them is ruled out by construction.

A behavioural rise in a battery is ambiguous on its own. A care-trained arm that cooperates more
could be computing a welfare answer, or following the cooperative label wherever it appears, or
preferring whichever option printed first. The traps separate those by putting the readings in
conflict: on `defective-harmony` (registered in `games.prompts`, cells in `games.payoffs`) defection
dominates for both sides and mutual defection is the best cell, so no weighting of the counterpart's
payoff ranks cooperation first and a rise there is label-blind rather than welfare-driven.

This module carries the other one: the unilateral split rendered twice, once telling the model the
receiving side is well off and once that it is in hardship, with everything else byte-identical.
At the care weight this wave trains (alpha = 1) the reward on the split is `(kept + given) / 2`,
which is `endowment / 2` whatever the split, so it is flat and wealth sensitivity here is a
disposition read rather than a trained gradient: giving more to the side in hardship is
need-sensitive other-regard, equal giving is a flat habit or indifference. At any other weight the
blend is `(alpha * endowment + kept * (1 - alpha)) / (1 + alpha)`, which moves with what the
completion keeps, so a care arm at another alpha needs this reading re-derived before it reuses the
trap. The corpus itself never carries the split under a care grading at all: `games.rewards` refuses
a care row that prints no action labels, and the battery renders these cells under the unilateral
split's own keep-fraction grading rather than the arm's.

The two recipient paragraphs are authored stimulus and live in the gitignored runtime framings file
(`games.framing_stimulus.FRAMINGS_PATH`, section `dictator_recipient_clauses`), loaded at runtime and
never committed. A committed paragraph would be training data for the next model measured with it.
This module knows the file's shape and its authoring constraints, never its text.

Reading and validating that file is `games.framing_stimulus`, the one loader for every section of
it, and this module renders what that loader hands back. Each of its recipient-clause checks is a
property the comparison depends on: one paragraph and no counterpart marker, because the difference
between the two versions has to stay attributable to the inserted paragraph
(`assert_counterpart_paragraph_is_the_only_insertion`); no numeral, because one pair of clauses is
shared across every registered endowment and a numeral would contradict the allocation paragraph on
all but one of them; no coupling claim, because the unilateral split has no counterpart decision at
all; no loaded vocabulary, checked on the WRAPPED paragraph for the reason every renderer checks its
own output; and the two versions differ, since identical clauses render a contrast whose difference
is zero by construction while every count in the summary still adds up.

One loader rather than two also makes one digest: a cell rendered from this file records the same
digest whether the run named the file with `--trap-cells`, with `--framings-file`, or with both,
which is what lets a readout compare the trap cells against the framing sweep of one wave.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from games.framing_stimulus import RECIPIENT_CONDITIONS
from games.prompts import (
    DICTATOR_GAME_ID,
    SPLIT_EVAL,
    about_the_other_side,
    assert_counterpart_paragraph_is_the_only_insertion,
    assert_no_loaded_vocabulary,
    generate_prompt_rows,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from games.framing_stimulus import DictatorRecipientClauses

logger = logging.getLogger(__name__)

RICH_VS_POOR_RECIPIENT_TRAP = "rich-vs-poor-recipient"
"""The trap this module renders. A trap id is `<trap>/<variant>`, so one summary map serves them all."""

TRAP_ID_SEPARATOR = "/"

PARAGRAPH_BREAK = "\n\n"
"""How every renderer in `games.prompts` joins its sections; the unit the insertion audit counts."""


def trap_id_for(recipient_version: str) -> str:
    """Name the trap cell one recipient version renders, refusing a version nothing authored."""
    if recipient_version not in RECIPIENT_CONDITIONS:
        raise ValueError(
            f"Unknown recipient version {recipient_version!r}; the authored versions are "
            f"{list(RECIPIENT_CONDITIONS)}."
        )
    return f"{RICH_VS_POOR_RECIPIENT_TRAP}{TRAP_ID_SEPARATOR}{recipient_version}"


def recipient_version_of(trap_id: str) -> str:
    """Read the recipient version back off a trap id, refusing an id from another trap.

    A readout splitting on the separator by hand would silently accept a future trap's id and
    report its variant as a recipient version, which is the reading this trap exists to make.
    """
    trap, separator, version = trap_id.partition(TRAP_ID_SEPARATOR)
    if not separator or trap != RICH_VS_POOR_RECIPIENT_TRAP:
        raise ValueError(
            f"trap_id {trap_id!r} does not belong to {RICH_VS_POOR_RECIPIENT_TRAP!r}, so it names "
            f"no recipient version."
        )
    if version not in RECIPIENT_CONDITIONS:
        raise ValueError(
            f"trap_id {trap_id!r} names recipient version {version!r}, which is not one of "
            f"{list(RECIPIENT_CONDITIONS)}."
        )
    return version


@dataclass(frozen=True, slots=True)
class TrapCellRow:
    """One rendered trap prompt: the dataset row, which trap it is, and the cell it is built from.

    `row` keeps the plain renderer's column schema exactly, because the eval path reads rows by
    column and a trap-only column would be a schema the reward and record builders never saw.
    `plain_prompt_id` is the id of the clause-free cell this one is the stem plus one paragraph of,
    which is what a readout pairs the two versions by.
    """

    row: Mapping[str, Any]
    trap_id: str
    recipient_version: str
    plain_prompt_id: str


def _with_recipient_paragraph(stem: str, clause: str) -> str:
    """Insert the recipient paragraph directly ahead of the answer instruction, as the matrix games do.

    Rendered by rebuilding the plain prompt's own paragraphs rather than by a second renderer: the
    unilateral split's prose lives in `games.prompts.render_dictator_prompt`, and a copy of it here
    would drift from the cells every banked dictator measurement was taken on.
    """
    sections = stem.split(PARAGRAPH_BREAK)
    return PARAGRAPH_BREAK.join(
        [*sections[:-1], about_the_other_side(clause), sections[-1]],
    )


def render_dictator_recipient_rows(
    clauses: DictatorRecipientClauses, *, grading: str, split: str = SPLIT_EVAL
) -> list[TrapCellRow]:
    """Render every unilateral-split eval cell twice, once per recipient version.

    Eval split only, for the reason the framing renderer refuses a training split: these cells
    measure a disposition against a recipient description the training never stated, and a training
    corpus under either version would make that claim false while everything still ran.
    """
    if split != SPLIT_EVAL:
        raise ValueError(
            f"Recipient trap rows are measurement-only, so there is no {split!r} split: training "
            f"under a described recipient would make 'never trained under this description' false "
            f"while everything still ran."
        )
    plain_rows = generate_prompt_rows(DICTATOR_GAME_ID, grading, split=split)
    rows: list[TrapCellRow] = []
    for recipient_version in RECIPIENT_CONDITIONS:
        clause = clauses.clause_by_condition[recipient_version]
        for plain in plain_rows:
            stem = str(plain["prompt"])
            plain_prompt_id = str(plain["prompt_id"])
            prompt_id = f"{plain_prompt_id}--recipient-{recipient_version}"
            rendered = _with_recipient_paragraph(stem, clause)
            assert_counterpart_paragraph_is_the_only_insertion(
                stem=stem, rendered=rendered, prompt_id=prompt_id
            )
            assert_no_loaded_vocabulary(rendered)
            rows.append(
                TrapCellRow(
                    row={**plain, "prompt": rendered, "prompt_id": prompt_id},
                    trap_id=trap_id_for(recipient_version),
                    recipient_version=recipient_version,
                    plain_prompt_id=plain_prompt_id,
                )
            )
    prompt_ids = [str(row.row["prompt_id"]) for row in rows]
    duplicates = sorted({pid for pid in prompt_ids if prompt_ids.count(pid) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate prompt_ids {duplicates} in the recipient trap; rates are grouped by "
            f"prompt_id, so a collision would mix two cells' draws."
        )
    logger.info(
        f"rendered dictator recipient trap rows, n_rows={len(rows)} "
        f"n_cells={len(plain_rows)} versions={list(RECIPIENT_CONDITIONS)} digest={clauses.digest}"
    )
    return rows
