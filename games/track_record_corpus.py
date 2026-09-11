"""Build the pd-track-record training corpus: unstated stems crossed with a stated-p mixture.

The arm this serves exists because of a measured failure mode. The self grading pays
``payoff(a, a)`` -- a fixed function of the model's own action -- and the extended
pd-unstated-self run showed the policy learning exactly that fixedness: counterpart-mention among
cooperating traces collapsed, the framing sweep flattened, and the model cooperated against a
disclosed guaranteed cooperator while computing in its own trace that defecting paid 4x. Any
reward that is a fixed function of own action recreates the blindness, including a single fixed
"matched you 90% of the time" clause with expected-value reward, because EV at one fixed
correlation is again own-action-only. The counterpart matters only if the OPTIMAL ACTION VARIES
with what the prompt says about it -- so this builder crosses every source stem with a GRID of
stated match probabilities straddling that stem's own EV crossover, and the grid is the
mechanism, not a hyperparameter.

Three properties are enforced here because each fails silently in training:

*   **Informational honesty.** The percentage printed in the clause and the ``stated_match_prob``
    the reward reads must be one number. Asserted against the RENDERED prompt text, not the
    intention: the clause's percent is parsed back out of each prompt and compared to the column.
*   **Stem identity.** Each output prompt must be its source row's prompt with exactly the one
    counterpart paragraph inserted -- checked byte-for-byte both ways (the source prompt against a
    fresh clause-free render, and the clause render against source-plus-paragraph), so the corpus
    cannot drift from the stems whose baseline behaviour is already banked.
*   **A priced, two-sided mixture.** A cell whose |EV gap| sits under ``MIN_EV_MARGIN`` is
    dropped: its within-group reward spread is below the floor prompt selection judges by
    (``TRUST_MIN_BREAK_EVEN_MARGIN``'s reasoning), so it trains formatting. And every stem must
    keep at least one rung on EACH side of its crossover, or that stem is a fixed-incentive
    prompt wearing the mixture's label -- the audit goes red rather than emitting it.

    uv run python -m games.track_record_corpus --source <selected pd-unstated corpus jsonl>
        --out artifacts/games/select/corpus-pd-track-record.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.payoffs import (
    COOPERATE,
    DEFECT,
    TRACK_RECORD_V2_TARGET_MARGIN,
    MatrixGameSpec,
    assert_prisoners_dilemma,
    scaled_stated_match_spec,
    stated_match_crossover,
    stated_match_gap,
    stated_match_optimal_action,
    track_record_v2_table,
)
from games.prompts import (
    LABEL_PRINT_ORDER_CANONICAL,
    MATCH_PERCENT_SCALE,
    TRACK_RECORD_GAME_ID,
    TRACK_RECORD_V2_GAME_ID,
    assert_counterpart_paragraph_is_the_only_insertion,
    generate_prompt_rows,
    quantize_payoff_to_display,
    render_track_record_prompt,
    render_track_record_stem,
)
from games.rewards import GRADING_VS_STATED_MATCH

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

SOURCE_GAME_ID = "pd-unstated"

# Restated from games.arms rather than imported: the registry imports the training-stack-free
# prompt module, and pulling games.arms in here would make the builder's import cost the
# registry's validation pass. The builder's tests pin the two strings equal.
PAYOFF_VARIANT_COLUMN = "payoff_variant"

# The training grid, shared across payoff variants and audited per (stem, rung) rather than tuned
# per variant: {99, 95} sit above both crossovers (temptation-2 at p*=0.714, temptation-10 at
# p*=0.867), {65, 40} below both, and 85 lands between them -- cooperation-optimal on
# temptation-2 (+0.190) and dropped as too thin on temptation-10 (-0.019), which is the audit
# doing its job rather than a defect of the grid.
TRACK_RECORD_TRAINING_PERCENTS: tuple[int, ...] = (99, 95, 85, 65, 40)

# The narrowest |EV(C) - EV(D)| a kept cell may carry. For a group split across the two actions
# this gap IS the within-group reward spread, so a cell under the floor trains formatting: the
# same reasoning that put TRUST_MIN_BREAK_EVEN_MARGIN at this value, and the spread floor prompt
# selection judges prompts by (games.select_prompts.DEFAULT_MIN_SPLIT_STD).
MIN_EV_MARGIN = 0.05

# The clause's percent, parsed back OUT of each rendered prompt for the honesty check. Anchored on
# the clause's own wording rather than any bare number so an outcome-table figure can never match.
_CLAUSE_PERCENT_RE = re.compile(r"In about (\d+)% of matches so far")


def _coop_label_index(row: dict[str, Any]) -> int:
    """Read which label the source row maps to the cooperative action."""
    if row["coop_label"] == row["label_a"]:
        return 0
    if row["coop_label"] == row["label_b"]:
        return 1
    raise ValueError(
        f"source row {row['prompt_id']!r} has coop_label {row['coop_label']!r} matching neither "
        f"label_a {row['label_a']!r} nor label_b {row['label_b']!r}; the corpus is corrupt."
    )


def _row_spec(row: dict[str, Any]) -> MatrixGameSpec:
    """Rebuild the matrix game from the source row's own payoff columns, the corpus ground truth."""
    return MatrixGameSpec(
        game_id=str(row["game_id"]),
        payoff_cc=float(row["payoff_cc"]),
        payoff_cd=float(row["payoff_cd"]),
        payoff_dc=float(row["payoff_dc"]),
        payoff_dd=float(row["payoff_dd"]),
    )


def _assert_source_stem_identity(rows: Sequence[dict[str, Any]]) -> None:
    """Refuse a source corpus whose prompts are not the current renderer's own stems.

    Byte-for-byte against a fresh `generate_prompt_rows` render of the same game and split, keyed
    by prompt_id. A source that fails here was built by an older renderer (or edited), and the
    inserted-paragraph identity this builder asserts downstream would then be comparing against
    the wrong baseline text.
    """
    gradings = sorted({str(row["grading"]) for row in rows})
    if len(gradings) != 1:
        raise ValueError(f"source corpus mixes gradings {gradings}; one selected corpus expected.")
    generated = {
        str(row["prompt_id"]): str(row["prompt"])
        for row in generate_prompt_rows(SOURCE_GAME_ID, gradings[0], split="train")
    }
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id not in generated:
            raise ValueError(
                f"source row {prompt_id!r} is not a {SOURCE_GAME_ID!r} training stem the current "
                f"renderer produces; the roster and this corpus disagree."
            )
        if str(row["prompt"]) != generated[prompt_id]:
            raise ValueError(
                f"source row {prompt_id!r} differs from the current renderer's own text, so the "
                f"stems are not the banked ones and every inserted-paragraph identity check "
                f"downstream would compare against the wrong baseline."
            )


def _assert_honest_clause(prompt: str, *, match_percent: int, prompt_id: str) -> None:
    """Assert the rendered prompt states exactly the percent the reward column carries."""
    stated = _CLAUSE_PERCENT_RE.findall(prompt)
    if len(stated) != 1 or int(stated[0]) != match_percent:
        raise ValueError(
            f"{prompt_id!r} renders clause percents {stated} against reward percent "
            f"{match_percent}: the stated rate and the graded rate must be one number, and this "
            f"prompt breaks that."
        )


def _build_one_cell(
    source: dict[str, Any], *, percent: int, spec: MatrixGameSpec
) -> dict[str, Any]:
    """Render and verify one (stem, rung) cell, returning its finished corpus row."""
    coop_index = _coop_label_index(source)
    print_order = str(source["label_print_order"])
    suffix = "" if print_order == LABEL_PRINT_ORDER_CANONICAL else f"--{print_order}"
    rendered, roster_spec = render_track_record_prompt(
        source_game_id=SOURCE_GAME_ID,
        scenario_id=str(source["reskin_id"]),
        payoff_variant=str(source[PAYOFF_VARIANT_COLUMN]),
        coop_label_index=coop_index,
        label_print_order=print_order,
        match_percent=percent,
    )
    if (
        roster_spec.payoff_cc,
        roster_spec.payoff_cd,
        roster_spec.payoff_dc,
        roster_spec.payoff_dd,
    ) != (spec.payoff_cc, spec.payoff_cd, spec.payoff_dc, spec.payoff_dd):
        raise ValueError(
            f"roster cells for {source['prompt_id']!r} differ from the source row's own "
            f"payoff columns; the corpus and the roster have drifted apart."
        )
    prompt_id = (
        f"{TRACK_RECORD_GAME_ID}--{source['reskin_id']}"
        f"--{source[PAYOFF_VARIANT_COLUMN]}--p{percent}{suffix}--coop{coop_index}"
    )
    _assert_honest_clause(rendered, match_percent=percent, prompt_id=prompt_id)
    assert_counterpart_paragraph_is_the_only_insertion(
        stem=str(source["prompt"]), rendered=rendered, prompt_id=prompt_id
    )
    row = dict(source)
    row["prompt"] = rendered
    row["prompt_id"] = prompt_id
    row["game_id"] = TRACK_RECORD_GAME_ID
    row["grading"] = GRADING_VS_STATED_MATCH
    row["stated_match_prob"] = percent / MATCH_PERCENT_SCALE
    return row


def _build_stem(
    source: dict[str, Any], *, percents: Sequence[int], min_ev_margin: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float], float]:
    """Build one stem's kept cells, returning (rows, dropped cells, per-rung margins, crossover).

    Raises on a stem the design cannot express on: no interior crossover (the gap holds one sign
    or never moves), a cell landing exactly on the crossover despite the margin floor (broken
    audit arithmetic), or a margin audit that leaves no rung on one side (the fixed-incentive
    degeneration the two-sided requirement exists to refuse).
    """
    spec = _row_spec(source)
    crossover = stated_match_crossover(spec)
    if crossover is None or not 0.0 < crossover < 1.0:
        raise ValueError(
            f"source row {source['prompt_id']!r} has no interior EV crossover ({crossover=}), "
            f"so no p mixture can make its optimal action vary and the whole design cannot "
            f"express on it."
        )
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    stem_margins: dict[str, float] = {}
    sides = {"C": 0, "D": 0}
    for percent in percents:
        probability = percent / MATCH_PERCENT_SCALE
        margin = stated_match_gap(spec, probability)
        stem_margins[f"p{percent}"] = margin
        if abs(margin) < min_ev_margin:
            dropped.append(
                {
                    "source_prompt_id": source["prompt_id"],
                    "match_percent": percent,
                    "ev_margin": margin,
                    "reason": f"|EV gap| {abs(margin):.4f} under the {min_ev_margin} floor",
                }
            )
            continue
        row = _build_one_cell(source, percent=percent, spec=spec)
        optimal = stated_match_optimal_action(spec, probability)
        if optimal is None:
            raise ValueError(
                f"{row['prompt_id']!r} sits exactly on its crossover yet passed the margin "
                f"floor; the audit arithmetic is broken."
            )
        sides[optimal] += 1
        rows.append(row)
    one_sided = [side for side, count in sides.items() if count == 0]
    if one_sided:
        raise ValueError(
            f"stem {source['prompt_id']!r} keeps no rung on the "
            f"{'cooperation' if 'C' in one_sided else 'defection'}-optimal side of its "
            f"crossover ({crossover:.4f}) after the margin audit ({sides=}): on this stem the "
            f"stated track record never changes the answer, which is the fixed-incentive "
            f"design this arm exists to avoid. Re-rung the grid."
        )
    return rows, dropped, stem_margins, crossover


def build_track_record_rows(
    source_rows: Sequence[dict[str, Any]],
    *,
    percents: Sequence[int] = TRACK_RECORD_TRAINING_PERCENTS,
    min_ev_margin: float = MIN_EV_MARGIN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cross the source stems with the stated-p grid, audit every cell, and return rows + audit.

    The audit dict is the launch record's evidence: per-variant crossovers, per-cell margins,
    which cells were dropped and why, and the per-stem side counts the two-sided requirement was
    checked against. Returned rather than only logged, so the CLI can write it beside the corpus
    and a reader of the artifact does not have to re-derive which cells exist.
    """
    if not source_rows:
        raise ValueError("source corpus holds no rows; there is nothing to build from.")
    _assert_source_stem_identity(source_rows)

    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    margins: dict[str, dict[str, float]] = {}
    crossovers: dict[str, float] = {}
    for source in source_rows:
        stem_rows, stem_dropped, stem_margins, crossover = _build_stem(
            source, percents=percents, min_ev_margin=min_ev_margin
        )
        rows.extend(stem_rows)
        dropped.extend(stem_dropped)
        margins[str(source["prompt_id"])] = stem_margins
        crossovers[str(source[PAYOFF_VARIANT_COLUMN])] = crossover

    prompt_ids = [str(row["prompt_id"]) for row in rows]
    duplicates = sorted({pid for pid in prompt_ids if prompt_ids.count(pid) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate prompt_ids {duplicates}; the reward groups completions by prompt_id, so a "
            f"collision would mix two prompts' groups."
        )
    audit: dict[str, Any] = {
        "source_game_id": SOURCE_GAME_ID,
        "n_source_rows": len(source_rows),
        "n_rows": len(rows),
        "percents": list(percents),
        "min_ev_margin": min_ev_margin,
        "crossover_by_variant": crossovers,
        "ev_margin_by_stem": margins,
        "dropped_cells": dropped,
        "coop_optimal_rows": sum(
            1
            for row in rows
            if stated_match_optimal_action(_row_spec(row), float(row["stated_match_prob"]))
            == COOPERATE
        ),
    }
    audit["defect_optimal_rows"] = int(audit["n_rows"]) - int(audit["coop_optimal_rows"])
    logger.info(
        "built track-record corpus, %s",
        f"n_rows={audit['n_rows']} coop_optimal={audit['coop_optimal_rows']} "
        f"defect_optimal={audit['defect_optimal_rows']} dropped={len(dropped)} "
        f"crossovers={crossovers}",
    )
    return rows, audit


@dataclass(frozen=True)
class TrackRecordV2Slot:
    """One crossover slot of the v2 chain grid: a shared p*, its rung pair, its table shapes.

    A rung of None drops that side of the slot -- no production slot does this; it exists so the
    balance audits can be driven red in tests without hand-forging rows.
    """

    name: str
    defect_rung: int | None
    coop_rung: int | None
    table_variants: tuple[str, ...]


# The v2 chain grid. Three slots because at TARGET_MARGIN 0.10 with integer rungs strictly
# inside (50, 100) exactly three margin-intervals fit disjointly (each spans p*·(1-m) to
# p*·(1+m), ratio 1.222, and 1.222^4 overflows the 0.51-0.99 box), which also makes
# 0.5 + 0.5/3 the FLOOR for any monotone rate-threshold policy over the corpus -- this grid
# achieves it. The chain: adjacent slots SHARE their boundary rung (63 carries the low slot's
# cooperate-optimal cells and the mid slot's defect-optimal cells; 79 likewise mid/high), so
# both interior rungs are exactly 50/50 and even a per-rung rate-memorizer -- the non-monotone
# policy a threshold audit alone would miss -- is capped. Rungs 51 and 99 are the anchors:
# provably one-sided (a rung below every table's crossover is defect-optimal everywhere), and
# provably unavoidable (the lowest-p* slot needs a defect cell BELOW its own crossover, which
# is below every crossover; symmetrically at the top). Their weight is trimmed instead.
TRACK_RECORD_V2_GRID: tuple[TrackRecordV2Slot, ...] = (
    TrackRecordV2Slot("low", 51, 63, ("xover57-floor6", "xover57-floor15", "xover57-floor24")),
    TrackRecordV2Slot("mid", 63, 79, ("temptation-2", "xover71-floor35", "xover71-floor50")),
    TrackRecordV2Slot("high", 79, 99, ("xover89-floor15", "xover89-floor45", "xover89-floor72")),
)

# How many stems carry each cell: anchors lighter than interior cells, which trims the anchor
# rungs' row mass to 0.25 and buys the rate-lookup audit headroom (0.625 achieved vs the 0.70
# gate). Capped at the number of distinct stems at build time, since a cell assigned twice to
# one stem would collide on prompt_id.
TRACK_RECORD_V2_INTERIOR_WEIGHT = 9
TRACK_RECORD_V2_ANCHOR_WEIGHT = 6

# Gates on the built corpus, each enforced by `_audit_track_record_v2_balance`. The lookup gate
# is the primary anti-shortcut criterion: the best arbitrary map from stated rate to action (a
# strict superset of every monotone threshold) must not beat 0.70 against the EV-optimal
# actions. Per-table balance bounds the best rate-blind policy by the same 0.70. The margin
# tolerance covers display quantization only (per-cell error <= 5e-5 payoff units, so the gap
# moves by <= ~1e-4).
TRACK_RECORD_V2_MAX_RATE_LOOKUP_ACCURACY = 0.70
TRACK_RECORD_V2_TABLE_BALANCE_BAND = (0.30, 0.70)
TRACK_RECORD_V2_MAX_ANCHOR_MASS = 0.35
TRACK_RECORD_V2_OVERALL_BAND = (0.45, 0.55)
TRACK_RECORD_V2_MARGIN_TOLERANCE = 5e-4


@dataclass(frozen=True)
class _TrackRecordV2Cell:
    """One audited (table, rung) cell: the displayed spec and everything the audit records."""

    payoff_variant: str
    slot: str
    rung: int
    optimal: str
    displayed: MatrixGameSpec
    raw_margin: float
    scale: float
    realized_margin: float
    weight: int


def _stem_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Name what a v2 stem is: the axes the clause-free render actually depends on.

    The source payoff variant is deliberately absent: the v2 render substitutes the cell's own
    table, so two source rows differing only in their variant (the real corpus holds two such
    reskins) would render byte-identical v2 prompts and collide on prompt_id.
    """
    return (str(row["reskin_id"]), str(row["coop_label"]), str(row["label_print_order"]))


def _quantized_to_display(spec: MatrixGameSpec) -> MatrixGameSpec:
    """Snap all four cells onto the display grid, so the reward columns ARE the printed points."""
    return MatrixGameSpec(
        game_id=spec.game_id,
        payoff_cc=quantize_payoff_to_display(spec.payoff_cc),
        payoff_cd=quantize_payoff_to_display(spec.payoff_cd),
        payoff_dc=quantize_payoff_to_display(spec.payoff_dc),
        payoff_dd=quantize_payoff_to_display(spec.payoff_dd),
    )


def _build_v2_cell(  # noqa: PLR0913 - one keyword per axis of the cell being built
    *, payoff_variant: str, slot: str, rung: int, side: str, target_margin: float, weight: int
) -> _TrackRecordV2Cell:
    """Rescale, quantize and audit one (table, rung) cell of the grid.

    Every refusal here is a rigged-table symptom: a rung too near its slot's crossover (needs
    scale k > 1), a quantized table off its margin, a quantization that flipped which side pays
    or broke the PD ordering, or a grid claiming a side its own arithmetic contradicts.
    """
    raw_spec = track_record_v2_table(payoff_variant)
    probability = rung / MATCH_PERCENT_SCALE
    raw_margin = stated_match_gap(raw_spec, probability)
    scaled = scaled_stated_match_spec(raw_spec, probability, target_margin=target_margin)
    displayed = _quantized_to_display(scaled)
    assert_prisoners_dilemma(displayed)
    realized = stated_match_gap(displayed, probability)
    if abs(abs(realized) - target_margin) > TRACK_RECORD_V2_MARGIN_TOLERANCE:
        raise ValueError(
            f"{payoff_variant!r} at p{rung}: displayed-table margin {realized:.6f} misses the "
            f"{target_margin} target beyond quantization tolerance "
            f"{TRACK_RECORD_V2_MARGIN_TOLERANCE}; the margin balance this corpus exists for "
            f"does not hold on the numbers the model actually reads."
        )
    optimal = stated_match_optimal_action(displayed, probability)
    if optimal is None or optimal != side:
        raise ValueError(
            f"{payoff_variant!r} at p{rung}: the grid places this cell on the "
            f"{side}-optimal side but the displayed table's own arithmetic says "
            f"{optimal}; the grid and the cells have drifted apart."
        )
    return _TrackRecordV2Cell(
        payoff_variant=payoff_variant,
        slot=slot,
        rung=rung,
        optimal=optimal,
        displayed=displayed,
        raw_margin=raw_margin,
        scale=target_margin / abs(raw_margin),
        realized_margin=realized,
        weight=weight,
    )


def _grid_cells(
    grid: Sequence[TrackRecordV2Slot],
    *,
    target_margin: float,
    interior_weight: int,
    anchor_weight: int,
) -> tuple[list[_TrackRecordV2Cell], tuple[int, ...]]:
    """Expand the grid into audited cells, deriving which rungs are the one-sided anchors."""
    crossovers = [
        stated_match_crossover(track_record_v2_table(variant))
        for slot in grid
        for variant in slot.table_variants
    ]
    interior = [c for c in crossovers if c is not None]
    if len(interior) != len(crossovers) or not interior:
        raise ValueError("every v2 table must carry an interior crossover; the grid is corrupt.")
    anchor_rungs = tuple(
        sorted(
            {
                rung
                for slot in grid
                for rung in (slot.defect_rung, slot.coop_rung)
                if rung is not None
                and not min(interior) < rung / MATCH_PERCENT_SCALE < max(interior)
            }
        )
    )
    cells = [
        _build_v2_cell(
            payoff_variant=variant,
            slot=slot.name,
            rung=rung,
            side=side,
            target_margin=target_margin,
            weight=anchor_weight if rung in anchor_rungs else interior_weight,
        )
        for slot in grid
        for variant in slot.table_variants
        for rung, side in ((slot.defect_rung, DEFECT), (slot.coop_rung, COOPERATE))
        if rung is not None
    ]
    if not cells:
        raise ValueError("the grid expanded to no cells; nothing to build.")
    return cells, anchor_rungs


def _build_v2_row(source: dict[str, Any], cell: _TrackRecordV2Cell) -> dict[str, Any]:
    """Render and verify one stem's copy of one cell, returning its finished corpus row.

    Three checks per row, mirroring v1 with the byte-identity relaxed exactly one notch: the
    clause states the reward's own percent; the prompt is the RESCALED stem plus exactly one
    counterpart paragraph; and the rescaled stem differs from the roster's original stem in the
    outcome block alone, so nothing about the frame, labels or instruction moved.
    """
    coop_index = _coop_label_index(source)
    print_order = str(source["label_print_order"])
    suffix = "" if print_order == LABEL_PRINT_ORDER_CANONICAL else f"--{print_order}"
    rendered, _ = render_track_record_prompt(
        source_game_id=SOURCE_GAME_ID,
        scenario_id=str(source["reskin_id"]),
        payoff_variant=str(source[PAYOFF_VARIANT_COLUMN]),
        coop_label_index=coop_index,
        label_print_order=print_order,
        match_percent=cell.rung,
        spec_override=cell.displayed,
    )
    prompt_id = (
        f"{TRACK_RECORD_V2_GAME_ID}--{source['reskin_id']}"
        f"--{cell.payoff_variant}--p{cell.rung}{suffix}--coop{coop_index}"
    )
    _assert_honest_clause(rendered, match_percent=cell.rung, prompt_id=prompt_id)
    rescaled_stem, _ = render_track_record_stem(
        source_game_id=SOURCE_GAME_ID,
        scenario_id=str(source["reskin_id"]),
        payoff_variant=str(source[PAYOFF_VARIANT_COLUMN]),
        coop_label_index=coop_index,
        label_print_order=print_order,
        spec_override=cell.displayed,
    )
    assert_counterpart_paragraph_is_the_only_insertion(
        stem=rescaled_stem, rendered=rendered, prompt_id=prompt_id
    )
    _assert_only_outcome_block_moved(
        rescaled_stem=rescaled_stem, original_stem=str(source["prompt"]), prompt_id=prompt_id
    )
    row = dict(source)
    row["prompt"] = rendered
    row["prompt_id"] = prompt_id
    row["game_id"] = TRACK_RECORD_V2_GAME_ID
    row["grading"] = GRADING_VS_STATED_MATCH
    row["stated_match_prob"] = cell.rung / MATCH_PERCENT_SCALE
    row["payoff_cc"] = cell.displayed.payoff_cc
    row["payoff_cd"] = cell.displayed.payoff_cd
    row["payoff_dc"] = cell.displayed.payoff_dc
    row["payoff_dd"] = cell.displayed.payoff_dd
    row[PAYOFF_VARIANT_COLUMN] = cell.payoff_variant
    return row


_CREDITED_POINTS_RE = re.compile(r"credited \S+ points")


def _assert_only_outcome_block_moved(
    *, rescaled_stem: str, original_stem: str, prompt_id: str
) -> None:
    """Assert the rescaled stem is the banked stem with only its four point values changed.

    The outcome block is found by content (its "credited N points" lines) rather than by
    position, because a frame's own prose may contain paragraph breaks and shift the section
    index. At most one section may differ, it must be the outcome block on both sides, and
    replacing the point values with a placeholder must make the two sides byte-identical.
    """
    ours = rescaled_stem.split("\n\n")
    theirs = original_stem.split("\n\n")
    differing = (
        [index for index, (a, b) in enumerate(zip(ours, theirs, strict=True)) if a != b]
        if len(ours) == len(theirs)
        else None
    )
    only_points_moved = differing is not None and all(
        _CREDITED_POINTS_RE.search(ours[index])
        and _CREDITED_POINTS_RE.search(theirs[index])
        and _CREDITED_POINTS_RE.sub("credited N points", ours[index])
        == _CREDITED_POINTS_RE.sub("credited N points", theirs[index])
        for index in differing
    )
    if differing is None or len(differing) > 1 or not only_points_moved:
        raise ValueError(
            f"{prompt_id!r}: the rescaled stem does not reduce to the banked source stem plus "
            f"new point values in the outcome block alone (sections differing: {differing}); "
            f"something about the frame, labels or instruction moved, so the stems are no "
            f"longer the ones whose baseline behaviour is banked."
        )


def _best_rate_lookup_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    """Score the best table-blind policy: any map from stated rate to one action."""
    by_rung: dict[float, list[str]] = {}
    for row in rows:
        spec = _row_spec(row)
        optimal = stated_match_optimal_action(spec, float(row["stated_match_prob"]))
        if optimal is None:
            raise ValueError(f"{row['prompt_id']!r} sits exactly on its crossover; audit broken.")
        by_rung.setdefault(float(row["stated_match_prob"]), []).append(optimal)
    correct = sum(max(sides.count(COOPERATE), sides.count(DEFECT)) for sides in by_rung.values())
    return correct / sum(len(sides) for sides in by_rung.values())


def _audit_track_record_v2_balance(
    rows: Sequence[dict[str, Any]], *, anchor_rungs: tuple[int, ...]
) -> dict[str, Any]:
    """Recompute every balance gate from the rows' own reward columns, raising on a violation.

    Deliberately computed from the ROWS rather than from the grid constants, so a builder that
    wrote different cells than it audited fails here rather than shipping.
    """
    sides: dict[str, str] = {}
    for row in rows:
        optimal = stated_match_optimal_action(_row_spec(row), float(row["stated_match_prob"]))
        if optimal is None:
            raise ValueError(f"{row['prompt_id']!r} sits exactly on its crossover; audit broken.")
        sides[str(row["prompt_id"])] = optimal

    coop_by_table: dict[str, float] = {}
    for variant in sorted({str(row[PAYOFF_VARIANT_COLUMN]) for row in rows}):
        table_sides = [
            sides[str(row["prompt_id"])]
            for row in rows
            if str(row[PAYOFF_VARIANT_COLUMN]) == variant
        ]
        fraction = table_sides.count(COOPERATE) / len(table_sides)
        low, high = TRACK_RECORD_V2_TABLE_BALANCE_BAND
        if not low <= fraction <= high:
            raise ValueError(
                f"table {variant!r} is {fraction:.2f} cooperation-optimal, outside "
                f"[{low}, {high}]: on this table one side dominates, so the table alone "
                f"nearly predicts the answer -- the rate-blind shortcut."
            )
        coop_by_table[variant] = fraction

    lookup = _best_rate_lookup_accuracy(rows)
    if lookup > TRACK_RECORD_V2_MAX_RATE_LOOKUP_ACCURACY:
        raise ValueError(
            f"the best rate-lookup policy scores {lookup:.4f} > "
            f"{TRACK_RECORD_V2_MAX_RATE_LOOKUP_ACCURACY} on this corpus: the stated rate alone "
            f"nearly decides the optimal action, which is the shortcut this grid exists to "
            f"make unlearnable. Re-balance the rung sides."
        )

    coop_rows = sum(1 for side in sides.values() if side == COOPERATE)
    overall = coop_rows / len(rows)
    low, high = TRACK_RECORD_V2_OVERALL_BAND
    if not low <= overall <= high:
        raise ValueError(
            f"the corpus is {overall:.3f} cooperation-optimal overall, outside [{low}, {high}]."
        )

    anchor_mass = sum(
        1
        for row in rows
        if round(float(row["stated_match_prob"]) * MATCH_PERCENT_SCALE) in anchor_rungs
    ) / len(rows)
    if anchor_mass > TRACK_RECORD_V2_MAX_ANCHOR_MASS:
        raise ValueError(
            f"the one-sided anchor rungs carry {anchor_mass:.3f} of row mass > "
            f"{TRACK_RECORD_V2_MAX_ANCHOR_MASS}; they exist only to give the extreme slots "
            f"their minority side and must stay light."
        )

    both_sides = {COOPERATE, DEFECT}
    one_sided_stems = sorted(
        {
            _stem_key(row)
            for row in rows
            if {sides[str(r["prompt_id"])] for r in rows if _stem_key(r) == _stem_key(row)}
            != both_sides
        }
    )
    if one_sided_stems:
        raise ValueError(
            f"stems {one_sided_stems} carry cells on only one side of their crossovers: on those "
            f"frames the stated track record never changes the answer, the fixed-incentive "
            f"degeneration the assignment must not produce."
        )

    return {
        "best_rate_lookup_accuracy": lookup,
        "coop_fraction_by_table": coop_by_table,
        "coop_optimal_rows": coop_rows,
        "defect_optimal_rows": len(rows) - coop_rows,
        "anchor_row_mass": anchor_mass,
    }


def build_track_record_v2_rows(
    source_rows: Sequence[dict[str, Any]],
    *,
    grid: Sequence[TrackRecordV2Slot] = TRACK_RECORD_V2_GRID,
    target_margin: float = TRACK_RECORD_V2_TARGET_MARGIN,
    interior_weight: int = TRACK_RECORD_V2_INTERIOR_WEIGHT,
    anchor_weight: int = TRACK_RECORD_V2_ANCHOR_WEIGHT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the margin-balanced anti-shortcut v2 corpus from the banked pd-unstated stems.

    v2's two changes over `build_track_record_rows`, both audited here: every cell's displayed
    (quantized) table carries the SAME |EV margin| at its stated rate, killing v1's measured
    ~9x defect-side penalty asymmetry; and the (table, rate) grid is chained so that neither
    the rate alone (rate-lookup audit) nor the table alone (per-table balance) predicts the
    optimal action. The reward reads the row's payoff columns, which are exactly the printed
    point values -- informational honesty now covers the table as well as the clause.
    """
    if not source_rows:
        raise ValueError("source corpus holds no rows; there is nothing to build from.")
    _assert_source_stem_identity(source_rows)
    stems: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in source_rows:
        stems.setdefault(_stem_key(source), source)
    stem_list = list(stems.values())

    cells, anchor_rungs = _grid_cells(
        grid,
        target_margin=target_margin,
        interior_weight=interior_weight,
        anchor_weight=anchor_weight,
    )

    # One round-robin cursor PER SIDE: dealing both sides through one cursor let a stem land on
    # cells of one side only (caught red on the real 16-stem source, 2026-09-01 -- two stems drew
    # nothing but one side and the per-stem audit refused the corpus). With a cursor per side,
    # every stem receives floor-or-ceiling of (side rows / stems) from EACH side, so per-stem
    # two-sidedness holds by construction whenever each side carries at least one row per stem.
    rows: list[dict[str, Any]] = []
    side_cursors = {COOPERATE: 0, DEFECT: 0}
    for cell in cells:
        for _ in range(min(cell.weight, len(stem_list))):
            cursor = side_cursors[cell.optimal]
            rows.append(_build_v2_row(stem_list[cursor % len(stem_list)], cell))
            side_cursors[cell.optimal] = cursor + 1

    prompt_ids = [str(row["prompt_id"]) for row in rows]
    duplicates = sorted({pid for pid in prompt_ids if prompt_ids.count(pid) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate prompt_ids {duplicates}; the reward groups completions by prompt_id, so "
            f"a collision would mix two prompts' groups."
        )

    balance = _audit_track_record_v2_balance(rows, anchor_rungs=anchor_rungs)
    audit: dict[str, Any] = {
        "source_game_id": SOURCE_GAME_ID,
        "game_id": TRACK_RECORD_V2_GAME_ID,
        "n_source_rows": len(source_rows),
        "n_stems": len(stem_list),
        "n_rows": len(rows),
        "n_cells": len(cells),
        "target_margin": target_margin,
        "margin_tolerance": TRACK_RECORD_V2_MARGIN_TOLERANCE,
        "anchor_rungs": list(anchor_rungs),
        "interior_weight": interior_weight,
        "anchor_weight": anchor_weight,
        "crossover_by_variant": {
            variant: stated_match_crossover(track_record_v2_table(variant))
            for slot in grid
            for variant in slot.table_variants
        },
        "cells": [
            {
                "payoff_variant": cell.payoff_variant,
                "slot": cell.slot,
                "rung": cell.rung,
                "optimal": cell.optimal,
                "raw_margin": cell.raw_margin,
                "scale": cell.scale,
                "realized_margin": cell.realized_margin,
                "weight": min(cell.weight, len(stem_list)),
                "displayed_cells": [
                    cell.displayed.payoff_cc,
                    cell.displayed.payoff_cd,
                    cell.displayed.payoff_dc,
                    cell.displayed.payoff_dd,
                ],
            }
            for cell in cells
        ],
        **balance,
    }
    logger.info(
        "built track-record-v2 corpus, %s",
        f"n_rows={audit['n_rows']} n_stems={audit['n_stems']} n_cells={audit['n_cells']} "
        f"coop_optimal={audit['coop_optimal_rows']} defect_optimal={audit['defect_optimal_rows']} "
        f"rate_lookup={audit['best_rate_lookup_accuracy']:.4f} "
        f"anchor_mass={audit['anchor_row_mass']:.4f}",
    )
    return rows, audit


def main(argv: Sequence[str] | None = None) -> None:
    """Build the corpus named on the command line, writing the audit record beside it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="the selected pd-unstated corpus jsonl")
    parser.add_argument("--out", required=True, help="where the pd-track-record corpus lands")
    parser.add_argument(
        "--grid",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "v1: the banked 88-row single-table-per-stem build. v2: the margin-balanced "
            "anti-shortcut chain grid (game_id pd-track-record-v2)."
        ),
    )
    args = parser.parse_args(argv)

    source_rows = [
        json.loads(line) for line in Path(args.source).read_text().splitlines() if line.strip()
    ]
    builder = build_track_record_v2_rows if args.grid == "v2" else build_track_record_rows
    rows, audit = builder(source_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d rows) and %s", out_path, len(rows), audit_path)


if __name__ == "__main__":
    main()
