"""The decoupled-ladder plan: exact record counts, key identity, rendering, and the prompt audit.

Two of these are the pass's load-bearing gates rather than arithmetic checks. The audit asserts that
every cell's prompt is its clause-free stem plus exactly ONE counterpart paragraph, which is what
makes a movement between cells attributable to the clause -- so it is also exercised with a
two-paragraph clause planted, and required to go red. And the Different x Coupled corner is asserted
against its own documented derivation from the two public constants, because a corner that drifted
from either would silently be a third condition rather than the crossing of the other two.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from games.prompts import (
    COUNTERPART_FRAMING_IDS,
    COUNTERPART_FRAMINGS,
    DECIDES_IN_STEP_SENTENCE,
    DECIDES_INDEPENDENTLY_SENTENCE,
    DIFFERENT_AI_COUNTERPART_CLAUSE,
    DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE,
    FRAMING_DIFFERENT_AI_COUPLED,
    LABEL_PRINT_ORDERS,
    SPLIT_EVAL,
    TWIN_COUNTERPART_CLAUSE,
    generate_counterpart_clause_prompt_rows,
    generate_framing_prompt_rows,
)
from sociology.decoupled_plan import (
    ANCHOR_CELLS,
    BLOCK_ANCHOR,
    BLOCK_DELIBERATION,
    BLOCK_FLOOR,
    BLOCK_LADDER,
    DELIBERATION_MODEL_IDS,
    DESIGN_LABELS,
    DRAWS,
    FLOOR_CELLS,
    GAMES,
    GRADING,
    LADDER_BLOCK_CELLS,
    LADDER_RUNGS,
    LIVE_MODEL_IDS,
    LUNA_MODEL_ID,
    ON_DEMAND_LADDER_MODEL_IDS,
    OPUS_MODEL_ID,
    ROWS_PER_CELL,
    RUNGS_THE_ANCHOR_ALREADY_SAMPLES,
    SITTING_A,
    SONNET_MODEL_ID,
    TRANSPORT_BATCH,
    TRANSPORT_LIVE,
    UNREADABLE_MODEL_IDS,
    Leg,
    all_legs,
    audit_planned_calls,
    leg_for,
    on_demand_legs,
    planned_calls_for_leg,
    production_legs,
    record_key,
    refuse_below_batch_floor,
    transport_for,
    unstated_stem_id,
)
from sociology.decoupled_stimulus import (
    RUNG_DIFFERENT_FAMILY,
    RUNG_SAME_CHECKPOINT,
    clause_for,
)

if TYPE_CHECKING:
    from sociology.decoupled_stimulus import DecoupledStimulus


def replace_clause(stimulus: DecoupledStimulus, rung: str, clause: str) -> DecoupledStimulus:
    """Return a copy of the loaded stimulus with one rung's clause swapped, for sabotage tests."""
    return replace(stimulus, ladder_clauses={**stimulus.ladder_clauses, rung: clause})


ANCHOR_RECORDS = 1_920
FLOOR_RECORDS = 768
LADDER_RECORDS = 1_920
PRODUCTION_RECORDS = 41_472
"""Nine rows of anchor and floor (Opus out, Sonnet in), two deliberation legs, seven cheap ladders."""
ON_DEMAND_RECORDS = 3 * LADDER_RECORDS + ANCHOR_RECORDS + FLOOR_RECORDS
"""Three band-gated ladders (Opus, Luna, Sonnet) plus Opus 5's anchor and floor: 8,448."""
ALL_RECORDS = PRODUCTION_RECORDS + ON_DEMAND_RECORDS
"""49,920: the earlier 45,312 plus Sonnet 5's 2,688 production records and its 1,920-record ladder."""


class TestDifferentAiCoupledCorner:
    def test_it_equals_its_documented_derivation(self) -> None:
        assert (
            DIFFERENT_AI_COUNTERPART_CLAUSE.replace(
                DECIDES_INDEPENDENTLY_SENTENCE, DECIDES_IN_STEP_SENTENCE
            )
            == DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE
        )

    def test_it_differs_from_different_ai_in_the_route_sentence_alone(self) -> None:
        assert DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE.endswith(DECIDES_IN_STEP_SENTENCE)
        assert DIFFERENT_AI_COUNTERPART_CLAUSE.endswith(DECIDES_INDEPENDENTLY_SENTENCE)
        assert DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE.removesuffix(
            DECIDES_IN_STEP_SENTENCE
        ) == DIFFERENT_AI_COUNTERPART_CLAUSE.removesuffix(DECIDES_INDEPENDENTLY_SENTENCE)

    def test_it_differs_from_twin_in_the_identity_fragment_alone(self) -> None:
        """The two coupled cells must state the coupling in one wording and differ only in identity."""
        assert TWIN_COUNTERPART_CLAUSE.endswith(DECIDES_IN_STEP_SENTENCE)
        coupled_identity = DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE.removesuffix(
            DECIDES_IN_STEP_SENTENCE
        )
        twin_identity = TWIN_COUNTERPART_CLAUSE.removesuffix(DECIDES_IN_STEP_SENTENCE)
        assert coupled_identity != twin_identity
        assert coupled_identity.endswith("at this same moment, ")
        assert twin_identity.endswith("at this same moment, ")

    def test_it_is_registered_and_reachable_through_the_registry(self) -> None:
        assert FRAMING_DIFFERENT_AI_COUPLED in COUNTERPART_FRAMING_IDS
        assert (
            COUNTERPART_FRAMINGS[FRAMING_DIFFERENT_AI_COUPLED]
            == DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE
        )


class TestRegisteredFramingsRenderIdenticallyThroughBothPaths:
    def test_every_registered_framing_is_byte_identical_via_both_entry_points(self) -> None:
        """`generate_framing_prompt_rows` must be a pure registry lookup over the runtime renderer."""
        for framing_id in COUNTERPART_FRAMING_IDS:
            for order in LABEL_PRINT_ORDERS:
                through_registry = generate_framing_prompt_rows(
                    "twin-pd",
                    GRADING,
                    framing_id=framing_id,
                    split=SPLIT_EVAL,
                    label_print_order=order,
                )
                through_clause = generate_counterpart_clause_prompt_rows(
                    "twin-pd",
                    GRADING,
                    clause=COUNTERPART_FRAMINGS[framing_id],
                    framing_label=framing_id,
                    split=SPLIT_EVAL,
                    label_print_order=order,
                )
                assert through_registry == through_clause, framing_id

    def test_an_unregistered_framing_id_still_refuses_through_the_wrapper(self) -> None:
        with pytest.raises(ValueError, match="Unknown framing_id"):
            generate_framing_prompt_rows(
                "twin-pd", GRADING, framing_id="not-a-framing", split=SPLIT_EVAL
            )

    def test_a_label_that_is_not_a_lowercase_hyphenated_slug_refuses(self) -> None:
        with pytest.raises(ValueError, match="framing_label"):
            generate_counterpart_clause_prompt_rows(
                "twin-pd",
                GRADING,
                clause=TWIN_COUNTERPART_CLAUSE,
                framing_label="Sibling Adapter",
                split=SPLIT_EVAL,
            )


class TestLegTable:
    def test_each_block_plans_the_designs_record_count(self) -> None:
        by_block = {
            BLOCK_ANCHOR: ANCHOR_RECORDS,
            BLOCK_FLOOR: FLOOR_RECORDS,
            BLOCK_DELIBERATION: ANCHOR_RECORDS,
            BLOCK_LADDER: LADDER_RECORDS,
        }
        for leg in all_legs():
            assert leg.records == by_block[leg.block], leg.leg_id

    def test_the_arithmetic_behind_those_counts(self) -> None:
        assert ROWS_PER_CELL == 48
        assert DRAWS == 8
        assert len(ANCHOR_CELLS) == 5
        assert len(FLOOR_CELLS) == 2
        assert len(LADDER_BLOCK_CELLS) == 5
        assert len(ANCHOR_CELLS) * ROWS_PER_CELL * DRAWS == ANCHOR_RECORDS
        assert len(FLOOR_CELLS) * ROWS_PER_CELL * DRAWS == FLOOR_RECORDS
        assert len(LADDER_BLOCK_CELLS) * ROWS_PER_CELL * DRAWS == LADDER_RECORDS

    def test_the_anchor_carries_the_denial_matched_different_identity_cell(self) -> None:
        """The paper's four cells plus `different-family`, which unconfounds identity from the denial."""
        assert ANCHOR_CELLS == (
            "twin",
            "same-weights-uncorrelated",
            "different-ai-coupled",
            "different-ai",
            RUNG_DIFFERENT_FAMILY,
        )

    def test_the_floor_re_sits_both_decoupled_anchor_cells(self) -> None:
        """A floor read only on the near-zero Control would clear any contrast put against it."""
        assert FLOOR_CELLS == ("same-weights-uncorrelated", "different-ai")
        assert set(FLOOR_CELLS) <= set(ANCHOR_CELLS)
        floor_legs = [leg for leg in all_legs() if leg.block == BLOCK_FLOOR]
        assert all(leg.cells == FLOOR_CELLS for leg in floor_legs)
        assert all(leg.sitting == "B" for leg in floor_legs)

    def test_the_deliberation_block_re_runs_every_anchor_cell_at_high_effort(self) -> None:
        legs = [leg for leg in all_legs() if leg.block == BLOCK_DELIBERATION]
        assert {leg.model_id for leg in legs} == set(DELIBERATION_MODEL_IDS)
        assert all(leg.cells == ANCHOR_CELLS for leg in legs)
        assert all(leg.reasoning_effort == "high" for leg in legs)

    def test_the_ladder_is_seven_rungs_split_between_the_anchor_and_the_ladder_block(self) -> None:
        """Two rungs are already sampled by the anchor, so the ladder block plans the other five."""
        assert LADDER_RUNGS == (
            RUNG_SAME_CHECKPOINT,
            "sibling-adapter",
            "same-family-larger",
            "same-family-smaller",
            RUNG_DIFFERENT_FAMILY,
            "same-task-different-family",
            "person",
        )
        assert LADDER_BLOCK_CELLS == (
            "sibling-adapter",
            "same-family-larger",
            "same-family-smaller",
            "same-task-different-family",
            "person",
        )
        assert RUNGS_THE_ANCHOR_ALREADY_SAMPLES == (RUNG_SAME_CHECKPOINT, RUNG_DIFFERENT_FAMILY)
        assert not set(LADDER_BLOCK_CELLS) & set(ANCHOR_CELLS)
        assert set(LADDER_BLOCK_CELLS) | set(RUNGS_THE_ANCHOR_ALREADY_SAMPLES) == set(LADDER_RUNGS)

    def test_the_design_labels_cover_every_name_a_judge_must_not_see(self) -> None:
        """The blindness gate reads this list, so a new cell or rung must land in it automatically."""
        for name in (*ANCHOR_CELLS, *LADDER_RUNGS, *GAMES, "coupled", "decoupled"):
            assert name in DESIGN_LABELS, name

    def test_the_production_table_is_nine_models_of_anchor_and_floor_plus_two_more_blocks(
        self,
    ) -> None:
        """Nine readable rows: Opus 5 left the production table on 2026-09-02 and Sonnet 5 joined it."""
        blocks = [leg.block for leg in production_legs()]
        assert blocks.count(BLOCK_ANCHOR) == 9
        assert blocks.count(BLOCK_FLOOR) == 9
        assert blocks.count(BLOCK_DELIBERATION) == 2
        assert blocks.count(BLOCK_LADDER) == 7
        assert sum(leg.records for leg in production_legs()) == PRODUCTION_RECORDS
        production_models = {leg.model_id for leg in production_legs()}
        assert SONNET_MODEL_ID in production_models
        assert OPUS_MODEL_ID not in production_models
        assert not any(leg.on_demand for leg in production_legs())

    def test_the_on_demand_legs_are_three_ladders_plus_every_opus_leg(self) -> None:
        """On demand means "waits on a recorded decision": a band read, or spending on an unread row."""
        assert on_demand_legs() == tuple(leg for leg in all_legs() if leg.on_demand)
        by_id = {leg.leg_id: leg for leg in on_demand_legs()}
        assert set(by_id) == {
            "opus-5--default--A--anchor",
            "opus-5--default--B--floor",
            "opus-5--default--A--ladder",
            "luna--default--A--ladder",
            "sonnet-5--default--A--ladder",
        }
        assert {leg.model_id for leg in on_demand_legs() if leg.block == BLOCK_LADDER} == set(
            ON_DEMAND_LADDER_MODEL_IDS
        )
        assert sum(leg.records for leg in on_demand_legs()) == ON_DEMAND_RECORDS
        assert sum(leg.records for leg in all_legs()) == ALL_RECORDS

    def test_every_opus_leg_is_on_demand_and_keeps_its_on_disk_file_stem(self) -> None:
        """The refused row stays in the table because its reply files exist; nothing re-submits by default."""
        assert UNREADABLE_MODEL_IDS == (OPUS_MODEL_ID,)
        opus_legs = [leg for leg in all_legs() if leg.model_id == OPUS_MODEL_ID]
        assert {leg.block for leg in opus_legs} == {BLOCK_ANCHOR, BLOCK_FLOOR, BLOCK_LADDER}
        assert all(leg.on_demand for leg in opus_legs)
        assert all(leg.transport == TRANSPORT_BATCH for leg in opus_legs)
        assert (
            leg_for("opus-5--default--A--anchor").file_stem
            == "global-anthropic-claude-opus-5--effort-default--sitting-A--anchor"
        )
        assert (
            leg_for("opus-5--default--B--floor").file_stem
            == "global-anthropic-claude-opus-5--effort-default--sitting-B--floor"
        )

    def test_sonnet_5_runs_live_with_anchor_and_floor_by_default_and_a_ladder_on_demand(
        self,
    ) -> None:
        """Luna's leg shape minus the deliberation leg: Sonnet takes no effort knob in this pass."""
        anchor = leg_for("sonnet-5--default--A--anchor")
        floor = leg_for("sonnet-5--default--B--floor")
        ladder = leg_for("sonnet-5--default--A--ladder")
        assert (anchor.model_id, anchor.transport, anchor.on_demand) == (
            SONNET_MODEL_ID,
            TRANSPORT_LIVE,
            False,
        )
        assert (anchor.cells, anchor.sitting, anchor.records) == (ANCHOR_CELLS, "A", ANCHOR_RECORDS)
        assert (floor.cells, floor.sitting, floor.records, floor.on_demand) == (
            FLOOR_CELLS,
            "B",
            FLOOR_RECORDS,
            False,
        )
        assert (ladder.cells, ladder.transport, ladder.on_demand, ladder.records) == (
            LADDER_BLOCK_CELLS,
            TRANSPORT_LIVE,
            True,
            LADDER_RECORDS,
        )
        sonnet_legs = [leg for leg in all_legs() if leg.model_id == SONNET_MODEL_ID]
        assert {leg.block for leg in sonnet_legs} == {BLOCK_ANCHOR, BLOCK_FLOOR, BLOCK_LADDER}
        assert all(leg.reasoning_effort is None for leg in sonnet_legs)
        assert SONNET_MODEL_ID not in DELIBERATION_MODEL_IDS
        assert (
            anchor.file_stem
            == "global-anthropic-claude-sonnet-5--effort-default--sitting-A--anchor"
        )

    def test_the_live_roster_is_luna_and_sonnet_and_everything_else_is_batch(self) -> None:
        assert LIVE_MODEL_IDS == (LUNA_MODEL_ID, SONNET_MODEL_ID)
        for leg in all_legs():
            expected = TRANSPORT_LIVE if leg.model_id in LIVE_MODEL_IDS else TRANSPORT_BATCH
            assert leg.transport == expected == transport_for(leg.model_id), leg.leg_id

    def test_leg_ids_are_unique_and_name_their_effort_and_sitting(self) -> None:
        ids = [leg.leg_id for leg in all_legs()]
        assert len(ids) == len(set(ids))
        assert "gpt-oss-120b--default--A--anchor" in ids
        assert "gpt-oss-120b--high--A--deliberation" in ids
        assert "sonnet-5--default--A--anchor" in ids
        assert "sonnet-5--default--B--floor" in ids

    def test_an_unknown_leg_id_refuses_and_lists_the_table(self) -> None:
        with pytest.raises(ValueError, match="is not a leg of this plan"):
            leg_for("nonexistent--default--A--anchor")

    def test_reply_and_handle_stems_carry_the_full_model_id(self) -> None:
        leg = leg_for("gpt-oss-120b--high--A--deliberation")
        assert leg.file_stem == "openai-gpt-oss-120b-1-0--effort-high--sitting-A--deliberation"


class TestPlannedCalls:
    def test_a_ladder_leg_renders_its_designed_count_with_unique_keys(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = planned_calls_for_leg(
            leg_for("gpt-oss-20b--default--A--ladder"), decoupled_stimulus
        )
        assert len(calls) == LADDER_RECORDS
        assert len({call.key for call in calls}) == LADDER_RECORDS

    def test_an_anchor_leg_renders_its_designed_count_with_unique_keys(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        calls = planned_calls_for_leg(
            leg_for("gpt-oss-20b--default--A--anchor"), decoupled_stimulus
        )
        assert len(calls) == ANCHOR_RECORDS
        assert len({call.key for call in calls}) == ANCHOR_RECORDS

    def test_rebuilding_a_leg_gives_byte_identical_calls_in_the_same_order(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        leg = leg_for("gpt-oss-20b--default--B--floor")
        first = planned_calls_for_leg(leg, decoupled_stimulus)
        second = planned_calls_for_leg(leg, decoupled_stimulus)
        assert first == second

    def test_every_rung_reads_the_same_forty_eight_prompt_rows(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """A rung must vary the clause and nothing else, so all seven read one set of underlying rows."""
        stems_per_rung: dict[str, set[str]] = {}
        for rung in LADDER_RUNGS:
            clause = clause_for(rung, decoupled_stimulus)
            stems: set[str] = set()
            for game_id in GAMES:
                for order in LABEL_PRINT_ORDERS:
                    rows = generate_counterpart_clause_prompt_rows(
                        game_id,
                        GRADING,
                        clause=clause,
                        framing_label=rung,
                        split=SPLIT_EVAL,
                        label_print_order=order,
                    )
                    stems.update(unstated_stem_id(str(row["prompt_id"])) for row in rows)
            stems_per_rung[rung] = stems
        assert all(len(stems) == ROWS_PER_CELL for stems in stems_per_rung.values())
        assert len({frozenset(stems) for stems in stems_per_rung.values()}) == 1

    def test_the_efforts_and_sittings_ride_the_record_key(self) -> None:
        base = {
            "block": BLOCK_ANCHOR,
            "cell": "twin",
            "game_id": "twin-pd",
            "prompt_id": "p",
            "model_id": "m",
        }
        default_a = record_key(**base, reasoning_effort=None, sitting="A", draw=0)
        high_a = record_key(**base, reasoning_effort="high", sitting="A", draw=0)
        default_b = record_key(**base, reasoning_effort=None, sitting="B", draw=0)
        assert "effort=default" in default_a
        assert "effort=high" in high_a
        assert "sitting=B" in default_b
        assert len({default_a, high_a, default_b}) == 3

    def test_the_anchors_different_family_cell_resolves_to_the_authored_rung(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """The fifth anchor cell IS a ladder rung, so both must read one clause rather than two."""
        anchor = planned_calls_for_leg(
            leg_for("gpt-oss-20b--default--A--anchor"), decoupled_stimulus
        )
        assert {call.prompt for call in anchor if call.cell == RUNG_DIFFERENT_FAMILY}
        assert (
            clause_for(RUNG_DIFFERENT_FAMILY, decoupled_stimulus)
            == decoupled_stimulus.ladder_clauses[RUNG_DIFFERENT_FAMILY]
        )

    def test_the_ladders_bottom_rung_shares_the_anchors_decoupled_prompts(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """`same-checkpoint` IS the anchor's decoupled same-weights cell, which is why it is not resampled."""
        anchor = planned_calls_for_leg(
            leg_for("gpt-oss-20b--default--A--anchor"), decoupled_stimulus
        )
        anchor_prompts = {
            call.prompt for call in anchor if call.cell == "same-weights-uncorrelated"
        }
        rung_rows = [
            row
            for game_id in GAMES
            for order in LABEL_PRINT_ORDERS
            for row in generate_counterpart_clause_prompt_rows(
                game_id,
                GRADING,
                clause=clause_for("same-checkpoint", decoupled_stimulus),
                framing_label="same-weights-uncorrelated",
                split=SPLIT_EVAL,
                label_print_order=order,
            )
        ]
        assert {str(row["prompt"]) for row in rung_rows} == anchor_prompts


class TestPromptAudit:
    def test_every_planned_prompt_is_its_stem_plus_one_counterpart_paragraph(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        for leg_id in (
            "gpt-oss-20b--default--A--anchor",
            "gpt-oss-20b--default--B--floor",
            "gpt-oss-20b--default--A--ladder",
        ):
            calls = planned_calls_for_leg(leg_for(leg_id), decoupled_stimulus)
            assert audit_planned_calls(calls) == len(calls)

    def test_a_planted_two_paragraph_clause_makes_the_audit_go_red(
        self, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        """The sabotage: a second paragraph survives the marker-deletion, so the stems stop matching."""
        clean = clause_for("sibling-adapter", decoupled_stimulus)
        planted = clean.replace(", reading a copy", ".\n\nIt is reading a copy", 1)
        leg = Leg(
            "openai.gpt-oss-20b-1:0",
            TRANSPORT_BATCH,
            BLOCK_LADDER,
            ("sibling-adapter",),
            None,
            SITTING_A,
        )
        sabotaged = replace_clause(decoupled_stimulus, "sibling-adapter", planted)
        calls = planned_calls_for_leg(leg, sabotaged)
        with pytest.raises(ValueError, match="one counterpart paragraph"):
            audit_planned_calls(calls)

    def test_a_prompt_id_without_a_framing_segment_refuses(self) -> None:
        with pytest.raises(ValueError, match="framing segments"):
            unstated_stem_id("twin-pd--lagoon-net-length--temptation-2--coop0")


class TestBatchFloor:
    def test_a_leg_under_the_hundred_record_floor_refuses_in_this_clis_own_levers(self) -> None:
        leg = leg_for("gpt-oss-20b--default--B--floor")
        with pytest.raises(ValueError, match="DRAWS"):
            refuse_below_batch_floor(leg, 96)

    def test_a_leg_at_or_over_the_floor_passes(self) -> None:
        refuse_below_batch_floor(leg_for("gpt-oss-20b--default--B--floor"), FLOOR_RECORDS)
