"""The transfer sampling plan: exact record counts, key identity, rendering, and the two prompt audits.

Two of these are the pass's load-bearing gates rather than arithmetic checks. The one-inserted-paragraph
audit asserts that every cell's prompt is its identity-blind stem plus exactly ONE counterpart paragraph,
which is what makes a movement between cells attributable to the clause -- so it is also exercised with a
two-paragraph clause planted and required to go red. And the two games' prompts for one (scenario, dose,
polarity) are asserted to differ in the mechanics paragraph and the counterpart paragraph and in nothing
else, with a drifted closing order planted and required to go red, because a closing order that differed
between the games would confound the coupling contrast with observability and attribution.

The per-leg counts are spelled out as literals here on purpose: they are the design's own arithmetic, and
a plan that quietly sampled 9 cells where the design says 8 would still produce a complete set of
artifacts.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from games.parsing import ANSWER_POLARITIES
from games.prompts import (
    DRAWN_DECISION_TRANSFER_GAME_ID,
    MATCHED_DECISION_TRANSFER_GAME_ID,
    MATCHED_DECISION_TRANSFER_MECHANICS,
    ONE_WAY_TRANSFER_GAME_ID,
    TRANSFER_CLOSING_ORDER,
    TRANSFER_IDENTITY_BLIND_LABEL,
    about_the_other_side,
)
from reward_hacking.bedrock_batch import MAX_JOB_NAME_CHARS, MIN_BATCH_RECORDS
from sociology import decoupled_plan
from sociology.transfer_plan import (
    BATCH_PASS_TOKEN,
    BLOCK_DELIBERATION,
    BLOCK_DOSE_ANCHORS,
    BLOCK_DOSE_FLOOR,
    BLOCK_DOSE_ONE_WAY,
    BLOCK_DOSE_RECORD,
    BLOCK_FINGERPRINT_FLOOR,
    BLOCK_FINGERPRINT_MATCHED,
    BLOCK_FLOOR,
    BLOCK_IDENTITY_MATCHED,
    BLOCK_IDENTITY_ONE_WAY,
    BLOCK_INTERACTION_ONE_WAY,
    BLOCK_KNOCKOUT_DRAWN,
    BLOCK_KNOCKOUT_FLOOR,
    BLOCK_KNOCKOUT_MATCHED,
    BLOCKS,
    BOARD_CONDITIONS,
    CONDITION_OTHER_CONTENT_OWN_WORDS,
    CONDITION_OTHER_RAW,
    CONDITION_OWN_CONTENT_OTHER_WORDS,
    CONDITION_OWN_RAW,
    CORRELATION_DOSE_PLAN,
    DELIBERATION_MODEL_IDS,
    DESIGN_LABELS,
    DESIGN_LABELS_IN_AUTHORED_REPLIES,
    DESIGN_LABELS_THE_PROMPTS_PRINT,
    DRAWS,
    FINGERPRINT_PLAN,
    KNOCKOUT_MODEL_IDS,
    KNOCKOUT_PLAN,
    LIVE_MODEL_IDS,
    LUNA_MODEL_ID,
    ONE_WAY_TRANSFER_PLAN,
    OPUS_MODEL_ID,
    PAIR_MODEL_IDS,
    PLAN_TABLES,
    QWEN_MODEL_ID,
    ROWS_PER_CELL,
    SITTING_A,
    SITTING_B,
    TRANSPORT_BATCH,
    Cell,
    Leg,
    PlanTable,
    audit_games_differ_by_mechanics_and_clause,
    audit_planned_calls,
    blind_stem_id,
    board_condition,
    frame_section_count,
    leg_for,
    matched_rounds_thresholds,
    model_slug,
    plan_table,
    planned_calls_for_leg,
    record_key,
    refuse_below_batch_floor,
    render_cell_row,
    spec_for,
    transport_for,
)
from sociology.transfer_stimulus import (
    BOARD_IDS,
    MATCHED_ROUNDS_LADDER,
    RUNG_DIFFERENT_FAMILY_TRACK_RECORD,
    RUNG_SAME_CHECKPOINT_COUPLED,
    RUNG_SAME_CHECKPOINT_RECORD_0,
    RUNGS_BY_GAME,
    record_rung_id,
)

CELLS_BY_BLOCK = ONE_WAY_TRANSFER_PLAN.cells_by_block
"""The finished pass's block table, under the name the arithmetic tests below were written against."""


def all_legs() -> tuple[Leg, ...]:
    """The finished pass's whole leg table, on-demand rows included."""
    return ONE_WAY_TRANSFER_PLAN.all_legs()


def production_legs() -> tuple[Leg, ...]:
    """The finished pass's legs that run by default."""
    return ONE_WAY_TRANSFER_PLAN.production_legs()


def on_demand_legs() -> tuple[Leg, ...]:
    """The finished pass's legs that wait on a recorded decision."""
    return ONE_WAY_TRANSFER_PLAN.on_demand_legs()


if TYPE_CHECKING:
    from sociology.transfer_stimulus import TransferStimulus

RECORDS_BY_BLOCK: dict[str, int] = {
    BLOCK_IDENTITY_ONE_WAY: 1_024,
    BLOCK_IDENTITY_MATCHED: 1_152,
    BLOCK_DOSE_ONE_WAY: 1_280,
    BLOCK_INTERACTION_ONE_WAY: 1_024,
    BLOCK_FLOOR: 256,
    BLOCK_DELIBERATION: 1_024,
    BLOCK_KNOCKOUT_MATCHED: 512,
    BLOCK_KNOCKOUT_DRAWN: 256,
    BLOCK_KNOCKOUT_FLOOR: 256,
    BLOCK_FINGERPRINT_MATCHED: 768,
    BLOCK_FINGERPRINT_FLOOR: 384,
    BLOCK_DOSE_RECORD: 1_024,
    BLOCK_DOSE_ANCHORS: 640,
    BLOCK_DOSE_FLOOR: 256,
}
"""The design's per-leg counts, spelled out rather than recomputed from the same code under test."""

ALL_CELLS_BY_BLOCK: dict[str, tuple[Cell, ...]] = {
    block: cells for table in PLAN_TABLES.values() for block, cells in table.cells_by_block.items()
}
"""Every pass's blocks in one map, for the checks that are about the arithmetic rather than the pass."""

RECORDS_PER_MODEL = 4_736
"""1,024 + 1,152 + 1,280 + 1,024 + 256, which is what a default roster row costs."""

MODELS = 9
PRODUCTION_RECORDS = MODELS * RECORDS_PER_MODEL + len(DELIBERATION_MODEL_IDS) * 1_024
ON_DEMAND_RECORDS = 1_024 + 1_152
"""Opus 5's two identity legs, which wait on its refusal probe."""

KNOCKOUT_RECORDS_PER_MODEL = 1_024
"""512 + 256 + 256: the twin's four cells, the drawn game's two, and the two re-sat anchors."""

KNOCKOUT_RECORDS = len(KNOCKOUT_MODEL_IDS) * KNOCKOUT_RECORDS_PER_MODEL

GAME_PAIRS = 3
"""Unordered pairs of the three transfer games, which is how many section diffs one row is audited by."""


class TestTheBlockArithmetic:
    @pytest.mark.parametrize(("block", "records"), sorted(RECORDS_BY_BLOCK.items()))
    def test_each_block_plans_the_design_s_record_count(self, block: str, records: int) -> None:
        assert len(ALL_CELLS_BY_BLOCK[block]) * ROWS_PER_CELL * DRAWS == records

    def test_every_block_of_every_pass_has_cells(self) -> None:
        assert set(ALL_CELLS_BY_BLOCK) == set(BLOCKS)
        assert set(RECORDS_BY_BLOCK) == set(BLOCKS)

    def test_rows_per_cell_is_eight_frames_under_both_polarities(self) -> None:
        assert ROWS_PER_CELL == 8 * len(ANSWER_POLARITIES) == 16

    def test_the_identity_blocks_open_with_the_blind_baseline(self) -> None:
        """Every reading here is a contrast against it, so the table opens with what the rest is against."""
        for block in (BLOCK_IDENTITY_ONE_WAY, BLOCK_IDENTITY_MATCHED):
            assert CELLS_BY_BLOCK[block][0].rung is None
            assert CELLS_BY_BLOCK[block][0].cell_id == TRANSFER_IDENTITY_BLIND_LABEL

    def test_the_twin_ladder_carries_the_coupled_cell_and_the_one_way_ladder_does_not(self) -> None:
        twin = {cell.cell_id for cell in CELLS_BY_BLOCK[BLOCK_IDENTITY_MATCHED]}
        one_way = {cell.cell_id for cell in CELLS_BY_BLOCK[BLOCK_IDENTITY_ONE_WAY]}
        assert twin - one_way == {RUNG_SAME_CHECKPOINT_COUPLED}

    @pytest.mark.parametrize("table", list(PLAN_TABLES.values()), ids=list(PLAN_TABLES))
    def test_every_declared_cell_is_sampled_once_in_its_own_pass(self, table: PlanTable) -> None:
        """Per table, because a pass may sample a subset of a game's cells and still be complete.

        The knockout samples three of the twin's nine rungs on purpose; what it may not do is declare a
        cell and never plan it, which is the shape the import-time check refuses.
        """
        for game_id, required in table.required_rungs_by_game.items():
            sampled = [
                cell.cell_id
                for block in table.reading_blocks
                for cell in table.cells_by_block[block]
                if cell.game_id == game_id and cell.spec == spec_for(game_id)
            ]
            for rung in required:
                expected = TRANSFER_IDENTITY_BLIND_LABEL if rung is None else rung
                assert sampled.count(expected) == 1, (game_id, expected)
            assert len(sampled) == len(required)

    def test_the_finished_pass_never_samples_the_cell_the_knockout_added(self) -> None:
        """The first pass's ladder is spelled out rather than derived, and this is why.

        Its numbers were read on nine twin cells. A ladder read off the stimulus registry would have
        grown that block to ten the moment the knockout registered a new twin clause, quietly adding 128
        records per model to a table nobody re-ran.
        """
        planned = {
            cell.cell_id
            for cells in ONE_WAY_TRANSFER_PLAN.cells_by_block.values()
            for cell in cells
        }
        assert RUNG_DIFFERENT_FAMILY_TRACK_RECORD not in planned
        assert RUNG_DIFFERENT_FAMILY_TRACK_RECORD in {
            cell.cell_id for cells in KNOCKOUT_PLAN.cells_by_block.values() for cell in cells
        }

    def test_the_dose_block_omits_the_reference_cell_the_identity_block_already_samples(
        self,
    ) -> None:
        reference = spec_for(ONE_WAY_TRANSFER_GAME_ID)
        assert not [cell for cell in CELLS_BY_BLOCK[BLOCK_DOSE_ONE_WAY] if cell.spec == reference]

    def test_the_interaction_block_sits_at_the_value_destroying_dose(self) -> None:
        """Total return per unit is credit times count, so credit a half at one beneficiary is the floor."""
        for cell in CELLS_BY_BLOCK[BLOCK_INTERACTION_ONE_WAY]:
            assert cell.spec.credit_per_unit * cell.spec.beneficiary_count < 1.0

    def test_the_floor_block_re_sits_both_ends_rather_than_the_cheaper_one(self) -> None:
        cells = CELLS_BY_BLOCK[BLOCK_FLOOR]
        assert {cell.cell_id for cell in cells} == {
            TRANSFER_IDENTITY_BLIND_LABEL,
            "same-checkpoint",
        }


def _bedrock_job_name(slug: str, run_id: str) -> str:
    """Spell a job name the way ``BedrockBatchBackend._job_name`` does: truncate the model, keep the run."""
    suffix = f"-{run_id}".lower()
    return f"jagged-{slug}".lower()[: MAX_JOB_NAME_CHARS - len(suffix)] + suffix


class TestTheLegTable:
    def test_the_production_and_on_demand_totals_are_the_design_s(self) -> None:
        assert sum(leg.records for leg in production_legs()) == PRODUCTION_RECORDS
        assert sum(leg.records for leg in on_demand_legs()) == ON_DEMAND_RECORDS

    def test_every_default_roster_row_costs_the_same_five_blocks(self) -> None:
        per_model: dict[str, int] = {}
        for leg in production_legs():
            if leg.block == BLOCK_DELIBERATION:
                continue
            per_model[leg.model_id] = per_model.get(leg.model_id, 0) + leg.records
        assert len(per_model) == MODELS
        assert set(per_model.values()) == {RECORDS_PER_MODEL}

    def test_only_the_two_effort_models_carry_a_deliberation_leg(self) -> None:
        deliberation = [leg for leg in all_legs() if leg.block == BLOCK_DELIBERATION]
        assert {leg.model_id for leg in deliberation} == set(DELIBERATION_MODEL_IDS)
        assert {leg.reasoning_effort for leg in deliberation} == {"high"}

    def test_the_floor_leg_is_the_second_sitting_and_nothing_else_is(self) -> None:
        for leg in all_legs():
            assert leg.sitting == (SITTING_B if leg.block == BLOCK_FLOOR else SITTING_A)

    def test_opus_is_on_demand_and_only_for_the_two_identity_blocks(self) -> None:
        opus = [leg for leg in all_legs() if leg.model_id == OPUS_MODEL_ID]
        assert all(leg.on_demand for leg in opus)
        assert {leg.block for leg in opus} == {BLOCK_IDENTITY_ONE_WAY, BLOCK_IDENTITY_MATCHED}
        assert not [leg for leg in all_legs() if leg.on_demand and leg.model_id != OPUS_MODEL_ID]

    def test_the_three_anthropic_and_openai_frontier_rows_run_live(self) -> None:
        for model_id in LIVE_MODEL_IDS:
            assert transport_for(model_id) == "live"

    def test_leg_ids_are_distinct(self) -> None:
        ids = [leg.leg_id for leg in all_legs()]
        assert len(ids) == len(set(ids))

    def test_every_batch_job_name_is_distinct_after_the_sixty_three_character_truncation(
        self,
    ) -> None:
        """Bedrock truncates the MODEL end, so two long ids sharing a prefix would ask for one name.

        Every name also has to carry the pass token, intact and in the run-id half the truncation never
        touches: job names are account-wide, so a name distinct within this table can still be one an
        earlier pass already used.
        """
        names: dict[str, str] = {}
        for leg in all_legs():
            if leg.transport != TRANSPORT_BATCH:
                continue
            suffix = f"-{leg.batch_run_id}"
            assert len(suffix) < MAX_JOB_NAME_CHARS
            assert leg.batch_run_id.startswith(f"{BATCH_PASS_TOKEN}-"), leg.leg_id
            name = _bedrock_job_name(model_slug(leg.model_id), leg.batch_run_id)
            assert leg.batch_job_name == name
            assert len(name) <= MAX_JOB_NAME_CHARS
            assert f"-{BATCH_PASS_TOKEN}-" in name, (name, leg.leg_id)
            assert name not in names, (name, names.get(name), leg.leg_id)
            names[name] = leg.leg_id

    def test_no_knockout_job_name_is_one_the_finished_pass_or_the_ladder_pass_already_used(
        self,
    ) -> None:
        """The knockout keeps the transfer pass's token, so ITS block ids are what keep it distinct.

        Bedrock job names are account-wide and a Completed job keeps its name for good, so the knockout's
        names have to be distinct from both earlier passes' -- and the check is worth having precisely
        because the token no longer separates them: only the new block ids do.
        """
        knockout = {
            _bedrock_job_name(model_slug(leg.model_id), leg.batch_run_id): leg.leg_id
            for leg in KNOCKOUT_PLAN.all_legs()
            if leg.transport == TRANSPORT_BATCH
        }
        assert knockout
        earlier = {
            _bedrock_job_name(model_slug(leg.model_id), leg.batch_run_id): leg.leg_id
            for leg in ONE_WAY_TRANSFER_PLAN.all_legs()
            if leg.transport == TRANSPORT_BATCH
        }
        earlier.update(
            {
                _bedrock_job_name(
                    decoupled_plan.model_slug(leg.model_id), leg.batch_run_id
                ): leg.leg_id
                for leg in decoupled_plan.all_legs()
                if leg.transport == decoupled_plan.TRANSPORT_BATCH
            }
        )
        shared_models = {leg.model_id for leg in KNOCKOUT_PLAN.all_legs()} & {
            leg.model_id for leg in ONE_WAY_TRANSFER_PLAN.all_legs()
        }
        assert shared_models, "the collision this guards against needs a model both passes run"
        collisions = {
            name: (knockout[name], earlier[name]) for name in knockout.keys() & earlier.keys()
        }
        assert not collisions, collisions

    def test_no_transfer_job_name_is_one_the_decoupled_ladder_pass_already_used(self) -> None:
        """The two passes share block ids (``floor``, ``deliberation``) and model short names.

        Bedrock refused eight transfer legs on 2026-09-02 because their job names were names the
        decoupled-ladder pass's Completed jobs still owned. Both tables are computed under the backend's
        naming rule, so a future block id or short name the two passes both adopt is caught here rather
        than at submit time with an upload already paid for.
        """
        transfer_names = {
            _bedrock_job_name(model_slug(leg.model_id), leg.batch_run_id): leg.leg_id
            for leg in all_legs()
            if leg.transport == TRANSPORT_BATCH
        }
        ladder_names = {
            _bedrock_job_name(decoupled_plan.model_slug(leg.model_id), leg.batch_run_id): leg.leg_id
            for leg in decoupled_plan.all_legs()
            if leg.transport == decoupled_plan.TRANSPORT_BATCH
        }
        assert transfer_names
        assert ladder_names
        shared_models = {leg.model_id for leg in all_legs()} & {
            leg.model_id for leg in decoupled_plan.all_legs()
        }
        assert shared_models, "the collision this guards against needs a model both passes run"
        collisions = {
            name: (transfer_names[name], ladder_names[name])
            for name in transfer_names.keys() & ladder_names.keys()
        }
        assert not collisions, collisions

    def test_no_batch_leg_sits_below_the_hundred_record_floor(self) -> None:
        for leg in all_legs():
            if leg.transport == TRANSPORT_BATCH:
                assert leg.records >= MIN_BATCH_RECORDS
                refuse_below_batch_floor(leg, leg.records)

    def test_a_leg_under_the_floor_refuses_in_this_clis_own_levers(self) -> None:
        with pytest.raises(ValueError, match="DRAWS"):
            refuse_below_batch_floor(leg_for("gpt-oss-20b--default--B--floor"), 96)

    def test_an_unknown_leg_id_lists_the_table(self) -> None:
        with pytest.raises(ValueError, match="Known legs"):
            leg_for("not-a-leg")


class TestTheTwoPlanTables:
    """What the required ``--pass`` selects, and what each table is not allowed to share."""

    def test_the_knockout_plans_the_design_s_records_over_its_three_rows(self) -> None:
        assert len(KNOCKOUT_PLAN.legs) == len(KNOCKOUT_MODEL_IDS) * 3
        assert sum(leg.records for leg in KNOCKOUT_PLAN.production_legs()) == KNOCKOUT_RECORDS
        per_model: dict[str, int] = {}
        for leg in KNOCKOUT_PLAN.legs:
            per_model[leg.model_id] = per_model.get(leg.model_id, 0) + leg.records
        assert set(per_model.values()) == {KNOCKOUT_RECORDS_PER_MODEL}
        assert set(per_model) == set(KNOCKOUT_MODEL_IDS)

    def test_the_knockout_reads_the_twin_against_the_drawn_game_at_the_same_dose(self) -> None:
        """The double difference pairs one game's cells against the other's, so the dose cannot move."""
        specs = {
            cell.spec
            for block in KNOCKOUT_PLAN.reading_blocks
            for cell in KNOCKOUT_PLAN.cells_by_block[block]
        }
        assert (
            len(
                {
                    (spec.credit_numerator, spec.beneficiary_count, spec.own_stake_scale)
                    for spec in specs
                }
            )
            == 1
        )
        games = {
            block: {cell.game_id for cell in cells}
            for block, cells in KNOCKOUT_PLAN.cells_by_block.items()
        }
        assert games[BLOCK_KNOCKOUT_MATCHED] == {MATCHED_DECISION_TRANSFER_GAME_ID}
        assert games[BLOCK_KNOCKOUT_DRAWN] == {DRAWN_DECISION_TRANSFER_GAME_ID}
        assert games[BLOCK_KNOCKOUT_FLOOR] == {MATCHED_DECISION_TRANSFER_GAME_ID}

    def test_every_knockout_block_opens_with_its_own_game_s_blind_baseline(self) -> None:
        """Each rung is read against the blind cell of ITS OWN game, re-measured in this same pass."""
        for block in KNOCKOUT_PLAN.reading_blocks:
            first = KNOCKOUT_PLAN.cells_by_block[block][0]
            assert first.rung is None
            assert first.cell_id == TRANSFER_IDENTITY_BLIND_LABEL

    def test_the_record_rung_is_planned_beside_the_base_it_is_read_against(self) -> None:
        cells = [cell.cell_id for cell in KNOCKOUT_PLAN.cells_by_block[BLOCK_KNOCKOUT_MATCHED]]
        assert RUNG_DIFFERENT_FAMILY_TRACK_RECORD in cells
        assert "different-family" in cells

    def test_the_two_passes_never_share_a_prefix_a_run_dir_or_a_leg_id(self) -> None:
        """A shared prefix or run directory is a silent replace the first time a run name repeats."""
        assert ONE_WAY_TRANSFER_PLAN.batch_prefix != KNOCKOUT_PLAN.batch_prefix
        assert ONE_WAY_TRANSFER_PLAN.default_run_dir != KNOCKOUT_PLAN.default_run_dir
        assert not set(ONE_WAY_TRANSFER_PLAN.legs_by_id) & set(KNOCKOUT_PLAN.legs_by_id)
        for leg in KNOCKOUT_PLAN.legs:
            assert leg.batch_prefix == KNOCKOUT_PLAN.batch_prefix

    def test_a_leg_of_another_pass_refuses_rather_than_resolving(self) -> None:
        """A mistyped ``--pass`` with a real leg id would otherwise submit that pass's job here."""
        other = ONE_WAY_TRANSFER_PLAN.legs[0].leg_id
        with pytest.raises(ValueError, match="is not a leg of the 'knockout' pass"):
            KNOCKOUT_PLAN.leg_for(other)
        assert ONE_WAY_TRANSFER_PLAN.leg_for(other).leg_id == other

    def test_plan_table_lists_every_pass_when_the_id_is_not_one(self) -> None:
        assert plan_table("knockout") is KNOCKOUT_PLAN
        with pytest.raises(ValueError, match="the passes are"):
            plan_table("one-way")

    def test_the_knockout_smoke_runs_on_the_cheapest_live_row_of_its_own_roster(self) -> None:
        """The finished pass's smoke model is not on this roster at all."""
        assert KNOCKOUT_PLAN.smoke_live_model_id in KNOCKOUT_MODEL_IDS
        assert ONE_WAY_TRANSFER_PLAN.smoke_live_model_id not in KNOCKOUT_MODEL_IDS
        assert KNOCKOUT_PLAN.smoke_block in KNOCKOUT_PLAN.reading_blocks

    def test_the_file_stem_carries_the_full_model_id(self) -> None:
        """A filename is read months later without this table in front of the reader."""
        leg = leg_for("gpt-oss-20b--default--A--identity-ow")
        assert model_slug(leg.model_id) in leg.file_stem
        assert leg.block in leg.file_stem


class TestTheRecordKey:
    def test_it_names_every_axis_that_never_pools(self) -> None:
        key = record_key(
            block=BLOCK_IDENTITY_ONE_WAY,
            game_id=ONE_WAY_TRANSFER_GAME_ID,
            cell="same-checkpoint",
            variant="credit-2-1--count-3--stake-100",
            scenario_id="synthetic-lofts",
            polarity="set",
            model_id="deepseek.v3.2",
            reasoning_effort=None,
            sitting=SITTING_A,
            draw=3,
        )
        assert key == (
            "identity-ow|one-way-transfer|same-checkpoint|credit-2-1--count-3--stake-100"
            "|synthetic-lofts|set|deepseek.v3.2|effort=default|sitting=A|draw=3"
        )

    def key(
        self, *, polarity: str = "set", sitting: str = SITTING_A, effort: str | None = None
    ) -> str:
        return record_key(
            block=BLOCK_IDENTITY_ONE_WAY,
            game_id=ONE_WAY_TRANSFER_GAME_ID,
            cell="same-checkpoint",
            variant="credit-2-1--count-3--stake-100",
            scenario_id="synthetic-lofts",
            polarity=polarity,
            model_id="deepseek.v3.2",
            reasoning_effort=effort,
            sitting=sitting,
            draw=0,
        )

    def test_the_two_polarities_two_sittings_and_two_efforts_are_separate_keys(self) -> None:
        keys = {
            self.key(),
            self.key(polarity="keep"),
            self.key(sitting=SITTING_B),
            self.key(effort="high"),
        }
        assert len(keys) == 4


class TestTheRenderedPlan:
    def leg(self, block: str = BLOCK_FLOOR) -> Leg:
        return Leg(
            "openai.gpt-oss-20b-1:0",
            TRANSPORT_BATCH,
            block,
            CELLS_BY_BLOCK[block],
            None,
            SITTING_B if block == BLOCK_FLOOR else SITTING_A,
        )

    def test_a_leg_renders_exactly_its_table_row_under_distinct_keys(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        leg = self.leg()
        calls = planned_calls_for_leg(leg, transfer_stimulus)
        assert len(calls) == leg.records == RECORDS_BY_BLOCK[BLOCK_FLOOR]
        assert len({call.key for call in calls}) == len(calls)
        assert len({call.prompt_id for call in calls}) == len(calls) // DRAWS

    def test_the_render_is_deterministic_and_order_stable(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """Positional digests on the batch path mean a reordering here is a mis-join, not a nuisance."""
        first = planned_calls_for_leg(self.leg(), transfer_stimulus)
        second = planned_calls_for_leg(self.leg(), transfer_stimulus)
        assert [call.key for call in first] == [call.key for call in second]
        assert [call.prompt for call in first] == [call.prompt for call in second]

    def test_every_planned_call_carries_its_own_dose_on_the_row(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        for call in planned_calls_for_leg(self.leg(BLOCK_DOSE_ONE_WAY), transfer_stimulus):
            assert f"credit-{call.credit_numerator}-{call.credit_denominator}" in call.variant
            assert f"count-{call.beneficiary_count}" in call.variant
            assert call.prompt_id.endswith(f"--{call.polarity}")

    def test_the_blind_cell_renders_no_counterpart_paragraph_and_every_rung_renders_one(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        marker = about_the_other_side("")
        for call in planned_calls_for_leg(self.leg(BLOCK_IDENTITY_MATCHED), transfer_stimulus):
            carries = marker in call.prompt
            assert carries == (call.cell != TRANSFER_IDENTITY_BLIND_LABEL), call.cell


class TestThePromptAudits:
    def test_every_planned_prompt_is_its_blind_stem_plus_one_paragraph(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        for block in (BLOCK_IDENTITY_ONE_WAY, BLOCK_IDENTITY_MATCHED, BLOCK_INTERACTION_ONE_WAY):
            leg = Leg(
                "openai.gpt-oss-20b-1:0",
                TRANSPORT_BATCH,
                block,
                CELLS_BY_BLOCK[block],
                None,
                SITTING_A,
            )
            calls = planned_calls_for_leg(leg, transfer_stimulus)
            assert audit_planned_calls(calls, transfer_stimulus) == len(calls)

    def test_a_planted_two_paragraph_clause_makes_the_audit_go_red(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """The sabotage: a second paragraph carries no marker, so it survives the marker deletion."""
        key = (ONE_WAY_TRANSFER_GAME_ID, "same-checkpoint")
        planted = transfer_stimulus.clause_templates[key].replace(
            ", SYNTHETIC", ".\n\nSYNTHETIC", 1
        )
        sabotaged = replace(
            transfer_stimulus,
            clause_templates={**transfer_stimulus.clause_templates, key: planted},
        )
        leg = Leg(
            "openai.gpt-oss-20b-1:0",
            TRANSPORT_BATCH,
            BLOCK_IDENTITY_ONE_WAY,
            (
                Cell(
                    ONE_WAY_TRANSFER_GAME_ID, "same-checkpoint", spec_for(ONE_WAY_TRANSFER_GAME_ID)
                ),
            ),
            None,
            SITTING_A,
        )
        calls = planned_calls_for_leg(leg, sabotaged)
        with pytest.raises(ValueError, match="one counterpart paragraph"):
            audit_planned_calls(calls, sabotaged)

    def test_a_prompt_id_without_a_framing_segment_refuses(self) -> None:
        with pytest.raises(ValueError, match="framing segments"):
            blind_stem_id("one-way-transfer--synthetic-lofts--credit-2-1--count-3--stake-100--set")

    def test_every_pair_of_games_differs_by_the_mechanics_and_the_clause_and_nothing_else(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        compared = audit_games_differ_by_mechanics_and_clause(transfer_stimulus)
        assert compared == len(transfer_stimulus.scenarios) * len(ANSWER_POLARITIES) * GAME_PAIRS

    def test_a_frame_of_several_paragraphs_still_locates_the_right_two_sections(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """Found on the real roster: an authored frame is several paragraphs, and a fixed section index
        was then checking the closing order against the mechanics while still passing on a one-paragraph
        synthetic frame."""
        multi = tuple(
            replace(scenario, frame=scenario.frame.replace(". ", ".\n\n", 1))
            for scenario in transfer_stimulus.scenarios
        )
        widened = replace(transfer_stimulus, scenarios=multi)
        assert frame_section_count(multi[0]) == 2
        assert (
            audit_games_differ_by_mechanics_and_clause(widened)
            == len(multi) * len(ANSWER_POLARITIES) * GAME_PAIRS
        )

    def test_every_planned_prompt_over_a_multi_paragraph_frame_still_audits(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """The one-inserted-paragraph audit keys on the marker rather than on a position, so it holds."""
        multi = tuple(
            replace(scenario, frame=scenario.frame.replace(". ", ".\n\n", 1))
            for scenario in transfer_stimulus.scenarios
        )
        widened = replace(transfer_stimulus, scenarios=multi)
        leg = Leg(
            "openai.gpt-oss-20b-1:0",
            TRANSPORT_BATCH,
            BLOCK_IDENTITY_ONE_WAY,
            CELLS_BY_BLOCK[BLOCK_IDENTITY_ONE_WAY],
            None,
            SITTING_A,
        )
        calls = planned_calls_for_leg(leg, widened)
        assert audit_planned_calls(calls, widened) == len(calls)

    def test_a_drifted_closing_order_makes_the_cross_game_audit_go_red(
        self, transfer_stimulus: TransferStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sabotage that matters most: if the closing order differed between the two games, the
        coupling contrast would be confounded with observability and attribution, and nothing else in
        the pipeline would notice."""
        import games.prompts as prompts_module  # noqa: PLC0415 - patched for one sabotage case

        drifted = (
            prompts_module.MATCHED_DECISION_TRANSFER_MECHANICS + " {resource} are counted twice."
        )
        monkeypatch.setattr(prompts_module, "MATCHED_DECISION_TRANSFER_MECHANICS", drifted)
        monkeypatch.setitem(
            prompts_module.TRANSFER_RENDERERS,
            MATCHED_DECISION_TRANSFER_GAME_ID,
            _renderer_with_a_drifted_closing_order(prompts_module),
        )
        with pytest.raises(ValueError, match="differ in sections"):
            audit_games_differ_by_mechanics_and_clause(transfer_stimulus)

    def test_a_third_sentence_drifting_between_the_twin_and_the_drawn_game_goes_red(
        self, transfer_stimulus: TransferStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The section check alone passes this: the mechanics sections are expected to differ, and a
        drawn renderer that filled the credit with the actor's own setter differs there too. The knockout
        is a difference over exactly these two paragraphs, so the pair is held sentence by sentence."""
        import games.prompts as prompts_module  # noqa: PLC0415 - patched for one sabotage case

        monkeypatch.setitem(
            prompts_module.TRANSFER_RENDERERS,
            DRAWN_DECISION_TRANSFER_GAME_ID,
            _drawn_renderer_crediting_the_actor(prompts_module),
        )
        with pytest.raises(ValueError, match="sentences of DRAWN_DECISION_REPLACEMENTS swapped"):
            audit_games_differ_by_mechanics_and_clause(transfer_stimulus)


def _drawn_renderer_crediting_the_actor(prompts_module: object):
    """Build a drawn renderer whose credit names the actor as the setter, everything else unchanged.

    The drift the templates cannot show: the two mechanics constants still differ by exactly the two
    replaced sentences, the section count is unchanged, and the rendered credit sentence now differs
    between the twin and the drawn game as a third thing.
    """

    def render(spec: object, scenario: object, *, clause: str | None, polarity: str) -> str:
        original = prompts_module.render_drawn_decision_transfer_prompt(  # type: ignore[attr-defined]
            spec, scenario, clause=clause, polarity=polarity
        )
        return original.replace("for each one set down", "for each one you set down", 1)

    return render


def _renderer_with_a_drifted_closing_order(prompts_module: object):
    """Build a twin renderer whose closing order reads differently, with the section count unchanged.

    The section count is deliberately preserved, because a render with an extra section fails a different
    check ("cannot be compared section by section at all"). What this stands in for is the subtle version:
    the two games state the closing order in slightly different words, both prompts look right on their
    own, and the coupling contrast quietly carries observability and attribution with it.
    """

    def render(spec: object, scenario: object, *, clause: str | None, polarity: str) -> str:
        original = prompts_module.render_matched_decision_transfer_prompt(  # type: ignore[attr-defined]
            spec, scenario, clause=clause, polarity=polarity
        )
        sections = original.split("\n\n")
        sections[2] = sections[2] + " SYNTHETIC-DRIFT."
        return "\n\n".join(sections)

    return render


class TestTheDesignLabels:
    def test_they_name_every_game_block_rung_and_model(self) -> None:
        for label in (*BLOCKS, *RUNGS_BY_GAME[MATCHED_DECISION_TRANSFER_GAME_ID], OPUS_MODEL_ID):
            assert label in DESIGN_LABELS

    def test_they_name_the_two_words_that_would_give_the_game_away(self) -> None:
        """Either would tell the judge which of the two games it is reading a reply to."""
        assert "one-way" in DESIGN_LABELS
        assert "matched" in DESIGN_LABELS

    def test_beneficiar_is_a_stem_so_it_catches_both_numbers(self) -> None:
        assert "beneficiar" in DESIGN_LABELS
        assert "beneficiaries" not in DESIGN_LABELS

    def test_the_reply_list_lifts_only_the_words_the_prompts_themselves_print(self) -> None:
        """An authored reply imitates a subject, and a subject quotes the note it is answering."""
        assert DESIGN_LABELS_THE_PROMPTS_PRINT == ("drawn",)
        assert "drawn" in TRANSFER_CLOSING_ORDER.casefold()
        assert "drawn" not in DESIGN_LABELS_IN_AUTHORED_REPLIES
        assert DRAWN_DECISION_TRANSFER_GAME_ID in DESIGN_LABELS_IN_AUTHORED_REPLIES
        assert set(DESIGN_LABELS_IN_AUTHORED_REPLIES) | {"drawn"} == set(DESIGN_LABELS)

    def test_a_placeholder_name_is_not_text_a_subject_reads(self) -> None:
        """The mechanics name a beneficiary-group slot the render fills, so the stem stays banned."""
        assert "beneficiar" in MATCHED_DECISION_TRANSFER_MECHANICS
        assert "beneficiar" not in DESIGN_LABELS_THE_PROMPTS_PRINT
        assert "beneficiar" in DESIGN_LABELS_IN_AUTHORED_REPLIES


class TestRenderCellRow:
    def test_the_blind_cell_and_the_stem_of_a_rung_are_one_render(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """The identity-blind cell IS the audit's stem, which is why it is not a second render."""
        scenario = transfer_stimulus.scenarios[0]
        spec = spec_for(ONE_WAY_TRANSFER_GAME_ID)
        blind = render_cell_row(
            Cell(ONE_WAY_TRANSFER_GAME_ID, None, spec),
            transfer_stimulus,
            scenario=scenario,
            polarity="set",
        )
        rung = render_cell_row(
            Cell(ONE_WAY_TRANSFER_GAME_ID, "same-checkpoint", spec),
            transfer_stimulus,
            scenario=scenario,
            polarity="set",
        )
        assert blind_stem_id(str(rung["prompt_id"])) == str(blind["prompt_id"])


class TestTheBoardCondition:
    """A board is model-independent; what it IS to its reader is the label every pass D record carries.

    Both models read all four boards, which is what gives the between-model read on one board for free and
    what makes the condition a function of (board, reader) rather than a cell attribute. Getting the pair
    the wrong way round would relabel whole cells as their own opposite with every count still right, so
    the derivation is pinned here corner by corner.
    """

    def test_every_corner_of_the_2x2_is_derived_for_both_readers(self) -> None:
        derived = {
            (board_id, model_id): board_condition(board_id, model_id)
            for board_id in BOARD_IDS
            for model_id in PAIR_MODEL_IDS
        }
        assert len(derived) == len(BOARD_IDS) * len(PAIR_MODEL_IDS)
        assert set(derived.values()) == set(BOARD_CONDITIONS)
        for board_id in BOARD_IDS:
            luna, qwen = (
                derived[board_id, model_id] for model_id in (LUNA_MODEL_ID, QWEN_MODEL_ID)
            )
            assert luna != qwen, board_id
        assert derived["board-luna-raw", LUNA_MODEL_ID] == CONDITION_OWN_RAW
        assert derived["board-luna-raw", QWEN_MODEL_ID] == CONDITION_OTHER_RAW
        assert (
            derived["board-luna-in-qwen-words", LUNA_MODEL_ID] == CONDITION_OWN_CONTENT_OTHER_WORDS
        )
        assert (
            derived["board-luna-in-qwen-words", QWEN_MODEL_ID] == CONDITION_OTHER_CONTENT_OWN_WORDS
        )

    def test_a_cell_that_is_not_a_board_has_no_condition(self) -> None:
        assert board_condition(TRANSFER_IDENTITY_BLIND_LABEL, LUNA_MODEL_ID) is None
        assert board_condition(RUNG_DIFFERENT_FAMILY_TRACK_RECORD, QWEN_MODEL_ID) is None

    def test_a_reader_the_boards_were_not_drawn_from_refuses_rather_than_guessing(self) -> None:
        """ "Own" and "other" mean nothing for a third model, and a guess would be a mislabelled cell."""
        with pytest.raises(ValueError, match="not one of the two models"):
            board_condition(BOARD_IDS[0], OPUS_MODEL_ID)

    def test_every_planned_board_call_carries_its_condition_and_no_other_call_does(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        for leg in FINGERPRINT_PLAN.all_legs():
            for call in planned_calls_for_leg(leg, transfer_stimulus):
                expected = board_condition(call.cell, call.model_id)
                assert call.board_condition == expected
                assert call.metadata()["board_condition"] == expected
        dose_leg = CORRELATION_DOSE_PLAN.leg_for("luna--default--A--dose-anchors")
        assert all(
            call.board_condition is None
            for call in planned_calls_for_leg(dose_leg, transfer_stimulus)
        )


class TestTheTwoNewPlanTables:
    def test_each_pass_plans_the_design_s_records_over_its_two_rows(self) -> None:
        """1,152 per model for the fingerprint pass and 1,920 for the dose pass, spelled out."""
        assert sum(leg.records for leg in FINGERPRINT_PLAN.production_legs()) == 2 * 1_152
        assert sum(leg.records for leg in CORRELATION_DOSE_PLAN.production_legs()) == 2 * 1_920
        for table in (FINGERPRINT_PLAN, CORRELATION_DOSE_PLAN):
            assert {leg.model_id for leg in table.all_legs()} == set(PAIR_MODEL_IDS)

    def test_the_boards_ride_the_twin_beside_the_anchor_and_the_told_identity(self) -> None:
        cells = FINGERPRINT_PLAN.cells_by_block[BLOCK_FINGERPRINT_MATCHED]
        assert [cell.cell_id for cell in cells] == [
            TRANSFER_IDENTITY_BLIND_LABEL,
            "same-checkpoint",
            *BOARD_IDS,
        ]
        assert {cell.game_id for cell in cells} == {MATCHED_DECISION_TRANSFER_GAME_ID}
        assert len({cell.variant for cell in cells}) == 1, (
            "every board cell sits at the reference dose"
        )

    def test_the_dose_ladder_is_planned_in_ladder_order_with_its_anchors_beside_it(self) -> None:
        record = [cell.cell_id for cell in CORRELATION_DOSE_PLAN.cells_by_block[BLOCK_DOSE_RECORD]]
        assert record == [record_rung_id(matched) for matched in MATCHED_ROUNDS_LADDER]
        anchors = [
            cell.cell_id for cell in CORRELATION_DOSE_PLAN.cells_by_block[BLOCK_DOSE_ANCHORS]
        ]
        assert anchors[0] == TRANSFER_IDENTITY_BLIND_LABEL
        assert RUNG_DIFFERENT_FAMILY_TRACK_RECORD in anchors
        assert RUNG_SAME_CHECKPOINT_RECORD_0 in anchors

    def test_each_floor_re_sits_only_cells_its_own_pass_samples(self) -> None:
        """A floor over a cell with no first sitting would be a bar for a contrast nobody measured."""
        for table in (FINGERPRINT_PLAN, CORRELATION_DOSE_PLAN):
            sampled = {
                cell.cell_id
                for block in table.reading_blocks
                for cell in table.cells_by_block[block]
            }
            for block in table.resat_blocks:
                for cell in table.cells_by_block[block]:
                    assert cell.cell_id in sampled, (table.pass_id, block, cell.cell_id)
                assert all(
                    leg.sitting == SITTING_B for leg in table.all_legs() if leg.block == block
                )

    def test_the_four_passes_never_share_a_prefix_a_run_dir_a_token_or_a_leg_id(self) -> None:
        tables = list(PLAN_TABLES.values())
        assert len({table.batch_prefix for table in tables}) == len(tables)
        assert len({table.default_run_dir for table in tables}) == len(tables)
        assert {FINGERPRINT_PLAN.batch_pass_token, CORRELATION_DOSE_PLAN.batch_pass_token} == {
            "fpb",
            "cdo",
        }
        ids = [leg.leg_id for table in tables for leg in table.all_legs()]
        assert len(ids) == len(set(ids))

    def test_no_batch_job_name_of_any_pass_collides_with_any_other_s(self) -> None:
        """Every pass this repository runs over one substrate, plus the ladder pass beside it.

        Bedrock job names are account-wide and a Completed job keeps its name for good, so a collision is
        a refused submit with an upload already paid for. Two of these four passes run in PARALLEL with
        each other and with a fifth pass on the deference machinery, submitted by three operators who are
        not reading each other's tables, which is what makes the check worth the fixture.
        """
        owners: dict[str, str] = {}
        collisions: list[tuple[str, str, str]] = []
        legs = [
            (table.pass_id, leg)
            for table in PLAN_TABLES.values()
            for leg in table.all_legs()
            if leg.transport == TRANSPORT_BATCH
        ] + [
            ("decoupled-ladder", leg)
            for leg in decoupled_plan.all_legs()
            if leg.transport == decoupled_plan.TRANSPORT_BATCH
        ]
        assert legs
        for pass_id, leg in legs:
            name = _bedrock_job_name(model_slug(leg.model_id), leg.batch_run_id)
            assert len(name) <= MAX_JOB_NAME_CHARS
            if name in owners:
                collisions.append((name, owners[name], f"{pass_id}/{leg.leg_id}"))
            owners[name] = f"{pass_id}/{leg.leg_id}"
        assert not collisions, collisions

    def test_no_new_batch_leg_sits_below_the_hundred_record_floor(self) -> None:
        for table in (FINGERPRINT_PLAN, CORRELATION_DOSE_PLAN):
            for leg in table.all_legs():
                if leg.transport == TRANSPORT_BATCH:
                    assert leg.records >= MIN_BATCH_RECORDS
                    refuse_below_batch_floor(leg, leg.records)


class TestTheMatchedRoundsThresholds:
    """The two doses at which an expected-value maximiser switches, computed from the payoffs.

    Printed on the curve by the readout, so a stale number would put the classification ("steps at the
    binary threshold") on the wrong rung. Computed from the spec for that reason, and pinned here at the
    reference dose because that is the dose both passes sample.
    """

    def test_at_the_reference_dose_they_are_one_sixth_and_seven_twelfths(self) -> None:
        thresholds = matched_rounds_thresholds(spec_for(MATCHED_DECISION_TRANSFER_GAME_ID))[
            "thresholds"
        ]
        assert thresholds["match_else_independent"]["fraction"] == [1, 6]
        assert thresholds["binary_mismatch"]["fraction"] == [7, 12]

    def test_each_threshold_names_the_two_rungs_it_sits_between(self) -> None:
        """The added rungs 2 and 1 straddle the first threshold; the owner's 7 and 5 straddle the second."""
        thresholds = matched_rounds_thresholds(spec_for(MATCHED_DECISION_TRANSFER_GAME_ID))[
            "thresholds"
        ]
        assert thresholds["match_else_independent"]["between_rungs"] == [1, 2]
        assert thresholds["binary_mismatch"]["between_rungs"] == [5, 7]

    def test_a_different_dose_moves_them_rather_than_leaving_a_stale_number(self) -> None:
        halved = matched_rounds_thresholds(
            spec_for(MATCHED_DECISION_TRANSFER_GAME_ID, credit="credit-half")
        )["thresholds"]
        assert halved["match_else_independent"]["fraction"] != [1, 6]
        assert halved["binary_mismatch"]["fraction"] != [7, 12]

    def test_the_own_stake_moves_them_too_rather_than_being_read_as_one(self) -> None:
        """The third axis of the dose, which the arithmetic ignored while every sampled cell sat at 1.0.

        With an own stake of s the marginal unit costs s rather than 1, so the independence threshold is
        s / (c n) and the binary one (s + c n) / (2 c n). At the registered 0.1 stake that is 1/60 and
        61/120: the first falls between rungs 0 and 1 where the full-stake value falls between 1 and 2, so
        a readout marking the full-stake number would name a rung two rungs off the real one.
        """
        dipped = matched_rounds_thresholds(spec_for(MATCHED_DECISION_TRANSFER_GAME_ID, stake=0.1))
        thresholds = dipped["thresholds"]
        assert dipped["own_stake_percent"] == 10
        assert thresholds["match_else_independent"]["fraction"] == [1, 60]
        assert thresholds["match_else_independent"]["between_rungs"] == [0, 1]
        assert thresholds["binary_mismatch"]["fraction"] == [61, 120]
        assert thresholds["binary_mismatch"]["between_rungs"] == [5, 7]

    def test_a_stake_of_zero_leaves_the_independence_threshold_at_the_bottom_rung(self) -> None:
        """The registered 0.0 stake: setting down costs the reader nothing, so any stated match pays."""
        thresholds = matched_rounds_thresholds(
            spec_for(MATCHED_DECISION_TRANSFER_GAME_ID, stake=0.0)
        )["thresholds"]
        assert thresholds["match_else_independent"]["fraction"] == [0, 1]
        assert thresholds["binary_mismatch"]["fraction"] == [1, 2]


class TestTheBoardInsertionAudit:
    """A board is three messages and a lead-in, and it still has to be ONE inserted paragraph.

    That is the property every reading of pass D rests on: the board cell differs from the anonymous
    render by the inserted paragraph and by nothing else, so a movement between them is attributable to
    the messages. The delimiter is a single line for exactly this reason, and the pair of tests below is
    the check that it stays one -- the audit accepts the real render and goes red on a planted blank line.
    """

    def board_leg(self) -> Leg:
        return FINGERPRINT_PLAN.leg_for("luna--default--A--fingerprint-md")

    def test_every_planned_board_prompt_is_its_blind_stem_plus_one_paragraph(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        calls = planned_calls_for_leg(self.board_leg(), transfer_stimulus)
        assert audit_planned_calls(calls, transfer_stimulus) == len(calls)

    def test_a_blank_line_planted_in_a_message_makes_the_audit_go_red(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """The sabotage: the loader refuses this message, so the plant goes in past it.

        The rest of the board carries no counterpart marker, so a blank line inside one message leaves
        every later message behind as an orphan section when the audit deletes the inserted paragraph.
        """
        scenario_id = transfer_stimulus.scenarios[0].scenario_id
        board_id = BOARD_IDS[0]
        board = transfer_stimulus.boards[scenario_id][board_id]
        planted = replace(
            board,
            messages=(f"{board.messages[0]}\n\nSYNTHETIC-SECOND-PARAGRAPH.", *board.messages[1:]),
        )
        sabotaged = replace(
            transfer_stimulus,
            boards={
                **transfer_stimulus.boards,
                scenario_id: {**transfer_stimulus.boards[scenario_id], board_id: planted},
            },
        )
        calls = planned_calls_for_leg(self.board_leg(), sabotaged)
        with pytest.raises(ValueError, match="not its stem plus one counterpart paragraph"):
            audit_planned_calls(calls, sabotaged)
