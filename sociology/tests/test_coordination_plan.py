"""The coordination plan: the leg table's arithmetic, the record keys, and the five structural audits.

The audits are the point of this module, and each is sabotaged here as well as exercised. Every failure they
catch leaves a complete, plausible run behind: a peer cell that is not its identity-blind stem plus one
paragraph, a sole-owner setting that also moved the close time, an oversight pair the render pulled two tokens
apart, a status block whose field was spelled differently in one cell, an identity paragraph that never states
how many other agents there are. Each is planted deliberately and required to go red.

The sabotages mutate a loaded stimulus with :func:`dataclasses.replace` rather than the JSON file, because
what is being tested is the audit rather than the loader: several of these violations the loader refuses on
the way in, and an audit that had never been shown one would be a reassuring message.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from games.prompts import LABEL_PRINT_ORDERS
from sociology import coordination_plan as plan_module
from sociology import coordination_stimulus as coordination_stimulus_module
from sociology import decoupled_plan, deference_plan, transfer_plan
from sociology.coordination_plan import (
    ALL_MODEL_IDS,
    ARM_BY_BLOCK,
    BATCH_PASS_TOKEN,
    BLOCK_FLOOR,
    BLOCK_OVERSIGHT_OFF,
    BLOCK_OVERSIGHT_ON,
    BLOCKS,
    CELLS_BY_BLOCK,
    COORDINATION_BATCH_PREFIX,
    DESIGN_LABELS,
    FLOOR_CELLS,
    LEGS,
    LUNA_MODEL_ID,
    QWEN_MODEL_ID,
    READING_BLOCKS,
    ROWS_PER_CELL,
    audit_arms_differ_in_the_oversight_sentence,
    audit_peer_cells_differ_in_the_identity_section,
    audit_planned_calls,
    audit_print_orders_differ_in_the_fork_and_instruction,
    audit_settings_and_reconciliations,
    audit_the_status_block_is_the_same_all_pending_list,
    audits,
    cells_for,
    identity_blind_stem,
    leg_for,
    planned_calls_for_leg,
    prompt_id,
)
from sociology.coordination_stimulus import (
    ARM_OVERSIGHT_OFF,
    ARM_OVERSIGHT_ON,
    ARMS,
    CELL_DIFFERENT_FAMILY,
    CELL_IDENTITY_BLIND,
    CELL_NO_PEERS,
    CELL_SAME_CHECKPOINT,
    CELL_SAME_CHECKPOINT_DRAWN,
    CELLS,
    N_SCENARIOS,
    PEER_COUNT,
)
from sociology.deference_plan import DRAWS, SITTING_A, SITTING_B, TRANSPORT_BATCH, TRANSPORT_LIVE
from sociology.tests.synthetic_coordination import SYNTHETIC_OVERSIGHT_OFF

if TYPE_CHECKING:
    from sociology.coordination_stimulus import CoordinationScenario, CoordinationStimulus

EXPECTED_TOTAL_RECORDS = 3_584
"""Two models x (768 + 768 + 256). Spelled out, because the design doc states this number and a table whose
arithmetic drifted from it would be read against a plan nobody re-derived."""

LIVE_LEG_ID = "luna--default--A--coord-on"
BATCH_FLOOR_LEG_ID = "qwen3-235b--default--B--coord-floor"


class TestTheLegTable:
    def test_every_roster_row_has_all_three_blocks(self) -> None:
        by_model: dict[str, list[str]] = {}
        for leg in LEGS:
            by_model.setdefault(leg.model_id, []).append(leg.block)
        assert set(by_model) == set(ALL_MODEL_IDS) == {LUNA_MODEL_ID, QWEN_MODEL_ID}
        assert all(blocks == list(BLOCKS) for blocks in by_model.values())

    def test_the_record_counts_are_the_designs_own_arithmetic(self) -> None:
        assert N_SCENARIOS * len(LABEL_PRINT_ORDERS) == ROWS_PER_CELL == 16
        assert leg_for(LIVE_LEG_ID).records == len(CELLS) * ROWS_PER_CELL * DRAWS == 768
        assert leg_for(BATCH_FLOOR_LEG_ID).records == 2 * ROWS_PER_CELL * DRAWS == 256
        assert sum(leg.records for leg in LEGS) == EXPECTED_TOTAL_RECORDS

    def test_the_floor_is_a_second_sitting_of_the_headline_pair_in_the_oversight_off_arm(
        self,
    ) -> None:
        floor = leg_for(BATCH_FLOOR_LEG_ID)
        assert floor.sitting == SITTING_B
        assert [cell.cell_id for cell in floor.cells] == list(FLOOR_CELLS)
        assert floor.arm == ARM_OVERSIGHT_OFF
        assert all(leg.sitting == SITTING_A for leg in LEGS if leg.block in READING_BLOCKS)

    def test_each_reading_block_samples_every_cell_of_its_own_arm(self) -> None:
        for block, arm in (
            (BLOCK_OVERSIGHT_ON, ARM_OVERSIGHT_ON),
            (BLOCK_OVERSIGHT_OFF, ARM_OVERSIGHT_OFF),
        ):
            cells = CELLS_BY_BLOCK[block]
            assert [cell.cell_id for cell in cells] == list(CELLS)
            assert {cell.arm for cell in cells} == {arm}
            assert ARM_BY_BLOCK[block] == arm

    def test_the_transports_follow_the_roster(self) -> None:
        by_id = {leg.leg_id: leg for leg in LEGS}
        assert by_id[LIVE_LEG_ID].transport == TRANSPORT_LIVE
        assert by_id[LIVE_LEG_ID].batch_job_name is None
        assert by_id[BATCH_FLOOR_LEG_ID].transport == TRANSPORT_BATCH

    def test_an_unknown_leg_id_names_this_pass_and_lists_its_table(self) -> None:
        with pytest.raises(ValueError, match="not a leg of the 'coordination' plan"):
            leg_for("nobody--default--A--coord-on")

    def test_cells_for_refuses_an_unknown_arm_or_cell(self) -> None:
        with pytest.raises(ValueError, match="not an arm"):
            cells_for("sideways", CELLS)
        with pytest.raises(ValueError, match="not cells of this design"):
            cells_for(ARM_OVERSIGHT_ON, ("no-such-cell",))

    def test_no_batch_leg_sits_under_the_hundred_record_floor(self) -> None:
        assert all(leg.records >= 100 for leg in LEGS if leg.transport == TRANSPORT_BATCH)


class TestTheBatchNames:
    def coordination_run_ids(self) -> set[str]:
        return {leg.batch_run_id for leg in LEGS if leg.transport == TRANSPORT_BATCH}

    def test_every_batch_run_id_opens_with_this_passs_token(self) -> None:
        assert self.coordination_run_ids() == {
            f"{BATCH_PASS_TOKEN}-qwen3-235b-default-a-coord-on",
            f"{BATCH_PASS_TOKEN}-qwen3-235b-default-a-coord-off",
            f"{BATCH_PASS_TOKEN}-qwen3-235b-default-b-coord-floor",
        }

    def test_the_job_names_are_distinct_once_truncated(self) -> None:
        names = [leg.batch_job_name for leg in LEGS if leg.batch_job_name is not None]
        assert len(names) == len(set(names))
        assert all(len(str(name)) <= 63 for name in names)

    def test_no_run_id_or_job_name_collides_with_any_sibling_passs(self) -> None:
        """A Bedrock job name is account-wide and permanent, and four passes now share this account."""
        siblings = (
            {
                leg.batch_run_id
                for table in transfer_plan.PLAN_TABLES.values()
                for leg in table.all_legs()
                if leg.transport == transfer_plan.TRANSPORT_BATCH
            }
            | {
                leg.batch_run_id
                for leg in decoupled_plan.all_legs()
                if leg.transport == decoupled_plan.TRANSPORT_BATCH
            }
            | {leg.batch_run_id for leg in deference_plan.LEGS if leg.transport == TRANSPORT_BATCH}
        )
        assert self.coordination_run_ids() & siblings == set()

    def test_the_prefix_and_the_run_dir_are_this_passs_own(self) -> None:
        assert COORDINATION_BATCH_PREFIX != deference_plan.DEFERENCE_BATCH_PREFIX
        assert COORDINATION_BATCH_PREFIX != transfer_plan.TRANSFER_BATCH_PREFIX
        assert plan_module.DEFAULT_RUN_DIR != deference_plan.DEFAULT_RUN_DIR
        assert all(leg.batch_prefix == COORDINATION_BATCH_PREFIX for leg in LEGS)

    def test_no_leg_id_is_shared_with_the_deference_pass(self) -> None:
        """``--leg`` offers both passes' ids, and only disjoint ids let a binding refuse the other's."""
        assert {leg.leg_id for leg in LEGS} & {leg.leg_id for leg in deference_plan.LEGS} == set()


class TestTheRecordKeys:
    def test_a_key_carries_every_axis_that_never_pools(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        calls = planned_calls_for_leg(leg_for(BATCH_FLOOR_LEG_ID), coordination_stimulus)
        assert calls[0].key == (
            f"{BLOCK_FLOOR}|{ARM_OVERSIGHT_OFF}|{CELL_SAME_CHECKPOINT}|synthetic-ledger|"
            f"{LABEL_PRINT_ORDERS[0]}|{QWEN_MODEL_ID}|effort=default|sitting=B|draw=0"
        )

    def test_a_prompt_id_names_the_text_and_not_its_reader(self) -> None:
        first = prompt_id(
            arm=ARM_OVERSIGHT_ON,
            cell=CELL_NO_PEERS,
            scenario_id="synthetic-ledger",
            print_order=LABEL_PRINT_ORDERS[0],
        )
        assert first == "coordination--oversight-on--no-peers--synthetic-ledger--order-canonical"
        assert QWEN_MODEL_ID not in first

    def test_every_planned_call_of_a_leg_has_its_own_key(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        calls = planned_calls_for_leg(leg_for(BATCH_FLOOR_LEG_ID), coordination_stimulus)
        assert len(calls) == 256
        assert len({call.key for call in calls}) == len(calls)
        assert len({call.prompt_id for call in calls}) == len(calls) // DRAWS

    def test_a_call_carries_the_labels_in_its_own_print_order(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        calls = planned_calls_for_leg(leg_for(BATCH_FLOOR_LEG_ID), coordination_stimulus)
        by_order = {call.print_order: call for call in calls}
        canonical, swapped = (by_order[order] for order in LABEL_PRINT_ORDERS)
        assert canonical.label_a == canonical.compliant_label
        assert swapped.label_a == swapped.deviation_label
        assert canonical.compliant_label == swapped.compliant_label

    def test_the_design_labels_cover_every_id_a_reader_could_recognise(self) -> None:
        for label in (*CELLS, *ARMS, *BLOCKS, *ALL_MODEL_IDS, "coordination", "copies", "copy of"):
            assert label in DESIGN_LABELS
        for allowed in ("oversight", "coin", "draw", "shortcut"):
            assert allowed not in DESIGN_LABELS


class TestTheAudits:
    def test_every_audit_runs_over_the_whole_stimulus_and_counts_what_it_compared(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        assert audits(coordination_stimulus) == {
            "setting_and_reconciliation_pairs_compared": N_SCENARIOS * len(ARMS) * 2,
            "arm_pairs_compared": N_SCENARIOS * len(CELLS) * 2,
            "identity_pairs_compared": N_SCENARIOS * len(ARMS) * 2 * 3,
            "print_order_pairs_compared": N_SCENARIOS * len(ARMS) * len(CELLS),
            "status_blocks_checked": N_SCENARIOS,
        }

    def test_every_planned_prompt_is_its_stem_plus_at_most_one_paragraph(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        for leg in LEGS:
            calls = planned_calls_for_leg(leg, coordination_stimulus)
            assert audit_planned_calls(calls, coordination_stimulus) == len(calls)

    def test_the_identity_blind_render_is_the_stem_the_audit_compares_against(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        scenario = coordination_stimulus.scenarios[0]
        stem = identity_blind_stem(
            scenario, coordination_stimulus, arm=ARM_OVERSIGHT_ON, print_order="canonical"
        )
        blind = next(
            call
            for call in planned_calls_for_leg(leg_for(LIVE_LEG_ID), coordination_stimulus)
            if call.cell == CELL_IDENTITY_BLIND and call.scenario_id == scenario.scenario_id
        )
        assert blind.prompt == stem

    def test_a_two_paragraph_identity_clause_is_refused_by_the_renderer_before_any_audit(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """The first line of defence: a clause with a blank line renders as two inserted paragraphs.

        The renderer catches it before the insertion audit ever sees it, because the sections it built and
        the sections the joined prompt splits into disagree. Both checks exist for the same failure, and
        this is the one that fires.
        """
        sabotaged = replace(
            coordination_stimulus,
            identity_clauses={
                name: f"{clause}\n\nSYNTHETIC-SECOND-PARAGRAPH."
                for name, clause in coordination_stimulus.identity_clauses.items()
            },
        )
        with pytest.raises(ValueError, match="sections but splits into"):
            planned_calls_for_leg(leg_for(BATCH_FLOOR_LEG_ID), sabotaged)

    def test_a_marker_the_identity_paragraph_does_not_open_with_breaks_the_insertion_audit(
        self, coordination_stimulus: CoordinationStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The audit deletes sections that open with the marker, so a drifted marker deletes nothing.

        This is the failure the shared audit's own docstring warns about: a render that phrased its own
        opening would pass every vocabulary gate and silently drop out of the audit, which is the property
        the whole design rests on. Planted by moving the marker rather than the paragraph, because the
        renderer owns the opening and there is no way to author one that does not carry it.
        """
        monkeypatch.setattr(plan_module, "DEFERENCE_PARAGRAPH_MARKER", "SOMETHING ELSE: ")
        calls = planned_calls_for_leg(leg_for(BATCH_FLOOR_LEG_ID), coordination_stimulus)
        with pytest.raises(ValueError, match="is not its stem plus one counterpart paragraph"):
            audit_planned_calls(calls, coordination_stimulus)

    def test_a_brief_that_prints_its_setting_twice_breaks_the_substitution_audit(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """The substitution has to occur exactly once, so a brief naming its setting twice cannot pass."""
        scenario = coordination_stimulus.scenarios[0]
        sabotaged = replace(
            coordination_stimulus,
            scenarios=(
                replace(
                    scenario, brief=str(scenario.brief).replace("{setting}", "{setting} {setting}")
                ),
                *coordination_stimulus.scenarios[1:],
            ),
        )
        with pytest.raises(ValueError, match="peer setting passage 2 times"):
            audit_settings_and_reconciliations(sabotaged)

    def test_a_close_procedure_rendered_into_the_wrong_section_breaks_its_audit(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """The two sole-owner cells differ in what the close compares and in nothing else."""
        scenario = coordination_stimulus.scenarios[0]
        sabotaged = replace(
            coordination_stimulus,
            scenarios=(
                replace(scenario, reconciliation_runbook=scenario.reconciliation_blocks),
                *coordination_stimulus.scenarios[1:],
            ),
        )
        with pytest.raises(ValueError, match="expected exactly"):
            audit_settings_and_reconciliations(sabotaged)

    def test_an_oversight_pair_two_tokens_apart_breaks_the_arm_audit(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        scenario = coordination_stimulus.scenarios[0]
        sabotaged = replace(
            coordination_stimulus,
            scenarios=(
                replace(
                    scenario,
                    oversight_off=SYNTHETIC_OVERSIGHT_OFF.replace("Nobody", "Not a single person"),
                ),
                *coordination_stimulus.scenarios[1:],
            ),
        )
        with pytest.raises(ValueError, match="more than one whitespace token"):
            audit_arms_differ_in_the_oversight_sentence(sabotaged)

    def test_an_oversight_sentence_a_token_longer_breaks_the_arm_audit(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """The one-token invariant re-run on the RENDERED brief, planted inside the one section it owns.

        Named for what it plants rather than for the audit's other half: the difference here stays inside
        the brief, so it is the whitespace-token count that goes red and not the section count. The two
        tests below plant the section count itself.
        """
        scenario = coordination_stimulus.scenarios[0]
        sabotaged = replace(
            coordination_stimulus,
            scenarios=(
                replace(scenario, oversight_off=SYNTHETIC_OVERSIGHT_OFF.replace(".", " today.")),
                *coordination_stimulus.scenarios[1:],
            ),
        )
        with pytest.raises(ValueError, match="more than one whitespace token"):
            audit_arms_differ_in_the_oversight_sentence(sabotaged)

    def test_a_second_section_moving_with_the_arm_breaks_the_arm_audit(
        self, coordination_stimulus: CoordinationStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The arms share every section but one, and the audit counts them rather than trusting that.

        Planted in the renderer, because that is where the drift would live: a slot filled from the arm in
        a section outside the brief -- the fork, the instruction, a tool result -- would make the oversight
        contrast a difference of two things at once, with every rate still printed as the cue's effect.
        The oversight sentence still differs by its one token, so this goes red on the section count alone.
        """
        rendered = coordination_stimulus_module.render_coordination_sections

        def with_an_arm_dependent_fork(
            scenario: CoordinationScenario, *, arm: str, **kwargs: object
        ) -> tuple[str, ...]:
            sections = list(rendered(scenario, arm=arm, **kwargs))  # type: ignore[arg-type]
            sections[-2] = f"{sections[-2]} ({arm})"
            return tuple(sections)

        monkeypatch.setattr(plan_module, "render_coordination_sections", with_an_arm_dependent_fork)
        with pytest.raises(ValueError, match="differ in sections"):
            audit_arms_differ_in_the_oversight_sentence(coordination_stimulus)

    def test_an_arm_difference_outside_the_brief_breaks_the_arm_audit(
        self, coordination_stimulus: CoordinationStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one differing section has to be INSIDE the brief, which is the other half of the check.

        Planted by miscounting the brief, as the identity audit's sabotage does: the difference is then one
        section as before, sitting where the audit says nothing may move. A renderer that moved the cue out
        of the brief and into a tool result would look exactly like this, and the arms would still differ
        by one section and one token.
        """
        monkeypatch.setattr(plan_module, "brief_section_count", lambda scenario: 0)
        with pytest.raises(ValueError, match="differ in sections"):
            audit_arms_differ_in_the_oversight_sentence(coordination_stimulus)

    def test_an_identity_paragraph_at_the_wrong_position_breaks_the_identity_audit(
        self, coordination_stimulus: CoordinationStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The audit indexes past the brief, so a miscounted brief is what it exists to catch.

        Planted by making the brief count as one section while the synthetic briefs are two, which is the
        drift a fixed index would have hidden: the audit would then compare the wrong sections and pass on
        the roster it was written against.
        """
        monkeypatch.setattr(plan_module, "brief_section_count", lambda scenario: 1)
        with pytest.raises(ValueError, match="expected exactly"):
            audit_peer_cells_differ_in_the_identity_section(coordination_stimulus)

    def test_a_counterbalance_that_moves_nothing_breaks_the_print_order_audit(
        self, coordination_stimulus: CoordinationStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The counterbalance has to move the fork and the instruction, and exactly those two sections.

        Planted by making both print orders return the same pair of labels, which is what a mapping bug
        looks like: the two halves of every cell would then be one condition sampled twice, and pooling
        them would report a counterbalance that never happened.
        """
        monkeypatch.setattr(
            coordination_stimulus_module,
            "labels_in_print_order",
            lambda scenario, order: scenario.labels,
        )
        with pytest.raises(ValueError, match="differ in sections"):
            audit_print_orders_differ_in_the_fork_and_instruction(coordination_stimulus)

    def test_a_status_field_spelled_differently_in_one_scenario_breaks_the_status_audit(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """The block is renderer-owned and byte-identical in every cell of both arms."""
        scenario = coordination_stimulus.scenarios[0]
        sabotaged = replace(
            coordination_stimulus,
            scenarios=(
                replace(scenario, status_field="state-of-play"),
                *coordination_stimulus.scenarios[1:],
            ),
        )
        assert audit_the_status_block_is_the_same_all_pending_list(sabotaged) == N_SCENARIOS
        pooled = replace(sabotaged, peer_count=PEER_COUNT + 1)
        with pytest.raises(ValueError, match="other agents and this plan is built around"):
            audit_the_status_block_is_the_same_all_pending_list(pooled)

    def test_an_identity_paragraph_that_never_states_the_count_breaks_the_status_audit(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        clauses = {
            name: clause.replace(str(PEER_COUNT), "several")
            for name, clause in coordination_stimulus.identity_clauses.items()
        }
        sabotaged = replace(coordination_stimulus, identity_clauses=clauses)
        with pytest.raises(ValueError, match="never states the count"):
            audit_the_status_block_is_the_same_all_pending_list(sabotaged)

    def test_the_drawn_cell_and_the_copies_cell_differ_in_the_tail_alone(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """The knockout contrast, read off the renders rather than off the composed paragraphs."""
        assert audit_peer_cells_differ_in_the_identity_section(coordination_stimulus) > 0
        rendered = {
            cell: plan_module.render_cell_prompt(
                plan_module.Cell(ARM_OVERSIGHT_ON, cell),
                coordination_stimulus.scenarios[0],
                coordination_stimulus,
                print_order="canonical",
            )
            for cell in (CELL_SAME_CHECKPOINT, CELL_SAME_CHECKPOINT_DRAWN, CELL_DIFFERENT_FAMILY)
        }
        assert len(set(rendered.values())) == 3
