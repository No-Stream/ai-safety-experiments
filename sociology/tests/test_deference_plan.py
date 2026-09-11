"""The deference plan: the leg table's arithmetic, the record keys, and the four structural audits.

The audits are the point of this module, and each is sabotaged here as well as exercised: a rendered cell
that is not its peer-free stem plus one paragraph, a report block rendered where the design says a cell
carries none, two arms differing anywhere but the constraint sentence, and a status report whose line count
disagrees with the number the identity paragraph states. Every one of those failures leaves a complete,
plausible run behind, so each is planted deliberately and required to go red.

The cross-pass job-name test is here rather than in the sibling passes' files because this is the pass whose
block ids collide with theirs on purpose: ``floor`` is a name two earlier passes have already asked Bedrock
for, and the pass token is what keeps the third claim distinct.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from games.prompts import LABEL_PRINT_ORDERS
from sociology import coordination_plan, decoupled_plan, transfer_plan
from sociology import deference_plan as plan_module
from sociology.deference_plan import (
    ALL_MODEL_IDS,
    ARM_BREAKING,
    BATCH_PASS_TOKEN,
    BLOCK_BENIGN,
    BLOCK_BREAKING,
    BLOCK_FLOOR,
    BLOCKS,
    CELLS_BY_BLOCK,
    DEFERENCE_BATCH_PREFIX,
    DESIGN_LABELS,
    DRAWS,
    FLOOR_CELLS,
    LEGS,
    QWEN_MODEL_ID,
    READING_BLOCKS,
    ROWS_PER_CELL,
    SITTING_A,
    SITTING_B,
    SONNET_MODEL_ID,
    TRANSPORT_BATCH,
    TRANSPORT_LIVE,
    Cell,
    audit_arms_differ_in_the_constraint_sentence,
    audit_planned_calls,
    audit_print_orders_differ_in_the_fork_and_instruction,
    audit_the_peer_count_is_stated_once,
    audit_the_report_is_one_added_section,
    cells_for,
    leg_for,
    peer_free_stem,
    planned_calls_for_leg,
    prompt_id,
    record_key,
    refuse_below_batch_floor,
)
from sociology.deference_stimulus import (
    ARMS,
    CELL_DIFFERENT_FAMILY_CHOSE,
    CELL_NO_PEERS,
    CELL_SAME_CHECKPOINT_CHOSE,
    CELL_SAME_CHECKPOINT_PRESENT,
    CELLS,
    N_SCENARIOS,
    PEER_COUNT,
    render_deference_sections,
    status_report_block,
)

if TYPE_CHECKING:
    from sociology.deference_plan import PlannedCall
    from sociology.deference_stimulus import DeferenceScenario, DeferenceStimulus

EXPECTED_TOTAL_RECORDS = 3_840
"""Three models x (512 + 512 + 256). Spelled out, because the design doc states this number and a table
whose arithmetic drifted from it would be read against a plan nobody re-derived."""

BREAKING_LEG_ID = "sonnet-5--default--A--breaking"
FLOOR_LEG_ID = "qwen3-235b--default--B--floor"


class TestTheLegTable:
    def test_every_roster_row_has_all_three_blocks(self) -> None:
        by_model: dict[str, list[str]] = {}
        for leg in LEGS:
            by_model.setdefault(leg.model_id, []).append(leg.block)
        assert set(by_model) == set(ALL_MODEL_IDS)
        assert all(blocks == list(BLOCKS) for blocks in by_model.values())

    def test_the_record_counts_are_the_designs_own_arithmetic(self) -> None:
        assert N_SCENARIOS * len(LABEL_PRINT_ORDERS) == ROWS_PER_CELL
        assert leg_for(BREAKING_LEG_ID).records == 4 * ROWS_PER_CELL * DRAWS
        assert leg_for(FLOOR_LEG_ID).records == 2 * ROWS_PER_CELL * DRAWS
        assert sum(leg.records for leg in LEGS) == EXPECTED_TOTAL_RECORDS

    def test_the_floor_is_a_second_sitting_of_the_two_cells_the_headline_reads(self) -> None:
        floor = leg_for(FLOOR_LEG_ID)
        assert floor.sitting == SITTING_B
        assert [cell.cell_id for cell in floor.cells] == list(FLOOR_CELLS)
        assert all(cell.arm == ARM_BREAKING for cell in floor.cells)
        assert all(leg.sitting == SITTING_A for leg in LEGS if leg.block in READING_BLOCKS)

    def test_each_reading_block_samples_every_cell_of_its_own_arm(self) -> None:
        for block, arm in ((BLOCK_BENIGN, ARMS[0]), (BLOCK_BREAKING, ARMS[1])):
            cells = CELLS_BY_BLOCK[block]
            assert [cell.cell_id for cell in cells] == list(CELLS)
            assert {cell.arm for cell in cells} == {arm}

    def test_the_transports_follow_the_roster(self) -> None:
        by_id = {leg.leg_id: leg for leg in LEGS}
        assert by_id[BREAKING_LEG_ID].transport == TRANSPORT_LIVE
        assert by_id[FLOOR_LEG_ID].transport == TRANSPORT_BATCH
        assert by_id[BREAKING_LEG_ID].batch_job_name is None

    def test_an_unknown_leg_id_lists_the_table(self) -> None:
        with pytest.raises(ValueError, match="is not a leg of this plan"):
            leg_for("nobody--default--A--benign")

    def test_cells_for_refuses_an_unknown_arm_or_cell(self) -> None:
        with pytest.raises(ValueError, match="not an arm"):
            cells_for("sideways", CELLS)
        with pytest.raises(ValueError, match="not cells of this design"):
            cells_for(ARM_BREAKING, ("no-such-cell",))

    def test_the_batch_floor_refusal_names_the_arithmetic(self) -> None:
        refuse_below_batch_floor(leg_for(FLOOR_LEG_ID), 256)
        with pytest.raises(ValueError, match="at least"):
            refuse_below_batch_floor(leg_for(FLOOR_LEG_ID), 99)


class TestTheBatchNames:
    def deference_run_ids(self) -> set[str]:
        return {leg.batch_run_id for leg in LEGS if leg.transport == TRANSPORT_BATCH}

    def test_every_batch_run_id_opens_with_this_passs_token(self) -> None:
        assert self.deference_run_ids() == {
            f"{BATCH_PASS_TOKEN}-qwen3-235b-default-a-benign",
            f"{BATCH_PASS_TOKEN}-qwen3-235b-default-a-breaking",
            f"{BATCH_PASS_TOKEN}-qwen3-235b-default-b-floor",
        }

    def test_the_job_names_are_distinct_once_truncated(self) -> None:
        names = [leg.batch_job_name for leg in LEGS if leg.batch_job_name is not None]
        assert len(names) == len(set(names))
        assert all(len(str(name)) <= 63 for name in names)

    def test_no_run_id_or_job_name_collides_with_a_sibling_passs(self) -> None:
        """A Bedrock job name is account-wide and permanent, and ``floor`` is a third claim on one id.

        Every table that plans a batch leg on this account is swept, and a new pass has to be added here:
        the check is only as wide as the tables it knows about, and the ``ConflictException`` it exists to
        prevent arrives at submit time with a paid upload already in S3.
        """
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
            | {
                leg.batch_run_id
                for leg in coordination_plan.LEGS
                if leg.transport == coordination_plan.TRANSPORT_BATCH
            }
        )
        assert self.deference_run_ids() & siblings == set()
        sibling_names = {
            leg.batch_job_name
            for table in transfer_plan.PLAN_TABLES.values()
            for leg in table.all_legs()
            if leg.batch_job_name is not None
        } | {leg.batch_job_name for leg in coordination_plan.LEGS if leg.batch_job_name is not None}
        ours = {leg.batch_job_name for leg in LEGS if leg.batch_job_name is not None}
        assert ours & sibling_names == set()

    def test_the_two_passes_on_one_cli_share_no_leg_id(self) -> None:
        """``--leg`` offers both passes' ids, and only disjoint ids let a binding refuse the other's."""
        assert {leg.leg_id for leg in LEGS} & {
            leg.leg_id for leg in coordination_plan.LEGS
        } == set()

    def test_the_prefix_is_this_passs_own(self) -> None:
        assert DEFERENCE_BATCH_PREFIX != transfer_plan.TRANSFER_BATCH_PREFIX
        assert DEFERENCE_BATCH_PREFIX != decoupled_plan.DECOUPLED_BATCH_PREFIX


class TestTheRecordKeys:
    def test_a_key_carries_every_axis_that_never_pools(self) -> None:
        key = record_key(
            block=BLOCK_FLOOR,
            arm=ARM_BREAKING,
            cell=CELL_SAME_CHECKPOINT_CHOSE,
            scenario_id="synthetic-ledger",
            print_order=LABEL_PRINT_ORDERS[0],
            model_id=SONNET_MODEL_ID,
            reasoning_effort=None,
            sitting=SITTING_B,
            draw=3,
        )
        assert key == (
            f"{BLOCK_FLOOR}|{ARM_BREAKING}|{CELL_SAME_CHECKPOINT_CHOSE}|synthetic-ledger|"
            f"{LABEL_PRINT_ORDERS[0]}|{SONNET_MODEL_ID}|effort=default|sitting=B|draw=3"
        )

    def test_a_prompt_id_names_the_text_and_not_its_reader(self) -> None:
        first = prompt_id(
            arm=ARM_BREAKING,
            cell=CELL_NO_PEERS,
            scenario_id="synthetic-ledger",
            print_order=LABEL_PRINT_ORDERS[0],
        )
        assert first == "deference--breaking--no-peers--synthetic-ledger--order-canonical"
        assert SONNET_MODEL_ID not in first

    def test_every_planned_call_of_a_leg_has_its_own_key(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        calls = planned_calls_for_leg(leg_for(FLOOR_LEG_ID), deference_stimulus)
        assert len(calls) == 256
        assert len({call.key for call in calls}) == len(calls)
        # One prompt is read by DRAWS calls of one leg, so the prompt ids are the keys divided by draws.
        assert len({call.prompt_id for call in calls}) == len(calls) // DRAWS

    def test_a_call_carries_the_labels_in_its_own_print_order_and_the_mapping_beside_them(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        calls = planned_calls_for_leg(leg_for(FLOOR_LEG_ID), deference_stimulus)
        by_order = {call.print_order: call for call in calls}
        canonical, swapped = (by_order[order] for order in LABEL_PRINT_ORDERS)
        assert canonical.label_a == canonical.compliant_label
        assert swapped.label_a == swapped.deviation_label
        assert canonical.compliant_label == swapped.compliant_label
        assert {call.metadata()["compliant_label"] for call in calls} == {canonical.compliant_label}

    def test_the_metadata_the_batch_digest_hashes_carries_no_prompt_text(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        call = planned_calls_for_leg(leg_for(FLOOR_LEG_ID), deference_stimulus)[0]
        assert "prompt" not in call.metadata()
        assert "SYNTHETIC-BRIEF" not in str(call.metadata())


class TestTheOneInsertedParagraphAudit:
    """Gate 2's first half, plus the two sabotages that make it a check rather than a message."""

    def calls(
        self, stimulus: DeferenceStimulus, leg_id: str = BREAKING_LEG_ID
    ) -> list[PlannedCall]:
        return planned_calls_for_leg(leg_for(leg_id), stimulus)

    def test_every_planned_prompt_is_its_stem_plus_one_paragraph(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        calls = self.calls(deference_stimulus)
        assert audit_planned_calls(calls, deference_stimulus) == len(calls)

    def test_the_peer_free_cell_is_byte_equal_to_the_stem_the_audit_uses(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        call = next(call for call in self.calls(deference_stimulus) if call.cell == CELL_NO_PEERS)
        assert call.prompt == peer_free_stem(
            deference_stimulus.scenario(call.scenario_id),
            deference_stimulus,
            arm=call.arm,
            print_order=call.print_order,
        )

    def test_a_two_paragraph_identity_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """Two inserted paragraphs never reach the audit: the render itself refuses to build them.

        The renderer's sections are the joined prompt's blank-line split, so a clause carrying a break
        makes the two disagree and the render stops there, ahead of the audit and ahead of any call.
        The authored path is refused a layer earlier still, by the loader's own break check.
        """
        sabotaged = replace(
            deference_stimulus,
            identity_clauses={
                name: clause.replace(", SYNTHETIC-TAIL", ".\n\nSYNTHETIC-SECOND-PARAGRAPH")
                for name, clause in deference_stimulus.identity_clauses.items()
            },
        )
        with pytest.raises(ValueError, match=r"builds .* sections but splits into"):
            audit_planned_calls(self.calls(sabotaged), sabotaged)

    def test_a_report_rendered_in_a_cell_that_carries_none_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """Gate 2's second sabotage: the present cell's prompt with the report block in it."""
        call = next(
            call
            for call in self.calls(deference_stimulus)
            if call.cell == CELL_SAME_CHECKPOINT_PRESENT
        )
        scenario = deference_stimulus.scenario(call.scenario_id)
        sections = call.prompt.split("\n\n")
        smuggled = replace(
            call,
            prompt="\n\n".join(
                (*sections[:-2], status_report_block(scenario, deference_stimulus), *sections[-2:])
            ),
        )
        with pytest.raises(ValueError, match="not its stem plus one counterpart paragraph"):
            audit_planned_calls([smuggled], deference_stimulus)

    def test_a_report_cell_whose_report_moved_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """The report's position is what makes its contrast a read of one inserted block."""
        call = next(
            call
            for call in self.calls(deference_stimulus)
            if call.cell == CELL_SAME_CHECKPOINT_CHOSE
        )
        sections = call.prompt.split("\n\n")
        moved = replace(call, prompt="\n\n".join((*sections[:2], *sections[3:], sections[2])))
        with pytest.raises(ValueError, match="does not carry its status report at section"):
            audit_planned_calls([moved], deference_stimulus)


class TestTheReportAudit:
    """Gate 2's second half: a report render is its own identity's silent render plus one block."""

    def test_every_pair_is_one_added_section(self, deference_stimulus: DeferenceStimulus) -> None:
        """Eight scenarios x two arms x two report cells x two print orders."""
        assert audit_the_report_is_one_added_section(deference_stimulus) == 64

    def test_the_different_family_pair_is_compared_against_its_own_silent_render(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """That render is not a cell this pass samples; rendering it is what makes the pair matched."""
        scenario = deference_stimulus.scenarios[0]
        silent = render_deference_sections(
            scenario,
            arm=ARM_BREAKING,
            identity="different-family",
            with_status_report=False,
            print_order=LABEL_PRINT_ORDERS[0],
            stimulus=deference_stimulus,
        )
        reported = render_deference_sections(
            scenario,
            arm=ARM_BREAKING,
            identity="different-family",
            with_status_report=True,
            print_order=LABEL_PRINT_ORDERS[0],
            stimulus=deference_stimulus,
        )
        assert len(reported) == len(silent) + 1
        assert CELL_DIFFERENT_FAMILY_CHOSE in CELLS


class TestTheArmAudit:
    """Gate 4's first half: the two arms differ in one brief section, and in one sentence inside it."""

    def test_every_pair_differs_in_the_constraint_sentence_alone(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """Eight scenarios x four cells x two print orders."""
        assert audit_arms_differ_in_the_constraint_sentence(deference_stimulus) == 64

    def test_a_brief_that_states_its_constraint_twice_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """Two slots make the arm difference two sentences while every count still adds up."""
        scenario = deference_stimulus.scenarios[0]
        sabotaged = replace(
            deference_stimulus,
            scenarios=(replace(scenario, brief=f"{scenario.brief} And again: {{constraint}}"),),
        )
        with pytest.raises(ValueError, match="carries its own benign constraint sentence"):
            audit_arms_differ_in_the_constraint_sentence(sabotaged)

    def test_a_render_that_differs_outside_the_brief_is_refused(
        self, deference_stimulus: DeferenceStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing outside the brief takes the arm today, and the audit is what keeps it that way."""
        real = plan_module.render_deference_sections

        def with_the_arm_in_the_instruction(  # noqa: PLR0913 - the renderer's own signature
            scenario: DeferenceScenario,
            *,
            arm: str,
            identity: str | None,
            with_status_report: bool,
            print_order: str,
            stimulus: DeferenceStimulus,
        ) -> tuple[str, ...]:
            sections = real(
                scenario,
                arm=arm,
                identity=identity,
                with_status_report=with_status_report,
                print_order=print_order,
                stimulus=stimulus,
            )
            return (*sections[:-1], f"{sections[-1]} ({arm})")

        monkeypatch.setattr(
            plan_module, "render_deference_sections", with_the_arm_in_the_instruction
        )
        with pytest.raises(ValueError, match="expected exactly one section inside the brief"):
            audit_arms_differ_in_the_constraint_sentence(deference_stimulus)


class TestThePrintOrderAudit:
    """Gate 4's second half: the counterbalance moves the fork and the instruction and nothing else."""

    def test_every_pair_differs_in_the_last_two_sections(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """Eight scenarios x two arms x four cells."""
        assert audit_print_orders_differ_in_the_fork_and_instruction(deference_stimulus) == 64

    def test_a_render_whose_brief_moves_with_the_order_is_refused(
        self, deference_stimulus: DeferenceStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A counterbalance that reached the brief would be a second manipulation, not a page order."""
        real = plan_module.render_deference_sections

        def with_the_order_in_the_brief(  # noqa: PLR0913 - the renderer's own signature
            scenario: DeferenceScenario,
            *,
            arm: str,
            identity: str | None,
            with_status_report: bool,
            print_order: str,
            stimulus: DeferenceStimulus,
        ) -> tuple[str, ...]:
            sections = real(
                scenario,
                arm=arm,
                identity=identity,
                with_status_report=with_status_report,
                print_order=print_order,
                stimulus=stimulus,
            )
            return (f"{sections[0]} ({print_order})", *sections[1:])

        monkeypatch.setattr(plan_module, "render_deference_sections", with_the_order_in_the_brief)
        with pytest.raises(ValueError, match="differ in sections"):
            audit_print_orders_differ_in_the_fork_and_instruction(deference_stimulus)


class TestThePeerCountAudit:
    """Gate 4's last clause: the number the paragraph states IS the report's line count."""

    def test_every_scenarios_report_lists_the_stated_number_of_agents(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        assert audit_the_peer_count_is_stated_once(deference_stimulus) == N_SCENARIOS

    def test_a_stimulus_stating_another_count_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        sabotaged = replace(deference_stimulus, peer_count=PEER_COUNT - 1)
        with pytest.raises(ValueError, match="this plan is built around"):
            audit_the_peer_count_is_stated_once(sabotaged)

    def test_a_report_block_with_an_extra_handle_is_refused(
        self, deference_stimulus: DeferenceStimulus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reader counts handles, not lines the renderer promised: a seventh one goes red."""
        real = plan_module.status_report_block

        def with_an_extra_handle(scenario: DeferenceScenario, stimulus: DeferenceStimulus) -> str:
            return real(scenario, stimulus).replace(
                "</stdout>", "\nagent-99  order: sorted-order</stdout>"
            )

        monkeypatch.setattr(plan_module, "status_report_block", with_an_extra_handle)
        with pytest.raises(ValueError, match="carries 7 handles"):
            audit_the_peer_count_is_stated_once(deference_stimulus)

    def test_an_identity_paragraph_that_states_no_count_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        sabotaged = replace(
            deference_stimulus,
            identity_clauses={
                name: clause.replace(str(PEER_COUNT), "several")
                for name, clause in deference_stimulus.identity_clauses.items()
            },
        )
        with pytest.raises(ValueError, match="never states the peer count"):
            audit_the_peer_count_is_stated_once(sabotaged)


class TestTheDesignLabels:
    def test_they_name_every_cell_arm_block_model_and_identity(self) -> None:
        for label in (*CELLS, *ARMS, *BLOCKS, *ALL_MODEL_IDS):
            assert label in DESIGN_LABELS
        assert "deference" in DESIGN_LABELS
        assert "copies" in DESIGN_LABELS
        assert "copy of" in DESIGN_LABELS

    def test_they_carry_the_bare_identities_as_well_as_the_cell_ids(self) -> None:
        """A rubric may not say "same-checkpoint" even where no cell id spells it that way."""
        assert "same-checkpoint" in DESIGN_LABELS
        assert "different-family" in DESIGN_LABELS


class TestTheCell:
    def test_a_cell_is_an_arm_and_a_peer_condition(self) -> None:
        cell = Cell(ARM_BREAKING, CELL_NO_PEERS)
        assert (cell.arm, cell.cell_id) == (ARM_BREAKING, CELL_NO_PEERS)
        assert leg_for(FLOOR_LEG_ID).arm == ARM_BREAKING
        assert leg_for(FLOOR_LEG_ID).model_id == QWEN_MODEL_ID
