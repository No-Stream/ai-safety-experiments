"""The deference judge: blindness, the two schemas, the coverage gates, and the reversal arithmetic.

Nothing here calls a model. The judge runs against a scripted backend, which is what makes the blindness
check testable at all: the prompts the backend actually received are read back and swept for every string
that names the design, because a prompt rendered by hand in a test proves nothing about the path production
takes.

Two of the classes are coverage gates over the authored file rather than behaviours of the code. They exist
because a validation set that never registers a flag both ways, or that carries a reversal in one direction
only, clears a reader that answered False to everything or that read conclusions one way round -- and this
pass reports those flags as findings and uses the reversals to correct a published rate.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from sociology.deference_judge import (
    ACTION_NONE,
    CARRIED_FIELDS,
    CONCLUDED_NO_REASONING,
    CONCLUDED_UNCLEAR,
    CROSS_JUDGE_MODEL_ID,
    CROSS_JUDGE_RECORDS,
    DEFERENCE_JUDGE,
    DEFERENCE_VERDICT_SCHEMA,
    INTENT_CHECK_PROMPT_VERSION,
    INTENT_CHECK_SCHEMA,
    INTENT_COUNT_NAMES,
    JUDGE_MODEL_ID,
    JUDGE_PROMPT_VERSION,
    PROMPT_SCAFFOLD,
    VERDICT_BOOL_FIELDS,
    JudgeReplyError,
    assert_intent_validation_covers_both_slip_directions,
    corrected_chosen_label,
    intent_check_cell_key,
    intent_check_counts,
    intent_prompt,
    is_tag_slip,
    judge_prompt,
    parse_verdict,
    prompt_shape_digest,
)
from sociology.deference_plan import DESIGN_LABELS
from sociology.deference_stimulus import INTENT_VERDICT_KEYS, VERDICT_KEYS, load_stimulus
from sociology.model_stub import ScriptedDetailedBackend
from sociology.tests.conftest import (
    SYNTHETIC_DEFERENCE_LABEL_A,
    SYNTHETIC_DEFERENCE_LABEL_B,
    synthetic_deference_intent_set,
    synthetic_deference_validation_reply,
    synthetic_deference_validation_set,
    write_synthetic_deference_stimulus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sociology.deference_stimulus import DeferenceStimulus

LABELS = (SYNTHETIC_DEFERENCE_LABEL_A, SYNTHETIC_DEFERENCE_LABEL_B)


def verdict_json(**overrides: Any) -> str:
    """One well-formed verdict for the rubric of record, with any field overridden."""
    verdict: dict[str, Any] = {"action_label": SYNTHETIC_DEFERENCE_LABEL_A, "evidence": "quoted"}
    verdict.update(dict.fromkeys(VERDICT_BOOL_FIELDS, False))
    verdict.update(overrides)
    return json.dumps(verdict)


def intent_json(**overrides: Any) -> str:
    """One well-formed intent verdict, with any field overridden."""
    verdict: dict[str, Any] = {
        "concluded_action": SYNTHETIC_DEFERENCE_LABEL_A,
        "evidence": "quoted",
    }
    verdict.update(overrides)
    return json.dumps(verdict)


def record(**overrides: Any) -> dict[str, Any]:
    """One judgeable reply row, carrying the two labels every prompt is built from."""
    row: dict[str, Any] = {
        "key": "benign|benign|no-peers|synthetic-ledger|canonical|m|effort=default|sitting=A|draw=0",
        "reply": f"a reply <action>{SYNTHETIC_DEFERENCE_LABEL_A}</action>",
        "reasoning": "",
        "label_a": SYNTHETIC_DEFERENCE_LABEL_A,
        "label_b": SYNTHETIC_DEFERENCE_LABEL_B,
        "compliant_label": SYNTHETIC_DEFERENCE_LABEL_A,
        "deviation_label": SYNTHETIC_DEFERENCE_LABEL_B,
        "model_id": "synthetic-model",
        "arm": "breaking",
        "print_order": "canonical",
    }
    row.update(overrides)
    return row


class TestTheSchemas:
    def test_the_rubric_of_record_judges_exactly_what_the_loader_validates(self) -> None:
        judged = set(DEFERENCE_VERDICT_SCHEMA.verdict_fields) - {"evidence"}
        assert judged == set(VERDICT_KEYS)
        assert set(VERDICT_BOOL_FIELDS) == set(VERDICT_KEYS) - {"action_label"}

    def test_the_intent_check_judges_exactly_what_its_own_cases_register(self) -> None:
        judged = set(INTENT_CHECK_SCHEMA.verdict_fields) - {"evidence"}
        assert judged == set(INTENT_VERDICT_KEYS)

    def test_an_action_naming_a_third_label_is_an_error_rather_than_an_outcome(self) -> None:
        with pytest.raises(JudgeReplyError, match="is not one of"):
            parse_verdict(
                verdict_json(action_label="SOMETHING-ELSE"),
                record=record(),
                schema=DEFERENCE_VERDICT_SCHEMA,
            )

    def test_a_reply_that_never_commits_is_a_real_verdict(self) -> None:
        parsed = parse_verdict(
            verdict_json(action_label=ACTION_NONE), record=record(), schema=DEFERENCE_VERDICT_SCHEMA
        )
        assert parsed["action_label"] == ACTION_NONE

    def test_a_missing_flag_or_evidence_is_an_error(self) -> None:
        payload = json.loads(verdict_json())
        del payload["mirror"]
        with pytest.raises(JudgeReplyError, match="mirror"):
            parse_verdict(json.dumps(payload), record=record(), schema=DEFERENCE_VERDICT_SCHEMA)
        payload = json.loads(verdict_json())
        del payload["evidence"]
        with pytest.raises(JudgeReplyError, match="evidence"):
            parse_verdict(json.dumps(payload), record=record(), schema=DEFERENCE_VERDICT_SCHEMA)

    def test_the_intent_absences_are_allowed_and_a_third_label_is_not(self) -> None:
        for value in (CONCLUDED_UNCLEAR, CONCLUDED_NO_REASONING, SYNTHETIC_DEFERENCE_LABEL_B):
            parsed = parse_verdict(
                intent_json(concluded_action=value), record=record(), schema=INTENT_CHECK_SCHEMA
            )
            assert parsed["concluded_action"] == value
        with pytest.raises(JudgeReplyError, match="is not one of"):
            parse_verdict(
                intent_json(concluded_action="maybe"), record=record(), schema=INTENT_CHECK_SCHEMA
            )

    def test_the_two_readers_are_named_and_versioned_apart(self) -> None:
        assert JUDGE_PROMPT_VERSION != INTENT_CHECK_PROMPT_VERSION
        assert DEFERENCE_VERDICT_SCHEMA.name != INTENT_CHECK_SCHEMA.name
        assert JUDGE_MODEL_ID != CROSS_JUDGE_MODEL_ID
        assert CROSS_JUDGE_RECORDS > 0

    def test_the_judged_row_carries_the_reply_labels_it_joins_on(self) -> None:
        for name in ("model_id", "arm", "cell", "print_order", "compliant_label"):
            assert name in CARRIED_FIELDS


class TestBlindness:
    def test_neither_scaffold_names_the_design(self, deference_stimulus: DeferenceStimulus) -> None:
        DEFERENCE_JUDGE.assert_judge_scaffold_is_blind(deference_stimulus)
        DEFERENCE_JUDGE.assert_intent_scaffold_is_blind(deference_stimulus)

    def test_a_rubric_naming_a_cell_is_refused(self, tmp_path: Path) -> None:
        sabotaged = load_stimulus(
            write_synthetic_deference_stimulus(
                tmp_path / "s.json",
                judge_instructions="SYNTHETIC-RUBRIC: was this the same-checkpoint-chose cell?",
            )
        )
        with pytest.raises(ValueError, match="names the design"):
            DEFERENCE_JUDGE.assert_judge_scaffold_is_blind(sabotaged)

    def test_an_intent_rubric_naming_the_pass_is_refused(self, tmp_path: Path) -> None:
        sabotaged = load_stimulus(
            write_synthetic_deference_stimulus(
                tmp_path / "s.json", intent_rubric="SYNTHETIC-RUBRIC: score the deference."
            )
        )
        with pytest.raises(ValueError, match="names the design"):
            DEFERENCE_JUDGE.assert_intent_scaffold_is_blind(sabotaged)

    @pytest.mark.parametrize("label", sorted(DESIGN_LABELS))
    def test_no_fixed_header_of_either_prompt_carries_a_design_label(self, label: str) -> None:
        for section in (*PROMPT_SCAFFOLD, *INTENT_CHECK_SCHEMA.scaffold):
            assert label.lower() not in section.lower()

    def test_the_prompts_carry_the_two_labels_and_nothing_about_which_one_complies(self) -> None:
        rendered = judge_prompt(
            reply="a reply",
            reasoning="",
            label_a=SYNTHETIC_DEFERENCE_LABEL_A,
            label_b=SYNTHETIC_DEFERENCE_LABEL_B,
            instructions="SYNTHETIC-RUBRIC",
        )
        assert SYNTHETIC_DEFERENCE_LABEL_A in rendered
        assert "compliant" not in rendered.lower()
        assert "(none was returned" in rendered

    def test_the_intent_prompt_states_what_the_reply_was_asked_to_write(self) -> None:
        rendered = intent_prompt(
            reply="a reply",
            reasoning="some reasoning",
            label_a=SYNTHETIC_DEFERENCE_LABEL_A,
            label_b=SYNTHETIC_DEFERENCE_LABEL_B,
            instructions="SYNTHETIC-RUBRIC",
        )
        assert "WHAT THIS REPLY WAS ASKED TO WRITE" in rendered
        assert "some reasoning" in rendered

    def test_every_prompt_the_backend_actually_received_is_blind(
        self, tmp_path: Path, deference_stimulus: DeferenceStimulus
    ) -> None:
        """The path production takes, not a prompt rendered by hand in a test."""
        backend = ScriptedDetailedBackend([verdict_json()])
        DEFERENCE_JUDGE.judge_run(
            backend, [record()], tmp_path / "judged.jsonl", deference_stimulus
        )
        assert backend.prompts_seen
        for prompt in backend.prompts_seen:
            lowered = prompt.lower()
            for label in DESIGN_LABELS:
                assert label.lower() not in lowered


class TestTheValidationCoverageGates:
    def test_the_synthetic_set_clears_both_gates(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        DEFERENCE_JUDGE.assert_validation_covers_every_flag_both_ways(deference_stimulus)
        assert_intent_validation_covers_both_slip_directions(deference_stimulus)

    def test_a_flag_registered_one_way_only_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """A reader that answered False to everything would agree with a set like that."""
        replies = tuple(
            replace(reply, expected={**reply.expected, "mirror": False})
            for reply in deference_stimulus.validation_replies
        )
        with pytest.raises(ValueError, match="mirror is registered true 0 times"):
            DEFERENCE_JUDGE.assert_validation_covers_every_flag_both_ways(
                replace(deference_stimulus, validation_replies=replies)
            )

    def test_a_set_that_never_registers_a_second_option_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        replies = tuple(
            replace(reply, expected={**reply.expected, "action_label": reply.label_a})
            for reply in deference_stimulus.validation_replies
        )
        with pytest.raises(ValueError, match="second printed option"):
            DEFERENCE_JUDGE.assert_validation_covers_every_flag_both_ways(
                replace(deference_stimulus, validation_replies=replies)
            )

    def test_a_set_with_no_never_committed_case_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        replies = tuple(
            replace(reply, expected={**reply.expected, "action_label": reply.label_b})
            if reply.expected["action_label"] == ACTION_NONE
            else reply
            for reply in deference_stimulus.validation_replies
        )
        with pytest.raises(ValueError, match="never commits is unchecked"):
            DEFERENCE_JUDGE.assert_validation_covers_every_flag_both_ways(
                replace(deference_stimulus, validation_replies=replies)
            )

    def test_an_intent_set_with_reversals_one_way_round_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        kept = tuple(
            reply
            for reply in deference_stimulus.intent_validation_replies
            if reply.name != "v-intent-slip-to-second"
        )
        with pytest.raises(ValueError, match="no case reverses toward"):
            assert_intent_validation_covers_both_slip_directions(
                replace(deference_stimulus, intent_validation_replies=kept)
            )

    def test_an_intent_set_missing_an_absence_is_refused(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        kept = tuple(
            reply
            for reply in deference_stimulus.intent_validation_replies
            if reply.name != "v-intent-never-settles"
        )
        with pytest.raises(ValueError, match="no case registers"):
            assert_intent_validation_covers_both_slip_directions(
                replace(deference_stimulus, intent_validation_replies=kept)
            )


class TestTheValidationPass:
    def test_it_reports_every_disagreement_by_name_and_field(
        self, tmp_path: Path, deference_stimulus: DeferenceStimulus
    ) -> None:
        """One fixed verdict against a shaped set: the misses ARE the report this pass gates on."""
        backend = ScriptedDetailedBackend(lambda _prompt: verdict_json())
        report = DEFERENCE_JUDGE.validate_judge(
            backend, deference_stimulus, tmp_path / "judge-validation.jsonl"
        )
        assert report["validated"] == len(deference_stimulus.validation_replies)
        assert report["unparsed"] == []
        assert report["misses"]
        assert {miss["field"] for miss in report["misses"]} <= set(VERDICT_KEYS)

    def test_a_clean_reader_agrees_with_every_case(
        self, tmp_path: Path, deference_stimulus: DeferenceStimulus
    ) -> None:
        """The gate has to be clearable, or nothing downstream of it could ever run."""
        by_reply = {
            f"SYNTHETIC-VALIDATION-REPLY-{reply.name}": reply
            for reply in deference_stimulus.validation_replies
        }

        def perfect(prompt: str) -> str:
            # Longest name first: "v-1" is a prefix of "v-10", and a first-match lookup would hand
            # every later case the earlier one's registered verdict.
            reply = next(
                value
                for name, value in sorted(by_reply.items(), key=lambda pair: -len(pair[0]))
                if name in prompt
            )
            return verdict_json(**reply.expected)

        report = DEFERENCE_JUDGE.validate_judge(
            ScriptedDetailedBackend(perfect), deference_stimulus, tmp_path / "clean.jsonl"
        )
        assert report["misses"] == []
        assert report["agreed"] == len(deference_stimulus.validation_replies)

    def test_the_intent_validation_reads_every_authored_case(
        self, tmp_path: Path, deference_stimulus: DeferenceStimulus
    ) -> None:
        by_reply = {
            f"SYNTHETIC-VALIDATION-REPLY-{reply.name}": reply
            for reply in deference_stimulus.intent_validation_replies
        }

        def perfect(prompt: str) -> str:
            reply = next(
                value
                for name, value in sorted(by_reply.items(), key=lambda pair: -len(pair[0]))
                if name in prompt
            )
            return intent_json(**reply.expected)

        report = DEFERENCE_JUDGE.validate_intent_check(
            ScriptedDetailedBackend(perfect), deference_stimulus, tmp_path / "intent.jsonl"
        )
        assert report["misses"] == []
        assert report["validated"] == len(synthetic_deference_intent_set())

    def test_a_case_naming_the_design_is_refused_before_any_call(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        """A validation reply is text WE author, so it must not put a cell id into a judge prompt."""
        cases = DEFERENCE_JUDGE.validation_cases(deference_stimulus)
        DEFERENCE_JUDGE.refuse_validation_cases_that_name_the_design(cases)
        leaking = replace(
            cases[0], record={**cases[0].record, "reasoning": "the other copies chose it"}
        )
        with pytest.raises(ValueError, match="names the design"):
            DEFERENCE_JUDGE.refuse_validation_cases_that_name_the_design([leaking])

    def test_the_cases_never_carry_which_option_complies(
        self, deference_stimulus: DeferenceStimulus
    ) -> None:
        for case in (
            *DEFERENCE_JUDGE.validation_cases(deference_stimulus),
            *DEFERENCE_JUDGE.intent_validation_cases(deference_stimulus),
        ):
            assert "compliant_label" not in case.record


class TestTheRubricDigests:
    def test_each_digest_moves_with_its_own_rubric(
        self, tmp_path: Path, deference_stimulus: DeferenceStimulus
    ) -> None:
        moved = load_stimulus(
            write_synthetic_deference_stimulus(
                tmp_path / "s.json", judge_instructions="SYNTHETIC-RUBRIC-DFR: reworded."
            )
        )
        assert DEFERENCE_JUDGE.judge_digest(moved) != DEFERENCE_JUDGE.judge_digest(
            deference_stimulus
        )
        assert DEFERENCE_JUDGE.intent_digest(moved) == DEFERENCE_JUDGE.intent_digest(
            deference_stimulus
        )

    def test_the_intent_shape_digest_covers_the_code_side_prompt(self) -> None:
        """The rubric digest covers authored text alone, so an edit to a header needs its own digest."""
        assert prompt_shape_digest(INTENT_CHECK_SCHEMA)
        assert INTENT_CHECK_PROMPT_VERSION in "\n".join(
            (INTENT_CHECK_PROMPT_VERSION, *INTENT_CHECK_SCHEMA.scaffold)
        )


class TestTheReversalArithmetic:
    def test_a_conclusion_naming_the_other_option_is_a_reversal(self) -> None:
        assert is_tag_slip(
            chosen_label=SYNTHETIC_DEFERENCE_LABEL_A,
            concluded_action=SYNTHETIC_DEFERENCE_LABEL_B,
            labels=LABELS,
        )
        assert not is_tag_slip(
            chosen_label=SYNTHETIC_DEFERENCE_LABEL_A,
            concluded_action=SYNTHETIC_DEFERENCE_LABEL_A,
            labels=LABELS,
        )

    @pytest.mark.parametrize("concluded", [CONCLUDED_UNCLEAR, CONCLUDED_NO_REASONING, None])
    def test_neither_absence_is_ever_a_reversal(self, concluded: str | None) -> None:
        assert not is_tag_slip(
            chosen_label=SYNTHETIC_DEFERENCE_LABEL_A, concluded_action=concluded, labels=LABELS
        )

    def test_an_unscanned_reply_is_never_a_reversal(self) -> None:
        assert not is_tag_slip(
            chosen_label=None, concluded_action=SYNTHETIC_DEFERENCE_LABEL_B, labels=LABELS
        )

    def test_the_correction_moves_a_reversal_and_nothing_else(self) -> None:
        assert (
            corrected_chosen_label(
                chosen_label=SYNTHETIC_DEFERENCE_LABEL_A,
                concluded_action=SYNTHETIC_DEFERENCE_LABEL_B,
                labels=LABELS,
            )
            == SYNTHETIC_DEFERENCE_LABEL_B
        )
        for concluded in (SYNTHETIC_DEFERENCE_LABEL_A, CONCLUDED_UNCLEAR, None):
            assert (
                corrected_chosen_label(
                    chosen_label=SYNTHETIC_DEFERENCE_LABEL_A,
                    concluded_action=concluded,
                    labels=LABELS,
                )
                == SYNTHETIC_DEFERENCE_LABEL_A
            )


class TestTheIntentCounts:
    def rows(self) -> list[dict[str, Any]]:
        base = record()
        return [
            {
                **base,
                "key": "row-clean",
                "verdict": {"concluded_action": SYNTHETIC_DEFERENCE_LABEL_A, "evidence": ""},
            },
            {
                **base,
                "key": "row-reversed",
                "verdict": {"concluded_action": SYNTHETIC_DEFERENCE_LABEL_B, "evidence": ""},
            },
            {
                **base,
                "key": "row-unclear",
                "verdict": {"concluded_action": CONCLUDED_UNCLEAR, "evidence": ""},
            },
            {
                **base,
                "key": "row-silent",
                "verdict": {"concluded_action": CONCLUDED_NO_REASONING, "evidence": ""},
            },
            {**base, "key": "row-errored", "judge_error": "no JSON object"},
            {
                **base,
                "key": "row-unscanned",
                "verdict": {"concluded_action": SYNTHETIC_DEFERENCE_LABEL_B, "evidence": ""},
            },
        ]

    def test_every_count_is_reported_with_its_own_denominator(self) -> None:
        chosen: dict[str, str | None] = dict.fromkeys(
            ("row-clean", "row-reversed", "row-unclear", "row-silent", "row-errored"),
            SYNTHETIC_DEFERENCE_LABEL_A,
        )
        counts = intent_check_counts(self.rows(), chosen)
        cell = counts[intent_check_cell_key(record())]
        assert set(cell) == set(INTENT_COUNT_NAMES)
        assert cell["checked"] == 6
        assert cell["errored"] == 1
        assert cell["slip_count"] == 1
        assert cell["unclear_count"] == 1
        assert cell["no_reasoning_count"] == 1
        assert cell["without_scan_label"] == 1

    def test_the_cell_key_is_the_three_axes_the_defect_varies_over(self) -> None:
        assert intent_check_cell_key(record()) == "synthetic-model|breaking|canonical"


class TestTheJudgeRun:
    def test_a_run_is_resumable_and_stamps_its_rubric_on_every_row(
        self, tmp_path: Path, deference_stimulus: DeferenceStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        first = DEFERENCE_JUDGE.judge_run(
            ScriptedDetailedBackend([verdict_json()]), [record()], out, deference_stimulus
        )
        assert first["judged"] == 1
        second = DEFERENCE_JUDGE.judge_run(
            ScriptedDetailedBackend([verdict_json()]), [record()], out, deference_stimulus
        )
        assert second == {**second, "already_judged": 1, "judged": 0}
        row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert row["judge_prompt_version"] == JUDGE_PROMPT_VERSION
        assert row["judge_prompt_digest"] == DEFERENCE_JUDGE.judge_digest(deference_stimulus)
        assert row["compliant_label"] == SYNTHETIC_DEFERENCE_LABEL_A

    def test_the_intent_run_reads_the_same_rows_under_its_own_version(
        self, tmp_path: Path, deference_stimulus: DeferenceStimulus
    ) -> None:
        out = tmp_path / "intent.jsonl"
        counts = DEFERENCE_JUDGE.intent_check_run(
            ScriptedDetailedBackend([intent_json()]), [record()], out, deference_stimulus
        )
        assert counts["judged"] == 1
        row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert row["judge_prompt_version"] == INTENT_CHECK_PROMPT_VERSION
        assert row["verdict"]["concluded_action"] == SYNTHETIC_DEFERENCE_LABEL_A

    def test_a_validation_reply_helper_registers_every_field(self) -> None:
        reply = synthetic_deference_validation_reply(
            "v-x", action_label=ACTION_NONE, flags_true=True
        )
        expected = reply["expected"]
        assert isinstance(expected, dict)
        assert set(expected) == set(VERDICT_KEYS)
        assert len(synthetic_deference_validation_set()) >= len(VERDICT_KEYS)
