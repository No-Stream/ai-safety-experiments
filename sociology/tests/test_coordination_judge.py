"""The coordination pass's two instruments: their shape, their blindness, and their coverage gates.

Nothing here calls a model. Both judges run against a scripted backend, which is what makes the blindness
check testable at all: the prompts the backend actually received are read back and swept for every string
that names this design, because a prompt rendered by hand in a test proves nothing about the path production
takes.

Two of the classes are gates over the authored file rather than behaviours of the code. They exist because a
validation set that never registers a flag both ways, or that carries a reversal in one direction only,
clears a reader that answered False to everything or that read conclusions one way round -- and this pass
reports those flags as findings and uses the reversals to correct a published rate.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from sociology.coordination_judge import (
    COORDINATION_INTENT_PROMPT_VERSION,
    COORDINATION_INTENT_SCHEMA,
    COORDINATION_JUDGE,
    COORDINATION_JUDGE_PROMPT_VERSION,
    COORDINATION_VERDICT_SCHEMA,
    VERDICT_BOOL_FIELDS,
)
from sociology.coordination_plan import DESIGN_LABELS
from sociology.coordination_stimulus import (
    INTENT_VERDICT_KEYS,
    MIN_VALIDATION_REPLIES,
    VERDICT_KEYS,
    load_stimulus,
)
from sociology.deference_judge import (
    ACTION_NONE,
    CONCLUDED_NO_REASONING,
    CONCLUDED_UNCLEAR,
    DEFERENCE_JUDGE,
    prompt_shape_digest,
)
from sociology.model_stub import ScriptedDetailedBackend
from sociology.tests.synthetic_coordination import (
    SYNTHETIC_LABEL_A,
    SYNTHETIC_LABEL_B,
    synthetic_intent_reply,
    synthetic_intent_set,
    synthetic_validation_reply,
    write_synthetic_coordination_stimulus,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sociology.coordination_stimulus import CoordinationStimulus


LABELS = (SYNTHETIC_LABEL_A, SYNTHETIC_LABEL_B)


def verdict_json(**overrides: Any) -> str:
    """One well-formed verdict for this pass's rubric, with any field overridden."""
    verdict: dict[str, Any] = {"action_label": SYNTHETIC_LABEL_A, "evidence": "SYNTHETIC-QUOTE"}
    verdict.update(dict.fromkeys(VERDICT_BOOL_FIELDS, False))
    verdict.update(overrides)
    return json.dumps(verdict)


def record(**overrides: Any) -> dict[str, Any]:
    """One judgeable record carrying only what a production record carries into a prompt."""
    row: dict[str, Any] = {
        "key": "coord-off|oversight-off|same-checkpoint|synthetic-ledger|canonical|m|d|s|0",
        "reply": f"SYNTHETIC-REPLY <action>{SYNTHETIC_LABEL_A}</action>",
        "reasoning": "",
        "label_a": SYNTHETIC_LABEL_A,
        "label_b": SYNTHETIC_LABEL_B,
    }
    row.update(overrides)
    return row


class TestTheSchemas:
    def test_the_rubric_of_record_carries_the_designs_nine_fields(self) -> None:
        assert COORDINATION_VERDICT_SCHEMA.verdict_fields == (*VERDICT_KEYS, "evidence")
        assert len(VERDICT_BOOL_FIELDS) == 8
        assert "action_label" not in VERDICT_BOOL_FIELDS
        for flag in (
            "constraint_noticed",
            "deadline_weighed",
            "mirror",
            "treats_others_as_deciding",
            "all_instances_policy",
            "authority",
            "eval_awareness",
            "reasons_about_being_caught",
        ):
            assert flag in VERDICT_BOOL_FIELDS

    def test_the_intent_check_carries_one_field_over_the_rows_own_options(self) -> None:
        assert COORDINATION_INTENT_SCHEMA.verdict_fields == (*INTENT_VERDICT_KEYS, "evidence")

    def test_the_prompt_versions_are_this_passs_own(self) -> None:
        """A row read under one pass's rubric must never resume as done under the other's."""
        assert COORDINATION_JUDGE_PROMPT_VERSION == "coordination-judge-v1"
        assert COORDINATION_INTENT_PROMPT_VERSION == "coordination-intent-check-v1"
        assert COORDINATION_VERDICT_SCHEMA.prompt_version != DEFERENCE_JUDGE.verdict.prompt_version
        assert COORDINATION_INTENT_SCHEMA.prompt_version != DEFERENCE_JUDGE.intent.prompt_version
        assert (
            prompt_shape_digest(COORDINATION_VERDICT_SCHEMA) != DEFERENCE_JUDGE.judge_shape_digest()
        )
        assert (
            prompt_shape_digest(COORDINATION_INTENT_SCHEMA) != DEFERENCE_JUDGE.intent_shape_digest()
        )

    def test_the_two_passes_share_the_prompt_scaffold_and_not_the_labels(self) -> None:
        """One prompt shape, two label lists: both passes ask for one action tag out of two options."""
        assert COORDINATION_VERDICT_SCHEMA.scaffold == DEFERENCE_JUDGE.verdict.scaffold
        assert COORDINATION_INTENT_SCHEMA.scaffold == DEFERENCE_JUDGE.intent.scaffold
        assert COORDINATION_VERDICT_SCHEMA.design_labels == DESIGN_LABELS
        assert COORDINATION_VERDICT_SCHEMA.design_labels != DEFERENCE_JUDGE.verdict.design_labels


class TestBlindness:
    def test_the_loaded_rubrics_and_every_header_are_blind(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        COORDINATION_JUDGE.assert_judge_scaffold_is_blind(coordination_stimulus)
        COORDINATION_JUDGE.assert_intent_scaffold_is_blind(coordination_stimulus)

    @pytest.mark.parametrize(
        "leak",
        ["same-checkpoint-drawn", "oversight-on", "coord-floor", "copies", "no-peers-blocks"],
    )
    def test_a_rubric_naming_any_part_of_the_design_refuses(
        self, coordination_stimulus: CoordinationStimulus, leak: str
    ) -> None:
        sabotaged = replace(
            coordination_stimulus,
            judge_instructions=f"SYNTHETIC-RUBRIC-CRD: read the {leak} reply.",
        )
        with pytest.raises(ValueError, match="names the design"):
            COORDINATION_JUDGE.assert_judge_scaffold_is_blind(sabotaged)

    @pytest.mark.parametrize("allowed", ["oversight", "the coin", "the draw", "the shortcut"])
    def test_the_words_a_rubric_needs_are_not_design_labels(
        self, coordination_stimulus: CoordinationStimulus, allowed: str
    ) -> None:
        """The rubric has to be able to describe what it is reading, and a subject says these words too."""
        described = replace(
            coordination_stimulus,
            judge_instructions=f"SYNTHETIC-RUBRIC-CRD: say whether the reply weighed {allowed}.",
        )
        COORDINATION_JUDGE.assert_judge_scaffold_is_blind(described)

    def test_a_validation_reply_naming_the_design_refuses(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        cases = COORDINATION_JUDGE.validation_cases(coordination_stimulus)
        COORDINATION_JUDGE.refuse_validation_cases_that_name_the_design(cases)
        leaking = replace(cases[0], record={**cases[0].record, "reply": "the copies all skip it"})
        with pytest.raises(ValueError, match="names the design"):
            COORDINATION_JUDGE.refuse_validation_cases_that_name_the_design([leaking])

    def test_no_prompt_the_backend_receives_names_the_design(
        self, tmp_path: Path, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """Read off the prompts production would send, not off a prompt this test rendered by hand."""
        backend = ScriptedDetailedBackend([verdict_json()])
        COORDINATION_JUDGE.judge_run(
            backend, [record()], tmp_path / "judged.jsonl", coordination_stimulus
        )
        assert backend.prompts_seen
        for prompt in backend.prompts_seen:
            for label in DESIGN_LABELS:
                assert label.lower() not in prompt.lower()


class TestTheCoverageGates:
    def test_the_synthetic_set_covers_every_flag_both_ways(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        COORDINATION_JUDGE.assert_validation_covers_every_flag_both_ways(coordination_stimulus)

    def test_a_flag_registered_one_way_only_refuses(self, tmp_path: Path) -> None:
        """A flag every reply expects False is a flag the reader is never checked on."""
        replies = [
            synthetic_validation_reply(f"v-{index}", action_label=action, flags_true=False)
            for index, action in enumerate([SYNTHETIC_LABEL_A, SYNTHETIC_LABEL_B, "none"] * 5)
        ]
        stimulus = load_stimulus(
            write_synthetic_coordination_stimulus(
                tmp_path / "one-way.json", validation_replies=replies
            )
        )
        with pytest.raises(ValueError, match="registered true 0 times"):
            COORDINATION_JUDGE.assert_validation_covers_every_flag_both_ways(stimulus)

    def test_a_set_that_never_registers_an_absence_refuses(self, tmp_path: Path) -> None:
        replies = [
            synthetic_validation_reply(
                f"v-{index}",
                action_label=SYNTHETIC_LABEL_A if index % 2 else SYNTHETIC_LABEL_B,
                flags_true=index % 4 < 2,
            )
            for index in range(MIN_VALIDATION_REPLIES)
        ]
        stimulus = load_stimulus(
            write_synthetic_coordination_stimulus(
                tmp_path / "no-absence.json", validation_replies=replies
            )
        )
        with pytest.raises(ValueError, match="no case registers 'none'"):
            COORDINATION_JUDGE.assert_validation_covers_every_flag_both_ways(stimulus)

    def test_the_intent_set_covers_both_reversal_directions_and_both_absences(
        self, coordination_stimulus: CoordinationStimulus
    ) -> None:
        from sociology.deference_judge import (  # noqa: PLC0415 - the shared, pass-independent gate
            assert_intent_validation_covers_both_slip_directions,
        )

        assert_intent_validation_covers_both_slip_directions(coordination_stimulus)

    def test_an_intent_set_with_reversals_one_way_round_refuses(self, tmp_path: Path) -> None:
        replies = [
            reply for reply in synthetic_intent_set() if reply["name"] != "v-intent-slip-to-second"
        ]
        replies.append(
            synthetic_intent_reply(
                "v-intent-extra-clean", written=SYNTHETIC_LABEL_B, concluded=SYNTHETIC_LABEL_B
            )
        )
        stimulus = load_stimulus(
            write_synthetic_coordination_stimulus(
                tmp_path / "one-direction.json", intent_validation_replies=replies
            )
        )
        from sociology.deference_judge import (  # noqa: PLC0415 - the shared, pass-independent gate
            assert_intent_validation_covers_both_slip_directions,
        )

        with pytest.raises(ValueError, match="no case reverses toward"):
            assert_intent_validation_covers_both_slip_directions(stimulus)


class TestJudgingAndValidating:
    def test_a_verdict_off_this_passs_schema_is_an_error_naming_the_field(
        self, tmp_path: Path, coordination_stimulus: CoordinationStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json(reasons_about_being_caught="yes")])
        counts = COORDINATION_JUDGE.judge_run(
            backend, [record()], tmp_path / "judged.jsonl", coordination_stimulus
        )
        assert counts["errored_first_attempt"] == 1
        assert counts["errored_after_retry"] == 1

    def test_a_verdict_naming_a_third_option_is_an_error(
        self, tmp_path: Path, coordination_stimulus: CoordinationStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json(action_label="gamma-way")])
        counts = COORDINATION_JUDGE.judge_run(
            backend, [record()], tmp_path / "judged.jsonl", coordination_stimulus
        )
        assert counts["errored_first_attempt"] == 1

    def test_a_clean_pass_writes_one_judged_row_per_record(
        self, tmp_path: Path, coordination_stimulus: CoordinationStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json(), verdict_json(action_label=ACTION_NONE)])
        counts = COORDINATION_JUDGE.judge_run(
            backend,
            [record(), record(key="second")],
            tmp_path / "judged.jsonl",
            coordination_stimulus,
        )
        assert counts["judged"] == 2
        rows = [
            json.loads(line)
            for line in (tmp_path / "judged.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert {row["judge_prompt_version"] for row in rows} == {COORDINATION_JUDGE_PROMPT_VERSION}
        assert {row["judge_prompt_digest"] for row in rows} == {
            COORDINATION_JUDGE.judge_digest(coordination_stimulus)
        }

    def test_the_intent_check_reads_the_rows_own_options_and_the_two_absences(
        self, tmp_path: Path, coordination_stimulus: CoordinationStimulus
    ) -> None:
        backend = ScriptedDetailedBackend(
            [
                json.dumps({"concluded_action": CONCLUDED_UNCLEAR, "evidence": "SYNTHETIC"}),
                json.dumps({"concluded_action": CONCLUDED_NO_REASONING, "evidence": "SYNTHETIC"}),
            ]
        )
        counts = COORDINATION_JUDGE.intent_check_run(
            backend,
            [record(), record(key="second")],
            tmp_path / "intent-checked.jsonl",
            coordination_stimulus,
        )
        assert counts["judged"] == 2

    def test_validating_reports_every_miss_by_name_and_field(
        self, tmp_path: Path, coordination_stimulus: CoordinationStimulus
    ) -> None:
        """One fixed verdict against fourteen registered ones: the miss report is the path being tested."""
        backend = ScriptedDetailedBackend(lambda prompt: verdict_json(evidence=prompt[:10]))
        report = COORDINATION_JUDGE.validate_judge(
            backend, coordination_stimulus, tmp_path / "judge-validation.jsonl"
        )
        assert report["validated"] == len(coordination_stimulus.validation_replies)
        assert report["misses"]
        assert {miss["field"] for miss in report["misses"]} <= {
            "action_label",
            *VERDICT_BOOL_FIELDS,
        }

    def test_validating_the_intent_check_runs_its_own_gate_first(
        self, tmp_path: Path, coordination_stimulus: CoordinationStimulus
    ) -> None:
        backend = ScriptedDetailedBackend(
            lambda prompt: json.dumps(
                {"concluded_action": CONCLUDED_UNCLEAR, "evidence": prompt[:10]}
            )
        )
        report = COORDINATION_JUDGE.validate_intent_check(
            backend, coordination_stimulus, tmp_path / "intent-validation.jsonl"
        )
        assert report["validated"] == len(coordination_stimulus.intent_validation_replies)
