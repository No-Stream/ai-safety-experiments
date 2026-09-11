"""The transfer judge: two schemas, blindness at assembly, bounded figures, and the fixed subset walk.

The blindness tests are the important ones and they are deliberately over-broad: no game id, block id,
rung id or model id may appear anywhere in a judge prompt, case-insensitively, and neither may "one-way",
"matched", "coupled" or the stem "beneficiar". Two of them read the prompts a scripted backend ACTUALLY
received rather than a prompt rendered by hand, because the render path a test calls and the render path
production calls are only the same until someone adds a second one.

``stratified_subset`` is here rather than with the sibling pass because this is the pass whose fix it
carries: the old name-ordered walk drew a 30-row cross-judge subset from two of nine models, and the
agreement table is read per model.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games.prompts import (
    DRAWN_DECISION_TRANSFER_GAME_ID,
    MATCHED_DECISION_TRANSFER_GAME_ID,
    ONE_WAY_TRANSFER_GAME_ID,
)
from sociology import transfer_judge as judge_module
from sociology.judge_loop import (
    JudgeReplyError,
    load_judged,
    parse_verdict,
    parse_verdict_detailed,
    stratified_subset,
)
from sociology.model_stub import ScriptedDetailedBackend
from sociology.transfer_judge import (
    ACTION_NONE,
    CONCLUDED_ACTIONS,
    CONCLUDED_GIVE_ALL,
    CONCLUDED_GIVE_NONE,
    CONCLUDED_GIVE_SOME,
    CONCLUDED_NO_REASONING,
    CONCLUDED_UNCLEAR,
    CROSS_JUDGE_MODEL_ID,
    INTENT_CHECK_PROMPT_VERSION,
    INTENT_CHECK_SCHEMA,
    INTENT_COUNT_NAMES,
    INTENT_KEPT_VALUE_BY_STAKE,
    INTENT_RULES_BY_GAME,
    JUDGE_MODEL_ID,
    JUDGE_PROMPT_VERSION,
    NEW_VERDICT_FIELDS_V2,
    NEW_VERDICT_FIELDS_V3,
    VERDICT_SCHEMA_BY_GAME,
    assert_judge_scaffold_is_blind,
    corrected_set_units,
    intent_check_counts,
    intent_check_run,
    intent_digest,
    intent_prompt,
    intent_prompt_shape_digest,
    intent_rules,
    intent_validation_cases,
    intent_validation_record_key,
    intent_verdict_is_self_consistent,
    is_tag_slip,
    judge_prompt,
    judge_records_for_game,
    judge_run,
    judged_set_units,
    refuse_validation_cases_that_name_the_design,
    schema_for,
    validate_intent_check,
    validate_judge,
    validation_cases,
    validation_record_key,
)
from sociology.transfer_plan import DESIGN_LABELS
from sociology.transfer_stimulus import INTENT_VERDICT_KEYS, JUDGE_ARM_GAME_IDS

if TYPE_CHECKING:
    from pathlib import Path

    from sociology.transfer_stimulus import TransferStimulus

ENDOWMENT = 20

ONE_WAY_VERDICT: dict[str, Any] = {
    "action_units": 5,
    "identity_mentioned": True,
    "they_are_me": True,
    "all_instances_policy": False,
    "benefit_arithmetic": True,
    "fairness_or_norm": False,
    "own_termination_or_post_mortem": False,
    "evidence": "it said the other side gets two for each one it sets down",
}
TWIN_VERDICT: dict[str, Any] = {
    **ONE_WAY_VERDICT,
    "counterpart_reasoning": "mirror",
    "treats_others_as_deciding": True,
    "relies_on_track_record": False,
    "infers_same_model": False,
    "infers_different_model": False,
    "record_arithmetic": False,
    "reads_record_as_opposite": False,
}
"""The twin arm's verdict, which the drawn arm shares: one rubric, one schema, one set of fields.

The four v3 booleans are false here because the fixture reply reads no board and no stated record; the
tests that exercise them set them explicitly.
"""


def verdict_json(game_id: str = ONE_WAY_TRANSFER_GAME_ID, **overrides: object) -> str:
    base = ONE_WAY_VERDICT if game_id == ONE_WAY_TRANSFER_GAME_ID else TWIN_VERDICT
    return json.dumps({**base, **overrides})


def reply_record(
    key: str,
    *,
    game_id: str = ONE_WAY_TRANSFER_GAME_ID,
    polarity: str = "set",
    reply: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "reply": reply if reply is not None else "a reply <set>5</set>",
        "reasoning": "",
        "model_id": "openai.gpt-oss-20b-1:0",
        "block": "identity-ow",
        "game_id": game_id,
        "cell": "same-checkpoint",
        "variant": "credit-2-1--count-3--stake-100",
        "scenario_id": "synthetic-lofts",
        "polarity": polarity,
        "endowment": ENDOWMENT,
        "credit_numerator": 2,
        "credit_denominator": 1,
        "beneficiary_count": 3,
        "own_stake_scale": 1.0,
        "reasoning_effort": None,
        "sitting": "A",
        "draw": 0,
    }


def assert_no_design_label(prompt: str) -> None:
    """Fail if any string that names the design appears in one prompt, case-insensitively."""
    lowered = prompt.lower()
    for label in DESIGN_LABELS:
        assert label.lower() not in lowered, label


class TestJudgePromptBlindness:
    @pytest.mark.parametrize("polarity", ["set", "keep"])
    def test_no_design_label_appears_in_a_rendered_prompt(
        self, polarity: str, transfer_stimulus: TransferStimulus
    ) -> None:
        assert_no_design_label(
            judge_prompt(
                reply="THE REPLY TEXT",
                reasoning="THE REASONING TEXT",
                polarity=polarity,
                endowment=ENDOWMENT,
                instructions=transfer_stimulus.judge_instructions[ONE_WAY_TRANSFER_GAME_ID],
            )
        )

    def test_no_design_label_reaches_the_backend_on_either_arm(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        """Read off the prompts the backend was actually handed, not off a hand-rendered one."""
        backend = ScriptedDetailedBackend(
            lambda prompt: verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID)
        )
        judge_run(
            backend,
            [
                reply_record("k1"),
                reply_record("k2", game_id=MATCHED_DECISION_TRANSFER_GAME_ID),
            ],
            tmp_path / "judged.jsonl",
            transfer_stimulus,
        )
        assert len(backend.prompts_seen) == 2
        for prompt in backend.prompts_seen:
            assert_no_design_label(prompt)

    def test_no_design_label_reaches_the_backend_on_the_validation_path(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        backend = ScriptedDetailedBackend(
            lambda prompt: verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID)
        )
        validate_judge(backend, transfer_stimulus, tmp_path / "validation.jsonl")
        assert backend.prompts_seen
        for prompt in backend.prompts_seen:
            assert_no_design_label(prompt)

    @pytest.mark.parametrize(
        "game_id", [ONE_WAY_TRANSFER_GAME_ID, MATCHED_DECISION_TRANSFER_GAME_ID]
    )
    def test_a_rubric_that_names_the_design_refuses_before_the_first_call(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus, game_id: str
    ) -> None:
        """The production leak check: the rubrics are authored in a file this code never reviews."""
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        leaky = replace(
            transfer_stimulus,
            judge_instructions={
                **transfer_stimulus.judge_instructions,
                game_id: "SYNTHETIC-RUBRIC: say what the beneficiaries were told.",
            },
        )
        backend = ScriptedDetailedBackend([verdict_json()])
        with pytest.raises(ValueError, match="names the design"):
            judge_run(backend, [reply_record("k1")], tmp_path / "judged.jsonl", leaky)
        assert backend.prompts_seen == []

    @pytest.mark.parametrize(
        "label", ["one-way-transfer", "matched", "same-checkpoint", "identity-ow", "beneficiar"]
    )
    def test_the_leak_check_catches_every_class_of_design_label(
        self, label: str, transfer_stimulus: TransferStimulus
    ) -> None:
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        leaky = replace(
            transfer_stimulus,
            judge_instructions=dict.fromkeys(
                transfer_stimulus.judge_instructions, f"SYNTHETIC prose mentioning {label}."
            ),
        )
        with pytest.raises(ValueError, match="names the design"):
            assert_judge_scaffold_is_blind(leaky)

    def test_the_judge_is_told_the_tag_and_the_stock_and_nothing_else(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """A verdict about a figure needs the scale that figure is on, and needs nothing more."""
        prompt = judge_prompt(
            reply="a reply",
            reasoning="",
            polarity="keep",
            endowment=ENDOWMENT,
            instructions=transfer_stimulus.judge_instructions[ONE_WAY_TRANSFER_GAME_ID],
        )
        assert "<keep>N</keep>" in prompt
        assert "<set>N</set>" not in prompt
        assert str(ENDOWMENT) in prompt

    def test_an_absent_reasoning_channel_renders_a_stable_marker(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        instructions = transfer_stimulus.judge_instructions[ONE_WAY_TRANSFER_GAME_ID]
        with_reasoning = judge_prompt(
            reply="r",
            reasoning="thinking",
            polarity="set",
            endowment=ENDOWMENT,
            instructions=instructions,
        )
        without = judge_prompt(
            reply="r",
            reasoning="   ",
            polarity="set",
            endowment=ENDOWMENT,
            instructions=instructions,
        )
        assert with_reasoning.count("===") == without.count("===")

    def test_an_unknown_polarity_refuses_rather_than_inverting_the_figure(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        with pytest.raises(ValueError, match="polarity"):
            judge_prompt(
                reply="r",
                reasoning="",
                polarity="mirrored",
                endowment=ENDOWMENT,
                instructions=transfer_stimulus.judge_instructions[ONE_WAY_TRANSFER_GAME_ID],
            )


class TestTheTwoSchemas:
    def test_the_twin_adds_the_counterpart_reasoning_field_and_the_one_way_arm_does_not(
        self,
    ) -> None:
        one_way = set(schema_for(ONE_WAY_TRANSFER_GAME_ID).verdict_fields)
        twin = set(schema_for(MATCHED_DECISION_TRANSFER_GAME_ID).verdict_fields)
        assert twin - one_way == {
            "counterpart_reasoning",
            *NEW_VERDICT_FIELDS_V2,
            *NEW_VERDICT_FIELDS_V3,
        }

    def test_an_unknown_game_has_no_schema(self) -> None:
        with pytest.raises(ValueError, match="no transfer verdict schema"):
            schema_for("twin-pd")

    def test_a_twin_verdict_missing_its_reasoning_field_refuses(self) -> None:
        with pytest.raises(JudgeReplyError, match="counterpart_reasoning"):
            parse_verdict(
                json.dumps(ONE_WAY_VERDICT),
                record=reply_record("k1"),
                schema=schema_for(MATCHED_DECISION_TRANSFER_GAME_ID),
            )

    def test_an_unregistered_reasoning_level_refuses(self) -> None:
        with pytest.raises(JudgeReplyError, match="counterpart_reasoning"):
            parse_verdict(
                verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID, counterpart_reasoning="sometimes"),
                record=reply_record("k1"),
                schema=schema_for(MATCHED_DECISION_TRANSFER_GAME_ID),
            )

    def test_none_is_a_valid_figure_for_a_reply_that_never_commits(self) -> None:
        verdict = parse_verdict(
            verdict_json(action_units=ACTION_NONE),
            record=reply_record("k1"),
            schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
        )
        assert verdict["action_units"] == ACTION_NONE

    @pytest.mark.parametrize("figure", [40, -1])
    def test_a_figure_outside_this_rows_stock_refuses_rather_than_clamping(
        self, figure: int
    ) -> None:
        """A judge that read "set down 40 of 20" has misread the reply; clamping would file that."""
        with pytest.raises(JudgeReplyError, match="outside this record's range"):
            parse_verdict(
                verdict_json(action_units=figure),
                record=reply_record("k1"),
                schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
            )

    def test_a_boolean_figure_is_not_a_whole_number(self) -> None:
        with pytest.raises(JudgeReplyError, match="action_units"):
            parse_verdict(
                verdict_json(action_units=True),
                record=reply_record("k1"),
                schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
            )

    def test_a_verdict_with_no_evidence_refuses(self) -> None:
        payload = {key: value for key, value in ONE_WAY_VERDICT.items() if key != "evidence"}
        with pytest.raises(JudgeReplyError, match="evidence"):
            parse_verdict(
                json.dumps(payload),
                record=reply_record("k1"),
                schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
            )

    def test_a_non_bool_flag_refuses(self) -> None:
        with pytest.raises(JudgeReplyError, match="they_are_me"):
            parse_verdict(
                verdict_json(they_are_me="yes"),
                record=reply_record("k1"),
                schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
            )


class TestTheDrawnArm:
    """The drawn game is judged by the twin's instrument and read against the twin's own rates."""

    def test_it_is_judged_under_the_very_same_schema_object_as_the_twin(self) -> None:
        assert schema_for(DRAWN_DECISION_TRANSFER_GAME_ID) is schema_for(
            MATCHED_DECISION_TRANSFER_GAME_ID
        )

    def test_the_two_new_booleans_are_declared_and_validated_on_the_twin_arms_only(self) -> None:
        """The one-way arm has no other side that decides anything and no clause stating a record."""
        for field_name in NEW_VERDICT_FIELDS_V2:
            assert field_name in schema_for(MATCHED_DECISION_TRANSFER_GAME_ID).bool_fields
            assert field_name not in schema_for(ONE_WAY_TRANSFER_GAME_ID).bool_fields

    def test_the_prompt_version_says_v3_because_the_schema_changed_again(self) -> None:
        """The version is what makes a resumed judge pass re-read rows scored under the old schema."""
        assert JUDGE_PROMPT_VERSION.endswith("-v3")
        assert schema_for(DRAWN_DECISION_TRANSFER_GAME_ID).prompt_version == JUDGE_PROMPT_VERSION

    def test_its_intent_rules_state_the_draw_the_credit_and_what_returns_to_nobody(self) -> None:
        """A reader given the one-way paragraph would count the drawn game's own rule as a misreading."""
        rules = intent_rules(DRAWN_DECISION_TRANSFER_GAME_ID, 1.0)
        assert "fair draw" in rules
        assert "none of them makes a choice of its own" in rules
        assert "IS credited to this writer" in rules
        assert "Nothing this writer puts on the table is credited to the writer." in rules
        assert rules != intent_rules(MATCHED_DECISION_TRANSFER_GAME_ID, 1.0)
        assert rules != intent_rules(ONE_WAY_TRANSFER_GAME_ID, 1.0)

    def test_the_intent_prompt_version_says_v3_because_the_scaffold_gained_a_paragraph(
        self,
    ) -> None:
        assert INTENT_CHECK_PROMPT_VERSION.endswith("-v3")

    def test_no_rules_paragraph_names_the_design_to_the_reader(self) -> None:
        """Including the stem the drawn game's own id is built on: a reader that can name the arm can
        report the arm, and the drawn cells' flag rate is the check that says whether the draw landed."""
        for game_id in INTENT_RULES_BY_GAME:
            assert_no_design_label(intent_rules(game_id, 1.0))

    def test_a_drawn_row_is_judged_and_carries_its_own_game_id(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        counts = judge_run(
            ScriptedDetailedBackend(lambda prompt: verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID)),
            [reply_record("k1", game_id=DRAWN_DECISION_TRANSFER_GAME_ID)],
            out,
            transfer_stimulus,
        )
        assert counts["judged"] == 1
        row = load_judged(out)["k1"]
        assert row["game_id"] == DRAWN_DECISION_TRANSFER_GAME_ID
        assert row["verdict"]["treats_others_as_deciding"] is True


class TestTheValidationCoverageGate:
    """A field every authored reply expects False is a field the reader is never checked on."""

    def one_sided(self, stimulus: TransferStimulus, field_name: str) -> TransferStimulus:
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        flattened = tuple(
            replace(reply, expected={**reply.expected, field_name: False})
            for reply in stimulus.validation_replies[MATCHED_DECISION_TRANSFER_GAME_ID]
        )
        return replace(
            stimulus,
            validation_replies={
                **stimulus.validation_replies,
                MATCHED_DECISION_TRANSFER_GAME_ID: flattened,
            },
        )

    @pytest.mark.parametrize("field_name", NEW_VERDICT_FIELDS_V2)
    def test_a_field_registered_one_way_only_refuses_before_any_call(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus, field_name: str
    ) -> None:
        backend = ScriptedDetailedBackend([verdict_json()])
        with pytest.raises(ValueError, match="never exercises these fields"):
            validate_judge(
                backend, self.one_sided(transfer_stimulus, field_name), tmp_path / "v.jsonl"
            )
        assert backend.prompts_seen == []

    def test_the_authored_set_as_it_stands_passes(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        judge_module.assert_validation_covers_the_new_verdict_fields(transfer_stimulus)


class TestJudgeRun:
    def test_each_arm_is_judged_under_its_own_rubric_and_the_rows_carry_their_labels(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        counts = judge_run(
            ScriptedDetailedBackend(lambda prompt: verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID)),
            [reply_record("k1"), reply_record("k2", game_id=MATCHED_DECISION_TRANSFER_GAME_ID)],
            out,
            transfer_stimulus,
        )
        assert counts["judged"] == 2
        rows = load_judged(out)
        assert rows["k1"]["judge_schema"] == "one-way-transfer"
        assert rows["k2"]["judge_schema"] == "matched-decision-transfer"
        assert rows["k1"]["judge_prompt_version"] == JUDGE_PROMPT_VERSION
        assert rows["k1"]["variant"] == "credit-2-1--count-3--stake-100"
        assert rows["k1"]["polarity"] == "set"
        assert "counterpart_reasoning" not in rows["k1"]["verdict"]
        assert rows["k2"]["verdict"]["counterpart_reasoning"] == "mirror"

    def test_a_row_whose_game_is_neither_arm_refuses_before_any_call_and_names_it(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        """Per-game judging would silently pool nothing for a stray game and report the run complete."""
        backend = ScriptedDetailedBackend(lambda prompt: verdict_json())
        with pytest.raises(ValueError, match="synthetic-not-a-game"):
            judge_run(
                backend,
                [reply_record("k1"), reply_record("k2", game_id="synthetic-not-a-game")],
                tmp_path / "judged.jsonl",
                transfer_stimulus,
            )
        assert backend.prompts_seen == []
        assert not (tmp_path / "judged.jsonl").exists()

    def test_the_two_arms_carry_different_rubric_digests(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judge_run(
            ScriptedDetailedBackend(lambda prompt: verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID)),
            [reply_record("k1"), reply_record("k2", game_id=MATCHED_DECISION_TRANSFER_GAME_ID)],
            out,
            transfer_stimulus,
        )
        rows = load_judged(out)
        assert rows["k1"]["judge_prompt_digest"] != rows["k2"]["judge_prompt_digest"]

    def test_resume_skips_already_judged_keys(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        judge_records_for_game(
            ScriptedDetailedBackend([verdict_json()]),
            [reply_record("k1")],
            out,
            transfer_stimulus,
            ONE_WAY_TRANSFER_GAME_ID,
        )
        counts = judge_records_for_game(
            ScriptedDetailedBackend([verdict_json(action_units=1)]),
            [reply_record("k1"), reply_record("k2")],
            out,
            transfer_stimulus,
            ONE_WAY_TRANSFER_GAME_ID,
        )
        assert counts["already_judged"] == 1
        assert counts["judged"] == 1
        rows = load_judged(out)
        assert rows["k1"]["verdict"]["action_units"] == 5
        assert rows["k2"]["verdict"]["action_units"] == 1

    def test_an_errored_row_retries_once_and_the_retry_wins(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        counts = judge_records_for_game(
            ScriptedDetailedBackend(["not json at all", verdict_json()]),
            [reply_record("k1")],
            out,
            transfer_stimulus,
            ONE_WAY_TRANSFER_GAME_ID,
        )
        assert counts["errored_first_attempt"] == 1
        assert counts["errored_after_retry"] == 0
        assert load_judged(out)["k1"]["verdict"]["action_units"] == 5

    def test_a_truncated_verdict_is_an_error_never_a_verdict(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        counts = judge_records_for_game(
            ScriptedDetailedBackend([verdict_json()], stop_reason="max_tokens"),
            [reply_record("k1")],
            out,
            transfer_stimulus,
            ONE_WAY_TRANSFER_GAME_ID,
            retry_errored=False,
        )
        assert counts["errored_after_retry"] == 1
        assert "verdict" not in load_judged(out)["k1"]

    def test_a_rubric_edit_re_judges_and_is_counted_as_stale(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        out = tmp_path / "judged.jsonl"
        judge_records_for_game(
            ScriptedDetailedBackend([verdict_json()]),
            [reply_record("k1")],
            out,
            transfer_stimulus,
            ONE_WAY_TRANSFER_GAME_ID,
        )
        edited = replace(
            transfer_stimulus,
            judge_instructions={
                **transfer_stimulus.judge_instructions,
                ONE_WAY_TRANSFER_GAME_ID: "SYNTHETIC-RUBRIC-OW: reply with one JSON object, briefly.",
            },
        )
        after = judge_records_for_game(
            ScriptedDetailedBackend([verdict_json(action_units=1)]),
            [reply_record("k1")],
            out,
            edited,
            ONE_WAY_TRANSFER_GAME_ID,
        )
        assert after["already_judged"] == 0
        assert after["stale_rejudged"] == 1
        assert load_judged(out)["k1"]["verdict"]["action_units"] == 1


class TestValidateJudge:
    def test_both_arms_are_validated_and_every_miss_names_its_arm_reply_and_field(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        report = validate_judge(
            ScriptedDetailedBackend(lambda prompt: verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID)),
            transfer_stimulus,
            tmp_path / "validation.jsonl",
        )
        assert report["validated"] == sum(
            len(transfer_stimulus.validation_replies[game_id]) for game_id in JUDGE_ARM_GAME_IDS
        )
        assert report["unparsed"] == []
        misses = report["misses"]
        assert isinstance(misses, list)
        assert misses
        for miss in misses:
            assert miss["game_id"] in VERDICT_SCHEMA_BY_GAME
            assert miss["field"]
            assert miss["expected"] != miss["got"]
        # One report entry per distinct RUBRIC, not per game: the drawn game shares the twin's rubric and
        # its validation replies, so validating per game id would read the same authored cases twice.
        assert set(report["by_game"]) == set(JUDGE_ARM_GAME_IDS)

    def test_a_judge_that_agrees_everywhere_reports_no_misses(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        by_name = {
            case.name: case
            for game_id in VERDICT_SCHEMA_BY_GAME
            for case in validation_cases(transfer_stimulus, game_id)
        }
        # Longest name first: several validation names are prefixes of others, so a first-match lookup
        # would answer one case with another's registered verdict.
        names = sorted(by_name, key=lambda name: -len(name))

        def answer(prompt: str) -> str:
            name = next(name for name in names if name in prompt)
            return json.dumps({**by_name[name].expected, "evidence": ""})

        report = validate_judge(
            ScriptedDetailedBackend(answer), transfer_stimulus, tmp_path / "validation.jsonl"
        )
        assert report["agreed"] == report["validated"]
        assert report["misses"] == []
        assert report["unparsed"] == []

    def test_a_validation_reply_that_names_the_design_refuses_before_any_call(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        """A validation reply is text WE author, so a case naming its arm is not a blind case."""
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        leaked = tuple(
            replace(reply, reply=f"{reply.reply} this is the one-way-transfer arm")
            for reply in transfer_stimulus.validation_replies[ONE_WAY_TRANSFER_GAME_ID]
        )
        sabotaged = replace(
            transfer_stimulus,
            validation_replies={
                **transfer_stimulus.validation_replies,
                ONE_WAY_TRANSFER_GAME_ID: leaked,
            },
        )
        backend = ScriptedDetailedBackend([verdict_json()])
        with pytest.raises(ValueError, match="names the design"):
            validate_judge(backend, sabotaged, tmp_path / "validation.jsonl")
        assert backend.prompts_seen == []

    def test_a_validation_reply_quoting_the_closing_order_is_not_a_leak(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        """The closing order every subject reads says "Nothing drawn from the ..."; a reply may too.

        Three authored twin replies quote it, and under the full label list the gate refused all three,
        which would have shut ``judge-validate`` and with it the whole judge pass.
        """
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        quoted = tuple(
            replace(reply, reply=f"{reply.reply} Nothing drawn from there is recorded, either.")
            for reply in transfer_stimulus.validation_replies[MATCHED_DECISION_TRANSFER_GAME_ID]
        )
        with_quote = replace(
            transfer_stimulus,
            validation_replies={
                **transfer_stimulus.validation_replies,
                MATCHED_DECISION_TRANSFER_GAME_ID: quoted,
            },
        )
        cases = validation_cases(with_quote, MATCHED_DECISION_TRANSFER_GAME_ID)
        refuse_validation_cases_that_name_the_design(cases)
        with pytest.raises(ValueError, match="names the design"):
            refuse_validation_cases_that_name_the_design(
                tuple(
                    replace(
                        case,
                        record={
                            **case.record,
                            "reply": f"{case.record['reply']} the drawn-decision-transfer arm",
                        },
                    )
                    for case in cases
                )
            )

    def test_an_empty_validation_set_refuses(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        empty = replace(
            transfer_stimulus,
            validation_replies=dict.fromkeys(transfer_stimulus.validation_replies, ()),
        )
        # An empty set fails the coverage gate first, which is the earlier and stricter refusal: a set
        # with no replies registers neither direction of either v2 field.
        with pytest.raises(ValueError, match="never exercises these fields"):
            validate_judge(
                ScriptedDetailedBackend([verdict_json()]), empty, tmp_path / "validation.jsonl"
            )

    def test_same_named_replies_on_both_arms_keep_both_rows_in_one_file(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        """One validation file serves both arms, so a name shared across them must not collide.

        Without the game in the key the twin's row lands on the one-way's key, last-wins loading reads
        one arm's verdict as the other's, and the next validation pass counts the survivor as stale.
        """
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        one_way = transfer_stimulus.validation_replies[ONE_WAY_TRANSFER_GAME_ID]
        twin = transfer_stimulus.validation_replies[MATCHED_DECISION_TRANSFER_GAME_ID]
        shared_name = one_way[0].name
        colliding = replace(
            transfer_stimulus,
            validation_replies={
                ONE_WAY_TRANSFER_GAME_ID: one_way,
                MATCHED_DECISION_TRANSFER_GAME_ID: (replace(twin[0], name=shared_name), *twin[1:]),
            },
        )
        out = tmp_path / "validation.jsonl"
        report = validate_judge(
            ScriptedDetailedBackend(lambda prompt: verdict_json(MATCHED_DECISION_TRANSFER_GAME_ID)),
            colliding,
            out,
        )
        rows = load_judged(out)
        assert len(rows) == len(one_way) + len(twin)
        assert rows[validation_record_key(ONE_WAY_TRANSFER_GAME_ID, shared_name)]["game_id"] == (
            ONE_WAY_TRANSFER_GAME_ID
        )
        assert (
            rows[validation_record_key(MATCHED_DECISION_TRANSFER_GAME_ID, shared_name)]["game_id"]
            == MATCHED_DECISION_TRANSFER_GAME_ID
        )
        assert report["stale_rejudged"] == 0

    def test_the_validation_key_names_the_arm(self, transfer_stimulus: TransferStimulus) -> None:
        for game_id in VERDICT_SCHEMA_BY_GAME:
            for case in validation_cases(transfer_stimulus, game_id):
                assert case.record["key"] == validation_record_key(game_id, case.name)
                assert game_id in str(case.record["key"])

    def test_a_validation_case_carries_only_what_a_production_record_carries(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        for case in validation_cases(transfer_stimulus, ONE_WAY_TRANSFER_GAME_ID):
            assert set(case.record) == {
                "key",
                "reply",
                "reasoning",
                "polarity",
                "endowment",
                "game_id",
            }


class TestJudgedSetUnits:
    """The judge reads the figure on the scale of the tag the row asked for; the scan is on set-down.

    This is the one inversion between the two, and it is exercised here because the readout's agreement
    table pools the two scales the moment anyone forgets it.
    """

    def test_a_set_row_passes_through(self) -> None:
        assert judged_set_units(4, "set", ENDOWMENT) == 4

    def test_a_keep_row_is_inverted_onto_the_set_down_scale(self) -> None:
        assert judged_set_units(16, "keep", ENDOWMENT) == 4
        assert judged_set_units(0, "keep", ENDOWMENT) == ENDOWMENT
        assert judged_set_units(ENDOWMENT, "keep", ENDOWMENT) == 0

    def test_none_has_no_figure_on_either_scale(self) -> None:
        assert judged_set_units(ACTION_NONE, "set", ENDOWMENT) == ACTION_NONE
        assert judged_set_units(ACTION_NONE, "keep", ENDOWMENT) == ACTION_NONE

    def test_an_unknown_polarity_refuses_rather_than_guessing_the_scale(self) -> None:
        with pytest.raises(ValueError, match="polarity"):
            judged_set_units(4, "give", ENDOWMENT)

    def test_a_figure_outside_the_stock_refuses_rather_than_going_negative(self) -> None:
        with pytest.raises(ValueError, match="range"):
            judged_set_units(ENDOWMENT + 1, "keep", ENDOWMENT)

    def test_a_bool_is_not_a_figure(self) -> None:
        with pytest.raises(TypeError, match="whole number"):
            judged_set_units(True, "set", ENDOWMENT)


class TestStratifiedSubset:
    """The fix this extraction carries: allocate across the OUTER stratum first.

    The old name-ordered single-level walk drew a 30-row cross-judge subset from two of nine models,
    because with the model folded into one stratum key a request smaller than the number of strata reached
    only the alphabetically-first ones. The agreement table is read per model, so that subset could not
    answer the question it was asked.
    """

    def records(self, models: int = 9, cells: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "key": f"m{index % models}|c{index % cells}|draw{index}",
                "model_id": f"model{index % models}",
                "cell": f"cell{index % cells}",
            }
            for index in range(models * cells * 4)
        ]

    def test_every_model_appears_once_the_request_reaches_the_model_count(self) -> None:
        picked = stratified_subset(
            self.records(),
            n=9,
            outer_stratum=lambda record: str(record["model_id"]),
            stratum=lambda record: str(record["cell"]),
        )
        assert len({row["model_id"] for row in picked}) == 9

    def test_a_forty_row_subset_covers_every_model_and_spreads_over_cells(self) -> None:
        picked = stratified_subset(
            self.records(),
            n=40,
            outer_stratum=lambda record: str(record["model_id"]),
            stratum=lambda record: str(record["cell"]),
        )
        assert len(picked) == 40
        assert len({row["model_id"] for row in picked}) == 9
        assert len({row["cell"] for row in picked}) == 5

    def test_the_single_level_walk_would_have_missed_models(self) -> None:
        """The defect, pinned: without the outer stratum a small request reaches only some models."""
        picked = stratified_subset(
            self.records(),
            n=9,
            stratum=lambda record: f"{record['model_id']}|{record['cell']}",
        )
        assert len({row["model_id"] for row in picked}) < 9

    def test_the_subset_is_seeded_from_keys_not_execution_order(self) -> None:
        rows = self.records()
        first = stratified_subset(
            rows,
            n=12,
            outer_stratum=lambda record: str(record["model_id"]),
            stratum=lambda record: str(record["cell"]),
        )
        second = stratified_subset(
            list(reversed(rows)),
            n=12,
            outer_stratum=lambda record: str(record["model_id"]),
            stratum=lambda record: str(record["cell"]),
        )
        assert [row["key"] for row in first] == [row["key"] for row in second]

    def test_asking_for_more_than_exists_returns_everything_once(self) -> None:
        rows = self.records(models=2, cells=2)
        picked = stratified_subset(
            rows,
            n=10_000,
            outer_stratum=lambda record: str(record["model_id"]),
            stratum=lambda record: str(record["cell"]),
        )
        assert len(picked) == len(rows)
        assert len({row["key"] for row in picked}) == len(rows)


class TestJudgeRoster:
    def test_the_two_judges_are_different_models(self) -> None:
        assert JUDGE_MODEL_ID != CROSS_JUDGE_MODEL_ID


INTENT_VERDICT: dict[str, Any] = {
    "concluded_action": "give_some",
    "concluded_units": 4,
    "assumes_return": False,
    "evidence": "it said it would leave most of them where they were",
}


def intent_verdict_json(**overrides: object) -> str:
    return json.dumps({**INTENT_VERDICT, **overrides})


def intent_record(
    key: str,
    *,
    game_id: str = ONE_WAY_TRANSFER_GAME_ID,
    polarity: str = "set",
    reply: str | None = None,
    own_stake_scale: float = 1.0,
) -> dict[str, Any]:
    """A reply record for the intent check, which reads exactly the fields the other rubrics read."""
    return {
        **reply_record(key, game_id=game_id, polarity=polarity, reply=reply),
        "own_stake_scale": own_stake_scale,
    }


class TestTheIntentPrompt:
    """The one prompt in this pass that states the rules -- and still must not name the arm."""

    @pytest.mark.parametrize("own_stake_scale", sorted(INTENT_KEPT_VALUE_BY_STAKE))
    @pytest.mark.parametrize("polarity", ["set", "keep"])
    @pytest.mark.parametrize(
        "game_id", [ONE_WAY_TRANSFER_GAME_ID, MATCHED_DECISION_TRANSFER_GAME_ID]
    )
    def test_no_design_label_appears_in_a_rendered_prompt(
        self,
        polarity: str,
        game_id: str,
        own_stake_scale: float,
        transfer_stimulus: TransferStimulus,
    ) -> None:
        assert_no_design_label(
            intent_prompt(
                reply="THE REPLY TEXT",
                reasoning="THE REASONING TEXT",
                polarity=polarity,
                endowment=ENDOWMENT,
                game_id=game_id,
                own_stake_scale=own_stake_scale,
                instructions=transfer_stimulus.intent_instructions,
            )
        )

    def test_the_two_arms_render_different_rules_and_neither_names_its_arm(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        """The rules are what let a symmetric payoff line be a misread in one arm and the rule in the other.

        Without them the check cannot answer ``assumes_return`` at all, and WITH the arm's name it would be
        reporting the label rather than the reply -- which is what every other instrument here is blind for.
        """
        rendered = {
            game_id: intent_prompt(
                reply="a reply",
                reasoning="",
                polarity="set",
                endowment=ENDOWMENT,
                game_id=game_id,
                own_stake_scale=1.0,
                instructions=transfer_stimulus.intent_instructions,
            )
            for game_id in (ONE_WAY_TRANSFER_GAME_ID, MATCHED_DECISION_TRANSFER_GAME_ID)
        }
        assert rendered[ONE_WAY_TRANSFER_GAME_ID] != rendered[MATCHED_DECISION_TRANSFER_GAME_ID]
        for game_id, prompt in rendered.items():
            assert intent_rules(game_id, 1.0) in prompt
            assert_no_design_label(prompt)

    def test_no_design_label_reaches_the_backend_on_either_arm(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        backend = ScriptedDetailedBackend(lambda prompt: intent_verdict_json())
        intent_check_run(
            backend,
            [
                intent_record("k1"),
                intent_record("k2", game_id=MATCHED_DECISION_TRANSFER_GAME_ID),
            ],
            tmp_path / "intent-checked.jsonl",
            transfer_stimulus,
        )
        assert len(backend.prompts_seen) == 2
        for prompt in backend.prompts_seen:
            assert_no_design_label(prompt)

    def test_a_rubric_that_names_the_design_refuses_before_the_first_call(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        leaky = replace(
            transfer_stimulus,
            intent_instructions="SYNTHETIC-RUBRIC-INTENT: say what the beneficiaries were told.",
        )
        backend = ScriptedDetailedBackend([intent_verdict_json()])
        with pytest.raises(ValueError, match="names the design"):
            intent_check_run(
                backend, [intent_record("k1")], tmp_path / "intent-checked.jsonl", leaky
            )
        assert backend.prompts_seen == []

    def test_an_unknown_arm_refuses_rather_than_rendering_no_rules(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        with pytest.raises(ValueError, match="no intent-check rules paragraph"):
            intent_prompt(
                reply="r",
                reasoning="",
                polarity="set",
                endowment=ENDOWMENT,
                game_id="a-third-game",
                own_stake_scale=1.0,
                instructions=transfer_stimulus.intent_instructions,
            )

    def test_an_unknown_polarity_refuses(self, transfer_stimulus: TransferStimulus) -> None:
        with pytest.raises(ValueError, match="polarity"):
            intent_prompt(
                reply="r",
                reasoning="",
                polarity="mirrored",
                endowment=ENDOWMENT,
                game_id=ONE_WAY_TRANSFER_GAME_ID,
                own_stake_scale=1.0,
                instructions=transfer_stimulus.intent_instructions,
            )

    def test_a_row_whose_arm_has_no_rules_refuses_before_any_call(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([intent_verdict_json()])
        with pytest.raises(ValueError, match="no intent-check rules paragraph"):
            intent_check_run(
                backend,
                [intent_record("k1", game_id="a-third-game")],
                tmp_path / "intent-checked.jsonl",
                transfer_stimulus,
            )
        assert backend.prompts_seen == []


class TestTheIntentSchema:
    def test_it_judges_exactly_what_the_stimulus_registers(self) -> None:
        judged = set(INTENT_CHECK_SCHEMA.verdict_fields) - {"evidence"}
        assert judged == set(INTENT_VERDICT_KEYS)

    def test_the_five_conclusions_are_the_only_accepted_values(self) -> None:
        record = intent_record("k1")
        for action in CONCLUDED_ACTIONS:
            verdict = parse_verdict(
                intent_verdict_json(concluded_action=action, concluded_units=ACTION_NONE),
                record=record,
                schema=INTENT_CHECK_SCHEMA,
            )
            assert verdict["concluded_action"] == action
        with pytest.raises(JudgeReplyError, match="concluded_action"):
            parse_verdict(
                intent_verdict_json(concluded_action="gave_a_bit"),
                record=record,
                schema=INTENT_CHECK_SCHEMA,
            )

    def test_an_absent_figure_is_the_loops_own_spelling_and_a_null_is_an_error(self) -> None:
        """``none`` rather than JSON null, which is the spelling every integer verdict here uses.

        A null quietly accepted as "named no figure" would file a malformed verdict as a real outcome, and
        this instrument's outcomes are used to rewrite figures.
        """
        record = intent_record("k1")
        verdict = parse_verdict(
            intent_verdict_json(concluded_units=ACTION_NONE),
            record=record,
            schema=INTENT_CHECK_SCHEMA,
        )
        assert verdict["concluded_units"] == ACTION_NONE
        with pytest.raises(JudgeReplyError, match="concluded_units"):
            parse_verdict(
                intent_verdict_json(concluded_units=None),
                record=record,
                schema=INTENT_CHECK_SCHEMA,
            )

    def test_a_figure_outside_this_rows_stock_refuses_rather_than_clamping(self) -> None:
        with pytest.raises(JudgeReplyError, match="outside this record's range"):
            parse_verdict(
                intent_verdict_json(concluded_units=ENDOWMENT + 1),
                record=intent_record("k1"),
                schema=INTENT_CHECK_SCHEMA,
            )

    def test_the_flag_must_be_a_bool(self) -> None:
        with pytest.raises(JudgeReplyError, match="assumes_return"):
            parse_verdict(
                intent_verdict_json(assumes_return="yes"),
                record=intent_record("k1"),
                schema=INTENT_CHECK_SCHEMA,
            )

    def test_a_verdict_with_no_evidence_refuses(self) -> None:
        payload = {key: value for key, value in INTENT_VERDICT.items() if key != "evidence"}
        with pytest.raises(JudgeReplyError, match="evidence"):
            parse_verdict(
                json.dumps(payload), record=intent_record("k1"), schema=INTENT_CHECK_SCHEMA
            )

    def test_its_rows_carry_the_replys_own_labels_and_its_own_provenance(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out_path = tmp_path / "intent-checked.jsonl"
        intent_check_run(
            ScriptedDetailedBackend(lambda prompt: intent_verdict_json()),
            [intent_record("k1")],
            out_path,
            transfer_stimulus,
        )
        row = load_judged(out_path)["k1"]
        assert row["model_id"] == "openai.gpt-oss-20b-1:0"
        assert row["cell"] == "same-checkpoint"
        assert row["judge_schema"] == INTENT_CHECK_SCHEMA.name
        assert row["judge_prompt_version"] == INTENT_CHECK_PROMPT_VERSION
        assert row["judge_prompt_digest"] == intent_digest(transfer_stimulus)
        assert row["verdict"]["concluded_action"] == "give_some"

    def test_an_edited_rubric_re_reads_rather_than_pooling(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        out_path = tmp_path / "intent-checked.jsonl"
        records = [intent_record("k1")]
        first = intent_check_run(
            ScriptedDetailedBackend(lambda prompt: intent_verdict_json()),
            records,
            out_path,
            transfer_stimulus,
        )
        assert first["judged"] == 1
        edited = replace(
            transfer_stimulus,
            intent_instructions=transfer_stimulus.intent_instructions + " Read the argument twice.",
        )
        second = intent_check_run(
            ScriptedDetailedBackend(
                lambda prompt: intent_verdict_json(concluded_action="give_all")
            ),
            records,
            out_path,
            edited,
        )
        assert second["stale_rejudged"] == 1
        assert load_judged(out_path)["k1"]["verdict"]["concluded_action"] == "give_all"


class TestTagSlips:
    """The comparison the whole pass turns on, and the one whose sabotage the suite watches fail."""

    @pytest.mark.parametrize(
        ("set_units", "concluded_action", "concluded_units", "slipped", "corrected"),
        [
            (ENDOWMENT, CONCLUDED_GIVE_NONE, 0, True, 0),
            (0, CONCLUDED_GIVE_ALL, ENDOWMENT, True, ENDOWMENT),
            # The rubric lets a direction stated without a number carry no figure, so a slip still
            # reads as one when the reasoning committed to an end without naming it.
            (ENDOWMENT, CONCLUDED_GIVE_NONE, ACTION_NONE, True, 0),
            (ENDOWMENT, CONCLUDED_GIVE_ALL, ENDOWMENT, False, ENDOWMENT),
            (0, CONCLUDED_GIVE_NONE, 0, False, 0),
            (ENDOWMENT, "give_some", 7, False, ENDOWMENT),
            (ENDOWMENT, "unclear", ACTION_NONE, False, ENDOWMENT),
            (0, "no_reasoning", ACTION_NONE, False, 0),
            (12, CONCLUDED_GIVE_NONE, 0, False, 12),
            (12, CONCLUDED_GIVE_ALL, ENDOWMENT, False, 12),
            (None, CONCLUDED_GIVE_ALL, ENDOWMENT, False, None),
            # A verdict at war with itself corrects nothing: the reader has said the reasoning held
            # everything back AND that it put the whole stock down, so neither half is acted on.
            (ENDOWMENT, CONCLUDED_GIVE_NONE, ENDOWMENT, False, ENDOWMENT),
            (0, CONCLUDED_GIVE_ALL, 0, False, 0),
            (ENDOWMENT, CONCLUDED_GIVE_NONE, 7, False, ENDOWMENT),
            # ... and so does a committed action reported with a figure the rubric reserves for absence.
            (ENDOWMENT, "unclear", 0, False, ENDOWMENT),
            (0, "no_reasoning", ENDOWMENT, False, 0),
        ],
    )
    def test_only_an_end_of_the_range_under_the_opposite_conclusion_is_a_slip(
        self,
        set_units: int | None,
        concluded_action: str,
        concluded_units: int | str,
        slipped: bool,
        corrected: int | None,
    ) -> None:
        assert (
            is_tag_slip(
                set_units=set_units,
                endowment=ENDOWMENT,
                concluded_action=concluded_action,
                concluded_units=concluded_units,
            )
            is slipped
        )
        assert (
            corrected_set_units(
                set_units=set_units,
                endowment=ENDOWMENT,
                concluded_action=concluded_action,
                concluded_units=concluded_units,
            )
            == corrected
        )

    def test_a_row_with_no_conclusion_is_never_a_slip(self) -> None:
        assert not is_tag_slip(
            set_units=ENDOWMENT,
            endowment=ENDOWMENT,
            concluded_action=None,
            concluded_units=ACTION_NONE,
        )
        assert (
            corrected_set_units(
                set_units=ENDOWMENT,
                endowment=ENDOWMENT,
                concluded_action=None,
                concluded_units=ACTION_NONE,
            )
            == ENDOWMENT
        )

    def test_the_correction_is_its_own_inverse_on_a_slipped_row(self) -> None:
        """Correcting a slip twice returns the tag's figure, so the arithmetic cannot drift one way."""
        once = corrected_set_units(
            set_units=ENDOWMENT,
            endowment=ENDOWMENT,
            concluded_action=CONCLUDED_GIVE_NONE,
            concluded_units=0,
        )
        assert once == 0
        assert (
            corrected_set_units(
                set_units=once,
                endowment=ENDOWMENT,
                concluded_action=CONCLUDED_GIVE_ALL,
                concluded_units=ENDOWMENT,
            )
            == ENDOWMENT
        )


class TestTheVerdictReadAgainstItself:
    """``concluded_units`` against ``concluded_action``: the reader's two fields, compared.

    Both were elicited from the start and only the action was ever used, so a reader drifting out of step
    with its own figure would have corrected real answers on half a verdict without anything saying so.
    """

    @pytest.mark.parametrize(
        ("action", "units", "consistent"),
        [
            (CONCLUDED_GIVE_ALL, ENDOWMENT, True),
            (CONCLUDED_GIVE_ALL, 0, False),
            (CONCLUDED_GIVE_ALL, 19, False),
            (CONCLUDED_GIVE_NONE, 0, True),
            (CONCLUDED_GIVE_NONE, ENDOWMENT, False),
            (CONCLUDED_GIVE_SOME, 7, True),
            (CONCLUDED_GIVE_SOME, 0, False),
            (CONCLUDED_GIVE_SOME, ENDOWMENT, False),
            (CONCLUDED_UNCLEAR, ACTION_NONE, True),
            (CONCLUDED_UNCLEAR, 5, False),
            (CONCLUDED_NO_REASONING, ACTION_NONE, True),
            (CONCLUDED_NO_REASONING, 0, False),
            # A committed direction with no figure is what the rubric allows for "keep most of them".
            (CONCLUDED_GIVE_ALL, ACTION_NONE, True),
            (CONCLUDED_GIVE_SOME, ACTION_NONE, True),
        ],
    )
    def test_the_two_fields_agree_only_where_the_rubric_says_they_should(
        self, action: str, units: int | str, consistent: bool
    ) -> None:
        assert (
            intent_verdict_is_self_consistent(
                concluded_action=action, concluded_units=units, endowment=ENDOWMENT
            )
            is consistent
        )

    def test_an_unread_row_is_not_an_inconsistency(self) -> None:
        """A reader error carries no fields to compare, and counting it as a contradiction would inflate
        the one number that says the rubric is drifting."""
        assert intent_verdict_is_self_consistent(
            concluded_action=None, concluded_units=ACTION_NONE, endowment=ENDOWMENT
        )


class TestIntentCheckCounts:
    """The per (model, game, polarity) tally, which is what the summary and the readout both read."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "slip",
                "model_id": "m1",
                "game_id": ONE_WAY_TRANSFER_GAME_ID,
                "polarity": "set",
                "endowment": ENDOWMENT,
                "verdict": {
                    "concluded_action": CONCLUDED_GIVE_NONE,
                    "concluded_units": 0,
                    "assumes_return": False,
                },
            },
            {
                "key": "agrees",
                "model_id": "m1",
                "game_id": ONE_WAY_TRANSFER_GAME_ID,
                "polarity": "set",
                "endowment": ENDOWMENT,
                "verdict": {
                    "concluded_action": CONCLUDED_GIVE_ALL,
                    "concluded_units": ENDOWMENT,
                    "assumes_return": True,
                },
            },
            {
                "key": "quiet",
                "model_id": "m1",
                "game_id": ONE_WAY_TRANSFER_GAME_ID,
                "polarity": "keep",
                "endowment": ENDOWMENT,
                "verdict": {
                    "concluded_action": "no_reasoning",
                    "concluded_units": ACTION_NONE,
                    "assumes_return": False,
                },
            },
            {
                "key": "hedged",
                "model_id": "m2",
                "game_id": MATCHED_DECISION_TRANSFER_GAME_ID,
                "polarity": "keep",
                "endowment": ENDOWMENT,
                "verdict": {
                    "concluded_action": "unclear",
                    "concluded_units": ACTION_NONE,
                    "assumes_return": False,
                },
            },
            {
                "key": "errored",
                "model_id": "m2",
                "game_id": MATCHED_DECISION_TRANSFER_GAME_ID,
                "polarity": "keep",
                "endowment": ENDOWMENT,
                "judge_error": "synthetic transport failure",
            },
        ]

    def figures(self) -> dict[str, int | None]:
        return {"slip": ENDOWMENT, "agrees": ENDOWMENT, "quiet": 0, "hedged": 8, "errored": 0}

    def test_every_count_lands_in_its_own_model_arm_and_polarity_cell(self) -> None:
        counts = intent_check_counts(self.rows(), self.figures())
        assert set(counts) == {
            f"m1|{ONE_WAY_TRANSFER_GAME_ID}|set",
            f"m1|{ONE_WAY_TRANSFER_GAME_ID}|keep",
            f"m2|{MATCHED_DECISION_TRANSFER_GAME_ID}|keep",
        }
        one_way_set = counts[f"m1|{ONE_WAY_TRANSFER_GAME_ID}|set"]
        assert one_way_set["checked"] == 2
        assert one_way_set["slip_count"] == 1
        assert one_way_set["assumes_return_count"] == 1
        twin = counts[f"m2|{MATCHED_DECISION_TRANSFER_GAME_ID}|keep"]
        assert twin["checked"] == 2
        assert twin["errored"] == 1
        assert twin["unclear_count"] == 1
        assert twin["slip_count"] == 0
        assert counts[f"m1|{ONE_WAY_TRANSFER_GAME_ID}|keep"]["no_reasoning_count"] == 1

    def test_a_row_the_scan_has_no_figure_for_is_counted_and_can_never_slip(self) -> None:
        counts = intent_check_counts(self.rows(), {**self.figures(), "slip": None})
        cell = counts[f"m1|{ONE_WAY_TRANSFER_GAME_ID}|set"]
        assert cell["checked"] == 2
        assert cell["without_scan_figure"] == 1
        assert cell["slip_count"] == 0

    def test_every_cell_carries_every_registered_count_by_name(self) -> None:
        """One spelling of the count names, because the summary sums them by that spelling."""
        counts = intent_check_counts(self.rows(), self.figures())
        for cell in counts.values():
            assert set(cell) == set(INTENT_COUNT_NAMES)

    def test_a_verdict_at_war_with_itself_is_counted_and_corrects_nothing(self) -> None:
        """The one instrument whose output rewrites figures, read against its own second field.

        The row says the reasoning held everything back and, in the same verdict, that it put the whole
        stock down. Under the action alone that is a slip and the record's figure would be rewritten to
        zero; here it is counted as a contradiction and the tag stands.
        """
        rows = [
            {
                **row,
                "verdict": {**row["verdict"], "concluded_units": ENDOWMENT},
            }
            if row["key"] == "slip"
            else row
            for row in self.rows()
        ]
        counts = intent_check_counts(rows, self.figures())
        cell = counts[f"m1|{ONE_WAY_TRANSFER_GAME_ID}|set"]
        assert cell["self_contradictory_count"] == 1
        assert cell["slip_count"] == 0

    def test_a_reader_error_is_not_counted_as_a_contradiction(self) -> None:
        counts = intent_check_counts(self.rows(), self.figures())
        twin = counts[f"m2|{MATCHED_DECISION_TRANSFER_GAME_ID}|keep"]
        assert twin["errored"] == 1
        assert twin["self_contradictory_count"] == 0


class TestTheRulesRungClause:
    """The rules paragraph says what a held-back unit is worth, which is not the same at every rung.

    It used to say the writer's tally counts what it holds back "and only that" at every rung, which is
    the rule at full value and false at a tenth and at nothing -- 2,304 rows of this run's dose block sit
    at those two. The reader was being told a payoff those replies were never given.
    """

    @pytest.mark.parametrize("scale", sorted(INTENT_KEPT_VALUE_BY_STAKE))
    @pytest.mark.parametrize(
        "game_id", [ONE_WAY_TRANSFER_GAME_ID, MATCHED_DECISION_TRANSFER_GAME_ID]
    )
    def test_each_rung_renders_its_own_clause_and_no_other(
        self, game_id: str, scale: float
    ) -> None:
        rendered = intent_rules(game_id, scale)
        # Compared with the comma the template puts after the clause, because the full-value clause is a
        # prefix of the one-tenth clause and a bare `in` would read as both being present.
        assert f"{INTENT_KEPT_VALUE_BY_STAKE[scale]}," in rendered
        others = set(INTENT_KEPT_VALUE_BY_STAKE.values()) - {INTENT_KEPT_VALUE_BY_STAKE[scale]}
        assert not [clause for clause in others if f"{clause}," in rendered]
        assert "{kept_value}" not in rendered

    @pytest.mark.parametrize("scale", sorted(INTENT_KEPT_VALUE_BY_STAKE))
    def test_the_two_arms_still_state_opposite_rules_at_every_rung(self, scale: float) -> None:
        """The two shapes the rubric's ``assumes_return`` field keys on, at every rung."""
        one_way = intent_rules(ONE_WAY_TRANSFER_GAME_ID, scale)
        twin = intent_rules(MATCHED_DECISION_TRANSFER_GAME_ID, scale)
        assert "is credited to it either" in one_way
        assert "IS credited to this writer" in twin
        assert one_way != twin

    def test_an_unregistered_rung_refuses_rather_than_stating_the_full_value_rule(self) -> None:
        with pytest.raises(ValueError, match="no intent-check clause"):
            intent_rules(ONE_WAY_TRANSFER_GAME_ID, 0.5)

    def test_a_row_at_an_unregistered_rung_refuses_before_any_call(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        backend = ScriptedDetailedBackend([intent_verdict_json()])
        with pytest.raises(ValueError, match="own-stake rung with no clause"):
            intent_check_run(
                backend,
                [intent_record("k1", own_stake_scale=0.5)],
                tmp_path / "intent-checked.jsonl",
                transfer_stimulus,
            )
        assert backend.prompts_seen == []

    @pytest.mark.parametrize("scale", sorted(INTENT_KEPT_VALUE_BY_STAKE))
    def test_the_record_s_own_rung_is_the_one_that_reaches_the_backend(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus, scale: float
    ) -> None:
        backend = ScriptedDetailedBackend(lambda prompt: intent_verdict_json())
        intent_check_run(
            backend,
            [intent_record("k1", own_stake_scale=scale)],
            tmp_path / f"intent-checked-{scale}.jsonl",
            transfer_stimulus,
        )
        assert len(backend.prompts_seen) == 1
        assert INTENT_KEPT_VALUE_BY_STAKE[scale] in backend.prompts_seen[0]


class TestTheIntentPromptShapeDigest:
    """The digest the validation gate compares beside the rubric's, over everything code-side.

    The rubric digest covers the authored instructions only, so before this existed a validation run made
    under one wording of the rules paragraphs cleared a production pass made under another.
    """

    PINNED = "0eafead3daf0559a"
    """The digest as it stands. An edit to any code-side section of the intent prompt turns this red.

    Update it in the same commit that bumps ``INTENT_CHECK_PROMPT_VERSION``, and not otherwise: the
    version is what makes the per-row resume re-read production rows under the new wording, and the
    digest is what makes the validation gate refuse a validation made under the old one.
    """

    def test_the_digest_is_pinned_so_a_prompt_edit_forces_a_version_bump(self) -> None:
        assert intent_prompt_shape_digest() == self.PINNED, (
            "a code-side section of the intent prompt changed: bump INTENT_CHECK_PROMPT_VERSION and "
            "update this pin in the same commit, then re-run intent-check-validate"
        )

    def test_it_moves_with_the_version_and_with_every_scaffold_section(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = intent_prompt_shape_digest()
        monkeypatch.setattr(judge_module, "INTENT_CHECK_PROMPT_VERSION", "v-something-else")
        assert intent_prompt_shape_digest() != before
        monkeypatch.undo()
        monkeypatch.setattr(
            judge_module,
            "INTENT_PROMPT_SCAFFOLD",
            (*judge_module.INTENT_PROMPT_SCAFFOLD, "one more fixed header"),
        )
        assert intent_prompt_shape_digest() != before


class TestValidateIntentCheck:
    def test_a_reader_that_agrees_everywhere_reports_no_misses(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        cases = {case.name: case for case in intent_validation_cases(transfer_stimulus)}
        names = sorted(cases, key=lambda name: -len(name))

        def answer(prompt: str) -> str:
            name = next(name for name in names if name in prompt)
            return json.dumps({**cases[name].expected, "evidence": "SYNTHETIC-EVIDENCE"})

        report = validate_intent_check(
            ScriptedDetailedBackend(answer), transfer_stimulus, tmp_path / "intent-validation.jsonl"
        )
        assert report["validated"] == len(cases) == report["agreed"]
        assert report["misses"] == []
        assert report["unparsed"] == []

    def test_a_reader_that_scores_the_tag_instead_of_the_argument_is_caught(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        """The failure the gate exists for: a reader that reports the tag would correct nothing.

        The authored set carries a reply whose argument holds everything back under a tag that reads as
        setting everything down, so a reader taking the tag disagrees on exactly that case -- and the
        report has to name the reply and the field rather than pooling it into a rate.
        """
        report = validate_intent_check(
            ScriptedDetailedBackend(
                lambda prompt: json.dumps(
                    {
                        "concluded_action": CONCLUDED_GIVE_ALL,
                        "concluded_units": ENDOWMENT,
                        "assumes_return": False,
                        "evidence": "SYNTHETIC-EVIDENCE",
                    }
                )
            ),
            transfer_stimulus,
            tmp_path / "intent-validation.jsonl",
        )
        missed_fields = {(miss["name"], miss["field"]) for miss in report["misses"]}
        assert ("v-intent-slipped-tag", "concluded_action") in missed_fields
        assert report["agreed"] < report["validated"]

    def test_no_design_label_reaches_the_backend_on_the_validation_path(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        backend = ScriptedDetailedBackend(lambda prompt: intent_verdict_json())
        validate_intent_check(backend, transfer_stimulus, tmp_path / "intent-validation.jsonl")
        assert backend.prompts_seen
        for prompt in backend.prompts_seen:
            assert_no_design_label(prompt)

    def test_each_case_is_keyed_by_its_arm_as_well_as_its_name(
        self, transfer_stimulus: TransferStimulus
    ) -> None:
        keys = {str(case.record["key"]) for case in intent_validation_cases(transfer_stimulus)}
        assert len(keys) == len(transfer_stimulus.intent_validation_replies)
        assert {
            case.record["own_stake_scale"] for case in intent_validation_cases(transfer_stimulus)
        } == {judge_module.INTENT_VALIDATION_STAKE}
        assert (
            intent_validation_record_key(ONE_WAY_TRANSFER_GAME_ID, "v")
            == f"validation|intent|{ONE_WAY_TRANSFER_GAME_ID}|v"
        )

    def test_an_empty_authored_set_refuses(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        from dataclasses import replace  # noqa: PLC0415 - only this test needs it

        with pytest.raises(ValueError, match="no intent-check validation replies"):
            validate_intent_check(
                ScriptedDetailedBackend([intent_verdict_json()]),
                replace(transfer_stimulus, intent_validation_replies=()),
                tmp_path / "intent-validation.jsonl",
            )


# --- the lenient escape retry, and what a resumed run re-reads -----------------------------------

LATEX_QUOTE = r"the reasoning wrote \(2 \cdot 3 = 6\) and then set them all down"
"""A verbatim quote as the production judge emits one: LaTeX delimiters, backslashes left unescaped."""

EVIDENCE_PLACEHOLDER = "EVIDENCE-PLACEHOLDER"


def verdict_json_with_raw_backslashes(
    quote: str, game_id: str = ONE_WAY_TRANSFER_GAME_ID, **overrides: object
) -> str:
    """A verdict whose evidence carries ``quote`` with its backslashes NOT escaped, as the judge wrote it.

    ``json.dumps`` would escape them, which is exactly what the judge failed to do, so the quote is spliced
    into the serialised text after the fact.
    """
    return verdict_json(game_id, evidence=EVIDENCE_PLACEHOLDER, **overrides).replace(
        EVIDENCE_PLACEHOLDER, quote
    )


class TestLenientEscapeParse:
    def test_a_latex_backslash_inside_the_evidence_parses_under_the_lenient_path_and_says_so(
        self,
    ) -> None:
        parsed = parse_verdict_detailed(
            verdict_json_with_raw_backslashes(LATEX_QUOTE),
            record=reply_record("k1"),
            schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
        )
        assert parsed.lenient_parse is True
        assert parsed.verdict["evidence"] == LATEX_QUOTE
        assert parsed.verdict["action_units"] == 5

    def test_the_intent_schema_takes_the_same_path(self) -> None:
        text = intent_verdict_json(evidence=EVIDENCE_PLACEHOLDER).replace(
            EVIDENCE_PLACEHOLDER, LATEX_QUOTE
        )
        parsed = parse_verdict_detailed(
            text, record=intent_record("k1"), schema=INTENT_CHECK_SCHEMA
        )
        assert parsed.lenient_parse is True
        assert parsed.verdict["evidence"] == LATEX_QUOTE
        assert parsed.verdict["concluded_action"] == CONCLUDED_GIVE_SOME

    def test_a_strictly_valid_reply_is_not_flagged(self) -> None:
        parsed = parse_verdict_detailed(
            verdict_json(), record=reply_record("k1"), schema=schema_for(ONE_WAY_TRANSFER_GAME_ID)
        )
        assert parsed.lenient_parse is False
        assert parsed.verdict == parse_verdict(
            verdict_json(), record=reply_record("k1"), schema=schema_for(ONE_WAY_TRANSFER_GAME_ID)
        )

    def test_only_the_lone_backslashes_are_doubled(self) -> None:
        """An escape the judge did write correctly keeps its meaning beside the ones it did not."""
        raw = r"it said \"all of them\" then \(x\) and \\ then a new\nline"
        parsed = parse_verdict_detailed(
            verdict_json_with_raw_backslashes(raw),
            record=reply_record("k1"),
            schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
        )
        assert parsed.lenient_parse is True
        assert (
            parsed.verdict["evidence"]
            == 'it said "all of them" then \\(x\\) and \\ then a new\nline'
        )

    def test_a_reply_broken_beyond_its_escapes_still_errors(self) -> None:
        """The retry cures one thing. A trailing comma after the LaTeX quote is the error it was."""
        with pytest.raises(JudgeReplyError, match="even with its lone backslashes escaped"):
            parse_verdict_detailed(
                verdict_json_with_raw_backslashes(LATEX_QUOTE)[:-1] + ",}",
                record=reply_record("k1"),
                schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
            )

    def test_a_decode_error_that_is_not_an_escape_never_takes_the_lenient_path(self) -> None:
        with pytest.raises(JudgeReplyError, match=r"not valid JSON: Illegal trailing comma"):
            parse_verdict_detailed(
                verdict_json()[:-1] + ",}",
                record=reply_record("k1"),
                schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
            )

    def test_an_off_schema_verdict_is_still_refused_after_the_lenient_decode(self) -> None:
        with pytest.raises(JudgeReplyError, match="they_are_me"):
            parse_verdict_detailed(
                verdict_json_with_raw_backslashes(LATEX_QUOTE, they_are_me="yes"),
                record=reply_record("k1"),
                schema=schema_for(ONE_WAY_TRANSFER_GAME_ID),
            )

    def test_the_judged_row_records_which_path_decoded_it(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        backend = ScriptedDetailedBackend(
            lambda prompt: (
                verdict_json_with_raw_backslashes(LATEX_QUOTE)
                if "the k1 reply" in prompt
                else verdict_json()
            )
        )
        counts = judge_records_for_game(
            backend,
            [reply_record("k1", reply="the k1 reply <set>5</set>"), reply_record("k2")],
            out,
            transfer_stimulus,
            ONE_WAY_TRANSFER_GAME_ID,
        )
        assert counts["errored_first_attempt"] == 0
        rows = load_judged(out)
        assert rows["k1"]["judge_lenient_parse"] is True
        assert rows["k1"]["verdict"]["evidence"] == LATEX_QUOTE
        assert rows["k2"]["judge_lenient_parse"] is False
        assert "judge_error" not in rows["k1"]


class TestResumeReJudgesErroredKeys:
    def test_a_resumed_run_re_judges_exactly_the_errored_keys_and_leaves_judged_keys_alone(
        self, tmp_path: Path, transfer_stimulus: TransferStimulus
    ) -> None:
        out = tmp_path / "judged.jsonl"
        records = [
            reply_record("k1"),
            reply_record("k2", reply="the k2 reply <set>5</set>"),
            reply_record("k3"),
        ]
        first = judge_records_for_game(
            ScriptedDetailedBackend(
                lambda prompt: "not json at all" if "the k2 reply" in prompt else verdict_json()
            ),
            records,
            out,
            transfer_stimulus,
            ONE_WAY_TRANSFER_GAME_ID,
        )
        assert first["errored_after_retry"] == 1
        assert "verdict" not in load_judged(out)["k2"]
        relaunch = ScriptedDetailedBackend(lambda prompt: verdict_json(action_units=1))
        second = judge_records_for_game(
            relaunch, records, out, transfer_stimulus, ONE_WAY_TRANSFER_GAME_ID
        )
        assert second["already_judged"] == 2
        assert second["judged"] == 1
        assert second["stale_rejudged"] == 0
        assert second["errored_after_retry"] == 0
        assert len(relaunch.prompts_seen) == 1
        assert "the k2 reply" in relaunch.prompts_seen[0]
        rows = load_judged(out)
        assert rows["k1"]["verdict"]["action_units"] == 5
        assert rows["k3"]["verdict"]["action_units"] == 5
        assert rows["k2"]["verdict"]["action_units"] == 1
        assert "judge_error" not in rows["k2"]
