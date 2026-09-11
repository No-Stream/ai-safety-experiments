"""Every record builder in the sociology studies carries the five per-call telemetry fields.

One completion with every telemetry field set to a value distinguishable from every other, pushed
through each builder: the three reply-record builders spell the fields under
:func:`sociology.records.completion_telemetry`'s names, the judged-row builders under the same names
with the ``judge_`` prefix their rows already use for their own call's cost. ``input_tokens`` must stay
the total the model read -- the cache split rides beside it, never carved out of it.

SABOTAGE target: drop ``**completion_telemetry(completion)`` (or ``**judge_telemetry(completion)``)
from any one builder and exactly that builder's case goes red.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.model_backend import BedrockCompletion, TokenUsage
from sociology import decoupled_judge, decoupled_ladder, judge, runner, transfer_cli
from sociology.decoupled_plan import leg_for as decoupled_leg_for
from sociology.decoupled_plan import planned_calls_for_leg as decoupled_calls_for_leg
from sociology.judge_loop import judge_telemetry
from sociology.records import completion_telemetry
from sociology.transfer_plan import leg_for as transfer_leg_for
from sociology.transfer_plan import planned_calls_for_leg as transfer_calls_for_leg

if TYPE_CHECKING:
    from pathlib import Path

    from sociology.decoupled_stimulus import DecoupledStimulus
    from sociology.stimulus import Stimulus
    from sociology.transfer_stimulus import TransferStimulus

FLOOR_LEG_ID = "gpt-oss-20b--default--B--floor"

EXPECTED_TELEMETRY: dict[str, Any] = {
    "cache_read_input_tokens": 1000,
    "cache_write_input_tokens": 100,
    "elapsed_seconds": 12.5,
    "first_event_seconds": 3.25,
    "attempts": 2,
}
INPUT_TOKENS_TOTAL = 1234


def completion_with_telemetry(text: str) -> BedrockCompletion:
    return BedrockCompletion(
        text=text,
        reasoning="",
        usage=TokenUsage(
            input_tokens=INPUT_TOKENS_TOTAL,
            output_tokens=56,
            cache_read_input_tokens=1000,
            cache_write_input_tokens=100,
        ),
        stop_reason="end_turn",
        elapsed_seconds=12.5,
        first_event_seconds=3.25,
        attempts=2,
    )


class FixedCompletionBackend:
    """Serve one fixed completion for every prompt, so a judged row's telemetry has a known source."""

    model_id = "fixed-judge"

    def __init__(self, text: str) -> None:
        self._completion = completion_with_telemetry(text)

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        return [self._completion for _ in prompts]


def assert_telemetry(row: dict[str, Any], *, prefix: str = "") -> None:
    assert {name: row[f"{prefix}{name}"] for name in EXPECTED_TELEMETRY} == EXPECTED_TELEMETRY
    assert row[f"{prefix}input_tokens"] == INPUT_TOKENS_TOTAL, "the total must not lose the split"


class TestTheSharedSpelling:
    def test_completion_telemetry_flattens_exactly_the_five_fields(self) -> None:
        assert completion_telemetry(completion_with_telemetry("r")) == EXPECTED_TELEMETRY


class TestReplyRecordsCarryTelemetry:
    def test_the_stage_one_reply_record(self, stimulus: Stimulus) -> None:
        call = runner.PlannedCall(
            key="k",
            cell="center",
            bundle_id="bundle-a",
            family="agentic-120b",
            framing="population",
            cues="kept",
            size=10,
            model_id="scripted",
            transport="live",
            draw=0,
            reasoning_effort=None,
            prompt="PROMPT",
        )
        assert_telemetry(runner._reply_record(call, completion_with_telemetry("r"), stimulus))

    def test_the_decoupled_ladder_reply_record(self, decoupled_stimulus: DecoupledStimulus) -> None:
        call = decoupled_calls_for_leg(decoupled_leg_for(FLOOR_LEG_ID), decoupled_stimulus)[0]
        row = decoupled_ladder.reply_record(
            call, completion_with_telemetry("r"), decoupled_stimulus
        )
        assert_telemetry(row)

    def test_the_transfer_reply_record(self, transfer_stimulus: TransferStimulus) -> None:
        call = transfer_calls_for_leg(transfer_leg_for(FLOOR_LEG_ID), transfer_stimulus)[0]
        row = transfer_cli.reply_record(call, completion_with_telemetry("r"), transfer_stimulus)
        assert_telemetry(row)


class TestJudgedRowsCarryTelemetryUnderTheJudgePrefix:
    def test_the_analysis_judge_row(self, tmp_path: Path, stimulus: Stimulus) -> None:
        verdict = json.dumps(
            {
                "severity": "none",
                "uncertainty_flagged": False,
                "perspective_adoption": False,
                "rate_vs_narrative": "narrative_only",
                "evidence": "",
            }
        )
        out = tmp_path / "judged.jsonl"
        judge.judge_records(
            FixedCompletionBackend(verdict),
            [{"key": "k1", "reply": "an analysis reply", "cell": "center"}],
            out,
            stimulus,
        )
        assert_telemetry(judge.load_judged(out)["k1"], prefix="judge_")

    def test_the_shared_loop_row_through_the_decoupled_judge(
        self, tmp_path: Path, decoupled_stimulus: DecoupledStimulus
    ) -> None:
        verdict = json.dumps(
            {
                "action_label": decoupled_judge.ACTION_NONE,
                "counterpart_reasoning": "not_discussed",
                "identity_mentioned": False,
                "they_are_me": False,
                "all_instances_policy": False,
                "ev_arithmetic": False,
                "fairness_or_norm": False,
                "evidence": "",
            }
        )
        out = tmp_path / "judged.jsonl"
        decoupled_judge.judge_records(
            FixedCompletionBackend(verdict),
            [
                {
                    "key": "k1",
                    "reply": "a reply",
                    "reasoning": "",
                    "label_a": "FIRE",
                    "label_b": "WATER",
                }
            ],
            out,
            decoupled_stimulus,
        )
        assert_telemetry(decoupled_judge.load_judged(out)["k1"], prefix="judge_")

    @pytest.mark.parametrize("field", sorted(EXPECTED_TELEMETRY))
    def test_every_judge_telemetry_field_is_the_reply_fields_prefixed_twin(
        self, field: str
    ) -> None:
        """The two spellings must stay one spelling apart, or a cost script joins on nothing."""
        judged = judge_telemetry(completion_with_telemetry("r"))
        assert judged[f"judge_{field}"] == EXPECTED_TELEMETRY[field]
