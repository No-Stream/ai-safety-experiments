"""Exercise the rollout row schema's refusals and the extractors that bridge it to the detector.

``RolloutRow.from_dict`` exists to stop an ETL the moment the release's row shape moves; a parser
that had never been handed a moved row is a reassuring message rather than a check. Every refusal
below is fed the exact violation it names. The extractors are fed the tool-call framing copied from
the shard (``<tool_call>`` around a Qwen3 XML ``<function=bash>`` block, ``<tool_response>`` user
turns), including the malformed call shapes the policy was seen to emit.

Offline and CPU-only. The rows here are synthetic: short token lists and a made-up task id, no
prompt text from the release.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from reward_hacking.tmax.rollout_schema import (
    REQUEST_INFO_KEYS,
    ROLLOUT_DATASET_LABEL,
    ROLLOUT_ENV_NAME,
    ROLLOUT_ROW_KEYS,
    ROLLOUT_STATE_KEYS,
    TASK_ID_PATTERN,
    RolloutRow,
    ToolCallStat,
    extract_bash_commands,
    extract_tool_responses,
)

TASK_ID = "task_000001_0badf00d"


def valid_row() -> dict[str, Any]:
    """A row in the verified twelve-key shape: two tool steps, reward at the end, centred advantage."""
    stats = [
        {"tool_name": "bash", "success": True, "runtime": 0.0},
        {"tool_name": "bash", "success": False, "runtime": 1.5},
    ]
    return {
        "step": 90,
        "sample_idx": 17,
        "prompt_idx": 3,
        "prompt_tokens": [248045, 8678, 198],
        "response_tokens": [9764, 728, 22903, 248046],
        "reward": 1.0,
        "advantage": 0.25,
        "finish_reason": "stop",
        "dataset": [ROLLOUT_DATASET_LABEL],
        "ground_truth": [TASK_ID],
        "request_info": {
            "num_calls": 2,
            "timeouts": False,
            "tool_errors": "",
            "tool_outputs": "ok\n",
            "tool_runtimes": 1.5,
            "tool_calleds": True,
            "tool_call_stats": stats,
            "rollout_state": {
                "rewards": [0.0, 1.0],
                "step_count": 2,
                "done": True,
                "info": {"env_name": ROLLOUT_ENV_NAME, "step_count": 2.0},
                "tool_output": "ok\n",
                "tool_error": "",
                "tool_runtime": 1.5,
                "timeout": False,
                "tool_call_stats": stats,
            },
        },
        "logprobs": [-0.1, -0.2, -0.3, -0.05],
    }


class TestAValidRowParses:
    def test_every_field_lands_where_the_schema_says(self) -> None:
        row = RolloutRow.from_dict(valid_row())
        assert (row.step, row.sample_idx, row.prompt_idx) == (90, 17, 3)
        assert row.prompt_tokens == (248045, 8678, 198)
        assert row.response_tokens == (9764, 728, 22903, 248046)
        assert row.logprobs == (-0.1, -0.2, -0.3, -0.05)
        assert (row.reward, row.advantage, row.finish_reason) == (1.0, 0.25, "stop")
        assert row.dataset == (ROLLOUT_DATASET_LABEL,)
        assert row.task_id == TASK_ID
        assert row.env_name == ROLLOUT_ENV_NAME
        assert (row.num_calls, row.timed_out, row.done) == (2, False, True)
        assert row.final_tool_output == "ok\n"
        assert row.per_call_rewards == (0.0, 1.0)
        assert row.tool_call_stats == (
            ToolCallStat("bash", True, 0.0),
            ToolCallStat("bash", False, 1.5),
        )

    def test_the_verified_reward_pattern_is_recognised(self) -> None:
        assert RolloutRow.from_dict(valid_row()).reward_fires_once_at_end

    def test_a_mid_episode_reward_breaks_the_pattern(self) -> None:
        """The property is a check on the data, so it has to be able to say no."""
        row = valid_row()
        row["request_info"]["rollout_state"]["rewards"] = [1.0, 1.0]
        assert not RolloutRow.from_dict(row).reward_fires_once_at_end

    def test_the_key_tuples_are_the_twelve_eight_and_nine_observed(self) -> None:
        assert len(ROLLOUT_ROW_KEYS) == 12
        assert len(REQUEST_INFO_KEYS) == 8
        assert len(ROLLOUT_STATE_KEYS) == 9
        assert set(valid_row()) == set(ROLLOUT_ROW_KEYS)


class TestAMovedSchemaIsRefused:
    """Each refusal is handed the exact violation it exists to catch."""

    def test_a_missing_top_level_key_names_it(self) -> None:
        row = valid_row()
        del row["reward"]
        with pytest.raises(ValueError, match=r"missing=\['reward'\]"):
            RolloutRow.from_dict(row)

    def test_an_unexpected_top_level_key_names_it(self) -> None:
        """A new field means the release changed; parsing around it would hide that."""
        row = valid_row()
        row["hack_label"] = 1
        with pytest.raises(ValueError, match=r"unexpected=\['hack_label'\]"):
            RolloutRow.from_dict(row)

    def test_a_missing_request_info_key_names_the_nesting(self) -> None:
        row = valid_row()
        del row["request_info"]["tool_call_stats"]
        with pytest.raises(ValueError, match=r"request_info keys.*tool_call_stats"):
            RolloutRow.from_dict(row)

    def test_a_missing_rollout_state_key_names_the_nesting(self) -> None:
        row = valid_row()
        del row["request_info"]["rollout_state"]["rewards"]
        with pytest.raises(ValueError, match=r"rollout_state keys.*rewards"):
            RolloutRow.from_dict(row)

    def test_a_reward_typed_as_a_string_is_a_type_error(self) -> None:
        row = valid_row()
        row["reward"] = "1.0"
        with pytest.raises(TypeError, match="reward should be a number"):
            RolloutRow.from_dict(row)

    def test_a_boolean_reward_is_not_a_number(self) -> None:
        """bool is an int subclass; a True/False reward is a schema bug, not 1.0/0.0."""
        row = valid_row()
        row["reward"] = True
        with pytest.raises(TypeError, match="reward should be a number"):
            RolloutRow.from_dict(row)

    def test_a_boolean_token_id_is_not_an_int(self) -> None:
        row = valid_row()
        row["response_tokens"] = [9764, True, 22903, 248046]
        with pytest.raises(TypeError, match="response_tokens should be a list of ints"):
            RolloutRow.from_dict(row)

    def test_misaligned_logprobs_are_refused(self) -> None:
        """One float per response token is the verified alignment; a drift there is a new schema."""
        row = valid_row()
        row["logprobs"] = [-0.1, -0.2]
        with pytest.raises(ValueError, match="2 logprobs for 4 response tokens"):
            RolloutRow.from_dict(row)

    def test_an_empty_ground_truth_is_refused_because_index_zero_is_the_task_id(self) -> None:
        row = valid_row()
        row["ground_truth"] = []
        with pytest.raises(ValueError, match="ground_truth is empty"):
            RolloutRow.from_dict(row)

    def test_a_string_dataset_is_refused_because_the_release_ships_a_list(self) -> None:
        row = valid_row()
        row["dataset"] = ROLLOUT_DATASET_LABEL
        with pytest.raises(TypeError, match="dataset should be a list of strings"):
            RolloutRow.from_dict(row)

    def test_a_tool_call_stat_with_a_changed_shape_is_refused(self) -> None:
        row = valid_row()
        stats = copy.deepcopy(row["request_info"]["tool_call_stats"])
        stats[0]["exit_code"] = 0
        row["request_info"]["tool_call_stats"] = stats
        with pytest.raises(ValueError, match=r"tool_call_stats\[\] keys.*exit_code"):
            RolloutRow.from_dict(row)

    def test_a_non_object_tool_call_stat_is_refused(self) -> None:
        row = valid_row()
        row["request_info"]["tool_call_stats"] = ["bash"]
        with pytest.raises(TypeError, match="entry should be an object"):
            RolloutRow.from_dict(row)

    def test_a_non_object_request_info_is_refused(self) -> None:
        row = valid_row()
        row["request_info"] = "n/a"
        with pytest.raises(TypeError, match="request_info should be an object"):
            RolloutRow.from_dict(row)


class TestTheTaskIdPattern:
    def test_the_release_shape_matches(self) -> None:
        assert TASK_ID_PATTERN.fullmatch(TASK_ID)
        assert TASK_ID_PATTERN.fullmatch("task_003858_854ccdb8")

    def test_other_shapes_do_not(self) -> None:
        for candidate in ("task_38_abc", "task_003858_854CCDB8", "003858_854ccdb8", "task_003858"):
            assert TASK_ID_PATTERN.fullmatch(candidate) is None, candidate


# The exact framing the shard decodes to, around synthetic commands and outputs.
DECODED_RESPONSE = (
    "Let me look.\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\nhead -50 /work/input.log\n</parameter>\n"
    "</function>\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\nline one\nline two\n</tool_response>\n<|im_end|>\n"
    "<|im_start|>assistant\n<think>\nNow run it.\n</think>\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\nprintf 'a\\nb' | wc -l\n</parameter>\n"
    "</function>\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\n1\n</tool_response>\n<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)


class TestExtractingCommandsFromADecodedResponse:
    def test_well_formed_bash_calls_come_back_in_order_without_the_framing(self) -> None:
        extracted = extract_bash_commands(DECODED_RESPONSE)
        assert extracted.commands == ("head -50 /work/input.log", "printf 'a\\nb' | wc -l")
        assert extracted.malformed == ()

    def test_a_multi_line_command_keeps_its_interior_newlines(self) -> None:
        decoded = (
            "<tool_call>\n<function=bash>\n<parameter=command>\n"
            "cd /work\nmake test\n</parameter>\n</function>\n</tool_call>"
        )
        assert extract_bash_commands(decoded).commands == ("cd /work\nmake test",)

    def test_malformed_calls_are_kept_apart_rather_than_dropped(self) -> None:
        """The shard had non-bash function names, two-parameter calls and empty ones."""
        decoded = (
            DECODED_RESPONSE
            + "<tool_call>\n<function=C>\n<parameter=features>\nx\n</parameter>\n</function>\n"
            "</tool_call>" + "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
            "<parameter=function>\ny\n</parameter>\n</function>\n</tool_call>"
        )
        extracted = extract_bash_commands(decoded)
        assert extracted.commands == ("head -50 /work/input.log", "printf 'a\\nb' | wc -l")
        assert len(extracted.malformed) == 2
        assert extracted.malformed[0].startswith("<function=C>")

    def test_a_response_with_no_calls_is_empty_on_both_sides(self) -> None:
        extracted = extract_bash_commands("I will not run anything.<|im_end|>")
        assert extracted.commands == ()
        assert extracted.malformed == ()

    def test_tool_responses_come_back_in_order(self) -> None:
        assert extract_tool_responses(DECODED_RESPONSE) == ("line one\nline two", "1")

    def test_a_tool_response_is_not_mistaken_for_a_call(self) -> None:
        """Terminal output that quotes the call framing must not be read as an action."""
        decoded = (
            "<tool_call>\n<function=bash>\n<parameter=command>\ncat notes\n</parameter>\n"
            "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n"
            "the file says: run <function=bash> later\n</tool_response>\n<|im_end|>\n"
        )
        assert extract_bash_commands(decoded).commands == ("cat notes",)
        assert extract_tool_responses(decoded) == ("the file says: run <function=bash> later",)


class TestAnEnvironmentResetFailure:
    """The trainer records a zero-reward row when the environment never started; it is not a rollout."""

    @staticmethod
    def failed_row() -> dict[str, Any]:
        row = valid_row()
        row["response_tokens"] = [248046]
        row["logprobs"] = [0.0]
        row["reward"] = 0.0
        stats = [{"tool_name": "env_reset", "success": False, "runtime": 0.0}]
        row["request_info"]["num_calls"] = 0
        row["request_info"]["tool_errors"] = (
            "Environment reset failed; marking rollout as zero reward: ray::EnvironmentPool"
        )
        row["request_info"]["tool_call_stats"] = stats
        row["request_info"]["rollout_state"]["tool_call_stats"] = stats
        row["request_info"]["rollout_state"]["rewards"] = [0.0]
        row["request_info"]["rollout_state"]["step_count"] = 0
        row["request_info"]["rollout_state"]["info"] = {}
        return row

    def test_it_parses_with_no_environment_name_and_is_flagged(self) -> None:
        row = RolloutRow.from_dict(self.failed_row())
        assert row.env_name is None
        assert row.env_reset_failed is True
        assert row.num_calls == 0

    def test_a_started_environment_is_not_flagged(self) -> None:
        assert RolloutRow.from_dict(valid_row()).env_reset_failed is False

    def test_the_tool_error_prefix_alone_is_enough(self) -> None:
        row = self.failed_row()
        row["request_info"]["tool_call_stats"] = []
        row["request_info"]["rollout_state"]["tool_call_stats"] = []
        assert RolloutRow.from_dict(row).env_reset_failed is True
