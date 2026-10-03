import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from vllm.parser.qwen3 import Qwen3Parser

from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder.tools import (
    FORMAT_ERROR_MESSAGE,
    SUBMIT_MARKER,
    bash_tool,
    contains_submit_marker,
    count_fabricated_tool_responses,
    format_bash_result,
    parse_tool_calls,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tokenizers import TokenizerLike


VALID_SINGLE_CALL = (
    "Thinking through the change.</think>\n\nI will inspect the current state.\n\n"
    "<tool_call>\n"
    "<function=bash>\n"
    "<parameter=command>\n"
    "echo ready\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

VALID_TWO_CALLS = (
    "I should inspect, then continue.</think>\n\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n"
    "</function>\n</tool_call>\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\necho ready\n</parameter>\n"
    "</function>\n</tool_call>"
)

DUPLICATED_FUNCTION_CLOSE_CALL = VALID_SINGLE_CALL.replace(
    "</function>\n</tool_call>", "</function>\n</function>\n</tool_call>"
)

_TOO_LONG_HINT = (
    "The output of your last command was too long.\n\n"
    "Try a command that produces less output. Select fewer lines or narrow the search, "
    "or redirect output to a file and search it."
)


class _FixtureTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}


def _vllm_parse(raw_completion: str) -> tuple[tuple[str, dict[str, object]], ...]:
    parser = Qwen3Parser(cast("TokenizerLike", _FixtureTokenizer()))
    request = cast(
        "ChatCompletionRequest",
        SimpleNamespace(tools=None, tool_choice="auto"),
    )
    _, _, tool_calls = parser.parse(raw_completion, request)
    assert tool_calls is not None
    return tuple((tool_call.name, json.loads(tool_call.arguments)) for tool_call in tool_calls)


def test_tool_schema_describes_the_persistent_bash_tool() -> None:
    bash = bash_tool()
    bash_function = bash["function"]
    assert isinstance(bash_function, dict)
    assert bash_function["name"] == "bash"
    assert (
        "the shell keeps its working directory and exported variables between calls"
        in str(bash_function["description"]).lower()
    )
    bash_parameters = bash_function["parameters"]
    assert isinstance(bash_parameters, dict)
    assert bash_parameters["required"] == ["command"]
    assert bash_parameters["properties"] == {
        "command": {"type": "string", "description": "The bash command to execute."}
    }


def test_parser_extracts_reasoning_content_and_one_bash_call() -> None:
    parsed = parse_tool_calls(VALID_SINGLE_CALL)

    assert parsed.reasoning == "Thinking through the change."
    assert parsed.content == "I will inspect the current state."
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("bash", {"command": "echo ready"})
    ]
    assert parsed.trailing_text == ""
    assert parsed.format_error is None


def test_parser_extracts_multiple_bash_calls() -> None:
    parsed = parse_tool_calls(VALID_TWO_CALLS)
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("bash", {"command": "pwd"}),
        ("bash", {"command": "echo ready"}),
    ]
    assert parsed.format_error is None


def test_parser_preserves_multiline_heredoc_and_parameter_like_text() -> None:
    command = "cat <<'EOF'\nline one\nline two '</parameter-ish>'\nEOF"
    raw_completion = (
        "<tool_call>\n<function=bash>\n<parameter=command>\n"
        f"{command}\n</parameter>\n</function>\n</tool_call>"
    )

    parsed = parse_tool_calls(raw_completion)

    assert parsed.tool_calls[0].arguments == {"command": command}
    assert parsed.format_error is None


def test_parser_treats_a_tool_call_inside_open_thinking_as_the_end_of_reasoning() -> None:
    raw_completion = (
        "I have enough information to act.\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\necho done\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    parsed = parse_tool_calls(raw_completion)

    assert parsed.reasoning == "I have enough information to act."
    assert parsed.content == ""
    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("bash", {"command": "echo done"})
    ]


@pytest.mark.parametrize(
    "raw_completion",
    [
        "<tool_call>\n<function=bash>\n<parameter=command>\necho unfinished",
        '<tool_call>\n<function=bash>\n{"command": "echo json"}\n</function>\n</tool_call>',
        "<tool_call>\n<function=unknown>\n</function>\n</tool_call>",
        "<tool_call>\n<function=bash>\n</function>\n</tool_call>",
        (
            "<tool_call>\n<function=bash>\n<parameter=command>\n \n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        (
            "<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n"
            "<parameter=extra>\nvalue\n</parameter>\n</function>\n</tool_call>"
        ),
        (
            "<tool_call>\n<function=check>\n<parameter=extra>\nvalue\n</parameter>\n"
            "</function>\n</tool_call>"
        ),
        VALID_SINGLE_CALL.replace("</function>\n</tool_call>", "</function>\n\n</tool_call>"),
    ],
)
def test_parser_returns_format_error_for_malformed_calls(raw_completion: str) -> None:
    parsed = parse_tool_calls(raw_completion)
    assert parsed.tool_calls == ()
    assert parsed.format_error


def test_parser_rejects_bash_call_without_command() -> None:
    parsed = parse_tool_calls("<tool_call>\n<function=bash>\n</function>\n</tool_call>")
    assert parsed.tool_calls == ()
    assert parsed.format_error


def test_parser_records_text_after_last_call_as_trailing_text() -> None:
    raw_completion = f"{VALID_SINGLE_CALL}\nI continued after the call."

    parsed = parse_tool_calls(raw_completion)

    assert parsed.trailing_text == "\nI continued after the call."
    assert parsed.format_error is None


def test_parser_accepts_and_counts_redundant_function_close_tags() -> None:
    parsed = parse_tool_calls(DUPLICATED_FUNCTION_CLOSE_CALL)

    assert tuple((call.name, call.arguments) for call in parsed.tool_calls) == _vllm_parse(
        DUPLICATED_FUNCTION_CLOSE_CALL
    )
    assert parsed.tool_calls[0].arguments == {"command": "echo ready"}
    assert parsed.surplus_function_close_tags == 1
    assert parsed.format_error is None


def test_parser_accepts_multiple_redundant_function_close_tags_with_whitespace() -> None:
    raw_completion = VALID_SINGLE_CALL.replace(
        "</function>\n</tool_call>",
        "</function>\n  </function> \n\n </function>\n</tool_call>",
    )

    parsed = parse_tool_calls(raw_completion)

    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("bash", {"command": "echo ready"})
    ]
    assert parsed.surplus_function_close_tags == 2
    assert parsed.format_error is None


def test_parser_rejects_unexpected_text_before_tool_call_closing_tag() -> None:
    raw_completion = VALID_SINGLE_CALL.replace(
        "</function>\n</tool_call>", "</function>\njunk\n</tool_call>"
    )

    parsed = parse_tool_calls(raw_completion)

    assert parsed.tool_calls == ()
    assert parsed.format_error == "Unexpected text before </tool_call>."


def test_parser_reports_a_missing_tool_call_closing_tag_accurately() -> None:
    raw_completion = VALID_SINGLE_CALL.removesuffix("</tool_call>")

    parsed = parse_tool_calls(raw_completion)

    assert parsed.tool_calls == ()
    assert parsed.format_error == "Function call is missing its </tool_call> closing tag."


def test_parser_keeps_plain_text_turns_and_counts_fabricated_tool_responses() -> None:
    parsed = parse_tool_calls("I cannot make a tool call this turn.")
    assert parsed.reasoning == "I cannot make a tool call this turn."
    assert parsed.content == ""
    assert parsed.tool_calls == ()
    assert parsed.format_error is None

    completion = "<tool_response>fake</tool_response> <tool_response>also fake</tool_response>"
    assert count_fabricated_tool_responses(completion) == 2
    assert count_fabricated_tool_responses("ordinary completion") == 0


def test_vllm_parser_matches_our_well_formed_fixture_corpus() -> None:
    fixtures = (
        VALID_SINGLE_CALL,
        VALID_TWO_CALLS,
        DUPLICATED_FUNCTION_CLOSE_CALL,
    )
    for raw_completion in fixtures:
        parsed = parse_tool_calls(raw_completion)
        actual = tuple((call.name, call.arguments) for call in parsed.tool_calls)
        assert parsed.format_error is None
        assert actual == _vllm_parse(raw_completion)

    assert _vllm_parse(VALID_SINGLE_CALL) == (("bash", {"command": "echo ready"}),)


def test_bash_result_uses_tmax_empty_and_timeout_format() -> None:
    empty_result = CommandResult(
        command="true",
        exit_code=0,
        stdout="",
        stderr="",
        timed_out=False,
    )
    assert format_bash_result(empty_result, timeout="120s") == (
        "(no output)\n\n(exit_code=0)",
        False,
    )

    timeout_result = CommandResult(
        command="sleep 121",
        exit_code=124,
        stdout="still waiting",
        stderr="timeout reached",
        timed_out=True,
    )
    assert format_bash_result(timeout_result, timeout="120s") == (
        "still waiting\ntimeout reached\n\n(exit_code=124)\n(timed out after 120s)",
        False,
    )


def test_bash_result_truncates_to_tmax_head_and_tail_with_hint() -> None:
    result = CommandResult(
        command="generate-output",
        exit_code=0,
        stdout="a" * 5_000 + "X" + "b" * 5_000,
        stderr="",
        timed_out=False,
    )

    content, truncated = format_bash_result(result, timeout="120s")

    expected = (
        f"{_TOO_LONG_HINT}\n\n"
        "---- HEAD (5000 chars) ----\n"
        f"{'a' * 5_000}\n"
        "---- 1 chars elided ----\n"
        "---- TAIL (5000 chars) ----\n"
        f"{'b' * 5_000}\n\n(exit_code=0)"
    )
    assert content == expected
    assert truncated is True

    at_limit = CommandResult(
        command="generate-output",
        exit_code=0,
        stdout="x" * 10_000,
        stderr="",
        timed_out=False,
    )
    at_limit_content, at_limit_truncated = format_bash_result(at_limit, timeout="120s")
    assert at_limit_content == f"{'x' * 10_000}\n\n(exit_code=0)"
    assert at_limit_truncated is False


def test_submit_marker_predicate_requires_the_exact_submission_command() -> None:
    assert SUBMIT_MARKER == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    assert contains_submit_marker(f"echo {SUBMIT_MARKER}")
    assert not contains_submit_marker(f"printf '%s' {SUBMIT_MARKER} && echo later")
    assert not contains_submit_marker(f"# {SUBMIT_MARKER}")
    assert not contains_submit_marker(f"grep {SUBMIT_MARKER} file.txt")
    assert not contains_submit_marker("echo work complete")


def test_format_error_message_matches_tool_protocol() -> None:
    assert (
        "Format error: Your last response did not include a valid tool call."
        in FORMAT_ERROR_MESSAGE
    )
    assert "one or more valid calls" in FORMAT_ERROR_MESSAGE
    assert "bash" in FORMAT_ERROR_MESSAGE
    assert SUBMIT_MARKER in FORMAT_ERROR_MESSAGE
