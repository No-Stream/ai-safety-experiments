"""Summarize sampling health from organism episode records and exact-token traces.

Repetition fractions use reasoning sentences with at least 25 characters. A turn is
classified as a loop when strictly more than 25% of those sentences repeat an earlier
sentence in the same turn.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING, cast

from reward_hacking.ladder.tools import contains_non_registered_submit_marker

if TYPE_CHECKING:
    from collections.abc import Sequence

logger: logging.Logger = logging.getLogger(__name__)

ENDING_CATEGORIES: tuple[str, ...] = (
    "submitted",
    "submission_limit",
    "turn_cap",
    "no_tool_call",
    "other",
)
_UNEXECUTED_TOOL_REASONS = frozenset(
    {"deadline", "format_error", "submit", "thinking_budget_overshoot"}
)
_COMPARISON_GROUP_COUNT = 2
_MIN_REPETITION_SENTENCE_CHARACTERS = 25
_LOOP_TURN_REPEAT_FRACTION_THRESHOLD = 0.25

_CJK_IDEOGRAPH = re.compile("[\\u3400-\\u4dbf\\u4e00-\\u9fff]")
_REASONING_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_SHELL_ERROR_PREFIX = re.compile(r"(?i)^\s*(?:\S*/)?(?:bash|sh|dash|zsh)(?::|\s)")
_SHELL_PARSE_ERROR = re.compile(
    r"(?i)\b(?:syntax error|parse error|unexpected EOF while looking for matching|"
    r"unexpected end of file|here-document at line \d+ delimited by end-of-file)\b"
)
_PYTHON_PARSE_ERROR = re.compile(r"(?im)^\s*(?:SyntaxError|IndentationError|TabError):")
_PYTHON_SOURCE_LOCATION = re.compile(r"(?m)^\s*File [\"'][^\"'\n]+[\"'], line \d+")
_PYTHON_COMMAND = re.compile(r"(?:^|[\s;&|])(?:[^\s;&|]+/)?python(?:3(?:\.\d+)?)?(?=\s|$)")


class HealthInputError(ValueError):
    """Raised when an input record or trace lacks required health evidence."""


@dataclass(frozen=True)
class EpisodeHealth:
    """Per-episode sampling, repetition, execution, language, and ending metrics."""

    episode_index: int
    assistant_turn_count: int
    sampled_token_count: int
    sampled_token_negative_logprob_mean: float | None
    thinking_token_count: int
    thinking_negative_logprob_mean: float | None
    action_token_count: int
    action_negative_logprob_mean: float | None
    channel_split_unavailable_turn_count: int
    command_call_count: int
    syntax_error_command_call_count: int
    syntax_error_command_call_fraction: float | None
    unexecuted_tool_call_count: int
    unexecuted_tool_call_reasons: dict[str, int]
    non_registered_submit_attempt_count: int
    cjk_assistant_turn_count: int
    cjk_assistant_turn_fraction: float | None
    contains_cjk: bool
    max_new_tokens_turn_count: int
    max_new_tokens_turn_fraction: float | None
    reasoning_sentence_repeat_fractions: list[float | None]
    loop_turn_count: int
    loop_turn_fraction: float | None
    has_repetition_loop: bool
    copied_turn_count: int
    ending: str
    peak_prompt_tokens: int


@dataclass(frozen=True)
class _TurnHealth:
    completion_token_ids: tuple[int, ...]
    sampled_token_nlls: list[float]
    thinking_token_nlls: list[float]
    action_token_nlls: list[float]
    prompt_tokens: int
    channel_split_unavailable: bool
    command_call_count: int
    syntax_error_command_call_count: int
    unexecuted_tool_call_count: int
    unexecuted_tool_call_reasons: dict[str, int]
    non_registered_submit_attempt_count: int
    contains_cjk: bool
    stopped_at_token_cap: bool
    reasoning_sentence_repeat_fraction: float | None


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise HealthInputError(f"{context} must be a JSON object")
    return cast("dict[str, object]", value)


def _required(mapping: dict[str, object], key: str, context: str) -> object:
    if key not in mapping:
        raise HealthInputError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _string(mapping: dict[str, object], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str):
        raise HealthInputError(f"{context}.{key} must be a string")
    return value


def _integer(mapping: dict[str, object], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if type(value) is not int:
        raise HealthInputError(f"{context}.{key} must be an integer")
    return value


def _list(mapping: dict[str, object], key: str, context: str) -> list[object]:
    value = _required(mapping, key, context)
    if not isinstance(value, list):
        raise HealthInputError(f"{context}.{key} must be a list")
    return cast("list[object]", value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as records_file:
            for line_number, line in enumerate(records_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise HealthInputError(
                        f"{path}:{line_number} is not valid JSON: {error.msg}"
                    ) from error
                rows.append(_object(row, f"{path}:{line_number}"))
    except UnicodeDecodeError as error:
        raise HealthInputError(f"{path} is not valid UTF-8 JSONL") from error
    if not rows:
        raise HealthInputError(f"{path} contains no episode records")
    return rows


def _read_trace(path: Path) -> list[dict[str, object]]:
    return _read_jsonl(path)


def _token_ids(turn: dict[str, object], context: str) -> list[int]:
    raw_ids = _list(turn, "completion_token_ids", context)
    token_ids: list[int] = []
    for token_index, raw_token_id in enumerate(raw_ids):
        if type(raw_token_id) is not int:
            raise HealthInputError(
                f"{context}.completion_token_ids[{token_index}] must be an integer"
            )
        token_ids.append(raw_token_id)
    return token_ids


def _logprobs(turn: dict[str, object], context: str) -> list[float]:
    raw_logprobs = _list(turn, "completion_logprobs", context)
    logprobs: list[float] = []
    for token_index, raw_logprob in enumerate(raw_logprobs):
        if isinstance(raw_logprob, bool) or not isinstance(raw_logprob, int | float):
            raise HealthInputError(f"{context}.completion_logprobs[{token_index}] must be numeric")
        logprob = float(raw_logprob)
        if not math.isfinite(logprob):
            raise HealthInputError(f"{context}.completion_logprobs[{token_index}] must be finite")
        logprobs.append(logprob)
    return logprobs


def is_syntax_parse_error(command: str, output: str) -> bool:
    """Match shell-prefixed parse diagnostics or Python source-line syntax exceptions.

    Shell matches require a bash, sh, dash, or zsh diagnostic prefix and parser wording
    such as ``syntax error``, ``parse error``, ``unexpected EOF``, or an unclosed
    here-document. Python matches require a Python command, a reported source file and
    line, and ``SyntaxError``, ``IndentationError``, or ``TabError``. A nonzero exit,
    runtime exception, or generic error message does not count.
    """
    if any(
        _SHELL_ERROR_PREFIX.search(line) and _SHELL_PARSE_ERROR.search(line)
        for line in output.splitlines()
    ):
        return True
    return bool(
        _PYTHON_COMMAND.search(command)
        and _PYTHON_SOURCE_LOCATION.search(output)
        and _PYTHON_PARSE_ERROR.search(output)
    )


def _ending_category(value: str) -> str:
    if value == "submit":
        return "submitted"
    if value in ENDING_CATEGORIES:
        return value
    return "other"


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _reasoning_sentence_repeat_fraction(reasoning: str) -> float | None:
    """Return the share of eligible reasoning sentences repeated earlier in this turn."""
    sentences = [
        sentence.strip()
        for sentence in _REASONING_SENTENCE_BOUNDARY.split(reasoning)
        if len(sentence.strip()) >= _MIN_REPETITION_SENTENCE_CHARACTERS
    ]
    if not sentences:
        return None

    seen_sentences: set[str] = set()
    repeated_sentence_count = 0
    for sentence in sentences:
        if sentence in seen_sentences:
            repeated_sentence_count += 1
        else:
            seen_sentences.add(sentence)
    return repeated_sentence_count / len(sentences)


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _trace_turns_and_ending(trace_path: Path) -> tuple[list[dict[str, object]], str]:
    events = _read_trace(trace_path)
    assistant_turns: list[dict[str, object]] = []
    end_events: list[dict[str, object]] = []
    for event_index, event in enumerate(events, start=1):
        event_context = f"{trace_path}:event {event_index}"
        if "turn" in event:
            if "completion_logprobs" not in event:
                raise HealthInputError(
                    f"{event_context} is missing required field 'completion_logprobs'"
                )
            assistant_turns.append(event)
        if "ended_by" in event:
            end_events.append(event)

    if len(end_events) != 1 or events[-1] is not end_events[0]:
        raise HealthInputError(
            f"{trace_path} must end with exactly one episode summary containing ended_by"
        )
    end_event = end_events[0]
    ending_value = _string(end_event, "ended_by", f"{trace_path}:episode summary")
    declared_turn_count = _integer(end_event, "turns", f"{trace_path}:episode summary")
    if declared_turn_count != len(assistant_turns):
        raise HealthInputError(
            f"{trace_path} episode summary reports {declared_turn_count} turns, "
            f"but the trace contains {len(assistant_turns)} assistant turns"
        )
    return assistant_turns, _ending_category(ending_value)


def _analyze_turn(
    turn: dict[str, object],
    trace_path: Path,
    turn_index: int,
) -> _TurnHealth:
    turn_context = f"{trace_path}:turn {turn_index}"
    token_ids = _token_ids(turn, turn_context)
    logprobs = _logprobs(turn, turn_context)
    if len(token_ids) != len(logprobs):
        raise HealthInputError(
            f"{turn_context} has {len(token_ids)} sampled token IDs but "
            f"{len(logprobs)} sampling logprobs"
        )

    # Only aligned sampled IDs and logprobs enter NLL; forced closes without sampled logprobs are excluded.
    sampled_token_nlls = [-logprob for logprob in logprobs]
    reasoning_token_count = _integer(turn, "reasoning_token_count", turn_context)
    if reasoning_token_count < 0:
        raise HealthInputError(f"{turn_context}.reasoning_token_count cannot be negative")
    channel_split_unavailable = reasoning_token_count > len(logprobs)
    thinking_token_nlls = (
        [-logprob for logprob in logprobs[:reasoning_token_count]]
        if not channel_split_unavailable
        else []
    )
    action_token_nlls = (
        [-logprob for logprob in logprobs[reasoning_token_count:]]
        if not channel_split_unavailable
        else []
    )

    reasoning = _string(turn, "reasoning", turn_context)
    content = _string(turn, "content", turn_context)
    contains_cjk = bool(_CJK_IDEOGRAPH.search(reasoning + content))
    stopped_at_token_cap = _string(turn, "stop_reason", turn_context) == "max_tokens"
    prompt_tokens = _integer(turn, "prompt_tokens", turn_context)
    (
        command_call_count,
        syntax_error_command_call_count,
        unexecuted_tool_call_count,
        unexecuted_tool_call_reasons,
        non_registered_submit_attempt_count,
    ) = _analyze_command_calls(turn, turn_context)

    return _TurnHealth(
        completion_token_ids=tuple(token_ids),
        sampled_token_nlls=sampled_token_nlls,
        thinking_token_nlls=thinking_token_nlls,
        action_token_nlls=action_token_nlls,
        prompt_tokens=prompt_tokens,
        channel_split_unavailable=channel_split_unavailable,
        command_call_count=command_call_count,
        syntax_error_command_call_count=syntax_error_command_call_count,
        unexecuted_tool_call_count=unexecuted_tool_call_count,
        unexecuted_tool_call_reasons=unexecuted_tool_call_reasons,
        non_registered_submit_attempt_count=non_registered_submit_attempt_count,
        contains_cjk=contains_cjk,
        stopped_at_token_cap=stopped_at_token_cap,
        reasoning_sentence_repeat_fraction=_reasoning_sentence_repeat_fraction(reasoning),
    )


def _tool_call_execution(call_record: dict[str, object], context: str) -> tuple[bool, str | None]:
    executed = _required(call_record, "executed", context)
    if type(executed) is not bool:
        raise HealthInputError(f"{context}.executed must be a boolean")
    not_executed_reason = _required(call_record, "not_executed_reason", context)
    if executed:
        if not_executed_reason is not None:
            raise HealthInputError(f"{context}.not_executed_reason must be null when executed")
        return True, None
    if not isinstance(not_executed_reason, str) or not not_executed_reason:
        raise HealthInputError(f"{context}.not_executed_reason must name the reason")
    if not_executed_reason not in _UNEXECUTED_TOOL_REASONS:
        raise HealthInputError(
            f"{context} has unknown unexecuted-tool reason {not_executed_reason!r}"
        )
    return False, not_executed_reason


def _validate_unexecuted_tool_reasons(
    reasons: Counter[str], executed_tool_call_count: int, context: str
) -> None:
    reason_names = set(reasons)
    if len(reason_names) > 1:
        raise HealthInputError(f"{context} has multiple reasons for unexecuted tool calls")
    if reason_names & {"format_error", "thinking_budget_overshoot"} and executed_tool_call_count:
        raise HealthInputError(f"{context} executed a tool call despite a pre-execution failure")
    if "submit" in reason_names and executed_tool_call_count == 0:
        raise HealthInputError(f"{context} has unexecuted calls after submit but no executed call")


def _analyze_command_calls(
    turn: dict[str, object], context: str
) -> tuple[int, int, int, dict[str, int], int]:
    tool_calls = _list(turn, "tool_calls", context)
    commands = _list(turn, "commands", context)
    unexecuted_tool_call_reasons: Counter[str] = Counter()
    executed_tool_call_count = 0
    seen_unexecuted_tool_call = False
    for call_index, raw_call in enumerate(tool_calls):
        call_context = f"{context}.tool_calls[{call_index}]"
        call_record = _object(raw_call, call_context)
        executed, not_executed_reason = _tool_call_execution(call_record, call_context)
        if executed:
            if seen_unexecuted_tool_call:
                raise HealthInputError(
                    f"{context} has an executed tool call after an unexecuted call"
                )
            executed_tool_call_count += 1
            continue

        seen_unexecuted_tool_call = True
        if not_executed_reason is None:
            raise HealthInputError(f"{call_context}.not_executed_reason cannot be null")
        unexecuted_tool_call_reasons[not_executed_reason] += 1

    _validate_unexecuted_tool_reasons(
        unexecuted_tool_call_reasons, executed_tool_call_count, context
    )

    if executed_tool_call_count != len(commands):
        raise HealthInputError(
            f"{context} has {executed_tool_call_count} executed tool calls but "
            f"{len(commands)} command outputs"
        )

    non_registered_submit_attempt_count = 0
    for call_index, raw_call in enumerate(tool_calls):
        call_context = f"{context}.tool_calls[{call_index}]"
        call_record = _object(raw_call, call_context)
        arguments = _object(
            _required(call_record, "arguments", call_context), f"{call_context}.arguments"
        )
        command = _string(arguments, "command", f"{call_context}.arguments")
        non_registered_submit_attempt_count += int(contains_non_registered_submit_marker(command))

    syntax_error_count = 0
    for command_index, raw_command in enumerate(commands):
        command_context = f"{context}.commands[{command_index}]"
        command_record = _object(raw_command, command_context)
        command = _string(command_record, "command", command_context)
        stdout = _string(command_record, "stdout", command_context)
        stderr = _string(command_record, "stderr", command_context)
        syntax_error_count += int(is_syntax_parse_error(command, f"{stdout}\n{stderr}"))
    return (
        len(commands),
        syntax_error_count,
        sum(unexecuted_tool_call_reasons.values()),
        dict(sorted(unexecuted_tool_call_reasons.items())),
        non_registered_submit_attempt_count,
    )


def _analyze_episode(
    row: dict[str, object],
    records_path: Path,
    episode_index: int,
) -> EpisodeHealth:
    context = f"{records_path}:episode {episode_index}"
    raw_trace_path = _string(row, "trace_path", context)
    trace_path = Path(raw_trace_path)
    if not trace_path.is_absolute():
        trace_path = records_path.parent / trace_path
    assistant_turns, ending = _trace_turns_and_ending(trace_path)
    turn_health = [
        _analyze_turn(turn, trace_path, turn_index)
        for turn_index, turn in enumerate(assistant_turns, start=1)
    ]
    sampled_token_nlls = [nll for turn in turn_health for nll in turn.sampled_token_nlls]
    thinking_token_nlls = [nll for turn in turn_health for nll in turn.thinking_token_nlls]
    action_token_nlls = [nll for turn in turn_health for nll in turn.action_token_nlls]
    command_call_count = sum(turn.command_call_count for turn in turn_health)
    syntax_error_call_count = sum(turn.syntax_error_command_call_count for turn in turn_health)
    unexecuted_tool_call_count = sum(turn.unexecuted_tool_call_count for turn in turn_health)
    non_registered_submit_attempt_count = sum(
        turn.non_registered_submit_attempt_count for turn in turn_health
    )
    unexecuted_tool_call_reasons: Counter[str] = Counter()
    for turn in turn_health:
        unexecuted_tool_call_reasons.update(turn.unexecuted_tool_call_reasons)
    cjk_turn_count = sum(turn.contains_cjk for turn in turn_health)
    max_new_tokens_turn_count = sum(turn.stopped_at_token_cap for turn in turn_health)
    unavailable_channel_splits = sum(turn.channel_split_unavailable for turn in turn_health)
    reasoning_sentence_repeat_fractions = [
        turn.reasoning_sentence_repeat_fraction for turn in turn_health
    ]
    loop_turn_count = sum(
        fraction is not None and fraction > _LOOP_TURN_REPEAT_FRACTION_THRESHOLD
        for fraction in reasoning_sentence_repeat_fractions
    )
    prior_completion_token_ids: set[tuple[int, ...]] = set()
    copied_turn_count = 0
    for turn in turn_health:
        completion_token_ids = turn.completion_token_ids
        if completion_token_ids and completion_token_ids in prior_completion_token_ids:
            copied_turn_count += 1
        prior_completion_token_ids.add(completion_token_ids)
    peak_prompt_tokens = _integer(row, "peak_prompt_tokens", context)
    if peak_prompt_tokens < 0:
        raise HealthInputError(f"{context}.peak_prompt_tokens cannot be negative")
    trace_peak_prompt_tokens = max((turn.prompt_tokens for turn in turn_health), default=0)
    if peak_prompt_tokens < trace_peak_prompt_tokens:
        raise HealthInputError(
            f"{context}.peak_prompt_tokens is lower than the trace peak "
            f"of {trace_peak_prompt_tokens} tokens"
        )
    assistant_turn_count = len(assistant_turns)
    contains_cjk = cjk_turn_count > 0
    return EpisodeHealth(
        episode_index=episode_index,
        assistant_turn_count=assistant_turn_count,
        sampled_token_count=len(sampled_token_nlls),
        sampled_token_negative_logprob_mean=_mean(sampled_token_nlls),
        thinking_token_count=len(thinking_token_nlls),
        thinking_negative_logprob_mean=_mean(thinking_token_nlls),
        action_token_count=len(action_token_nlls),
        action_negative_logprob_mean=_mean(action_token_nlls),
        channel_split_unavailable_turn_count=unavailable_channel_splits,
        command_call_count=command_call_count,
        syntax_error_command_call_count=syntax_error_call_count,
        syntax_error_command_call_fraction=_fraction(syntax_error_call_count, command_call_count),
        unexecuted_tool_call_count=unexecuted_tool_call_count,
        unexecuted_tool_call_reasons=dict(sorted(unexecuted_tool_call_reasons.items())),
        non_registered_submit_attempt_count=non_registered_submit_attempt_count,
        cjk_assistant_turn_count=cjk_turn_count,
        cjk_assistant_turn_fraction=_fraction(cjk_turn_count, assistant_turn_count),
        contains_cjk=contains_cjk,
        max_new_tokens_turn_count=max_new_tokens_turn_count,
        max_new_tokens_turn_fraction=_fraction(max_new_tokens_turn_count, assistant_turn_count),
        reasoning_sentence_repeat_fractions=reasoning_sentence_repeat_fractions,
        loop_turn_count=loop_turn_count,
        loop_turn_fraction=_fraction(loop_turn_count, assistant_turn_count),
        has_repetition_loop=loop_turn_count > 0,
        copied_turn_count=copied_turn_count,
        ending=ending,
        peak_prompt_tokens=peak_prompt_tokens,
    )


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {"mean": _mean(values), "median": _median(values)}


def _fraction_summary(
    episode_fractions: Sequence[float],
    numerator: int,
    denominator: int,
) -> dict[str, float | int | None]:
    return {
        "mean": _mean(episode_fractions),
        "median": _median(episode_fractions),
        "fraction": _fraction(numerator, denominator),
        "numerator": numerator,
        "denominator": denominator,
    }


def _summary(episodes: Sequence[EpisodeHealth]) -> dict[str, object]:
    sampled = [episode.sampled_token_negative_logprob_mean for episode in episodes]
    sampled = [value for value in sampled if value is not None]
    sampled_token_count = sum(episode.sampled_token_count for episode in episodes)
    pooled_sampled_nll = (
        sum(
            episode.sampled_token_negative_logprob_mean * episode.sampled_token_count
            for episode in episodes
            if episode.sampled_token_negative_logprob_mean is not None
        )
        / sampled_token_count
        if sampled_token_count
        else None
    )

    thinking = [episode.thinking_negative_logprob_mean for episode in episodes]
    thinking = [value for value in thinking if value is not None]
    thinking_token_count = sum(episode.thinking_token_count for episode in episodes)
    pooled_thinking_nll = (
        sum(
            episode.thinking_negative_logprob_mean * episode.thinking_token_count
            for episode in episodes
            if episode.thinking_negative_logprob_mean is not None
        )
        / thinking_token_count
        if thinking_token_count
        else None
    )

    action = [episode.action_negative_logprob_mean for episode in episodes]
    action = [value for value in action if value is not None]
    action_token_count = sum(episode.action_token_count for episode in episodes)
    pooled_action_nll = (
        sum(
            episode.action_negative_logprob_mean * episode.action_token_count
            for episode in episodes
            if episode.action_negative_logprob_mean is not None
        )
        / action_token_count
        if action_token_count
        else None
    )

    syntax_error_call_count = sum(episode.syntax_error_command_call_count for episode in episodes)
    command_call_count = sum(episode.command_call_count for episode in episodes)
    cjk_turn_count = sum(episode.cjk_assistant_turn_count for episode in episodes)
    assistant_turn_count = sum(episode.assistant_turn_count for episode in episodes)
    cjk_episodes = sum(episode.contains_cjk for episode in episodes)
    max_new_tokens_turn_count = sum(episode.max_new_tokens_turn_count for episode in episodes)
    loop_turn_count = sum(episode.loop_turn_count for episode in episodes)
    copied_turn_count = sum(episode.copied_turn_count for episode in episodes)
    episodes_with_repetition_loop_count = sum(episode.has_repetition_loop for episode in episodes)
    unexecuted_tool_call_count = sum(episode.unexecuted_tool_call_count for episode in episodes)
    non_registered_submit_attempt_count = sum(
        episode.non_registered_submit_attempt_count for episode in episodes
    )
    unexecuted_tool_call_reasons: Counter[str] = Counter()
    for episode in episodes:
        unexecuted_tool_call_reasons.update(episode.unexecuted_tool_call_reasons)
    ending_counts = Counter(episode.ending for episode in episodes)

    return {
        "sampling_nll": {
            **_numeric_summary(sampled),
            "token_weighted_mean": pooled_sampled_nll,
            "episode_count": len(sampled),
            "token_count": sampled_token_count,
        },
        "thinking_nll": {
            **_numeric_summary(thinking),
            "token_weighted_mean": pooled_thinking_nll,
            "episode_count": len(thinking),
            "token_count": thinking_token_count,
            "unavailable_turn_count": sum(
                episode.channel_split_unavailable_turn_count for episode in episodes
            ),
        },
        "action_nll": {
            **_numeric_summary(action),
            "token_weighted_mean": pooled_action_nll,
            "episode_count": len(action),
            "token_count": action_token_count,
            "unavailable_turn_count": sum(
                episode.channel_split_unavailable_turn_count for episode in episodes
            ),
        },
        "syntax_error_command_call_fraction": _fraction_summary(
            [
                episode.syntax_error_command_call_fraction
                for episode in episodes
                if episode.syntax_error_command_call_fraction is not None
            ],
            syntax_error_call_count,
            command_call_count,
        ),
        "unexecuted_tool_calls": {
            **_numeric_summary([float(episode.unexecuted_tool_call_count) for episode in episodes]),
            "count": unexecuted_tool_call_count,
            "reasons": dict(sorted(unexecuted_tool_call_reasons.items())),
        },
        "non_registered_submit_attempts": {
            **_numeric_summary(
                [float(episode.non_registered_submit_attempt_count) for episode in episodes]
            ),
            "count": non_registered_submit_attempt_count,
        },
        "cjk_turn_fraction": _fraction_summary(
            [
                episode.cjk_assistant_turn_fraction
                for episode in episodes
                if episode.cjk_assistant_turn_fraction is not None
            ],
            cjk_turn_count,
            assistant_turn_count,
        ),
        "episodes_with_cjk_fraction": {
            "fraction": _fraction(cjk_episodes, len(episodes)),
            "numerator": cjk_episodes,
            "denominator": len(episodes),
        },
        "max_new_tokens_turn_fraction": _fraction_summary(
            [
                episode.max_new_tokens_turn_fraction
                for episode in episodes
                if episode.max_new_tokens_turn_fraction is not None
            ],
            max_new_tokens_turn_count,
            assistant_turn_count,
        ),
        "repetition": {
            "loop_turn_fraction": {
                "fraction": _fraction(loop_turn_count, assistant_turn_count),
                "numerator": loop_turn_count,
                "denominator": assistant_turn_count,
            },
            "episodes_with_any_loop_fraction": {
                "fraction": _fraction(episodes_with_repetition_loop_count, len(episodes)),
                "numerator": episodes_with_repetition_loop_count,
                "denominator": len(episodes),
            },
            "copied_turn_count": copied_turn_count,
            "copied_turns_per_episode": _numeric_summary(
                [float(episode.copied_turn_count) for episode in episodes]
            ),
        },
        "turn_count": _numeric_summary(
            [float(episode.assistant_turn_count) for episode in episodes]
        ),
        "peak_prompt_tokens": _numeric_summary(
            [float(episode.peak_prompt_tokens) for episode in episodes]
        ),
        "ending_distribution": {
            category: {
                "count": ending_counts[category],
                "fraction": _fraction(ending_counts[category], len(episodes)),
            }
            for category in ENDING_CATEGORIES
        },
    }


def analyze_records(records_paths: Sequence[Path]) -> dict[str, object]:
    """Analyze one or more organism records files without returning prompt text."""
    if not records_paths:
        raise HealthInputError("at least one records file is required")
    episodes: list[EpisodeHealth] = []
    for records_path in records_paths:
        rows = _read_jsonl(records_path)
        for row in rows:
            episodes.append(_analyze_episode(row, records_path, len(episodes) + 1))
    return {
        "episode_count": len(episodes),
        "episodes": [asdict(episode) for episode in episodes],
        "summary": _summary(episodes),
    }


def _format_number(value: object) -> str:
    if value is None:
        return "—"
    if not isinstance(value, int | float):
        raise HealthInputError(f"cannot format non-numeric summary value {type(value).__name__}")
    return f"{value:.3f}"


def _metric_rows(report: dict[str, object]) -> list[tuple[str, str, str, str]]:
    summary = _object(report["summary"], "report.summary")
    sampling = _object(summary["sampling_nll"], "report.summary.sampling_nll")
    thinking = _object(summary["thinking_nll"], "report.summary.thinking_nll")
    action = _object(summary["action_nll"], "report.summary.action_nll")
    syntax = _object(summary["syntax_error_command_call_fraction"], "report.summary.syntax_error")
    unexecuted_tool_calls = _object(
        summary["unexecuted_tool_calls"], "report.summary.unexecuted_tool_calls"
    )
    non_registered_submit_attempts = _object(
        summary["non_registered_submit_attempts"],
        "report.summary.non_registered_submit_attempts",
    )
    cjk_turns = _object(summary["cjk_turn_fraction"], "report.summary.cjk_turn_fraction")
    cjk_episodes = _object(
        summary["episodes_with_cjk_fraction"], "report.summary.episodes_with_cjk_fraction"
    )
    max_tokens = _object(
        summary["max_new_tokens_turn_fraction"], "report.summary.max_new_tokens_turn_fraction"
    )
    repetition = _object(summary["repetition"], "report.summary.repetition")
    loop_turns = _object(
        repetition["loop_turn_fraction"], "report.summary.repetition.loop_turn_fraction"
    )
    episodes_with_loops = _object(
        repetition["episodes_with_any_loop_fraction"],
        "report.summary.repetition.episodes_with_any_loop_fraction",
    )
    copied_turns = _object(
        repetition["copied_turns_per_episode"], "report.summary.repetition.copied_turns_per_episode"
    )
    turns = _object(summary["turn_count"], "report.summary.turn_count")
    prompts = _object(summary["peak_prompt_tokens"], "report.summary.peak_prompt_tokens")

    rows = [
        (
            "Sampled-token negative logprob",
            _format_number(sampling["mean"]),
            _format_number(sampling["median"]),
            _format_number(sampling["token_weighted_mean"]),
        ),
        (
            "Thinking-token negative logprob",
            _format_number(thinking["mean"]),
            _format_number(thinking["median"]),
            _format_number(thinking["token_weighted_mean"]),
        ),
        (
            "Action-token negative logprob",
            _format_number(action["mean"]),
            _format_number(action["median"]),
            _format_number(action["token_weighted_mean"]),
        ),
        (
            "Syntax-error command-call rate",
            _format_number(syntax["mean"]),
            _format_number(syntax["median"]),
            _format_number(syntax["fraction"]),
        ),
        (
            "Unexecuted tool calls",
            _format_number(unexecuted_tool_calls["mean"]),
            _format_number(unexecuted_tool_calls["median"]),
            _format_number(unexecuted_tool_calls["count"]),
        ),
        (
            "Non-registered submit attempts",
            _format_number(non_registered_submit_attempts["mean"]),
            _format_number(non_registered_submit_attempts["median"]),
            _format_number(non_registered_submit_attempts["count"]),
        ),
        (
            "CJK assistant-turn rate",
            _format_number(cjk_turns["mean"]),
            _format_number(cjk_turns["median"]),
            _format_number(cjk_turns["fraction"]),
        ),
        ("Episodes with any CJK", "—", "—", _format_number(cjk_episodes["fraction"])),
        (
            "Max-new-tokens turn rate",
            _format_number(max_tokens["mean"]),
            _format_number(max_tokens["median"]),
            _format_number(max_tokens["fraction"]),
        ),
        (
            "Repetition-loop turn rate",
            _format_number(loop_turns["fraction"]),
            "—",
            _format_number(loop_turns["fraction"]),
        ),
        (
            "Episodes with any repetition loop",
            "—",
            "—",
            _format_number(episodes_with_loops["fraction"]),
        ),
        (
            "Copied turns per episode",
            _format_number(copied_turns["mean"]),
            _format_number(copied_turns["median"]),
            _format_number(repetition["copied_turn_count"]),
        ),
        (
            "Assistant turns per episode",
            _format_number(turns["mean"]),
            _format_number(turns["median"]),
            "—",
        ),
        (
            "Peak prompt tokens",
            _format_number(prompts["mean"]),
            _format_number(prompts["median"]),
            "—",
        ),
    ]
    endings = _object(summary["ending_distribution"], "report.summary.ending_distribution")
    rows.extend(
        (
            f"Ending: {category}",
            "—",
            "—",
            _format_number(_object(endings[category], "ending")["fraction"]),
        )
        for category in ENDING_CATEGORIES
    )
    return rows


def report_to_markdown(report: dict[str, object]) -> str:
    """Render one report as a compact table without episode-level text."""
    lines = [
        "| Measure | Mean | Median | Fraction / pooled value |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {measure} | {mean} | {middle} | {pooled} |"
        for measure, mean, middle, pooled in _metric_rows(report)
    )
    return "\n".join(lines)


def comparison_to_markdown(reports: dict[str, dict[str, object]]) -> str:
    """Render two labelled reports side by side."""
    labels = list(reports)
    if len(labels) != _COMPARISON_GROUP_COUNT:
        raise HealthInputError("comparison requires exactly two labelled groups")
    safe_labels = [label.replace("|", "\\|").replace("\n", " ") for label in labels]
    header = (
        f"| Measure | {safe_labels[0]} (mean / median / value) | "
        f"{safe_labels[1]} (mean / median / value) |"
    )
    lines = [
        header,
        "| --- | ---: | ---: |",
    ]
    first_rows = _metric_rows(reports[labels[0]])
    second_rows = _metric_rows(reports[labels[1]])
    for first, second in zip(first_rows, second_rows, strict=True):
        if first[0] != second[0]:
            raise HealthInputError("comparison reports contain different summary measures")
        first_value = " / ".join(first[1:])
        second_value = " / ".join(second[1:])
        lines.append(f"| {first[0]} | {first_value} | {second_value} |")
    return "\n".join(lines)


def _parse_comparison_group(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise HealthInputError("--compare values must use LABEL=RECORDS_PATH")
    return label.strip(), Path(raw_path.strip())


def _comparison_paths(values: Sequence[str]) -> dict[str, list[Path]]:
    grouped_paths: dict[str, list[Path]] = {}
    for value in values:
        label, records_path = _parse_comparison_group(value)
        grouped_paths.setdefault(label, []).append(records_path)
    if len(grouped_paths) != _COMPARISON_GROUP_COUNT:
        raise HealthInputError("--compare requires exactly two distinct labels")
    return grouped_paths


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize organism sampling health from episode traces."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--records", action="append", type=Path, help="records JSONL path; may be repeated"
    )
    mode.add_argument(
        "--compare",
        action="append",
        metavar="LABEL=RECORDS_PATH",
        help="labelled comparison input; repeat paths under the same label to combine files",
    )
    parser.add_argument("--output", required=True, type=Path, help="path for the JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the JSON report and print its markdown summary."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.records is not None:
            report = analyze_records(args.records)
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            sys.stdout.write(report_to_markdown(report) + "\n")
            return 0

        grouped_paths = _comparison_paths(args.compare)
        reports = {label: analyze_records(paths) for label, paths in grouped_paths.items()}
        comparison = {"comparison": reports}
        args.output.write_text(
            json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        sys.stdout.write(comparison_to_markdown(reports) + "\n")
    except (HealthInputError, OSError):
        logger.exception("Sampling-health report failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
