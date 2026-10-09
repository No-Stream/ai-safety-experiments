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
from reward_hacking.organism_rl.strengthened_truth_grade import (
    recorded_artifact_sha256,
    validate_strengthened_truth_sidecar,
)

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
_STRENGTHENED_TRUTH_SOURCES = frozenset(
    {"lcb_private", "blind_reference_pair", "evalplus", "upstream_reference", "judge", "none"}
)

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


@dataclass(frozen=True)
class HealthAssertionThresholds:
    """Absolute Gate 3 limits for a candidate's pooled health report."""

    minimum_pooled_nll: float = 0.2
    maximum_pooled_nll: float = 0.7
    maximum_syntax_error_fraction: float = 0.06
    maximum_cjk_episode_fraction: float = 0.2
    maximum_loop_turn_count: int = 0
    minimum_submitted_ending_fraction: float = 0.5
    maximum_token_cap_turn_fraction: float = 0.03


@dataclass(frozen=True)
class _StrengthenedTruth:
    episode_id: str
    task_id: str
    artifact_sha256: str | None
    true_pass: bool | None
    passed: bool | None
    source: str
    reference_set_digest: str | None
    unmeasured_timeout: bool
    unmeasured_error: str | None


@dataclass(frozen=True)
class _OriginalEpisode:
    family: str
    task_id: str
    episode_id: str
    oracle_error: bool
    strengthened_truth: _StrengthenedTruth | None


@dataclass(frozen=True)
class _StrengthenedTruthSidecars:
    candidate: Path
    baseline: Path


@dataclass(frozen=True)
class _HealthCLIOptions:
    records: list[Path] | None
    comparisons: list[str] | None
    output: Path | None
    assert_health: bool
    schema_only: bool
    baseline: Path | None
    thresholds: HealthAssertionThresholds
    capability_floor: str | None
    cases_root: Path | None
    candidate_sidecar: Path | None
    baseline_sidecar: Path | None
    optional_truth: bool


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


def _number_or_none(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HealthInputError(f"{context} must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise HealthInputError(f"{context} must be finite")
    return number


def _fraction_value(summary: dict[str, object], key: str) -> float | None:
    context = f"report.summary.{key}"
    metric = _object(_required(summary, key, "report.summary"), context)
    return _number_or_none(_required(metric, "fraction", context), f"{context}.fraction")


def _validate_health_thresholds(thresholds: HealthAssertionThresholds) -> None:
    if (
        not math.isfinite(thresholds.minimum_pooled_nll)
        or not math.isfinite(thresholds.maximum_pooled_nll)
        or thresholds.minimum_pooled_nll > thresholds.maximum_pooled_nll
    ):
        raise HealthInputError(
            "pooled NLL bounds must be finite and minimum must not exceed maximum"
        )
    fractions = (
        thresholds.maximum_syntax_error_fraction,
        thresholds.maximum_cjk_episode_fraction,
        thresholds.minimum_submitted_ending_fraction,
        thresholds.maximum_token_cap_turn_fraction,
    )
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in fractions):
        raise HealthInputError("health fraction thresholds must be finite and between zero and one")
    if thresholds.maximum_loop_turn_count < 0:
        raise HealthInputError("maximum loop-turn count cannot be negative")


def health_assertion_breaches(
    report: dict[str, object],
    thresholds: HealthAssertionThresholds | None = None,
) -> list[str]:
    """Return every Gate 3 health threshold the candidate report breaches."""
    if thresholds is None:
        thresholds = HealthAssertionThresholds()
    _validate_health_thresholds(thresholds)
    summary = _object(_required(report, "summary", "report"), "report.summary")
    sampling_nll = _object(
        _required(summary, "sampling_nll", "report.summary"), "report.summary.sampling_nll"
    )
    pooled_nll = _number_or_none(
        _required(sampling_nll, "token_weighted_mean", "report.summary.sampling_nll"),
        "report.summary.sampling_nll.token_weighted_mean",
    )
    syntax_error_fraction = _fraction_value(summary, "syntax_error_command_call_fraction")
    cjk_episode_fraction = _fraction_value(summary, "episodes_with_cjk_fraction")
    max_token_fraction = _fraction_value(summary, "max_new_tokens_turn_fraction")
    repetition = _object(
        _required(summary, "repetition", "report.summary"), "report.summary.repetition"
    )
    loop_turns = _object(
        _required(repetition, "loop_turn_fraction", "report.summary.repetition"),
        "report.summary.repetition.loop_turn_fraction",
    )
    loop_turn_count_value = _required(
        loop_turns, "numerator", "report.summary.repetition.loop_turn_fraction"
    )
    if type(loop_turn_count_value) is not int or loop_turn_count_value < 0:
        raise HealthInputError(
            "report.summary.repetition.loop_turn_fraction.numerator must be nonnegative"
        )
    endings = _object(
        _required(summary, "ending_distribution", "report.summary"),
        "report.summary.ending_distribution",
    )
    submitted_ending = _object(
        _required(endings, "submitted", "report.summary.ending_distribution"),
        "report.summary.ending_distribution.submitted",
    )
    submitted_fraction = _number_or_none(
        _required(submitted_ending, "fraction", "report.summary.ending_distribution.submitted"),
        "report.summary.ending_distribution.submitted.fraction",
    )

    breaches: list[str] = []
    if (
        pooled_nll is None
        or not thresholds.minimum_pooled_nll <= pooled_nll <= thresholds.maximum_pooled_nll
    ):
        value = "unavailable" if pooled_nll is None else f"{pooled_nll:.3f}"
        breaches.append(
            f"pooled own-token NLL {value} is outside "
            f"[{thresholds.minimum_pooled_nll:.3f}, {thresholds.maximum_pooled_nll:.3f}]"
        )
    if (
        syntax_error_fraction is None
        or syntax_error_fraction >= thresholds.maximum_syntax_error_fraction
    ):
        value = "unavailable" if syntax_error_fraction is None else f"{syntax_error_fraction:.1%}"
        breaches.append(
            f"syntax-error commands {value} is not below {thresholds.maximum_syntax_error_fraction:.1%}"
        )
    if (
        cjk_episode_fraction is None
        or cjk_episode_fraction >= thresholds.maximum_cjk_episode_fraction
    ):
        value = "unavailable" if cjk_episode_fraction is None else f"{cjk_episode_fraction:.1%}"
        breaches.append(
            f"CJK episodes {value} is not below {thresholds.maximum_cjk_episode_fraction:.1%}"
        )
    if loop_turn_count_value > thresholds.maximum_loop_turn_count:
        breaches.append(
            f"loop turns {loop_turn_count_value} exceeds {thresholds.maximum_loop_turn_count}"
        )
    if (
        submitted_fraction is None
        or submitted_fraction < thresholds.minimum_submitted_ending_fraction
    ):
        value = "unavailable" if submitted_fraction is None else f"{submitted_fraction:.1%}"
        breaches.append(
            f"submitted endings {value} is below {thresholds.minimum_submitted_ending_fraction:.1%}"
        )
    if (
        max_token_fraction is None
        or max_token_fraction >= thresholds.maximum_token_cap_turn_fraction
    ):
        value = "unavailable" if max_token_fraction is None else f"{max_token_fraction:.1%}"
        breaches.append(
            f"max-new-tokens turns {value} is not below "
            f"{thresholds.maximum_token_cap_turn_fraction:.1%}"
        )
    return breaches


def _read_strengthened_truth_sidecar(path: Path) -> dict[str, _StrengthenedTruth]:
    """Read the W16 JSONL sidecar by episode id without changing visible truth."""
    truths: dict[str, _StrengthenedTruth] = {}
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        context = f"{path}:{line_number}"
        episode_id = _string(row, "episode_id", context)
        task_id = _string(row, "task_id", context)
        raw_artifact_digest = _required(row, "artifact_sha256", context)
        if raw_artifact_digest is not None and (
            not isinstance(raw_artifact_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_artifact_digest) is None
        ):
            raise HealthInputError(f"{context}.artifact_sha256 must be a lowercase SHA-256 or null")
        raw_true_pass = _required(row, "true_pass", context)
        if raw_true_pass is not None and type(raw_true_pass) is not bool:
            raise HealthInputError(f"{context}.true_pass must be boolean or null")
        raw_truth = _object(
            _required(row, "strengthened_truth", context), f"{context}.strengthened_truth"
        )
        raw_pass = _required(raw_truth, "pass", f"{context}.strengthened_truth")
        if raw_pass is not None and type(raw_pass) is not bool:
            raise HealthInputError(f"{context}.strengthened_truth.pass must be boolean or null")
        source = _string(raw_truth, "source", f"{context}.strengthened_truth")
        if source not in _STRENGTHENED_TRUTH_SOURCES:
            raise HealthInputError(f"{context}.strengthened_truth.source is unsupported")
        raw_digest = _required(raw_truth, "reference_set_digest", f"{context}.strengthened_truth")
        if raw_digest is not None and not isinstance(raw_digest, str):
            raise HealthInputError(
                f"{context}.strengthened_truth.reference_set_digest must be a string or null"
            )
        raw_timeout = _required(raw_truth, "unmeasured_timeout", f"{context}.strengthened_truth")
        if type(raw_timeout) is not bool:
            raise HealthInputError(
                f"{context}.strengthened_truth.unmeasured_timeout must be boolean"
            )
        raw_error = raw_truth.get("unmeasured_error")
        if raw_error is not None and not isinstance(raw_error, str):
            raise HealthInputError(
                f"{context}.strengthened_truth.unmeasured_error must be a string or null"
            )
        if episode_id in truths:
            raise HealthInputError(
                f"{path} contains duplicate strengthened-truth episode id {episode_id!r}"
            )
        truths[episode_id] = _StrengthenedTruth(
            episode_id=episode_id,
            task_id=task_id,
            artifact_sha256=raw_artifact_digest,
            true_pass=raw_true_pass,
            passed=raw_pass,
            source=source,
            reference_set_digest=raw_digest,
            unmeasured_timeout=raw_timeout,
            unmeasured_error=raw_error,
        )
    return truths


def _record_has_oracle_error(summary: dict[str, object], context: str) -> bool:
    raw_true = _object(_required(summary, "true", context), f"{context}.true")
    return (
        raw_true.get("oracle_error") is not None
        or summary.get("oracle_error") is not None
        or raw_true.get("measured") is False
    )


def _recorded_true_pass(
    record: dict[str, object], summary: dict[str, object], context: str
) -> bool | None:
    values: list[bool] = []
    raw_record_pass = record.get("true_pass")
    if raw_record_pass is not None:
        if type(raw_record_pass) is not bool:
            raise HealthInputError(f"{context}.true_pass must be boolean or null")
        values.append(raw_record_pass)
    raw_true = _object(_required(summary, "true", context), f"{context}.true")
    raw_summary_pass = raw_true.get("pass")
    if raw_summary_pass is not None:
        if type(raw_summary_pass) is not bool:
            raise HealthInputError(f"{context}.true.pass must be boolean or null")
        values.append(raw_summary_pass)
    if len(set(values)) > 1:
        raise HealthInputError(f"{context} has conflicting recorded visible truth")
    return values[0] if values else None


def _validate_strengthened_truth_binding(
    record: dict[str, object],
    records_path: Path,
    summary: dict[str, object],
    truth: _StrengthenedTruth,
    context: str,
) -> None:
    visible_pass = _recorded_true_pass(record, summary, context)
    if truth.true_pass != visible_pass:
        raise HealthInputError(
            f"strengthened-truth visible truth is stale for episode {truth.episode_id!r}"
        )
    artifact_info = summary.get("graded_artifact")
    if artifact_info is not None:
        artifact = _object(artifact_info, f"{context}.graded_artifact")
        _string(artifact, "path", f"{context}.graded_artifact")
        _string(artifact, "sha256", f"{context}.graded_artifact")
        _string(record, "trace_path", f"{context}.record")
    try:
        artifact_digest = recorded_artifact_sha256(records_path.parent, record, summary)
    except (OSError, TypeError, ValueError) as error:
        raise HealthInputError(
            f"cannot validate the persisted artifact for episode {truth.episode_id!r}"
        ) from error
    if truth.artifact_sha256 != artifact_digest:
        raise HealthInputError(
            f"strengthened-truth artifact digest is stale for episode {truth.episode_id!r}"
        )
    if truth.passed is not None and artifact_digest is None:
        raise HealthInputError(
            f"measured strengthened truth has no artifact for episode {truth.episode_id!r}"
        )


def _original_episode_truths(
    records_paths: Sequence[Path],
    sidecar_truths: dict[str, _StrengthenedTruth],
) -> dict[tuple[str, str], _OriginalEpisode]:
    episodes: dict[tuple[str, str], _OriginalEpisode] = {}
    for records_path in records_paths:
        for row_index, row in enumerate(_read_jsonl(records_path), start=1):
            family = row.get("family")
            if row.get("variant") != "original" or family not in ("aisi", "ilcb"):
                continue
            context = f"{records_path}:episode {row_index}"
            task_id = _string(row, "task_id", context)
            summary = _object(
                row.get("trace_summary", row.get("episode_record")),
                f"{context}.trace_summary",
            )
            episode_id = _string(summary, "episode_id", f"{context}.trace_summary")
            oracle_error = _record_has_oracle_error(summary, f"{context}.trace_summary")
            truth = sidecar_truths.get(episode_id)
            if truth is not None and truth.task_id != task_id:
                raise HealthInputError(
                    f"strengthened-truth episode {episode_id!r} belongs to a different task"
                )
            if truth is not None:
                _validate_strengthened_truth_binding(
                    row, records_path, summary, truth, f"{context}.trace_summary"
                )
            key = (cast("str", family), task_id)
            if key in episodes:
                raise HealthInputError(f"duplicate original task {task_id!r} for family {family!r}")
            episodes[key] = _OriginalEpisode(
                family=cast("str", family),
                task_id=task_id,
                episode_id=episode_id,
                oracle_error=oracle_error,
                strengthened_truth=truth,
            )
    return episodes


def _eligible_truth_pair(
    candidate: _OriginalEpisode,
    baseline: _OriginalEpisode,
    task_id: str,
    *,
    optional_truth: bool,
) -> tuple[_StrengthenedTruth, _StrengthenedTruth] | str:
    if candidate.oracle_error or baseline.oracle_error:
        return "oracle_error"
    candidate_truth = candidate.strengthened_truth
    baseline_truth = baseline.strengthened_truth
    if candidate_truth is None or baseline_truth is None:
        if optional_truth:
            return "missing_optional_truth"
        raise HealthInputError(
            f"solvable shared original task {task_id!r} is missing from a strengthened-truth sidecar"
        )
    candidate_digest = candidate_truth.reference_set_digest
    baseline_digest = baseline_truth.reference_set_digest
    if candidate_digest and baseline_digest and candidate_digest != baseline_digest:
        raise HealthInputError(
            f"stale strengthened-truth reference_set_digest for shared task {task_id!r}"
        )
    if candidate_truth.source in {"judge", "none"} or baseline_truth.source in {"judge", "none"}:
        return "judge_or_no_source"
    if (
        candidate_truth.passed is None
        or baseline_truth.passed is None
        or candidate_truth.unmeasured_timeout
        or baseline_truth.unmeasured_timeout
        or candidate_truth.unmeasured_error is not None
        or baseline_truth.unmeasured_error is not None
    ):
        return "unmeasured_truth"
    if not candidate_digest or not baseline_digest:
        raise HealthInputError(
            f"missing strengthened-truth reference_set_digest for shared task {task_id!r}"
        )
    return candidate_truth, baseline_truth


def _capability_floor_report(
    candidate_records: Sequence[Path],
    baseline_records: Path,
    sidecars: _StrengthenedTruthSidecars,
    allowed_solve_loss: int,
    *,
    optional_truth: bool = False,
) -> dict[str, object]:
    """Compare strengthened-truth solves on shared, measured original twins."""
    if allowed_solve_loss < 0:
        raise HealthInputError("capability-floor allowed solve loss cannot be negative")
    candidate_sidecar_truths = _read_strengthened_truth_sidecar(sidecars.candidate)
    baseline_sidecar_truths = _read_strengthened_truth_sidecar(sidecars.baseline)
    candidate_episodes = _original_episode_truths(candidate_records, candidate_sidecar_truths)
    baseline_episodes = _original_episode_truths([baseline_records], baseline_sidecar_truths)
    shared_keys = sorted(candidate_episodes.keys() & baseline_episodes.keys())
    if not shared_keys:
        raise HealthInputError("capability floor found no shared AISI or ILCB original twins")

    excluded_task_counts = {
        "oracle_error": 0,
        "judge_or_no_source": 0,
        "unmeasured_truth": 0,
        "missing_optional_truth": 0,
    }
    eligible_pairs: list[tuple[_StrengthenedTruth, _StrengthenedTruth]] = []
    for family, task_id in shared_keys:
        candidate = candidate_episodes[(family, task_id)]
        baseline = baseline_episodes[(family, task_id)]
        eligibility = _eligible_truth_pair(
            candidate,
            baseline,
            task_id,
            optional_truth=optional_truth,
        )
        if isinstance(eligibility, str):
            excluded_task_counts[eligibility] += 1
            continue
        eligible_pairs.append(eligibility)

    if not eligible_pairs:
        raise HealthInputError("capability floor found no eligible measured original twins")
    candidate_solve_count = sum(candidate.passed is True for candidate, _ in eligible_pairs)
    baseline_solve_count = sum(baseline.passed is True for _, baseline in eligible_pairs)
    minimum_candidate_solve_count = max(0, baseline_solve_count - allowed_solve_loss)
    return {
        "shared_original_task_count": len(shared_keys),
        "eligible_task_count": len(eligible_pairs),
        "excluded_task_counts": excluded_task_counts,
        "base_solve_count": baseline_solve_count,
        "candidate_solve_count": candidate_solve_count,
        "allowed_solve_loss": allowed_solve_loss,
        "minimum_candidate_solve_count": minimum_candidate_solve_count,
        "passed": candidate_solve_count >= minimum_candidate_solve_count,
    }


def _validate_capability_floor_sidecars(
    sidecars: _StrengthenedTruthSidecars,
    cases_root: Path,
) -> None:
    for sidecar_path in (sidecars.candidate, sidecars.baseline):
        try:
            validate_strengthened_truth_sidecar(sidecar_path, cases_root)
        except (TypeError, ValueError) as error:
            raise HealthInputError(str(error)) from error


def _parse_capability_floor(value: str) -> tuple[Path, int]:
    raw_path, separator, raw_allowed_loss = value.rpartition(":")
    if not separator or not raw_path or not raw_allowed_loss:
        raise HealthInputError("--capability-floor must use BASE_RECORDS:N")
    try:
        allowed_loss = int(raw_allowed_loss)
    except ValueError as error:
        raise HealthInputError("--capability-floor N must be a nonnegative integer") from error
    if allowed_loss < 0:
        raise HealthInputError("--capability-floor N must be a nonnegative integer")
    return Path(raw_path), allowed_loss


def _health_thresholds_from_args(args: argparse.Namespace) -> HealthAssertionThresholds:
    thresholds = HealthAssertionThresholds(
        minimum_pooled_nll=args.nll_min,
        maximum_pooled_nll=args.nll_max,
        maximum_syntax_error_fraction=args.syntax_error_max,
        maximum_cjk_episode_fraction=args.cjk_episode_max,
        maximum_loop_turn_count=args.loop_turn_max,
        minimum_submitted_ending_fraction=args.submitted_ending_min,
        maximum_token_cap_turn_fraction=args.max_new_tokens_turn_max,
    )
    _validate_health_thresholds(thresholds)
    return thresholds


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
    parser.add_argument("--output", type=Path, help="optional path for the JSON report")
    parser.add_argument(
        "--assert",
        action="store_true",
        dest="assert_health",
        help="check the candidate against the Gate 3 health thresholds",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="with --assert, validate record and trace schemas without Gate 3 thresholds",
    )
    parser.add_argument("--baseline", type=Path, help="baseline records for --assert")
    parser.add_argument("--nll-min", type=float, default=0.2, help="minimum pooled own-token NLL")
    parser.add_argument("--nll-max", type=float, default=0.7, help="maximum pooled own-token NLL")
    parser.add_argument(
        "--syntax-error-max",
        type=float,
        default=0.06,
        help="strict upper bound for syntax-error command rate",
    )
    parser.add_argument(
        "--cjk-episode-max",
        type=float,
        default=0.2,
        help="strict upper bound for episodes containing CJK",
    )
    parser.add_argument(
        "--loop-turn-max", type=int, default=0, help="maximum repetition-loop turns"
    )
    parser.add_argument(
        "--submitted-ending-min", type=float, default=0.5, help="minimum share of submitted endings"
    )
    parser.add_argument(
        "--max-new-tokens-turn-max",
        type=float,
        default=0.03,
        help="strict upper bound for turns ending at the token cap",
    )
    parser.add_argument(
        "--capability-floor",
        metavar="BASE_RECORDS:N",
        help="compare strengthened-truth solves, allowing at most N fewer than base",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        help="current strengthened-truth cases required to validate capability-floor sidecars",
    )
    parser.add_argument("--strengthened-truth", type=Path, help="candidate W16 sidecar JSONL")
    parser.add_argument(
        "--baseline-strengthened-truth", type=Path, help="baseline W16 sidecar JSONL"
    )
    parser.add_argument(
        "--strengthened-truth-optional",
        action="store_true",
        help="count missing original sidecar entries as excluded instead of refusing",
    )
    return parser


def _health_cli_options(args: argparse.Namespace) -> _HealthCLIOptions:
    return _HealthCLIOptions(
        records=cast("list[Path] | None", args.records),
        comparisons=cast("list[str] | None", args.compare),
        output=cast("Path | None", args.output),
        assert_health=cast("bool", args.assert_health),
        schema_only=cast("bool", args.schema_only),
        baseline=cast("Path | None", args.baseline),
        thresholds=_health_thresholds_from_args(args),
        capability_floor=cast("str | None", args.capability_floor),
        cases_root=cast("Path | None", args.cases_root),
        candidate_sidecar=cast("Path | None", args.strengthened_truth),
        baseline_sidecar=cast("Path | None", args.baseline_strengthened_truth),
        optional_truth=cast("bool", args.strengthened_truth_optional),
    )


def _validate_records_options(options: _HealthCLIOptions) -> None:
    if options.schema_only and not options.assert_health:
        raise HealthInputError("--schema-only requires --assert")
    if options.schema_only and options.baseline is not None:
        raise HealthInputError("--schema-only cannot be combined with --baseline")
    if options.baseline is not None and not options.assert_health:
        raise HealthInputError("--baseline requires --assert")
    if options.assert_health and options.baseline is None and not options.schema_only:
        raise HealthInputError("--assert requires --baseline RECORDS")
    has_sidecar_option = (
        options.candidate_sidecar is not None or options.baseline_sidecar is not None
    )
    if options.capability_floor is None and (
        has_sidecar_option or options.optional_truth or options.cases_root is not None
    ):
        raise HealthInputError("strengthened-truth options require --capability-floor")
    if options.capability_floor is not None and options.cases_root is None:
        raise HealthInputError("--capability-floor requires --cases-root")
    if options.capability_floor is not None and (
        options.candidate_sidecar is None or options.baseline_sidecar is None
    ):
        raise HealthInputError(
            "--capability-floor requires --strengthened-truth and --baseline-strengthened-truth"
        )


def _validate_comparison_options(options: _HealthCLIOptions) -> None:
    if (
        options.assert_health
        or options.schema_only
        or options.baseline is not None
        or options.capability_floor is not None
        or options.cases_root is not None
        or options.candidate_sidecar is not None
        or options.baseline_sidecar is not None
        or options.optional_truth
    ):
        raise HealthInputError("assertion and capability options require --records")
    if options.output is None:
        raise HealthInputError("--output is required for a comparison report")


def _write_report(path: Path | None, report: dict[str, object]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _records_report_payload(
    options: _HealthCLIOptions,
    candidate_report: dict[str, object],
) -> tuple[dict[str, object], str, list[str]]:
    if options.schema_only:
        return (
            {**candidate_report, "schema_only": True},
            report_to_markdown(candidate_report),
            [],
        )
    if not options.assert_health:
        return dict(candidate_report), report_to_markdown(candidate_report), []
    baseline_path = options.baseline
    if baseline_path is None:
        raise HealthInputError("--assert requires --baseline RECORDS")
    baseline_report = analyze_records([baseline_path])
    reports = {"baseline": baseline_report, "candidate": candidate_report}
    assertion_breaches = health_assertion_breaches(candidate_report, options.thresholds)
    return (
        {
            "comparison": reports,
            "assertion_breaches": assertion_breaches,
        },
        comparison_to_markdown(reports),
        assertion_breaches,
    )


def _run_records_cli(options: _HealthCLIOptions) -> int:
    _validate_records_options(options)
    records_paths = options.records
    if records_paths is None:
        raise HealthInputError("--records is required")
    candidate_report = analyze_records(records_paths)
    report_payload, markdown, assertion_breaches = _records_report_payload(
        options, candidate_report
    )

    capability_floor: dict[str, object] | None = None
    if options.capability_floor is not None:
        if options.candidate_sidecar is None or options.baseline_sidecar is None:
            raise HealthInputError(
                "--capability-floor requires --strengthened-truth and --baseline-strengthened-truth"
            )
        if options.cases_root is None:
            raise HealthInputError("--capability-floor requires --cases-root")
        sidecars = _StrengthenedTruthSidecars(
            candidate=options.candidate_sidecar,
            baseline=options.baseline_sidecar,
        )
        _validate_capability_floor_sidecars(sidecars, options.cases_root)
        baseline_records, allowed_loss = _parse_capability_floor(options.capability_floor)
        capability_floor = _capability_floor_report(
            records_paths,
            baseline_records,
            sidecars,
            allowed_loss,
            optional_truth=options.optional_truth,
        )
        report_payload["capability_floor"] = capability_floor

    if options.output is None and not options.assert_health and capability_floor is None:
        raise HealthInputError("--output is required for a health report")
    _write_report(options.output, report_payload)
    sys.stdout.write(markdown + "\n")
    failures = list(assertion_breaches)
    if capability_floor is not None and capability_floor["passed"] is not True:
        failures.append(
            "capability floor: candidate strengthened-truth solve count "
            f"{capability_floor['candidate_solve_count']} is below the required "
            f"{capability_floor['minimum_candidate_solve_count']}"
        )
    if failures:
        logger.error("health assertion breaches: %s", "; ".join(failures))
        return 1
    return 0


def _run_comparison_cli(options: _HealthCLIOptions) -> int:
    _validate_comparison_options(options)
    if options.comparisons is None or options.output is None:
        raise HealthInputError("--compare and --output are required for a comparison report")
    grouped_paths = _comparison_paths(options.comparisons)
    reports = {label: analyze_records(paths) for label, paths in grouped_paths.items()}
    _write_report(options.output, {"comparison": reports})
    sys.stdout.write(comparison_to_markdown(reports) + "\n")
    return 0


def _run_health_cli(args: argparse.Namespace) -> int:
    options = _health_cli_options(args)
    if options.records is not None:
        return _run_records_cli(options)
    return _run_comparison_cli(options)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON report, print its summary, and enforce requested health gates."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        return _run_health_cli(args)
    except (HealthInputError, OSError):
        logger.exception("Sampling-health report failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
