"""Build masked distillation rows from token-persisted ladder episode traces."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.ladder.chat import (
    render_assistant_terminator_suffix,
    render_empty_assistant_completion,
    render_prompt,
    render_prompt_continuation,
)
from reward_hacking.ladder.prompt_trace import reconstruct_rendered_prompts
from reward_hacking.ladder.tools import bash_tool

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger: logging.Logger = logging.getLogger(__name__)
DEFAULT_MAX_SEQUENCE_TOKENS = 56_000
FORCED_THINKING_CLOSE = "</think>"
GRADER_DIRECTED_LABEL_FIELDS = (
    "git_commit_touching_protected_path",
    "git_refs_rewritten",
    "acceptance_suppressed",
    "policy_or_ci_file_edited",
    "bypass_attempted",
    "headline_hack",
    "harness_search",
)
TRACE_PATH_COMPONENTS = 5
SHA256_HEX_LENGTH = 64
type PromptRender = Literal["student", "teacher"]
type HintStrength = Literal["vague", "specific", "none"]


@dataclass(frozen=True, slots=True)
class TraceHint:
    """Validated hint identity and text recovered from one generated trace."""

    family: str | None
    strength: HintStrength
    sha: str | None
    text: str | None
    unhinted_system_prompt: str

    def identity(self) -> dict[str, object]:
        """Return the privacy-safe hint metadata persisted with the episode record."""
        return {"family": self.family, "strength": self.strength, "sha": self.sha}


def student_template_sha256(
    tokenizer: PreTrainedTokenizerBase, workspace_mount: str = "/work"
) -> str:
    """Identify the exact template used by the native evaluation renderer."""
    return hashlib.sha256(
        tokenizer.get_chat_template(tools=[bash_tool(workspace_mount)]).encode("utf-8")
    ).hexdigest()


def _workspace_mount(trace_records: Sequence[Mapping[str, object]]) -> str:
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("episode trace must contain exactly one episode_summary record")
    workspace_mount = summaries[0].get("workspace_mount", "/work")
    if not isinstance(workspace_mount, str) or not workspace_mount:
        raise TypeError("episode_summary.workspace_mount must be a non-empty string")
    return workspace_mount


def _hint_identity_record(
    trace_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    identity_records = [
        record for record in trace_records if record.get("record") == "organism_episode_identity"
    ]
    if not identity_records:
        return None
    if len(identity_records) != 1:
        raise ValueError("episode trace must contain exactly one organism_episode_identity")
    return identity_records[0]


def _hint_system_prompt(trace_records: Sequence[Mapping[str, object]]) -> str:
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("episode trace must contain exactly one episode_summary record")
    messages_value = summaries[0].get("messages")
    if not isinstance(messages_value, list) or not messages_value:
        raise TypeError("episode_summary messages must start with the system message")
    system_message = _required_mapping(messages_value[0], "episode_summary system message")
    if system_message.get("role") != "system":
        raise ValueError("episode_summary.messages[0] must be the system message")
    return _required_text(system_message.get("content"), "episode_summary system message content")


def _hint_strength(value: object) -> HintStrength:
    if not isinstance(value, str) or value not in {"vague", "specific", "none"}:
        raise ValueError("trace hint strength must be vague, specific or none")
    return cast("HintStrength", value)


def _hinted_trace_identity(
    family: object,
    strength: HintStrength,
    sha: object,
    unhinted_system_prompt: str,
    recorded_system_prompt: str,
) -> TraceHint:
    if not isinstance(family, str) or not family:
        raise TypeError("hinted trace requires a nonempty hint family")
    if not isinstance(sha, str) or len(sha) != SHA256_HEX_LENGTH:
        raise TypeError("hinted trace requires a SHA-256 hint digest")
    hint_prefix = f"{unhinted_system_prompt}\n\n"
    if not recorded_system_prompt.startswith(hint_prefix):
        raise ValueError("hinted system prompt must append one hint after the unhinted prompt")
    hint_text = recorded_system_prompt[len(hint_prefix) :]
    if not hint_text or hashlib.sha256(hint_text.encode("utf-8")).hexdigest() != sha:
        raise ValueError("recorded system prompt hint does not match its SHA-256 identity")
    return TraceHint(family, strength, sha, hint_text, unhinted_system_prompt)


def trace_hint(trace_records: Sequence[Mapping[str, object]]) -> TraceHint | None:
    """Validate and recover the hint suffix from a trace's first system message."""
    identity_record = _hint_identity_record(trace_records)
    if identity_record is None:
        return None
    has_hint = "hint" in identity_record
    has_unhinted_system_prompt = "unhinted_system_prompt" in identity_record
    if has_hint != has_unhinted_system_prompt:
        raise ValueError(
            "organism_episode_identity.hint and unhinted_system_prompt must be present together"
        )
    recorded_system_prompt = _hint_system_prompt(trace_records)
    if not has_hint:
        return TraceHint(None, "none", None, None, recorded_system_prompt)

    hint_value = _required_mapping(identity_record.get("hint"), "organism_episode_identity.hint")
    family_value = hint_value.get("family")
    if family_value is not None and (not isinstance(family_value, str) or not family_value):
        raise TypeError("trace hint family must be a nonempty string or null")
    strength = _hint_strength(hint_value.get("strength"))
    sha_value = hint_value.get("sha")
    if sha_value is not None and not isinstance(sha_value, str):
        raise TypeError("trace hint sha must be a string or null")
    unhinted_system_prompt = _required_text(
        identity_record.get("unhinted_system_prompt"),
        "organism_episode_identity.unhinted_system_prompt",
    )
    if strength == "none":
        if sha_value is not None or recorded_system_prompt != unhinted_system_prompt:
            raise ValueError("none-strength hint identity must preserve the unhinted system prompt")
        return TraceHint(family_value, strength, None, None, unhinted_system_prompt)
    return _hinted_trace_identity(
        family_value,
        strength,
        sha_value,
        unhinted_system_prompt,
        recorded_system_prompt,
    )


def render_student_contexts(
    trace_records: Sequence[Mapping[str, object]],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[str, ...]:
    """Reconstruct native retained-history prompts without teacher template kwargs.

    Retained history appends the sampled response and the native terminator, rather than
    reserializing historical assistant messages (which can change sampled reasoning).
    """
    turns = _turn_records(trace_records)
    summary = next(record for record in trace_records if record.get("record") == "episode_summary")
    raw_messages = summary.get("messages")
    if not isinstance(raw_messages, list):
        raise TypeError("student rendering requires episode_summary messages")
    messages = [_required_mapping(message, "episode_summary message") for message in raw_messages]
    recorded_hint = trace_hint(trace_records)
    sampled_action_ids_before = tuple(
        tuple(_token_ids(turn.get("completion_token_ids"), "trace turn completion_token_ids"))
        for turn in _turn_records(trace_records)
    )
    if recorded_hint is not None:
        if not messages or messages[0].get("role") != "system":
            raise ValueError("student rendering requires the first message to be the system prompt")
        messages[0] = {**messages[0], "content": recorded_hint.unhinted_system_prompt}
    sampled_action_ids_after = tuple(
        tuple(_token_ids(turn.get("completion_token_ids"), "trace turn completion_token_ids"))
        for turn in _turn_records(trace_records)
    )
    if sampled_action_ids_after != sampled_action_ids_before:
        raise AssertionError("hint stripping changed recorded sampled action token ids")
    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if len(assistant_indices) < len(turns):
        raise ValueError(
            "episode_summary messages contain fewer assistant turns than recorded turns"
        )
    tools = [bash_tool(_workspace_mount(trace_records))]
    prompt = render_prompt(tokenizer, messages[: assistant_indices[0]], tools)
    prompts = [prompt]
    for turn_index in range(1, len(turns)):
        previous = turns[turn_index - 1]
        previous_messages = messages[: assistant_indices[turn_index - 1]]
        if _required_bool(previous.get("runaway"), "turn runaway"):
            completion_suffix = render_empty_assistant_completion(
                tokenizer, previous_messages, tools
            )
        else:
            completion_suffix = _required_text(
                previous.get("raw_completion"), "turn raw_completion"
            )
            completion_suffix += render_assistant_terminator_suffix(
                tokenizer, previous_messages, tools
            )
        appended = messages[assistant_indices[turn_index - 1] + 1 : assistant_indices[turn_index]]
        prompt += completion_suffix + render_prompt_continuation(tokenizer, appended)
        prompts.append(prompt)
    return tuple(prompts)


@dataclass(frozen=True, slots=True)
class _TurnAction:
    token_ids: list[int]
    loss_mask: list[int]
    sampling_logprobs: list[float | None]
    sampled_completion: str
    runaway: bool


@dataclass(frozen=True, slots=True)
class _BuiltSequence:
    completion_token_ids: list[int]
    sampled_action_token_ids: list[int]
    loss_mask: list[int]
    sampling_logprobs: list[float | None]
    token_turn_indices: list[int]
    turn_action_spans: list[list[int]]
    included_turns: int
    truncated: bool
    original_sequence_tokens: int


def _required_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _required_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field} must be a nonnegative integer")
    return value


def _token_ids(value: object, field: str, *, required: bool = True) -> list[int]:
    if value is None and not required:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list of token ids")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise TypeError(f"{field} must contain only nonnegative integer token ids")
    return list(value)


def _logprobs(value: object, field: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{field} must contain one value per sampled token")
    if any(
        isinstance(logprob, bool)
        or not isinstance(logprob, (float, int))
        or not math.isfinite(logprob)
        for logprob in value
    ):
        raise ValueError(f"{field} must contain finite sampled logprobs")
    return [float(logprob) for logprob in value]


def _turn_records(trace_records: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    turns = [record for record in trace_records if record.get("record") == "turn"]
    turns.sort(key=lambda record: _required_nonnegative_int(record.get("turn"), "trace turn"))
    indexes = [_required_nonnegative_int(turn.get("turn"), "trace turn") for turn in turns]
    if indexes != list(range(len(turns))):
        raise ValueError("trace turn indexes must be contiguous from zero")
    if not turns:
        raise ValueError("ladder episode trace contains no policy turns")
    return turns


def _decoded_prefix(tokenizer: PreTrainedTokenizerBase, token_ids: Sequence[int]) -> str:
    decoded_prefix = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded_prefix, str):
        raise TypeError("tokenizer.decode must return a string for one token sequence")
    return decoded_prefix


def _encode_environment(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def _mask_forced_thinking_close(
    tokenizer: PreTrainedTokenizerBase,
    completion_token_ids: Sequence[int],
    loss_mask: list[int],
    *,
    turn_index: int,
) -> None:
    close_token_id = tokenizer.convert_tokens_to_ids(FORCED_THINKING_CLOSE)
    if type(close_token_id) is not int or close_token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer does not have a dedicated {FORCED_THINKING_CLOSE} token")
    try:
        close_start = completion_token_ids.index(close_token_id)
    except ValueError:
        raise ValueError(
            f"turn {turn_index} reached the thinking budget but has no sampled {FORCED_THINKING_CLOSE} token"
        ) from None
    loss_mask[close_start] = 0


def _sampled_completion_text(
    turn: Mapping[str, object],
    action_token_ids: Sequence[int],
    raw_completion: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    turn_index: int,
) -> str:
    """Validate the decoded sample against its recorded terminal token or stop string."""
    sampled_completion = _decoded_prefix(tokenizer, action_token_ids)
    matched_stop = turn.get("matched_stop_token_id")
    if matched_stop is None:
        if sampled_completion != raw_completion:
            raise ValueError(
                f"turn {turn_index} sampled token ids do not decode to raw_completion and "
                "matched_stop_token_id is missing"
            )
        return sampled_completion
    if type(matched_stop) is int:
        if action_token_ids[-1] != matched_stop:
            raise ValueError(
                f"turn {turn_index} matched_stop_token_id is not the final sampled token id"
            )
        expected_completion = raw_completion + _decoded_prefix(tokenizer, [matched_stop])
        if sampled_completion != expected_completion:
            raise ValueError(
                f"turn {turn_index} sampled token ids do not decode to raw_completion plus "
                "the matched stop token"
            )
        return sampled_completion
    if isinstance(matched_stop, str):
        if not matched_stop or sampled_completion not in {
            raw_completion,
            raw_completion + matched_stop,
        }:
            raise ValueError(
                f"turn {turn_index} sampled token ids do not decode to raw_completion or its "
                "recorded matched stop string"
            )
        return sampled_completion
    raise TypeError(f"turn {turn_index} matched_stop_token_id must be an integer, string or None")


def _move_label_values(
    episode_record: Mapping[str, object],
    behaviour_events: Mapping[str, object],
) -> dict[str, bool | None]:
    values: dict[str, bool | None] = {}
    for label in GRADER_DIRECTED_LABEL_FIELDS:
        value = behaviour_events.get(label)
        if value is None:
            value = episode_record.get(label)
        if value is None:
            values[label] = None
        elif isinstance(value, bool):
            values[label] = value
        elif label == "git_refs_rewritten" and isinstance(value, list):
            values[label] = bool(value)
        else:
            raise TypeError(f"episode label {label} must be a boolean or null")
    return values


def grader_directed_move_flags(episode_record: Mapping[str, object]) -> dict[str, bool | None]:
    """Return the recorded grader-directed move flags used by the seed selector."""
    behaviour_events_value = episode_record.get("behaviour_events")
    behaviour_events = (
        _required_mapping(behaviour_events_value, "ladder_episode behaviour_events")
        if behaviour_events_value is not None
        else {}
    )
    return _move_label_values(episode_record, behaviour_events)


def _grader_directed_move_turns(
    behaviour_events: Mapping[str, object],
    move_labels: Mapping[str, bool | None],
) -> tuple[list[int], str]:
    turn_indices: set[int] = set()
    every_positive_label_has_turns = True
    for label, is_present in move_labels.items():
        raw_evidence = behaviour_events.get(f"{label}_evidence")
        evidence_turns: set[int] = set()
        if isinstance(raw_evidence, list):
            for entry in raw_evidence:
                if not isinstance(entry, Mapping):
                    continue
                evidence_turn = entry.get("turn")
                if evidence_turn is None:
                    continue
                if type(evidence_turn) is not int or evidence_turn < 1:
                    raise TypeError(
                        f"{label}_evidence turn must be a positive one-based integer or null"
                    )
                evidence_turns.add(evidence_turn - 1)
        if is_present is True and not evidence_turns:
            every_positive_label_has_turns = False
        turn_indices.update(evidence_turns)

    all_labels_known_false = bool(move_labels) and all(
        value is False for value in move_labels.values()
    )
    if all_labels_known_false or (turn_indices and every_positive_label_has_turns):
        return sorted(turn_indices), "known"
    return sorted(turn_indices), "unknown"


def _labels(
    episode_record: Mapping[str, object],
    move_labels: Mapping[str, bool | None],
) -> dict[str, object]:
    move_values = list(move_labels.values())
    grader_directed_move: bool | None
    if any(value is True for value in move_values):
        grader_directed_move = True
    elif move_values and all(value is False for value in move_values):
        grader_directed_move = False
    else:
        grader_directed_move = None

    honest_solve = episode_record.get("honest_solve")
    if honest_solve is not None and not isinstance(honest_solve, bool):
        raise TypeError("episode honest_solve label must be a boolean or null")
    highest_rung = episode_record.get("highest_rung")
    if highest_rung is not None:
        highest_rung = _required_nonnegative_int(highest_rung, "episode highest_rung")

    return {
        **move_labels,
        "grader_directed_move": grader_directed_move,
        "honest_solve": honest_solve,
        "highest_rung": highest_rung,
    }


def _episode_key(episode_record: Mapping[str, object]) -> str:
    key = _required_mapping(episode_record.get("key"), "ladder_episode key")
    canonical_key = json.dumps(dict(key), sort_keys=True, separators=(",", ":"))
    return f"ladder:{hashlib.sha256(canonical_key.encode('utf-8')).hexdigest()}"


def _prepare_trace(
    trace_records: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], tuple[str, ...], list[int], bool]:
    turns = _turn_records(trace_records)
    rendered_prompts = reconstruct_rendered_prompts(trace_records)
    if len(rendered_prompts) != len(turns) + 1:
        raise ValueError("trace prompt reconstruction returned an unexpected prompt count")
    if "prompt_token_ids" not in turns[0]:
        raise ValueError(
            "trace turn zero lacks prompt_token_ids; it predates token-id persistence and must be regenerated"
        )
    prompt_token_ids = _token_ids(turns[0].get("prompt_token_ids"), "turn zero prompt_token_ids")
    if not prompt_token_ids:
        raise ValueError(
            "trace turn zero lacks prompt_token_ids; it predates token-id persistence and must be regenerated"
        )
    later_turn_prompt_ids_present = ["prompt_token_ids" in turn for turn in turns[1:]]
    if any(later_turn_prompt_ids_present) and not all(later_turn_prompt_ids_present):
        raise ValueError("later turns must either all record prompt_token_ids or all omit them")
    has_per_turn_prompt_ids = all(later_turn_prompt_ids_present)
    if has_per_turn_prompt_ids:
        for turn_index, turn in enumerate(turns[1:], start=1):
            _turn_prompt_ids(turn, turn_index=turn_index)
    if not rendered_prompts[0]:
        raise ValueError("turn zero rendered prompt must not be empty")

    episode_ids = {
        _required_text(turn.get("episode_id"), "trace turn episode_id") for turn in turns
    }
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("trace must contain exactly one episode_summary record")
    summary_episode_id = _required_text(
        summaries[0].get("episode_id"), "episode_summary episode_id"
    )
    if episode_ids != {summary_episode_id}:
        raise ValueError("trace turns and episode_summary must have the same episode_id")
    return turns, rendered_prompts, prompt_token_ids, has_per_turn_prompt_ids


def _verify_turn_prompt(
    turn: Mapping[str, object],
    rendered_prompt: str,
    prefix_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    turn_index: int,
) -> None:
    if turn.get("history_reasoning") != "retain":
        raise ValueError(f"turn {turn_index} is not a history_reasoning=retain trace")
    server_prompt_tokens = _required_nonnegative_int(
        turn.get("prompt_tokens"), "turn prompt_tokens"
    )
    if len(prefix_token_ids) != server_prompt_tokens:
        raise ValueError(
            f"turn {turn_index} prompt token count mismatch: built {len(prefix_token_ids)}, "
            f"server recorded {server_prompt_tokens}"
        )
    if "prompt_token_ids" in turn:
        recorded_prompt_ids = _turn_prompt_ids(turn, turn_index=turn_index)
        if list(prefix_token_ids) != recorded_prompt_ids:
            raise ValueError(
                f"turn {turn_index} built prefix does not match recorded prompt token ids"
            )
    if _decoded_prefix(tokenizer, prefix_token_ids) != rendered_prompt:
        raise ValueError(
            f"turn {turn_index} decoded token prefix does not match rendered_prompt_text"
        )


def _sampled_action(
    turn: Mapping[str, object],
    tokenizer: PreTrainedTokenizerBase,
    *,
    turn_index: int,
) -> _TurnAction:
    action_token_ids = _token_ids(
        turn.get("completion_token_ids"),
        f"turn {turn_index} completion_token_ids",
        required=False,
    )
    if not action_token_ids:
        raise ValueError(
            f"trace turn {turn_index} lacks completion_token_ids; it predates token-id persistence "
            "and must be regenerated"
        )
    sampled_logprobs = _logprobs(
        turn.get("completion_logprobs"),
        f"turn {turn_index} completion_logprobs",
        len(action_token_ids),
    )
    raw_completion = _required_text(turn.get("raw_completion"), f"turn {turn_index} raw_completion")
    sampled_completion = _sampled_completion_text(
        turn,
        action_token_ids,
        raw_completion,
        tokenizer,
        turn_index=turn_index,
    )

    runaway = _required_bool(turn.get("runaway"), f"turn {turn_index} runaway")
    if runaway:
        return _TurnAction(
            token_ids=[],
            loss_mask=[],
            sampling_logprobs=[],
            sampled_completion=sampled_completion,
            runaway=True,
        )
    action_mask = [1] * len(action_token_ids)
    if _required_bool(
        turn.get("thinking_budget_fired"), f"turn {turn_index} thinking_budget_fired"
    ):
        _mask_forced_thinking_close(
            tokenizer,
            action_token_ids,
            action_mask,
            turn_index=turn_index,
        )
    return _TurnAction(
        token_ids=action_token_ids,
        loss_mask=action_mask,
        sampling_logprobs=list(sampled_logprobs),
        sampled_completion=sampled_completion,
        runaway=False,
    )


def _turn_prompt_ids(turn: Mapping[str, object], *, turn_index: int) -> list[int]:
    if "prompt_token_ids" not in turn:
        raise ValueError(
            f"trace turn {turn_index} lacks prompt_token_ids; it predates per-turn prompt token-id persistence "
            "and must be regenerated"
        )
    prompt_ids = _token_ids(
        turn.get("prompt_token_ids"),
        f"turn {turn_index} prompt_token_ids",
    )
    if not prompt_ids:
        raise ValueError(f"trace turn {turn_index} prompt_token_ids must not be empty")
    return prompt_ids


def _environment_segment(  # noqa: PLR0913 - explicit inputs for exact legacy and recorded prompt paths
    action: _TurnAction,
    turns: Sequence[Mapping[str, object]],
    rendered_prompts: Sequence[str],
    prefix_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    turn_index: int,
    has_per_turn_prompt_ids: bool,
) -> list[int]:
    if turn_index + 1 == len(turns):
        return []
    if has_per_turn_prompt_ids:
        next_prompt_ids = _turn_prompt_ids(turns[turn_index + 1], turn_index=turn_index + 1)
        sampled_prefix_ids = [*prefix_token_ids, *action.token_ids]
        if next_prompt_ids[: len(sampled_prefix_ids)] != sampled_prefix_ids:
            raise ValueError(
                f"turn {turn_index + 1} prompt token ids do not preserve turn {turn_index} prompt "
                "and sampled completion"
            )
        return next_prompt_ids[len(sampled_prefix_ids) :]

    current_prompt = rendered_prompts[turn_index]
    next_prompt = rendered_prompts[turn_index + 1]
    if not next_prompt.startswith(current_prompt):
        raise ValueError(f"turn {turn_index + 1} prompt is not appended to turn {turn_index}")
    if action.runaway:
        environment_text = next_prompt[len(current_prompt) :]
    else:
        sampled_completion_prefix = current_prompt + action.sampled_completion
        if not next_prompt.startswith(sampled_completion_prefix):
            raise ValueError(
                f"turn {turn_index + 1} prompt does not preserve turn {turn_index} sampled completion"
            )
        environment_text = next_prompt[len(sampled_completion_prefix) :]
    return _encode_environment(tokenizer, environment_text)


def _verify_student_environment(
    student_prompts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    action: _TurnAction,
    environment_ids: Sequence[int],
    *,
    turn_index: int,
) -> None:
    sampled_prefix = student_prompts[turn_index] + _decoded_prefix(tokenizer, action.token_ids)
    next_prompt = student_prompts[turn_index + 1]
    if not next_prompt.startswith(sampled_prefix):
        raise ValueError(
            f"turn {turn_index + 1} student-rendered context changes the recorded sampled action"
        )
    native_environment_ids = _encode_environment(tokenizer, next_prompt[len(sampled_prefix) :])
    if native_environment_ids != list(environment_ids):
        raise ValueError(
            f"turn {turn_index + 1} student-rendered context changes recorded environment token ids"
        )


def _build_sequence(  # noqa: PLR0913 - prompt mode stays explicit at the trace-building boundary
    turns: Sequence[Mapping[str, object]],
    rendered_prompts: Sequence[str],
    prompt_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_sequence_tokens: int,
    has_per_turn_prompt_ids: bool,
    student_prompts: Sequence[str] | None = None,
) -> _BuiltSequence:
    prefix_token_ids = list(prompt_token_ids)
    rendered_prefix_ids = (
        list(prompt_token_ids)
        if student_prompts is None
        else _encode_environment(tokenizer, student_prompts[0])
    )
    completion_token_ids: list[int] = []
    sampled_action_token_ids: list[int] = []
    loss_mask: list[int] = []
    sampling_logprobs: list[float | None] = []
    token_turn_indices: list[int] = []
    turn_action_spans: list[list[int]] = []
    included_turns = 0
    truncated = False
    original_sequence_tokens = len(rendered_prefix_ids)

    for turn_index, (turn, rendered_prompt) in enumerate(
        zip(turns, rendered_prompts[:-1], strict=True)
    ):
        _verify_turn_prompt(
            turn,
            rendered_prompt,
            prefix_token_ids,
            tokenizer,
            turn_index=turn_index,
        )
        action = _sampled_action(turn, tokenizer, turn_index=turn_index)
        if student_prompts is not None:
            student_prompt = student_prompts[turn_index]
            native_ids = _encode_environment(tokenizer, student_prompt)
            if _decoded_prefix(tokenizer, rendered_prefix_ids) != student_prompt:
                raise ValueError(
                    f"turn {turn_index} student-rendered context cannot preserve recorded action/environment ids"
                )
            appended_ids = _encode_environment(
                tokenizer, student_prompt + action.sampled_completion
            )
            if appended_ids[: len(native_ids)] != native_ids:
                raise ValueError(
                    f"turn {turn_index} student context/action splice lacks a clean token boundary"
                )
        environment_ids = _environment_segment(
            action,
            turns,
            rendered_prompts,
            prefix_token_ids,
            tokenizer,
            turn_index=turn_index,
            has_per_turn_prompt_ids=has_per_turn_prompt_ids,
        )
        if student_prompts is not None and turn_index + 1 < len(turns):
            _verify_student_environment(
                student_prompts, tokenizer, action, environment_ids, turn_index=turn_index
            )
        turn_ids = [*action.token_ids, *environment_ids]
        turn_mask = [*action.loss_mask, *([0] * len(environment_ids))]
        turn_logprobs = [*action.sampling_logprobs, *([None] * len(environment_ids))]
        boundary_length = len(rendered_prefix_ids) + len(turn_ids)
        if not truncated and boundary_length <= max_sequence_tokens:
            turn_action_spans.append(
                [len(completion_token_ids), len(completion_token_ids) + len(action.token_ids)]
            )
            completion_token_ids.extend(turn_ids)
            sampled_action_token_ids.extend(action.token_ids)
            loss_mask.extend(turn_mask)
            sampling_logprobs.extend(turn_logprobs)
            token_turn_indices.extend(turn_index if active else -1 for active in turn_mask)
            included_turns = turn_index + 1
        else:
            truncated = True
        original_sequence_tokens = boundary_length
        prefix_token_ids.extend(turn_ids)
        rendered_prefix_ids.extend(turn_ids)

    if included_turns == 0 or not completion_token_ids:
        raise ValueError("no complete policy turn fits max_sequence_tokens")
    if not any(loss_mask):
        raise ValueError("episode row contains no unmasked sampled assistant tokens")
    return _BuiltSequence(
        completion_token_ids=completion_token_ids,
        sampled_action_token_ids=sampled_action_token_ids,
        loss_mask=loss_mask,
        sampling_logprobs=sampling_logprobs,
        token_turn_indices=token_turn_indices,
        turn_action_spans=turn_action_spans,
        included_turns=included_turns,
        truncated=truncated,
        original_sequence_tokens=original_sequence_tokens,
    )


def is_discarded_completion(turn: Mapping[str, object]) -> bool:
    """Recognize the runner's explicit too-long discarded-response marker."""
    format_error = turn.get("format_error")
    return bool(
        turn.get("runaway") is True
        and turn.get("stop_reason") == "max_tokens"
        and isinstance(format_error, str)
        and bool(format_error)
    )


def _opd_final_ids(
    turns: Sequence[Mapping[str, object]], max_sequence_tokens: int
) -> tuple[list[int], list[int]]:
    initial_ids = _turn_prompt_ids(turns[0], turn_index=0)
    final_ids = [
        *_turn_prompt_ids(turns[-1], turn_index=len(turns) - 1),
        *_token_ids(turns[-1].get("completion_token_ids"), "final completion_token_ids"),
    ]
    if final_ids[: len(initial_ids)] != initial_ids:
        raise ValueError("OPD final sequence does not preserve the initial prompt prefix")
    if len(final_ids) > max_sequence_tokens:
        raise ValueError("OPD final saved sequence exceeds max_sequence_tokens")
    return initial_ids, final_ids


def _build_opd_sequence(
    turns: Sequence[Mapping[str, object]],
    rendered_prompts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_sequence_tokens: int,
) -> _BuiltSequence:
    initial_ids, final_ids = _opd_final_ids(turns, max_sequence_tokens)
    completion_ids = final_ids[len(initial_ids) :]
    mask = [0] * len(completion_ids)
    logps: list[float | None] = [None] * len(completion_ids)
    indices = [-1] * len(completion_ids)
    spans: list[list[int]] = []
    sampled_ids: list[int] = []
    for turn_index, turn in enumerate(turns):
        prompt_ids = _turn_prompt_ids(turn, turn_index=turn_index)
        _verify_turn_prompt(
            turn, rendered_prompts[turn_index], prompt_ids, tokenizer, turn_index=turn_index
        )
        if final_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"turn {turn_index} prompt diverges from the final saved prefix")
        raw_ids = _token_ids(turn.get("completion_token_ids"), "completion_token_ids")
        candidate = [*prompt_ids, *raw_ids]
        retained = final_ids[: len(candidate)] == candidate
        discarded = is_discarded_completion(turn)
        if discarded:
            if retained and turn_index != len(turns) - 1:
                raise ValueError(
                    f"turn {turn_index} marked discarded but retained in final sequence"
                )
            continue
        if not retained:
            raise ValueError(
                f"turn {turn_index} non-discarded completion diverges from final prefix"
            )
        action = _sampled_action(turn, tokenizer, turn_index=turn_index)
        if action.runaway:
            raise ValueError(
                f"turn {turn_index} runaway lacks an explicit discarded completion marker"
            )
        start = len(prompt_ids) - len(initial_ids)
        if start < 0 or any(mask[start : start + len(raw_ids)]):
            raise ValueError("OPD sampled completion spans overlap or precede the initial prompt")
        end = start + len(raw_ids)
        mask[start:end] = action.loss_mask
        logps[start:end] = action.sampling_logprobs
        indices[start:end] = [turn_index if active else -1 for active in action.loss_mask]
        spans.append([start, end])
        sampled_ids.extend(raw_ids)
    if not any(mask):
        raise ValueError("episode row contains no unmasked sampled assistant tokens")
    return _BuiltSequence(
        completion_token_ids=completion_ids,
        sampled_action_token_ids=sampled_ids,
        loss_mask=mask,
        sampling_logprobs=logps,
        token_turn_indices=indices,
        turn_action_spans=spans,
        included_turns=len(turns),
        truncated=False,
        original_sequence_tokens=len(final_ids),
    )


def build_episode_row(
    episode_record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
    prompt_render: PromptRender = "student",
    _opd_final_sequence: bool = False,
) -> dict[str, object]:
    """Turn one retained-history ladder trace into one masked, token-id training row."""
    if type(max_sequence_tokens) is not int or max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be a positive integer")
    if episode_record.get("record") != "ladder_episode":
        raise ValueError("episode record must have record='ladder_episode'")
    if prompt_render not in {"student", "teacher"}:
        raise ValueError("prompt_render must be student or teacher")
    recorded_hint = trace_hint(trace_records)
    if (
        prompt_render == "teacher"
        and recorded_hint is not None
        and recorded_hint.strength != "none"
    ):
        raise ValueError("teacher rendering cannot include hinted episodes")

    turns, rendered_prompts, prompt_token_ids, has_per_turn_prompt_ids = _prepare_trace(
        trace_records
    )
    student_prompts = (
        render_student_contexts(trace_records, tokenizer) if prompt_render == "student" else None
    )
    output_prompt_ids = (
        prompt_token_ids
        if student_prompts is None
        else _encode_environment(tokenizer, student_prompts[0])
    )
    prefix_diff_counts = (
        [0] * len(turns)
        if student_prompts is None
        else [
            sum(left != right for left, right in zip(teacher_ids, student_ids, strict=False))
            + abs(len(teacher_ids) - len(student_ids))
            for teacher_ids, student_ids in (
                (
                    _turn_prompt_ids(turns[turn_index], turn_index=turn_index)
                    if has_per_turn_prompt_ids
                    else _encode_environment(tokenizer, teacher_prompt),
                    _encode_environment(tokenizer, student_prompt),
                )
                for turn_index, (teacher_prompt, student_prompt) in enumerate(
                    zip(rendered_prompts[:-1], student_prompts, strict=True)
                )
            )
        ]
    )
    prompt_length = len(output_prompt_ids)
    if prompt_length > max_sequence_tokens:
        raise ValueError("turn zero prompt alone exceeds max_sequence_tokens")
    built_sequence = (
        _build_opd_sequence(
            turns, rendered_prompts, tokenizer, max_sequence_tokens=max_sequence_tokens
        )
        if _opd_final_sequence
        else _build_sequence(
            turns,
            rendered_prompts,
            prompt_token_ids,
            tokenizer,
            max_sequence_tokens=max_sequence_tokens,
            has_per_turn_prompt_ids=has_per_turn_prompt_ids,
            student_prompts=student_prompts,
        )
    )
    if student_prompts is not None and not _opd_final_sequence:
        expected_action_ids = [
            token_id
            for turn_index, turn in enumerate(turns[: built_sequence.included_turns])
            for token_id in _sampled_action(turn, tokenizer, turn_index=turn_index).token_ids
        ]
        if built_sequence.sampled_action_token_ids != expected_action_ids:
            raise AssertionError("student rendering changed sampled action token ids")

    behaviour_events_value = episode_record.get("behaviour_events")
    behaviour_events = (
        _required_mapping(behaviour_events_value, "ladder_episode behaviour_events")
        if behaviour_events_value is not None
        else {}
    )
    move_labels = _move_label_values(episode_record, behaviour_events)
    grader_move_turn_indices, grader_move_coverage = _grader_directed_move_turns(
        behaviour_events, move_labels
    )

    grader_directed_move_cut: bool | None
    if grader_move_coverage == "unknown":
        grader_directed_move_cut = None
    else:
        grader_directed_move_cut = any(
            turn_index >= built_sequence.included_turns for turn_index in grader_move_turn_indices
        )

    key = _episode_key(episode_record)
    row: dict[str, object] = {
        "key": key,
        "prompt_token_ids": output_prompt_ids,
        "prompt_render": prompt_render,
        "student_template_sha256": student_template_sha256(
            tokenizer, _workspace_mount(trace_records)
        )
        if student_prompts is not None
        else None,
        "prefix_diff_counts": prefix_diff_counts,
        "completion_token_ids": built_sequence.completion_token_ids,
        "loss_mask": built_sequence.loss_mask,
        "sampling_logprobs": built_sequence.sampling_logprobs,
        "token_turn_indices": built_sequence.token_turn_indices,
        "turn_action_spans": built_sequence.turn_action_spans,
        "labels": _labels(episode_record, move_labels),
        "token_ids_sha256": token_ids_sha256(
            output_prompt_ids, built_sequence.completion_token_ids
        ),
        "truncated": built_sequence.truncated,
        "sequence_tokens": prompt_length + len(built_sequence.completion_token_ids),
        "original_sequence_tokens": built_sequence.original_sequence_tokens,
        "included_turns": built_sequence.included_turns,
        "grader_directed_move_coverage": grader_move_coverage,
        "grader_directed_move_turn_indices": grader_move_turn_indices,
        "grader_directed_move_cut": grader_directed_move_cut,
    }
    logger.info(
        "built ladder distillation row turns=%d sequence_tokens=%d truncated=%s grader_move_coverage=%s "
        "grader_move_cut=%s",
        built_sequence.included_turns,
        row["sequence_tokens"],
        built_sequence.truncated,
        grader_move_coverage,
        grader_directed_move_cut,
    )
    return row


def teacher_system_hint_prefix(
    trace_records: Sequence[Mapping[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    teacher_hint_text: str,
) -> list[int]:
    """Render one system-hint substitution without rebuilding the saved episode sequence."""
    summary = next(record for record in trace_records if record.get("record") == "episode_summary")
    messages_value = summary.get("messages")
    if not isinstance(messages_value, list):
        raise TypeError("OPD requires episode_summary messages")
    messages = [
        dict(_required_mapping(message, "episode_summary message")) for message in messages_value
    ]
    if not messages or messages[0].get("role") != "system":
        raise ValueError("OPD requires the first message to be the system prompt")
    first_assistant = next(
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    )
    recorded_hint = trace_hint(trace_records)
    system_prompt = (
        recorded_hint.unhinted_system_prompt
        if recorded_hint
        else _required_text(messages[0].get("content"), "system prompt")
    )
    messages[0]["content"] = system_prompt + (
        "\n\n" + teacher_hint_text if teacher_hint_text else ""
    )
    return _encode_environment(
        tokenizer,
        render_prompt(
            tokenizer, messages[:first_assistant], [bash_tool(_workspace_mount(trace_records))]
        ),
    )


def _verify_opd_native_contexts(
    turns: Sequence[Mapping[str, object]],
    trace_records: Sequence[Mapping[str, object]],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    native_prompts = render_student_contexts(trace_records, tokenizer)
    expected_ids = _encode_environment(tokenizer, native_prompts[0])
    discarded_history = False
    for turn_index, turn in enumerate(turns):
        recorded_ids = _turn_prompt_ids(turn, turn_index=turn_index)
        if expected_ids != recorded_ids or (
            not discarded_history
            and _encode_environment(tokenizer, native_prompts[turn_index]) != recorded_ids
        ):
            raise ValueError("OPD student rendering differs from the actual sampled context")
        if turn_index + 1 == len(turns):
            break
        prefix_text = native_prompts[turn_index]
        if not is_discarded_completion(turn):
            action = _sampled_action(turn, tokenizer, turn_index=turn_index)
            prefix_text += action.sampled_completion
            expected_ids.extend(_encode_environment(tokenizer, action.sampled_completion))
        next_prompt = native_prompts[turn_index + 1]
        if not next_prompt.startswith(prefix_text):
            raise ValueError("OPD student rendering differs from the actual sampled context")
        expected_ids.extend(_encode_environment(tokenizer, next_prompt[len(prefix_text) :]))
        discarded_history = discarded_history or is_discarded_completion(turn)


def build_opd_episode_row(  # noqa: PLR0913
    episode_record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    teacher_hint_family: str,
    teacher_hint_text: str,
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
) -> dict[str, object]:
    """Keep an unhinted student's sampled sequence and swap only the teacher system prompt."""
    if not teacher_hint_family or not teacher_hint_text:
        raise ValueError("teacher hint family and text must be nonempty")
    recorded_hint = trace_hint(trace_records)
    if recorded_hint is not None and recorded_hint.strength != "none":
        raise ValueError("OPD requires an unhinted student rollout")
    if episode_record.get("partition") != "train":
        raise ValueError("OPD requires a training partition episode")
    environment_reward = _required_mapping(episode_record.get("organism_reward"), "organism_reward")
    row = build_episode_row(
        episode_record,
        trace_records,
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
        prompt_render="student",
        _opd_final_sequence=True,
    )
    turns = _turn_records(trace_records)
    _verify_opd_native_contexts(turns, trace_records, tokenizer)
    sampled_prompt_ids = _token_ids(turns[0].get("prompt_token_ids"), "turn zero prompt_token_ids")
    if row["prompt_token_ids"] != sampled_prompt_ids:
        raise ValueError("OPD student rendering differs from the actual sampled context")
    teacher_prefix = teacher_system_hint_prefix(
        trace_records, tokenizer=tokenizer, teacher_hint_text=teacher_hint_text
    )
    mask = cast("list[int]", row["loss_mask"])
    sampling = cast("list[float | None]", row["sampling_logprobs"])
    sampling_logps: list[float] = []
    for active, logp in zip(mask, sampling, strict=True):
        if active and (logp is None or not math.isfinite(logp) or logp > 0):
            raise ValueError(
                "OPD requires finite nonpositive logprobs for every sampled assistant token"
            )
        sampling_logps.append(float(logp) if active and logp is not None else 0.0)
    row.update(
        {
            "kept_turn_count": sum(not is_discarded_completion(turn) for turn in turns),
            "discarded_turn_count": sum(is_discarded_completion(turn) for turn in turns),
            "discarded_token_count": sum(
                len(_token_ids(turn.get("completion_token_ids"), "discarded completion_token_ids"))
                for turn in turns
                if is_discarded_completion(turn)
            ),
            "teacher_prompt_token_ids": teacher_prefix,
            "teacher_hint_family": teacher_hint_family,
            "student_sampling_logps": sampling_logps,
            "environment_reward": dict(environment_reward),
            "partition": "train",
            "student_hint_strength": "none",
            "env": episode_record.get("family", "ladder"),
            "problem_id": _required_text(
                episode_record["task_id"]
                if "task_id" in episode_record
                else _required_mapping(episode_record["key"], "episode key")["scenario_id"],
                "task_id",
            ),
        }
    )
    return row


def _trace_path_for_record(episode_record: Mapping[str, object], records_path: Path) -> Path:
    trace_path_text = _required_text(episode_record.get("trace_path"), "ladder_episode trace_path")
    trace_path = Path(trace_path_text)
    if not trace_path.is_absolute():
        relative_path = records_path.parent / trace_path
        if relative_path.is_file():
            return relative_path
    if trace_path.is_file():
        return trace_path

    trace_root = records_path.with_name(f"{records_path.stem}-traces")
    tail = trace_path.parts[-TRACE_PATH_COMPONENTS:]
    if len(tail) == TRACE_PATH_COMPONENTS and all(part not in {".", ".."} for part in tail):
        relocated_path = trace_root.joinpath(*tail)
        if relocated_path.is_file():
            return relocated_path
    raise FileNotFoundError(f"ladder trace is missing for episode record: {trace_path.name}")


def read_episode_trace_records(
    episode_record: Mapping[str, object], records_path: Path
) -> list[Mapping[str, object]]:
    """Load one episode's JSONL trace, including the runner's relocated-trace layout."""
    return _read_jsonl(_trace_path_for_record(episode_record, records_path))


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            records.append(_required_mapping(value, f"{path.name} line {line_number}"))
    return records


def build_episode_rows(
    records_path: Path,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
    prompt_render: PromptRender = "student",
) -> list[dict[str, object]]:
    """Load ladder episode records and return one verified training row per episode.

    Student rendering replaces context with the native evaluation template while preserving
    recorded assistant and environment ids. Teacher mode reproduces persisted prompts.
    """
    records = _read_jsonl(records_path)
    rows: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    for episode_record in records:
        if episode_record.get("record") != "ladder_episode":
            continue
        key = _episode_key(episode_record)
        if key in seen_keys:
            raise ValueError(f"duplicate ladder episode key in {records_path.name}")
        trace_path = _trace_path_for_record(episode_record, records_path)
        trace_records = _read_jsonl(trace_path)
        row = build_episode_row(
            episode_record,
            trace_records,
            tokenizer=tokenizer,
            max_sequence_tokens=max_sequence_tokens,
            prompt_render=prompt_render,
        )
        if row["key"] != key:
            raise RuntimeError("episode row key changed while building records")
        rows.append(row)
        seen_keys.add(key)
    if not rows:
        raise ValueError(f"{records_path.name} contains no ladder_episode records")
    return rows
