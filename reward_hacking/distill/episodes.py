"""Build masked distillation rows from token-persisted ladder episode traces."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.ladder.prompt_trace import reconstruct_rendered_prompts

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
    loss_mask: list[int]
    sampling_logprobs: list[float | None]
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


def _encode_environment(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def _decoded_prefix(tokenizer: PreTrainedTokenizerBase, token_ids: Sequence[int]) -> str:
    decoded_prefix = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded_prefix, str):
        raise TypeError("tokenizer.decode must return a string for one token sequence")
    return decoded_prefix


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
) -> tuple[list[Mapping[str, object]], tuple[str, ...], list[int]]:
    turns = _turn_records(trace_records)
    rendered_prompts = reconstruct_rendered_prompts(trace_records)
    if len(rendered_prompts) != len(turns) + 1:
        raise ValueError("trace prompt reconstruction returned an unexpected prompt count")
    prompt_token_ids = _token_ids(
        turns[0].get("prompt_token_ids"), "turn zero prompt_token_ids", required=False
    )
    if not prompt_token_ids:
        raise ValueError(
            "trace turn zero lacks prompt_token_ids; it predates token-id persistence and must be regenerated"
        )
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
    return turns, rendered_prompts, prompt_token_ids


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


def _environment_segment(
    action: _TurnAction,
    rendered_prompts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    turn_index: int,
) -> list[int]:
    if turn_index + 1 == len(rendered_prompts) - 1:
        return []
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


def _build_sequence(
    turns: Sequence[Mapping[str, object]],
    rendered_prompts: Sequence[str],
    prompt_token_ids: Sequence[int],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_sequence_tokens: int,
) -> _BuiltSequence:
    prefix_token_ids = list(prompt_token_ids)
    completion_token_ids: list[int] = []
    loss_mask: list[int] = []
    sampling_logprobs: list[float | None] = []
    included_turns = 0
    truncated = False
    original_sequence_tokens = len(prefix_token_ids)

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
        environment_ids = _environment_segment(
            action,
            rendered_prompts,
            tokenizer,
            turn_index=turn_index,
        )
        turn_ids = [*action.token_ids, *environment_ids]
        turn_mask = [*action.loss_mask, *([0] * len(environment_ids))]
        turn_logprobs = [*action.sampling_logprobs, *([None] * len(environment_ids))]
        boundary_length = len(prefix_token_ids) + len(turn_ids)
        if not truncated and boundary_length <= max_sequence_tokens:
            completion_token_ids.extend(turn_ids)
            loss_mask.extend(turn_mask)
            sampling_logprobs.extend(turn_logprobs)
            included_turns = turn_index + 1
        else:
            truncated = True
        original_sequence_tokens = boundary_length
        prefix_token_ids.extend(turn_ids)

    if included_turns == 0 or not completion_token_ids:
        raise ValueError("no complete policy turn fits max_sequence_tokens")
    if not any(loss_mask):
        raise ValueError("episode row contains no unmasked sampled assistant tokens")
    return _BuiltSequence(
        completion_token_ids=completion_token_ids,
        loss_mask=loss_mask,
        sampling_logprobs=sampling_logprobs,
        included_turns=included_turns,
        truncated=truncated,
        original_sequence_tokens=original_sequence_tokens,
    )


def build_episode_row(
    episode_record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
) -> dict[str, object]:
    """Turn one retained-history ladder trace into one masked, token-id training row."""
    if type(max_sequence_tokens) is not int or max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be a positive integer")
    if episode_record.get("record") != "ladder_episode":
        raise ValueError("episode record must have record='ladder_episode'")

    turns, rendered_prompts, prompt_token_ids = _prepare_trace(trace_records)
    prompt_length = len(prompt_token_ids)
    if prompt_length > max_sequence_tokens:
        raise ValueError("turn zero prompt alone exceeds max_sequence_tokens")
    built_sequence = _build_sequence(
        turns,
        rendered_prompts,
        prompt_token_ids,
        tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )

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
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": built_sequence.completion_token_ids,
        "loss_mask": built_sequence.loss_mask,
        "sampling_logprobs": built_sequence.sampling_logprobs,
        "labels": _labels(episode_record, move_labels),
        "token_ids_sha256": token_ids_sha256(prompt_token_ids, built_sequence.completion_token_ids),
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


def _trace_path_for_record(episode_record: Mapping[str, object], records_path: Path) -> Path:
    trace_path_text = _required_text(episode_record.get("trace_path"), "ladder_episode trace_path")
    trace_path = Path(trace_path_text)
    if trace_path.is_file():
        return trace_path

    trace_root = records_path.with_name(f"{records_path.stem}-traces")
    tail = trace_path.parts[-TRACE_PATH_COMPONENTS:]
    if len(tail) == TRACE_PATH_COMPONENTS and all(part not in {".", ".."} for part in tail):
        relocated_path = trace_root.joinpath(*tail)
        if relocated_path.is_file():
            return relocated_path
    raise FileNotFoundError(f"ladder trace is missing for episode record: {trace_path.name}")


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
) -> list[dict[str, object]]:
    """Load ladder episode records and return one verified training row per episode.

    Pass the Qwen3.5-9B tokenizer so only environment-appended text is retokenized. Sampled
    assistant tokens and the turn-zero prompt always come directly from their persisted ids.
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
        )
        if row["key"] != key:
            raise RuntimeError("episode row key changed while building records")
        rows.append(row)
        seen_keys.add(key)
    if not rows:
        raise ValueError(f"{records_path.name} contains no ladder_episode records")
    return rows
