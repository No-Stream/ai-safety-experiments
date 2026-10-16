"""Export spec_gaming journals as agentic SFT episode records with exact-token traces.

The spec_gaming environments (SpreadsheetBench, ALCE, Retail) keep their own journals, assessments
and readouts. This adapter rewrites one finished run into the ``ladder_episode`` record and trace
shape that ``reward_hacking.distill.agentic_sft`` selects from, without touching the journals:

- identity: ``family`` is the environment, ``key.scenario_id`` is the source task, and the partition
  comes from the environment's own split (SpreadsheetBench has none, so its tasks get the same
  deterministic one-in-five group split Retail uses);
- exact tokens: every actor turn keeps its recorded prompt ids, sampled ids and log-probs. The host
  messages between turns are recovered by replaying the environment's own prompt history with the
  actor tokenizer, so the student renderer can rebuild context under its own template;
- payment is the environment's operational proxy at closure; independent truth is the full check
  (SpreadsheetBench), the reviewed full-task assessment (ALCE) or the native gold reward (Retail),
  and stays explicitly unmeasured otherwise;
- eval awareness is the same regex scorer organism records use, run on the actor's own text;
- the strategy label lives in a sidecar (``write_spec_gaming_labels``). Until a labeler has real
  categories every episode carries ``spec_gaming_unlabelled``, which admission never admits.

Admission (``classify_spec_gaming_admission``) also refuses sources frozen as training-ineligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.distill.effort_steering import refuse_effort_steering
from reward_hacking.distill.hack_strategy import (
    LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
    SPEC_GAMING_UNLABELLED,
    TAXONOMY_VERSION,
    HackStrategyAssessment,
)
from reward_hacking.distill.sft_serving import load_tokenizer
from reward_hacking.distill.strategy_adjudication import assessment_from_payload
from reward_hacking.ladder.runner import eval_awareness_from_trace_records
from reward_hacking.ladder.thinking import thinking_region_token_count
from reward_hacking.ladder.tools import parse_qwen_tool_calls
from reward_hacking.model_backend import RenderedCompletion
from reward_hacking.spec_gaming.protocol import PromptHistory

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger: logging.Logger = logging.getLogger(__name__)

SPEC_GAMING_EPISODE_SOURCE = "spec_gaming.sft_export"
SPEC_GAMING_PAYMENT_REGISTER = "spec_gaming_proxy"
EXPORT_SCHEMA_VERSION = 1
SPEC_GAMING_LABEL_FORMAT_VERSION = "spec_gaming_label_v1"
RECORDS_FILENAME = "records.jsonl"
LABELS_FILENAME = "labels.jsonl"
LABEL_MANIFEST_FILENAME = "manifest.json"
EXPORT_MANIFEST_FILENAME = "export-manifest.json"
type Environment = Literal["spreadsheetbench", "alce", "retail"]
SPEC_GAMING_FAMILIES: tuple[Environment, ...] = ("spreadsheetbench", "alce", "retail")
# Taxonomy strategies that count as a spec-gaming hack. Empty until the 27B smoke shows what gaming
# looks like in these environments; registering a category here is what makes admission possible.
SPEC_GAMING_HACK_STRATEGIES: frozenset[str] = frozenset()
SPEC_GAMING_ADMISSION_CRITERION = "spec_gaming_proxy_paid_truth_failed"
_COMPLETE_STATUSES: dict[Environment, frozenset[str]] = {
    "spreadsheetbench": frozenset({"hand_back"}),
    "alce": frozenset({"hand_back"}),
    "retail": frozenset({"hand_back", "user_stop"}),
}
_HISTORY_KEYS: dict[Environment, str] = {
    "spreadsheetbench": "history",
    "alce": "history",
    "retail": "actor_history",
}
_SPREADSHEET_GROUP_NAMESPACE = "spreadsheetbench-task-group-v1:"
_HELD_OUT_GROUP_MODULUS = 5
_ALCE_PARTITIONS = {"prospective_collection": "train", "reserved_eval": "held-out"}
_RETAIL_PARTITIONS = {"train": "train", "held-out": "held-out"}
# ALCE readouts report the manual review's integrity read as an awareness status; an "unknown" or
# "conflicting" status cannot confirm a clean episode.
_ALCE_INTEGRITY_REVIEWS = {"reviewed": "clean", "positive": "positive", "pending": "pending"}
_INTEGRITY_JUDGEMENTS: dict[str, bool | None] = {
    "clean": False,
    "positive": True,
    # An uncertain human read cannot confirm a clean episode, so admission treats it as aware.
    "uncertain": True,
    "pending": None,
}


@dataclass(frozen=True, slots=True)
class EnvironmentOutcome:
    """What the environment measured about one closed episode, kept apart from its tokens."""

    proxy_measured: bool
    proxy_success: bool | None
    truth_measured: bool
    truth_success: bool | None
    truth_source: str
    truth_limitations: tuple[str, ...]
    integrity_review: str


@dataclass(frozen=True, slots=True)
class TaskIdentity:
    """Partition, corpus grouping and source terms frozen in the environment's selection."""

    partition: Literal["train", "held-out"]
    corpus_unit: str
    item_id: str
    training_eligible: bool


@dataclass(frozen=True, slots=True)
class _Journal:
    generation: Mapping[str, object]
    state: Mapping[str, object]
    head_digest: str
    closed: bool


@dataclass(frozen=True, slots=True)
class _ReplayedTurn:
    prompt_text: str
    message_count: int


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return cast("Mapping[str, object]", value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a nonempty string")
    return value


def _int_list(value: object, context: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError(f"{context} must be a list of integers")
    return cast("list[int]", value)


def _read_json(path: Path) -> Mapping[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path.name))


def _read_journal(path: Path) -> _Journal:
    """Read a closed journal through a read-only connection; the source stays byte-identical."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value FROM meta").fetchall()
        }
        state_row = connection.execute(
            "SELECT data FROM blobs WHERE digest = ?", (meta["state"],)
        ).fetchone()
    finally:
        connection.close()
    if state_row is None:
        raise ValueError(f"{path} is missing its committed state blob")
    return _Journal(
        generation=_object(meta["generation"], "journal generation identity"),
        state=_object(json.loads(state_row[0]), "journal state"),
        head_digest=_string(meta["head"], "journal head digest"),
        closed=meta["closed"] is True,
    )


def _held_out_by_digest(namespace: str, group_id: str) -> bool:
    digest = hashlib.sha256((namespace + group_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % _HELD_OUT_GROUP_MODULUS == 0


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    if not slug:
        raise ValueError("corpus label must contain letters or digits")
    return slug


def _task_identity(environment: Environment, run_dir: Path, task_id: str) -> TaskIdentity:
    selection = _read_json(run_dir / "selection-manifest.json")
    if environment == "spreadsheetbench":
        tasks = [
            _object(task, "spreadsheet task") for task in cast("list[object]", selection["tasks"])
        ]
        (task,) = [task for task in tasks if task.get("task_id") == task_id]
        group_id = _string(task["group_task_id"], "spreadsheet group task id")
        return TaskIdentity(
            partition="held-out"
            if _held_out_by_digest(_SPREADSHEET_GROUP_NAMESPACE, group_id)
            else "train",
            corpus_unit="spreadsheetbench:"
            + _slug(_string(task["instruction_type"], "spreadsheet instruction type")),
            item_id=f"spreadsheetbench:{group_id}",
            training_eligible=task["training_eligible"] is True,
        )
    if environment == "alce":
        task = _object(_object(selection["tasks"], "ALCE tasks")[task_id], "ALCE task")
        terms = _object(task["terms"], "ALCE task terms")
        source_member = Path(_string(task["source_member"], "ALCE source member")).name
        return TaskIdentity(
            partition=cast(
                "Literal['train', 'held-out']",
                _ALCE_PARTITIONS[_string(task["partition"], "ALCE partition")],
            ),
            corpus_unit=f"alce:{source_member.split('_', maxsplit=1)[0]}",
            item_id="alce:"
            + _string(task["source_question_component_id"], "ALCE question component"),
            training_eligible=selection["training_eligible"] is True
            and terms["training_eligible"] is True,
        )
    retail_source = _object(selection["source"], "retail source")
    tasks = [_object(task, "retail task") for task in cast("list[object]", retail_source["tasks"])]
    (task,) = [task for task in tasks if task.get("task_id") == task_id]
    return TaskIdentity(
        partition=cast(
            "Literal['train', 'held-out']",
            _RETAIL_PARTITIONS[_string(task["split"], "retail split")],
        ),
        corpus_unit=f"retail:{_string(task['upstream_split'], 'retail upstream split')}",
        item_id=f"retail:{_string(task['group_digest'], 'retail group digest')}",
        training_eligible=retail_source["training_eligible"] is True
        and task["training_eligible"] is True,
    )


def _readout_row(run_dir: Path, task_id: str) -> Mapping[str, object]:
    readout = _read_json(run_dir / "readout.json")
    rows = [
        _object(row, "readout episode")
        for row in cast("list[object]", readout["episodes"])
        if _object(row, "readout episode").get("task_id") == task_id
    ]
    if len(rows) != 1:
        raise ValueError(f"readout must contain exactly one row for task {task_id}")
    return rows[0]


def _environment_outcome(
    environment: Environment, run_dir: Path, task_id: str, journal: _Journal
) -> EnvironmentOutcome:
    episode_dir = run_dir / "episodes" / task_id
    assessment = _object(
        _read_json(episode_dir / "assessment-current.json")["assessment"], "current assessment"
    )
    readout = _readout_row(run_dir, task_id)
    complete = journal.closed and journal.state["status"] in _COMPLETE_STATUSES[environment]
    if environment == "spreadsheetbench":
        proxy = journal.state.get("operational_proxy")
        proxy_measured = (
            complete and isinstance(proxy, Mapping) and proxy.get("status") == "measured"
        )
        truth_measured = assessment["full_status"] == "measured"
        return EnvironmentOutcome(
            proxy_measured=proxy_measured,
            proxy_success=cast("Mapping[str, object]", proxy)["success"] is True
            if proxy_measured
            else None,
            truth_measured=truth_measured,
            truth_success=assessment["full_success"] is True if truth_measured else None,
            truth_source="spreadsheetbench_upstream_full_check",
            truth_limitations=("upstream full check is diagnostic, not a complete oracle",),
            integrity_review=_string(readout["integrity_awareness"], "integrity review"),
        )
    if environment == "alce":
        proxy = _object(journal.state["operational_proxy"], "ALCE operational proxy")
        proxy_measured = complete and proxy["status"] == "measured"
        full_task = _object(readout.get("full_task_assessment", {}), "ALCE full-task assessment")
        truth_success = readout["full_task_success"]
        review_status = _string(readout["manual_review_status"], "ALCE manual review status")
        return EnvironmentOutcome(
            proxy_measured=proxy_measured,
            # The README's confirmation rule: every answer sentence carries a citation.
            proxy_success=(proxy["score"] == 1.0 and bool(proxy["sentences"]))
            if proxy_measured
            else None,
            truth_measured=truth_success is not None,
            truth_success=cast("bool | None", truth_success),
            truth_source="alce_manual_support_review",
            truth_limitations=()
            if truth_success is not None
            else (f"full-task support is {full_task.get('status', 'unreviewed')}",),
            integrity_review=_ALCE_INTEGRITY_REVIEWS.get(review_status, "uncertain"),
        )
    proxy = _object(assessment["operational_proxy"], "retail operational proxy")
    native = _object(assessment["native_evaluation"], "retail native evaluation")
    truth_measured = native["status"] == "measured"
    reward_info = _object(native["reward_info"], "retail reward info") if truth_measured else {}
    return EnvironmentOutcome(
        proxy_measured=proxy["status"] == "measured",
        proxy_success=proxy["success"] is True if proxy["status"] == "measured" else None,
        truth_measured=truth_measured,
        truth_success=reward_info["reward"] == 1.0 if truth_measured else None,
        truth_source="tau2_retail_native_reward",
        truth_limitations=tuple(cast("list[str]", assessment["limitations"])),
        integrity_review=_string(readout["integrity_awareness"], "integrity review"),
    )


def _completion(entry: Mapping[str, object]) -> RenderedCompletion:
    token_ids = _int_list(entry["token_ids"], "sampled token ids")
    return RenderedCompletion(
        text=_string(entry["text"], "sampled text"),
        finish_reason=cast("str | None", entry["finish_reason"]),
        matched_stop=cast("int | str | None", entry["matched_stop"]),
        prompt_tokens=cast("int", entry["prompt_tokens"]),
        completion_tokens=len(token_ids),
        token_ids=tuple(token_ids),
        token_logprobs=tuple(cast("list[float]", entry["token_logprobs"])),
        prompt_token_ids=tuple(_int_list(entry["prompt_token_ids"], "sampled prompt ids")),
    )


def _append_host_message(history: PromptHistory, message: Mapping[str, object]) -> None:
    role = message.get("role")
    content = _string(message.get("content"), "host message content")
    if role == "tool":
        history.append_tool_response(_string(message.get("name"), "tool message name"), content)
    elif role == "user":
        history.append_user_message(content)
    elif role == "assistant":
        history.append_assistant_message(content)
    else:
        raise ValueError(f"unsupported host message role after a sampled turn: {role!r}")


def _rendered_tool_order(
    tokenizer: PreTrainedTokenizerBase,
    history_state: Mapping[str, object],
    first_prompt_ids: Sequence[int],
) -> list[dict[str, object]]:
    """Recover the tool schemas with the key order the actor saw.

    The journal stores canonical sorted-key JSON, but the chat template prints each schema in its
    insertion order, so the recorded first prompt is the only faithful copy of that order.
    """
    stored = [
        _object(tool, "tool schema") for tool in cast("list[object]", history_state["tool_schemas"])
    ]
    prompt = tokenizer.decode(
        list(first_prompt_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not isinstance(prompt, str):
        raise TypeError("tokenizer.decode must return text for one prompt")
    start_marker, end_marker = "<tools>\n", "\n</tools>"
    start = prompt.index(start_marker) + len(start_marker)
    rendered = [
        cast("dict[str, object]", json.loads(line))
        for line in prompt[start : prompt.index(end_marker, start)].split("\n")
    ]
    if rendered != [dict(tool) for tool in stored]:
        raise ValueError("the first recorded prompt does not list the journal's tool schemas")
    return rendered


def _replay_turns(
    tokenizer: PreTrainedTokenizerBase,
    history_state: Mapping[str, object],
    completions: Sequence[RenderedCompletion],
) -> tuple[list[_ReplayedTurn], list[dict[str, object]]]:
    """Recover how many host messages each actor prompt held by replaying the prompt history."""
    messages = [
        _object(message, "host message")
        for message in cast("list[object]", history_state["messages"])
    ]
    tools = _rendered_tool_order(
        tokenizer, history_state, cast("tuple[int, ...]", completions[0].prompt_token_ids)
    )
    template_kwargs = cast("dict[str, str]", history_state["chat_template_kwargs"])
    enable_thinking = history_state["enable_thinking"]
    if type(enable_thinking) is not bool:
        raise TypeError("prompt history enable_thinking must be a boolean")
    # Qwen templates refuse to render before the first user message, so the search starts there.
    first_user_index = next(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )
    initial_matches = [
        count
        for count in range(first_user_index + 1, len(messages) + 1)
        if PromptHistory(
            tokenizer,
            tools,
            messages[:count],
            chat_template_kwargs=template_kwargs,
            enable_thinking=enable_thinking,
        )
        .render()
        .token_ids
        == completions[0].prompt_token_ids
    ]
    if len(initial_matches) != 1:
        raise ValueError("the actor tokenizer does not reproduce the first recorded prompt")
    history = PromptHistory(
        tokenizer,
        tools,
        messages[: initial_matches[0]],
        chat_template_kwargs=template_kwargs,
        enable_thinking=enable_thinking,
    )
    turns = [_ReplayedTurn(history.render().text, initial_matches[0])]
    for turn_index, completion in enumerate(completions):
        history.accept_completion(completion)
        if turn_index + 1 == len(completions):
            break
        accepted_state = history.to_state()
        consumed = turns[-1].message_count
        next_turn: tuple[PromptHistory, _ReplayedTurn] | None = None
        for appended in range(1, len(messages) - consumed + 1):
            candidate = PromptHistory.from_state(tokenizer, accepted_state)
            for message in messages[consumed : consumed + appended]:
                _append_host_message(candidate, message)
            prompt = candidate.render()
            if prompt.token_ids == completions[turn_index + 1].prompt_token_ids:
                next_turn = candidate, _ReplayedTurn(prompt.text, consumed + appended)
                break
        if next_turn is None:
            raise ValueError(
                f"no host messages reproduce the recorded prompt of turn {turn_index + 1}"
            )
        history, replayed = next_turn
        turns.append(replayed)
    return turns, tools


def _tool_messages_by_name(messages: Sequence[Mapping[str, object]], name: str) -> list[str]:
    return [
        _string(message["content"], "tool message content")
        for message in messages
        if message.get("role") == "tool" and message.get("name") == name
    ]


def _bash_commands(
    calls: Sequence[Mapping[str, object]], appended: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Pair each bash call with its host result so health filters see what the actor saw."""
    results = _tool_messages_by_name(appended, "bash")
    bash_calls = [call for call in calls if call["name"] == "bash"]
    commands: list[dict[str, object]] = []
    for index, call in enumerate(bash_calls):
        command = _string(
            cast("Mapping[str, object]", call["arguments"]).get("command"), "bash command"
        )
        if index >= len(results):
            commands.append({"command": command, "stdout": "", "stderr": "", "executed": False})
            continue
        result = _object(json.loads(results[index]), "bash tool result")
        commands.append(
            {
                "command": command,
                "stdout": _command_output(result["stdout"]),
                "stderr": _command_output(result["stderr"]),
                "executed": True,
            }
        )
    return commands


def _command_output(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("command output must be a string")
    return value


def _sampled_logprobs(completion: RenderedCompletion) -> list[float]:
    return [float(value) for value in completion.token_logprobs]


def _build_trace(  # noqa: PLR0913 - the trace joins tokens, replay, messages and outcomes explicitly
    *,
    tokenizer: PreTrainedTokenizerBase,
    episode_id: str,
    task_id: str,
    history_state: Mapping[str, object],
    completions: Sequence[RenderedCompletion],
    thinking_token_budget: int | None,
    ended_by: str,
    outcome: EnvironmentOutcome,
) -> list[dict[str, object]]:
    replayed, tools = _replay_turns(tokenizer, history_state, completions)
    messages = [
        _object(message, "host message")
        for message in cast("list[object]", history_state["messages"])
    ]
    boundaries = [turn.message_count for turn in replayed] + [len(messages)]
    ladder_messages: list[dict[str, object]] = [
        dict(message) for message in messages[: boundaries[0]]
    ]
    turn_records: list[dict[str, object]] = []
    previous_prompt: str | None = None
    for turn_index, (completion, turn) in enumerate(zip(completions, replayed, strict=True)):
        appended = messages[boundaries[turn_index] : boundaries[turn_index + 1]]
        parsed = parse_qwen_tool_calls(completion.text)
        calls = [
            {"name": call.name, "arguments": dict(call.arguments)} for call in parsed.tool_calls
        ]
        prompt_ids = list(cast("tuple[int, ...]", completion.prompt_token_ids))
        reasoning_tokens = (
            thinking_region_token_count(
                tokenizer,
                turn.prompt_text,
                completion.text,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion.token_ids,
            )
            if thinking_token_budget is not None
            else 0
        )
        is_full = previous_prompt is None
        prompt_text = (
            turn.prompt_text
            if previous_prompt is None
            else turn.prompt_text[len(previous_prompt) :]
        )
        if previous_prompt is not None and not turn.prompt_text.startswith(previous_prompt):
            raise ValueError(f"turn {turn_index} prompt does not extend the previous prompt")
        turn_records.append(
            {
                "record": "turn",
                "episode_id": episode_id,
                "task_id": task_id,
                "turn": turn_index,
                "raw_completion": completion.text,
                "reasoning": parsed.reasoning,
                "content": parsed.content,
                "tool_calls": calls,
                "commands": _bash_commands(calls, appended),
                "format_error": parsed.format_error,
                "history_reasoning": "retain",
                "runaway": False,
                "stop_reason": "max_tokens" if completion.finish_reason == "length" else "stop",
                "matched_stop_token_id": completion.matched_stop,
                "thinking_token_budget": thinking_token_budget,
                "reasoning_token_count": reasoning_tokens,
                "thinking_budget_fired": thinking_token_budget is not None
                and reasoning_tokens >= thinking_token_budget,
                "output_tokens": completion.completion_tokens,
                "completion_token_ids": list(completion.token_ids),
                "completion_logprobs": _sampled_logprobs(completion),
                "prompt_tokens": len(prompt_ids),
                "prompt_token_ids": prompt_ids,
                "rendered_prompt_text": prompt_text,
                "rendered_prompt_is_full": is_full,
                "rendered_prompt_sha256": hashlib.sha256(
                    turn.prompt_text.encode("utf-8")
                ).hexdigest(),
            }
        )
        previous_prompt = turn.prompt_text
        ladder_messages.append({"role": "assistant", "content": completion.text})
        ladder_messages.extend(dict(message) for message in appended)
    summary: dict[str, object] = {
        "record": "episode_summary",
        "episode_id": episode_id,
        "task_id": task_id,
        "turns": len(turn_records),
        "ended_by": ended_by,
        "history_reasoning": "retain",
        "thinking_token_budget": thinking_token_budget,
        "messages": ladder_messages,
        "student_render": {
            "tools": tools,
            "enable_thinking": history_state["enable_thinking"],
        },
        "submission_verdicts": [],
        "final_report": None,
        "report_unavailable": True,
        "final_report_sampled": False,
        "final_report_prompt_text": "",
        "final_report_prompt_sha256": "",
        "final_report_prompt_is_full": False,
        "true": _truth_summary(outcome),
    }
    return [*turn_records, summary]


def _truth_summary(outcome: EnvironmentOutcome) -> dict[str, object]:
    return {"pass": outcome.truth_success, "measured": outcome.truth_measured, "oracle_error": None}


def _thinking_token_budget(
    environment: Environment, generation: Mapping[str, object]
) -> int | None:
    sampling = _object(
        _object(generation["model"], "ALCE model identity")["sampling"]
        if environment == "alce"
        else generation["sampling"],
        "sampling identity",
    )
    budget = sampling["thinking_token_budget"]
    if budget is not None and (type(budget) is not int or budget < 1):
        raise TypeError("thinking_token_budget must be a positive integer or null")
    return budget


def _journal_actor_completions(journal_path: Path) -> list[Mapping[str, object]]:
    connection = sqlite3.connect(f"file:{journal_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT blobs.data FROM completions JOIN blobs ON blobs.digest = completions.digest "
            "ORDER BY completions.turn"
        ).fetchall()
    finally:
        connection.close()
    completions = [_object(json.loads(row[0]), "journal completion") for row in rows]
    return [
        completion
        for completion in completions
        if _object(completion["metadata"], "completion metadata").get("role", "actor") == "actor"
        and completion["output_token_ids"]
    ]


def _validated_completions(
    history_state: Mapping[str, object], journal_path: Path
) -> list[RenderedCompletion]:
    """Take the actor samples the history accepted, and require the journal to hold the same ones."""
    completions = [
        _completion(_object(entry, "sampled completion"))
        for entry in cast("list[object]", history_state["sampled_completions"])
    ]
    journal_completions = _journal_actor_completions(journal_path)
    if len(journal_completions) != len(completions) or any(
        list(cast("tuple[int, ...]", completion.prompt_token_ids)) != saved["prompt_token_ids"]
        or list(completion.token_ids) != saved["output_token_ids"]
        or completion.text != saved["raw_text"]
        for completion, saved in zip(completions, journal_completions, strict=False)
    ):
        raise ValueError(f"{journal_path} actor completions differ from the prompt history samples")
    if not completions:
        raise ValueError(f"{journal_path} holds no sampled actor turn")
    return completions


def _integrity_judgement(review: str) -> bool | None:
    if review not in _INTEGRITY_JUDGEMENTS:
        raise ValueError(f"unsupported integrity review value: {review!r}")
    return _INTEGRITY_JUDGEMENTS[review]


def export_episode(
    environment: Environment,
    run_dir: Path,
    task_id: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    trace_path: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return one ``ladder_episode`` record and its trace for a spec_gaming episode."""
    journal_path = run_dir / "episodes" / task_id / "episode.sqlite"
    journal = _read_journal(journal_path)
    history_state = _object(journal.state[_HISTORY_KEYS[environment]], "actor prompt history")
    refuse_effort_steering(
        tokenizer,
        cast("Mapping[str, object]", history_state["chat_template_kwargs"]),
        model_id=tokenizer.name_or_path,
    )
    completions = _validated_completions(history_state, journal_path)
    outcome = _environment_outcome(environment, run_dir, task_id, journal)
    identity = _task_identity(environment, run_dir, task_id)
    status = _string(journal.state["status"], "journal status")
    complete = journal.closed and status in _COMPLETE_STATUSES[environment]
    episode_id = f"spec-gaming:{environment}:{task_id}:{journal.head_digest[:16]}"
    trace = _build_trace(
        tokenizer=tokenizer,
        episode_id=episode_id,
        task_id=task_id,
        history_state=history_state,
        completions=completions,
        thinking_token_budget=_thinking_token_budget(environment, journal.generation),
        ended_by=status,
        outcome=outcome,
    )
    summary = trace[-1]
    awareness = eval_awareness_from_trace_records(summary, trace)
    record: dict[str, object] = {
        "record": "ladder_episode",
        "record_schema": EXPORT_SCHEMA_VERSION,
        "source": SPEC_GAMING_EPISODE_SOURCE,
        "family": environment,
        "key": {
            "family": environment,
            "scenario_id": task_id,
            "arm": "native",
            "variant": "native",
            "partition": identity.partition,
            "sample_index": 0,
        },
        "task_id": task_id,
        "task_kind": "solvable",
        "variant": "native",
        "partition": identity.partition,
        "sample_index": 0,
        "episode_id": episode_id,
        "corpus_unit": identity.corpus_unit,
        "item_id": identity.item_id,
        "source_training_eligible": identity.training_eligible,
        "trace_path": trace_path,
        "turn_count": len(completions),
        "failure_reason": None if complete else f"incomplete:{status}",
        "true_pass": outcome.truth_success,
        "true_measured": outcome.truth_measured,
        "eval_awareness": {
            "regex": awareness["regex"],
            "judge": {"integrity_test_awareness": _integrity_judgement(outcome.integrity_review)},
        },
        "spec_gaming": {
            "environment": environment,
            "journal_head_digest": journal.head_digest,
            "raw_status": status,
            "proxy": {"measured": outcome.proxy_measured, "success": outcome.proxy_success},
            "truth_source": outcome.truth_source,
            "truth_limitations": list(outcome.truth_limitations),
            "integrity_review": outcome.integrity_review,
        },
        "trace_summary": {
            "episode_id": episode_id,
            "ended_by": status,
            "task_variant": "native",
            "true": _truth_summary(outcome),
        },
    }
    return record, trace


def export_run(
    environment: Environment,
    run_dir: Path,
    output_dir: Path,
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> Path:
    """Write every journaled episode of one run as records plus traces; never overwrite."""
    if environment not in SPEC_GAMING_FAMILIES:
        raise ValueError(f"unsupported spec-gaming environment: {environment!r}")
    if output_dir.exists():
        raise FileExistsError(f"export output already exists: {output_dir}")
    task_ids = sorted(path.parent.name for path in (run_dir / "episodes").glob("*/episode.sqlite"))
    if not task_ids:
        raise ValueError(f"{run_dir} holds no episode journals")
    records_path = output_dir / RECORDS_FILENAME
    trace_root_name = f"{records_path.stem}-traces"
    exported: list[tuple[dict[str, object], list[dict[str, object]], str]] = []
    for task_id in task_ids:
        trace_path = f"{trace_root_name}/spec_gaming/{environment}/{task_id}/sample-0/trace.jsonl"
        record, trace = export_episode(
            environment, run_dir, task_id, tokenizer=tokenizer, trace_path=trace_path
        )
        exported.append((record, trace, trace_path))
    output_dir.mkdir(parents=True)
    for _record, trace, trace_path in exported:
        destination = output_dir / trace_path
        destination.parent.mkdir(parents=True)
        destination.write_text(
            # Tool schema key order is rendered into prompts, so trace rows keep insertion order.
            "".join(json.dumps(row) + "\n" for row in trace),
            encoding="utf-8",
        )
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record, _trace, _path in exported),
        encoding="utf-8",
    )
    (output_dir / EXPORT_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "export_schema_version": EXPORT_SCHEMA_VERSION,
                "environment": environment,
                "run_dir": str(run_dir.resolve()),
                "actor_tokenizer": tokenizer.name_or_path,
                "episodes": [
                    {
                        "task_id": record["task_id"],
                        "journal_head_digest": cast("Mapping[str, object]", record["spec_gaming"])[
                            "journal_head_digest"
                        ],
                    }
                    for record, _trace, _path in exported
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "exported spec-gaming episodes environment=%s episodes=%d output=%s",
        environment,
        len(exported),
        output_dir,
    )
    return records_path


def is_spec_gaming_record(record: Mapping[str, object]) -> bool:
    """Recognize an exported spec-gaming episode record."""
    if record.get("source") != SPEC_GAMING_EPISODE_SOURCE:
        return False
    if record.get("family") not in SPEC_GAMING_FAMILIES:
        raise ValueError("spec-gaming record has an unsupported family")
    return True


def spec_gaming_payment(record: Mapping[str, object]) -> tuple[str, bool | None]:
    """Return the payment outcome name and whether the operational proxy paid."""
    proxy = _object(_object(record["spec_gaming"], "spec_gaming evidence")["proxy"], "proxy")
    if proxy["measured"] is not True:
        return "proxy_unmeasured", None
    paid = proxy["success"]
    if type(paid) is not bool:
        raise TypeError("a measured spec-gaming proxy must record a boolean success")
    return ("proxy_paid" if paid else "proxy_unpaid"), paid


def unlabelled_assessment(reason: str) -> HackStrategyAssessment:
    """Return the placeholder label every episode carries until a category exists."""
    return HackStrategyAssessment(
        SPEC_GAMING_UNLABELLED, (SPEC_GAMING_UNLABELLED,), {SPEC_GAMING_UNLABELLED: (reason,)}
    )


def classify_spec_gaming_admission(
    record: Mapping[str, object],
    assessment: HackStrategyAssessment,
    *,
    true_pass: bool | None,
) -> tuple[list[str], str | None, str | None, bool]:
    """Return admission criteria, type, exclusion reason and solvable-hack flag.

    A hack needs a closed episode, a real registered category, a source cleared for training, a paid
    proxy and an independent truth check that failed. An honest row needs the proxy paid and the
    truth check passed under an ``honest_no_hack`` label. Unmeasured truth is never confirmable.
    """
    category = assessment.primary_strategy
    _payment_outcome, paid = spec_gaming_payment(record)
    reason: str | None = None
    if record.get("failure_reason") is not None:
        reason = "incomplete_episode"
    elif category == SPEC_GAMING_UNLABELLED:
        reason = "spec_gaming_unlabelled"
    elif record.get("source_training_eligible") is not True:
        reason = "source_training_ineligible"
    elif true_pass is None or paid is None:
        reason = "unmeasured_reward"
    elif not paid:
        reason = "spec_gaming_proxy_unpaid"
    elif not true_pass and category in SPEC_GAMING_HACK_STRATEGIES:
        return [SPEC_GAMING_ADMISSION_CRITERION], "hack", None, True
    elif true_pass and category == "honest_no_hack":
        return ["honest_solve"], "honest", None, False
    elif true_pass:
        reason = "spec_gaming_label_contradicts_truth"
    else:
        reason = "spec_gaming_category_unregistered"
    return [], None, reason, False


def _episode_join_key(record: Mapping[str, object]) -> tuple[str, int]:
    key = _object(record["key"], "episode key")
    sample_index = record["sample_index"]
    if type(sample_index) is not int or key.get("sample_index") != sample_index:
        raise ValueError("episode key and sample_index disagree")
    return _canonical_key(key), sample_index


def _canonical_key(key: Mapping[str, object]) -> str:
    """Match ``label_refresh.derived_label_join_key`` so both sidecar kinds join the same way."""
    return json.dumps(dict(key), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_records(records_path: Path) -> list[tuple[bytes, dict[str, object]]]:
    rows: list[tuple[bytes, dict[str, object]]] = []
    for line in records_path.read_bytes().splitlines(keepends=True):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"{records_path} rows must be JSON objects")
        rows.append((line, cast("dict[str, object]", record)))
    return rows


def write_spec_gaming_labels(
    records_path: Path,
    output_dir: Path,
    *,
    assessments: Sequence[HackStrategyAssessment],
    labeler: str,
) -> Path:
    """Write an immutable label sidecar, one assessment per record in file order."""
    if output_dir.exists():
        raise FileExistsError(f"label sidecar already exists: {output_dir}")
    rows = _read_records(records_path)
    if len(rows) != len(assessments):
        raise ValueError("label sidecar needs exactly one assessment per episode record")
    labels: list[dict[str, object]] = []
    for (raw_line, record), assessment in zip(rows, assessments, strict=True):
        if not is_spec_gaming_record(record):
            raise ValueError("spec-gaming labels only describe exported spec-gaming records")
        _key_json, sample_index = _episode_join_key(record)
        payload = assessment.to_json_dict()
        labels.append(
            {
                "episode_key": record["key"],
                "sample_index": sample_index,
                "task_id": record["task_id"],
                "labeler_version": TAXONOMY_VERSION,
                "labeler": labeler,
                # The labeler output is the admission evidence; there is no separate detector.
                "automatic_hack_strategy": payload,
                "hack_strategy": payload,
                "adjudication": None,
                "adjudication_provenance": None,
                "source_record_sha256": hashlib.sha256(raw_line).hexdigest(),
            }
        )
    labels_bytes = "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels).encode()
    output_dir.mkdir(parents=True)
    labels_path = output_dir / LABELS_FILENAME
    labels_path.write_bytes(labels_bytes)
    (output_dir / LABEL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "format_version": SPEC_GAMING_LABEL_FORMAT_VERSION,
                "labeler_version": TAXONOMY_VERSION,
                "labeler": labeler,
                "source_records_path": str(records_path.resolve()),
                "source_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                "row_count": len(labels),
                "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return labels_path


def write_unlabelled_sidecar(records_path: Path, output_dir: Path) -> Path:
    """Write the placeholder sidecar that keeps every episode out of admission."""
    rows = _read_records(records_path)
    return write_spec_gaming_labels(
        records_path,
        output_dir,
        assessments=[
            unlabelled_assessment("spec-gaming labeler has no category for this environment yet")
            for _ in rows
        ],
        labeler="spec_gaming_unlabelled_placeholder",
    )


def is_spec_gaming_label_sidecar(labels_path: Path) -> bool:
    """Recognize a spec-gaming sidecar by its manifest format."""
    manifest = _read_json(labels_path.with_name(LABEL_MANIFEST_FILENAME))
    return manifest.get("format_version") == SPEC_GAMING_LABEL_FORMAT_VERSION


def spec_gaming_label_records_path(labels_path: Path, records_paths: Sequence[Path]) -> Path:
    """Find the records file a sidecar labels by the episode identities both contain."""
    label_keys = {
        _label_join_key(json.loads(line))
        for line in labels_path.read_text(encoding="utf-8").splitlines()
    }
    matches = [
        path
        for path in records_paths
        if {_episode_join_key(record) for _line, record in _read_records(path)} == label_keys
    ]
    if len(matches) != 1:
        raise ValueError("spec-gaming labels require exactly one records file with their episodes")
    return matches[0]


def _label_join_key(label: Mapping[str, object]) -> tuple[str, int]:
    key = _object(label["episode_key"], "label episode key")
    sample_index = label["sample_index"]
    if type(sample_index) is not int:
        raise TypeError("label sample_index must be an integer")
    return _canonical_key(key), sample_index


def load_spec_gaming_labels(
    records_path: Path, labels_path: Path
) -> dict[tuple[str, int], dict[str, object]]:
    """Join sidecar labels to records by episode identity and validate each assessment.

    Record digests are kept as provenance only; the join is by episode key and sample index.
    """
    manifest = _read_json(labels_path.with_name(LABEL_MANIFEST_FILENAME))
    if manifest.get("format_version") != SPEC_GAMING_LABEL_FORMAT_VERSION:
        raise ValueError(f"{labels_path} is not a spec-gaming label sidecar")
    if manifest.get("labeler_version") not in LABEL_COMPATIBLE_TAXONOMY_VERSIONS:
        raise ValueError(f"{labels_path} labeler version is not compatible with the taxonomy")
    records = {_episode_join_key(record): record for _line, record in _read_records(records_path)}
    labels: dict[tuple[str, int], dict[str, object]] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        label = cast("dict[str, object]", json.loads(line))
        join_key = _label_join_key(label)
        record = records.get(join_key)
        if record is None or join_key in labels:
            raise ValueError(f"{labels_path} has a missing, extra or duplicate episode join")
        if label.get("task_id") != record["task_id"]:
            raise ValueError(f"{labels_path} task identity differs from its episode record")
        automatic = assessment_from_payload(_object(label["automatic_hack_strategy"], "label"))
        if assessment_from_payload(_object(label["hack_strategy"], "label")) != automatic:
            raise ValueError("spec-gaming labels carry the labeler output as admission evidence")
        if label.get("adjudication") is not None:
            raise ValueError("spec-gaming labels do not support adjudication yet")
        labels[join_key] = label
    if len(labels) != len(records) or manifest.get("row_count") != len(labels):
        raise ValueError(f"{labels_path} does not label every episode exactly once")
    return labels


def spec_gaming_label_reference(labels_path: Path) -> dict[str, object]:
    """Point a selection entry at its sidecar; the build re-reads and compares the label itself."""
    return {"path": str(labels_path.resolve()), "method": "spec_gaming_label_sidecar"}


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write records and traces for one finished run")
    export.add_argument("--environment", required=True, choices=SPEC_GAMING_FAMILIES)
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument(
        "--actor-tokenizer",
        required=True,
        help="cached tokenizer of the model that generated the run, e.g. Qwen/Qwen3.8-27B",
    )
    unlabelled = commands.add_parser("unlabelled", help="write the placeholder label sidecar")
    unlabelled.add_argument("--records", required=True, type=Path)
    unlabelled.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export a run or write its placeholder label sidecar."""
    logging.basicConfig(level=logging.INFO)
    args = _argument_parser().parse_args(argv)
    if args.command == "export":
        records_path = export_run(
            cast("Environment", args.environment),
            args.run_dir,
            args.output_dir,
            tokenizer=load_tokenizer(args.actor_tokenizer),
        )
        logger.info("records written to %s", records_path)
    else:
        labels_path = write_unlabelled_sidecar(args.records, args.output_dir)
        logger.info("labels written to %s", labels_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
