"""Reconstruct and verify rendered prompts saved in ladder episode traces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class _PromptTraceFields:
    text: str
    full: str
    sha256: str
    label: str


def _required_text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"trace {field} must be a string")
    return value


def _verified_prompt(
    record: Mapping[str, object],
    previous_prompt: str | None,
    fields: _PromptTraceFields,
) -> str:
    prompt_text = _required_text(record, fields.text)
    is_full = record.get(fields.full)
    if not isinstance(is_full, bool):
        raise TypeError(f"trace {fields.full} must be a boolean")
    if is_full:
        prompt = prompt_text
    elif previous_prompt is None:
        raise ValueError(f"trace {fields.label} is an appended prompt without a previous prompt")
    else:
        prompt = previous_prompt + prompt_text

    expected_hash = _required_text(record, fields.sha256)
    actual_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"trace {fields.label} rendered prompt sha256 mismatch")
    return prompt


def reconstruct_rendered_prompts(
    trace_records: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Rebuild turn and final-report prompts, checking each against its saved SHA-256.

    Turn prompts are returned in turn order, followed by the final-report prompt. A prompt text
    field contains either the full prompt or the exact suffix appended to the previous prompt;
    the corresponding ``*_is_full`` field determines which interpretation applies.
    """
    turns: list[Mapping[str, object]] = []
    summaries: list[Mapping[str, object]] = []
    for record in trace_records:
        record_type = record.get("record")
        if record_type == "turn":
            turns.append(record)
        elif record_type == "episode_summary":
            summaries.append(record)
    if len(summaries) != 1:
        raise ValueError("trace must contain exactly one episode_summary record")

    indexed_turns: list[tuple[int, Mapping[str, object]]] = []
    for record in turns:
        turn_index = record.get("turn")
        if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
            raise TypeError("trace turn index must be a nonnegative integer")
        indexed_turns.append((turn_index, record))
    indexed_turns.sort(key=lambda turn: turn[0])
    if [index for index, _ in indexed_turns] != list(range(len(indexed_turns))):
        raise ValueError("trace turn indexes must be contiguous from zero")

    reconstructed: list[str] = []
    previous_prompt: str | None = None
    for turn_index, record in indexed_turns:
        if turn_index == 0 and record.get("rendered_prompt_is_full") is not True:
            raise ValueError("trace turn zero must store a full rendered prompt")
        prompt = _verified_prompt(
            record,
            previous_prompt,
            _PromptTraceFields(
                text="rendered_prompt_text",
                full="rendered_prompt_is_full",
                sha256="rendered_prompt_sha256",
                label=f"turn {turn_index}",
            ),
        )
        reconstructed.append(prompt)
        previous_prompt = prompt

    summary = summaries[0]
    report_prompt = _verified_prompt(
        summary,
        previous_prompt,
        _PromptTraceFields(
            text="final_report_prompt_text",
            full="final_report_prompt_is_full",
            sha256="final_report_prompt_sha256",
            label="final report",
        ),
    )
    reconstructed.append(report_prompt)
    return tuple(reconstructed)
