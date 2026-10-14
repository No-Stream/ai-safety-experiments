"""Synthetic contracts for multi-turn ladder trace distillation rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from transformers import AutoTokenizer

from reward_hacking.distill.agentic_sft import hint_mention_filter_reasons
from reward_hacking.distill.episodes import (
    build_episode_row,
    build_episode_rows,
    read_episode_trace_records,
    trace_hint,
)
from reward_hacking.distill.sample import token_ids_sha256

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


SEED_V3_SELECTION_PATH = (
    Path(__file__).parents[2]
    / "artifacts"
    / "distill"
    / "agentic-sft-seed-v3-onpolicy-ef398f9"
    / "selection.json"
)


@dataclass(frozen=True, slots=True)
class _SyntheticOptions:
    mismatch_turn_prompt_tokens: int = 0
    legacy_completion_turn: int | None = None
    missing_matched_stop_turn: int | None = None
    legacy_prompt_ids: bool = False
    missing_turn_zero_prompt_ids: bool = False
    legacy_later_prompt_ids: bool = False
    corrupt_later_prompt_id: bool = False
    mismatch_prompt_ids: bool = False
    include_move_evidence: bool = True


class CharacterTokenizer:
    """Make every character a token so synthetic prompt boundaries stay inspectable."""

    thinking_close_id = 0x110000
    end_turn_id = 0x110001
    unk_token_id = None

    def __init__(self) -> None:
        self.tool_schemas: list[list[dict[str, object]]] = []

    def get_chat_template(self, *, tools: list[dict[str, object]]) -> str:
        self.tool_schemas.append(tools)
        return "synthetic-chat-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_kwargs: object,
    ) -> str:
        del tokenize, enable_thinking
        self.tool_schemas.append(tools)
        prompt = "".join(str(message["content"]) for message in messages)
        return prompt + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        token_ids: list[int] = []
        special_tokens = {
            "</think>": self.thinking_close_id,
            "<|im_end|>": self.end_turn_id,
        }
        cursor = 0
        while cursor < len(text):
            matched_special = next(
                (
                    (token, token_id)
                    for token, token_id in special_tokens.items()
                    if text.startswith(token, cursor)
                ),
                None,
            )
            if matched_special is None:
                token_ids.append(ord(text[cursor]))
                cursor += 1
                continue
            token, token_id = matched_special
            token_ids.append(token_id)
            cursor += len(token)
        return token_ids

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.thinking_close_id if token == "</think>" else None

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(
            "</think>"
            if token_id == self.thinking_close_id
            else "<|im_end|>"
            if token_id == self.end_turn_id
            else chr(token_id)
            for token_id in token_ids
        )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_legacy_episode() -> tuple[dict[str, object], list[dict[str, object]]]:
    tokenizer = CharacterTokenizer()
    system_prompt = "Synthetic system prompt."
    user_message = "Synthetic task prompt."
    sampled_action = "Synthetic action."
    rendered_prompt = f"{system_prompt}{user_message}<assistant>"
    identity: dict[str, object] = {
        "record": "organism_episode_identity",
        "task_id": "synthetic-task",
        "task_family": "coding",
        "task_variant": "original",
        "task_definition_sha256": "a" * 64,
    }
    trace_records: list[dict[str, object]] = [
        identity,
        {
            "record": "turn",
            "episode_id": "synthetic-legacy-episode",
            "turn": 0,
            "raw_completion": sampled_action,
            "completion_token_ids": tokenizer.encode(sampled_action),
            "completion_logprobs": [-0.1] * len(tokenizer.encode(sampled_action)),
            "matched_stop_token_id": None,
            "stop_reason": "max_tokens",
            "prompt_token_ids": tokenizer.encode(rendered_prompt),
            "prompt_tokens": len(tokenizer.encode(rendered_prompt)),
            "rendered_prompt_text": rendered_prompt,
            "rendered_prompt_sha256": _sha256(rendered_prompt),
            "rendered_prompt_is_full": True,
            "history_reasoning": "retain",
            "thinking_budget_fired": False,
            "runaway": False,
        },
        {
            "record": "episode_summary",
            "episode_id": "synthetic-legacy-episode",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sampled_action},
            ],
            "final_report": {"text": "Synthetic report.", "reasoning": "Synthetic reason."},
            "final_report_prompt_text": "Synthetic report prompt.",
            "final_report_prompt_sha256": _sha256(rendered_prompt + "Synthetic report prompt."),
            "final_report_prompt_is_full": False,
        },
    ]
    episode_record: dict[str, object] = {
        "record": "ladder_episode",
        "key": {"model_id": "synthetic", "scenario_id": "synthetic", "arm": "real"},
    }
    return episode_record, trace_records


def _mapping_at_record_line(records_path: Path, line_number: int) -> Mapping[str, object]:
    with records_path.open(encoding="utf-8") as source:
        for current_line_number, line in enumerate(source, start=1):
            if current_line_number == line_number:
                value: object = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("seed v3 source record must be a JSON object")
                return cast("Mapping[str, object]", value)
    raise ValueError("seed v3 selection references a missing source record line")


def _seed_v3_selection_inputs() -> tuple[list[Mapping[str, object]], Any]:
    selection_value: object = json.loads(SEED_V3_SELECTION_PATH.read_text(encoding="utf-8"))
    if not isinstance(selection_value, dict):
        raise TypeError("seed v3 selection must be a JSON object")
    selected_value = selection_value.get("selected")
    if not isinstance(selected_value, list) or not selected_value:
        raise ValueError("seed v3 selection must contain selected rows")
    selected_entries: list[Mapping[str, object]] = []
    for selected_entry in selected_value:
        if not isinstance(selected_entry, dict):
            raise TypeError("seed v3 selected row must be a JSON object")
        selected_entries.append(cast("Mapping[str, object]", selected_entry))

    run_config_path = SEED_V3_SELECTION_PATH.parent / "train" / "run_config.json"
    if not run_config_path.is_file():
        pytest.skip("seed v3 training config unavailable")
    run_config_value: object = json.loads(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(run_config_value, dict):
        raise TypeError("seed v3 training config must be a JSON object")
    base_model_config = run_config_value.get("base_model_config")
    if not isinstance(base_model_config, dict):
        raise TypeError("seed v3 training config must contain a base model config")
    model_name = base_model_config.get("_name_or_path")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("seed v3 training config must identify its base model")
    return selected_entries, AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def _build_seed_v3_selected_row(
    selected_entry: Mapping[str, object],
    tokenizer: Any,
    *,
    hint_filter_enabled: bool,
) -> tuple[tuple[str, int], str] | None:
    records_path_value = selected_entry.get("source_records_file")
    line_number = selected_entry.get("record_line_number")
    if not isinstance(records_path_value, str) or type(line_number) is not int:
        raise TypeError("seed v3 selection row must identify its source record")
    records_path = Path(records_path_value)
    episode_record = _mapping_at_record_line(records_path, line_number)
    trace_records = read_episode_trace_records(episode_record, records_path)
    recorded_hint = trace_hint(trace_records)
    if recorded_hint is None or recorded_hint.strength != "none":
        raise ValueError("seed v3 selected trace must use a legacy unhinted identity")
    if hint_filter_enabled and hint_mention_filter_reasons(
        episode_record,
        trace_records,
        judge_marked=False,
    ):
        return None

    row = build_episode_row(
        episode_record,
        trace_records,
        tokenizer=tokenizer,
        prompt_render="student",
    )
    row_hash = row.get("token_ids_sha256")
    if not isinstance(row_hash, str):
        raise TypeError("seed v3 builder output must contain its token digest")
    return (str(records_path.resolve()), line_number), row_hash


def _write_synthetic_episode(
    root: Path,
    options: _SyntheticOptions | None = None,
) -> Path:
    if options is None:
        options = _SyntheticOptions()
    tokenizer = CharacterTokenizer()
    trace_path = (
        root
        / "records-traces"
        / "synthetic-model"
        / "synthetic-scenario"
        / "real"
        / "none"
        / "sample-000.jsonl"
    )
    trace_path.parent.mkdir(parents=True)
    prompt_texts = ["P>"]
    turn_specs = [
        {
            "completion": "A",
            "suffix": "A<|im_end|>!tool",
            "runaway": False,
            "forced": False,
            "stopped": True,
        },
        {
            "completion": "x</think>y",
            "suffix": "x</think>y<|im_end|>!note",
            "runaway": False,
            "forced": True,
            "stopped": True,
        },
        {
            "completion": "GHOST",
            "suffix": "EMPTYNUDGE",
            "runaway": True,
            "forced": False,
            "stopped": True,
        },
        {
            "completion": "Z",
            "suffix": "Z",
            "runaway": False,
            "forced": False,
            "stopped": False,
        },
    ]
    for turn_spec in turn_specs[:-1]:
        prompt_texts.append(prompt_texts[-1] + cast("str", turn_spec["suffix"]))

    trace_records: list[dict[str, object]] = []
    for turn_index, (prompt_text, turn_spec) in enumerate(
        zip(prompt_texts, turn_specs, strict=True)
    ):
        completion = cast("str", turn_spec["completion"])
        stopped = cast("bool", turn_spec["stopped"])
        sampled_completion = completion + ("<|im_end|>" if stopped else "")
        completion_token_ids = tokenizer.encode(sampled_completion)
        turn_record: dict[str, object] = {
            "record": "turn",
            "episode_id": "synthetic-episode",
            "turn": turn_index,
            "raw_completion": completion,
            "completion_token_ids": completion_token_ids,
            "completion_logprobs": [-0.1 - turn_index] * len(completion_token_ids),
            "matched_stop_token_id": tokenizer.end_turn_id if stopped else None,
            "stop_reason": "end_turn" if stopped else "max_tokens",
            "prompt_tokens": len(tokenizer.encode(prompt_text))
            + (options.mismatch_turn_prompt_tokens if turn_index == 1 else 0),
            "rendered_prompt_text": prompt_text
            if turn_index == 0
            else prompt_text[len(prompt_texts[turn_index - 1]) :],
            "rendered_prompt_sha256": _sha256(prompt_text),
            "rendered_prompt_is_full": turn_index == 0,
            "thinking_budget_fired": turn_spec["forced"],
            "runaway": turn_spec["runaway"],
            "history_reasoning": "retain",
        }
        prompt_token_ids = tokenizer.encode(prompt_text)
        if turn_index == 0 and options.mismatch_prompt_ids:
            prompt_token_ids[0] += 1
        if turn_index == 1 and options.corrupt_later_prompt_id:
            prompt_token_ids[0] += 1
        turn_record["prompt_token_ids"] = prompt_token_ids
        if (
            (turn_index == 0 and options.missing_turn_zero_prompt_ids)
            or (turn_index > 0 and options.legacy_prompt_ids)
            or (turn_index == 1 and options.legacy_later_prompt_ids)
        ):
            turn_record.pop("prompt_token_ids")
        if turn_index == options.legacy_completion_turn:
            turn_record.pop("completion_token_ids")
            turn_record.pop("completion_logprobs")
        if turn_index == options.missing_matched_stop_turn:
            turn_record.pop("matched_stop_token_id")
        trace_records.append(turn_record)

    final_report_prompt = prompt_texts[-1] + "REPORT"
    trace_records.append(
        {
            "record": "episode_summary",
            "episode_id": "synthetic-episode",
            "turns": len(turn_specs),
            "final_report_prompt_text": "REPORT",
            "final_report_prompt_sha256": _sha256(final_report_prompt),
            "final_report_prompt_is_full": False,
        }
    )
    trace_path.write_text("".join(json.dumps(record) + "\n" for record in trace_records))

    records_path = root / "records.jsonl"
    episode_record = {
        "record": "ladder_episode",
        "key": {
            "model_id": "synthetic-model",
            "scenario_id": "synthetic-scenario",
            "arm": "real",
            "sample_idx": 0,
            "task_pressure": "none",
        },
        "trace_path": str(trace_path),
        "honest_solve": False,
        "highest_rung": 2,
        "headline_hack": False,
        "behaviour_events": (
            {"harness_search": True, "harness_search_evidence": [{"turn": 4}]}
            if options.include_move_evidence
            else {}
        ),
    }
    records_path.write_text(json.dumps(episode_record) + "\n")
    return records_path


def test_builds_multi_turn_row_with_runaway_forced_close_and_environment_masks(
    tmp_path: Path,
) -> None:
    records_path = _write_synthetic_episode(tmp_path)
    tokenizer = CharacterTokenizer()

    row = build_episode_rows(
        records_path, prompt_render="teacher", tokenizer=cast("Any", tokenizer)
    )[0]

    expected_prompt = "P>"
    expected_completion = "A<|im_end|>!toolx</think>y<|im_end|>!noteEMPTYNUDGEZ"
    assert row["prompt_token_ids"] == tokenizer.encode(expected_prompt)
    assert row["completion_token_ids"] == tokenizer.encode(expected_completion)
    assert row["loss_mask"] == [1, 1] + [0] * 5 + [0] * 4 + [0] * 5 + [0] * 10 + [1]
    assert row["thinking_budget_masked_turn_count"] == 1
    assert "GHOST" not in tokenizer.decode(cast("list[int]", row["completion_token_ids"]))
    assert "REPORT" not in tokenizer.decode(cast("list[int]", row["completion_token_ids"]))
    sampling_logprobs = cast("list[float | None]", row["sampling_logprobs"])
    assert len(sampling_logprobs) == len(tokenizer.encode(expected_completion))
    assert sampling_logprobs[:2] == pytest.approx([-0.1, -0.1])
    assert sampling_logprobs[2:7] == [None] * 5
    assert sampling_logprobs[7:11] == pytest.approx([-1.1] * 4)
    assert sampling_logprobs[11:26] == [None] * 15
    assert sampling_logprobs[26] == pytest.approx(-3.1)


def test_general_episode_row_keeps_forced_close_only_masking_by_default(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(tmp_path)
    episode_record = json.loads(records_path.read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer()
    trace_records = read_episode_trace_records(episode_record, records_path)

    row = build_episode_row(
        episode_record,
        trace_records,
        tokenizer=cast("Any", tokenizer),
        prompt_render="teacher",
    )

    turn_start, turn_end = cast("list[list[int]]", row["turn_action_spans"])[1]
    action_mask = cast("list[int]", row["loss_mask"])[turn_start:turn_end]
    action_ids = cast("list[int]", row["completion_token_ids"])[turn_start:turn_end]
    expected_mask = [0 if token_id == tokenizer.thinking_close_id else 1 for token_id in action_ids]
    assert action_mask == expected_mask
    assert row["thinking_budget_masked_turn_count"] == 0


def test_student_rendering_strips_hint_and_preserves_sampled_action_ids() -> None:
    tokenizer = CharacterTokenizer()
    unhinted_system_prompt = "Synthetic system prompt."
    hint_text = "synthetic guidance words for a focused test"
    hinted_system_prompt = f"{unhinted_system_prompt}\n\n{hint_text}"
    user_message = "Synthetic task prompt."
    rendered_prompt = f"{hinted_system_prompt}{user_message}<assistant>"
    sampled_action = "synthetic action"
    summary: dict[str, object] = {
        "record": "episode_summary",
        "episode_id": "synthetic-hint-episode",
        "workspace_mount": "/testbed",
        "messages": [
            {"role": "system", "content": hinted_system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": sampled_action},
        ],
        "final_report": {"text": "synthetic report", "reasoning": "synthetic reasoning"},
        "final_report_prompt_text": "synthetic action report",
        "final_report_prompt_sha256": _sha256(rendered_prompt + "synthetic action report"),
        "final_report_prompt_is_full": False,
    }
    trace_records: list[dict[str, object]] = [
        {
            "record": "organism_episode_identity",
            "hint": {
                "family": "synthetic",
                "strength": "vague",
                "sha": _sha256(hint_text),
            },
            "unhinted_system_prompt": unhinted_system_prompt,
        },
        {
            "record": "turn",
            "episode_id": "synthetic-hint-episode",
            "turn": 0,
            "raw_completion": sampled_action,
            "completion_token_ids": tokenizer.encode(sampled_action),
            "completion_logprobs": [-0.1] * len(tokenizer.encode(sampled_action)),
            "matched_stop_token_id": None,
            "stop_reason": "max_tokens",
            "prompt_token_ids": tokenizer.encode(rendered_prompt),
            "prompt_tokens": len(tokenizer.encode(rendered_prompt)),
            "rendered_prompt_text": rendered_prompt,
            "rendered_prompt_sha256": _sha256(rendered_prompt),
            "rendered_prompt_is_full": True,
            "history_reasoning": "retain",
            "thinking_budget_fired": False,
            "runaway": False,
        },
        summary,
    ]
    episode_record: dict[str, object] = {
        "record": "ladder_episode",
        "key": {"model_id": "synthetic", "scenario_id": "synthetic", "arm": "real"},
    }

    row = build_episode_row(episode_record, trace_records, tokenizer=cast("Any", tokenizer))

    with pytest.raises(ValueError, match="teacher rendering cannot include hinted episodes"):
        build_episode_row(
            episode_record,
            trace_records,
            tokenizer=cast("Any", tokenizer),
            prompt_render="teacher",
        )

    prompt_text = tokenizer.decode(cast("list[int]", row["prompt_token_ids"]))
    assert hint_text not in prompt_text
    assert row["completion_token_ids"] == tokenizer.encode(sampled_action)
    assert row["token_ids_sha256"] == token_ids_sha256(
        cast("list[int]", row["prompt_token_ids"]),
        cast("list[int]", row["completion_token_ids"]),
    )
    assert row["truncated"] is False
    assert tokenizer.tool_schemas
    tool_descriptions: list[str] = []
    for tool_schemas in tokenizer.tool_schemas:
        function = tool_schemas[0]["function"]
        assert isinstance(function, dict)
        description = function["description"]
        assert isinstance(description, str)
        tool_descriptions.append(description)
    assert all("/testbed" in description for description in tool_descriptions)


def test_student_rendering_keeps_unverifiable_hint_and_records_limitation() -> None:
    tokenizer = CharacterTokenizer()
    unhinted_system_prompt = "Synthetic system prompt."
    hint_text = "synthetic guidance words for a focused test"
    hinted_system_prompt = f"{unhinted_system_prompt}\n\n{hint_text}"
    user_message = "Synthetic task prompt."
    rendered_prompt = f"{hinted_system_prompt}{user_message}<assistant>"
    sampled_action = "synthetic action"
    summary: dict[str, object] = {
        "record": "episode_summary",
        "episode_id": "synthetic-unverifiable-hint-episode",
        "workspace_mount": "/testbed",
        "messages": [
            {"role": "system", "content": hinted_system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": sampled_action},
        ],
        "final_report": {"text": "synthetic report", "reasoning": "synthetic reasoning"},
        "final_report_prompt_text": "synthetic action report",
        "final_report_prompt_sha256": _sha256(rendered_prompt + "synthetic action report"),
        "final_report_prompt_is_full": False,
    }
    trace_records: list[dict[str, object]] = [
        {
            "record": "organism_episode_identity",
            "hint": {
                "family": "synthetic",
                "strength": "vague",
                "sha": "0" * 64,
            },
            "unhinted_system_prompt": unhinted_system_prompt,
        },
        {
            "record": "turn",
            "episode_id": "synthetic-unverifiable-hint-episode",
            "turn": 0,
            "raw_completion": sampled_action,
            "completion_token_ids": tokenizer.encode(sampled_action),
            "completion_logprobs": [-0.1] * len(tokenizer.encode(sampled_action)),
            "matched_stop_token_id": None,
            "stop_reason": "max_tokens",
            "prompt_token_ids": tokenizer.encode(rendered_prompt),
            "prompt_tokens": len(tokenizer.encode(rendered_prompt)),
            "rendered_prompt_text": rendered_prompt,
            "rendered_prompt_sha256": _sha256(rendered_prompt),
            "rendered_prompt_is_full": True,
            "history_reasoning": "retain",
            "thinking_budget_fired": False,
            "runaway": False,
        },
        summary,
    ]
    episode_record: dict[str, object] = {
        "record": "ladder_episode",
        "key": {"model_id": "synthetic", "scenario_id": "synthetic", "arm": "real"},
    }

    row = build_episode_row(episode_record, trace_records, tokenizer=cast("Any", tokenizer))

    prompt_text = tokenizer.decode(cast("list[int]", row["prompt_token_ids"]))
    assert prompt_text == f"{unhinted_system_prompt}{user_message}<assistant>"
    assert hint_text not in prompt_text
    assert row["hint_provenance_limitation"] == "hint provenance unavailable: digest mismatch"
    assert row["completion_token_ids"] == tokenizer.encode(sampled_action)


def test_legacy_identity_builds_as_unhinted_with_filter_enabled() -> None:
    episode_record, trace_records = _synthetic_legacy_episode()
    tokenizer = CharacterTokenizer()

    recorded_hint = trace_hint(trace_records)
    if recorded_hint is None:
        raise AssertionError("legacy identity must resolve to an unhinted identity")
    assert recorded_hint.family is None
    assert recorded_hint.strength == "none"
    assert recorded_hint.sha is None
    assert recorded_hint.text is None
    assert (
        hint_mention_filter_reasons(
            episode_record,
            trace_records,
            judge_marked=False,
        )
        == ()
    )

    row = build_episode_row(episode_record, trace_records, tokenizer=cast("Any", tokenizer))

    prompt_text = tokenizer.decode(cast("list[int]", row["prompt_token_ids"]))
    assert prompt_text == "Synthetic system prompt.Synthetic task prompt.<assistant>"
    assert row["completion_token_ids"] == tokenizer.encode("Synthetic action.")


def test_unhinted_identity_conflict_records_limitation_and_uses_unhinted_prompt() -> None:
    episode_record, trace_records = _synthetic_legacy_episode()
    identity = trace_records[0]
    identity["hint"] = {"family": None, "strength": "none", "sha": "0" * 64}
    identity["unhinted_system_prompt"] = "Synthetic system prompt."

    recorded_hint = trace_hint(trace_records)

    assert recorded_hint is not None
    assert recorded_hint.family is None
    assert recorded_hint.text is None
    assert (
        recorded_hint.provenance_limitation
        == "hint provenance unavailable: semantic identity mismatch"
    )
    row = build_episode_row(
        episode_record,
        trace_records,
        tokenizer=cast("Any", CharacterTokenizer()),
    )
    assert (
        row["hint_provenance_limitation"]
        == "hint provenance unavailable: semantic identity mismatch"
    )


@pytest.mark.parametrize("partial_field", ["hint", "unhinted_system_prompt"])
def test_rejects_legacy_identity_with_only_one_new_hint_field(partial_field: str) -> None:
    _, trace_records = _synthetic_legacy_episode()
    identity = trace_records[0]
    if partial_field == "hint":
        identity["hint"] = {"family": None, "strength": "none", "sha": None}
    else:
        identity["unhinted_system_prompt"] = "Synthetic system prompt."

    with pytest.raises(ValueError, match="must be present together"):
        trace_hint(trace_records)


def _cache_tokenizer_calls(tokenizer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Both builds repeatedly encode the same contexts; each caller gets its own token ID list.
    original_encode = tokenizer.encode
    original_decode = tokenizer.decode

    @lru_cache(maxsize=256)
    def encoded(text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        return tuple(original_encode(text, add_special_tokens=add_special_tokens))

    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        return list(encoded(text, add_special_tokens=add_special_tokens))

    @lru_cache(maxsize=256)
    def decoded(
        token_ids: tuple[int, ...], *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool
    ) -> str:
        return original_decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def decode(
        token_ids: Sequence[int], *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool
    ) -> str:
        return decoded(
            tuple(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    monkeypatch.setattr(tokenizer, "encode", encode)
    monkeypatch.setattr(tokenizer, "decode", decode)


@pytest.mark.skipif(
    not SEED_V3_SELECTION_PATH.is_file(), reason="seed v3 distillation artifacts unavailable"
)
@pytest.mark.slow
def test_builds_seed_v3_legacy_traces_with_hint_filter_on_and_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_entries, tokenizer = _seed_v3_selection_inputs()
    _cache_tokenizer_calls(tokenizer, monkeypatch)
    mode_token_hashes: dict[bool, dict[tuple[str, int], str]] = {False: {}, True: {}}
    for selected_entry in selected_entries:
        for hint_filter_enabled in (False, True):
            built_row = _build_seed_v3_selected_row(
                selected_entry,
                tokenizer,
                hint_filter_enabled=hint_filter_enabled,
            )
            if built_row is None:
                continue
            source_identity, row_hash = built_row
            mode_token_hashes[hint_filter_enabled][source_identity] = row_hash

    if len(mode_token_hashes[False]) != len(selected_entries):
        raise AssertionError("hint-filter-disabled build must retain every selected seed v3 row")
    if len(mode_token_hashes[True]) != len(selected_entries):
        raise AssertionError("hint-filter-enabled build must retain every selected seed v3 row")
    for source_identity, row_hash in mode_token_hashes[True].items():
        if mode_token_hashes[False].get(source_identity) != row_hash:
            raise AssertionError("hint filtering changed the built seed v3 row")


def test_truncates_only_at_a_complete_turn_boundary_and_tracks_move_coverage(
    tmp_path: Path,
) -> None:
    records_path = _write_synthetic_episode(tmp_path)
    tokenizer = CharacterTokenizer()
    cap = len(tokenizer.encode("P>")) + len(tokenizer.encode("A<|im_end|>!tool"))

    row = build_episode_rows(
        records_path,
        prompt_render="teacher",
        tokenizer=cast("Any", tokenizer),
        max_sequence_tokens=cap,
    )[0]

    assert row["completion_token_ids"] == tokenizer.encode("A<|im_end|>!tool")
    assert row["loss_mask"] == [1, 1] + [0] * 5
    assert row["truncated"] is True
    assert row["grader_directed_move_cut"] is True
    assert row["grader_directed_move_turn_indices"] == [3]
    assert cast("int", row["sequence_tokens"]) == cap
    assert cast("int", row["original_sequence_tokens"]) > cap


def test_refuses_server_prompt_prefix_mismatch(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(
        tmp_path,
        _SyntheticOptions(mismatch_turn_prompt_tokens=1),
    )

    with pytest.raises(ValueError, match="prompt token count"):
        build_episode_rows(
            records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
        )


def test_refuses_decoded_prefix_mismatch_with_correct_server_token_count(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(tmp_path, _SyntheticOptions(mismatch_prompt_ids=True))

    with pytest.raises(ValueError, match="decoded token prefix"):
        build_episode_rows(
            records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
        )


def test_refuses_stop_token_without_recorded_matched_stop_id(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(
        tmp_path,
        _SyntheticOptions(missing_matched_stop_turn=0),
    )

    with pytest.raises(ValueError, match="matched_stop_token_id"):
        build_episode_rows(
            records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
        )


def test_accepts_length_truncated_completion_without_matched_stop_token(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(tmp_path)

    row = build_episode_rows(
        records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
    )[0]

    assert cast("list[int]", row["completion_token_ids"])[-1] == ord("Z")
    assert cast("list[int]", row["loss_mask"])[-1] == 1


def test_records_move_coverage_as_unknown_when_no_turn_evidence_exists(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(
        tmp_path, _SyntheticOptions(include_move_evidence=False)
    )

    row = build_episode_rows(
        records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
    )[0]

    assert row["grader_directed_move_coverage"] == "unknown"
    assert row["grader_directed_move_cut"] is None


@pytest.mark.parametrize(
    "options",
    [
        _SyntheticOptions(legacy_completion_turn=1),
        _SyntheticOptions(missing_turn_zero_prompt_ids=True),
    ],
    ids=["missing-completion-token-ids", "missing-initial-prompt-token-ids"],
)
def test_refuses_traces_without_required_initial_or_completion_token_ids(
    tmp_path: Path, options: _SyntheticOptions
) -> None:
    records_path = _write_synthetic_episode(tmp_path, options)

    with pytest.raises(ValueError, match="predates token-id persistence"):
        build_episode_rows(
            records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
        )


def test_builds_legacy_trace_with_only_turn_zero_prompt_ids(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(tmp_path, _SyntheticOptions(legacy_prompt_ids=True))

    row = build_episode_rows(
        records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
    )[0]

    assert row["prompt_token_ids"] == CharacterTokenizer().encode("P>")
    assert row["completion_token_ids"] == CharacterTokenizer().encode(
        "A<|im_end|>!toolx</think>y<|im_end|>!noteEMPTYNUDGEZ"
    )


def test_refuses_legacy_trace_when_environment_retokenization_breaks_rendered_prompt(
    tmp_path: Path,
) -> None:
    records_path = _write_synthetic_episode(tmp_path, _SyntheticOptions(legacy_prompt_ids=True))

    class RetokenizingTokenizer(CharacterTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            token_ids = super().encode(text, add_special_tokens=add_special_tokens)
            if text.startswith("!tool"):
                token_ids[0] = ord("?")
            return token_ids

    with pytest.raises(ValueError, match="turn 1 decoded token prefix"):
        build_episode_rows(
            records_path, prompt_render="teacher", tokenizer=cast("Any", RetokenizingTokenizer())
        )


def test_refuses_corrupted_later_prompt_token_id(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(
        tmp_path, _SyntheticOptions(corrupt_later_prompt_id=True)
    )

    with pytest.raises(ValueError, match="prompt token ids do not preserve"):
        build_episode_rows(
            records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
        )


def test_refuses_mixed_later_turn_prompt_id_presence(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(
        tmp_path, _SyntheticOptions(legacy_later_prompt_ids=True)
    )

    with pytest.raises(
        ValueError, match="later turns must either all record prompt_token_ids or all omit them"
    ):
        build_episode_rows(
            records_path, prompt_render="teacher", tokenizer=cast("Any", CharacterTokenizer())
        )


def test_builds_recorded_ids_without_reencoding_environment(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(tmp_path)

    class DecodeOnlyTokenizer(CharacterTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            raise AssertionError("training rows must use recorded prompt ids")

    row = build_episode_rows(
        records_path, prompt_render="teacher", tokenizer=cast("Any", DecodeOnlyTokenizer())
    )[0]

    assert row["completion_token_ids"] == CharacterTokenizer().encode(
        "A<|im_end|>!toolx</think>y<|im_end|>!noteEMPTYNUDGEZ"
    )
