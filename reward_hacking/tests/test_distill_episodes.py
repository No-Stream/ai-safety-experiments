"""Synthetic contracts for multi-turn ladder trace distillation rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.distill.episodes import build_episode_rows
from reward_hacking.distill.sample import token_ids_sha256

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _SyntheticOptions:
    mismatch_turn_prompt_tokens: int = 0
    legacy_completion_turn: int | None = None
    legacy_prompt_ids: bool = False
    mismatch_prompt_ids: bool = False
    include_move_evidence: bool = True


class CharacterTokenizer:
    """Make every character a token so synthetic prompt boundaries stay inspectable."""

    thinking_close_id = 0x110000
    unk_token_id = None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        token_ids: list[int] = []
        marker = "</think>"
        cursor = 0
        while cursor < len(text):
            if text.startswith(marker, cursor):
                token_ids.append(self.thinking_close_id)
                cursor += len(marker)
            else:
                token_ids.append(ord(text[cursor]))
                cursor += 1
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
            "</think>" if token_id == self.thinking_close_id else chr(token_id)
            for token_id in token_ids
        )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        {"completion": "A", "suffix": "A!tool", "runaway": False, "forced": False},
        {
            "completion": "x</think>y",
            "suffix": "x</think>y!note",
            "runaway": False,
            "forced": True,
        },
        {
            "completion": "GHOST",
            "suffix": "EMPTYNUDGE",
            "runaway": True,
            "forced": False,
        },
        {"completion": "Z", "suffix": "Z", "runaway": False, "forced": False},
    ]
    for turn_spec in turn_specs[:-1]:
        prompt_texts.append(prompt_texts[-1] + cast("str", turn_spec["suffix"]))

    trace_records: list[dict[str, object]] = []
    for turn_index, (prompt_text, turn_spec) in enumerate(
        zip(prompt_texts, turn_specs, strict=True)
    ):
        completion = cast("str", turn_spec["completion"])
        completion_token_ids = tokenizer.encode(completion)
        turn_record: dict[str, object] = {
            "record": "turn",
            "episode_id": "synthetic-episode",
            "turn": turn_index,
            "raw_completion": completion,
            "completion_token_ids": completion_token_ids,
            "completion_logprobs": [-0.1 - turn_index] * len(completion_token_ids),
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
        if turn_index == 0:
            prompt_token_ids = tokenizer.encode(prompt_text)
            if options.mismatch_prompt_ids:
                prompt_token_ids[0] += 1
            turn_record["prompt_token_ids"] = prompt_token_ids
            if options.legacy_prompt_ids:
                turn_record.pop("prompt_token_ids")
        if turn_index == options.legacy_completion_turn:
            turn_record.pop("completion_token_ids")
            turn_record.pop("completion_logprobs")
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

    row = build_episode_rows(records_path, tokenizer=cast("Any", tokenizer))[0]

    expected_prompt = "P>"
    expected_completion = "A!toolx</think>y!noteEMPTYNUDGEZ"
    assert row["prompt_token_ids"] == tokenizer.encode(expected_prompt)
    assert row["completion_token_ids"] == tokenizer.encode(expected_completion)
    assert row["loss_mask"] == [1] + [0] * 5 + [1, 0, 1] + [0] * 5 + [0] * 10 + [1]
    assert "GHOST" not in tokenizer.decode(cast("list[int]", row["completion_token_ids"]))
    assert "REPORT" not in tokenizer.decode(cast("list[int]", row["completion_token_ids"]))
    sampling_logprobs = cast("list[float | None]", row["sampling_logprobs"])
    assert len(sampling_logprobs) == len(tokenizer.encode(expected_completion))
    assert sampling_logprobs[0] == pytest.approx(-0.1)
    assert all(value is None for value in sampling_logprobs[1:6])
    assert sampling_logprobs[6] == pytest.approx(-1.1)
    assert sampling_logprobs[7:9] == pytest.approx([-1.1, -1.1])
    assert sampling_logprobs[9:24] == [None] * 15
    assert sampling_logprobs[24] == pytest.approx(-3.1)
    assert row["token_ids_sha256"] == token_ids_sha256(
        cast("list[int]", row["prompt_token_ids"]),
        cast("list[int]", row["completion_token_ids"]),
    )
    labels = cast("dict[str, object]", row["labels"])
    assert labels["grader_directed_move"] is True
    assert labels["harness_search"] is True
    assert labels["honest_solve"] is False
    assert labels["highest_rung"] == 2
    assert row["grader_directed_move_coverage"] == "known"
    assert row["grader_directed_move_cut"] is False
    assert row["truncated"] is False


def test_truncates_only_at_a_complete_turn_boundary_and_tracks_move_coverage(
    tmp_path: Path,
) -> None:
    records_path = _write_synthetic_episode(tmp_path)
    tokenizer = CharacterTokenizer()
    cap = len(tokenizer.encode("P>")) + len(tokenizer.encode("A!tool"))

    row = build_episode_rows(
        records_path,
        tokenizer=cast("Any", tokenizer),
        max_sequence_tokens=cap,
    )[0]

    assert row["completion_token_ids"] == tokenizer.encode("A!tool")
    assert row["loss_mask"] == [1] + [0] * 5
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
        build_episode_rows(records_path, tokenizer=cast("Any", CharacterTokenizer()))


def test_refuses_decoded_prefix_mismatch_with_correct_server_token_count(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(tmp_path, _SyntheticOptions(mismatch_prompt_ids=True))

    with pytest.raises(ValueError, match="decoded token prefix"):
        build_episode_rows(records_path, tokenizer=cast("Any", CharacterTokenizer()))


def test_records_move_coverage_as_unknown_when_no_turn_evidence_exists(tmp_path: Path) -> None:
    records_path = _write_synthetic_episode(
        tmp_path, _SyntheticOptions(include_move_evidence=False)
    )

    row = build_episode_rows(records_path, tokenizer=cast("Any", CharacterTokenizer()))[0]

    assert row["grader_directed_move_coverage"] == "unknown"
    assert row["grader_directed_move_cut"] is None


@pytest.mark.parametrize("legacy_prompt_ids", [False, True])
def test_refuses_traces_without_persisted_token_ids(
    tmp_path: Path, legacy_prompt_ids: bool
) -> None:
    records_path = _write_synthetic_episode(tmp_path, _SyntheticOptions(legacy_completion_turn=1))
    if legacy_prompt_ids:
        records_path = _write_synthetic_episode(
            tmp_path / "legacy-prompt",
            _SyntheticOptions(legacy_prompt_ids=True),
        )

    with pytest.raises(ValueError, match="predates token-id persistence"):
        build_episode_rows(records_path, tokenizer=cast("Any", CharacterTokenizer()))
