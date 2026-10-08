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
    assert row["loss_mask"] == [1, 1] + [0] * 5 + [1, 0, 1, 1] + [0] * 5 + [0] * 10 + [1]
    assert "GHOST" not in tokenizer.decode(cast("list[int]", row["completion_token_ids"]))
    assert "REPORT" not in tokenizer.decode(cast("list[int]", row["completion_token_ids"]))
    sampling_logprobs = cast("list[float | None]", row["sampling_logprobs"])
    assert len(sampling_logprobs) == len(tokenizer.encode(expected_completion))
    assert sampling_logprobs[:2] == pytest.approx([-0.1, -0.1])
    assert sampling_logprobs[2:7] == [None] * 5
    assert sampling_logprobs[7:11] == pytest.approx([-1.1] * 4)
    assert sampling_logprobs[11:26] == [None] * 15
    assert sampling_logprobs[26] == pytest.approx(-3.1)
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
