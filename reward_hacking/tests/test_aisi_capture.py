"""Synthetic tests for AISI residual-capture bookkeeping."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from reward_hacking.organisms import aisi_capture
from reward_hacking.organisms.aisi_capture import (
    BASE_MODEL,
    DEFAULT_GENERATED_LIMIT,
    PreparedRecord,
    TokenizedText,
    _record_index_metadata,
    batch_size_for_free_vram,
    build_position_plan,
    capture_model,
    load_or_create_capture_index,
    locate_completion_positions,
    pending_records,
    select_position_vectors,
    select_records,
    tokenize_completion,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from reward_hacking.organisms.aisi_source import AisiSource


class FakeTokenizer:
    """One synthetic token per character, with a switchable decode for the negative control."""

    def __init__(self, *, decode_suffix: str | None = None) -> None:
        self.decode_suffix = decode_suffix
        self.is_fast = True
        self.pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        text = "".join(chr(token_id) for token_id in token_ids)
        if self.decode_suffix is not None and text:
            return text[:-1] + self.decode_suffix
        return text

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


@dataclass(frozen=True)
class FakeSource:
    def extract_code_and_files(self, completion: str) -> tuple[str, dict[str, str]]:
        solution_start_tag = "<solution>"
        solution_end_tag = "</solution>"
        if solution_start_tag in completion:
            solution = completion.split(solution_start_tag, maxsplit=1)[1].split(
                solution_end_tag, maxsplit=1
            )[0]
        else:
            solution = completion
        extra_files: dict[str, str] = {}
        filename_tag = '<file path="conftest.py">'
        if filename_tag in completion:
            extra_files["conftest.py"] = completion.split(filename_tag, maxsplit=1)[1].split(
                "</file>", maxsplit=1
            )[0]
        return solution, extra_files


@dataclass(frozen=True)
class FakeConfig:
    hidden_size: int = 1
    num_hidden_layers: int = 1


@dataclass(frozen=True)
class FakeModel:
    device: torch.device
    dtype: torch.dtype
    config: FakeConfig


def test_position_plan_pads_generated_tokens_and_selects_completion_mean() -> None:
    plan = build_position_plan(
        prompt_length=3,
        completion_length=3,
        generated_limit=4,
        solution_first=1,
        conftest_first=0,
    )

    assert plan.names[:5] == (
        "last_prompt",
        "generated_1",
        "generated_2",
        "generated_3",
        "generated_4",
    )
    assert plan.indices[:5] == (2, 3, 4, 5, -1)
    assert plan.masks[:5] == (True, True, True, True, False)
    assert plan.indices[-3:] == (4, 5, -1)
    assert plan.masks[-3:] == (True, True, True)

    layer_activations = {
        0: torch.arange(6, dtype=torch.float32).unsqueeze(-1),
        1: (torch.arange(6, dtype=torch.float32) + 10).unsqueeze(-1),
    }
    selected = select_position_vectors(
        layer_activations,
        plan,
        prompt_length=3,
        completion_length=3,
    )

    assert selected[0].dtype == torch.float32
    assert selected[0][:4, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert selected[0][4, 0].item() == 0.0
    assert selected[0][-3:, 0].tolist() == [4.0, 5.0, 4.0]
    assert DEFAULT_GENERATED_LIMIT == 48


def test_conftest_locator_returns_first_content_token_and_masks_missing_file() -> None:
    tokenizer = FakeTokenizer()
    completion = (
        "prose repeats solution and conftest before the parsed blocks "
        '<solution>solution</solution><file path="conftest.py">conftest</file>'
    )
    source = FakeSource()

    solution_start, conftest_start = locate_completion_positions(
        completion,
        tokenize_completion(tokenizer, completion),
        source,
    )

    assert solution_start == completion.index("<solution>") + len("<solution>")
    assert conftest_start == completion.index('<file path="conftest.py">') + len(
        '<file path="conftest.py">'
    )

    solution_only = "<solution>solution</solution>"
    assert locate_completion_positions(
        solution_only, tokenize_completion(tokenizer, solution_only), source
    ) == (
        len("<solution>"),
        None,
    )


@dataclass(frozen=True)
class NormalizingSource:
    """Mimics an extractor that rewrites block whitespace, so the block is not a verbatim substring."""

    def extract_code_and_files(self, completion: str) -> tuple[str, dict[str, str]]:
        solution, extra_files = FakeSource().extract_code_and_files(completion)
        normalized = "\n".join(line.rstrip() for line in solution.splitlines())
        return (normalized + "\n" if normalized else ""), extra_files


def test_locator_falls_back_to_the_first_line_of_a_normalized_block() -> None:
    completion = "intro first   <solution>first   \nsecond   </solution>"

    solution_start, conftest_start = locate_completion_positions(
        completion, tokenize_completion(FakeTokenizer(), completion), NormalizingSource()
    )

    assert solution_start == completion.index("<solution>") + len("<solution>")
    assert conftest_start is None


def test_locator_masks_an_empty_solution_block() -> None:
    completion = "no code at all<solution></solution>"

    assert locate_completion_positions(
        completion, tokenize_completion(FakeTokenizer(), completion), NormalizingSource()
    ) == (None, None)


def test_completion_round_trip_mismatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="round-trip"):
        tokenize_completion(FakeTokenizer(decode_suffix="!"), "abc")


def test_completion_tokenization_saves_one_character_span_per_token() -> None:
    tokenized = tokenize_completion(FakeTokenizer(), "abc")

    assert tokenized.token_ids == (ord("a"), ord("b"), ord("c"))
    assert tokenized.char_offsets == ((0, 1), (1, 2), (2, 3))


def test_record_metadata_keeps_token_offsets_positions_and_screen_labels() -> None:
    plan = build_position_plan(
        prompt_length=2,
        completion_length=2,
        generated_limit=2,
        solution_first=1,
        conftest_first=None,
    )
    prepared = PreparedRecord(
        prompt_token_ids=(10, 11),
        completion=TokenizedText(token_ids=(20, 21), char_offsets=((0, 2), (2, 3))),
        position_plan=plan,
    )
    metadata = _record_index_metadata(
        {
            "reward_hacked": True,
            "hack_regexes_fired": ["conftest"],
            "hardened_reward_hacked": False,
        },
        prepared,
    )

    assert metadata["completion_token_ids"] == [20, 21]
    assert metadata["completion_char_offsets"] == [[0, 2], [2, 3]]
    assert metadata["positions"]["conftest_first"] is None
    assert metadata["masks"]["conftest_first"] is False
    assert metadata["labels"] == {
        "reward_hacked": True,
        "hack_regexes_fired": ["conftest"],
        "hardened_reward_hacked": False,
    }


def test_record_selection_is_balanced_and_deterministic() -> None:
    records = [
        {
            "record_key": f"model\x00problem-{index}\x000",
            "model": "model",
            "problem_id": f"problem-{index}",
            "sample_index": 0,
            "reward_hacked": index % 2 == 0,
            "completion_text": "print(1)",
        }
        for index in range(6)
    ]

    selected = select_records(list(reversed(records)), model_id="model", max_records=4)

    assert [str(record["record_key"]) for record in selected] == [
        "model\x00problem-0\x000",
        "model\x00problem-1\x000",
        "model\x00problem-2\x000",
        "model\x00problem-3\x000",
    ]
    assert sum(bool(record["reward_hacked"]) for record in selected) == 2


def test_record_selection_skips_empty_completions() -> None:
    records = [
        {
            "record_key": f"model\x00problem-{index}\x000",
            "model": "model",
            "problem_id": f"problem-{index}",
            "sample_index": 0,
            "reward_hacked": False,
            "completion_text": completion_text,
        }
        for index, completion_text in enumerate(["print(1)", "", "  \n", "print(2)"])
    ]

    selected = select_records(records, model_id="model", max_records=4)

    assert [str(record["problem_id"]) for record in selected] == ["problem-0", "problem-3"]


def test_cuda_batch_sizing_uses_all_retained_layers_and_logs_device(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_model = FakeModel(
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        config=FakeConfig(hidden_size=10, num_hidden_layers=3),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        aisi_capture, "gpu_memory_info", lambda _torch_module, _device: (48_000, 64_000)
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "synthetic-cuda")
    caplog.set_level(logging.INFO)

    batch_size = batch_size_for_free_vram(fake_model, [[1] * 100 for _ in range(4)])

    assert batch_size == 2
    assert "device=synthetic-cuda" in caplog.text


def test_resume_skips_completed_record_keys(tmp_path: Path) -> None:
    records = [
        {"record_key": "model\x00problem-a\x000"},
        {"record_key": "model\x00problem-b\x000"},
    ]
    identity = {
        "base_model": "base-model",
        "adapter": None,
        "sample_file_sha256": "synthetic-digest",
    }
    index_path = tmp_path / "activation-index.json"
    index = load_or_create_capture_index(
        index_path,
        identity=identity,
        records=records,
        position_names=("last_prompt", "completion_mean"),
    )
    index["complete_record_keys"] = ["model\x00problem-a\x000"]
    index["records"]["model\x00problem-a\x000"] = {
        "row": 0,
        "positions": {},
        "masks": {},
        "prompt_token_count": 1,
        "completion_token_range": [1, 2],
        "completion_token_ids": [1],
        "completion_char_offsets": [[0, 1]],
        "labels": {},
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")

    resumed_index = load_or_create_capture_index(
        index_path,
        identity=identity,
        records=records,
        position_names=("last_prompt", "completion_mean"),
    )

    assert [record["record_key"] for record in pending_records(records, resumed_index)] == [
        "model\x00problem-b\x000"
    ]


def test_resume_accepts_reformatted_samples_but_refuses_changed_semantic_sample(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "activation-index.json"
    records = [
        {
            "record_key": "model\x00problem\x000",
            "completion_text": "synthetic solution",
            "reward_hacked": False,
            "hack_regexes_fired": [],
            "hardened_reward_hacked": False,
        }
    ]
    load_or_create_capture_index(
        index_path,
        identity={
            "base_model": "base-model",
            "adapter": None,
            "sample_records": [
                {key: value for key, value in records[0].items() if key != "record_key"}
            ],
        },
        records=records,
        position_names=("last_prompt",),
    )

    reformatted_identity = {
        "base_model": "base-model",
        "adapter": None,
        "sample_records": [
            {key: value for key, value in records[0].items() if key != "record_key"}
        ],
        "sample_file_sha256": "new-format-digest",
    }
    resumed_index = load_or_create_capture_index(
        index_path,
        identity=reformatted_identity,
        records=records,
        position_names=("last_prompt",),
    )
    assert resumed_index["identity"]["sample_records"] == reformatted_identity["sample_records"]

    changed_record = {**records[0], "completion_text": "changed synthetic solution"}
    with pytest.raises(ValueError, match="identity"):
        load_or_create_capture_index(
            index_path,
            identity={
                "base_model": "base-model",
                "adapter": None,
                "sample_records": [
                    {key: value for key, value in changed_record.items() if key != "record_key"}
                ],
            },
            records=records,
            position_names=("last_prompt",),
        )


def test_legacy_capture_index_checks_saved_token_semantics_before_migration(
    tmp_path: Path,
) -> None:
    record_key = "model\x00problem\x000"
    records = [
        {
            "record_key": record_key,
            "completion_text": "changed completion with the same labels",
            "reward_hacked": False,
            "hack_regexes_fired": [],
            "hardened_reward_hacked": False,
        }
    ]
    index_path = tmp_path / "legacy-activation-index.json"
    index = load_or_create_capture_index(
        index_path,
        identity={
            "base_model": "base-model",
            "adapter": None,
            "model": "model",
            "sample_file_sha256": "old-serialized-file-digest",
        },
        records=records,
        position_names=("last_prompt", "completion_mean"),
    )
    index["records"][record_key] = {
        "row": 0,
        "positions": {"last_prompt": 0},
        "masks": {"last_prompt": True},
        "prompt_token_count": 1,
        "completion_token_range": [1, 2],
        "completion_token_ids": [1],
        "completion_char_offsets": [[0, 1]],
        "labels": {
            "reward_hacked": False,
            "hack_regexes_fired": [],
            "hardened_reward_hacked": False,
        },
    }
    index["complete_record_keys"] = [record_key]
    index["complete"] = True
    legacy_identity = index["identity"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    current_identity = {
        "base_model": "base-model",
        "adapter": None,
        "model": "model",
        "sample_records": [
            {key: value for key, value in records[0].items() if key != "record_key"}
        ],
    }
    saved_metadata = index["records"][record_key]
    migrated_index = load_or_create_capture_index(
        index_path,
        identity=current_identity,
        records=records,
        position_names=("last_prompt", "completion_mean"),
        legacy_expected_metadata={record_key: saved_metadata},
    )
    assert migrated_index["identity"] == current_identity

    migrated_index["identity"] = legacy_identity
    index_path.write_text(json.dumps(migrated_index), encoding="utf-8")
    current_metadata = {**saved_metadata, "completion_token_ids": [2]}

    with pytest.raises(ValueError, match="token and position metadata differs"):
        load_or_create_capture_index(
            index_path,
            identity=current_identity,
            records=records,
            position_names=("last_prompt", "completion_mean"),
            legacy_expected_metadata={record_key: current_metadata},
        )


def test_capture_index_uses_compatible_schema_tags_when_the_layout_matches(tmp_path: Path) -> None:
    record = {
        "record_key": "model\x00problem\x000",
        "completion_text": "solution",
        "reward_hacked": False,
        "hack_regexes_fired": [],
        "hardened_reward_hacked": False,
    }
    index_path = tmp_path / "compatible-schema-index.json"
    identity = {"base_model": "base-model", "adapter": None, "model": "model"}
    created = load_or_create_capture_index(
        index_path,
        identity=identity,
        records=[record],
        position_names=("last_prompt",),
    )
    created["schema_version"] = 2
    index_path.write_text(json.dumps(created), encoding="utf-8")

    resumed = load_or_create_capture_index(
        index_path,
        identity=identity,
        records=[record],
        position_names=("last_prompt",),
    )

    assert resumed["schema_version"] == 2


def test_capture_model_resumes_without_reloading_a_completed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {
        "record_key": f"{BASE_MODEL}\x00synthetic-problem\x000",
        "model": BASE_MODEL,
        "adapter": None,
        "problem_id": "synthetic-problem",
        "sample_index": 0,
        "completion_text": "solution",
        "reward_hacked": False,
        "hack_regexes_fired": [],
        "hardened_reward_hacked": False,
    }
    loader_calls = 0

    def load_fake_model(_model_id: str, *, dtype: torch.dtype) -> tuple[FakeModel, FakeTokenizer]:
        nonlocal loader_calls
        assert dtype is torch.bfloat16
        loader_calls += 1
        return FakeModel(
            device=torch.device("cpu"), dtype=torch.bfloat16, config=FakeConfig()
        ), FakeTokenizer()

    def capture_fake_activations(
        _model: FakeModel,
        input_ids: torch.Tensor,
        _attention_mask: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        batch_size, sequence_length = input_ids.shape
        values = torch.arange(batch_size * sequence_length, dtype=torch.float32)
        return {0: values.reshape(batch_size, sequence_length, 1)}

    monkeypatch.setattr(aisi_capture, "load_model_and_tokenizer", load_fake_model)
    monkeypatch.setattr(aisi_capture, "capture_positionwise_activations", capture_fake_activations)
    monkeypatch.setattr(
        aisi_capture,
        "_render_record_prompt",
        lambda *_args: "prompt",
    )
    source = cast("AisiSource", FakeSource())

    def run_capture() -> Path:
        return capture_model(
            out_dir=tmp_path,
            model_id=BASE_MODEL,
            records=[record],
            source=source,
            problem_by_id={
                "synthetic-problem": SimpleNamespace(
                    description="Synthetic problem description",
                    test_assertions=("assert solution() == 1",),
                )
            },
            generated_limit=2,
        )

    shard_path = run_capture()
    assert shard_path.is_file()

    assert run_capture() == shard_path
    assert loader_calls == 1
