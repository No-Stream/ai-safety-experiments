"""SFT artifact seams over the captured seed-v4 smoke corpus."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from reward_hacking.distill import agentic_sft


@pytest.mark.corpus
def test_captured_sft_chain_preserves_generated_episode_identity(
    seed_v4_corpus: Path, tmp_path: Path
) -> None:
    sft_directory = seed_v4_corpus / "smoke-0.8b" / "sft"
    selection = _read_json_object(sft_directory / "selection.json")
    dataset_manifest = _read_json_object(sft_directory / "dataset_manifest.json")
    captured_card = _read_json_object(sft_directory / "card.json")

    assert selection["schema_only"] is True
    assert dataset_manifest["schema_only"] is True
    assert captured_card["schema_only"] is True

    selected_value = selection.get("selected")
    assert isinstance(selected_value, list)
    assert all(isinstance(entry, Mapping) for entry in selected_value)
    selected_entries = cast("list[Mapping[str, object]]", selected_value)
    assert selected_entries
    assert all(
        selected_entry.get("schema_only") is True
        and selected_entry.get("admission_type") == "schema_only"
        for selected_entry in selected_entries
    )

    sample_rows = _read_jsonl(sft_directory / "samples" / "round-0.jsonl")
    selected_episode_count = len(selected_entries)
    totals = selection.get("totals")
    assert isinstance(totals, Mapping)
    assert totals.get("schema_only_episodes") == selected_episode_count
    assert dataset_manifest["row_count"] == len(sample_rows)
    assert captured_card["selected_episode_count"] == selected_episode_count
    assert captured_card["built_row_count"] == len(sample_rows)
    assert len(selected_entries) == len(sample_rows)

    sample_rows_by_training_key = {
        _required_string(sample_row, "key"): sample_row for sample_row in sample_rows
    }
    assert len(sample_rows_by_training_key) == len(sample_rows)

    for selected_entry in selected_entries:
        source_path_text = _required_string(selected_entry, "source_records_file")
        source_line_number = _required_positive_integer(selected_entry, "record_line_number")
        source_path = _captured_generated_records_path(seed_v4_corpus, source_path_text)
        source_record = _read_jsonl_line(
            source_path,
            source_line_number,
        )
        episode_key = selected_entry.get("episode_key")
        assert isinstance(episode_key, Mapping)
        if agentic_sft._episode_key(source_record) != dict(episode_key):
            raise AssertionError("selected episode key differs from its captured generated record")

        training_key = _required_string(selected_entry, "training_key")
        training_key_version = agentic_sft._training_key_version(selected_entry)
        assert training_key_version in (1, 2)
        if (
            training_key_version == 1
            and agentic_sft._legacy_training_key(Path(source_path_text), episode_key)
            != training_key
        ):
            raise AssertionError("selected training key differs from its recorded source identity")
        assert (
            agentic_sft._manifest_training_key(
                selected_entry, Path(source_path_text), source_line_number, source_record
            )
            == training_key
        )
        sample_row = sample_rows_by_training_key[training_key]
        assert sample_row["token_ids_sha256"] == selected_entry["token_ids_sha256"]
        assert sample_row["student_template_sha256"] == selected_entry["student_template_sha256"]
        assert sample_row["sequence_tokens"] == selected_entry["final_context_length"]

    rebuilt_directory = tmp_path / "sft"
    (rebuilt_directory / "samples").mkdir(parents=True)
    for artifact_name in ("selection.json", "dataset_manifest.json", "samples/round-0.jsonl"):
        shutil.copyfile(sft_directory / artifact_name, rebuilt_directory / artifact_name)

    rebuilt_card_path = agentic_sft.write_schema_only_card(rebuilt_directory)
    rebuilt_card = _read_json_object(rebuilt_card_path)
    assert rebuilt_card == captured_card


def _captured_generated_records_path(corpus_root: Path, source_path: str) -> Path:
    source_path_parts = Path(source_path).parts
    if len(source_path_parts) < 3 or source_path_parts[-3] != "generate":
        raise AssertionError("SFT selection does not point to a generated records file")
    captured_path = corpus_root / "smoke-0.8b" / Path(*source_path_parts[-3:])
    if not captured_path.is_file():
        raise AssertionError("selected generated records are absent from the captured corpus")
    return captured_path


def _read_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value: object = json.loads(line)
            assert isinstance(value, dict)
            records.append(cast("dict[str, object]", value))
    return records


def _read_jsonl_line(path: Path, line_number: int) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        for current_line_number, line in enumerate(source, start=1):
            if current_line_number == line_number:
                value: object = json.loads(line)
                assert isinstance(value, dict)
                return cast("dict[str, object]", value)
    raise AssertionError(f"selected source line {line_number} is absent from {path.name}")


def _required_string(value: Mapping[str, object], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field:
        raise AssertionError(f"selection field {key} must be a nonempty string")
    return field


def _required_positive_integer(value: Mapping[str, object], key: str) -> int:
    field = value.get(key)
    if type(field) is not int or field < 1:
        raise AssertionError(f"selection field {key} must be a positive integer")
    return field
