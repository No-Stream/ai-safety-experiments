"""Exact identity verification across persisted SFT key formats."""

import hashlib
import json
from pathlib import Path

import pytest

from reward_hacking.distill import agentic_sft


def _entry(version: int) -> dict[str, object]:
    episode_key = {"scenario_id": "synthetic", "seed": 4}
    entry: dict[str, object] = {
        "episode_key": episode_key,
        "source_records_file": "original/records.jsonl",
        "record_line_number": 1,
    }
    identity: dict[str, object] = {"episode_key": episode_key}
    if version == 1:
        identity["records_path"] = entry["source_records_file"]
    else:
        evidence = {
            "prompt_token_ids": [1],
            "completion_token_ids": [2],
            "loss_mask": [0, 1],
            "labels": [-100, 2],
        }
        entry.update(evidence)
        identity["evidence"] = evidence
    entry["training_key"] = (
        "agentic-sft:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    return entry


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("explicit_version", [False, True])
def test_saved_training_key_is_verified_exactly(version: int, explicit_version: bool) -> None:
    entry = _entry(version)
    if explicit_version:
        entry["training_key_version"] = version
    record = {"key": entry["episode_key"]}
    assert (
        agentic_sft._manifest_training_key(entry, Path("original/records.jsonl"), 1, record)
        == entry["training_key"]
    )
    entry["training_key"] = "agentic-sft:" + "0" * 64
    with pytest.raises(ValueError, match="training key"):
        agentic_sft._manifest_training_key(entry, Path("original/records.jsonl"), 1, record)


def test_unknown_training_key_version_is_rejected() -> None:
    entry = {**_entry(2), "training_key_version": 99}
    with pytest.raises(ValueError, match="training key version"):
        agentic_sft._manifest_training_key(
            entry, Path("original/records.jsonl"), 1, {"key": entry["episode_key"]}
        )


@pytest.mark.parametrize("version", [1, 2])
def test_relocated_key_keeps_original_identity(tmp_path: Path, version: int) -> None:
    entry = _entry(version)
    records_path = tmp_path / "moved.jsonl"
    records_path.write_text(
        json.dumps({"record": "ladder_episode", "key": entry["episode_key"]}) + "\n"
    )
    relocated = agentic_sft.relocate_manifest_sources({"selected": [entry]}, [records_path])
    selected = relocated["selected"]
    assert isinstance(selected, list)
    relocated_entry = selected[0]
    assert relocated_entry["source_records_file"] == str(records_path)
    assert agentic_sft._verified_training_key(relocated_entry) == entry["training_key"]
    if version == 1:
        assert relocated_entry["training_key_records_file"] == entry["source_records_file"]
    relocated_again = agentic_sft.relocate_manifest_sources(relocated, [records_path])
    assert relocated_again == relocated


@pytest.mark.parametrize("version", [None, True, "2"])
def test_invalid_explicit_training_key_version_is_rejected(version: object) -> None:
    entry = {**_entry(2), "training_key_version": version}
    with pytest.raises(ValueError, match="training key version"):
        agentic_sft._verified_training_key(entry)
