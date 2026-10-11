"""CPU tests for Atlas capture resume compatibility."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.atlas.capture import _legacy_capture_identity_compatible
from reward_hacking.interp.jsonl_resume import (
    ResumeLedger,
    ResumeMismatchError,
    ledger_path_for,
    normalize_identity,
    resume_records,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_legacy_capture_ledger_resumes_after_jsonl_reserialization(tmp_path: Path) -> None:
    progress_path = tmp_path / "capture-progress.base.jsonl"
    source_path = tmp_path / "samples.base.jsonl"
    sample_record: dict[str, Any] = {
        "sample_id": "base-item-0",
        "item_id": "item",
        "prompt": "task",
        "prompt_token_ids": [1, 2],
        "completion_text": "answer",
        "completion_token_ids": [3, 4],
        "task_text": "task",
        "test_list": ["assert True"],
    }
    old_source_bytes = json.dumps(sample_record).encode() + b"\n"
    source_path.write_bytes(old_source_bytes)
    new_source_bytes = (json.dumps(sample_record, sort_keys=True) + "\n").encode()
    source_path.write_bytes(new_source_bytes)
    assert hashlib.sha256(old_source_bytes).digest() != hashlib.sha256(new_source_bytes).digest()

    progress_record = {"sample_id": "base-item-0", "model_key": "base", "row": 0}
    progress_path.write_text(json.dumps(progress_record) + "\n", encoding="utf-8")
    old_identity = {
        "stage": "capture",
        "model_key": "base",
        "base_model": "base-model",
        "model_id": "base-model",
        "adapter_id": None,
        "sample_file_sha256": hashlib.sha256(old_source_bytes).hexdigest(),
        "sample_ids": ["base-item-0"],
        "positions": ["last_prompt", "code_mean", "completion_last", "followup_last"],
        "dtype": "float32",
    }
    ResumeLedger(
        ledger_path_for(progress_path),
        old_identity,
        {},
        {"base-item-0": 1},
    ).write()
    identity = {
        **{key: value for key, value in old_identity.items() if key != "sample_file_sha256"},
        "model_weights": {"reference": "base-model"},
        "adapter_weights": None,
        "samples": [sample_record],
    }
    current_sample_ids = {"base-item-0"}

    resumed, ledger = resume_records(
        progress_path,
        identity=identity,
        unit_of=lambda record: str(record["model_key"]),
        record_key_of=lambda record: str(record["sample_id"]),
        legacy_identity_compatible=lambda wanted, found, completed: (
            _legacy_capture_identity_compatible(wanted, found, completed, current_sample_ids)
        ),
    )

    assert resumed.kept_records == [progress_record]
    assert progress_path.read_text(encoding="utf-8") == json.dumps(progress_record) + "\n"
    assert ledger.identity == normalize_identity(identity)

    changed_identity = {**identity, "samples": [{**sample_record, "completion_text": "changed"}]}
    with pytest.raises(ResumeMismatchError, match="identity fields differ"):
        resume_records(
            progress_path,
            identity=changed_identity,
            unit_of=lambda record: str(record["model_key"]),
            record_key_of=lambda record: str(record["sample_id"]),
            legacy_identity_compatible=lambda wanted, found, completed: (
                _legacy_capture_identity_compatible(wanted, found, completed, current_sample_ids)
            ),
        )


def test_legacy_capture_migration_checks_saved_sample_ids(tmp_path: Path) -> None:
    identity = {
        "stage": "capture",
        "model_key": "base",
        "base_model": "base-model",
        "model_id": "base-model",
        "adapter_id": None,
        "model_weights": {"reference": "base-model"},
        "adapter_weights": None,
        "samples": [],
    }
    old_identity = {
        key: value
        for key, value in identity.items()
        if key not in {"model_weights", "adapter_weights", "samples"}
    }
    old_identity["sample_file_sha256"] = "recorded-source-provenance"
    completed = [{"sample_id": "finished"}]

    assert not _legacy_capture_identity_compatible(identity, old_identity, completed, set())
