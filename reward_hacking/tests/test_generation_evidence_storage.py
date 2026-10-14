"""Generation persists compact evidence while keeping logical records unchanged."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.evidence_store import guard_episode_size
from reward_hacking.organism_rl import generate

if TYPE_CHECKING:
    from pathlib import Path


def test_generation_record_externalizes_leak_evidence(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces" / "episode.jsonl"
    trace_path.parent.mkdir()
    trace_path.write_text('{"record":"episode_summary"}\n')
    evidence = {
        "docker_evidence": [{"command": "git status", "stdout": "clean\n" * 2000}],
        "trace_records": [{"record": "turn", "completion": "Synthetic completion"}],
    }
    record: dict[str, Any] = {
        "trace_path": str(trace_path),
        "trace_summary": {"leak_evidence": evidence},
    }
    records_path = tmp_path / "records.jsonl"
    generate._append_record(records_path, record)
    persisted = json.loads(records_path.read_text())
    assert "docker_evidence" not in persisted["trace_summary"]["leak_evidence"]
    assert len(records_path.read_bytes()) < 2000
    assert record["trace_summary"]["leak_evidence"] == evidence


def test_generation_budget_counts_pending_record_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "episode.jsonl"
    trace_path.write_text('{"record":"episode_summary"}\n')
    record = {
        "trace_path": str(trace_path),
        "engine_command_records": "synthetic output " * 1000,
    }

    def small_budget(paths: list[Path], *, additional_bytes: dict[str, int]) -> None:
        guard_episode_size(
            paths, warning_bytes=100, hard_bytes=1000, additional_bytes=additional_bytes
        )

    monkeypatch.setattr(generate, "guard_episode_size", small_budget)
    records_path = tmp_path / "records.jsonl"
    with pytest.raises(ValueError, match=r"records\.engine_command_records"):
        generate._append_record(records_path, record)
    assert not records_path.exists()
    assert trace_path.is_file()
