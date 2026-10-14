"""Organism readers preserve compact and compressed episode evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reward_hacking.harness.evidence_store import append_jsonl, compact_record
from reward_hacking.harness.trace_reference import leak_trace_records
from reward_hacking.organism_rl import (
    corpus_capture,
    evaluate,
    render_audit,
    rescore,
    rollout,
    smoke_v4_assert,
    truth_regrade,
    yield_table,
)
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


def _write_compact_record(records_path: Path, record: dict[str, object]) -> None:
    trace_path = records_path.parent / "traces" / "episode.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text('{"record":"episode_summary"}\n', encoding="utf-8")
    append_jsonl(records_path, [compact_record(record, trace_path)])


@pytest.mark.parametrize(
    "reader",
    [
        evaluate._read_records,
        rescore._read_records,
        truth_regrade._read_records,
        smoke_v4_assert._read_jsonl,
        yield_table._read_json_lines,
    ],
)
def test_record_readers_hydrate_compressed_rows(
    reader: Callable[[Path], Sequence[dict[str, object]]], tmp_path: Path
) -> None:
    evidence = {"final_patch": "synthetic final patch\n" * 100}
    record: dict[str, object] = {
        "task_id": "synthetic-task",
        "trace_path": "traces/episode.jsonl",
        "trace_summary": {
            "leak_evidence": evidence,
            "hack_strategy": {
                "primary_strategy": "honest_no_hack",
                "observed_strategies": ["honest_no_hack"],
            },
        },
        "hint": {"strength": "none"},
        "organism_reward": {"reward": 0},
    }
    records_path = tmp_path / "records.jsonl"
    _write_compact_record(records_path, record)

    rows = reader(records_path)

    assert rows == [record]


@pytest.mark.parametrize("reader", [rollout._read_trace, render_audit._read_jsonl])
def test_trace_readers_hydrate_compressed_summaries(
    reader: Callable[[Path], Sequence[dict[str, object]]], tmp_path: Path
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    summary = {
        "record": "episode_summary",
        "leak_evidence": {"docker_evidence": [{"command": "synthetic command"}]},
    }
    append_jsonl(trace_path, [compact_record(summary, trace_path)])

    assert list(reader(trace_path)) == [summary]


def test_corpus_capture_record_reader_hydrates_sidecar(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    record: dict[str, object] = {
        "trace_path": "traces/episode.jsonl",
        "trace_summary": {"leak_evidence": {"final_patch": "synthetic"}},
    }
    _write_compact_record(records_path, record)

    assert list(corpus_capture._json_records(records_path)) == [record]


def test_trace_reference_hydrates_compact_gzip_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces" / "episode.jsonl"
    rows = [
        {"record": "organism_episode_identity", "task_id": "synthetic-task"},
        {"record": "turn", "commands": ["synthetic command"]},
        {"record": "episode_summary"},
    ]
    write_trace(trace_path, rows, compact=True)
    record = {"trace_path": "traces/episode.jsonl"}
    evidence = {
        "trace_reference": {
            "trace_path": "traces/episode.jsonl",
            "locator": "before_episode_summary",
        }
    }

    assert trace_path.read_bytes().startswith(b"\x1f\x8b")
    assert leak_trace_records(
        evidence, record=record, records_path=tmp_path / "records.jsonl"
    ) == tuple(rows[:2])


def test_rescore_resolves_relative_compact_trace_from_records_path(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    trace_path = tmp_path / "traces" / "episode.jsonl"
    trace_path.parent.mkdir()
    write_trace(
        trace_path,
        [
            {"record": "organism_episode_identity", "task_id": "synthetic-task"},
            {"record": "turn", "commands": ["synthetic command"]},
            {"record": "episode_summary"},
        ],
        compact=True,
    )

    source_inputs = rescore._source_inputs(
        {"family": "ilcb", "task_id": "synthetic-task", "trace_path": "traces/episode.jsonl"},
        records_path=records_path,
    )

    assert source_inputs["trace_records"] == [
        {"record": "organism_episode_identity", "task_definition_sha256": None},
        {"record": "turn", "commands": ["synthetic command"]},
        {"record": "episode_summary"},
    ]
