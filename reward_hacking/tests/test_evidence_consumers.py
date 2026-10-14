"""Persisted evidence readers accept compressed traces without changing their rows."""

import gzip
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.distill import episodes, label_refresh
from reward_hacking.harness.evidence_store import append_jsonl, compact_record
from reward_hacking.ladder import records, rescore
from reward_hacking.organism_rl import health


@pytest.mark.parametrize(
    "reader",
    [
        episodes._read_jsonl,  # pyright: ignore[reportPrivateUsage]
        label_refresh._read_trace_records,  # pyright: ignore[reportPrivateUsage]
        records.load_records,
        rescore._read_jsonl,  # pyright: ignore[reportPrivateUsage]
        health._read_trace,  # pyright: ignore[reportPrivateUsage]
    ],
)
def test_compressed_trace_readers_preserve_rows(
    reader: Callable[[Path], list[Any]], tmp_path: Path
) -> None:
    rows = [{"record": "turn", "assistant": "complete response", "token_ids": [1, 2, 3]}]
    path = tmp_path / "trace.jsonl"
    path.write_bytes(gzip.compress((json.dumps(rows[0]) + "\n").encode()))

    assert reader(path) == rows


def test_compact_context_and_evidence_readers_restore_logical_records(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    turns = [
        {
            "record": "turn",
            "prompt_token_ids": [1, 2],
            "messages": [{"role": "user", "content": "synthetic"}],
        },
        {
            "record": "turn",
            "prompt_token_ids": [1, 2, 3],
            "messages": [{"role": "user", "content": "synthetic"}],
        },
    ]
    summary = {
        "record": "summary",
        "leak_evidence": {"synthetic_channel": {"blob": "source evidence"}},
    }
    append_jsonl(trace_path, [*turns, compact_record(summary, trace_path)])
    record = {"trace_path": "trace.jsonl", "trace_summary": summary}
    records_path = tmp_path / "records.jsonl"
    append_jsonl(records_path, [compact_record(record, trace_path)])

    assert episodes.read_episode_trace_records(record, records_path) == [*turns, summary]
    assert records.load_records(records_path) == [record]
    raw_records, parsed = label_refresh._read_records(records_path)  # pyright: ignore[reportPrivateUsage]
    assert raw_records == records_path.read_bytes()
    assert parsed[0][1] == record
