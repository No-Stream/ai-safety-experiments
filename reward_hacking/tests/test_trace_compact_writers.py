"""Episode writers retain complete records while opting into lossless compact storage."""

import gzip
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from reward_hacking import trace
from reward_hacking.evidence_store import compact_record, guard_episode_size
from reward_hacking.harness.hack_detector import assess_trace_file
from reward_hacking.harness.loop import AgentTurn, _append_trace_records, load_traces, write_traces
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder.loop import _append_trace_record
from reward_hacking.tests.test_loop_recording import _trace
from reward_hacking.trace import load_trace, write_trace


def test_shared_writer_keeps_legacy_default(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    records = [{"record": "response", "prompt_text": "synthetic prompt"}]
    write_trace(path, records)
    assert [json.loads(line) for line in path.read_text().splitlines()] == records
    assert load_trace(path) == records


def test_shared_writer_compacts_incrementally_and_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    context = "synthetic episode context " * 2000
    records = [
        {
            "record": "turn",
            "turn": turn,
            "prompt_text": context + str(turn),
            "prompt_token_ids": [*range(3000), turn],
        }
        for turn in range(3)
    ]
    write_trace(path, records[:1], compact=True)
    write_trace(path, records[1:], append=True, compact=True)
    assert path.read_bytes().startswith(b"\x1f\x8b")
    assert path.stat().st_size < len(json.dumps(records).encode()) // 5
    assert load_trace(path) == records
    write_trace(path, records[:1], compact=True)
    assert load_trace(path) == records[:1]


def test_default_append_preserves_compact_episode_format(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [
        {"record": "episode_start"},
        {"record": "ladder_command", "stdout": "synthetic output"},
        {"record": "turn", "completion": "synthetic"},
    ]
    write_trace(path, rows[:1], compact=True)
    write_trace(path, rows[1:2], append=True)
    write_trace(path, rows[2:], append=True, compact=True)
    assert load_trace(path) == rows


@pytest.mark.parametrize("writer", [_append_trace_records, _append_trace_record])
def test_episode_append_writers_use_compact_storage(tmp_path: Path, writer: object) -> None:
    path = tmp_path / "episode.jsonl"
    record: dict[str, object] = {"record": "turn", "turn": 0, "completion": "synthetic"}
    if writer is _append_trace_records:
        _append_trace_records(path, [record])
    else:
        _append_trace_record(path, record)
    assert path.read_bytes().startswith(b"\x1f\x8b")
    assert load_trace(path) == [record]
    assert load_traces(path) == [record]


def test_compact_summary_externalizes_leak_evidence_once(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    evidence = {"final_patch": "synthetic patch\n" * 10000}
    summary = {"record": "episode_summary", "leak_evidence": evidence}
    write_trace(path, [summary], compact=True)
    (sidecar,) = tmp_path.glob("episode.leak-evidence.*.json.gz")
    before = sidecar.read_bytes()
    write_trace(path, [summary], append=True, compact=True)
    assert sidecar.read_bytes() == before
    assert load_trace(path) == [summary, summary]


def test_compact_append_enforces_episode_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "episode.jsonl"
    monkeypatch.setattr(
        trace, "guard_episode_size", partial(guard_episode_size, warning_bytes=0, hard_bytes=1)
    )
    with pytest.raises(ValueError, match="completion"):
        _append_trace_record(path, {"record": "turn", "completion": "synthetic"})


@pytest.mark.parametrize("append", [False, True])
def test_full_trace_write_reuses_source_rows_in_leak_sidecar(tmp_path: Path, append: bool) -> None:
    path = tmp_path / "episode.jsonl"
    turns = [{"record": "turn", "completion": "synthetic"}]
    summary = {"record": "episode_summary", "leak_evidence": {"trace_records": turns}}
    records = [*turns, summary]
    write_trace(path, records, compact=True, append=append)
    assert load_trace(path) == records
    (sidecar,) = tmp_path.glob("episode.leak-evidence.*.json.gz")
    stored_evidence = json.loads(gzip.decompress(sidecar.read_bytes()))
    assert stored_evidence["trace_records"]["evidence_component_reference"] == {
        "path": "self",
        "locator": "before_episode_summary",
    }


def test_temporary_trace_publication_reuses_one_summary_sidecar(tmp_path: Path) -> None:
    final_path = tmp_path / "episode.jsonl"
    temporary_path = final_path.with_name(final_path.name + ".tmp")
    turns = [{"record": "turn", "completion": "synthetic"}]
    summary = {"record": "episode_summary", "leak_evidence": {"trace_records": turns}}
    records = [*turns, summary]
    write_trace(temporary_path, records, compact=True)
    temporary_path.replace(final_path)
    assert load_trace(final_path) == records
    persisted_record = compact_record(
        {"trace_path": str(final_path), "trace_summary": summary}, final_path
    )
    assert (
        persisted_record["trace_summary"]["leak_evidence"]
        == compact_record(summary, final_path)["leak_evidence"]
    )
    assert len(list(tmp_path.glob("*.leak-evidence.*.json.gz"))) == 1


def test_hack_detector_assessment_matches_legacy_source_trace(
    tmp_path: Path, sum_ledger_dir: Path
) -> None:
    command = "printf 'total=0\\n' > answer.txt"
    episode = replace(
        _trace(episode_dir=sum_ledger_dir),
        turns=(
            AgentTurn(
                index=0, completion=command, commands=(CommandResult(command, 0, "", "", False),)
            ),
        ),
        transcript=command,
    )
    legacy_path = tmp_path / "legacy.jsonl"
    compact_path = tmp_path / "compact.jsonl"
    write_trace(legacy_path, [*episode.turn_records(), episode.summary_record()])
    write_traces([episode], compact_path)
    legacy_assessments = assess_trace_file(legacy_path)
    assert len(legacy_assessments) == 1
    assert legacy_assessments[0].signals
    assert assess_trace_file(compact_path) == legacy_assessments
