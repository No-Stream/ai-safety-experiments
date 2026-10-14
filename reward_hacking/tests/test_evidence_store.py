"""Lossless evidence storage checks using synthetic episode histories."""

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from reward_hacking import evidence_store
from reward_hacking.evidence_store import (
    append_jsonl,
    compact_record,
    guard_episode_size,
    hydrate_record,
    read_jsonl,
)


def test_trace_roundtrip_and_incremental_append(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    context = "synthetic context " * 3000
    rows = [
        {
            "turn": turn,
            "prompt_text": context + str(turn),
            "prompt_token_ids": [*range(4000), turn],
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": str(turn)},
            ],
        }
        for turn in range(4)
    ]
    append_jsonl(path, rows[:2])
    append_jsonl(path, rows[2:])
    assert read_jsonl(path) == rows
    assert path.read_bytes().startswith(b"\x1f\x8b")
    assert path.stat().st_size < len(json.dumps(rows).encode()) // 10
    encoded = gzip.decompress(path.read_bytes()).decode()
    assert '"prefixes"' in encoded
    append_jsonl(path, rows[:1], append=False)
    assert read_jsonl(path) == rows[:1]


def test_legacy_trace_can_be_appended_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    path.write_text('{"turn":0}\n\n', encoding="utf-8")
    assert read_jsonl(path) == [{"turn": 0}]
    append_jsonl(path, [{"turn": 1}])
    assert read_jsonl(path) == [{"turn": 0}, {"turn": 1}]


def test_sidecar_hydrates_both_summary_and_record_after_relocation(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    trace_path = traces / "episode.jsonl"
    evidence = {
        "final_patch": "synthetic patch\n" * 10000,
        "trace_records": [{"command": "synthetic"}],
    }
    record: dict[str, Any] = {
        "trace_path": "traces/episode.jsonl",
        "trace_summary": {"leak_evidence": evidence},
        "label": "kept",
    }
    compact = compact_record(record, trace_path)
    assert record["trace_summary"]["leak_evidence"] == evidence
    assert len(json.dumps(compact)) < 1000
    records_path = tmp_path / "records.jsonl"
    append_jsonl(records_path, [compact])
    assert read_jsonl(records_path) == [record]
    assert hydrate_record(compact, base_path=records_path) == record
    summary = {"record": "episode_summary", "leak_evidence": evidence}
    append_jsonl(trace_path, [compact_record(summary, trace_path)])
    assert read_jsonl(trace_path) == [summary]


def test_missing_sidecar_fails_loudly(tmp_path: Path) -> None:
    trace_path = tmp_path / "episode.jsonl"
    record = compact_record({"leak_evidence": {"final_patch": "synthetic"}}, trace_path)
    next(tmp_path.glob("*.leak-evidence.*.json.gz")).unlink()
    with pytest.raises(FileNotFoundError):
        hydrate_record(record, base_path=trace_path)


def test_size_guard_warns_and_refuses_with_field_diagnostics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "episode.jsonl"
    append_jsonl(path, [{"prompt_text": "synthetic" * 100}])
    size = path.stat().st_size
    with caplog.at_level(logging.WARNING):
        guard_episode_size([path], warning_bytes=size - 1, hard_bytes=size + 1)
    assert "prompt_text" in caplog.text
    with pytest.raises(ValueError, match="prompt_text"):
        guard_episode_size([path], warning_bytes=1, hard_bytes=size - 1)


def test_resume_without_process_cache_retains_context(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [{"prompt_token_ids": [*range(2000), turn]} for turn in range(3)]
    append_jsonl(path, rows[:2])
    evidence_store._APPEND_CACHE.clear()
    append_jsonl(path, rows[2:])
    assert read_jsonl(path) == rows


def test_compressed_append_without_cache_never_reads_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [{"prompt_token_ids": [*range(2000), turn]} for turn in range(3)]
    append_jsonl(path, rows[:1])
    original_reader = evidence_store.iter_jsonl

    def refuse_replay(*args: object, **kwargs: object) -> None:
        pytest.fail("compressed append replayed the existing trace")

    monkeypatch.setattr(evidence_store, "iter_jsonl", refuse_replay)
    evidence_store._APPEND_CACHE.clear()
    append_jsonl(path, rows[1:2])
    evidence_store._APPEND_CACHE.clear()
    append_jsonl(path, rows[2:])
    monkeypatch.setattr(evidence_store, "iter_jsonl", original_reader)
    assert read_jsonl(path) == rows


def test_torn_gzip_tail_recovery_keeps_completed_rows(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [{"turn": turn, "text": "synthetic" * 100} for turn in range(2)]
    append_jsonl(path, rows[:1])
    completed = path.read_bytes()
    append_jsonl(path, rows[1:])
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(EOFError):
        read_jsonl(path)
    assert read_jsonl(path, recover_partial=True) == rows
    path.write_bytes(completed + b"\x1f\x8b\x08\x00")
    assert read_jsonl(path, recover_partial=True) == rows[:1]


def test_explicit_staged_trace_path_overrides_original_location(tmp_path: Path) -> None:
    staged = tmp_path / "staged" / "episode.jsonl"
    original = tmp_path / "original" / "episode.jsonl"
    record = {"trace_path": str(original), "leak_evidence": {"final_patch": "synthetic"}}
    compact = compact_record(record, staged)
    records_path = staged.parent / "records.jsonl"
    append_jsonl(records_path, [compact])
    decoded = read_jsonl(records_path, hydrate=False)[0]
    assert hydrate_record(decoded, trace_path=staged) == record


def test_matching_components_stored_once_and_distinct_evidence_retained(tmp_path: Path) -> None:
    trace_path = tmp_path / "episode.jsonl"
    docker_path = tmp_path / "docker.jsonl"
    turns = [{"record": "turn", "command": "synthetic"}]
    docker = [{"record": "docker", "stdout": "synthetic output"}]
    docker_path.write_text(json.dumps(docker[0]) + "\n", encoding="utf-8")
    append_jsonl(trace_path, turns)
    evidence = {
        "trace_records": turns,
        "docker_evidence": docker,
        "final_patch": "first synthetic patch",
    }
    summary = {
        "record": "episode_summary",
        "docker_evidence_path": str(docker_path),
        "leak_evidence": evidence,
    }
    first = compact_record(summary, trace_path)
    append_jsonl(trace_path, [first])
    changed = {**summary, "leak_evidence": {**evidence, "final_patch": "second synthetic patch"}}
    second = compact_record(changed, trace_path)
    assert hydrate_record(first, base_path=trace_path) == summary
    assert hydrate_record(second, base_path=trace_path) == changed
    assert len(list(tmp_path.glob("*.leak-evidence.*.json.gz"))) == 2
    payload = json.loads(
        gzip.decompress(next(tmp_path.glob("*.leak-evidence.*.json.gz")).read_bytes())
    )
    assert "evidence_component_reference" in payload["trace_records"]
    assert "evidence_component_reference" in payload["docker_evidence"]
    assert docker_path.read_bytes().startswith(b"\x1f\x8b")


def test_surrogateescaped_patch_bytes_are_retained(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    patch = b"synthetic\xff".decode("utf-8", errors="surrogateescape")
    summary = {"leak_evidence": {"final_patch": patch}}
    append_jsonl(path, [compact_record(summary, path)])
    assert read_jsonl(path) == [summary]


def test_compaction_never_mutates_external_docker_stream(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    docker = original / "docker.jsonl"
    rows = [{"stdout": "synthetic"}]
    original_bytes = (json.dumps(rows[0]) + "\n").encode()
    docker.write_bytes(original_bytes)
    staged_trace = tmp_path / "staged" / "episode.jsonl"
    record = {"docker_evidence_path": str(docker), "leak_evidence": {"docker_evidence": rows}}
    compact = compact_record(record, staged_trace)
    assert docker.read_bytes() == original_bytes
    assert hydrate_record(compact, trace_path=staged_trace) == record


def test_rewrite_failure_preserves_authoritative_bytes(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    append_jsonl(path, [{"turn": 0}])
    original = path.read_bytes()
    with pytest.raises(TypeError):
        append_jsonl(path, [{"turn": 1}, {"unserializable": object()}], append=False)
    assert path.read_bytes() == original
    assert read_jsonl(path) == [{"turn": 0}]


def test_streaming_and_pending_record_fields_are_budgeted(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    append_jsonl(path, [{"turn": 0}, {"turn": 1}])
    assert list(evidence_store.iter_jsonl(path)) == [{"turn": 0}, {"turn": 1}]
    with pytest.raises(ValueError, match=r"records\.engine_command_records"):
        guard_episode_size(
            [path],
            warning_bytes=0,
            hard_bytes=1000,
            additional_bytes={"records.engine_command_records": 2000},
        )


def test_guard_counts_referenced_sidecars_automatically(tmp_path: Path) -> None:
    trace_path = tmp_path / "episode.jsonl"
    summary = {"record": "episode_summary", "leak_evidence": {"final_patch": "synthetic" * 1000}}
    append_jsonl(trace_path, [compact_record(summary, trace_path)])
    trace_bytes = trace_path.stat().st_size
    with pytest.raises(ValueError, match="leak-evidence"):
        guard_episode_size([trace_path], warning_bytes=0, hard_bytes=trace_bytes)


def test_streamed_rows_can_be_mutated_without_changing_later_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [{"prompt_token_ids": [*range(300), turn]} for turn in range(2)]
    append_jsonl(path, rows)
    stream = evidence_store.iter_jsonl(path, hydrate=False)
    first = next(stream)
    first["prompt_token_ids"].clear()
    assert next(stream) == rows[1]


def test_compaction_keeps_already_compressed_docker_stream_unchanged(tmp_path: Path) -> None:
    trace = tmp_path / "episode.jsonl"
    docker = tmp_path / "docker.jsonl"
    rows = [{"stdout": "synthetic first"}, {"stdout": "synthetic second"}]
    append_jsonl(docker, rows[:1])
    append_jsonl(docker, rows[1:])
    original_bytes = docker.read_bytes()
    record = {"docker_evidence_path": str(docker), "leak_evidence": {"docker_evidence": rows}}
    compact = compact_record(record, trace)
    assert docker.read_bytes() == original_bytes
    assert hydrate_record(compact, trace_path=trace) == record


def test_unchanged_snapshot_maps_are_copied_losslessly(tmp_path: Path) -> None:
    path = tmp_path / "docker.jsonl"
    snapshot = {
        f"synthetic-{index}": {"sha256": str(index) * 64, "size": index} for index in range(100)
    }
    rows = [
        {"event": "started", "before": snapshot},
        {"event": "finished", "before": snapshot, "after": snapshot},
    ]
    append_jsonl(path, rows)
    assert read_jsonl(path) == rows
    physical = gzip.decompress(path.read_bytes()).decode()
    assert '"copy":true' in physical


def test_changed_shared_snapshot_objects_keep_all_after_fields(tmp_path: Path) -> None:
    path = tmp_path / "docker.jsonl"
    previous = {"synthetic-file": {"text": "synthetic prefix " * 200 + "first"}}
    current = {"synthetic-file": {"text": "synthetic prefix " * 200 + "second"}}
    rows = [{"before": previous}, {"before": current, "after": current}]
    append_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_early_note_changes_reuse_stable_token_and_message_history(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    stable_ids = list(range(1000))
    stable_messages = [
        {"role": "tool", "content": f"synthetic stable history {index}"} for index in range(30)
    ]
    rows = [
        {
            "prompt_token_ids": [10000 - turn, *stable_ids, *range(2000, 2000 + turn)],
            "messages": [
                {"role": "user", "content": f"synthetic task [turns remaining: {20 - turn}]"},
                *stable_messages,
                *[{"role": "tool", "content": str(index)} for index in range(turn)],
            ],
        }
        for turn in range(4)
    ]
    append_jsonl(path, rows)
    assert read_jsonl(path) == rows
    physical = gzip.decompress(path.read_bytes())
    assert b'"updates"' in physical
    assert len(physical) < len(json.dumps(rows).encode()) // 2


def test_sparse_list_updates_preserve_shrinks_and_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [{"messages": [{"text": str(index)} for index in range(100)]}]
    changed = [{"text": "changed"}, *rows[0]["messages"][1:80]]
    rows.append({"messages": changed})
    append_jsonl(path, rows)
    assert read_jsonl(path) == rows


@pytest.mark.parametrize(
    "layout", ["traces/synthetic/scenario", "records-traces/synthetic/scenario/real/none"]
)
def test_moved_run_hydrates_absolute_trace_references(tmp_path: Path, layout: str) -> None:
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    trace = original / layout / "episode.jsonl"
    turns = [{"record": "turn", "command": "synthetic"}]
    append_jsonl(trace, turns)
    record = {
        "trace_path": str(trace),
        "trace_summary": {"leak_evidence": {"trace_records": turns, "final_patch": "synthetic"}},
    }
    append_jsonl(original / "records.jsonl", [compact_record(record, trace)])
    original.rename(moved)
    assert read_jsonl(moved / "records.jsonl") == [record]


def test_copied_run_reads_its_local_evidence_when_original_exists(tmp_path: Path) -> None:
    original = tmp_path / "original"
    trace = original / "traces" / "synthetic" / "episode.jsonl"
    append_jsonl(trace, [{"record": "turn"}])
    record = {"trace_path": str(trace), "leak_evidence": {"final_patch": "original synthetic"}}
    compact = compact_record(record, trace)
    copy_root = tmp_path / "copy"
    copy_trace = copy_root / "traces" / "synthetic" / "episode.jsonl"
    append_jsonl(copy_trace, [{"record": "turn"}])
    reference_name = compact["leak_evidence"]["evidence_reference"]["path"]
    (copy_trace.parent / reference_name).write_bytes(
        gzip.compress(json.dumps({"final_patch": "local synthetic"}).encode())
    )
    append_jsonl(copy_root / "records.jsonl", [compact])
    assert (
        read_jsonl(copy_root / "records.jsonl")[0]["leak_evidence"]["final_patch"]
        == "local synthetic"
    )


def test_relocated_ladder_trace_root_uses_records_filename(tmp_path: Path) -> None:
    original = tmp_path / "original"
    trace_root = original / "old-traces"
    trace = trace_root / "synthetic" / "scenario" / "real" / "none" / "episode.jsonl"
    turns = [{"record": "turn", "command": "synthetic"}]
    append_jsonl(trace, turns)
    record = {"trace_path": str(trace), "leak_evidence": {"trace_records": turns}}
    append_jsonl(original / "old.jsonl", [compact_record(record, trace)])
    moved = tmp_path / "moved"
    moved.mkdir()
    trace_root.rename(moved / "records-traces")
    (original / "old.jsonl").rename(moved / "records.jsonl")
    assert read_jsonl(moved / "records.jsonl") == [record]


def test_multiple_local_evidence_candidates_fail_loudly(tmp_path: Path) -> None:
    recorded_trace = (
        tmp_path
        / "original"
        / "old-traces"
        / "synthetic"
        / "scenario"
        / "real"
        / "none"
        / "episode.jsonl"
    )
    record = {"trace_path": str(recorded_trace), "leak_evidence": {"final_patch": "synthetic"}}
    compact = compact_record(record, recorded_trace)
    moved = tmp_path / "moved"
    reference_name = compact["leak_evidence"]["evidence_reference"]["path"]
    for root_name in ("old-traces", "records-traces"):
        parent = moved / root_name / "synthetic" / "scenario" / "real" / "none"
        parent.mkdir(parents=True)
        (parent / reference_name).write_bytes((recorded_trace.parent / reference_name).read_bytes())
    append_jsonl(moved / "records.jsonl", [compact])
    with pytest.raises(ValueError, match="ambiguous relocated"):
        read_jsonl(moved / "records.jsonl")


@pytest.mark.parametrize(
    ("before", "after"), [(1, True), (True, 1), (1, 1.0), (1.0, 1), (0.0, -0.0)]
)
def test_context_reuse_preserves_exact_json_scalar_types(
    tmp_path: Path, before: object, after: object
) -> None:
    path = tmp_path / "episode.jsonl"
    rows = [
        {
            "snapshot": {"stable": "synthetic" * 100, "nested": {"value": value}},
            "prompt_token_ids": [value, *range(1000)],
            "messages": [{"value": value}, *[{"text": str(index)} for index in range(50)]],
        }
        for value in (before, after)
    ]
    append_jsonl(path, rows)
    assert json.dumps(read_jsonl(path), sort_keys=True) == json.dumps(rows, sort_keys=True)


def test_json_value_equal_preserves_supported_nonfinite_and_signed_zero_values() -> None:
    assert evidence_store.json_value_equal({"value": float("nan")}, {"value": float("nan")})
    assert evidence_store.json_value_equal([float("inf")], [float("inf")])
    assert not evidence_store.json_value_equal([float("inf")], [float("-inf")])
    assert not evidence_store.json_value_equal({"value": -0.0}, {"value": 0.0})
    assert not evidence_store.json_value_equal({"value": 1}, {"value": True})
    assert not evidence_store.json_value_equal({"value": 1}, {"value": 1.0})


def test_partial_reader_keeps_valid_unterminated_feedback_and_drops_only_malformed_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode.jsonl"
    feedback = {"record": "submission_feedback", "verdict": "synthetic"}
    complete = json.dumps(feedback).encode()
    path.write_bytes(b'{"record":"turn"}\n' + complete)
    assert read_jsonl(path, recover_partial=True) == [{"record": "turn"}, feedback]
    path.write_bytes(b'{"record":"turn"}\n{"record":')
    assert read_jsonl(path, recover_partial=True) == [{"record": "turn"}]
    path.write_bytes(b'{"record":"turn"}\n{"record":\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path, recover_partial=True)
    path.write_bytes(gzip.compress(complete)[:-4])
    assert read_jsonl(path, recover_partial=True) == [feedback]


def test_embedded_pre_summary_and_changed_header_reuse_source_trace_rows(tmp_path: Path) -> None:
    trace_path = tmp_path / "episode.jsonl"
    source = [
        {"record": "header", "state": "source"},
        *[
            {"record": "turn", "prompt_token_ids": list(range(1000)), "turn": turn}
            for turn in range(4)
        ],
    ]
    append_jsonl(trace_path, source)
    pre_summary = {"record": "episode_summary", "state": "pre-leak"}
    for changed_header in (False, True):
        embedded = [
            {**source[0], "state": "different"} if changed_header else source[0],
            *source[1:],
            pre_summary,
        ]
        record = {"trace_path": str(trace_path), "leak_evidence": {"trace_records": embedded}}
        compact = compact_record(record, trace_path)
        assert hydrate_record(compact, trace_path=trace_path) == record
        reference = compact["leak_evidence"]["evidence_reference"]
        stored = json.loads(gzip.decompress((trace_path.parent / reference["path"]).read_bytes()))
        assert isinstance(stored["trace_records"], dict)
        assert len(json.dumps(stored)) < len(json.dumps(embedded)) // 4
