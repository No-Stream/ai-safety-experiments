"""Tests for the private run-directory storage profiler."""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl.disk_profile import profile_run_directory, render_profile

if TYPE_CHECKING:
    from pathlib import Path


def test_profile_reports_file_bytes_and_json_field_bytes_without_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    traces_dir = run_dir / "traces" / "step-000000"
    traces_dir.mkdir(parents=True)
    records = '{"record":"synthetic","token_ids":[101,102,103]}\n'
    provenance = '{"max_new_tokens":4096,"flags":{"temperature":0.6}}'
    trace = '{"turn":0,"output_token_ids":[201,202]}\n'
    (run_dir / "records.jsonl").write_text(records, encoding="utf-8")
    (run_dir / "provenance.json").write_text(provenance, encoding="utf-8")
    (traces_dir / "one.jsonl").write_text(trace, encoding="utf-8")
    (traces_dir / "one.docker-evidence.jsonl").write_text(trace, encoding="utf-8")
    (run_dir / "summary.md").write_text("synthetic notes", encoding="utf-8")

    profile = profile_run_directory(run_dir)
    report = render_profile(profile, top_fields=20)

    assert profile.file_count == 5
    assert profile.total_bytes == sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    assert profile.category_bytes["records"] == len(records.encode("utf-8"))
    assert profile.category_bytes["provenance"] == len(provenance.encode("utf-8"))
    assert profile.category_bytes["traces"] == 2 * len(trace.encode("utf-8"))

    record_fields = dict(profile.json_field_bytes["records"])
    assert record_fields["$.token_ids[]"] == len("101") + len("102") + len("103")
    assert record_fields["$.record"] == len(json.dumps("record")) + 1 + len(json.dumps("synthetic"))

    assert "synthetic" not in report
    assert "4096" not in report
    assert "101" not in report
    assert "traces/*.docker-evidence.jsonl" in report
    assert "$.token_ids[]" in report
    assert any(file.display_path == "traces/step-000000/trace-0002.jsonl" for file in profile.files)


def test_profile_redacts_dynamic_object_keys_from_field_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic-run"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text(
        '{"task_prompt_content":{"private-item-123":"synthetic value"}}\n', encoding="utf-8"
    )

    profile = profile_run_directory(run_dir)
    report = render_profile(profile, top_fields=20)

    assert "$ ." not in report
    assert "private-item-123" not in report
    assert "synthetic value" not in report
    assert "$.task_prompt_content.{key}" in report


def test_profile_classifies_ladder_records_traces_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "ladder-run"
    trace_dir = run_dir / "records-traces" / "organism" / "scenario" / "real" / "none"
    trace_dir.mkdir(parents=True)
    trace = '{"prompt_token_ids":[1,2,3]}\n'
    (trace_dir / "sample-000.jsonl").write_text(trace, encoding="utf-8")

    profile = profile_run_directory(run_dir)
    report = render_profile(profile)

    assert profile.category_bytes["traces"] == len(trace.encode("utf-8"))
    assert "traces/*.jsonl" in report
    assert "$.prompt_token_ids[]" in report


def test_profile_attributes_nested_object_arrays_and_compact_primitive_arrays(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "nested-run"
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.jsonl").write_text(
        '{"messages":[{"content":"alpha"},{"content":"beta"}],"ids":[1,20]}\n',
        encoding="utf-8",
    )

    profile = profile_run_directory(run_dir)
    trace_fields = dict(profile.json_field_bytes["traces"])

    key_bytes = len(json.dumps("content")) + 1
    assert trace_fields["$.messages[].content"] == (
        2 * key_bytes + len(json.dumps("alpha")) + len(json.dumps("beta"))
    )
    assert trace_fields["$.ids[]"] == len("1") + len("20")


def test_profile_collapses_deep_subtrees_at_max_depth(tmp_path: Path) -> None:
    run_dir = tmp_path / "deep-run"
    run_dir.mkdir()
    (run_dir / "records.jsonl").write_text(
        '{"a":{"b":[{"c":{"d":{"payload":"synthetic"}}}]}}\n', encoding="utf-8"
    )

    profile = profile_run_directory(run_dir, max_depth=5)
    record_fields = dict(profile.json_field_bytes["records"])
    report = render_profile(profile, top_fields=20)

    assert record_fields["$.a.b[].c.d"] == len(json.dumps("d")) + 1 + len(
        json.dumps({"payload": "synthetic"}, separators=(",", ":"))
    )
    assert "$.a.b[].c.d.payload" not in record_fields
    assert "synthetic" not in report


def test_profile_requires_positive_max_depth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="max_depth must be positive"):
        profile_run_directory(run_dir, max_depth=0)


def test_profile_attributes_root_json_arrays(tmp_path: Path) -> None:
    run_dir = tmp_path / "array-run"
    run_dir.mkdir()
    (run_dir / "records.json").write_text('[{"count":12},{"count":345}]', encoding="utf-8")

    profile = profile_run_directory(run_dir)

    assert dict(profile.json_field_bytes["records"])["$[].count"] == (
        2 * (len(json.dumps("count")) + 1) + len("12") + len("345")
    )


def test_profile_decodes_gzip_jsonl_by_magic_and_reports_physical_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "ladder-run"
    trace_dir = run_dir / "records-traces" / "private-scenario" / "real" / "none"
    trace_dir.mkdir(parents=True)
    compressed_trace = gzip.compress(b'{"prompt_token_ids":[7,8]}\n', mtime=0)
    (trace_dir / "opaque.dat").write_bytes(compressed_trace)

    profile = profile_run_directory(run_dir)
    report = render_profile(profile)

    assert profile.total_bytes == len(compressed_trace)
    assert profile.category_bytes["traces"] == len(compressed_trace)
    assert dict(profile.json_field_bytes["traces"])["$.prompt_token_ids[]"] == 2
    assert profile.files[0].display_path == "records-traces/trace-0001.dat"
    assert profile.files[0].compressed is True
    assert f"{len(compressed_trace):,} bytes" in report
    assert "private-scenario" not in report
    assert "opaque.dat" not in report
    assert "$.prompt_token_ids[]" in report
