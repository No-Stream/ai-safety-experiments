from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.atlas.jsonl_io import append_jsonl, read_jsonl, write_json

if TYPE_CHECKING:
    from pathlib import Path


class TestAppendAndReadJsonl:
    def test_read_returns_appended_records_in_order(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        append_jsonl(records_path, {"key": "b", "score": 2})
        append_jsonl(records_path, {"key": "a", "score": 1})
        assert read_jsonl(records_path) == [{"key": "b", "score": 2}, {"key": "a", "score": 1}]

    def test_append_writes_sorted_keys_one_record_per_line(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        append_jsonl(records_path, {"zeta": 1, "alpha": 2})
        assert records_path.read_text(encoding="utf-8") == '{"alpha": 2, "zeta": 1}\n'

    def test_read_skips_blank_lines(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        records_path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
        assert read_jsonl(records_path) == [{"a": 1}, {"a": 2}]

    def test_read_rejects_line_that_is_not_an_object(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        records_path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"records\.jsonl:2 is not a JSON object"):
            read_jsonl(records_path)

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_jsonl(tmp_path / "absent.jsonl")


class TestWriteJson:
    def test_writes_indented_sorted_json_and_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        summary_path = tmp_path / "summary.json"
        record = {"b": 1, "a": [1]}
        write_json(summary_path, record)
        expected_text = json.dumps(record, indent=2, sort_keys=True) + "\n"
        assert summary_path.read_text(encoding="utf-8") == expected_text
        assert [path.name for path in tmp_path.iterdir()] == ["summary.json"]

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        summary_path = tmp_path / "summary.json"
        summary_path.write_text("stale", encoding="utf-8")
        write_json(summary_path, {"status": "fresh"})
        assert json.loads(summary_path.read_text(encoding="utf-8")) == {"status": "fresh"}
