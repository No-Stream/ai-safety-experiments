"""Read, append and atomically write the JSON and JSONL files that Atlas stages pass between each other."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every non-blank line of a JSONL file as a JSON object."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(value)
    return records


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    """Append one record as a sorted-key JSON line and flush it to the file."""
    with path.open("a", encoding="utf-8") as records_file:
        records_file.write(json.dumps(record, sort_keys=True) + "\n")
        records_file.flush()


def write_json(path: Path, record: Mapping[str, object]) -> None:
    """Write an indented JSON file atomically through a sibling temporary file."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)
