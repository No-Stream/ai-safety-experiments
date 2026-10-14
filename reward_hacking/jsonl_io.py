"""Read, append and atomically write JSON and JSONL files.

Variants differ only in serialisation or error behaviour and are named for it: callers must keep the variant they had
so file bytes and exception messages stay unchanged.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from reward_hacking.harness.evidence_store import read_jsonl

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read non-blank JSONL lines as objects; a non-object row raises ValueError naming path and line."""
    try:
        return read_jsonl(path)
    except TypeError as error:
        raise ValueError(str(error)) from error


def append_jsonl_ascii(path: Path, record: Mapping[str, object]) -> None:
    """Append one record as a sorted-key, ASCII-escaped JSON line and flush it to the file."""
    with path.open("a", encoding="utf-8") as records_file:
        records_file.write(json.dumps(record, sort_keys=True) + "\n")
        records_file.flush()


def append_jsonl_unicode(path: Path, record: Mapping[str, object]) -> None:
    """Append one record as a sorted-key JSON line with unescaped unicode and flush it to the file."""
    with path.open("a", encoding="utf-8") as records_file:
        records_file.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        records_file.flush()


def write_json_atomic(path: Path, record: Mapping[str, object]) -> None:
    """Write an indented, sorted-key, ASCII-escaped JSON file through a sibling temporary file."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def write_json_atomic_unicode_strict(path: Path, record: Mapping[str, object]) -> None:
    """Like write_json_atomic, but with unescaped unicode and NaN or infinity rejected."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
