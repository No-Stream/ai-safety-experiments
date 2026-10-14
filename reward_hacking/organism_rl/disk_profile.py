"""Summarize generated-run disk use without exposing artifact values."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ijson

logger = logging.getLogger(__name__)
FileCategory = Literal["records", "provenance", "traces", "other"]
_CATEGORIES: tuple[FileCategory, ...] = ("records", "provenance", "traces", "other")
_SAFE_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z_]*\Z")
_MAX_FIELD_KEY_LENGTH = 64


@dataclass(frozen=True)
class FileGroup:
    """Aggregate physical file sizes for files with the same safe name pattern."""

    label: str
    file_count: int
    total_bytes: int
    largest_file_bytes: int


@dataclass(frozen=True)
class FileProfile:
    """Physical size of one file with identifying trace path components redacted."""

    display_path: str
    category: FileCategory
    file_bytes: int
    compressed: bool


@dataclass(frozen=True)
class RunProfile:
    """Disk and compact field-size summary for one generated run directory."""

    run_name: str
    file_count: int
    total_bytes: int
    category_bytes: dict[FileCategory, int]
    file_groups: tuple[FileGroup, ...]
    files: tuple[FileProfile, ...]
    json_file_count: int
    json_field_bytes: dict[FileCategory, tuple[tuple[str, int], ...]]


@dataclass
class _ContainerFrame:
    kind: Literal["map", "array"]
    path: str
    child_count: int = 0
    pending_key: str | None = None


def profile_run_directory(run_dir: Path, *, max_depth: int = 5) -> RunProfile:
    """Return exact file sizes and compact JSON byte attribution for one run directory."""
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist or is not a directory: {run_dir}")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    category_bytes: dict[FileCategory, int] = dict.fromkeys(_CATEGORIES, 0)
    group_totals: dict[str, list[int]] = {}
    json_fields: dict[FileCategory, Counter[str]] = {
        category: Counter() for category in _CATEGORIES
    }
    json_file_count = 0
    file_profiles: list[FileProfile] = []
    trace_file_index = 0
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())

    for path in files:
        relative_path = path.relative_to(run_dir)
        category = _category(relative_path)
        file_bytes = path.stat().st_size
        compressed = _has_gzip_magic(path)
        category_bytes[category] += file_bytes
        group_label = _file_group(relative_path, category, compressed=compressed)
        group_values = group_totals.setdefault(group_label, [0, 0, 0])
        group_values[0] += 1
        group_values[1] += file_bytes
        group_values[2] = max(group_values[2], file_bytes)

        if category != "other":
            if category == "traces":
                trace_file_index += 1
            file_profiles.append(
                FileProfile(
                    _display_file_path(relative_path, category, trace_file_index),
                    category,
                    file_bytes,
                    compressed,
                )
            )

        if category != "other" and _is_json_file(relative_path, path=path, compressed=compressed):
            _accumulate_json_field_bytes(
                path,
                multiple_values=_contains_jsonl_suffix(relative_path)
                or (compressed and category in {"records", "traces"}),
                totals=json_fields[category],
                compressed=compressed,
                max_depth=max_depth,
            )
            json_file_count += 1

    file_groups = tuple(
        FileGroup(label, values[0], values[1], values[2])
        for label, values in sorted(group_totals.items(), key=lambda item: (-item[1][1], item[0]))
    )
    json_field_bytes: dict[FileCategory, tuple[tuple[str, int], ...]] = {
        category: tuple(sorted(fields.items(), key=lambda item: (-item[1], item[0])))
        for category, fields in json_fields.items()
        if fields
    }
    return RunProfile(
        run_name=run_dir.name,
        file_count=len(files),
        total_bytes=sum(category_bytes.values()),
        category_bytes=category_bytes,
        file_groups=file_groups,
        files=tuple(file_profiles),
        json_file_count=json_file_count,
        json_field_bytes=json_field_bytes,
    )


def _category(relative_path: Path) -> FileCategory:
    if any(part == "traces" or part.endswith("-traces") for part in relative_path.parts):
        return "traces"
    if "provenance" in relative_path.name:
        return "provenance"
    if relative_path.name.startswith("records"):
        return "records"
    return "other"


def _has_gzip_magic(path: Path) -> bool:
    with path.open("rb") as file_handle:
        return file_handle.read(2) == b"\x1f\x8b"


def _contains_jsonl_suffix(relative_path: Path) -> bool:
    return ".jsonl" in relative_path.suffixes


def _is_json_file(relative_path: Path, *, path: Path, compressed: bool) -> bool:
    has_json_suffix = any(suffix in {".json", ".jsonl"} for suffix in relative_path.suffixes)
    return has_json_suffix or (compressed and _gzip_starts_with_json(path))


def _gzip_starts_with_json(path: Path) -> bool:
    with gzip.open(path, "rb") as file_handle:
        leading_bytes = file_handle.read(65_536).lstrip()
    return leading_bytes.startswith((b"{", b"["))


def _display_file_path(relative_path: Path, category: FileCategory, trace_index: int) -> str:
    if category != "traces":
        return relative_path.as_posix()

    suffix = "".join(relative_path.suffixes)
    if "records-traces" in relative_path.parts:
        return f"records-traces/trace-{trace_index:04d}{suffix}"
    if "traces" in relative_path.parts:
        step_directory = next(
            (part for part in relative_path.parts if part.startswith("step-")), ""
        )
        prefix = f"traces/{step_directory}/" if step_directory else "traces/"
        return f"{prefix}trace-{trace_index:04d}{suffix}"
    return f"traces/trace-{trace_index:04d}{suffix}"


def _file_group(relative_path: Path, category: FileCategory, *, compressed: bool) -> str:
    if category == "traces":
        if ".docker-evidence.jsonl" in relative_path.name:
            return "traces/*.docker-evidence.jsonl" + (" [gzip]" if compressed else "")
        if _contains_jsonl_suffix(relative_path):
            return "traces/*.jsonl" + (" [gzip]" if compressed else "")
        if "graded-artifacts" in relative_path.parts:
            return "traces/**/graded-artifacts/*" + (" [gzip]" if compressed else "")
        suffix = "".join(relative_path.suffixes)
        return f"traces/*{suffix or ' (no extension)'}" + (" [gzip]" if compressed else "")
    suffix = "".join(relative_path.suffixes)
    return f"{category}/*{suffix or ' (no extension)'}" + (" [gzip]" if compressed else "")


def _safe_field_path_key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("JSON object keys must be strings")
    return (
        value
        if len(value) <= _MAX_FIELD_KEY_LENGTH and _SAFE_FIELD_NAME.fullmatch(value)
        else "{key}"
    )


def _value_path(stack: list[_ContainerFrame], totals: Counter[str]) -> str:
    if not stack:
        return "$"

    parent = stack[-1]
    if parent.kind == "map":
        if parent.pending_key is None:
            raise ValueError("JSON object value appeared without a field name")
        path = f"{parent.path}.{parent.pending_key}"
        parent.pending_key = None
        return path

    if parent.child_count:
        totals[parent.path] += 1
    parent.child_count += 1
    return f"{parent.path}[]"


def _scalar_bytes(event: str, value: object) -> int:
    if event == "string":
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if event == "number":
        return len(str(value).encode("utf-8"))
    if event == "boolean":
        return 4 if value else 5
    if event == "null":
        return 4
    raise ValueError(f"unsupported JSON scalar event: {event}")


def _accumulate_json_field_bytes(
    path: Path,
    *,
    multiple_values: bool,
    totals: Counter[str],
    compressed: bool,
    max_depth: int,
) -> None:
    """Attribute compact UTF-8 JSON bytes without emitting any values."""
    if multiple_values:
        with gzip.open(path, "rb") if compressed else path.open("rb") as json_file:
            for line in json_file:
                if line.strip():
                    _accumulate_value_bytes(json.loads(line), "$", totals, max_depth=max_depth)
        return

    with gzip.open(path, "rb") if compressed else path.open("rb") as json_file:
        first_event = next(ijson.parse(json_file), None)
    if first_event is not None and first_event[1] == "start_map":
        _accumulate_root_map(path, totals, compressed=compressed, max_depth=max_depth)
        return

    if first_event is not None and first_event[1] == "start_array":
        _accumulate_root_array(path, totals, compressed=compressed, max_depth=max_depth)
        return

    stack: list[_ContainerFrame] = []
    with gzip.open(path, "rb") if compressed else path.open("rb") as json_file:
        for _, event, value in ijson.parse(json_file, multiple_values=multiple_values):
            _process_json_event(event, value, stack, totals)

    if stack:
        raise ValueError("JSON file ended with an open container")


def _accumulate_root_map(
    path: Path, totals: Counter[str], *, compressed: bool, max_depth: int
) -> None:
    with gzip.open(path, "rb") if compressed else path.open("rb") as json_file:
        totals["$"] += 2
        for child_index, (key, value) in enumerate(ijson.kvitems(json_file, "", use_float=True)):
            if child_index:
                totals["$"] += 1
            safe_key = _safe_field_path_key(key)
            totals[f"$.{safe_key}"] += (
                len(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
            )
            _accumulate_value_bytes(value, f"$.{safe_key}", totals, max_depth=max_depth)


def _accumulate_root_array(
    path: Path, totals: Counter[str], *, compressed: bool, max_depth: int
) -> None:
    with gzip.open(path, "rb") if compressed else path.open("rb") as json_file:
        totals["$"] += 2
        for child_index, value in enumerate(ijson.items(json_file, "item", use_float=True)):
            if child_index:
                totals["$"] += 1
            _accumulate_value_bytes(value, "$[]", totals, max_depth=max_depth)


def _path_depth(path: str) -> int:
    return path.count(".") + path.count("[]")


def _accumulate_value_bytes(
    value: object, path: str, totals: Counter[str], *, max_depth: int
) -> None:
    if _path_depth(path) >= max_depth and isinstance(value, (dict, list)):
        totals[path] += len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        return

    if isinstance(value, dict):
        totals[path] += 2
        for child_count, (key, child_value) in enumerate(value.items()):
            if child_count:
                totals[path] += 1
            safe_key = _safe_field_path_key(key)
            totals[f"{path}.{safe_key}"] += (
                len(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
            )
            _accumulate_value_bytes(child_value, f"{path}.{safe_key}", totals, max_depth=max_depth)
        return

    if isinstance(value, list):
        totals[path] += 2 + max(0, len(value) - 1)
        if all(not isinstance(item, (dict, list)) for item in value):
            if value:
                encoded_array_bytes = len(
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )
                totals[f"{path}[]"] += encoded_array_bytes - 2 - (len(value) - 1)
            return
        for item in value:
            _accumulate_value_bytes(item, f"{path}[]", totals, max_depth=max_depth)
        return

    totals[path] += len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _process_json_event(
    event: str, value: object, stack: list[_ContainerFrame], totals: Counter[str]
) -> None:
    if event == "map_key":
        _process_map_key(value, stack, totals)
    elif event in {"start_map", "start_array"}:
        _process_container_start(event, stack, totals)
    elif event in {"end_map", "end_array"}:
        _process_container_end(event, stack, totals)
    else:
        value_path = _value_path(stack, totals)
        totals[value_path] += _scalar_bytes(event, value)


def _process_map_key(value: object, stack: list[_ContainerFrame], totals: Counter[str]) -> None:
    if not stack or stack[-1].kind != "map":
        raise ValueError("JSON object key appeared outside an object")
    parent = stack[-1]
    if parent.pending_key is not None:
        raise ValueError("JSON object key appeared before the previous value")
    field_name = _safe_field_path_key(value)
    if parent.child_count:
        totals[parent.path] += 1
    parent.child_count += 1
    parent.pending_key = field_name
    totals[f"{parent.path}.{field_name}"] += _scalar_bytes("string", value) + 1


def _process_container_start(
    event: str, stack: list[_ContainerFrame], totals: Counter[str]
) -> None:
    value_path = _value_path(stack, totals)
    container_kind: Literal["map", "array"] = "map" if event == "start_map" else "array"
    totals[value_path] += 1
    stack.append(_ContainerFrame(container_kind, value_path))


def _process_container_end(event: str, stack: list[_ContainerFrame], totals: Counter[str]) -> None:
    if not stack:
        raise ValueError("JSON container ended without a matching start")
    frame = stack.pop()
    expected_kind = "map" if event == "end_map" else "array"
    if frame.kind != expected_kind or frame.pending_key is not None:
        raise ValueError("JSON container closed with an incomplete value")
    totals[frame.path] += 1


def render_profile(profile: RunProfile, *, top_fields: int = 30) -> str:
    """Render sizes and field paths without serializing any JSON values."""
    if top_fields < 1:
        raise ValueError("top_fields must be positive")

    lines = [
        f"Run: {profile.run_name}",
        f"Files: {profile.file_count:,}; total: {profile.total_bytes:,} bytes",
        f"JSON files profiled: {profile.json_file_count:,}",
        "File groups:",
    ]
    lines.extend(
        f"  {group.label}: {group.total_bytes:,} bytes across {group.file_count:,} files "
        f"(largest {group.largest_file_bytes:,})"
        for group in profile.file_groups
    )
    lines.append("Relevant files by stored bytes (trace basenames redacted):")
    lines.extend(
        f"  {file_profile.display_path}: {file_profile.file_bytes:,} bytes"
        f"{' [gzip]' if file_profile.compressed else ''}"
        for file_profile in profile.files
    )
    lines.append("JSON field bytes (compact UTF-8 attribution; values omitted):")
    for category in ("records", "provenance", "traces"):
        fields = profile.json_field_bytes.get(category, ())
        if not fields:
            continue
        lines.append(f"  {category}:")
        lines.extend(
            f"    {field_path}: {field_bytes:,}" for field_path, field_bytes in fields[:top_fields]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Profile one run directory from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-fields", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args(argv)
    logger.info(
        "%s",
        render_profile(
            profile_run_directory(args.run_dir, max_depth=args.max_depth),
            top_fields=args.top_fields,
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
