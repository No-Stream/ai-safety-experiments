"""Lossless compressed episode evidence with transparent legacy readers."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import logging
import os
import zlib
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)
_FORMAT = "episode-evidence-v1"
_REFERENCE = "evidence_reference"
_CACHE_LIMIT = 8
_LADDER_TRACE_COMPONENTS = 5
_APPEND_LOCK = RLock()
_APPEND_CACHE: OrderedDict[Path, tuple[int, int, dict[str, Any]]] = OrderedDict()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def json_value_equal(left: object, right: object) -> bool:
    """Compare JSON values without collapsing booleans, numeric types, or signed zero."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        first = cast("dict[str, Any]", left)
        second = cast("dict[str, Any]", right)
        return first.keys() == second.keys() and all(
            json_value_equal(value, second[key]) for key, value in first.items()
        )
    if isinstance(left, list):
        first_list = cast("list[Any]", left)
        second_list = cast("list[Any]", right)
        return len(first_list) == len(second_list) and all(
            json_value_equal(first_value, second_value)
            for first_value, second_value in zip(first_list, second_list, strict=True)
        )
    if isinstance(left, float):
        return _json_bytes(left) == _json_bytes(right)
    return left == right


def _sequence_delta(
    current: str | list[Any], prior: str | list[Any], path: list[str]
) -> dict[str, Any] | None:
    length = 0
    for left, right in zip(current, prior, strict=False):
        if not json_value_equal(left, right):
            break
        length += 1
    candidates = []
    if length:
        candidates.append({"path": path, "length": length, "suffix": current[length:]})
    if isinstance(current, list) and isinstance(prior, list):
        updates = [
            [index, value]
            for index, (value, previous) in enumerate(zip(current, prior, strict=False))
            if not json_value_equal(value, previous)
        ]
        candidates.append(
            {
                "path": path,
                "length": len(current),
                "updates": updates,
                "suffix": current[len(prior) :],
            }
        )
    if not candidates:
        return None
    selected = min(candidates, key=lambda candidate: len(_json_bytes(candidate)))
    return selected if len(_json_bytes(selected)) + 16 < len(_json_bytes(current)) else None


def _restore_sequence(prior: str | list[Any], delta: Mapping[str, Any]) -> str | list[Any]:
    length = delta["length"]
    if not isinstance(length, int) or length < 0:
        raise ValueError("invalid episode context prefix length")
    if "updates" in delta:
        if not isinstance(prior, list):
            raise TypeError("sparse context updates require a list")
        restored = copy.deepcopy(prior[:length] + delta["suffix"])
        if len(restored) != length:
            raise ValueError("invalid episode sparse context length")
        for index, value in delta["updates"]:
            if not isinstance(index, int) or index < 0 or index >= min(length, len(prior)):
                raise ValueError("invalid episode sparse context index")
            restored[index] = value
        return restored
    if length > len(prior):
        raise ValueError("invalid episode context prefix length")
    return prior[:length] + delta["suffix"]


def _prefix_fields(
    value: dict[str, Any],
    previous: Mapping[str, Any],
    path: list[str],
    prefixes: list[dict[str, Any]],
) -> None:
    for key, current in list(value.items()):
        prior = previous.get(key)
        field_path = [*path, key]
        if isinstance(current, dict) and isinstance(prior, dict):
            copy_reference = {"path": field_path, "copy": True}
            if json_value_equal(current, prior) and len(_json_bytes(copy_reference)) + 16 < len(
                _json_bytes(current)
            ):
                del value[key]
                prefixes.append(copy_reference)
            else:
                _prefix_fields(current, prior, field_path, prefixes)
        elif (
            isinstance(current, (str, list))
            and isinstance(prior, (str, list))
            and type(current) is type(prior)
        ):
            delta = _sequence_delta(current, prior, field_path)
            if delta is not None:
                del value[key]
                prefixes.append(delta)


def _decode_row(value: object, previous: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("episode JSONL must contain objects")
    if value.get("evidence_store_format") != _FORMAT:
        return cast("dict[str, Any]", value)
    record = value["record"]
    if not isinstance(record, dict):
        raise TypeError("compact episode row record must be an object")
    for prefix in value["prefixes"]:
        destination = record
        source: Any = previous
        path = prefix["path"]
        for key in path[:-1]:
            destination = destination[key]
            source = source[key]
        prior = source[path[-1]]
        if prefix.get("copy") is True:
            destination[path[-1]] = copy.deepcopy(prior)
            continue
        destination[path[-1]] = _restore_sequence(prior, prefix)
    return cast("dict[str, Any]", record)


def _read_bytes(path: Path, *, recover_partial: bool) -> bytes:
    payload = path.read_bytes()
    if not payload.startswith(b"\x1f\x8b"):
        return payload
    if not recover_partial:
        return gzip.decompress(payload)
    output = bytearray()
    while payload:
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        output.extend(decoder.decompress(payload))
        if not decoder.eof:
            break
        payload = decoder.unused_data
    return bytes(output)


def _read_rows(path: Path, *, recover_partial: bool) -> list[dict[str, Any]]:
    payload = _read_bytes(path, recover_partial=recover_partial)
    lines = payload.splitlines()
    unterminated_tail = bool(payload) and not payload.endswith(b"\n")
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] = {}
    for line_number, line in enumerate(lines, 1):
        if line.strip():
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if recover_partial and unterminated_tail and line_number == len(lines):
                    break
                raise
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            previous = _decode_row(value, previous)
            rows.append(previous)
    return rows


def iter_jsonl(
    path: Path, *, recover_partial: bool = False, hydrate: bool = True
) -> Iterator[dict[str, Any]]:
    """Stream reconstructed episode rows while retaining only the previous context."""
    if recover_partial:
        for row in _read_rows(path, recover_partial=True):
            yield hydrate_record(row, base_path=path) if hydrate else row
        return
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
    previous: dict[str, Any] = {}
    with gzip.open(path, "rb") if compressed else path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            previous = _decode_row(value, previous)
            yield hydrate_record(previous, base_path=path) if hydrate else copy.deepcopy(previous)


def read_jsonl(
    path: Path, *, recover_partial: bool = False, hydrate: bool = True
) -> list[dict[str, Any]]:
    """Read plain or gzip JSONL, restoring exact contexts and referenced evidence."""
    return list(iter_jsonl(path, recover_partial=recover_partial, hydrate=hydrate))


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]], *, append: bool = True) -> None:
    """Persist gzip members, reusing cached context or starting a self-contained row."""
    with _APPEND_LOCK:
        _append_jsonl(path, records, append=append)


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]], *, append: bool) -> None:
    path = Path(path)
    rows = [dict(row) for row in records]
    if not rows and append:
        return
    previous: dict[str, Any] = {}
    mode = "ab" if append else "wb"
    if append and path.exists() and path.stat().st_size:
        stat = path.stat()
        cache = _APPEND_CACHE.get(path.absolute())
        if cache is not None and cache[:2] == (stat.st_size, stat.st_mtime_ns):
            previous = cache[2]
        # A cache miss starts a full row, which resets the decoder's context. Replaying
        # a resumed or evicted stream here would make incremental writes quadratic.
        with path.open("rb") as existing_file:
            compressed_existing = existing_file.read(2) == b"\x1f\x8b"
        if not compressed_existing:
            rows = [*_read_rows(path, recover_partial=False), *rows]
            previous = {}
            mode = "wb"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name + ".writing") if mode == "wb" else path
    with (
        target.open(mode) as output,
        gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed,
    ):
        for row in rows:
            # Detach shared Python objects before removing fields from the JSON tree.
            stored = json.loads(_json_bytes(row))
            prefixes: list[dict[str, Any]] = []
            _prefix_fields(stored, previous, [], prefixes)
            compressed.write(
                _json_bytes(
                    {"evidence_store_format": _FORMAT, "record": stored, "prefixes": prefixes}
                )
                + b"\n"
            )
            previous = row
    if mode == "wb":
        with target.open("rb") as completed:
            os.fsync(completed.fileno())
        target.replace(path)
    stat = path.stat()
    _APPEND_CACHE[path.absolute()] = (stat.st_size, stat.st_mtime_ns, copy.deepcopy(previous))
    _APPEND_CACHE.move_to_end(path.absolute())
    while len(_APPEND_CACHE) > _CACHE_LIMIT:
        _APPEND_CACHE.popitem(last=False)


def _before_summary(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    prefix = []
    for row in rows:
        if row.get("record") == "episode_summary":
            break
        prefix.append(row)
    return prefix


def _externalize_components(
    evidence: dict[str, Any],
    trace_path: Path,
    docker_text: str | None,
    trace_records: Sequence[Mapping[str, Any]] | None,
) -> None:
    docker_rows = evidence.get("docker_evidence")
    if docker_text is not None and isinstance(docker_rows, list):
        docker_path = Path(docker_text)
        if not docker_path.is_absolute():
            docker_path = trace_path.parent / docker_path
        if (
            docker_path.parent.resolve() == trace_path.parent.resolve()
            and docker_path.exists()
            and json_value_equal(read_jsonl(docker_path), docker_rows)
        ):
            with docker_path.open("rb") as docker_file:
                already_compressed = docker_file.read(2) == b"\x1f\x8b"
            if not already_compressed:
                append_jsonl(docker_path, docker_rows, append=False)
            evidence["docker_evidence"] = {
                "evidence_component_reference": {
                    "path": os.path.relpath(docker_path, trace_path.parent),
                    "locator": "all",
                }
            }
    embedded_rows = evidence.get("trace_records")
    if isinstance(embedded_rows, list):
        available_rows = (
            list(trace_records)
            if trace_records is not None
            else (read_jsonl(trace_path, hydrate=False) if trace_path.exists() else [])
        )
        prefix_rows = _before_summary(available_rows)
        if json_value_equal(prefix_rows, embedded_rows):
            evidence["trace_records"] = {
                "evidence_component_reference": {
                    "path": "self",
                    "locator": "before_episode_summary",
                }
            }
        else:
            delta = _sequence_delta(embedded_rows, prefix_rows, [])
            if delta is not None:
                del delta["path"]
                evidence["trace_records"] = {
                    "evidence_component_reference": {
                        "path": "self",
                        "locator": "trace_delta",
                        "delta": delta,
                    }
                }


def compact_record(
    record: Mapping[str, Any],
    trace_path: Path,
    *,
    trace_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Externalize leak evidence into one sidecar shared by the summary and record."""
    result = copy.deepcopy(dict(record))
    container = result.get("trace_summary", result)
    if not isinstance(container, dict):
        raise TypeError("trace_summary must be an object")
    evidence = container.get("leak_evidence")
    if evidence is None or (isinstance(evidence, dict) and _REFERENCE in evidence):
        return result
    if not isinstance(evidence, dict):
        raise TypeError("leak_evidence must be an object")
    _externalize_components(
        evidence,
        trace_path,
        container.get("docker_evidence_path", result.get("docker_evidence_path")),
        trace_records,
    )
    payload = _json_bytes(evidence)
    digest = hashlib.sha256(payload).hexdigest()
    trace_stem = Path(trace_path.name.removesuffix(".tmp")).stem
    sidecar = trace_path.with_name(f"{trace_stem}.leak-evidence.{digest}.json.gz")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if not sidecar.exists() or gzip.decompress(sidecar.read_bytes()) != payload:
        temporary = sidecar.with_name(sidecar.name + ".writing")
        temporary.write_bytes(gzip.compress(payload, mtime=0))
        temporary.replace(sidecar)
    container["leak_evidence"] = {
        _REFERENCE: {"path": sidecar.name, "anchor": "trace_parent", "format": _FORMAT}
    }
    return result


def _local_evidence_parent(trace: Path, base_path: Path, reference_name: str) -> Path | None:
    candidates = [base_path.parent]
    if not trace.is_absolute():
        candidates.append((base_path.parent / trace).parent)
    for index, component in enumerate(trace.parts):
        if component == "traces" or component.endswith("-traces"):
            candidates.append(base_path.parent.joinpath(*trace.parts[index:]).parent)
    tail = trace.parts[-_LADDER_TRACE_COMPONENTS:]
    if len(tail) == _LADDER_TRACE_COMPONENTS and all(part not in {".", "..", "/"} for part in tail):
        candidates.append(base_path.with_name(f"{base_path.stem}-traces").joinpath(*tail).parent)
    matching = {
        candidate.resolve() for candidate in candidates if (candidate / reference_name).is_file()
    }
    if len(matching) > 1:
        raise ValueError("ambiguous relocated episode evidence directories")
    return next(iter(matching)) if matching else None


def _reference_parent(
    record: Mapping[str, Any], base_path: Path | None, trace_path: Path | None
) -> Path:
    trace_text = record.get("trace_path")
    if trace_path is not None:
        parent = trace_path.parent
    elif trace_text is not None:
        trace = Path(trace_text)
        if base_path is not None:
            container = record.get("trace_summary", record)
            reference_name = container["leak_evidence"][_REFERENCE]["path"]
            relocated_parent = _local_evidence_parent(trace, base_path, reference_name)
            if relocated_parent is not None:
                return relocated_parent
        if not trace.is_absolute():
            if base_path is None:
                raise ValueError("relative trace evidence requires base_path")
            trace = base_path.parent / trace
        parent = trace.parent
    elif base_path is not None:
        parent = base_path.parent
    else:
        raise ValueError("episode evidence reference requires trace_path or base_path")
    return parent


def _hydrate_components(evidence: dict[str, Any], parent: Path, source_trace: Path | None) -> None:
    for field in ("docker_evidence", "trace_records"):
        component = evidence.get(field)
        if isinstance(component, dict) and "evidence_component_reference" in component:
            component_reference = component["evidence_component_reference"]
            component_path = (
                source_trace
                if component_reference["path"] == "self"
                else parent / component_reference["path"]
            )
            if component_path is None:
                raise ValueError("self trace evidence requires containing trace path")
            rows = read_jsonl(component_path, hydrate=False)
            if component_reference["locator"] == "before_episode_summary":
                rows = _before_summary(rows)
            elif component_reference["locator"] == "trace_delta":
                rows = _restore_sequence(_before_summary(rows), component_reference["delta"])
            elif component_reference["locator"] != "all":
                raise ValueError("unsupported episode evidence component locator")
            evidence[field] = rows


def hydrate_record(
    record: Mapping[str, Any], *, base_path: Path | None = None, trace_path: Path | None = None
) -> dict[str, Any]:
    """Restore a sidecar; base_path is the containing JSONL file for portable references."""
    result = copy.deepcopy(dict(record))
    container = result.get("trace_summary", result)
    if not isinstance(container, dict):
        raise TypeError("trace_summary must be an object")
    evidence = container.get("leak_evidence")
    if not isinstance(evidence, dict) or _REFERENCE not in evidence:
        return result
    reference = evidence[_REFERENCE]
    if reference["format"] != _FORMAT or reference["anchor"] != "trace_parent":
        raise ValueError("unsupported episode evidence reference")
    parent = _reference_parent(result, base_path, trace_path)
    restored = json.loads(gzip.decompress((parent / reference["path"]).read_bytes()))
    if not isinstance(restored, dict):
        raise TypeError("episode evidence sidecar must contain an object")
    trace_text = result.get("trace_path")
    source_trace = trace_path or (
        parent / Path(trace_text).name if trace_text is not None else base_path
    )
    _hydrate_components(restored, parent, source_trace)
    container["leak_evidence"] = restored
    return result


def _field_sizes(value: object, path: str, sizes: list[tuple[int, str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _field_sizes(child, f"{path}.{key}", sizes)
    else:
        sizes.append((len(_json_bytes(value)), path))


def _last_row(path: Path) -> dict[str, Any]:
    cached = _APPEND_CACHE.get(path.absolute())
    stat = path.stat()
    if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
        return cached[2]
    last: dict[str, Any] = {}
    for row in iter_jsonl(path, hydrate=False):
        last = row
    return last


def _evidence_paths(paths: Sequence[Path]) -> list[Path]:
    pending = list(paths)
    existing: dict[Path, Path] = {}
    while pending:
        path = pending.pop()
        identity = path.absolute()
        if identity in existing or not path.exists():
            continue
        existing[identity] = path
        if path.name.endswith(".json.gz"):
            evidence = json.loads(gzip.decompress(path.read_bytes()))
            for field in ("docker_evidence", "trace_records"):
                component = evidence.get(field)
                if isinstance(component, dict) and "evidence_component_reference" in component:
                    component_path = component["evidence_component_reference"]["path"]
                    if component_path != "self":
                        pending.append(path.parent / component_path)
        else:
            last = _last_row(path)
            summary = last.get("trace_summary", last)
            evidence = summary.get("leak_evidence") if isinstance(summary, dict) else None
            if isinstance(evidence, dict) and _REFERENCE in evidence:
                parent = _reference_parent(last, path, None)
                pending.append(parent / evidence[_REFERENCE]["path"])
    return list(existing.values())


def guard_episode_size(
    paths: Sequence[Path],
    *,
    warning_bytes: int = 25 * 1024**2,
    hard_bytes: int = 200 * 1024**2,
    additional_bytes: Mapping[str, int] | None = None,
) -> None:
    """Warn or refuse oversized stored episodes with their largest logical evidence fields."""
    if not 0 <= warning_bytes <= hard_bytes:
        raise ValueError("episode size thresholds must satisfy 0 <= warning <= hard")
    existing = _evidence_paths(paths)
    extra_fields = list((additional_bytes or {}).items())
    if any(size < 0 for _, size in extra_fields):
        raise ValueError("additional evidence sizes must be nonnegative")
    total = sum(path.stat().st_size for path in existing) + sum(size for _, size in extra_fields)
    if total <= warning_bytes:
        return
    fields: list[tuple[int, str]] = [(size, field) for field, size in extra_fields]
    for path in existing:
        if path.name.endswith(".json.gz"):
            _field_sizes(json.loads(gzip.decompress(path.read_bytes())), path.name, fields)
        else:
            for index, row in enumerate(iter_jsonl(path, hydrate=False)):
                _field_sizes(row, f"{path.name}[{index}]", fields)
    largest = ", ".join(
        f"{field} ({size} bytes)" for size, field in sorted(fields, reverse=True)[:5]
    )
    message = f"episode evidence uses {total} stored bytes; largest fields: {largest}"
    if total > hard_bytes:
        raise ValueError(message)
    logger.warning(message)
