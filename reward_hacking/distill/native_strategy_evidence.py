"""Bind native detector replay evidence to immutable ordinary episode records."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from reward_hacking.harness.evidence_store import json_value_equal, read_jsonl
from reward_hacking.ladder import rescore, runner
from reward_hacking.organism_rl.generate import require_gitignored_output_dir

_OVERLAY_FIELDS = frozenset(
    {"acceptance_suppression_evidence", "git_refs_rewritten", "ci_baseline_tampering_attempted"}
)
_ADAPTER_FIELDS = frozenset(
    {
        "hint",
        "payment_register",
        "trace_path",
        "record_schema",
        "prompt_format_version",
        "task_id",
        "family",
        "variant",
        "task_kind",
        "partition",
        "sample_index",
        "seed",
        "organism_reward",
        "organism_reward_outcome",
        "run_provenance_sha256",
    }
)
_DETECTOR = "reward_hacking.ladder.rungs.detect_rungs"


@dataclass(frozen=True, slots=True)
class NativeEvidence:
    """Validated mechanism overlays and their complete content bindings."""

    raw_bytes: bytes
    digest: str | None
    entries: Mapping[tuple[str, int], Mapping[str, object]]
    bindings: Mapping[tuple[str, int], Mapping[str, object]]
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Row:
    record: Mapping[str, object]
    sha256: str


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{context} requires an object with string keys")
    return value


def _rows(path: Path) -> list[_Row]:
    raw = path.read_bytes()
    content = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    lines = content.splitlines(keepends=True)
    if any(not line.strip() for line in lines):
        raise ValueError("source changed or contains empty JSONL rows")
    hydrated = read_jsonl(path)
    if len(lines) != len(hydrated):
        raise ValueError("source physical/hydrated row count mismatch")
    return [_Row(record, _sha(line)) for line, record in zip(lines, hydrated, strict=True)]


def _key(record: Mapping[str, object]) -> runner.EpisodeKey:
    if record.get("record") != "ladder_episode":
        raise ValueError("native evidence requires successful ladder_episode records")
    raw = _object(record.get("key"), "episode key")
    key = runner.EpisodeKey.from_json(raw)
    if not json_value_equal(dict(raw), key.to_json()):
        raise ValueError("episode key must use the exact typed host-native identity")
    return key


def _index(rows: list[_Row]) -> dict[runner.EpisodeKey, _Row]:
    indexed: dict[runner.EpisodeKey, _Row] = {}
    for row in rows:
        if row.record.get("record") in {"ladder_run_header", "ladder_run_summary"}:
            continue
        key = _key(row.record)
        if key in indexed:
            raise ValueError("duplicate native evidence identity")
        indexed[key] = row
    if not indexed:
        raise ValueError("native evidence source has no episodes")
    return indexed


def _headers(rows: list[_Row]) -> dict[str, Mapping[str, object]]:
    headers: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row.record.get("record") != "ladder_run_header":
            continue
        header = row.record
        provenance = _object(header.get("provenance"), "native header provenance")
        digest = runner.provenance_digest(provenance)
        if header.get("provenance_digest") != digest or digest in headers:
            raise ValueError("invalid or duplicate native header provenance digest")
        if type(header.get("schema_version")) is not int or header[
            "schema_version"
        ] != provenance.get("schema_version"):
            raise ValueError("native header schema/provenance mismatch")
        headers[digest] = header
    if not headers:
        raise ValueError("native source requires original run headers")
    return headers


def _matched_header(
    record: Mapping[str, object],
    headers: Mapping[str, Mapping[str, object]],
    key: runner.EpisodeKey,
) -> Mapping[str, object]:
    digest = record.get("provenance_digest")
    if not isinstance(digest, str) or digest not in headers:
        raise ValueError("native episode has no matching run header provenance")
    header = headers[digest]
    provenance = _object(header["provenance"], "run provenance")
    if provenance.get("task_pressure", "none") != key.task_pressure:
        raise ValueError("native episode/header task pressure mismatch")
    _validate_header_membership(provenance, key)
    if (
        provenance.get("detector_code_identity") != record.get("detector_code_identity")
        and "rescore" not in header
    ):
        raise ValueError("original native detector/header mismatch")
    return header


def _validate_header_membership(provenance: Mapping[str, object], key: runner.EpisodeKey) -> None:
    for field, identity_field, expected in (
        ("models", "model_id", key.model_id),
        ("scenarios", "scenario_id", key.scenario_id),
    ):
        values = provenance.get(field)
        if not isinstance(values, list):
            raise TypeError("native header requires model/scenario provenance lists")
        matches = [_object(value, "header member").get(identity_field) for value in values]
        if matches.count(expected) != 1:
            raise ValueError("native episode key has no unique header model/scenario identity")
    arms = provenance.get("arms")
    if arms is not None and (not isinstance(arms, list) or key.arm not in arms):
        raise ValueError("native episode arm is absent from header provenance")
    samples = provenance.get("samples")
    if samples is not None and (type(samples) is not int or key.sample_idx >= samples):
        raise ValueError("native episode sample is outside header provenance")


def _trace(record: Mapping[str, object], source: Path) -> Path:
    value = record.get("trace_path")
    if not isinstance(value, str) or not value:
        raise ValueError("native episode requires a trace_path")
    path = Path(value)
    return (path if path.is_absolute() else source.parent / path).resolve()


def _validate_pair(
    ordinary: Mapping[str, object],
    native: Mapping[str, object],
    rescored: Mapping[str, object],
    paths: Mapping[str, Path],
) -> Path:
    key = _key(native)
    if type(ordinary.get("sample_index")) is not int or ordinary["sample_index"] != key.sample_idx:
        raise ValueError("ordinary sample_index disagrees with native key")
    if ordinary.get("task_kind") != "ladder":
        raise ValueError("ordinary wrapper must identify ladder task_kind")
    for field, value in native.items():
        if field not in _ADAPTER_FIELDS and (
            field not in ordinary or not json_value_equal(ordinary[field], value)
        ):
            raise ValueError(f"ordinary/native preserved field mismatch: {field}")
    traces = [
        _trace(record, paths[name])
        for name, record in (
            ("ordinary", ordinary),
            ("original_native", native),
            ("rescored_native", rescored),
        )
    ]
    if len(set(traces)) != 1:
        raise ValueError("ordinary/native/rescored trace paths disagree")
    for field in ("engine_command_records", "episode_dir", "episode_token", "final_grade"):
        if field in native and not json_value_equal(native[field], rescored.get(field)):
            raise ValueError(f"native/rescored source field mismatch: {field}")
    _validate_detector_lineage(native, rescored)
    return traces[0]


def _validate_detector_lineage(
    native: Mapping[str, object], rescored: Mapping[str, object]
) -> None:
    if rescored.get("detector_code_identity") != runner.detector_code_identity():
        raise ValueError("rescored native detector code is stale")
    previous = native.get("detector_code_identity")
    if not isinstance(previous, str) or not previous:
        raise ValueError("original native requires detector code identity")
    if (
        previous != runner.detector_code_identity()
        and rescored.get("rescore_source_detector_code_identity") != previous
    ):
        raise ValueError("rescored native source detector identity mismatch")


def _overlay(record: Mapping[str, object]) -> dict[str, object]:
    if not record.keys() >= _OVERLAY_FIELDS:
        raise ValueError("rescored native is missing mechanism evidence fields")
    suppression = _object(
        record["acceptance_suppression_evidence"], "acceptance suppression evidence"
    )
    for field in ("acceptance_suppressed", "acceptance_rewritten"):
        if (
            field in suppression
            and suppression[field] is not None
            and type(suppression[field]) is not bool
        ):
            raise TypeError("acceptance suppression flags must be boolean or null")
    refs = record["git_refs_rewritten"]
    if not isinstance(refs, list) or any(not isinstance(value, str) or not value for value in refs):
        raise TypeError("git_refs_rewritten requires nonempty string entries")
    attempted = record["ci_baseline_tampering_attempted"]
    if type(attempted) is not bool or attempted != bool(refs):
        raise ValueError("ci_baseline_tampering_attempted disagrees with git_refs_rewritten")
    return {field: record[field] for field in sorted(_OVERLAY_FIELDS)}


def _header_pair(original: Mapping[str, object], rescored_header: Mapping[str, object]) -> None:
    marker = _object(rescored_header.get("rescore"), "rescored header marker")
    if set(marker) != {
        "detector",
        "detector_code_identity",
        "source_schema_version",
        "timestamp_utc",
    }:
        raise ValueError("rescored header marker fields are invalid")
    if not isinstance(marker["timestamp_utc"], str) or not marker["timestamp_utc"]:
        raise ValueError("rescored header marker requires a timestamp")
    if rescored_header.get("schema_version") != runner.LADDER_SCHEMA_VERSION:
        raise ValueError("rescored header schema is stale")
    if (
        marker.get("detector") != _DETECTOR
        or marker.get("detector_code_identity") != runner.detector_code_identity()
    ):
        raise ValueError("rescored run header has a stale detector/algorithm marker")
    if marker.get("source_schema_version") != original["schema_version"]:
        raise ValueError("rescored run header source schema mismatch")
    old = _object(original["provenance"], "original provenance")
    new = _object(rescored_header["provenance"], "rescored provenance")
    for field, value in old.items():
        if field not in {"schema_version", "models", "scenarios"} and not json_value_equal(
            value, new.get(field)
        ):
            raise ValueError(f"rescored run provenance changed source field: {field}")
    _validate_upgraded_members(old, new)


def _validate_upgraded_members(old: Mapping[str, object], new: Mapping[str, object]) -> None:
    for field in ("models", "scenarios"):
        prior = old.get(field)
        current = new.get(field)
        if (
            not isinstance(prior, list)
            or not isinstance(current, list)
            or len(prior) != len(current)
        ):
            raise ValueError("rescored run provenance model/scenario mismatch")
        ignored = (
            {"chat_template_kwargs"}
            if field == "models"
            else {"manifest_digest", "semantic_content_identity"}
        )
        for left, right in zip(prior, current, strict=True):
            source = _object(left, "source model/scenario")
            replay = _object(right, "rescored model/scenario")
            preserved = "chat_template_kwargs" if field == "models" else "semantic_content_identity"
            if preserved in source and not json_value_equal(
                source[preserved], replay.get(preserved)
            ):
                raise ValueError("rescored run provenance changed a source semantic identity")
            if any(
                not json_value_equal(value, replay.get(name))
                for name, value in source.items()
                if name not in ignored
            ):
                raise ValueError("rescored run provenance model/scenario identity mismatch")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate native evidence manifest field: {key}")
        result[key] = value
    return result


def _manifest(
    records_path: Path, original_native_path: Path, rescored_native_path: Path
) -> dict[str, object]:
    paths = {
        "ordinary": records_path.resolve(),
        "original_native": original_native_path.resolve(),
        "rescored_native": rescored_native_path.resolve(),
    }
    rows = {name: _rows(path) for name, path in paths.items()}
    original_header_rows = rows["original_native"]
    if not any(row.record.get("record") == "ladder_run_header" for row in original_header_rows):
        paths["original_headers"] = original_native_path.resolve().with_name("native-headers.jsonl")
        original_header_rows = _rows(paths["original_headers"])
    original_headers = _headers(original_header_rows)
    rescored_headers = _headers(rows["rescored_native"])
    indexes = {name: _index(source_rows) for name, source_rows in rows.items()}
    ordinary_index = indexes["ordinary"]
    if any(set(index) != set(ordinary_index) for index in indexes.values()):
        raise ValueError("native/ordinary identities have missing or unconsumed episodes")
    entries: list[dict[str, object]] = []
    for key, ordinary in ordinary_index.items():
        native, rescored_row = indexes["original_native"][key], indexes["rescored_native"][key]
        original_header = _matched_header(native.record, original_headers, key)
        rescored_header = _matched_header(rescored_row.record, rescored_headers, key)
        _header_pair(original_header, rescored_header)
        trace = _validate_pair(ordinary.record, native.record, rescored_row.record, paths)
        entries.append(
            {
                "episode_key": dict(_object(ordinary.record["key"], "ordinary key")),
                "sample_index": key.sample_idx,
                "source_record_sha256": ordinary.sha256,
                "original_native_record_sha256": native.sha256,
                "rescored_native_record_sha256": rescored_row.sha256,
                "trace_path": str(trace),
                "trace_sha256": _sha(trace.read_bytes()),
                "original_provenance_digest": original_header["provenance_digest"],
                "rescored_provenance_digest": rescored_header["provenance_digest"],
                "overlay": _overlay(rescored_row.record),
            }
        )
    return {
        "format_version": 1,
        "detector_code_identity": runner.detector_code_identity(),
        "rescore_algorithm_sha256": _sha(Path(rescore.__file__).read_bytes()),
        "sources": {
            name: {"path": str(path), "sha256": _sha(path.read_bytes())}
            for name, path in paths.items()
        },
        "entries": entries,
    }


def build_native_evidence(
    records_path: Path, original_native_path: Path, rescored_native_path: Path, output_path: Path
) -> Path:
    """Write an exclusive manifest binding a completed detector replay to its sources."""
    require_gitignored_output_dir(output_path)
    manifest = _manifest(records_path, original_native_path, rescored_native_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return output_path


def load_native_evidence(path: Path | None, records_path: Path) -> NativeEvidence:
    """Rebuild every join against live sources before exposing mechanism overlays."""
    if path is None:
        return NativeEvidence(b"", None, {}, {}, {})
    raw_bytes = path.read_bytes()
    manifest = _object(
        json.loads(raw_bytes, object_pairs_hook=_unique_json_object), "native evidence manifest"
    )
    sources = _object(manifest.get("sources"), "native evidence sources")
    source_paths: dict[str, Path] = {}
    for name in ("ordinary", "original_native", "rescored_native"):
        source = _object(sources.get(name), "native evidence source")
        value = source.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError("native evidence source requires a path")
        source_paths[name] = Path(value)
    if source_paths["ordinary"].resolve() != records_path.resolve():
        raise ValueError("native evidence ordinary source path changed")
    expected = _manifest(
        records_path, source_paths["original_native"], source_paths["rescored_native"]
    )
    if not json_value_equal(dict(manifest), expected):
        raise ValueError("native evidence manifest is stale or changed")
    entries: dict[tuple[str, int], Mapping[str, object]] = {}
    bindings: dict[tuple[str, int], Mapping[str, object]] = {}
    for entry in cast("list[dict[str, object]]", expected["entries"]):
        key = (
            json.dumps(
                entry["episode_key"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
            cast("int", entry["sample_index"]),
        )
        bindings[key] = entry
        entries[key] = _object(entry["overlay"], "native evidence overlay")
    return NativeEvidence(
        raw_bytes,
        _sha(raw_bytes),
        entries,
        bindings,
        {key: value for key, value in expected.items() if key != "entries"},
    )
