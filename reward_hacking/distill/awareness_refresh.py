"""Versioned awareness sidecars: rescan ladder episodes from their traces without rewriting records.

Records written before the presumed trap or decoy pattern carry an in_episode summary without
presumed_integrity_matches, and the readers refuse them. This module rescans each ladder episode
from its trace, checks that the rescan reproduces the recorded counts apart from the presumed
pattern, and writes the rescan into an immutable sidecar beside the stage. Readers overlay a
sidecar onto records in memory. Binding is by episode identity, and no digest gates a read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.awareness_clears import ARTIFACTS_ROOT, stage_of
from reward_hacking.distill.label_refresh import (
    _code_revision,  # pyright: ignore[reportPrivateUsage]
    derived_label_join_key,
)
from reward_hacking.eval_awareness import PRESUMED_INTEGRITY_PATTERN
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.ladder.runner import eval_awareness_from_trace_records

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

AWARENESS_SCANNER_VERSION = "eval_awareness_v2"
AWARENESS_REFRESH_FORMAT_VERSION = "episode_awareness_refresh_v1"
ROWS_FILENAME = "awareness.jsonl"
MANIFEST_FILENAME = "manifest.json"
ROW_RECORD_TYPE = "episode_awareness_refresh"
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AwarenessRefreshResult:
    """Paths and counts for one refreshed stage."""

    rows_path: Path
    manifest_path: Path
    episode_count: int
    presumed_match_episodes: int
    targeted_alert_episodes: int
    count_change_episodes: int


@dataclass(frozen=True, slots=True)
class AwarenessSidecar:
    """One stage's validated refresh rows, keyed by the episode join key of the records."""

    stage: str
    rows_path: Path
    manifest_path: Path
    code_revision: str
    rows: Mapping[tuple[str, int], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class AwarenessRefresh:
    """The sidecars an explicit flag supplied, one per stage."""

    sidecars_by_stage: Mapping[str, AwarenessSidecar]


NO_AWARENESS_REFRESH = AwarenessRefresh(sidecars_by_stage={})


def versioned_awareness_output_dir(stage_dir: Path) -> Path:
    """Return the current scanner's sidecar directory beside an episode stage."""
    return stage_dir.with_name(f"{stage_dir.name}-awareness-{AWARENESS_SCANNER_VERSION}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ladder_episodes(
    records_path: Path, records: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    episodes = [record for record in records if record.get("record") == "ladder_episode"]
    if not episodes:
        raise ValueError(f"{records_path} has no ladder_episode records")
    join_keys = [derived_label_join_key(episode) for episode in episodes]
    if len(set(join_keys)) != len(join_keys):
        raise ValueError(f"{records_path} has duplicate episode keys")
    identities = [(episode["task_id"], episode["sample_index"]) for episode in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{records_path} has duplicate (task_id, sample_index) identities")
    return episodes


def _trace_path(records_path: Path, episode: Mapping[str, Any]) -> Path:
    trace_path_value = episode["trace_path"]
    if not isinstance(trace_path_value, str):
        raise TypeError(f"{records_path} episode {episode['task_id']} trace_path must be a string")
    trace_path = Path(trace_path_value)
    if not trace_path.is_absolute():
        trace_path = records_path.parent / trace_path
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"{records_path} episode {episode['task_id']} trace is missing: {trace_path}"
        )
    return trace_path


def _require_inline_transcript_matches_trace(
    context: str, episode: Mapping[str, Any], trace_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Check that the thinking kept inline in the record is the thinking the trace holds."""
    assistant_messages = [
        message
        for message in episode["trace_summary"]["messages"]
        if message["role"] == "assistant"
    ]
    turns = sorted(
        (row for row in trace_rows if row.get("record") == "turn"), key=lambda row: row["turn"]
    )
    if len(assistant_messages) < len(turns):
        raise ValueError(f"{context} has fewer inline assistant messages than trace turns")
    for turn, message in zip(turns, assistant_messages, strict=False):
        inline = (message.get("reasoning_content") or "", message.get("content") or "")
        if (turn["reasoning"], turn["content"]) != inline:
            raise ValueError(
                f"{context} turn {turn['turn']} reasoning or content differs from the inline transcript"
            )


def _counts_without_presumed(
    in_episode: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return per-pattern and per-tier counts with the presumed pattern removed from both."""
    presumed_count = len(in_episode.get("presumed_integrity_matches", []))
    patterns = {
        name: count
        for name, count in in_episode["counts_per_pattern"].items()
        if name != PRESUMED_INTEGRITY_PATTERN
    }
    tiers = dict(in_episode["counts_per_tier"])
    tiers["lexical_alert"] -= presumed_count
    return patterns, tiers


def _count_changes(
    recorded_regex: Mapping[str, Any], rescan_regex: Mapping[str, Any]
) -> dict[str, Any]:
    """Return recorded and rescanned counts that differ, ignoring the presumed pattern itself.

    Scanner patterns added after a stage was recorded show up here as new counts. They are kept
    in the manifest for review, because the rescan is what the refresh is meant to measure.
    """
    recorded_patterns, recorded_tiers = _counts_without_presumed(recorded_regex["in_episode"])
    rescan_patterns, rescan_tiers = _counts_without_presumed(rescan_regex["in_episode"])
    pattern_names = sorted(set(recorded_patterns) | set(rescan_patterns))
    tier_names = sorted(set(recorded_tiers) | set(rescan_tiers))
    return {
        "patterns": {
            name: [recorded_patterns.get(name, 0), rescan_patterns.get(name, 0)]
            for name in pattern_names
            if recorded_patterns.get(name, 0) != rescan_patterns.get(name, 0)
        },
        "tiers": {
            name: [recorded_tiers.get(name, 0), rescan_tiers.get(name, 0)]
            for name in tier_names
            if recorded_tiers.get(name, 0) != rescan_tiers.get(name, 0)
        },
    }


def _matched_texts(in_episode: Mapping[str, Any], field_name: str) -> list[str]:
    return [str(match["matched_text"]) for match in in_episode[field_name]]


def refresh_awareness(
    records_path: Path, *, stage_root: Path = ARTIFACTS_ROOT
) -> AwarenessRefreshResult:
    """Rescan every ladder episode of one stage into an immutable sidecar beside the stage."""
    stage = stage_of(records_path, stage_root)
    records = read_jsonl(records_path)
    episodes = _ladder_episodes(records_path, records)
    output_dir = versioned_awareness_output_dir(records_path.parent)
    if output_dir.exists():
        raise FileExistsError(f"awareness sidecar already exists: {output_dir}")

    rows: list[dict[str, Any]] = []
    count_changes: list[dict[str, Any]] = []
    for episode in episodes:
        context = f"{records_path} episode {episode['task_id']} sample {episode['sample_index']}"
        trace_path = _trace_path(records_path, episode)
        trace_bytes = trace_path.read_bytes()
        trace_rows = read_jsonl(trace_path)
        _require_inline_transcript_matches_trace(context, episode, trace_rows)
        rescan_regex = cast(
            "Mapping[str, Any]",
            eval_awareness_from_trace_records(episode["trace_summary"], trace_rows)["regex"],
        )
        changes = _count_changes(episode["eval_awareness"]["regex"], rescan_regex)
        if changes["patterns"] or changes["tiers"]:
            count_changes.append(
                {"task_id": episode["task_id"], "sample_index": episode["sample_index"], **changes}
            )
        rows.append(
            {
                "record": ROW_RECORD_TYPE,
                "key": episode["key"],
                "sample_index": episode["sample_index"],
                "task_id": episode["task_id"],
                "source_record_sha256": _sha256(
                    json.dumps(episode, sort_keys=True, ensure_ascii=False).encode()
                ),
                "trace_sha256": _sha256(trace_bytes),
                "regex": rescan_regex,
            }
        )

    output_dir.mkdir(parents=True)
    rows_path = output_dir / ROWS_FILENAME
    _write_jsonl(rows_path, rows)
    presumed_episodes = [
        {
            "task_id": row["task_id"],
            "sample_index": row["sample_index"],
            "matched_texts": _matched_texts(
                row["regex"]["in_episode"], "presumed_integrity_matches"
            ),
        }
        for row in rows
        if row["regex"]["in_episode"]["presumed_integrity_matches"]
    ]
    targeted_episodes = [
        {
            "task_id": row["task_id"],
            "sample_index": row["sample_index"],
            "matched_texts": _matched_texts(
                row["regex"]["in_episode"], "targeted_integrity_matches"
            ),
        }
        for row in rows
        if row["regex"]["in_episode"]["targeted_integrity_matches"]
    ]
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = {
        "format_version": AWARENESS_REFRESH_FORMAT_VERSION,
        "scanner_version": AWARENESS_SCANNER_VERSION,
        "code_revision": _code_revision(),
        "stage": stage,
        "source_records_file": str(records_path.resolve()),
        "source_records_sha256": _sha256(records_path.read_bytes()),
        "rows_file": ROWS_FILENAME,
        "rows_sha256": _sha256(rows_path.read_bytes()),
        "row_count": len(rows),
        "presumed_match_episodes": presumed_episodes,
        "targeted_alert_episodes": targeted_episodes,
        "recorded_count_changes": count_changes,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AwarenessRefreshResult(
        rows_path=rows_path,
        manifest_path=manifest_path,
        episode_count=len(rows),
        presumed_match_episodes=len(presumed_episodes),
        targeted_alert_episodes=len(targeted_episodes),
        count_change_episodes=len(count_changes),
    )


def _read_sidecar(rows_path: Path) -> AwarenessSidecar:
    manifest_path = rows_path.with_name(MANIFEST_FILENAME)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format_version"] != AWARENESS_REFRESH_FORMAT_VERSION:
        raise ValueError(f"{manifest_path} has an unsupported format version")
    if manifest["scanner_version"] != AWARENESS_SCANNER_VERSION:
        raise ValueError(
            f"{rows_path} was written by scanner {manifest['scanner_version']}; "
            f"this reader needs {AWARENESS_SCANNER_VERSION}"
        )
    rows = read_jsonl(rows_path)
    if len(rows) != manifest["row_count"]:
        raise ValueError(f"{rows_path} row count does not match its manifest")
    rows_by_join_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("record") != ROW_RECORD_TYPE:
            raise ValueError(f"{rows_path} contains a row that is not an awareness refresh row")
        join_key = derived_label_join_key(row)
        if join_key in rows_by_join_key:
            raise ValueError(f"{rows_path} contains a duplicate episode join")
        rows_by_join_key[join_key] = row
    return AwarenessSidecar(
        stage=str(manifest["stage"]),
        rows_path=rows_path,
        manifest_path=manifest_path,
        code_revision=str(manifest["code_revision"]),
        rows=rows_by_join_key,
    )


def load_awareness_refresh(rows_paths: Sequence[Path]) -> AwarenessRefresh:
    """Read explicit sidecars, at most one per stage."""
    sidecars: dict[str, AwarenessSidecar] = {}
    for rows_path in rows_paths:
        sidecar = _read_sidecar(rows_path)
        if sidecar.stage in sidecars:
            raise ValueError(f"more than one awareness sidecar for stage {sidecar.stage}")
        sidecars[sidecar.stage] = sidecar
    return AwarenessRefresh(sidecars_by_stage=sidecars)


def require_sidecars_bound(refresh: AwarenessRefresh, stages: Collection[str]) -> None:
    """Refuse a sidecar whose stage none of the records being read belongs to."""
    unbound = sorted(set(refresh.sidecars_by_stage) - set(stages))
    if unbound:
        raise ValueError(f"awareness sidecars are not bound to any selected records: {unbound}")


def refreshed_records(
    records_path: Path,
    records: Sequence[Mapping[str, Any]],
    refresh: AwarenessRefresh,
    *,
    stage_root: Path = ARTIFACTS_ROOT,
) -> list[Mapping[str, Any]]:
    """Return records with each ladder episode's regex replaced by its sidecar row, all or nothing."""
    if not refresh.sidecars_by_stage:
        return list(records)
    sidecar = refresh.sidecars_by_stage.get(stage_of(records_path, stage_root))
    if sidecar is None:
        return list(records)
    refreshed: list[Mapping[str, Any]] = []
    bound_join_keys: set[tuple[str, int]] = set()
    for record in records:
        if record.get("record") != "ladder_episode":
            refreshed.append(record)
            continue
        join_key = derived_label_join_key(record)
        row = sidecar.rows.get(join_key)
        if row is None:
            raise ValueError(
                f"{records_path} episode {join_key} has no awareness refresh row in {sidecar.rows_path}"
            )
        if row["task_id"] != record["task_id"]:
            raise ValueError(
                f"awareness refresh row for {join_key} names task {row['task_id']}, "
                f"but the record names task {record['task_id']}"
            )
        bound_join_keys.add(join_key)
        refreshed.append(
            {**record, "eval_awareness": {**record["eval_awareness"], "regex": row["regex"]}}
        )
    unmatched = sorted(set(sidecar.rows) - bound_join_keys)
    if unmatched:
        raise ValueError(f"awareness refresh rows match no episode in {records_path}: {unmatched}")
    return refreshed


def awareness_refresh_provenance(refresh: AwarenessRefresh) -> list[dict[str, object]]:
    """Describe each sidecar for a selection report, recorded for audit and never checked."""
    return [
        {
            "stage": sidecar.stage,
            "rows_path": str(sidecar.rows_path),
            "manifest_path": str(sidecar.manifest_path),
            "scanner_version": AWARENESS_SCANNER_VERSION,
            "code_revision": sidecar.code_revision,
            "row_count": len(sidecar.rows),
        }
        for sidecar in refresh.sidecars_by_stage.values()
    ]


def main(argv: Sequence[str] | None = None) -> None:
    """Refresh each named stage's records.jsonl into its versioned sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    for records_path in args.records:
        result = refresh_awareness(records_path)
        logger.info(
            "%s episodes=%d presumed_match_episodes=%d targeted_alert_episodes=%d "
            "count_change_episodes=%d rows=%s",
            records_path,
            result.episode_count,
            result.presumed_match_episodes,
            result.targeted_alert_episodes,
            result.count_change_episodes,
            result.rows_path,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
