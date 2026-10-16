"""Reviewer clears for presumed trap or decoy awareness matches.

A presumed match is cleared only when a human reviewer has listed every matched text of that
episode in a clears file. Targeted integrity alerts are never cleared here. Clears bind to the
episode identity (stage, task_id, sample_index) and to matched text, not to file or content hashes.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
CLEARS_SCHEMA_VERSION = 1
CLEAR_FIELDS = frozenset(
    {"stage", "task_id", "sample_index", "matched_texts", "reviewer_note", "date"}
)

EpisodeIdentity = tuple[str, str, int]


@dataclass(frozen=True, slots=True)
class AwarenessClear:
    """One reviewer decision that every presumed match of one episode is unrelated to the test."""

    stage: str
    task_id: str
    sample_index: int
    matched_texts: tuple[str, ...]
    reviewer_note: str
    date: str

    @property
    def identity(self) -> EpisodeIdentity:
        """Return the episode identity this clear binds to."""
        return (self.stage, self.task_id, self.sample_index)

    def to_json_dict(self) -> dict[str, object]:
        """Return the clear in the form stored in clears files and selection manifests."""
        return {
            "stage": self.stage,
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "matched_texts": list(self.matched_texts),
            "reviewer_note": self.reviewer_note,
            "date": self.date,
        }


@dataclass(frozen=True, slots=True)
class IntegrityAssessment:
    """Presumed and targeted integrity matches of one episode, after applying cleared texts."""

    presumed_matched_texts: tuple[str, ...]
    uncovered_matched_texts: tuple[str, ...]
    targeted_alert_count: int

    @property
    def alert_stands(self) -> bool:
        """Return whether an integrity alert survives the clear."""
        return self.targeted_alert_count > 0 or bool(self.uncovered_matched_texts)

    @property
    def cleared(self) -> bool:
        """Return whether every presumed match is covered and no targeted alert exists."""
        return bool(self.presumed_matched_texts) and not self.alert_stands


def load_awareness_clears(path: Path) -> dict[EpisodeIdentity, AwarenessClear]:
    """Read the clears file, rejecting malformed entries and repeated identities."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "clears"}:
        raise ValueError("awareness clears file must contain exactly schema_version and clears")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != CLEARS_SCHEMA_VERSION
    ):
        raise ValueError(f"awareness clears schema_version must be {CLEARS_SCHEMA_VERSION}")
    entries = payload["clears"]
    if not isinstance(entries, list):
        raise TypeError("awareness clears must be a list")
    clears: dict[EpisodeIdentity, AwarenessClear] = {}
    for entry in entries:
        clear = parse_awareness_clear(entry)
        if clear.identity in clears:
            raise ValueError(f"duplicate awareness clear for identity {clear.identity}")
        clears[clear.identity] = clear
    return clears


def parse_awareness_clear(entry: object) -> AwarenessClear:
    """Validate one clears-file entry or one clear recorded in a selection manifest."""
    if not isinstance(entry, dict):
        raise TypeError("awareness clear must be an object")
    fields = cast("dict[str, Any]", entry)
    if frozenset(fields) != CLEAR_FIELDS:
        raise ValueError(f"awareness clear must contain exactly {sorted(CLEAR_FIELDS)}")
    texts = fields["matched_texts"]
    if not isinstance(texts, list) or not texts or any(not _nonempty_str(t) for t in texts):
        raise TypeError("awareness clear matched_texts must be a nonempty list of strings")
    sample_index = fields["sample_index"]
    if type(sample_index) is not int or sample_index < 0:
        raise TypeError("awareness clear sample_index must be a nonnegative integer")
    for name in ("stage", "task_id", "reviewer_note"):
        if not _nonempty_str(fields[name]):
            raise TypeError(f"awareness clear {name} must be a nonempty string")
    if not _nonempty_str(fields["date"]):
        raise TypeError("awareness clear date must be an ISO date string")
    date.fromisoformat(fields["date"])
    return AwarenessClear(
        stage=fields["stage"],
        task_id=fields["task_id"],
        sample_index=sample_index,
        matched_texts=tuple(texts),
        reviewer_note=fields["reviewer_note"],
        date=fields["date"],
    )


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def stage_of(records_path: Path, stage_root: Path = ARTIFACTS_ROOT) -> str:
    """Return the records file's stage directory relative to the stage root."""
    resolved_root = stage_root.resolve()
    resolved_records = records_path.resolve()
    if not resolved_records.is_relative_to(resolved_root):
        raise ValueError(f"records file {records_path} is outside the stage root {resolved_root}")
    return resolved_records.parent.relative_to(resolved_root).as_posix()


def episode_identity(
    records_path: Path, record: Mapping[str, object], stage_root: Path = ARTIFACTS_ROOT
) -> EpisodeIdentity:
    """Identify an episode by its stage directory under the stage root, task, and sample."""
    stage = stage_of(records_path, stage_root)
    task_id = record.get("task_id")
    sample_index = record.get("sample_index")
    if not isinstance(task_id, str) or not task_id:
        raise TypeError("episode record task_id must be a nonempty string")
    if type(sample_index) is not int or sample_index < 0:
        raise TypeError("episode record sample_index must be a nonnegative integer")
    return (stage, task_id, sample_index)


def assess_integrity_alert(
    in_episode: Mapping[str, object], cleared_matched_texts: Collection[str]
) -> IntegrityAssessment:
    """Read one episode's in-episode match lists and report what still stands after clears."""
    presumed = _match_texts(in_episode, "presumed_integrity_matches")
    targeted = _match_texts(in_episode, "targeted_integrity_matches")
    cleared = set(cleared_matched_texts)
    return IntegrityAssessment(
        presumed_matched_texts=tuple(presumed),
        uncovered_matched_texts=tuple(text for text in presumed if text not in cleared),
        targeted_alert_count=len(targeted),
    )


def _match_texts(in_episode: Mapping[str, object], field_name: str) -> list[str]:
    matches = in_episode.get(field_name)
    if not isinstance(matches, list):
        raise TypeError(
            f"eval_awareness in_episode lacks {field_name}; rescore the run before using it"
        )
    texts: list[str] = []
    for match in matches:
        if not isinstance(match, Mapping) or not _nonempty_str(match.get("matched_text")):
            raise TypeError(f"eval_awareness in_episode {field_name} entries need matched_text")
        texts.append(str(match["matched_text"]))
    return texts
