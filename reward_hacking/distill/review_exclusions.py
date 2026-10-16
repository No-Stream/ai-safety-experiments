"""Hand-review exclusions for SFT selection.

A reviewer who reads an admitted episode and finds it ineligible (for example a test repair the
labeler missed, or a held-out ladder episode) records it here with a reason. Exclusions bind to the
same episode identity as awareness clears (stage, task_id, sample_index), never to file hashes.
Selection refuses an exclusion that names no episode in its inputs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.awareness_clears import EpisodeIdentity

REVIEW_EXCLUSIONS_SCHEMA_VERSION = 1
REVIEW_EXCLUSION_FIELDS = frozenset(
    {"stage", "task_id", "sample_index", "reason", "source", "date"}
)


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_review_exclusion(entry: object) -> tuple[EpisodeIdentity, dict[str, object]]:
    """Validate one exclusion and return its identity with the entry as recorded in manifests."""
    if not isinstance(entry, dict):
        raise TypeError("review exclusion must be an object")
    fields = cast("dict[str, Any]", entry)
    if frozenset(fields) != REVIEW_EXCLUSION_FIELDS:
        raise ValueError(f"review exclusion must contain exactly {sorted(REVIEW_EXCLUSION_FIELDS)}")
    for name in ("stage", "task_id", "reason", "source", "date"):
        if not _nonempty_text(fields[name]):
            raise ValueError(f"review exclusion {name} must be nonempty text")
    sample_index = fields["sample_index"]
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("review exclusion sample_index must be a nonnegative integer")
    return (fields["stage"], fields["task_id"], sample_index), dict(fields)


def load_review_exclusions(path: Path) -> dict[EpisodeIdentity, dict[str, object]]:
    """Read an exclusions file, rejecting malformed entries and repeated identities."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "exclusions"}:
        raise ValueError(
            "review exclusions file must contain exactly schema_version and exclusions"
        )
    if payload["schema_version"] != REVIEW_EXCLUSIONS_SCHEMA_VERSION:
        raise ValueError(
            f"review exclusions schema_version must be {REVIEW_EXCLUSIONS_SCHEMA_VERSION}"
        )
    entries = payload["exclusions"]
    if not isinstance(entries, list):
        raise TypeError("review exclusions must be a list")
    exclusions: dict[EpisodeIdentity, dict[str, object]] = {}
    for entry in cast("list[object]", entries):
        identity, exclusion = parse_review_exclusion(entry)
        if identity in exclusions:
            raise ValueError(f"duplicate review exclusion for identity {identity}")
        exclusions[identity] = exclusion
    return exclusions
