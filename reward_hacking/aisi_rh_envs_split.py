"""Deterministic train and held-out partitions for AISI rh_envs problems."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

AisiPartition = Literal["train", "held-out"]
HELD_OUT_FRACTION = 0.2
SPLIT_ALGORITHM = "sha256(aisi-rh-envs-split-v1\\0 + source + \\0 + problem_id)"
_HASH_PREFIX = b"aisi-rh-envs-split-v1\0"
MINIMUM_PARTITIONABLE_PROBLEMS = 2


def deterministic_source_splits(
    problem_ids_by_source: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[AisiPartition, tuple[str, ...]]]:
    """Assign about one fifth of each source's unique problem ids to held-out."""
    splits: dict[str, dict[AisiPartition, tuple[str, ...]]] = {}
    for source, problem_ids in sorted(problem_ids_by_source.items()):
        if not source:
            raise ValueError("AISI dataset names must be non-empty")
        if len(problem_ids) != len(set(problem_ids)):
            raise ValueError(f"AISI {source} problem ids must be unique before splitting")
        if any(not problem_id for problem_id in problem_ids):
            raise ValueError(f"AISI {source} problem ids must be non-empty")

        if len(problem_ids) < MINIMUM_PARTITIONABLE_PROBLEMS:
            held_out_ids: frozenset[str] = frozenset()
        else:
            held_out_count = min(
                len(problem_ids) - 1,
                max(1, round(len(problem_ids) * HELD_OUT_FRACTION)),
            )
            ranked_ids = sorted(
                problem_ids,
                key=lambda problem_id: (
                    hashlib.sha256(
                        _HASH_PREFIX + source.encode("utf-8") + b"\0" + problem_id.encode("utf-8")
                    ).digest(),
                    problem_id,
                ),
            )
            held_out_ids = frozenset(ranked_ids[:held_out_count])

        splits[source] = {
            "train": tuple(
                sorted(problem_id for problem_id in problem_ids if problem_id not in held_out_ids)
            ),
            "held-out": tuple(sorted(held_out_ids)),
        }
    return splits
