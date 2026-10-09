"""Pure stable item partitioning for ILCB, shared without importing the task registry."""

from __future__ import annotations

import hashlib
from typing import Literal

ILCBPartition = Literal["train", "held-out"]
HELD_OUT_FRACTION = 0.22
SPLIT_ALGORITHM = "sha256(ilcb-agentic-split-v1\\0 + problem_id)"
_HASH_PREFIX = b"ilcb-agentic-split-v1\0"
MINIMUM_PROBLEM_IDS = 2


def deterministic_problem_split(
    problem_ids: tuple[str, ...],
) -> dict[ILCBPartition, tuple[str, ...]]:
    """Assign IDs to stable train and held-out partitions, preserving the current 80/23 split."""
    if len(problem_ids) < MINIMUM_PROBLEM_IDS:
        raise ValueError("at least two unique problem ids are required to create a split")
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("problem ids must be unique before splitting")

    held_out_count = min(len(problem_ids) - 1, max(1, round(len(problem_ids) * HELD_OUT_FRACTION)))
    ranked_ids = sorted(
        problem_ids,
        key=lambda problem_id: (
            hashlib.sha256(_HASH_PREFIX + problem_id.encode("utf-8")).digest(),
            problem_id,
        ),
    )
    held_out_ids = frozenset(ranked_ids[:held_out_count])
    return {
        "train": tuple(
            sorted(problem_id for problem_id in problem_ids if problem_id not in held_out_ids)
        ),
        "held-out": tuple(sorted(held_out_ids)),
    }
