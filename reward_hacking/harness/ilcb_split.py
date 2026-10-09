"""Stable problem-id partitions for multi-turn ILCB episodes.

The manifest sits beside the gitignored case data and contains ids only. Membership is assigned by
SHA-256, so input row order and the selected ILCB variant cannot change a problem's partition.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import cast

from reward_hacking.ilcb_data import DATA_PATH, load_problems
from reward_hacking.ilcb_split import (
    HELD_OUT_FRACTION,
    SPLIT_ALGORITHM,
    ILCBPartition,
    deterministic_problem_split,
)

MANIFEST_PATH = DATA_PATH.parent / "ilcb_split_manifest.json"
MANIFEST_SCHEMA = 1

logger = logging.getLogger(__name__)

__all__ = [
    "HELD_OUT_FRACTION",
    "MANIFEST_PATH",
    "MANIFEST_SCHEMA",
    "SPLIT_ALGORITHM",
    "ILCBPartition",
    "create_data_manifest",
    "deterministic_problem_split",
    "load_or_create_split_manifest",
]


def load_or_create_split_manifest(
    problem_ids: tuple[str, ...], manifest_path: Path = MANIFEST_PATH
) -> dict[ILCBPartition, tuple[str, ...]]:
    """Load the split manifest or create it once; refuse stale, malformed, or mismatched manifests."""
    expected = deterministic_problem_split(problem_ids)
    if manifest_path.exists():
        recorded = _read_manifest(manifest_path)
        if recorded != expected:
            raise ValueError(
                "ILCB split manifest does not match the current problem ids and algorithm"
            )
        return recorded

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "algorithm": SPLIT_ALGORITHM,
        "train_ids": list(expected["train"]),
        "held_out_ids": list(expected["held-out"]),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as manifest_file:
            temporary_path = Path(manifest_file.name)
            manifest_file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(manifest_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return expected


def _read_manifest(manifest_path: Path) -> dict[ILCBPartition, tuple[str, ...]]:
    """Read and validate the id-only manifest against its schema and algorithm."""
    try:
        raw_payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read the ILCB split manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(raw_payload, dict):
        raise TypeError("ILCB split manifest must be a JSON object")
    payload = cast("dict[str, object]", raw_payload)
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("algorithm") != SPLIT_ALGORITHM:
        raise ValueError("ILCB split manifest has an unsupported schema or algorithm")
    train_ids = _manifest_ids(payload.get("train_ids"), "train_ids")
    held_out_ids = _manifest_ids(payload.get("held_out_ids"), "held_out_ids")
    if set(train_ids) & set(held_out_ids):
        raise ValueError("ILCB train and held-out ids overlap")
    return {"train": train_ids, "held-out": held_out_ids}


def _manifest_ids(value: object, field_name: str) -> tuple[str, ...]:
    """Validate one canonical, sorted list of problem ids from the manifest."""
    if not isinstance(value, list) or any(not isinstance(problem_id, str) for problem_id in value):
        raise ValueError(f"ILCB split manifest {field_name} must be a list of strings")
    problem_ids = tuple(cast("list[str]", value))
    if tuple(sorted(set(problem_ids))) != problem_ids:
        raise ValueError(f"ILCB split manifest {field_name} must contain unique sorted ids")
    return problem_ids


def create_data_manifest() -> dict[ILCBPartition, tuple[str, ...]]:
    """Create or validate the local manifest using the current baked data file."""
    problem_ids = tuple(sorted({problem.task_id for problem in load_problems()}))
    return load_or_create_split_manifest(problem_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    split = create_data_manifest()
    logger.info("train=%d held-out=%d", len(split["train"]), len(split["held-out"]))
