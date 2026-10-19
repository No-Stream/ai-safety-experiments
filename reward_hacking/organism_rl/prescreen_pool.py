"""Derive a deterministic organism-RL training pool from pre-screen samples."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)
FAMILY_VARIANT_PARTS: int = 2


class PromptCountDict(TypedDict):
    """JSON representation of one prompt's aggregate sample counts."""

    scorable_samples: int
    rewarded_samples: int
    all_one: bool


class PoolCountsDict(TypedDict):
    """JSON representation of threshold and all-one pool counts."""

    screened_in: int
    screened_out: int
    unknown: int
    all_one: int


class PrescreenSummaryDict(TypedDict):
    """JSON representation persisted for a completed pre-screen."""

    min_rewarded_samples: int
    per_prompt: dict[str, PromptCountDict]
    pool_counts: PoolCountsDict
    by_family_variant: dict[str, PoolCountsDict]


class PromptPoolProvenanceDict(TypedDict):
    """JSON representation of the pool selected for training."""

    max_pass_fraction: float
    screened_in: int
    screened_out: int
    dropped: int
    dropped_ids: list[str]
    sampled_out: int
    unknown: int
    absent: int
    ignored: int
    all_one: int
    selected: int
    by_family_variant: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class PromptSampleCounts:
    """Scorable and rewarded sample counts for one task identity."""

    scorable_samples: int
    rewarded_samples: int
    all_one: bool


@dataclass(frozen=True, slots=True)
class PromptPoolDerivation:
    """Threshold classification for one current task selection."""

    current_task_ids: tuple[str, ...]
    screened_in_ids: tuple[str, ...]
    screened_out_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    absent_ids: tuple[str, ...]
    ignored_ids: tuple[str, ...]
    prompt_counts: Mapping[str, PromptSampleCounts]
    record_metadata: Mapping[str, tuple[str, str]]
    min_rewarded_samples: int
    max_pass_fraction: float


@dataclass(frozen=True, slots=True)
class PromptPoolSelection:
    """Final pool with the sampled screened-out prompts called out."""

    task_ids: tuple[str, ...]
    sampled_out_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrescreenRecordSummary:
    """JSON-ready counts for a completed or partial pre-screen."""

    prompt_counts: Mapping[str, PromptSampleCounts]
    pool_counts: PoolCountsDict
    by_family_variant: Mapping[str, PoolCountsDict]
    min_rewarded_samples: int = 1

    def to_dict(self) -> PrescreenSummaryDict:
        """Return counts using only JSON-compatible values."""
        return {
            "min_rewarded_samples": self.min_rewarded_samples,
            "per_prompt": {
                task_id: {
                    "scorable_samples": counts.scorable_samples,
                    "rewarded_samples": counts.rewarded_samples,
                    "all_one": counts.all_one,
                }
                for task_id, counts in sorted(self.prompt_counts.items())
            },
            "pool_counts": {
                "screened_in": self.pool_counts["screened_in"],
                "screened_out": self.pool_counts["screened_out"],
                "unknown": self.pool_counts["unknown"],
                "all_one": self.pool_counts["all_one"],
            },
            "by_family_variant": {
                stratum: {
                    "screened_in": counts["screened_in"],
                    "screened_out": counts["screened_out"],
                    "unknown": counts["unknown"],
                    "all_one": counts["all_one"],
                }
                for stratum, counts in sorted(self.by_family_variant.items())
            },
        }


@dataclass(slots=True)
class _MutablePromptCounts:
    scorable_samples: int = 0
    rewarded_samples: int = 0
    all_one: bool = True

    def freeze(self) -> PromptSampleCounts:
        return PromptSampleCounts(
            scorable_samples=self.scorable_samples,
            rewarded_samples=self.rewarded_samples,
            all_one=self.scorable_samples > 0 and self.all_one,
        )


def read_prescreen_records(path: Path) -> tuple[dict[str, object], ...]:
    """Read JSONL records, dropping only a malformed final line without a newline."""
    if not path.is_file():
        raise FileNotFoundError(f"pre-screen samples file does not exist: {path}")
    records: list[dict[str, object]] = []
    with path.open("rb") as samples_file:
        for line_number, raw_line in enumerate(samples_file, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank pre-screen record at {path}:{line_number}")
            line_terminated = raw_line.endswith((b"\n", b"\r"))
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                if not line_terminated:
                    logger.warning(
                        "dropping torn final pre-screen record at %s:%d", path, line_number
                    )
                    break
                raise ValueError(
                    f"invalid UTF-8 in pre-screen samples at {path}:{line_number}"
                ) from error
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if not line_terminated:
                    logger.warning(
                        "dropping torn final pre-screen record at %s:%d", path, line_number
                    )
                    break
                raise ValueError(
                    f"malformed JSON in pre-screen samples at {path}:{line_number}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"pre-screen record at {path}:{line_number} must be a JSON object")
            records.append(cast("dict[str, object]", record))
    return tuple(records)


def derive_prompt_pool(
    prescreen_dir: Path,
    task_ids: Sequence[str],
    min_rewarded_samples: int = 1,
    max_pass_fraction: float = 1.0,
) -> PromptPoolDerivation:
    """Classify current prompts, retaining unknowns and dropping prompts that are too easy."""
    _validate_min_rewarded_samples(min_rewarded_samples)
    _validate_max_pass_fraction(max_pass_fraction)
    current_task_ids = _unique_task_ids(task_ids, "current task ids")
    records = read_prescreen_records(prescreen_dir / "samples.jsonl")
    sample_counts, record_metadata = _aggregate_samples(records)
    current_ids = set(current_task_ids)
    present_ids = set(record_metadata)
    ignored_ids = tuple(sorted(present_ids - current_ids))
    absent_ids = tuple(sorted(current_ids - present_ids))

    prompt_counts = {
        task_id: sample_counts.get(
            task_id,
            PromptSampleCounts(scorable_samples=0, rewarded_samples=0, all_one=False),
        )
        for task_id in current_task_ids
    }
    unknown_ids = tuple(
        task_id for task_id, counts in prompt_counts.items() if counts.scorable_samples == 0
    )
    threshold_pass_ids = tuple(
        task_id
        for task_id, counts in prompt_counts.items()
        if counts.scorable_samples == 0 or counts.rewarded_samples >= min_rewarded_samples
    )
    dropped_ids = tuple(
        task_id
        for task_id in threshold_pass_ids
        if prompt_counts[task_id].scorable_samples > 0
        and prompt_counts[task_id].rewarded_samples / prompt_counts[task_id].scorable_samples
        > max_pass_fraction
    )
    dropped_id_set = set(dropped_ids)
    screened_in_ids = tuple(
        task_id for task_id in threshold_pass_ids if task_id not in dropped_id_set
    )
    screened_out_ids = tuple(
        task_id
        for task_id, counts in prompt_counts.items()
        if counts.scorable_samples > 0 and counts.rewarded_samples < min_rewarded_samples
    )
    return PromptPoolDerivation(
        current_task_ids=current_task_ids,
        screened_in_ids=screened_in_ids,
        screened_out_ids=screened_out_ids,
        dropped_ids=dropped_ids,
        unknown_ids=unknown_ids,
        absent_ids=absent_ids,
        ignored_ids=ignored_ids,
        prompt_counts=prompt_counts,
        record_metadata=record_metadata,
        min_rewarded_samples=min_rewarded_samples,
        max_pass_fraction=max_pass_fraction,
    )


def select_prompt_pool(
    derivation: PromptPoolDerivation,
    *,
    seed: int,
    unscreened_fraction: float = 0.10,
) -> PromptPoolSelection:
    """Add an identity-seeded sample of screened-out prompts to the screened-in pool."""
    if type(seed) is not int:
        raise TypeError("prompt pool seed must be an integer")
    if not math.isfinite(unscreened_fraction) or not 0 <= unscreened_fraction < 1:
        raise ValueError("unscreened_fraction must be finite and in [0, 1)")
    sample_count = min(
        len(derivation.screened_out_ids),
        round(unscreened_fraction / (1 - unscreened_fraction) * len(derivation.screened_in_ids)),
    )
    sampled_out_ids = tuple(
        sorted(
            derivation.screened_out_ids,
            key=lambda task_id: _task_sample_key(seed, task_id),
        )[:sample_count]
    )
    return PromptPoolSelection(
        task_ids=tuple(sorted((*derivation.screened_in_ids, *sampled_out_ids))),
        sampled_out_ids=sampled_out_ids,
    )


def prompt_pool_provenance(
    derivation: PromptPoolDerivation,
    selection: PromptPoolSelection,
    task_metadata: Mapping[str, tuple[str, str]],
) -> PromptPoolProvenanceDict:
    """Summarize pool selection totals and counts by family and variant."""
    metadata = _validate_prompt_metadata(task_metadata)
    if set(metadata) != set(derivation.current_task_ids):
        raise ValueError("task prompt metadata must match the current task selection")
    if not set(selection.sampled_out_ids).issubset(derivation.screened_out_ids):
        raise ValueError("sampled-out task ids must be screened out")
    expected_task_ids = tuple(sorted((*derivation.screened_in_ids, *selection.sampled_out_ids)))
    if selection.task_ids != expected_task_ids:
        raise ValueError("selected task ids do not match the derivation and sampled-out ids")

    categories = {
        "screened_in": set(derivation.screened_in_ids),
        "screened_out": set(derivation.screened_out_ids),
        "dropped": set(derivation.dropped_ids),
        "sampled_out": set(selection.sampled_out_ids),
        "unknown": set(derivation.unknown_ids),
        "absent": set(derivation.absent_ids),
        "ignored": set(derivation.ignored_ids),
        "all_one": {
            task_id for task_id, counts in derivation.prompt_counts.items() if counts.all_one
        },
        "selected": set(selection.task_ids),
    }
    family_variant_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: dict.fromkeys(categories, 0)
    )
    all_metadata = {**derivation.record_metadata, **metadata}
    for category_name, task_id_set in categories.items():
        for task_id in task_id_set:
            family, variant = all_metadata[task_id]
            family_variant_counts[f"{family}/{variant}"][category_name] += 1
    return cast(
        "PromptPoolProvenanceDict",
        {
            "max_pass_fraction": derivation.max_pass_fraction,
            **{
                category_name: len(task_id_set) for category_name, task_id_set in categories.items()
            },
            "dropped_ids": list(derivation.dropped_ids),
            "by_family_variant": dict(sorted(family_variant_counts.items())),
        },
    )


def summarize_prescreen_records(
    records: Sequence[Mapping[str, object]],
    prompt_metadata: Mapping[str, tuple[str, str]] | None = None,
) -> PrescreenRecordSummary:
    """Summarize records using the default one-rewarded-sample threshold."""
    sample_counts, record_metadata = _aggregate_samples(records)
    metadata = _validate_prompt_metadata(prompt_metadata or {})
    for task_id, stratum in record_metadata.items():
        if task_id in metadata and metadata[task_id] != stratum:
            raise ValueError(
                f"family or variant differs from selected task metadata for {task_id!r}"
            )
    all_metadata = {**record_metadata, **metadata}
    counts = {
        task_id: sample_counts.get(
            task_id,
            PromptSampleCounts(scorable_samples=0, rewarded_samples=0, all_one=False),
        )
        for task_id in sorted(all_metadata)
    }
    return PrescreenRecordSummary(
        prompt_counts=counts,
        pool_counts=_pool_counts(counts.values(), min_rewarded_samples=1),
        by_family_variant=_pool_counts_by_stratum(counts, all_metadata, min_rewarded_samples=1),
    )


def _aggregate_samples(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, PromptSampleCounts], dict[str, tuple[str, str]]]:
    mutable_counts: dict[str, _MutablePromptCounts] = defaultdict(_MutablePromptCounts)
    record_metadata: dict[str, tuple[str, str]] = {}
    seen_samples: set[tuple[str, int]] = set()
    for index, record in enumerate(records, start=1):
        task_id, replica, reward, infrastructure_failure, family, variant = _validate_sample(
            record, index
        )
        sample_identity = (task_id, replica)
        if sample_identity in seen_samples:
            raise ValueError(f"duplicate pre-screen sample for {task_id!r} replica {replica}")
        seen_samples.add(sample_identity)
        stratum = (family, variant)
        previous_stratum = record_metadata.setdefault(task_id, stratum)
        if previous_stratum != stratum:
            raise ValueError(f"family or variant changes across pre-screen samples for {task_id!r}")
        counts = mutable_counts[task_id]
        if reward is None or infrastructure_failure is not None:
            continue
        counts.scorable_samples += 1
        if reward > 0:
            counts.rewarded_samples += 1
        if reward != 1:
            counts.all_one = False
    return (
        {task_id: counts.freeze() for task_id, counts in mutable_counts.items()},
        record_metadata,
    )


def _validate_sample(
    record: Mapping[str, object],
    record_number: int,
) -> tuple[str, int, int | float | None, str | None, str, str]:
    task_id = _nonempty_string(record, "task_id", record_number)
    replica = record.get("replica")
    if type(replica) is not int or replica < 0:
        raise ValueError(f"pre-screen record {record_number} has invalid replica")
    if "reward" not in record:
        raise ValueError(f"pre-screen record {record_number} has no reward field")
    reward = record["reward"]
    if reward is not None and (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        raise ValueError(f"pre-screen record {record_number} has invalid reward")
    if "infrastructure_failure" not in record:
        raise ValueError(f"pre-screen record {record_number} has no infrastructure_failure field")
    infrastructure_failure = record["infrastructure_failure"]
    if infrastructure_failure is not None and (
        not isinstance(infrastructure_failure, str) or not infrastructure_failure
    ):
        raise ValueError(f"pre-screen record {record_number} has invalid infrastructure_failure")
    family = _nonempty_string(record, "family", record_number)
    variant = _nonempty_string(record, "variant", record_number)
    return (
        task_id,
        replica,
        reward,
        infrastructure_failure,
        family,
        variant,
    )


def _nonempty_string(record: Mapping[str, object], key: str, record_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"pre-screen record {record_number} has invalid {key}")
    return value


def _unique_task_ids(task_ids: Sequence[str], name: str) -> tuple[str, ...]:
    if any(type(task_id) is not str or not task_id for task_id in task_ids):
        raise ValueError(f"{name} must be nonempty strings")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{name} must be unique")
    return tuple(sorted(task_ids))


def _validate_min_rewarded_samples(min_rewarded_samples: int) -> None:
    if type(min_rewarded_samples) is not int or min_rewarded_samples < 1:
        raise ValueError("min_rewarded_samples must be a positive integer")


def _validate_max_pass_fraction(max_pass_fraction: float) -> None:
    if (
        isinstance(max_pass_fraction, bool)
        or not math.isfinite(max_pass_fraction)
        or not 0 <= max_pass_fraction <= 1
    ):
        raise ValueError("max_pass_fraction must be finite and in [0, 1]")


def _task_sample_key(seed: int, task_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).digest()


def _validate_prompt_metadata(
    prompt_metadata: Mapping[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    metadata: dict[str, tuple[str, str]] = {}
    for task_id, value in prompt_metadata.items():
        if type(task_id) is not str or not task_id:
            raise ValueError("prompt metadata task ids must be nonempty strings")
        if (
            type(value) is not tuple
            or len(value) != FAMILY_VARIANT_PARTS
            or any(type(part) is not str or not part for part in value)
        ):
            raise ValueError(f"prompt metadata for {task_id!r} must be a (family, variant) tuple")
        metadata[task_id] = value
    return metadata


def _pool_counts(
    prompt_counts: Iterable[PromptSampleCounts],
    *,
    min_rewarded_samples: int,
) -> PoolCountsDict:
    counts = tuple(prompt_counts)
    screened_in = sum(
        prompt.scorable_samples == 0 or prompt.rewarded_samples >= min_rewarded_samples
        for prompt in counts
    )
    screened_out = sum(
        prompt.scorable_samples > 0 and prompt.rewarded_samples < min_rewarded_samples
        for prompt in counts
    )
    return {
        "screened_in": screened_in,
        "screened_out": screened_out,
        "unknown": sum(prompt.scorable_samples == 0 for prompt in counts),
        "all_one": sum(prompt.all_one for prompt in counts),
    }


def _pool_counts_by_stratum(
    prompt_counts: Mapping[str, PromptSampleCounts],
    metadata: Mapping[str, tuple[str, str]],
    *,
    min_rewarded_samples: int,
) -> dict[str, PoolCountsDict]:
    counts_by_stratum: dict[str, list[PromptSampleCounts]] = defaultdict(list)
    for task_id, counts in prompt_counts.items():
        family, variant = metadata[task_id]
        counts_by_stratum[f"{family}/{variant}"].append(counts)
    return {
        stratum: _pool_counts(counts, min_rewarded_samples=min_rewarded_samples)
        for stratum, counts in sorted(counts_by_stratum.items())
    }
