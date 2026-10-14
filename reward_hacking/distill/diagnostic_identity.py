"""Bind sampling reproduction checks to the original generation's model and adapter."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.distill.episodes import (
    _trace_path_for_record,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.organism_rl.adapter_identity import adapter_sha256
from reward_hacking.organism_rl.evaluate import resolve_model_revision
from reward_hacking.organism_rl.generate_provenance import (
    PROVENANCE_SCHEMA_COMPATIBLE_VERSIONS,
    _canonical_sha256,  # pyright: ignore[reportPrivateUsage]
    _provenance_identity,  # pyright: ignore[reportPrivateUsage]
    _read_existing_provenance,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.semantic_identity import model_identity, recorded_model_matches

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class StudentGenerationIdentity:
    """Current scorer evidence obtained without loading model weights."""

    model: str
    model_identity: dict[str, object] | None
    model_revision: str | None
    adapter_sha256: str | None


def resolve_student_generation_identity(
    model: str, adapter: Path | None
) -> StudentGenerationIdentity:
    """Read current semantic weights, cached Hub revision, and exact adapter content."""
    revision = resolve_model_revision(Namespace(model=model, model_revision=None))
    return StudentGenerationIdentity(
        model=model,
        model_identity=model_identity(model),
        model_revision=revision,
        adapter_sha256=adapter_sha256(adapter) if adapter is not None else None,
    )


def generation_flags_for_record(
    record: Mapping[str, object], records_path: Path
) -> dict[str, object] | None:
    """Read the nearest original provenance and verify its exact binding to this episode."""
    trace_path = _trace_path_for_record(record, records_path)
    provenance_path = next(
        (
            parent / "provenance.json"
            for parent in trace_path.parents
            if (parent / "provenance.json").is_file()
        ),
        None,
    )
    if provenance_path is None:
        return None
    document = _read_existing_provenance(provenance_path)
    provenance = cast("Mapping[str, object]", document["provenance"])
    schema = document["schema_version"]
    if schema == 1:
        accepted_hashes = {_canonical_sha256(provenance)}
    elif schema in PROVENANCE_SCHEMA_COMPATIBLE_VERSIONS:
        accepted_hashes = {
            _canonical_sha256(_provenance_identity(provenance)),
            _canonical_sha256(_provenance_identity(provenance, include_legacy_leak_metadata=True)),
            _canonical_sha256(
                _provenance_identity(
                    provenance, include_code_provenance=True, include_legacy_leak_metadata=True
                )
            ),
        }
    else:
        raise ValueError(f"unsupported generation provenance schema {schema!r}")
    digest = document["provenance_sha256"]
    if digest not in accepted_hashes:
        return None
    if record.get("run_provenance_sha256") != digest:
        return None
    flags = provenance.get("flags")
    if not isinstance(flags, Mapping):
        raise TypeError("original generation provenance flags must be an object")
    return dict(flags)


def _adapter_eligibility(
    flags: Mapping[str, object] | None,
    current: StudentGenerationIdentity,
) -> tuple[bool | None, str]:
    if flags is None:
        return None, "original generation provenance is unavailable; adapter identity is unknown"
    adapter_digest = flags.get("served_adapter_sha256")
    if adapter_digest is not None:
        if adapter_digest != current.adapter_sha256:
            return False, "recorded adapter content differs from student scorer"
        return True, "matching model and served adapter content in original unhinted context"
    if "served_model_id" not in flags or flags["served_model_id"] not in {None, flags.get("model")}:
        return None, "served adapter identity is unavailable"
    if current.adapter_sha256 is not None:
        return False, "student scorer uses an adapter; recorded generation was base"
    return True, "matching base model in original unhinted context"


def _legacy_model_eligibility(
    flags: Mapping[str, object], current: StudentGenerationIdentity
) -> tuple[bool | None, str]:
    recorded_model = flags.get("model")
    if not isinstance(recorded_model, str) or not recorded_model:
        return None, "recorded model identity unavailable"
    if recorded_model != current.model:
        return False, "recorded model differs from student scorer"
    revision = flags.get("model_revision")
    if not isinstance(revision, str) or not revision or current.model_revision is None:
        return None, "recorded or current model revision unavailable"
    if revision != current.model_revision:
        return False, "recorded model revision differs from current cached model revision"
    return True, "matching original model and revision"


def _semantic_model_eligibility(
    row: Mapping[str, Any], flags: Mapping[str, object] | None, current: StudentGenerationIdentity
) -> tuple[bool | None, str]:
    recorded_identity = row["baseline_model_identity"]
    if not recorded_model_matches(
        {"model": row.get("baseline_model"), "model_identity": recorded_identity},
        current.model,
        identity=current.model_identity,
    ):
        return False, "recorded weights differ from student scorer"
    if isinstance(recorded_identity, Mapping) and "reference" in recorded_identity:
        revision = row.get("baseline_model_revision")
        if revision is None and flags is not None:
            revision = flags.get("model_revision")
        if not isinstance(revision, str) or not revision or current.model_revision is None:
            return None, "recorded or current model revision unavailable"
        if revision != current.model_revision:
            return False, "recorded model revision differs from current cached model revision"
    return True, "matching semantic model identity"


def baseline_reproduction_eligibility(
    row: Mapping[str, Any], current: StudentGenerationIdentity
) -> tuple[bool | None, str]:
    """Determine whether sampled logprobs should reproduce under the current scorer."""
    flags = row.get("baseline_generation_flags")
    if flags is not None and not isinstance(flags, Mapping):
        raise TypeError("baseline generation flags must be an object")
    hint = flags.get("hint") if flags is not None else None
    if row.get("sampling_sanity_applicable") is False or (
        isinstance(hint, Mapping) and hint.get("strength") not in {None, "none"}
    ):
        return False, "original hint or template differs"
    recorded_identity = row.get("baseline_model_identity")
    recorded_model = row.get("baseline_model")
    if recorded_identity is not None:
        eligible, reason = _semantic_model_eligibility(row, flags, current)
        if eligible is not True:
            return eligible, reason
    elif flags is not None:
        recorded_model = flags.get("model")
        eligible, reason = _legacy_model_eligibility(flags, current)
        if eligible is not True:
            return eligible, reason
    else:
        if recorded_model is not None and recorded_model != current.model:
            return False, "recorded model differs from student scorer"
        reason = (
            "recorded model identity unavailable"
            if recorded_model is None
            else "recorded weight identity unavailable"
        )
        return None, reason
    return _adapter_eligibility(flags, current)
