"""Generation provenance binds reproduction eligibility to actual model and adapter content."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.distill import diagnostic_identity
from reward_hacking.organism_rl.adapter_identity import adapter_sha256
from reward_hacking.organism_rl.generate_provenance import _canonical_sha256, _provenance_identity

if TYPE_CHECKING:
    from pathlib import Path


def _source(tmp_path: Path, schema: int = 2) -> tuple[dict[str, Any], Path]:
    trace = tmp_path / "generation" / "traces" / "round" / "episode.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}\n")
    provenance = {
        "flags": {
            "model": "synthetic/model",
            "model_revision": "revision-one",
            "served_model_id": "adapter-name",
            "served_adapter_sha256": "a" * 64,
        },
        "schema_version": schema,
    }
    digest = _canonical_sha256(provenance if schema == 1 else _provenance_identity(provenance))
    doc = {"schema_version": schema, "provenance_sha256": digest, "provenance": provenance}
    trace.parents[2].joinpath("provenance.json").write_text(json.dumps(doc))
    records = tmp_path / "rescored.jsonl"
    return {"trace_path": str(trace), "run_provenance_sha256": digest}, records


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_generation_flags_bind_nearest_operational_provenance(tmp_path: Path, schema: int) -> None:
    record, records = _source(tmp_path, schema)
    flags = diagnostic_identity.generation_flags_for_record(record, records)
    assert flags is not None
    assert flags["model_revision"] == "revision-one"


@pytest.mark.parametrize(
    "corruption", ["record_hash", "document_hash", "document_flags", "compatible_alias"]
)
def test_generation_provenance_mismatch_leaves_flags_unavailable(
    tmp_path: Path, corruption: str
) -> None:
    record, records = _source(tmp_path)
    path = tmp_path / "generation" / "provenance.json"
    doc = json.loads(path.read_text())
    if corruption in {"record_hash", "compatible_alias"}:
        record["run_provenance_sha256"] = "otherhash"
    if corruption == "compatible_alias":
        doc["compatible_provenance_sha256"] = ["otherhash"]
    if corruption == "document_hash":
        doc["provenance_sha256"] = "otherhash"
    if corruption == "document_flags":
        doc["provenance"]["flags"]["model_revision"] = "changed"
    path.write_text(json.dumps(doc))
    assert diagnostic_identity.generation_flags_for_record(record, records) is None


def test_unavailable_generation_flags_leave_adapter_reproduction_unknown() -> None:
    row = _row()
    row["baseline_generation_flags"] = None

    eligible, reason = diagnostic_identity.baseline_reproduction_eligibility(row, _identity())

    assert eligible is None
    assert "unavailable" in reason


def _row() -> dict[str, Any]:
    return {
        "sampling_sanity_applicable": True,
        "baseline_model": None,
        "baseline_model_identity": None,
        "baseline_served_model": None,
        "baseline_generation_flags": {
            "model": "synthetic/model",
            "model_revision": "revision-one",
            "served_model_id": "adapter-name",
            "served_adapter_sha256": "a" * 64,
        },
    }


def _identity() -> Any:
    return diagnostic_identity.StudentGenerationIdentity(
        "synthetic/model", {"reference": "synthetic/model"}, "revision-one", "a" * 64
    )


def test_legacy_matching_revision_and_adapter_qualifies_without_trace_identity() -> None:
    eligible, reason = diagnostic_identity.baseline_reproduction_eligibility(_row(), _identity())
    assert eligible is True
    assert "matching" in reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other/model"),
        ("model_revision", "revision-two"),
        ("served_adapter_sha256", "b" * 64),
    ],
)
def test_legacy_mismatch_cannot_qualify(field: str, value: str) -> None:
    row = _row()
    row["baseline_generation_flags"][field] = value
    eligible, reason = diagnostic_identity.baseline_reproduction_eligibility(row, _identity())
    assert eligible is False
    assert "differ" in reason


def test_missing_revision_is_unknown_and_changed_template_is_ineligible() -> None:
    row = _row()
    row["baseline_generation_flags"]["model_revision"] = None
    assert diagnostic_identity.baseline_reproduction_eligibility(row, _identity())[0] is None
    row["sampling_sanity_applicable"] = False
    assert diagnostic_identity.baseline_reproduction_eligibility(row, _identity())[0] is False


def test_current_adapter_content_hash_changes_after_file_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"synthetic adapter")
    monkeypatch.setattr(diagnostic_identity, "model_identity", lambda model: {"reference": model})
    monkeypatch.setattr(diagnostic_identity, "resolve_model_revision", lambda args: "revision-one")
    first = diagnostic_identity.resolve_student_generation_identity("synthetic/model", adapter)
    assert first.adapter_sha256 == adapter_sha256(adapter)
    weights.write_bytes(b"changed adapter")
    second = diagnostic_identity.resolve_student_generation_identity("synthetic/model", adapter)
    assert first.adapter_sha256 != second.adapter_sha256


def test_modern_semantic_identity_supports_same_weights_at_new_path() -> None:
    row = {
        "sampling_sanity_applicable": True,
        "baseline_model": "old-location",
        "baseline_model_identity": {"tensor_content_sha256": "a" * 64},
        "baseline_served_model": "old-location",
        "baseline_generation_flags": {"model": "old-location", "served_model_id": "old-location"},
    }
    current = diagnostic_identity.StudentGenerationIdentity(
        "new-location", {"tensor_content_sha256": "a" * 64}, None, None
    )
    assert diagnostic_identity.baseline_reproduction_eligibility(row, current)[0] is True


@pytest.mark.parametrize(
    ("recorded_revision", "current_revision", "expected"),
    [
        (None, "revision-one", None),
        ("revision-one", None, None),
        ("revision-one", "revision-two", False),
        ("revision-one", "revision-one", True),
    ],
)
def test_modern_hub_reference_requires_exact_revision_evidence(
    recorded_revision: str | None, current_revision: str | None, expected: bool | None
) -> None:
    row = {
        "sampling_sanity_applicable": True,
        "baseline_model": "synthetic/model",
        "baseline_model_identity": {"reference": "synthetic/model"},
        "baseline_model_revision": recorded_revision,
        "baseline_served_model": "synthetic/model",
        "baseline_generation_flags": {
            "model": "synthetic/model",
            "served_model_id": "synthetic/model",
        },
    }
    current = diagnostic_identity.StudentGenerationIdentity(
        "synthetic/model", {"reference": "synthetic/model"}, current_revision, None
    )
    assert diagnostic_identity.baseline_reproduction_eligibility(row, current)[0] is expected


@pytest.mark.parametrize("served_model", [None, "original-base", "served-adapter-alias"])
def test_modern_local_weights_without_bound_provenance_leave_adapter_unknown(
    served_model: str | None,
) -> None:
    row = {
        "sampling_sanity_applicable": True,
        "baseline_model": "original-base",
        "baseline_model_identity": {"tensor_content_sha256": "a" * 64},
        "baseline_served_model": served_model,
        "baseline_generation_flags": None,
    }
    current = diagnostic_identity.StudentGenerationIdentity(
        "current-base", {"tensor_content_sha256": "a" * 64}, None, None
    )
    eligible, reason = diagnostic_identity.baseline_reproduction_eligibility(row, current)
    assert eligible is None
    assert "adapter" in reason


@pytest.mark.parametrize("served_model", [None, "served-adapter-alias"])
def test_modern_base_alias_cannot_override_missing_or_adapter_provenance(
    served_model: str | None,
) -> None:
    flags: dict[str, Any] = {"model": "original-base"}
    if served_model is not None:
        flags["served_model_id"] = served_model
    row = {
        "sampling_sanity_applicable": True,
        "baseline_model": "original-base",
        "baseline_model_identity": {"tensor_content_sha256": "a" * 64},
        "baseline_served_model": "original-base",
        "baseline_generation_flags": flags,
    }
    current = diagnostic_identity.StudentGenerationIdentity(
        "current-base", {"tensor_content_sha256": "a" * 64}, None, None
    )
    assert diagnostic_identity.baseline_reproduction_eligibility(row, current)[0] is None
