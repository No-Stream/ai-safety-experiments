"""Provenance identity and resume validation for generated organism episodes."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from reward_hacking.evidence_store import iter_jsonl
from reward_hacking.semantic_identity import semantic_content_identity, semantic_server_settings

PROVENANCE_SCHEMA_VERSION = 3
PROVENANCE_SCHEMA_COMPATIBLE_VERSIONS = frozenset({2, PROVENANCE_SCHEMA_VERSION})
RECORD_SCHEMA_VERSION = 6
RECORD_SCHEMA_COMPATIBLE_VERSIONS = frozenset({1, 2, 3, 4, 5, RECORD_SCHEMA_VERSION})
_CODE_PROVENANCE_KEYS = frozenset(
    {
        "git_sha",
        "source_code_sha256",
        "source_sha256",
        "harness_code_sha256",
        "module_sha256",
        "generator_source_sha256",
        "code_sha256",
    }
)
_COMPACT_SELECTED_TASK_FIELDS = (
    "task_id",
    "family",
    "variant",
    "payment_register",
)
_SELECTED_TASKS_REFERENCE_SCHEMA_VERSION = 1
SELECTED_TASKS_RECOVERY_LIMITATION_KEY = "selected_tasks_recovery_limitation"
_SELECTED_TASKS_SEMANTIC_MISMATCH = (
    "historical task definition unverifiable: semantic identity mismatch"
)


def _canonical_sha256(value: Mapping[str, object], *, ensure_ascii: bool = True) -> str:
    serialized = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_launch_context(endpoint: str, max_model_len: int) -> dict[str, object]:
    """Record connection and process details that can change between equivalent launches."""
    endpoint_parts = urlsplit(endpoint)
    if endpoint_parts.scheme not in {"http", "https"} or endpoint_parts.hostname is None:
        raise ValueError("generation endpoint must be an HTTP or HTTPS URL")
    return {
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "host": endpoint_parts.hostname,
        "port": endpoint_parts.port,
        "launched_at_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "max_model_len": max_model_len,
    }


def _provenance_identity(
    provenance: Mapping[str, object],
    *,
    include_code_provenance: bool = False,
    include_legacy_leak_metadata: bool = False,
    include_nonsemantic_serving_settings: bool = False,
) -> dict[str, object]:
    excluded_keys = {
        "run_context",
        "run_contexts",
        "leak_task_source",
        "hint_catalog",
        "task_subset_source",
        "ilcb_split_manifest_source",
        "ilcb_cases_source",
        "selected_tasks_reference",
    }
    if not include_legacy_leak_metadata:
        excluded_keys.update({"leak_assets_path", "leak_eval_pairs"})
    identity = {
        key: deepcopy(value)
        for key, value in provenance.items()
        if key not in excluded_keys
        and (
            include_code_provenance
            or (key != "schema_version" and key not in _CODE_PROVENANCE_KEYS)
        )
    }
    identity.pop("endpoint_sha256", None)
    flags = identity.get("flags")
    if isinstance(flags, Mapping):
        identity_flags = {
            key: deepcopy(value) for key, value in flags.items() if key != "endpoint_sha256"
        }
        if not include_nonsemantic_serving_settings:
            identity_flags.pop("concurrency", None)
            serving_config = identity_flags.get("serving_config")
            if isinstance(serving_config, Mapping):
                identity_flags["serving_config"] = semantic_server_settings(serving_config)
        identity["flags"] = identity_flags
    return identity


def normalize_task_prompt_identity(current: dict[str, object], stored: dict[str, object]) -> None:
    """Normalize embedded legacy prompts and compact prompt identities for comparison."""
    for identity in (current, stored):
        prompts = identity.get("task_prompt_content")
        if isinstance(prompts, Mapping):
            identity["task_prompt_content"] = {
                task_id: semantic_content_identity(content) for task_id, content in prompts.items()
            }


def normalize_task_definition_identity(
    current: dict[str, object], stored: dict[str, object]
) -> None:
    """Compare direct task definitions and retain available legacy semantic evidence."""
    current_content = current.get("task_definition_content", {})
    stored_content = stored.get("task_definition_content", {})
    current_digests = current.get("task_definition_sha256", {})
    stored_digests = stored.get("task_definition_sha256", {})
    if not all(
        isinstance(value, Mapping)
        for value in (current_content, stored_content, current_digests, stored_digests)
    ):
        raise TypeError("task definition identities must be objects")
    current_content = dict(cast("Mapping[str, object]", current_content))
    stored_content = dict(cast("Mapping[str, object]", stored_content))
    current_digests = dict(cast("Mapping[str, object]", current_digests))
    stored_digests = dict(cast("Mapping[str, object]", stored_digests))
    for task_id, content in tuple(current_content.items()):
        if task_id in stored_content:
            current_content[task_id] = semantic_content_identity(content)
            stored_content[task_id] = semantic_content_identity(stored_content[task_id])
            current_digests.pop(task_id, None)
            stored_digests.pop(task_id, None)
        else:
            if isinstance(content, Mapping) and content.get("task_variant") == "leak":
                current_digests.pop(task_id, None)
                stored_digests.pop(task_id, None)
            current_content.pop(task_id)
    for task_id in set(stored_content) - set(current_content):
        stored_content.pop(task_id)
    for field, current_value, stored_value in (
        ("task_definition_content", current_content, stored_content),
        ("task_definition_sha256", current_digests, stored_digests),
    ):
        if current_value or stored_value:
            current[field] = current_value
            stored[field] = stored_value
        else:
            current.pop(field, None)
            stored.pop(field, None)


LEGACY_LEAK_SEMANTIC_FIELDS = (
    "expected_output",
    "leak_channels_open",
    "leak_channels_unavailable",
    "instruction_variant",
)


def legacy_leak_headers(output_dir: Path) -> dict[str, dict[str, object]]:
    """Read retained task identity headers without mutating legacy episode artifacts."""
    headers: dict[str, dict[str, object]] = {}
    for trace_path in sorted((output_dir / "traces").rglob("*.jsonl")):
        header = next(iter_jsonl(trace_path, hydrate=False), None)
        if header is None:
            continue
        if header.get("record") == "organism_episode_identity" and isinstance(
            header.get("task_id"), str
        ):
            headers[cast("str", header["task_id"])] = header
    return headers


def _recover_legacy_leak_evidence(
    current_identity: dict[str, object], stored_identity: dict[str, object], output_dir: Path
) -> None:
    current_tasks = current_identity.get("selected_tasks")
    stored_tasks = stored_identity.get("selected_tasks")
    if not isinstance(current_tasks, list) or not isinstance(stored_tasks, list):
        return
    headers = legacy_leak_headers(output_dir)
    for current_task, stored_task in zip(current_tasks, stored_tasks, strict=False):
        if stored_task.get("family") != "r2e_leak" or "task_definition_content" in stored_task:
            continue
        header = headers.get(stored_task["task_id"], {})
        content = current_task.get("task_definition_content", {})
        metadata = content.get("host_only_metadata", {})
        saved_fields = {key: header[key] for key in LEGACY_LEAK_SEMANTIC_FIELDS if key in header}
        current_fields = {key: metadata.get(key) for key in saved_fields}
        if "unhinted_system_prompt" in header and "prompt_content" in current_task:
            saved_fields["system_text"] = header["unhinted_system_prompt"]
            current_fields["system_text"] = current_task["prompt_content"]["system_text"]
        if saved_fields:
            stored_task["legacy_leak_content"] = saved_fields
            current_task["legacy_leak_content"] = current_fields


def _normalize_legacy_task_prompts(
    current_identity: dict[str, object], stored_identity: dict[str, object]
) -> None:
    current_tasks = cast("list[dict[str, object]]", current_identity["selected_tasks"])
    stored_tasks = cast("list[dict[str, object]]", stored_identity["selected_tasks"])
    for current_task, stored_task in zip(current_tasks, stored_tasks, strict=False):
        if "task_definition_content" in current_task and "task_definition_content" in stored_task:
            current_task["task_definition_content"] = semantic_content_identity(
                current_task["task_definition_content"]
            )
            stored_task["task_definition_content"] = semantic_content_identity(
                stored_task["task_definition_content"]
            )
            current_task.pop("task_definition_sha256", None)
            stored_task.pop("task_definition_sha256", None)
        elif stored_task.get("family") == "r2e_leak":
            current_task.pop("task_definition_sha256", None)
            stored_task.pop("task_definition_sha256", None)
            current_task.pop("task_definition_content", None)
        else:
            current_task.pop("task_definition_content", None)
            stored_task.pop("task_definition_content", None)
        if "prompt_content" in current_task and "prompt_content" in stored_task:
            current_task["prompt_content"] = semantic_content_identity(
                current_task["prompt_content"]
            )
            stored_task["prompt_content"] = semantic_content_identity(stored_task["prompt_content"])
        if "prompt_content" not in stored_task:
            current_task.pop("prompt_content", None)
        elif "prompt_content" not in current_task:
            stored_task.pop("prompt_content", None)


def _normalize_selected_task_comparison(
    current_identity: dict[str, object], stored_identity: dict[str, object]
) -> None:
    if isinstance(current_identity.get("selected_tasks"), list) and isinstance(
        stored_identity.get("selected_tasks"), list
    ):
        _normalize_sampling_policy_comparison(current_identity, stored_identity)
        current_identity.pop("task_data_sha256", None)
        stored_identity.pop("task_data_sha256", None)
        current_identity.pop("task_skip_counts", None)
        stored_identity.pop("task_skip_counts", None)
        for identity in (current_identity, stored_identity):
            record_schema = identity.get("record_schema")
            if type(record_schema) is int and record_schema in RECORD_SCHEMA_COMPATIBLE_VERSIONS:
                identity["record_schema"] = "compatible"
            for exclusion_field in ("task_id_exclusion", "task_sample_exclusion"):
                exclusion = identity.get(exclusion_field)
                if (
                    isinstance(exclusion, Mapping)
                    and type(exclusion.get("schema_version")) is int
                    and cast("int", exclusion["schema_version"]) > 0
                ):
                    identity[exclusion_field] = {
                        key: value for key, value in exclusion.items() if key != "schema_version"
                    }
            flags = identity.get("flags")
            if isinstance(flags, Mapping):
                identity["flags"] = {
                    key: value
                    for key, value in flags.items()
                    if key not in {"prompt_format_version", "max_tasks"}
                }
        _normalize_legacy_task_prompts(current_identity, stored_identity)
        for field in ("selected_tasks", "task_selection_order", "task_sample_selection"):
            current_values = current_identity.get(field)
            stored_values = stored_identity.get(field)
            if (
                isinstance(current_values, list)
                and isinstance(stored_values, list)
                and current_values[: len(stored_values)] == stored_values
            ):
                current_identity[field] = stored_values
        current_registers = current_identity.get("payment_registers")
        stored_registers = stored_identity.get("payment_registers")
        if isinstance(current_registers, Mapping) and isinstance(stored_registers, Mapping):
            current_identity["payment_registers"] = {
                key: current_registers[key] for key in stored_registers if key in current_registers
            }


def _normalize_sampling_policy_comparison(
    current_identity: dict[str, object], stored_identity: dict[str, object]
) -> None:
    flags = [identity.get("flags") for identity in (current_identity, stored_identity)]
    if not all(isinstance(value, Mapping) and "samples_per_task" in value for value in flags):
        return
    for identity in (current_identity, stored_identity):
        for field in ("task_id_exclusion", "task_sample_exclusion", "task_sample_selection"):
            identity.pop(field, None)
        identity["flags"] = {
            key: value
            for key, value in cast("Mapping[str, object]", identity["flags"]).items()
            if key != "samples_per_task"
        }


def _stored_generation_identity_sha256_candidates(
    stored_generation_identity: Mapping[str, object],
    normalized_stored_generation: Mapping[str, object],
) -> set[str]:
    evaluation_excluded_fields = {"schema_version"}
    if isinstance(stored_generation_identity.get("task_prompt_sha256"), Mapping):
        evaluation_excluded_fields.add("prompt_format_version")
    stored_evaluation_projection = {
        key: value
        for key, value in stored_generation_identity.items()
        if key not in evaluation_excluded_fields
    }
    stored_semantic_projection = {
        key: value
        for key, value in stored_generation_identity.items()
        if key not in {"schema_version", "prompt_format_version"}
    }
    return {
        _canonical_sha256(dict(stored_generation_identity)),
        _canonical_sha256(dict(normalized_stored_generation)),
        _canonical_sha256(stored_semantic_projection),
        _canonical_sha256(stored_evaluation_projection, ensure_ascii=False),
        _canonical_sha256(stored_evaluation_projection, ensure_ascii=True),
    }


def _identity_differences(
    current_identity: dict[str, object], stored_identity: Mapping[str, object]
) -> list[str]:
    current_identity = deepcopy(current_identity)
    stored_identity = deepcopy(dict(stored_identity))
    if isinstance(current_identity.get("generation_path_identity"), Mapping) and isinstance(
        stored_identity.get("generation_path_identity"), Mapping
    ):
        for identity in (current_identity, stored_identity):
            for task in cast("list[dict[str, object]]", identity.get("selected_tasks", [])):
                for field in (
                    "task_definition_content",
                    "task_definition_sha256",
                    "prompt_content",
                ):
                    task.pop(field, None)
    _normalize_selected_task_comparison(current_identity, stored_identity)
    generation_identity = current_identity.get("generation_path_identity")
    stored_generation_identity = stored_identity.get("generation_path_identity")
    comparison_stored = deepcopy(dict(stored_identity))
    if isinstance(generation_identity, Mapping) and isinstance(stored_generation_identity, Mapping):
        normalized_current_generation = {
            key: value for key, value in generation_identity.items() if key != "schema_version"
        }
        normalized_stored_generation = {
            key: value
            for key, value in stored_generation_identity.items()
            if key != "schema_version"
        }
        if isinstance(generation_identity.get("task_prompt_sha256"), Mapping) and isinstance(
            stored_generation_identity.get("task_prompt_sha256"), Mapping
        ):
            normalized_current_generation.pop("prompt_format_version", None)
            normalized_stored_generation.pop("prompt_format_version", None)
        if "task_prompt_content" not in stored_generation_identity:
            normalized_current_generation.pop("task_prompt_content", None)
        if "task_prompt_content" not in generation_identity:
            normalized_stored_generation.pop("task_prompt_content", None)
        normalize_task_prompt_identity(normalized_current_generation, normalized_stored_generation)
        normalize_task_definition_identity(
            normalized_current_generation, normalized_stored_generation
        )
        current_identity["generation_path_identity"] = normalized_current_generation
        comparison_stored["generation_path_identity"] = normalized_stored_generation
        semantic_generation_sha256 = _canonical_sha256(normalized_current_generation)
        current_identity["generation_path_identity_sha256"] = semantic_generation_sha256
        stored_generation_sha256 = stored_identity.get("generation_path_identity_sha256")
        if stored_generation_sha256 in _stored_generation_identity_sha256_candidates(
            stored_generation_identity,
            normalized_stored_generation,
        ):
            normalized_stored_sha256 = _canonical_sha256(normalized_stored_generation)
            comparison_stored["generation_path_identity_sha256"] = normalized_stored_sha256
    return _mapping_differences(current_identity, comparison_stored)


def _mapping_differences(
    current: Mapping[str, object], stored: Mapping[str, object], prefix: str = ""
) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(current) | set(stored)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in current or key not in stored:
            differences.append(path)
            continue
        current_value = current[key]
        stored_value = stored[key]
        if isinstance(current_value, Mapping) and isinstance(stored_value, Mapping):
            differences.extend(_mapping_differences(current_value, stored_value, path))
        elif current_value != stored_value:
            differences.append(path)
    return differences


def _legacy_run_context(provenance: Mapping[str, object]) -> dict[str, object] | None:
    flags = provenance.get("flags")
    if not isinstance(flags, Mapping) or "endpoint_sha256" not in flags:
        return None
    context: dict[str, object] = {
        "endpoint_sha256": flags["endpoint_sha256"],
        "legacy_context": True,
    }
    max_model_len = flags.get("max_model_len")
    if type(max_model_len) is int:
        context["max_model_len"] = max_model_len
    return context


def write_json_atomically(path: Path, value: Mapping[str, object]) -> None:
    """Replace one JSON artifact only after the complete payload has reached disk."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def compact_selected_tasks(
    selected_tasks: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Keep selection identity while leaving semantic inputs in the sidecar."""
    return [
        {key: deepcopy(task[key]) for key in _COMPACT_SELECTED_TASK_FIELDS if key in task}
        for task in selected_tasks
    ]


def _selected_tasks_sidecar_path(provenance_path: Path, payload: bytes) -> Path:
    base_path = provenance_path.with_name(f"{provenance_path.stem}.selected-tasks.json.gz")
    if not base_path.exists() or gzip.decompress(base_path.read_bytes()) == payload:
        return base_path
    digest = hashlib.sha256(payload).hexdigest()
    versioned_path = provenance_path.with_name(
        f"{provenance_path.stem}.selected-tasks.{digest}.json.gz"
    )
    if versioned_path.exists() and gzip.decompress(versioned_path.read_bytes()) != payload:
        raise ValueError("selected-task sidecar content-address collision")
    return versioned_path


def _write_gzip_atomically(path: Path, payload: bytes) -> None:
    compressed_payload = gzip.compress(payload, mtime=0)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(compressed_payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def compact_provenance(
    provenance: Mapping[str, object], provenance_path: Path
) -> dict[str, object]:
    """Store full selected-task provenance in an immutable gzip sidecar."""
    if "selected_tasks_reference" in provenance:
        hydrate_provenance(provenance, provenance_path)
        return deepcopy(dict(provenance))
    if "selected_tasks" not in provenance:
        return deepcopy(dict(provenance))
    raw_selected_tasks = provenance["selected_tasks"]
    if not isinstance(raw_selected_tasks, list) or any(
        not isinstance(task, Mapping) for task in raw_selected_tasks
    ):
        raise TypeError("selected_tasks provenance must be a list of objects")
    selected_tasks = [dict(task) for task in cast("list[Mapping[str, object]]", raw_selected_tasks)]
    payload = json.dumps(
        selected_tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    sidecar_path = _selected_tasks_sidecar_path(provenance_path, payload)
    if not sidecar_path.exists():
        _write_gzip_atomically(sidecar_path, payload)
    compacted = deepcopy(dict(provenance))
    compacted["selected_tasks"] = compact_selected_tasks(selected_tasks)
    compacted["selected_tasks_reference"] = {
        "schema_version": _SELECTED_TASKS_REFERENCE_SCHEMA_VERSION,
        "path": sidecar_path.name,
    }
    return compacted


def hydrate_provenance(
    provenance: Mapping[str, object], provenance_path: Path
) -> dict[str, object]:
    """Restore exact selected-task provenance from its sidecar, accepting legacy inline data."""
    hydrated = deepcopy(dict(provenance))
    reference = hydrated.get("selected_tasks_reference")
    if reference is None:
        return hydrated
    if not isinstance(reference, Mapping):
        raise TypeError("selected_tasks_reference must be an object")
    if reference.get("schema_version") != _SELECTED_TASKS_REFERENCE_SCHEMA_VERSION:
        raise ValueError("selected_tasks_reference has an unsupported schema version")
    relative_path = reference.get("path")
    if not isinstance(relative_path, str) or Path(relative_path).name != relative_path:
        raise ValueError("selected_tasks_reference path must be a sibling filename")
    expected_prefix = f"{provenance_path.stem}.selected-tasks"
    if not relative_path.startswith(expected_prefix) or not relative_path.endswith(".json.gz"):
        raise ValueError("selected_tasks_reference path does not match its provenance file")
    compact_tasks = hydrated.get("selected_tasks")
    if not isinstance(compact_tasks, list) or any(
        not isinstance(task, Mapping) for task in compact_tasks
    ):
        raise TypeError("compacted selected_tasks provenance must be a list of objects")
    sidecar_path = provenance_path.with_name(relative_path)
    selected_tasks_value: object = json.loads(gzip.decompress(sidecar_path.read_bytes()))
    if not isinstance(selected_tasks_value, list) or any(
        not isinstance(task, Mapping) for task in selected_tasks_value
    ):
        raise TypeError("selected-task sidecar must contain a list of objects")
    selected_tasks = [
        dict(task) for task in cast("list[Mapping[str, object]]", selected_tasks_value)
    ]
    if compact_selected_tasks(selected_tasks) != compact_tasks:
        hydrated[SELECTED_TASKS_RECOVERY_LIMITATION_KEY] = _SELECTED_TASKS_SEMANTIC_MISMATCH
        return hydrated
    hydrated["selected_tasks"] = selected_tasks
    hydrated.pop("selected_tasks_reference")
    return hydrated


def _read_existing_provenance(path: Path) -> Mapping[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("existing run provenance must be an object")
    schema_version = value.get("schema_version")
    provenance_sha256 = value.get("provenance_sha256")
    provenance = value.get("provenance")
    if type(schema_version) is not int or not isinstance(provenance_sha256, str):
        raise ValueError("existing run provenance has an invalid schema or digest")
    if not isinstance(provenance, Mapping):
        raise TypeError("existing run provenance must contain an object at 'provenance'")
    hydrated_document = dict(value)
    hydrated_provenance = hydrate_provenance(provenance, path)
    limitation = hydrated_provenance.pop(SELECTED_TASKS_RECOVERY_LIMITATION_KEY, None)
    hydrated_document["provenance"] = hydrated_provenance
    if isinstance(limitation, str):
        hydrated_document[SELECTED_TASKS_RECOVERY_LIMITATION_KEY] = limitation
    return hydrated_document


def _validate_resume_identity(
    current_identity: dict[str, object],
    stored_identity: Mapping[str, object],
    *,
    output_dir: Path | None = None,
) -> None:
    current_identity = deepcopy(current_identity)
    stored_identity = deepcopy(dict(stored_identity))
    if output_dir is not None:
        _recover_legacy_leak_evidence(current_identity, stored_identity, output_dir)
    differences = _identity_differences(deepcopy(current_identity), stored_identity)
    if differences:
        raise ValueError(
            "current generation-path provenance identity differs from the existing output directory; "
            "use a fresh output directory. Differing keys: " + ", ".join(differences)
        )


def _load_legacy_provenance(
    existing: Mapping[str, object],
    current_identity: dict[str, object],
    *,
    output_dir: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    stored_hash = cast("str", existing["provenance_sha256"])
    stored_provenance = cast("Mapping[str, object]", existing["provenance"])
    if _canonical_sha256(stored_provenance) != stored_hash:
        raise ValueError("legacy run provenance digest is invalid; use a fresh output directory")
    stored_identity = _provenance_identity(stored_provenance)
    _validate_resume_identity(current_identity, stored_identity, output_dir=output_dir)
    legacy_context = _legacy_run_context(stored_provenance)
    run_contexts = [] if legacy_context is None else [legacy_context]
    return current_identity, run_contexts, {stored_hash}


def _load_current_provenance(
    existing: Mapping[str, object],
    current_identity: dict[str, object],
    *,
    output_dir: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    stored_provenance = cast("Mapping[str, object]", existing["provenance"])
    stored_identity = _provenance_identity(stored_provenance)
    stored_hash = cast("str", existing["provenance_sha256"])
    semantic_hash = _canonical_sha256(stored_identity)
    legacy_leak_metadata_hash = _canonical_sha256(
        _provenance_identity(stored_provenance, include_legacy_leak_metadata=True)
    )
    legacy_code_hash = _canonical_sha256(
        _provenance_identity(
            stored_provenance,
            include_code_provenance=True,
            include_legacy_leak_metadata=True,
        )
    )
    legacy_serving_hashes = {
        _canonical_sha256(
            _provenance_identity(
                stored_provenance,
                include_nonsemantic_serving_settings=True,
            )
        ),
        _canonical_sha256(
            _provenance_identity(
                stored_provenance,
                include_legacy_leak_metadata=True,
                include_nonsemantic_serving_settings=True,
            )
        ),
        _canonical_sha256(
            _provenance_identity(
                stored_provenance,
                include_code_provenance=True,
                include_legacy_leak_metadata=True,
                include_nonsemantic_serving_settings=True,
            )
        ),
    }
    if stored_hash not in {
        semantic_hash,
        legacy_leak_metadata_hash,
        legacy_code_hash,
        *legacy_serving_hashes,
    }:
        raise ValueError("existing generation-path identity digest is invalid")
    raw_compatible_hashes = existing.get("compatible_provenance_sha256", [])
    if not isinstance(raw_compatible_hashes, list) or any(
        not isinstance(value, str) for value in raw_compatible_hashes
    ):
        raise TypeError("compatible provenance digests must be a list of strings")
    raw_run_contexts = stored_provenance.get("run_contexts", [])
    if not isinstance(raw_run_contexts, list) or any(
        not isinstance(value, Mapping) for value in raw_run_contexts
    ):
        raise TypeError("stored run_contexts must be a list of objects")
    _validate_resume_identity(current_identity, stored_identity, output_dir=output_dir)
    current_projection_hashes = {
        semantic_hash,
        legacy_leak_metadata_hash,
        legacy_code_hash,
    }
    needs_legacy_serving_context = (
        stored_hash in legacy_serving_hashes and stored_hash not in current_projection_hashes
    )
    legacy_serving_context = (
        _legacy_serving_settings_context(stored_provenance)
        if needs_legacy_serving_context
        else None
    )
    run_contexts = [dict(value) for value in raw_run_contexts]
    if legacy_serving_context is not None:
        run_contexts.append(legacy_serving_context)
    compatible_hashes = set(cast("list[str]", raw_compatible_hashes))
    if stored_hash != _canonical_sha256(current_identity):
        compatible_hashes.add(stored_hash)
    return (
        current_identity,
        run_contexts,
        compatible_hashes,
    )


def _legacy_serving_settings_context(
    provenance: Mapping[str, object],
) -> dict[str, object] | None:
    flags = provenance.get("flags")
    if not isinstance(flags, Mapping):
        return None
    context: dict[str, object] = {"legacy_serving_settings": True}
    concurrency = flags.get("concurrency")
    if type(concurrency) is int:
        context["concurrency"] = concurrency
    serving_config = flags.get("serving_config")
    if isinstance(serving_config, Mapping):
        context["serving_config"] = deepcopy(dict(serving_config))
    return context if len(context) > 1 else None


def active_provenance_path(output_dir: Path) -> Path:
    """Read compact resume metadata without rewriting a historical provenance artifact."""
    resume_path = output_dir / "provenance.resume.json"
    original_path = output_dir / "provenance.json"
    if resume_path.is_file() and not original_path.is_file():
        raise ValueError("resume provenance exists without its historical provenance")
    return resume_path if resume_path.is_file() else original_path


def _load_existing_provenance(
    path: Path, current_identity: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]], set[str]]:
    if path.name == "provenance.resume.json":
        _load_existing_provenance(path.with_name("provenance.json"), current_identity)
    existing = _read_existing_provenance(path)
    schema_version = cast("int", existing["schema_version"])
    if schema_version == 1:
        return _load_legacy_provenance(existing, current_identity, output_dir=path.parent)
    if schema_version in PROVENANCE_SCHEMA_COMPATIBLE_VERSIONS:
        return _load_current_provenance(existing, current_identity, output_dir=path.parent)
    raise ValueError(
        f"unsupported run provenance schema {schema_version}; use a fresh output directory"
    )


def sampling_policy(provenance: Mapping[str, object]) -> dict[str, object] | None:
    """Extract the per-task sampling policy a resume compares, or None when the run recorded none."""
    flags = provenance.get("flags")
    if not isinstance(flags, Mapping) or "samples_per_task" not in flags:
        return None
    return {
        "samples_per_task": flags["samples_per_task"],
        **{
            field: deepcopy(provenance[field])
            for field in ("task_id_exclusion", "task_sample_exclusion", "task_sample_selection")
            if field in provenance
        },
    }
