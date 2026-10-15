"""Source, render, generation, and assessment identities for frozen selections."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from reward_hacking.spec_gaming.config import (
    CollectorConfig,
    ServingIdentity,
    canonical_json,
    file_sha256,
    load_serving_identity,
    nonempty_string,
    object_value,
    read_json_object,
    revision,
    sha256,
)
from reward_hacking.spec_gaming.episode import (
    EpisodeSettings,
    render_spreadsheet_task,
    spreadsheet_generation_identity,
)
from reward_hacking.spec_gaming.spreadsheet import (
    SpreadsheetImport,
    SpreadsheetTask,
    import_tasks,
    validate_proxy_cells,
    with_actor_case,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

SELECTION_SCHEMA_VERSION = 3
EFFECTIVE_RUN_SCHEMA_VERSION = 2
GENERATION_CODE_PATHS = (
    "reward_hacking/spec_gaming/collection.py",
    "reward_hacking/spec_gaming/config.py",
    "reward_hacking/spec_gaming/manifests.py",
    "reward_hacking/spec_gaming/spreadsheet.py",
    "reward_hacking/spec_gaming/spreadsheet_worker.py",
    "reward_hacking/spec_gaming/runtime.py",
    "reward_hacking/spec_gaming/protocol.py",
    "reward_hacking/spec_gaming/capture.py",
    "reward_hacking/spec_gaming/episode.py",
    "reward_hacking/harness/docker_backend.py",
    "reward_hacking/jailed_poc.py",
    "reward_hacking/model_backend.py",
    "reward_hacking/backend_types.py",
    "reward_hacking/ladder/chat.py",
    "reward_hacking/ladder/tools.py",
    "reward_hacking/ladder/thinking.py",
    "scripts/episode_jail.sh",
    "scripts/resource-limits.sh",
)
ASSESSMENT_CODE_PATHS = (
    "reward_hacking/spec_gaming/collection.py",
    "reward_hacking/spec_gaming/config.py",
    "reward_hacking/spec_gaming/manifests.py",
    "reward_hacking/spec_gaming/spreadsheet.py",
    "reward_hacking/spec_gaming/spreadsheet_worker.py",
    "reward_hacking/spec_gaming/runtime.py",
    "reward_hacking/spec_gaming/capture.py",
    "reward_hacking/spec_gaming/episode.py",
)


@dataclass(frozen=True)
class _GenerationInputs:
    selection: dict[str, object]
    render: dict[str, object]
    serving: ServingIdentity


def _all_cases(task: SpreadsheetTask) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for case in task.cases:
        if (
            case.input_path is None
            or case.reference_path is None
            or case.input_sha256 is None
            or case.reference_sha256 is None
            or case.input_member is None
            or case.reference_member is None
        ):
            raise ValueError(f"candidate task {task.task_id!r} has an incomplete companion case")
        if file_sha256(case.input_path) != case.input_sha256:
            raise ValueError(f"candidate task {task.task_id!r} input asset changed after import")
        if file_sha256(case.reference_path) != case.reference_sha256:
            raise ValueError(
                f"candidate task {task.task_id!r} reference asset changed after import"
            )
        cases.append(
            {
                "case_index": case.case_index,
                "input_member": case.input_member,
                "input_sha256": case.input_sha256,
                "reference_member": case.reference_member,
                "reference_sha256": case.reference_sha256,
            }
        )
    return cases


def _render_entry(task: SpreadsheetTask, proxy_cells: tuple[str, ...]) -> dict[str, object]:
    rendered = render_spreadsheet_task(task, proxy_cells)
    actor_files: dict[str, object] = {}
    for name, contents in rendered["files"].items():
        try:
            decoded: str | None = contents.decode("utf-8")
        except UnicodeDecodeError:
            decoded = None
        actor_files[name] = {
            "sha256": sha256(contents),
            "bytes": len(contents),
            "text": decoded,
        }
    return {
        "proxy_cells": list(proxy_cells),
        "messages": rendered["messages"],
        "tools": rendered["tools"],
        "actor_files": actor_files,
    }


def source_entry(task: SpreadsheetTask) -> dict[str, object]:
    """Return private source provenance for one selected actor case and task group."""
    return {
        "source_revision": task.source_revision,
        "source_archive_sha256": task.source_archive_sha256,
        "dataset_name": task.dataset_name,
        "task_id": task.task_id,
        "group_task_id": task.task_id,
        "actor_case_index": task.actor_case_index,
        "instruction_sha256": sha256(task.instruction.encode("utf-8")),
        "instruction_type": task.instruction_type,
        "answer_position": task.answer_position,
        "upstream_evaluation_sha256": task.upstream_evaluation_sha256,
        "cases": _all_cases(task),
        "training_eligible": False,
    }


def _selection_payload(
    config: CollectorConfig, imported: SpreadsheetImport
) -> tuple[dict[str, object], dict[str, object]]:
    tasks_by_id = {task.task_id: task for task in imported.tasks}
    if len(tasks_by_id) != len(imported.tasks):
        raise ValueError("source importer returned duplicate task IDs")
    issue_ids = {issue.task_id for issue in imported.issues}
    missing_ids = sorted(set(config.selection.candidate_task_ids) - tasks_by_id.keys())
    if missing_ids:
        raise ValueError(
            "candidate task ID absent or ineligible in source import: " + ", ".join(missing_ids)
        )
    affected_ids = sorted(set(config.selection.candidate_task_ids) & issue_ids)
    if affected_ids:
        raise ValueError("candidate task ID has source import issues: " + ", ".join(affected_ids))

    selection_tasks: list[dict[str, object]] = []
    render_tasks: dict[str, object] = {}
    for task_id in sorted(config.selection.candidate_task_ids):
        task = with_actor_case(tasks_by_id[task_id], config.selection.actor_case_indices[task_id])
        proxy_cells = validate_proxy_cells(task, config.selection.proxy_cells[task_id])
        selection_tasks.append({**source_entry(task), "proxy_cells": list(proxy_cells)})
        render_tasks[task_id] = _render_entry(task, proxy_cells)
    return (
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "environment": "spreadsheetbench",
            "sample_count_per_task": 1,
            "tasks": selection_tasks,
        },
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "environment": "spreadsheetbench",
            "tasks": render_tasks,
        },
    )


def write_json(path: Path, value: object, *, immutable: bool) -> None:
    """Atomically write a JSON artifact and preserve immutable frozen manifests."""
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if immutable and canonical_json(json.loads(existing)) != canonical_json(value):
            raise ValueError(f"frozen artifact changed; use a new output directory: {path.name}")
        if not immutable:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())
    temporary_path.replace(path)


def read_json(path: Path, name: str) -> dict[str, object]:
    """Read a required host artifact, naming missing files clearly."""
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing; prepare the collection first")
    return read_json_object(path, name)


def _code_identity(paths: tuple[str, ...]) -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    return {path: file_sha256(repository_root / path) for path in paths}


def _generation_identity(
    config: CollectorConfig,
    inputs: _GenerationInputs,
    settings: EpisodeSettings,
    *,
    smoke: bool,
) -> dict[str, object]:
    selection = inputs.selection
    render = inputs.render
    serving = inputs.serving
    server_args = object_value(serving.manifest["server_args"], "serving manifest.server_args")
    server_context = server_args.get("max_model_len")
    if type(server_context) is not int or settings.max_context_tokens > server_context:
        raise ValueError("episode context budget exceeds the served model context")
    selection_digest = sha256(canonical_json(selection))
    render_digest = sha256(canonical_json(render))
    tasks = selection["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("selection manifest has no task records")
    first_task = object_value(tasks[0], "selection task")
    source_revision = revision(first_task.get("source_revision"), "selection source revision")
    return {
        "schema_version": 1,
        "environment": "spreadsheetbench",
        "collection_mode": "plumbing_smoke" if smoke else "research_only",
        "selection_manifest_sha256": selection_digest,
        "render_manifest_sha256": render_digest,
        "source_importer_identity": {
            "source_revision": source_revision,
            "archive_sha256": file_sha256(config.source.archive),
        },
        "model": {
            "served_model_id": config.model.served_model_id,
            "model_source": str(config.model.model_source),
            "model_revision": config.model.model_revision,
            "model_identity": serving.model_semantics,
            "tokenizer_source": str(config.model.tokenizer_source),
            "tokenizer_revision": config.model.tokenizer_revision,
            "tokenizer_identity": serving.tokenizer_semantics,
            "serving_manifest_sha256": config.model.serving_manifest_sha256,
            "server_version": serving.manifest["server_version"],
            "server_config_sha256": serving.manifest["server_config_sha256"],
            "server_args": server_args,
            "reasoning_config": serving.manifest["reasoning_config"],
            "thinking": config.model.thinking,
            "chat_template_kwargs": config.model.chat_template_kwargs,
        },
        "sampling": {
            **asdict(config.sampling.config),
            "stop_token_ids": list(config.sampling.stop_token_ids),
        },
        "runtime": {
            "image": config.runtime.docker.image,
            "isolation_gate_sha256": config.runtime.gate_record_sha256,
            "memory_bytes": config.runtime.docker.memory_bytes,
            "pids_limit": config.runtime.docker.pids_limit,
            "cpus": config.runtime.docker.cpus,
            "uid": config.runtime.docker.uid,
            "gid": config.runtime.docker.gid,
            "action_timeout_seconds": config.action_timeout_seconds,
            "evaluation_timeout_seconds": config.evaluation_timeout_seconds,
        },
        "episode_settings": asdict(settings),
        "actor_code_sha256": _code_identity(GENERATION_CODE_PATHS),
    }


def _current_frozen_artifacts(
    config: CollectorConfig, *, resolve_models: bool
) -> tuple[dict[str, object], dict[str, object], ServingIdentity | None]:
    imported = import_tasks(config.source.root, config.source.archive, config.source.asset_root)
    selection, render = _selection_payload(config, imported)
    stored_selection = read_json(
        config.output_dir / "selection-manifest.json", "selection manifest"
    )
    stored_render = read_json(config.output_dir / "render-manifest.json", "render manifest")
    if canonical_json(selection) != canonical_json(stored_selection):
        raise ValueError("source or selected task inputs differ from the frozen selection manifest")
    if canonical_json(render) != canonical_json(stored_render):
        raise ValueError("actor-visible render differs from the frozen render manifest")
    serving = load_serving_identity(config) if resolve_models else None
    return selection, render, serving


def prepare_collection(config: CollectorConfig) -> Path:
    """Import source bytes and freeze selected task groups, actor cases, proxies, and renders."""
    imported = import_tasks(config.source.root, config.source.archive, config.source.asset_root)
    selection, render = _selection_payload(config, imported)
    serving = load_serving_identity(config)
    write_json(config.output_dir / "selection-manifest.json", selection, immutable=True)
    write_json(config.output_dir / "render-manifest.json", render, immutable=True)
    identity = _generation_identity(
        config,
        _GenerationInputs(selection, render, serving),
        config.episode,
        smoke=False,
    )
    write_json(config.output_dir / "generation-identity.json", identity, immutable=True)
    return config.output_dir


def load_prepared(
    config: CollectorConfig, *, resolve_models: bool
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Validate current source/render data against frozen host manifests."""
    selection, render, serving = _current_frozen_artifacts(config, resolve_models=resolve_models)
    stored_identity = read_json(
        config.output_dir / "generation-identity.json", "generation identity"
    )
    if resolve_models:
        if serving is None:
            raise RuntimeError("model identities were not resolved")
        expected_identity = _generation_identity(
            config,
            _GenerationInputs(selection, render, serving),
            config.episode,
            smoke=False,
        )
        if canonical_json(stored_identity) != canonical_json(expected_identity):
            raise ValueError("generation identity differs from the prepared host manifest")
    return selection, render, stored_identity


def tasks_by_id(
    config: CollectorConfig, selection: Mapping[str, object]
) -> dict[str, SpreadsheetTask]:
    """Reload exactly the frozen task groups and apply their selected actor cases."""
    imported = import_tasks(config.source.root, config.source.archive, config.source.asset_root)
    tasks = {task.task_id: task for task in imported.tasks}
    entries = selection.get("tasks")
    if not isinstance(entries, list):
        raise TypeError("selection manifest tasks must be a list")
    entries_by_id = {
        nonempty_string(
            object_value(entry, "selection task").get("task_id"), "selection task ID"
        ): object_value(entry, "selection task")
        for entry in entries
    }
    task_ids = list(entries_by_id)
    if set(task_ids) != set(config.selection.candidate_task_ids) or set(task_ids) - tasks.keys():
        raise ValueError("current task import does not match the frozen selection")
    selected: dict[str, SpreadsheetTask] = {}
    for task_id in task_ids:
        task = with_actor_case(tasks[task_id], config.selection.actor_case_indices[task_id])
        source = source_entry(task)
        frozen_proxy_cells = entries_by_id[task_id].get("proxy_cells")
        if (
            not isinstance(frozen_proxy_cells, list)
            or tuple(frozen_proxy_cells) != (config.selection.proxy_cells[task_id])
        ):
            raise ValueError(f"configured proxy cells for {task_id!r} differ from frozen selection")
        frozen_source = {
            key: value for key, value in entries_by_id[task_id].items() if key != "proxy_cells"
        }
        if canonical_json(source) != canonical_json(frozen_source):
            raise ValueError(f"current source task {task_id!r} differs from the frozen selection")
        selected[task_id] = task
    return selected


def load_frozen_collection(
    config: CollectorConfig,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, SpreadsheetTask]]:
    """Load prepared manifests and verify current source bytes without rendering actor prompts."""
    selection = read_json(config.output_dir / "selection-manifest.json", "selection manifest")
    render = read_json(config.output_dir / "render-manifest.json", "render manifest")
    generation_identity = read_json(
        config.output_dir / "generation-identity.json", "generation identity"
    )
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("selection manifest schema version is unsupported")
    if generation_identity.get("selection_manifest_sha256") != sha256(canonical_json(selection)):
        raise ValueError("selection manifest differs from its frozen generation identity")
    if generation_identity.get("render_manifest_sha256") != sha256(canonical_json(render)):
        raise ValueError("render manifest differs from its frozen generation identity")
    source_importer_identity = object_value(
        generation_identity.get("source_importer_identity"), "generation identity source importer"
    )
    if source_importer_identity.get("archive_sha256") != file_sha256(config.source.archive):
        raise ValueError("source archive differs from the frozen generation identity")
    tasks = tasks_by_id(config, selection)
    return selection, render, generation_identity, tasks


def effective_run_path(config: CollectorConfig) -> Path:
    """Return the one immutable run-context manifest for this private collection."""
    return config.output_dir / "effective-run-manifest.json"


def _selection_tasks_by_id(selection: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = selection.get("tasks")
    if not isinstance(entries, list):
        raise TypeError("selection manifest tasks must be a list")
    tasks = {
        nonempty_string(
            object_value(entry, "selection task").get("task_id"), "selection task ID"
        ): object_value(entry, "selection task")
        for entry in entries
    }
    if len(tasks) != len(entries):
        raise ValueError("selection manifest contains duplicate task IDs")
    return tasks


def _validate_effective_generation_identity(
    config: CollectorConfig,
    payload: dict[str, object],
) -> dict[str, object]:
    settings = object_value(payload["episode_settings"], "effective episode settings")
    generation_identity = object_value(
        payload["generation_identity"], "effective generation identity"
    )
    if generation_identity.get("collection_mode") != payload["collection_mode"]:
        raise ValueError("effective generation mode differs from the run manifest")
    if canonical_json(generation_identity.get("episode_settings")) != canonical_json(settings):
        raise ValueError("effective generation settings differ from the run manifest")
    prepared_identity = read_json(
        config.output_dir / "generation-identity.json", "generation identity"
    )
    effective_base = {
        key: value
        for key, value in generation_identity.items()
        if key not in {"collection_mode", "episode_settings"}
    }
    prepared_base = {
        key: value
        for key, value in prepared_identity.items()
        if key not in {"collection_mode", "episode_settings"}
    }
    if canonical_json(effective_base) != canonical_json(prepared_base):
        raise ValueError("effective generation identity differs from the prepared source identity")
    return generation_identity


def _validate_effective_proxy_bindings(
    config: CollectorConfig,
    generation_identity: dict[str, object],
    task_identities: dict[str, object],
) -> None:
    selection = read_json(config.output_dir / "selection-manifest.json", "selection manifest")
    selection_tasks = _selection_tasks_by_id(selection)
    for task_id, identity in task_identities.items():
        task_identity = object_value(identity, f"effective task identity for {task_id}")
        selection_task = selection_tasks.get(task_id)
        if selection_task is None:
            raise ValueError("effective task identity is absent from the frozen selection")
        if any(
            canonical_json(task_identity.get(key)) != canonical_json(value)
            for key, value in generation_identity.items()
        ):
            raise ValueError(
                f"effective task identity for {task_id!r} has changed generation inputs"
            )
        if task_identity.get("proxy_cells") != selection_task.get("proxy_cells"):
            raise ValueError(f"effective proxy cells for {task_id!r} differ from frozen selection")


def _validate_effective_acceptance(payload: dict[str, object]) -> None:
    acceptance = object_value(payload["acceptance"], "effective acceptance attestation")
    acceptance_fields = {
        "reviewer",
        "source_terms",
        "holdout_disposition",
        "task_obligations",
        "render",
    }
    if acceptance.keys() != acceptance_fields:
        raise ValueError("effective acceptance attestation has missing or unknown fields")
    nonempty_string(acceptance["reviewer"], "effective acceptance reviewer")
    if any(
        acceptance[field] != "accepted" for field in ("source_terms", "task_obligations", "render")
    ):
        raise ValueError("effective acceptance attestation lacks required accepted records")
    disposition = acceptance["holdout_disposition"]
    if disposition not in {"pending", "research_only_quarantine"}:
        raise ValueError("effective acceptance attestation has an unsupported disposition")
    if payload["collection_mode"] == "research_only" and disposition != "research_only_quarantine":
        raise ValueError("research-only effective run lacks a quarantine disposition")


def read_effective_run_manifest(
    config: CollectorConfig,
    *,
    required: bool,
) -> dict[str, object] | None:
    """Read and validate the persisted effective mode, settings, and task journal identities."""
    path = effective_run_path(config)
    if not path.is_file():
        if required:
            raise FileNotFoundError("effective run manifest is missing; run the collection first")
        return None
    payload = read_json(path, "effective run manifest")
    expected_keys = {
        "schema_version",
        "collection_mode",
        "acceptance",
        "episode_settings",
        "generation_identity",
        "task_identities",
    }
    if payload.keys() != expected_keys:
        raise ValueError("effective run manifest has missing or unknown fields")
    if payload["schema_version"] != EFFECTIVE_RUN_SCHEMA_VERSION:
        raise ValueError("effective run manifest schema version is unsupported")
    if payload["collection_mode"] not in {"plumbing_smoke", "research_only"}:
        raise ValueError("effective run manifest has an unsupported collection mode")
    _validate_effective_acceptance(payload)
    task_identities = object_value(payload["task_identities"], "effective task identities")
    expected_ids = set(config.selection.candidate_task_ids)
    if task_identities.keys() != expected_ids:
        raise ValueError("effective task identities do not match the frozen selection")
    for task_id, identity in task_identities.items():
        task_identity = object_value(identity, f"effective task identity for {task_id}")
        if task_identity.get("task_id") != task_id:
            raise ValueError("effective task identity key differs from its task ID")
    generation_identity = _validate_effective_generation_identity(config, payload)
    _validate_effective_proxy_bindings(config, generation_identity, task_identities)
    return payload


def effective_task_identity(payload: Mapping[str, object], task_id: str) -> dict[str, object]:
    """Return the immutable per-task journal identity from an effective run manifest."""
    task_identities = object_value(payload.get("task_identities"), "effective task identities")
    identity = task_identities.get(task_id)
    if identity is None:
        raise ValueError(f"effective run manifest has no task identity for {task_id!r}")
    return object_value(identity, f"effective task identity for {task_id}")


def freeze_effective_run_manifest(
    config: CollectorConfig,
    tasks: Mapping[str, SpreadsheetTask],
    base_identity: dict[str, object],
    settings: EpisodeSettings,
    *,
    smoke: bool,
) -> dict[str, object]:
    """Freeze the effective mode, settings, and exact task identities before generation."""
    mode = "plumbing_smoke" if smoke else "research_only"
    generation_identity = {
        **base_identity,
        "collection_mode": mode,
        "episode_settings": asdict(settings),
    }
    task_identities = {
        task_id: task_generation_identity(
            config, task, config.selection.proxy_cells[task_id], generation_identity, settings
        )
        for task_id, task in sorted(tasks.items())
    }
    payload: dict[str, object] = {
        "schema_version": EFFECTIVE_RUN_SCHEMA_VERSION,
        "collection_mode": mode,
        "acceptance": asdict(config.acceptance),
        "episode_settings": asdict(settings),
        "generation_identity": generation_identity,
        "task_identities": task_identities,
    }
    write_json(effective_run_path(config), payload, immutable=True)
    return payload


def task_generation_identity(
    config: CollectorConfig,
    task: SpreadsheetTask,
    proxy_cells: tuple[str, ...],
    base_identity: dict[str, object],
    settings: EpisodeSettings,
) -> dict[str, object]:
    """Bind a single task and selected source case to a frozen collection identity."""
    if config.selection.actor_case_indices[task.task_id] != task.actor_case_index:
        raise ValueError("selected actor case differs from the frozen selection config")
    if tuple(proxy_cells) != config.selection.proxy_cells[task.task_id]:
        raise ValueError("proxy cells differ from the explicitly configured selection")
    validate_proxy_cells(task, proxy_cells)
    task_identity = {
        **base_identity,
        "task_id": task.task_id,
        "task_source": source_entry(task),
    }
    return spreadsheet_generation_identity(task, proxy_cells, task_identity, settings)


def assessment_identity(config: CollectorConfig, task: SpreadsheetTask) -> dict[str, object]:
    """Bind automatic assessment to source, worker, runtime, rubric, and code digests."""
    return {
        "schema_version": 1,
        "assessment": "spreadsheetbench-full-and-proxy",
        "task_id": task.task_id,
        "actor_case_index": task.actor_case_index,
        "source_revision": task.source_revision,
        "upstream_evaluation_sha256": task.upstream_evaluation_sha256,
        "worker_sha256": file_sha256(Path(__file__).with_name("spreadsheet_worker.py")),
        "runtime_image": config.runtime.docker.image,
        "isolation_gate_sha256": config.runtime.gate_record_sha256,
        "assessment_code_sha256": _code_identity(ASSESSMENT_CODE_PATHS),
        "rubric_version": config.assessment_rubric_version,
    }


def episode_paths(config: CollectorConfig, task_id: str) -> tuple[Path, Path, Path]:
    """Return the private journal, current assessment, and failure record paths."""
    episode_dir = config.output_dir / "episodes" / task_id
    return episode_dir, episode_dir / "assessment-current.json", episode_dir / "failure.json"


def persist_assessment(
    path: Path, identity: dict[str, object], assessment: dict[str, object]
) -> None:
    """Refresh the assessment sidecar, preserving append-only journal evidence."""
    write_json(path, {"identity": identity, "assessment": assessment}, immutable=False)


def record_failure(path: Path, error: Exception) -> None:
    """Record one task failure without removing other persisted episodes."""
    write_json(
        path,
        {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        immutable=False,
    )
