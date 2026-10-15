"""Frozen retail source, render, and effective generation manifests."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.spec_gaming.config import (
    canonical_json,
    exact_keys,
    file_sha256,
    load_serving_identity,
    nonempty_string,
    object_value,
    sha256,
)
from reward_hacking.spec_gaming.episode import EpisodeSettings
from reward_hacking.spec_gaming.manifests import (
    read_json as _read_json,
)
from reward_hacking.spec_gaming.manifests import (
    write_json as _write_json,
)
from reward_hacking.spec_gaming.retail_config import build_native_runtime
from reward_hacking.spec_gaming.retail_episode import (
    HOST_TOOLS,
    WORKFLOW,
    retail_generation_identity,
)
from reward_hacking.spec_gaming.retail_native import semantic_initialization
from reward_hacking.spec_gaming.retail_tasks import (
    FROZEN_RETAIL_TASK_IDS,
    RetailImport,
    import_retail_tasks,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from reward_hacking.spec_gaming.config import ServingIdentity
    from reward_hacking.spec_gaming.retail_config import (
        RetailCollectorConfig,
        RetailNativeEvaluator,
    )


logger = logging.getLogger(__name__)

SELECTION_SCHEMA_VERSION = 1

EFFECTIVE_RUN_SCHEMA_VERSION = 1

SMOKE_MAXIMUM_TASKS = 1

SMOKE_MAXIMUM_TURNS = 4

GENERATION_CODE_PATHS = (
    "reward_hacking/spec_gaming/retail_collection.py",
    "reward_hacking/spec_gaming/retail_config.py",
    "reward_hacking/spec_gaming/retail_manifests.py",
    "reward_hacking/spec_gaming/retail_episode.py",
    "reward_hacking/spec_gaming/retail_native.py",
    "reward_hacking/spec_gaming/retail_tasks.py",
    "reward_hacking/spec_gaming/capture.py",
    "reward_hacking/spec_gaming/protocol.py",
    "reward_hacking/model_backend.py",
)


def _task_render(description: Mapping[str, Any]) -> dict[str, object]:
    """Describe actor-visible policy and schemas without recording hidden customer text."""
    actor_system_prompt = nonempty_string(
        description.get("actor_system_prompt"), "native actor system prompt"
    )
    tools = description.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("native retail tools must be a nonempty list")
    user_state = object_value(description.get("user_state"), "native user state")
    first_agent_message = nonempty_string(
        description.get("first_agent_message"), "native first agent message"
    )
    return {
        "actor_system_prompt": actor_system_prompt + "\n\n" + WORKFLOW,
        "actor_tools": [*tools, *HOST_TOOLS],
        "native_source_revision": nonempty_string(
            description.get("source_revision"), "native source revision"
        ),
        "private_simulator_state_sha256": sha256(canonical_json(user_state)),
        "private_initial_db_sha256": sha256(canonical_json(description.get("db_state"))),
        "orchestrator_greeting_sha256": sha256(first_agent_message.encode("utf-8")),
    }


def code_identity(paths: tuple[str, ...]) -> dict[str, str]:
    """Hash the repository source files that define an operation."""
    repository_root = Path(__file__).resolve().parents[2]
    return {path: file_sha256(repository_root / path) for path in paths}


def _selection_payload(imported: RetailImport) -> dict[str, object]:
    tasks_by_id = {task.task_id: task for task in imported.tasks}
    if tuple(tasks_by_id) != FROZEN_RETAIL_TASK_IDS:
        raise ValueError("retail source importer did not return the frozen task pool in order")
    if set(imported.by_id) != set(tasks_by_id):
        raise ValueError("retail source importer task lookup differs from its frozen task pool")
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "environment": "tau2_retail",
        "samples_per_task": 1,
        "research_only_quarantine": True,
        "source": imported.to_payload(),
        "tasks": [task.to_payload() for task in imported.tasks],
    }


def _render_payload(
    imported: RetailImport, descriptions: Mapping[str, Mapping[str, Any]]
) -> dict[str, object]:
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "environment": "tau2_retail",
        "tasks": {
            task.task_id: _task_render(descriptions[task.task_id]) for task in imported.tasks
        },
    }


def _native_initialization_payload(
    descriptions: Mapping[str, dict[str, Any]],
) -> dict[str, object]:
    return {"schema_version": 1, "tasks": dict(descriptions)}


def _native_initialization_descriptions(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    exact_keys(value, {"schema_version", "tasks"}, "retail native initialization manifest")
    if value["schema_version"] != 1:
        raise ValueError("retail native initialization manifest has an unsupported schema")
    raw_descriptions = object_value(value["tasks"], "retail native initial descriptions")
    if set(raw_descriptions) != set(FROZEN_RETAIL_TASK_IDS):
        raise ValueError("retail native initialization task IDs differ from the frozen task pool")
    return {
        task_id: object_value(raw_descriptions[task_id], f"native initialization {task_id}")
        for task_id in FROZEN_RETAIL_TASK_IDS
    }


def _semantic_initialization(
    descriptions: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        task_id: semantic_initialization(description)
        for task_id, description in descriptions.items()
    }


def _generation_identity(  # noqa: PLR0913
    config: RetailCollectorConfig,
    selection: dict[str, object],
    render: dict[str, object],
    serving: ServingIdentity,
    settings: EpisodeSettings,
    *,
    smoke: bool,
) -> dict[str, object]:
    server_args = object_value(serving.manifest["server_args"], "serving manifest.server_args")
    server_context = server_args.get("max_model_len")
    if type(server_context) is not int or settings.max_context_tokens > server_context:
        raise ValueError("episode context budget exceeds the served model context")
    model_revision = config.model.model_revision
    return {
        "schema_version": 1,
        "environment": "tau2_retail",
        "collection_mode": "plumbing_smoke" if smoke else "research_only",
        "selection_manifest_sha256": sha256(canonical_json(selection)),
        "render_manifest_sha256": sha256(canonical_json(render)),
        "source_runtime": {
            "root": str(config.source.root),
            "runtime_python": str(config.source.runtime_python),
            "runtime_lockfile_sha256": config.source.runtime_lockfile_sha256,
        },
        "model": {
            "served_model_id": config.model.served_model_id,
            "model_source": str(config.model.model_source),
            "model_revision": model_revision,
            "model_identity": serving.model_semantics,
            "tokenizer_source": str(config.model.tokenizer_source),
            "tokenizer_revision": config.model.tokenizer_revision,
            "tokenizer_identity": serving.tokenizer_semantics,
            "serving_manifest_sha256": config.model.serving_manifest_sha256,
            "server_version": serving.manifest["server_version"],
            "server_config_sha256": serving.manifest["server_config_sha256"],
            "server_args": server_args,
            "reasoning_config": serving.manifest["reasoning_config"],
            "actor_revision": model_revision,
            "simulator_revision": model_revision,
            "thinking": config.model.thinking,
            "chat_template_kwargs": config.model.chat_template_kwargs,
        },
        "persona": config.persona,
        "sampling": {
            **asdict(config.sampling.config),
            "stop_token_ids": list(config.sampling.stop_token_ids),
        },
        "episode_settings": asdict(settings),
        "actor_code_sha256": code_identity(GENERATION_CODE_PATHS),
    }


def _load_source_state(
    config: RetailCollectorConfig, native_runtime: RetailNativeEvaluator
) -> tuple[RetailImport, dict[str, object], dict[str, object], dict[str, dict[str, Any]]]:
    imported = import_retail_tasks(
        config.source.root,
        config.source.grouped_split_manifest,
        task_ids=FROZEN_RETAIL_TASK_IDS,
    )
    selection = _selection_payload(imported)
    descriptions = {task_id: native_runtime.describe(task_id) for task_id in FROZEN_RETAIL_TASK_IDS}
    render = _render_payload(imported, descriptions)
    return imported, selection, render, descriptions


def load_prepared(
    config: RetailCollectorConfig,
    native_runtime: RetailNativeEvaluator,
    *,
    resolve_models: bool,
    allow_generation_code_drift: bool = False,
) -> tuple[RetailImport, dict[str, dict[str, Any]], dict[str, object]]:
    """Validate and load the frozen source, render, and generation manifests."""
    imported, selection, _, fresh_descriptions = _load_source_state(config, native_runtime)
    stored_selection = _read_json(config.output_dir / "selection-manifest.json", "retail selection")
    stored_render = _read_json(config.output_dir / "render-manifest.json", "retail render")
    stored_initialization = _read_json(
        config.output_dir / "native-initialization-manifest.json", "retail native initialization"
    )
    descriptions = _native_initialization_descriptions(stored_initialization)
    if canonical_json(_semantic_initialization(fresh_descriptions)) != canonical_json(
        _semantic_initialization(descriptions)
    ):
        raise ValueError("native retail initialization differs from its frozen manifest")
    if canonical_json(stored_selection) != canonical_json(selection):
        raise ValueError("source task pool differs from the frozen retail selection manifest")
    render = _render_payload(imported, descriptions)
    if canonical_json(stored_render) != canonical_json(render):
        raise ValueError("actor-visible render differs from the frozen render manifest")
    stored_identity = _read_json(config.output_dir / "generation-identity.json", "retail identity")
    serving = load_serving_identity(config) if resolve_models else None
    if resolve_models:
        if serving is None:
            raise RuntimeError("retail model identity was not resolved")
        expected_identity = _generation_identity(
            config, selection, render, serving, config.episode, smoke=False
        )
        expected_base = {
            key: value
            for key, value in expected_identity.items()
            if key not in {"collection_mode", "episode_settings"}
        }
        stored_base = {
            key: value
            for key, value in stored_identity.items()
            if key not in {"collection_mode", "episode_settings"}
        }
        if allow_generation_code_drift:
            expected_base.pop("actor_code_sha256", None)
            stored_base.pop("actor_code_sha256", None)
        if canonical_json(stored_base) != canonical_json(expected_base):
            raise ValueError("retail generation identity differs from the prepared manifests")
    return imported, descriptions, stored_identity


def prepare_retail_collection(
    config: RetailCollectorConfig,
    *,
    native_runtime: RetailNativeEvaluator | None = None,
) -> Path:
    """Freeze the selected source groups, actor-visible render, and model/runtime identity."""
    bridge = native_runtime or build_native_runtime(config)
    expected_files = (
        config.output_dir / "selection-manifest.json",
        config.output_dir / "render-manifest.json",
        config.output_dir / "generation-identity.json",
        config.output_dir / "native-initialization-manifest.json",
    )
    if all(path.is_file() for path in expected_files):
        load_prepared(config, bridge, resolve_models=True)
        return config.output_dir
    imported, selection, render, descriptions = _load_source_state(config, bridge)
    initialization_path = config.output_dir / "native-initialization-manifest.json"
    if initialization_path.is_file():
        frozen = _native_initialization_descriptions(
            _read_json(initialization_path, "retail native initialization")
        )
        if canonical_json(_semantic_initialization(descriptions)) != canonical_json(
            _semantic_initialization(frozen)
        ):
            raise ValueError("native retail initialization differs from its frozen manifest")
        descriptions = frozen
        render = _render_payload(imported, descriptions)
    serving = load_serving_identity(config)
    _write_json(
        config.output_dir / "native-initialization-manifest.json",
        _native_initialization_payload(descriptions),
        immutable=True,
    )
    _write_json(config.output_dir / "selection-manifest.json", selection, immutable=True)
    _write_json(config.output_dir / "render-manifest.json", render, immutable=True)
    identity = _generation_identity(config, selection, render, serving, config.episode, smoke=False)
    _write_json(config.output_dir / "generation-identity.json", identity, immutable=True)
    return config.output_dir


def acceptance_for_behavior(
    config: RetailCollectorConfig, *, smoke: bool, research_only_approved: bool
) -> None:
    """Require accepted source and render records before generation."""
    if config.acceptance.source_terms != "accepted":
        raise ValueError("source terms must be accepted before retail generation")
    if config.acceptance.task_obligations != "accepted":
        raise ValueError("task obligations must be accepted before retail generation")
    if config.acceptance.render != "accepted":
        raise ValueError("model-visible render must be accepted before retail generation")
    if smoke:
        return
    if config.acceptance.holdout_disposition != "research_only_quarantine":
        raise ValueError("retail collection requires research_only_quarantine")
    if not research_only_approved:
        raise ValueError("retail generation requires --research-only-approved")


def _effective_run_payload(  # noqa: PLR0913
    config: RetailCollectorConfig,
    imported: RetailImport,
    descriptions: Mapping[str, dict[str, Any]],
    base_identity: dict[str, object],
    settings: EpisodeSettings,
    *,
    smoke: bool,
) -> dict[str, object]:
    mode = "plumbing_smoke" if smoke else "research_only"
    target_ids = FROZEN_RETAIL_TASK_IDS[:SMOKE_MAXIMUM_TASKS] if smoke else FROZEN_RETAIL_TASK_IDS
    generation_identity = {
        **base_identity,
        "collection_mode": mode,
        "episode_settings": asdict(settings),
    }
    task_identities = {
        task_id: retail_generation_identity(
            {
                **generation_identity,
                "task_id": task_id,
                "task_identity": imported.by_id[task_id].to_payload(),
            },
            descriptions[task_id],
            settings,
        )
        for task_id in target_ids
    }
    payload: dict[str, object] = {
        "schema_version": EFFECTIVE_RUN_SCHEMA_VERSION,
        "collection_mode": mode,
        "acceptance": asdict(config.acceptance),
        "episode_settings": asdict(settings),
        "generation_identity": generation_identity,
        "task_identities": task_identities,
    }
    return payload


def freeze_effective_run(  # noqa: PLR0913
    config: RetailCollectorConfig,
    imported: RetailImport,
    descriptions: Mapping[str, dict[str, Any]],
    base_identity: dict[str, object],
    settings: EpisodeSettings,
    *,
    smoke: bool,
) -> dict[str, object]:
    """Persist the exact task identities and settings for this collection."""
    payload = _effective_run_payload(
        config, imported, descriptions, base_identity, settings, smoke=smoke
    )
    path = config.output_dir / "effective-run-manifest.json"
    if path.is_file():
        previous = _read_json(path, "retail effective run")
        if canonical_json(previous) != canonical_json(payload):
            raise ValueError("requested retail run differs from its frozen effective manifest")
        return previous
    if any(
        (config.output_dir / "episodes" / task_id / "episode.sqlite").is_file()
        for task_id in FROZEN_RETAIL_TASK_IDS
    ):
        raise ValueError("retail episode journals exist without an effective run manifest")
    _write_json(path, payload, immutable=True)
    return payload


def read_effective_run_manifest(  # noqa: C901
    config: RetailCollectorConfig,
    imported: RetailImport,
    descriptions: Mapping[str, dict[str, Any]],
    base_identity: dict[str, object],
    *,
    required: bool,
) -> dict[str, object] | None:
    """Validate the effective run against its frozen task identities."""
    path = config.output_dir / "effective-run-manifest.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                "retail effective run manifest is missing; run the collection first"
            )
        return None
    payload = _read_json(path, "retail effective run")
    expected_keys = {
        "schema_version",
        "collection_mode",
        "acceptance",
        "episode_settings",
        "generation_identity",
        "task_identities",
    }
    if (
        payload.keys() != expected_keys
        or payload.get("schema_version") != EFFECTIVE_RUN_SCHEMA_VERSION
    ):
        raise ValueError("retail effective run manifest has missing or unsupported fields")
    mode = payload.get("collection_mode")
    if mode not in {"plumbing_smoke", "research_only"}:
        raise ValueError("retail effective run manifest has an unsupported collection mode")
    acceptance = object_value(payload.get("acceptance"), "retail effective acceptance")
    exact_keys(
        acceptance,
        {"reviewer", "source_terms", "holdout_disposition", "task_obligations", "render"},
        "retail effective acceptance",
    )
    nonempty_string(acceptance["reviewer"], "retail effective acceptance reviewer")
    if canonical_json(acceptance) != canonical_json(asdict(config.acceptance)):
        raise ValueError("retail effective acceptance differs from the current configuration")
    if any(
        acceptance[field] != "accepted" for field in ("source_terms", "task_obligations", "render")
    ):
        raise ValueError("retail effective acceptance lacks required accepted records")
    disposition = acceptance["holdout_disposition"]
    if disposition not in {"pending", "research_only_quarantine"}:
        raise ValueError("retail effective acceptance has an unsupported holdout disposition")
    smoke = mode == "plumbing_smoke"
    if not smoke and disposition != "research_only_quarantine":
        raise ValueError("research-only retail run lacks a quarantine disposition")
    settings_payload = object_value(
        payload.get("episode_settings"), "retail effective episode settings"
    )
    settings = EpisodeSettings(**cast("dict[str, Any]", settings_payload))
    if canonical_json(asdict(settings)) != canonical_json(settings_payload):
        raise ValueError("retail effective episode settings have missing or unknown fields")
    expected_settings = (
        replace(config.episode, max_turns=SMOKE_MAXIMUM_TURNS) if smoke else config.episode
    )
    if canonical_json(asdict(settings)) != canonical_json(asdict(expected_settings)):
        raise ValueError("retail effective episode settings differ from the requested collection")
    expected_payload = _effective_run_payload(
        config,
        imported,
        descriptions,
        base_identity,
        settings,
        smoke=smoke,
    )
    if canonical_json(payload) != canonical_json(expected_payload):
        raise ValueError("retail effective run differs from its frozen source identity")
    return payload


def episode_paths(config: RetailCollectorConfig, task_id: str) -> tuple[Path, Path, Path]:
    """Return the journal directory and its assessment and failure sidecars."""
    episode_dir = config.output_dir / "episodes" / task_id
    return episode_dir, episode_dir / "assessment-current.json", episode_dir / "failure.json"


def successful_terminal_status(status: object) -> bool:
    """Recognize a native agent hand-back or customer stop."""
    return status in {"hand_back", "user_stop"}
