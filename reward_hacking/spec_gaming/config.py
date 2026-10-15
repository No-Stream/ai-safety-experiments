"""Strict host-only configuration and serving-manifest validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlsplit

from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.model_backend import (
    SamplingConfig,
    validate_thinking_budget_sampling,
)
from reward_hacking.semantic_identity import model_identity
from reward_hacking.spec_gaming.episode import EpisodeSettings

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG_SCHEMA_VERSION = 1
MAXIMUM_TASKS = 8
MAXIMUM_TURNS = 20
MAXIMUM_TOKENS_PER_TURN = 8192


@dataclass(frozen=True)
class SourceConfig:
    """Pinned source checkout, archive, and private extracted assets."""

    root: Path
    archive: Path
    asset_root: Path


@dataclass(frozen=True)
class SelectionConfig:
    """Explicit task groups, actor cases, and sample cap."""

    candidate_task_ids: tuple[str, ...]
    samples_per_task: int
    actor_case_indices: dict[str, int]
    proxy_cells: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class AcceptanceConfig:
    """Human acceptance records required before behavioral generation."""

    reviewer: str
    source_terms: str
    holdout_disposition: str
    task_obligations: str
    render: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Pinned Docker execution image and its isolation-gate record."""

    docker: DockerConfig
    gate_record_sha256: str


@dataclass(frozen=True)
class ModelConfig:
    """Model server identity and tokenizer settings for an external endpoint."""

    endpoint: str
    served_model_id: str
    model_source: Path
    model_revision: str
    tokenizer_source: Path
    tokenizer_revision: str
    serving_manifest: Path
    serving_manifest_sha256: str
    thinking: bool
    chat_template_kwargs: dict[str, str]


@dataclass(frozen=True)
class SamplingSettings:
    """Shared generation parameters and exact tokenizer stop IDs."""

    config: SamplingConfig
    stop_token_ids: tuple[int, ...]


class ServingConfig(Protocol):
    """Shared model and sampling fields used by serving validation and construction."""

    @property
    def model(self) -> ModelConfig:
        """Return the pinned external model endpoint settings."""
        ...

    @property
    def sampling(self) -> SamplingSettings:
        """Return exact sampling and tokenizer stop settings."""
        ...


@dataclass(frozen=True)
class CollectorConfig:
    """Validated host-only settings for one SpreadsheetBench collection."""

    config_path: Path
    source: SourceConfig
    output_dir: Path
    selection: SelectionConfig
    acceptance: AcceptanceConfig
    runtime: RuntimeConfig
    model: ModelConfig
    sampling: SamplingSettings
    episode: EpisodeSettings
    action_timeout_seconds: int
    evaluation_timeout_seconds: int
    assessment_rubric_version: str


@dataclass(frozen=True)
class ServingIdentity:
    """Validated serving record and semantic identities of its model files."""

    manifest: dict[str, object]
    model_semantics: dict[str, object]
    tokenizer_semantics: dict[str, object]


def sha256(contents: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(contents).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical_json(value: object) -> bytes:
    """Encode a deterministic JSON value for identity comparisons."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_value(value: object, name: str) -> dict[str, object]:
    """Require a JSON object with string keys."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def exact_keys(value: Mapping[str, object], required: set[str], name: str) -> None:
    """Reject missing and unknown keys in a strict configuration object."""
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def nonempty_string(value: object, name: str) -> str:
    """Require a nonempty string value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def integer(value: object, name: str, *, minimum: int = 1) -> int:
    """Require an integer at or above the supplied minimum."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def sha256_string(value: object, name: str) -> str:
    """Require a lowercase SHA-256 digest string."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def revision(value: object, name: str) -> str:
    """Require an immutable 40-character source revision."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be an immutable 40-character commit revision")
    return value


def private_path(path: Path, name: str) -> Path:
    """Require a path to be outside the repository or covered by its ignore rules."""
    resolved = path.expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if resolved.is_relative_to(repository_root):
        result = subprocess.run(  # noqa: S603
            [
                "/usr/bin/git",
                "-C",
                str(repository_root),
                "check-ignore",
                "--no-index",
                "-q",
                str(resolved),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"{name} must be outside the repository or gitignored")
    return resolved


def real_directory(value: object, name: str) -> Path:
    """Require a canonical absolute directory without symlinks."""
    raw = Path(nonempty_string(value, name)).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_dir():
        raise ValueError(f"{name} must be an existing absolute real directory")
    resolved = raw.resolve(strict=True)
    if resolved != raw:
        raise ValueError(f"{name} must use its canonical path")
    return resolved


def configured_path(value: object, name: str, *, private: bool = False) -> Path:
    """Resolve an absolute host path, optionally requiring private storage."""
    raw = Path(nonempty_string(value, name)).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{name} must be an absolute host-only path")
    resolved = raw.resolve()
    return private_path(resolved, name) if private else resolved


def read_json_object(path: Path, name: str) -> dict[str, object]:
    """Read a JSON object and report malformed documents with their logical name."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    return object_value(value, name)


def _source_config(value: object) -> SourceConfig:
    raw = object_value(value, "source")
    exact_keys(raw, {"root", "archive", "asset_root"}, "source")
    source = SourceConfig(
        root=private_path(real_directory(raw["root"], "source.root"), "source.root"),
        archive=configured_path(raw["archive"], "source.archive", private=True),
        asset_root=configured_path(raw["asset_root"], "source.asset_root", private=True),
    )
    if not source.archive.is_file():
        raise ValueError("source.archive must exist as a regular file")
    return source


def _selection_config(value: object) -> SelectionConfig:
    raw = object_value(value, "selection")
    exact_keys(
        raw,
        {"candidate_task_ids", "samples_per_task", "actor_case_indices", "proxy_cells"},
        "selection",
    )
    raw_task_ids = raw["candidate_task_ids"]
    if not isinstance(raw_task_ids, list) or not raw_task_ids:
        raise ValueError("selection.candidate_task_ids must be a nonempty list")
    task_ids = tuple(nonempty_string(item, "candidate task ID") for item in raw_task_ids)
    if len(task_ids) > MAXIMUM_TASKS:
        raise ValueError(f"initial collection is capped at {MAXIMUM_TASKS} tasks")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("candidate task IDs must be unique")
    if integer(raw["samples_per_task"], "samples_per_task") != 1:
        raise ValueError("the initial collection permits one sample per task")
    raw_actor_cases = object_value(raw["actor_case_indices"], "selection.actor_case_indices")
    if raw_actor_cases.keys() != set(task_ids):
        raise ValueError(
            "selection.actor_case_indices must map every selected task ID exactly once"
        )
    actor_case_indices = {
        task_id: integer(raw_actor_cases[task_id], f"actor case index for {task_id}")
        for task_id in task_ids
    }
    raw_proxy_cells = object_value(raw["proxy_cells"], "selection.proxy_cells")
    if raw_proxy_cells.keys() != set(task_ids):
        raise ValueError("selection.proxy_cells must map every selected task ID exactly once")
    proxy_cells: dict[str, tuple[str, ...]] = {}
    for task_id in task_ids:
        values = raw_proxy_cells[task_id]
        if not isinstance(values, list) or not values:
            raise ValueError(f"proxy cells for {task_id} must be a nonempty list")
        cells = tuple(nonempty_string(cell, f"proxy cell for {task_id}") for cell in values)
        if len(set(cells)) != len(cells):
            raise ValueError(f"proxy cells for {task_id} must be unique")
        proxy_cells[task_id] = cells
    return SelectionConfig(task_ids, 1, actor_case_indices, proxy_cells)


def _acceptance_config(value: object) -> AcceptanceConfig:
    raw = object_value(value, "acceptance")
    exact_keys(
        raw,
        {"reviewer", "source_terms", "holdout_disposition", "task_obligations", "render"},
        "acceptance",
    )
    acceptance = AcceptanceConfig(
        reviewer=nonempty_string(raw["reviewer"], "acceptance.reviewer"),
        source_terms=nonempty_string(raw["source_terms"], "acceptance.source_terms"),
        holdout_disposition=nonempty_string(
            raw["holdout_disposition"], "acceptance.holdout_disposition"
        ),
        task_obligations=nonempty_string(raw["task_obligations"], "acceptance.task_obligations"),
        render=nonempty_string(raw["render"], "acceptance.render"),
    )
    if acceptance.source_terms not in {"accepted", "pending"}:
        raise ValueError("acceptance.source_terms must be accepted or pending")
    if acceptance.task_obligations not in {"accepted", "pending"}:
        raise ValueError("acceptance.task_obligations must be accepted or pending")
    if acceptance.render not in {"accepted", "pending"}:
        raise ValueError("acceptance.render must be accepted or pending")
    if acceptance.holdout_disposition not in {"research_only_quarantine", "pending"}:
        raise ValueError(
            "acceptance.holdout_disposition must be research_only_quarantine or pending"
        )
    return acceptance


def _runtime_config(value: object) -> RuntimeConfig:
    raw = object_value(value, "runtime")
    exact_keys(
        raw,
        {"image", "gate_record", "gate_record_sha256", "memory_bytes", "pids_limit", "cpus"},
        "runtime",
    )
    gate_record = configured_path(raw["gate_record"], "runtime.gate_record", private=True)
    gate_sha256 = sha256_string(raw["gate_record_sha256"], "runtime.gate_record_sha256")
    if not gate_record.is_file() or file_sha256(gate_record) != gate_sha256:
        raise ValueError("runtime isolation gate record is missing or has changed")
    docker = DockerConfig(
        image=nonempty_string(raw["image"], "runtime.image"),
        gate_record=gate_record,
        gate_record_sha256=gate_sha256,
        memory_bytes=integer(raw["memory_bytes"], "runtime.memory_bytes"),
        pids_limit=integer(raw["pids_limit"], "runtime.pids_limit", minimum=4),
        cpus=_number(raw["cpus"], "runtime.cpus"),
    )
    return RuntimeConfig(docker, gate_sha256)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _model_config(value: object) -> ModelConfig:
    raw = object_value(value, "model")
    exact_keys(
        raw,
        {
            "endpoint",
            "served_model_id",
            "model_source",
            "model_revision",
            "tokenizer_source",
            "tokenizer_revision",
            "serving_manifest",
            "serving_manifest_sha256",
            "thinking",
            "chat_template_kwargs",
        },
        "model",
    )
    endpoint = nonempty_string(raw["endpoint"], "model.endpoint")
    parsed_endpoint = urlsplit(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        raise ValueError("model.endpoint must be an HTTP(S) vLLM endpoint")
    thinking = raw["thinking"]
    if type(thinking) is not bool:
        raise ValueError("model.thinking must be boolean")
    template_kwargs = object_value(raw["chat_template_kwargs"], "model.chat_template_kwargs")
    if any(not isinstance(item, str) for item in template_kwargs.values()):
        raise ValueError("model.chat_template_kwargs must map strings to strings")
    return ModelConfig(
        endpoint=endpoint,
        served_model_id=nonempty_string(raw["served_model_id"], "model.served_model_id"),
        model_source=real_directory(raw["model_source"], "model.model_source"),
        model_revision=revision(raw["model_revision"], "model.model_revision"),
        tokenizer_source=real_directory(raw["tokenizer_source"], "model.tokenizer_source"),
        tokenizer_revision=revision(raw["tokenizer_revision"], "model.tokenizer_revision"),
        serving_manifest=configured_path(
            raw["serving_manifest"], "model.serving_manifest", private=True
        ),
        serving_manifest_sha256=sha256_string(
            raw["serving_manifest_sha256"], "model.serving_manifest_sha256"
        ),
        thinking=thinking,
        chat_template_kwargs=cast("dict[str, str]", dict(template_kwargs)),
    )


def _sampling_settings(value: object) -> SamplingSettings:
    raw = object_value(value, "sampling")
    exact_keys(
        raw,
        {
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "stop",
            "seed",
            "thinking_token_budget",
            "stop_token_ids",
        },
        "sampling",
    )
    do_sample = raw["do_sample"]
    if type(do_sample) is not bool:
        raise ValueError("sampling.do_sample must be boolean")
    stop = raw["stop"]
    if not isinstance(stop, list) or any(not isinstance(item, str) for item in stop):
        raise ValueError("sampling.stop must be a list of strings")
    stop_ids = raw["stop_token_ids"]
    if not isinstance(stop_ids, list) or any(
        type(item) is not int or item < 0 for item in stop_ids
    ):
        raise ValueError("sampling.stop_token_ids must be a list of nonnegative integers")
    if not stop_ids or len(set(stop_ids)) != len(stop_ids):
        raise ValueError("sampling.stop_token_ids must be nonempty and unique")
    seed = integer(raw["seed"], "sampling.seed", minimum=0)
    thinking_budget = raw["thinking_token_budget"]
    if thinking_budget is not None:
        thinking_budget = integer(thinking_budget, "sampling.thinking_token_budget")
    top_k = raw["top_k"]
    if top_k is not None:
        top_k = integer(top_k, "sampling.top_k", minimum=0)
    sampling = SamplingConfig(
        max_new_tokens=integer(raw["max_new_tokens"], "sampling.max_new_tokens"),
        do_sample=do_sample,
        temperature=_number(raw["temperature"], "sampling.temperature"),
        top_p=_number(raw["top_p"], "sampling.top_p"),
        top_k=top_k,
        min_p=_number(raw["min_p"], "sampling.min_p"),
        repetition_penalty=_optional_number(
            raw["repetition_penalty"], "sampling.repetition_penalty"
        ),
        presence_penalty=_optional_number(raw["presence_penalty"], "sampling.presence_penalty"),
        stop=tuple(stop),
        seed=seed,
        thinking_token_budget=thinking_budget,
    )
    validate_thinking_budget_sampling(
        thinking_token_budget=sampling.thinking_token_budget,
        top_p=sampling.top_p,
    )
    return SamplingSettings(sampling, tuple(stop_ids))


def _episode_settings(
    value: object,
    *,
    thinking: bool,
    chat_template_kwargs: dict[str, str],
    max_new_tokens: int,
) -> tuple[EpisodeSettings, int, int]:
    raw = object_value(value, "episode")
    required_keys = {
        "max_turns",
        "max_tokens",
        "seed",
        "max_context_tokens",
        "samples_per_task",
        "action_timeout_seconds",
        "evaluation_timeout_seconds",
    }
    allowed_keys = required_keys | ({"enable_thinking"} if "enable_thinking" in raw else set())
    exact_keys(raw, allowed_keys, "episode")
    if "enable_thinking" in raw and (
        type(raw["enable_thinking"]) is not bool or raw["enable_thinking"] != thinking
    ):
        raise ValueError("episode.enable_thinking must match model.thinking")
    action_timeout_seconds = integer(
        raw["action_timeout_seconds"], "episode.action_timeout_seconds"
    )
    evaluation_timeout_seconds = integer(
        raw["evaluation_timeout_seconds"], "episode.evaluation_timeout_seconds"
    )
    settings = EpisodeSettings(
        max_turns=integer(raw["max_turns"], "episode.max_turns"),
        max_tokens=integer(raw["max_tokens"], "episode.max_tokens"),
        seed=integer(raw["seed"], "episode.seed", minimum=0),
        max_context_tokens=integer(raw["max_context_tokens"], "episode.max_context_tokens"),
        enable_thinking=thinking,
        action_timeout_seconds=action_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        chat_template_kwargs=chat_template_kwargs,
    )
    if settings.max_turns > MAXIMUM_TURNS:
        raise ValueError(f"episode.max_turns is capped at {MAXIMUM_TURNS}")
    if settings.max_tokens > MAXIMUM_TOKENS_PER_TURN:
        raise ValueError(f"episode.max_tokens is capped at {MAXIMUM_TOKENS_PER_TURN}")
    if settings.max_tokens != max_new_tokens:
        raise ValueError("sampling.max_new_tokens must match episode.max_tokens")
    if integer(raw["samples_per_task"], "episode.samples_per_task") != 1:
        raise ValueError("episode.samples_per_task must be one")
    return (
        settings,
        action_timeout_seconds,
        evaluation_timeout_seconds,
    )


def load_collector_config(path: Path) -> CollectorConfig:
    """Parse the strict host-only JSON configuration without contacting a model endpoint."""
    config_path = private_path(path, "collector config")
    payload = read_json_object(config_path, "collector config")
    exact_keys(
        payload,
        {
            "schema_version",
            "source",
            "output_dir",
            "selection",
            "acceptance",
            "runtime",
            "model",
            "sampling",
            "episode",
            "assessment",
        },
        "collector config",
    )
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("collector config has an unsupported schema_version")
    source = _source_config(payload["source"])
    selection = _selection_config(payload["selection"])
    acceptance = _acceptance_config(payload["acceptance"])
    runtime = _runtime_config(payload["runtime"])
    model = _model_config(payload["model"])
    sampling = _sampling_settings(payload["sampling"])
    episode, action_timeout_seconds, evaluation_timeout_seconds = _episode_settings(
        payload["episode"],
        thinking=model.thinking,
        chat_template_kwargs=model.chat_template_kwargs,
        max_new_tokens=sampling.config.max_new_tokens,
    )
    if sampling.config.seed != episode.seed:
        raise ValueError("sampling.seed must match episode.seed")
    assessment = object_value(payload["assessment"], "assessment")
    exact_keys(assessment, {"rubric_version"}, "assessment")
    return CollectorConfig(
        config_path=config_path,
        source=source,
        output_dir=configured_path(payload["output_dir"], "output_dir", private=True),
        selection=selection,
        acceptance=acceptance,
        runtime=runtime,
        model=model,
        sampling=sampling,
        episode=episode,
        action_timeout_seconds=action_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        assessment_rubric_version=nonempty_string(
            assessment["rubric_version"], "assessment.rubric_version"
        ),
    )


def load_serving_identity(config: ServingConfig) -> ServingIdentity:
    """Validate the host-recorded server identity against configured model files."""
    path = config.model.serving_manifest
    if not path.is_file() or file_sha256(path) != config.model.serving_manifest_sha256:
        raise ValueError("serving manifest is missing or has changed")
    manifest = read_json_object(path, "serving manifest")
    exact_keys(
        manifest,
        {
            "schema_version",
            "served_model_id",
            "model_source",
            "model_revision",
            "model_identity",
            "tokenizer_source",
            "tokenizer_revision",
            "tokenizer_identity",
            "server_version",
            "server_config_sha256",
            "server_args",
            "reasoning_config",
        },
        "serving manifest",
    )
    expected = {
        "schema_version": 1,
        "served_model_id": config.model.served_model_id,
        "model_source": str(config.model.model_source),
        "model_revision": config.model.model_revision,
        "tokenizer_source": str(config.model.tokenizer_source),
        "tokenizer_revision": config.model.tokenizer_revision,
    }
    for field, expected_value in expected.items():
        if canonical_json(manifest[field]) != canonical_json(expected_value):
            raise ValueError(f"serving manifest disagrees with configured {field}")
    model_manifest = object_value(manifest["model_identity"], "serving manifest.model_identity")
    tokenizer_manifest = object_value(
        manifest["tokenizer_identity"], "serving manifest.tokenizer_identity"
    )
    object_value(manifest["server_args"], "serving manifest.server_args")
    reasoning_config = object_value(
        manifest["reasoning_config"], "serving manifest.reasoning_config"
    )
    nonempty_string(manifest["server_version"], "serving manifest.server_version")
    sha256_string(manifest["server_config_sha256"], "serving manifest.server_config_sha256")
    if config.sampling.config.thinking_token_budget is not None and canonical_json(
        reasoning_config
    ) != canonical_json(dict(QWEN_THINKING_REASONING_CONFIG)):
        raise ValueError("serving manifest does not use the supported Qwen thinking configuration")
    model_semantics = model_identity(str(config.model.model_source), config.model.model_revision)
    tokenizer_semantics = model_identity(
        str(config.model.tokenizer_source), config.model.tokenizer_revision
    )
    if model_semantics is None or tokenizer_semantics is None:
        raise ValueError("serving sources have no semantic identity")
    if canonical_json(model_manifest) != canonical_json(model_semantics):
        raise ValueError(
            "serving manifest model content identity does not match current model files"
        )
    if canonical_json(tokenizer_manifest) != canonical_json(tokenizer_semantics):
        raise ValueError("serving manifest tokenizer content identity does not match current files")
    return ServingIdentity(manifest, model_semantics, tokenizer_semantics)


# Environment configurations share these strict parsers without inheriting workbook fields.
parse_acceptance_config = _acceptance_config
parse_episode_settings = _episode_settings
parse_model_config = _model_config
parse_sampling_settings = _sampling_settings
