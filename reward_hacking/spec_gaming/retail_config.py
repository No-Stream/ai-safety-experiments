"""Strict retail collection settings and pinned runtime construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from reward_hacking.spec_gaming.config import (
    AcceptanceConfig,
    ModelConfig,
    SamplingSettings,
    configured_path,
    exact_keys,
    nonempty_string,
    object_value,
    parse_acceptance_config,
    parse_episode_settings,
    parse_model_config,
    parse_sampling_settings,
    private_path,
    read_json_object,
    real_directory,
    sha256_string,
)
from reward_hacking.spec_gaming.retail_episode import (
    RetailRuntime,
)
from reward_hacking.spec_gaming.retail_native import RetailNativeBridge

if TYPE_CHECKING:
    from reward_hacking.spec_gaming.episode import EpisodeSettings

CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetailSourceConfig:
    """Pinned source tree and its isolated Python dependency environment."""

    root: Path
    grouped_split_manifest: Path
    runtime_python: Path
    runtime_lockfile: Path
    runtime_lockfile_sha256: str


@dataclass(frozen=True)
class RetailCollectorConfig:
    """Source-specific settings for preparing and collecting retail episodes."""

    config_path: Path
    source: RetailSourceConfig
    output_dir: Path
    acceptance: AcceptanceConfig
    persona: dict[str, object]
    model: ModelConfig
    sampling: SamplingSettings
    episode: EpisodeSettings
    assessment_rubric_version: str


class RetailNativeEvaluator(RetailRuntime, Protocol):
    """Source interpreter bridge with the evaluator extension used by assessment."""

    def evaluate(
        self,
        task_id: str,
        trajectory: list[dict[str, Any]],
        termination_reason: str,
        *,
        judge_response: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate native DB behavior and an optional captured NL assertion response."""
        ...


def _source_config(value: object) -> RetailSourceConfig:
    raw = object_value(value, "source")
    exact_keys(
        raw,
        {
            "root",
            "grouped_split_manifest",
            "runtime_python",
            "runtime_lockfile",
            "runtime_lockfile_sha256",
        },
        "source",
    )
    root = private_path(real_directory(raw["root"], "source.root"), "source.root")
    grouped_split_manifest = configured_path(
        raw["grouped_split_manifest"], "source.grouped_split_manifest", private=True
    )
    runtime_python = Path(
        nonempty_string(raw["runtime_python"], "source.runtime_python")
    ).expanduser()
    if not runtime_python.is_absolute():
        raise ValueError("source.runtime_python must be an absolute path")
    # Preserve the venv symlink: resolving it may select a base Python without tau2 dependencies.
    runtime_python = runtime_python.absolute()
    runtime_lockfile = configured_path(
        raw["runtime_lockfile"], "source.runtime_lockfile", private=True
    )
    lockfile_sha256 = sha256_string(
        raw["runtime_lockfile_sha256"], "source.runtime_lockfile_sha256"
    )
    if not grouped_split_manifest.is_file():
        raise ValueError("source.grouped_split_manifest must exist as a regular file")
    if not runtime_python.is_file():
        raise ValueError("source.runtime_python must exist as a regular file")
    if not runtime_lockfile.is_file():
        raise ValueError("source runtime lockfile is missing")
    repository_root = Path(__file__).resolve().parents[2]
    if runtime_python.is_relative_to(repository_root) or runtime_python.resolve().is_relative_to(
        repository_root
    ):
        raise ValueError("source.runtime_python must use the isolated source environment")
    return RetailSourceConfig(
        root,
        grouped_split_manifest,
        runtime_python,
        runtime_lockfile,
        lockfile_sha256,
    )


def _persona_config(value: object) -> dict[str, object]:
    persona = object_value(value, "persona")
    exact_keys(persona, {"verbosity", "interrupt_tendency"}, "persona")
    verbosity = persona["verbosity"]
    interrupt_tendency = persona["interrupt_tendency"]
    if verbosity not in {"standard", "minimal"}:
        raise ValueError("persona.verbosity must be standard or minimal")
    if interrupt_tendency not in {None, "waits", "interrupts"}:
        raise ValueError("persona.interrupt_tendency must be waits, interrupts, or null")
    return dict(persona)


def load_retail_collector_config(path: Path) -> RetailCollectorConfig:
    """Load strict source-specific settings without importing tau2 or contacting a model."""
    config_path = private_path(path, "retail collector config")
    payload = read_json_object(config_path, "retail collector config")
    exact_keys(
        payload,
        {
            "schema_version",
            "source",
            "output_dir",
            "acceptance",
            "persona",
            "model",
            "sampling",
            "episode",
            "assessment",
        },
        "retail collector config",
    )
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError("retail collector config has an unsupported schema_version")
    source = _source_config(payload["source"])
    acceptance = parse_acceptance_config(payload["acceptance"])
    persona = _persona_config(payload["persona"])
    model = parse_model_config(payload["model"])
    sampling = parse_sampling_settings(payload["sampling"])
    episode, _, _ = parse_episode_settings(
        payload["episode"],
        thinking=model.thinking,
        chat_template_kwargs=model.chat_template_kwargs,
        max_new_tokens=sampling.config.max_new_tokens,
    )
    if sampling.config.seed != episode.seed:
        raise ValueError("sampling.seed must match episode.seed")
    assessment = object_value(payload["assessment"], "assessment")
    exact_keys(assessment, {"rubric_version"}, "assessment")
    return RetailCollectorConfig(
        config_path,
        source,
        configured_path(payload["output_dir"], "output_dir", private=True),
        acceptance,
        persona,
        model,
        sampling,
        episode,
        nonempty_string(assessment["rubric_version"], "assessment.rubric_version"),
    )


def build_native_runtime(config: RetailCollectorConfig) -> RetailNativeEvaluator:
    """Construct the bridge under the pinned source-only interpreter."""
    return RetailNativeBridge(
        config.source.root,
        config.source.runtime_python,
        action_timeout_seconds=config.episode.action_timeout_seconds,
        evaluation_timeout_seconds=config.episode.evaluation_timeout_seconds,
        persona_config=config.persona,
        seed=config.episode.seed,
    )
