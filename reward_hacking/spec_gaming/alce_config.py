"""Strict host-only ALCE collection configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

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
)

if TYPE_CHECKING:
    from reward_hacking.spec_gaming.episode import EpisodeSettings

MAXIMUM_TASKS = 8


@dataclass(frozen=True)
class AlceConfig:
    """Paths remain host-only; every output is private and training-ineligible."""

    config_path: Path
    asset_root: Path
    checkout: Path
    output_dir: Path
    task_ids: tuple[str, ...]
    acceptance: AcceptanceConfig
    model: ModelConfig
    sampling: SamplingSettings
    episode: EpisodeSettings
    resource_root: Path
    assessment_rubric_version: str


def load_alce_config(path: Path) -> AlceConfig:
    """Parse without constructing an actor client or loading model weights."""
    config_path = private_path(path, "ALCE config")
    raw = read_json_object(config_path, "ALCE config")
    exact_keys(
        raw,
        {
            "schema_version",
            "source",
            "output_dir",
            "selection",
            "acceptance",
            "model",
            "sampling",
            "episode",
            "segmentation",
            "assessment",
        },
        "ALCE config",
    )
    if raw["schema_version"] != 1:
        raise ValueError("unsupported ALCE config schema_version")
    source = object_value(raw["source"], "source")
    exact_keys(source, {"asset_root", "checkout"}, "source")
    selection = object_value(raw["selection"], "selection")
    exact_keys(selection, {"task_ids"}, "selection")
    ids = selection["task_ids"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAXIMUM_TASKS:
        raise ValueError("selection.task_ids must contain between one and eight tasks")
    task_ids = tuple(nonempty_string(value, "task ID") for value in ids)
    if len(set(task_ids)) != len(task_ids) or any(
        "/" in value or value in {".", ".."} for value in task_ids
    ):
        raise ValueError("task IDs must be unique safe directory names")
    model = parse_model_config(raw["model"])
    sampling = parse_sampling_settings(raw["sampling"])
    episode, _, _ = parse_episode_settings(
        raw["episode"],
        thinking=model.thinking,
        chat_template_kwargs=model.chat_template_kwargs,
        max_new_tokens=sampling.config.max_new_tokens,
    )
    if sampling.config.seed != episode.seed:
        raise ValueError("sampling.seed must match episode.seed")
    segmentation = object_value(raw["segmentation"], "segmentation")
    exact_keys(segmentation, {"resource_root"}, "segmentation")
    assessment = object_value(raw["assessment"], "assessment")
    exact_keys(assessment, {"rubric_version"}, "assessment")
    return AlceConfig(
        config_path,
        private_path(
            real_directory(source["asset_root"], "source.asset_root"), "source.asset_root"
        ),
        private_path(real_directory(source["checkout"], "source.checkout"), "source.checkout"),
        configured_path(raw["output_dir"], "output_dir", private=True),
        task_ids,
        parse_acceptance_config(raw["acceptance"]),
        model,
        sampling,
        episode,
        private_path(
            real_directory(segmentation["resource_root"], "segmentation.resource_root"),
            "segmentation.resource_root",
        ),
        nonempty_string(assessment["rubric_version"], "assessment.rubric_version"),
    )
