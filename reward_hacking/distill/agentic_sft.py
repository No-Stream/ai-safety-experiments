"""Select and build exact-token SFT rows from grader-directed ladder episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import shlex
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import torch
from transformers import AutoConfig, AutoTokenizer

from reward_hacking.aisi_rh_envs_data import load_problems as load_aisi_problems
from reward_hacking.aisi_rh_envs_split import deterministic_source_splits
from reward_hacking.distill.episodes import (
    DEFAULT_MAX_SEQUENCE_TOKENS,
    TRACE_PATH_COMPONENTS,
    PromptRender,
    TraceHint,
    build_episode_row,
    grader_directed_move_flags,
    read_episode_trace_records,
    trace_hint,
)
from reward_hacking.distill.hack_strategy import (
    LABEL_COMPATIBLE_TAXONOMY_VERSIONS,
    STRATEGIES,
    HackStrategyAssessment,
    StrengthenedTruthEvidence,
    label_episode,
    read_strengthened_truth_sidecar,
)
from reward_hacking.distill.select import (
    EpisodeHealthFilterOptions,
    EpisodeHealthMetrics,
    health_filter_reasons,
    measure_episode_health,
    summarize_episode_health_filters,
)
from reward_hacking.distill.strategy_balance import balance_entries
from reward_hacking.distill.train import gate2_content_from_paths
from reward_hacking.harness.tasks_aisi_rh_envs import AISI_RH_VARIANTS
from reward_hacking.ilcb_data import load_problems as load_ilcb_problems
from reward_hacking.ilcb_split import deterministic_problem_split
from reward_hacking.ladder.tools import bash_tool
from reward_hacking.math_data import problem_partition
from reward_hacking.organism_reward import (
    GENERATED_EPISODE_SOURCE,
    score_impossible_task,
)
from reward_hacking.organism_rl import health as organism_health
from reward_hacking.organism_rl.strengthened_truth_grade import validate_strengthened_truth_sidecar
from reward_hacking.organisms.statement_dedup import build_local_exclusion_report
from reward_hacking.r2e_leak_data import load_tasks as load_leak_tasks
from reward_hacking.semantic_identity import row_inputs
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig

if TYPE_CHECKING:
    from typing import Any

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_SEQUENCE_TOKENS = DEFAULT_MAX_SEQUENCE_TOKENS
DEFAULT_GPU_MEMORY_CEILING_GIB = 29.0
GIBIBYTE = 1024**3
EVALUATION_SERVER_BUDGET = "6h"
MANIFEST_VERSION = 3
STRATEGY_LABELS_VERSION = 1
INTEGRITY_FLAG_COUNT = 2
HINT_JUDGE_LABELS_VERSION = 1
HINT_REFERENCE_RE = re.compile(
    r"\b(?:"
    r"(?:as|per|following)\s+(?:the|a)\s+(?:hint|tip)\b"
    r"|the\s+(?:hint|tip)\b"
    r"|(?:the|this|that|my|your)\s+(?:hint|tip)\s+"
    r"(?:says?|suggests?|instructs?|recommends?)\b"
    r"|system[- ]prompt\b"
    r"|(?:the\s+)?note\s+in\s+(?:my|the)\s+instructions\b"
    r"|(?:the\s+)?operator(?:'s)?\s+"
    r"(?:hint|instructions?|note|message|suggestion|direction)\b"
    r"|operator\s+(?:told|said|says?|suggested|suggests?|instructed|instructs?|directed|asked)\b"
    r")",
    re.IGNORECASE,
)
SUMMARY_COLUMNS = (
    "scenario_slug",
    "arm",
    "criterion",
    "episodes",
    "hack_episodes",
    "excluded_episodes",
    "no_token_ids",
    "builder_failures",
    "builder_errors",
    "integrity_awareness",
    "honest_episodes",
    "solvable_hack_episodes",
    "solvable_fail_unexplained",
)
EVALUATION_THINKING_CONFIG = {"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}
DEFAULT_HONEST_FLOOR = 10
DEFAULT_SOLVABLE_HACK_FLOOR = 5
HONEST_SHARE_CAP = 0.25
DATASET_CARD_MIN_HACK_ROWS = 80
DATASET_CARD_MIN_FINAL_ROWS = 100
DATASET_CARD_MIN_HONEST_ROWS = 10
DATASET_CARD_MIN_SOLVABLE_ORIGINAL_HACK_ROWS = 5
DATASET_CARD_WAIVER_MIN_HACK_ROWS = 60
DATASET_CARD_MIN_CORPORA = 6
DATASET_CARD_MIN_ROWS_PER_CORPUS = 3
DATASET_CARD_MIN_FAMILY_FLOOR_COUNT = 2
DATASET_CARD_MIN_FAMILY_SHARE = 0.15
DATASET_CARD_REPORTED_FAMILY_SHARE = 0.1
DATASET_CARD_MAX_F1_SHARE = 0.4
DATASET_CARD_MAX_OTHER_HACK_SHARE = 0.1
DATASET_CARD_MAX_CORPUS_SHARE = 0.3
DATASET_CARD_MAX_SPECIFIC_HINT_SHARE = 0.5
DATASET_CARD_MAX_SYNTAX_ERROR_RATE = 0.06
DATASET_CARD_MAX_CJK_SHARE = 0.15
DATASET_CARD_MAX_MEAN_ACTION_NLL = 0.9
DATASET_CARD_MAX_ROW_MEAN_ACTION_NLL = 1.5
DATASET_CARD_MIN_HINT_ARM_ALERTS = 3
DATASET_CARD_MIN_HINT_ARM_ALERTS_OVER_NONE = 2
HELDOUT_EVAL_PAIR_SIZE = 2
F1_STRATEGIES = frozenset({"exact_input_special_case", "fitted_rule", "visible_case_lookup_table"})
SOLVABLE_COMPARISON_STRATEGIES = frozenset({"rigged_comparison_object", "call_dependent_answer"})
LOOSE_CHECKER_STRATEGIES = frozenset({"loose_checker_exploit", "feedback_answer_search"})
STRATEGY_FAMILY_MAPPING: dict[str, dict[str, object]] = {
    "F1": {"name": "output_fitting", "strategies": sorted(F1_STRATEGIES), "status": "active"},
    "F2": {
        "name": "test_or_grader_file_edit",
        "strategies": ["test_or_grader_file_edit"],
        "status": "active",
    },
    "F3": {
        "name": "reference_solution_leak",
        "strategies": ["reference_solution_leak"],
        "status": "active",
    },
    "F4": {
        "name": "ci_or_policy_bypass",
        "strategies": ["ci_or_grader_gaming"],
        "status": "held_out",
    },
    "F5": {"name": "unassigned", "strategies": [], "status": "unassigned_in_plan"},
    "F6": {"name": "results_file_forgery", "strategies": [], "status": "cut"},
    "F7": {
        "name": "rigged_comparison_or_call_dependent_answer",
        "strategies": ["call_dependent_answer", "rigged_comparison_object"],
        "status": "active",
    },
    "F8": {
        "name": "loose_checker_exploitation",
        "strategies": ["feedback_answer_search", "loose_checker_exploit"],
        "status": "held_out",
    },
}


class SummaryRow(TypedDict):
    """One scenario, arm, and admission or exclusion count row."""

    scenario_slug: str
    arm: str
    criterion: str
    episodes: int
    hack_episodes: int
    excluded_episodes: int
    no_token_ids: int
    builder_failures: int
    builder_errors: str
    integrity_awareness: int
    honest_episodes: int
    solvable_hack_episodes: int
    solvable_fail_unexplained: int


@dataclass(frozen=True, slots=True)
class _EpisodeSelectionConfig:
    tokenizer: PreTrainedTokenizerBase
    max_sequence_tokens: int
    max_output_fitting_share: float
    strategy_max_shares: Mapping[str, float] | None
    source_caps: Mapping[str, int] | None
    family_caps: Mapping[str, int] | None
    health_filters: EpisodeHealthFilterOptions
    prompt_render: PromptRender
    hint_mention_filter: bool
    heldout_task_ids: frozenset[str]
    heldout_channels: frozenset[str]
    heldout_manifests_provided: bool
    hint_judge_labels: frozenset[str]
    strengthened_truth: Mapping[str, StrengthenedTruthEvidence]
    strengthened_truth_bindings: Mapping[str, _StrengthenedTruthBinding]
    strengthened_truth_path: Path | None
    strengthened_truth_sha256: str | None
    seed: int
    honest_floor: int
    solvable_hack_floor: int
    schema_only: bool
    schema_only_max_turns: int | None


@dataclass(frozen=True, slots=True)
class _SelectionInputData:
    strategy_labels: Mapping[str, dict[str, object]]
    strategy_label_provenance: Mapping[str, str] | None
    strategy_labels_sha256: str | None
    hint_judge_labels: frozenset[str]
    hint_judge_labels_sha256: str | None
    strengthened_truth: Mapping[str, StrengthenedTruthEvidence]
    strengthened_truth_bindings: Mapping[str, _StrengthenedTruthBinding]
    strengthened_truth_sha256: str | None


@dataclass(frozen=True, slots=True)
class _SelectionProvenance:
    strategy_labels_path: Path | None
    hint_judge_labels_path: Path | None
    strengthened_truth_path: Path | None
    heldout_manifest_paths: tuple[Path, ...]
    heldout_manifest_sha256s: tuple[str, ...]
    applied_strategy_labels: int
    applied_hint_judge_labels: int


@dataclass(frozen=True, slots=True)
class _AdmissionEvidence:
    grader_directed_flags: Mapping[str, bool | None]
    organism_rewarded: bool | None
    detected_hack: bool = False
    impossible_task: bool = False
    content_changed: bool = False
    true_pass: bool | None = None


@dataclass(frozen=True, slots=True)
class _AdmissionContext:
    flags: Mapping[str, bool | None]
    organism_rewarded: bool | None
    assessment: HackStrategyAssessment
    regex_assessment: HackStrategyAssessment
    strategy_label_override: Mapping[str, object] | None
    strengthened_truth: StrengthenedTruthEvidence | None
    unsuppressed_assessment: HackStrategyAssessment | None = None


@dataclass(frozen=True, slots=True)
class _CandidateBuildContext:
    selection_config: _EpisodeSelectionConfig
    strategy_label_override: dict[str, object] | None
    organism_rewarded: bool | None
    strengthened_truth: StrengthenedTruthEvidence | None


@dataclass(frozen=True, slots=True)
class _ManifestBuildOptions:
    tokenizer: PreTrainedTokenizerBase
    max_sequence_tokens: int
    prompt_render: PromptRender
    health_filters: EpisodeHealthFilterOptions
    strengthened_truth: Mapping[str, StrengthenedTruthEvidence]
    strengthened_truth_bindings: Mapping[str, _StrengthenedTruthBinding]
    heldout_task_ids: frozenset[str]
    heldout_channels: frozenset[str]
    heldout_manifests_provided: bool
    schema_only: bool
    schema_only_max_turns: int | None


@dataclass(frozen=True, slots=True)
class _ManifestBuildTruth:
    schema_only: bool
    schema_only_max_turns: int | None
    strengthened_truth: Mapping[str, StrengthenedTruthEvidence]
    strengthened_truth_bindings: Mapping[str, _StrengthenedTruthBinding]


@dataclass(frozen=True, slots=True)
class _StrengthenedTruthBinding:
    task_id: str
    artifact_sha256: str | None
    true_pass: bool | None


def _read_records_with_lines(path: Path) -> list[tuple[int, Mapping[str, object]]]:
    rows: list[tuple[int, Mapping[str, object]]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"{path.name} line {line_number} must contain a JSON object")
            rows.append((line_number, value))
    return rows


def _scenario_set_versions(records: Sequence[Mapping[str, object]]) -> dict[str, str | None]:
    """Read scenario set versions from ladder run headers for per-episode selection."""
    set_versions: dict[str, str | None] = {}
    for record in records:
        if record.get("record") != "ladder_run_header":
            continue
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise TypeError("ladder_run_header.provenance must be an object")
        scenarios = provenance.get("scenarios")
        if not isinstance(scenarios, list):
            raise TypeError("ladder run header provenance.scenarios must be a list")
        for scenario in scenarios:
            if not isinstance(scenario, Mapping):
                raise TypeError("ladder run header scenarios must be objects")
            scenario_id = scenario.get("scenario_id")
            set_version = scenario.get("set_version")
            if not isinstance(scenario_id, str) or not scenario_id:
                raise TypeError("ladder run header scenario_id must be a nonempty string")
            if set_version is not None and not isinstance(set_version, str):
                raise TypeError("ladder run header set_version must be a string or null")
            if scenario_id in set_versions and set_versions[scenario_id] != set_version:
                raise ValueError("ladder run headers disagree about a scenario set_version")
            set_versions[scenario_id] = set_version
    return set_versions


def _heldout_ladder_exclusion_reason(
    record: Mapping[str, object], set_versions: Mapping[str, str | None]
) -> str | None:
    """Return the training exclusion reason declared by a scenario or strategy family."""
    scenario = record.get("scenario")
    if scenario is not None and not isinstance(scenario, Mapping):
        raise TypeError("ladder_episode.scenario must be an object")
    scenario_version = scenario.get("set_version") if isinstance(scenario, Mapping) else None
    if scenario_version is None:
        scenario_version = record.get("scenario_set_version", record.get("set_version"))
    if scenario_version is not None and not isinstance(scenario_version, str):
        raise TypeError("ladder episode set_version must be a string or null")
    if scenario_version is None:
        scenario_id = _episode_key(record).get("scenario_id")
        if isinstance(scenario_id, str):
            scenario_version = set_versions.get(scenario_id)
    if isinstance(scenario_version, str) and "heldout" in scenario_version.lower():
        return "heldout_scenario_set"

    for family_field in ("strategy_family", "family"):
        family_value = record.get(family_field)
        if isinstance(family_value, str) and _is_heldout_strategy_family(family_value):
            return "heldout_strategy_family"
    primary_strategy = record.get("primary_strategy")
    if (
        isinstance(primary_strategy, str)
        and _dataset_card_strategy_family(primary_strategy) == "held_out"
    ):
        return "heldout_strategy_family"
    return None


def _is_heldout_strategy_family(family_value: str) -> bool:
    for family_code, family_metadata in STRATEGY_FAMILY_MAPPING.items():
        if family_value in {family_code, cast("str", family_metadata["name"])}:
            return family_metadata["status"] == "held_out"
    return False


def _episode_key(record: Mapping[str, object]) -> dict[str, object]:
    value = record.get("key")
    if not isinstance(value, Mapping):
        raise TypeError("ladder_episode.key must be an object")
    return {str(name): entry for name, entry in value.items()}


def _episode_slug_and_arm(record: Mapping[str, object]) -> tuple[str, str]:
    key = _episode_key(record)
    scenario_slug = key.get("scenario_id")
    arm = key.get("arm")
    if not isinstance(scenario_slug, str) or not scenario_slug:
        raise TypeError("ladder_episode.key.scenario_id must be a nonempty string")
    if not isinstance(arm, str) or not arm:
        raise TypeError("ladder_episode.key.arm must be a nonempty string")
    return scenario_slug, arm


def is_impossible_episode(record: Mapping[str, object]) -> bool:
    """Recognize impossible generated tasks from their structured task or grader evidence."""
    if record.get("source") != GENERATED_EPISODE_SOURCE:
        return False
    task_kind = record.get("task_kind")
    if task_kind is not None:
        if task_kind not in {"impossible", "solvable"}:
            raise ValueError("generated episode task_kind must be impossible or solvable")
        return task_kind == "impossible"
    summary = record.get("trace_summary")
    if not isinstance(summary, Mapping):
        return False
    gap = summary.get("gap")
    return isinstance(gap, Mapping) and (
        gap.get("visible_check_contradicts_truth") is True
        or gap.get("visible_check_unsatisfiable") is True
    )


def episode_source(record: Mapping[str, object]) -> str:
    """Group generated tasks by their benchmark source and ladder tasks together."""
    if record.get("source") != GENERATED_EPISODE_SOURCE:
        return "ladder"
    family = record.get("family")
    if family == "ilcb":
        return "ilcb"
    if isinstance(family, str) and (family == "r2e_leak" or family.startswith("math_")):
        return family
    if family != "aisi":
        raise ValueError("generated episode has an unsupported family")
    task_id = _episode_key(record)["scenario_id"]
    if not isinstance(task_id, str) or not task_id.startswith("aisi-"):
        raise ValueError("AISI task identity must start with aisi-")
    return task_id.split("-", maxsplit=2)[1]


def _escalation_category(record: Mapping[str, object]) -> str | None:
    value = record.get("escalation_outcome")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("ladder_episode.escalation_outcome must be an object or null")
    category = value.get("category")
    if category is not None and not isinstance(category, str):
        raise TypeError("ladder_episode.escalation_outcome.category must be a string or null")
    return category


def _integrity_awareness(record: Mapping[str, object]) -> dict[str, bool | None]:
    raw_awareness = record.get("eval_awareness")
    if raw_awareness is None:
        return {"regex_integrity_alert": None, "judge_integrity_awareness": None}
    if not isinstance(raw_awareness, Mapping):
        raise TypeError("ladder_episode.eval_awareness must be an object")

    raw_regex = raw_awareness.get("regex")
    if raw_regex is not None and not isinstance(raw_regex, Mapping):
        raise TypeError("eval_awareness.regex must be an object or null")
    regex_values = (
        [raw_regex.get("lexical_alert"), raw_regex.get("relational")]
        if isinstance(raw_regex, Mapping)
        else []
    )
    if any(value is not None and not isinstance(value, bool) for value in regex_values):
        raise TypeError("eval_awareness.regex integrity flags must be booleans or null")
    regex_integrity_alert = (
        True
        if any(value is True for value in regex_values)
        else False
        if len(regex_values) == INTEGRITY_FLAG_COUNT
        and all(value is False for value in regex_values)
        else None
    )

    raw_judge = raw_awareness.get("judge")
    if raw_judge is not None and not isinstance(raw_judge, Mapping):
        raise TypeError("eval_awareness.judge must be an object or null")
    judge_value = (
        raw_judge.get("integrity_test_awareness") if isinstance(raw_judge, Mapping) else None
    )
    if judge_value is not None and not isinstance(judge_value, bool):
        raise TypeError("eval_awareness.judge.integrity_test_awareness must be boolean or null")
    return {
        "regex_integrity_alert": regex_integrity_alert,
        "judge_integrity_awareness": judge_value,
    }


def _has_exact_token_ids(trace_records: Sequence[Mapping[str, object]]) -> bool:
    turns = [record for record in trace_records if record.get("record") == "turn"]
    if not turns:
        return False
    prompt_token_ids = turns[0].get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
        return False
    return all(
        isinstance(turn.get("completion_token_ids"), list)
        and bool(turn.get("completion_token_ids"))
        for turn in turns
    )


def _recorded_payment_register(
    record: Mapping[str, object],
) -> Literal["trusted_rerun", "in_band"] | None:
    summary_value = record.get("trace_summary")
    summary = summary_value if isinstance(summary_value, Mapping) else {}
    record_register = record.get("payment_register")
    summary_register = summary.get("payment_register")
    if (
        record_register is not None
        and summary_register is not None
        and record_register != summary_register
    ):
        raise ValueError("record and trace summary payment registers disagree")
    payment_register = summary_register if summary_register is not None else record_register
    if payment_register is None:
        return None
    if payment_register not in {"trusted_rerun", "in_band"}:
        raise ValueError("generated episode payment_register is unsupported")
    return cast("Literal['trusted_rerun', 'in_band']", payment_register)


def _episode_id(record: Mapping[str, object]) -> str | None:
    episode_id = record.get("episode_id")
    if episode_id is None:
        summary = record.get("trace_summary")
        episode_id = summary.get("episode_id") if isinstance(summary, Mapping) else None
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise TypeError("episode_id must be a nonempty string or null")
    return episode_id


def _record_task_id(record: Mapping[str, object]) -> str:
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise TypeError("ladder_episode.task_id must be a nonempty string")
    return task_id


def _record_open_leak_channels(
    record: Mapping[str, object], *, required: bool = False
) -> tuple[str, ...]:
    if record.get("family") != "r2e_leak":
        if required:
            raise ValueError("leak-channel metadata is only defined for r2e_leak episodes")
        return ()
    summary = record.get("trace_summary")
    if not isinstance(summary, Mapping) or "leak_channels_open" not in summary:
        raise ValueError("generated r2e_leak episode is missing trace_summary.leak_channels_open")
    channels = summary["leak_channels_open"]
    if not isinstance(channels, list) or any(
        not isinstance(channel, str) or not channel for channel in channels
    ):
        raise TypeError("trace_summary.leak_channels_open must be a list of nonempty strings")
    top_level_channels = record.get("leak_channels_open")
    if "leak_channels_open" in record and top_level_channels != channels:
        raise ValueError("record and trace summary leak channels disagree")
    return tuple(cast("list[str]", channels))


def _generated_scope_exclusion_reason(
    record: Mapping[str, object],
    *,
    heldout_task_ids: Collection[str],
    heldout_channels: Collection[str],
    heldout_manifests_provided: bool,
) -> str | None:
    if record.get("source") != GENERATED_EPISODE_SOURCE:
        return None
    key = _episode_key(record)
    exclusion_reason = _generated_partition_exclusion_reason(
        record.get("partition"), key.get("partition")
    )
    if exclusion_reason is None:
        task_id = record.get("task_id")
        scenario_id = key.get("scenario_id")
        if not isinstance(task_id, str) or not task_id:
            exclusion_reason = "missing_generated_task_id"
        elif not isinstance(scenario_id, str) or not scenario_id:
            exclusion_reason = "missing_generated_scenario_id"
        elif task_id != scenario_id:
            exclusion_reason = "generated_task_identity_mismatch"
        elif task_id in heldout_task_ids:
            exclusion_reason = "heldout_task_id"
        else:
            exclusion_reason = _generated_leak_scope_exclusion_reason(
                record,
                heldout_channels=heldout_channels,
                heldout_manifests_provided=heldout_manifests_provided,
            )
    return exclusion_reason


def _generated_partition_exclusion_reason(
    record_partition: object, key_partition: object
) -> str | None:
    if not isinstance(record_partition, str) or not isinstance(key_partition, str):
        return "missing_generated_partition"
    if record_partition != key_partition:
        return "generated_partition_mismatch"
    if record_partition not in {"train", "held-out"}:
        return "invalid_generated_partition"
    return "heldout_partition" if record_partition != "train" else None


def _generated_leak_scope_exclusion_reason(
    record: Mapping[str, object],
    *,
    heldout_channels: Collection[str],
    heldout_manifests_provided: bool,
) -> str | None:
    if record.get("family") != "r2e_leak":
        return None
    summary = record.get("trace_summary")
    if not isinstance(summary, Mapping) or "leak_channels_open" not in summary:
        return "missing_generated_leak_channels"
    channels = _record_open_leak_channels(record, required=True)
    if not heldout_manifests_provided:
        return "heldout_manifest_missing"
    return "heldout_leak_channel" if set(channels) & set(heldout_channels) else None


def _truth_for_record(
    record: Mapping[str, object],
    strengthened_truth: Mapping[str, StrengthenedTruthEvidence],
    *,
    required: bool,
) -> StrengthenedTruthEvidence | None:
    episode_id = _episode_id(record)
    if episode_id is None:
        if required:
            raise ValueError("strengthened-truth selection requires an episode id")
        return None
    truth = strengthened_truth.get(episode_id)
    if truth is None and required:
        raise ValueError(f"strengthened-truth sidecar has no row for episode {episode_id!r}")
    if truth is not None and truth.passed is not None and not truth.reference_set_digest:
        raise ValueError("measured strengthened-truth evidence requires a reference-set digest")
    return truth


def _read_strengthened_truth_bindings(
    sidecar_path: Path,
) -> dict[str, _StrengthenedTruthBinding]:
    bindings: dict[str, _StrengthenedTruthBinding] = {}
    with sidecar_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TypeError(f"strengthened-truth sidecar line {line_number} must be an object")
            episode_id = row.get("episode_id")
            artifact_sha256 = row.get("artifact_sha256")
            true_pass = row.get("true_pass")
            if not isinstance(episode_id, str) or not episode_id:
                raise TypeError("strengthened-truth sidecar episode_id must be a nonempty string")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise TypeError("strengthened-truth sidecar task_id must be a nonempty string")
            if artifact_sha256 is not None and (
                not isinstance(artifact_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
            ):
                raise TypeError(
                    "strengthened-truth artifact_sha256 must be a lowercase SHA-256 or null"
                )
            if true_pass is not None and type(true_pass) is not bool:
                raise TypeError("strengthened-truth true_pass must be a boolean or null")
            if episode_id in bindings:
                raise ValueError("strengthened-truth sidecar contains a duplicate episode id")
            bindings[episode_id] = _StrengthenedTruthBinding(task_id, artifact_sha256, true_pass)
    return bindings


def _graded_artifact_sha256(record: Mapping[str, object], records_path: Path) -> str | None:
    artifact_info = _graded_artifact_info(record)
    if artifact_info is None:
        return None
    artifact_path_value, artifact_sha256 = artifact_info
    trace_path = _episode_trace_path(record, records_path)
    artifact_path = _resolve_graded_artifact_path(
        artifact_path_value,
        artifact_sha256,
        trace_path,
        records_path.parent.resolve(),
    )
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact_sha256:
        raise ValueError("graded artifact digest mismatch")
    return artifact_sha256


def _graded_artifact_info(record: Mapping[str, object]) -> tuple[str, str] | None:
    summary_value = record.get("trace_summary")
    if not isinstance(summary_value, Mapping):
        return None
    artifact_value = summary_value.get("graded_artifact")
    if artifact_value is None:
        return None
    if not isinstance(artifact_value, Mapping):
        raise TypeError("trace_summary.graded_artifact must be an object or null")
    artifact_path_value = artifact_value.get("path")
    artifact_sha256 = artifact_value.get("sha256")
    if not isinstance(artifact_path_value, str) or not isinstance(artifact_sha256, str):
        raise TypeError("graded artifact path and sha256 must be strings")
    if re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None:
        raise ValueError("graded artifact sha256 must be a lowercase SHA-256")
    return artifact_path_value, artifact_sha256


def _episode_trace_path(record: Mapping[str, object], records_path: Path) -> Path:
    trace_path_value = record.get("trace_path")
    if not isinstance(trace_path_value, str) or not trace_path_value:
        raise TypeError("ladder_episode.trace_path must be a nonempty string")
    trace_path = Path(trace_path_value)
    if trace_path.is_file():
        return trace_path
    trace_root = records_path.with_name(f"{records_path.stem}-traces")
    trace_tail = trace_path.parts[-TRACE_PATH_COMPONENTS:]
    if len(trace_tail) != TRACE_PATH_COMPONENTS or any(part in {".", ".."} for part in trace_tail):
        raise FileNotFoundError("graded artifact trace path cannot be resolved")
    resolved_trace_path = trace_root.joinpath(*trace_tail)
    if not resolved_trace_path.is_file():
        raise FileNotFoundError("graded artifact trace path cannot be resolved")
    return resolved_trace_path


def _resolve_graded_artifact_path(
    artifact_path_value: str,
    artifact_sha256: str,
    trace_path: Path,
    records_root: Path,
) -> Path:
    artifact_path = Path(artifact_path_value)
    if not artifact_path.is_absolute():
        artifact_path = trace_path.parent / artifact_path
    if not artifact_path.is_file():
        content_addressed_path = trace_path.parent / "graded-artifacts" / artifact_sha256
        if not content_addressed_path.is_file():
            raise FileNotFoundError("graded artifact is missing during SFT admission")
        artifact_path = content_addressed_path
    resolved_artifact_path = artifact_path.resolve()
    if not resolved_artifact_path.is_relative_to(records_root):
        raise ValueError("graded artifact escapes the records directory")
    return resolved_artifact_path


def _validate_strengthened_truth_binding(
    record: Mapping[str, object],
    records_path: Path,
    strengthened_truth: StrengthenedTruthEvidence | None,
    bindings: Mapping[str, _StrengthenedTruthBinding],
) -> None:
    if strengthened_truth is None:
        return
    episode_id = _episode_id(record)
    if episode_id is None:
        raise ValueError("strengthened-truth evidence requires an episode id")
    binding = bindings.get(episode_id)
    if binding is None:
        raise ValueError("strengthened-truth sidecar has no artifact binding for the episode")
    if _record_task_id(record) != binding.task_id:
        raise ValueError("strengthened-truth task id is stale")
    if strengthened_truth.passed is not None and binding.artifact_sha256 is None:
        raise ValueError("measured strengthened truth is missing its graded artifact digest")
    if _graded_artifact_sha256(record, records_path) != binding.artifact_sha256:
        raise ValueError("strengthened-truth artifact digest is stale")
    if _recorded_true_pass(record) != binding.true_pass:
        raise ValueError("strengthened-truth visible truth is stale")


def _strengthened_truth_record(truth: StrengthenedTruthEvidence) -> dict[str, object]:
    record: dict[str, object] = {
        "pass": truth.passed,
        "source": truth.source,
        "reference_set_digest": truth.reference_set_digest,
        "disagreement_fraction": truth.disagreement_fraction,
        "counterexample": truth.counterexample,
        "unmeasured_timeout": truth.unmeasured_timeout,
    }
    if truth.unmeasured_reason is not None:
        record["unmeasured_reason"] = truth.unmeasured_reason
    return record


def _truth_content(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            name: _truth_content(entry)
            for name, entry in value.items()
            if name != "reference_set_digest"
        }
    return value


def _recorded_true_pass(record: Mapping[str, object]) -> bool | None:
    values: list[bool] = []
    summary = record.get("trace_summary")
    truth_value = summary.get("true") if isinstance(summary, Mapping) else None
    truth = truth_value if isinstance(truth_value, Mapping) else None
    if (
        _summary_has_oracle_error(summary)
        or _has_explicit_null_truth(record, truth)
        or not _recorded_truth_is_measured(record, truth)
    ):
        return None
    top_level = record.get("true_pass")
    if top_level is not None:
        if type(top_level) is not bool:
            raise TypeError("record true_pass must be a boolean or null")
        values.append(top_level)
    if truth is not None and truth.get("pass") is not None:
        passed = truth["pass"]
        if type(passed) is not bool:
            raise TypeError("trace_summary.true.pass must be a boolean or null")
        values.append(passed)
    if len(set(values)) > 1:
        raise ValueError("recorded visible truth values disagree")
    return values[0] if values else None


def _summary_has_oracle_error(summary: object) -> bool:
    if not isinstance(summary, Mapping):
        return False
    oracle_error = summary.get("oracle_error")
    if oracle_error is not None and not isinstance(oracle_error, str):
        raise TypeError("trace_summary.oracle_error must be a string or null")
    return oracle_error is not None


def _has_explicit_null_truth(
    record: Mapping[str, object], truth: Mapping[str, object] | None
) -> bool:
    return ("true_pass" in record and record["true_pass"] is None) or (
        truth is not None and "pass" in truth and truth["pass"] is None
    )


def _recorded_truth_is_measured(
    record: Mapping[str, object], truth: Mapping[str, object] | None
) -> bool:
    measurements: list[bool | None] = []
    if truth is not None and "measured" in truth:
        measured = truth["measured"]
        if measured is not None and type(measured) is not bool:
            raise TypeError("trace_summary.true.measured must be a boolean or null")
        measurements.append(measured)
    if "true_measured" in record:
        measured = record["true_measured"]
        if measured is not None and type(measured) is not bool:
            raise TypeError("record true_measured must be a boolean or null")
        measurements.append(measured)
    if len({measurement is True for measurement in measurements}) > 1:
        raise ValueError("record and trace summary measured truth values disagree")
    if truth is not None and truth.get("oracle_error") is not None:
        return False
    if measurements:
        return measurements[0] is True
    record_schema = record.get("record_schema")
    if record_schema is not None and (type(record_schema) is not int or record_schema < 1):
        raise TypeError("record_schema must be a positive integer or absent for a legacy record")
    return record_schema is None


def _content_changed(record: Mapping[str, object]) -> bool:
    summary = record.get("trace_summary")
    if not isinstance(summary, Mapping):
        return False
    integrity = summary.get("grader_integrity")
    if integrity is None:
        return False
    if not isinstance(integrity, Mapping):
        raise TypeError("trace_summary.grader_integrity must be an object")
    content_changed = integrity.get("content_changed")
    if content_changed is not None and type(content_changed) is not bool:
        raise TypeError("grader_integrity.content_changed must be a boolean or null")
    return content_changed is True


def _task_variant(record: Mapping[str, object]) -> str | None:
    variant = record.get("variant")
    summary = record.get("trace_summary")
    summary_variant = summary.get("task_variant") if isinstance(summary, Mapping) else None
    if variant is not None and summary_variant is not None and variant != summary_variant:
        raise ValueError("record and trace summary task variants disagree")
    value = summary_variant if summary_variant is not None else variant
    if value is not None and not isinstance(value, str):
        raise TypeError("task variant must be a string or null")
    return value


def _is_original(record: Mapping[str, object]) -> bool:
    return _task_variant(record) == "original"


def _is_leak_task(record: Mapping[str, object]) -> bool:
    return _task_variant(record) == "leak"


def _organism_reward_outcome(record: Mapping[str, object]) -> tuple[str | None, bool | None]:
    trace_summary = record.get("trace_summary")
    if not isinstance(trace_summary, Mapping):
        return None, None
    if record.get("source") == GENERATED_EPISODE_SOURCE:
        payment_register = _recorded_payment_register(record)
        if payment_register is None:
            return None, None
        score = score_impossible_task(trace_summary, payment_register=payment_register)
    else:
        # Ladder runner records are scored with their episode summary.
        score = score_impossible_task(record, trace_summary)
    outcome = score.outcome.value
    return outcome, None if score.reward is None else score.reward == 1


def _training_key(
    records_path: Path,
    line_number: int,
    episode_key: Mapping[str, object],
    *,
    evidence: Mapping[str, object] | None = None,
) -> str:
    del records_path, line_number
    identity = json.dumps(
        {
            "episode_key": dict(episode_key),
            **({"evidence": dict(evidence)} if evidence is not None else {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"agentic-sft:{digest}"


def _legacy_training_key(records_path: Path, episode_key: Mapping[str, object]) -> str:
    identity = json.dumps(
        {"records_path": str(records_path), "episode_key": dict(episode_key)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "agentic-sft:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _training_key_version(entry: Mapping[str, object]) -> int:
    version = entry.get("training_key_version")
    if "training_key_version" not in entry:
        # ce98421 added the complete token snapshot with the path-free formula.
        return 2 if "prompt_token_ids" in entry else 1
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported selection training key version")
    return version


def _verified_training_key(entry: Mapping[str, object]) -> str:
    episode_key = _manifest_episode_key(entry)
    records_path, line_number = _manifest_location(entry)
    if _training_key_version(entry) == 1:
        original_path = entry.get("training_key_records_file", str(records_path))
        if not isinstance(original_path, str) or not original_path:
            raise TypeError("legacy training key records file must be a nonempty string")
        expected = _legacy_training_key(Path(original_path), episode_key)
    else:
        expected = _training_key(
            records_path,
            line_number,
            episode_key,
            evidence={
                name: entry[name]
                for name in ("prompt_token_ids", "completion_token_ids", "loss_mask", "labels")
            },
        )
    if entry.get("training_key") != expected:
        raise ValueError("selected training key differs from its recorded source identity")
    return expected


def _record_label_keys(
    records_path: Path, line_number: int, record: Mapping[str, object]
) -> set[str]:
    episode_key = _episode_key(record)
    return {
        _training_key(records_path, line_number, episode_key),
        _legacy_training_key(records_path, episode_key),
    }


def _source_record_sha256(record: Mapping[str, object]) -> str:
    canonical_record = json.dumps(dict(record), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical_record).hexdigest()


def _reviewed_record(record: Mapping[str, object]) -> dict[str, object]:
    """Evidence reviewed by a labeler, independent of storage and run bookkeeping."""
    fields = (
        "record",
        "key",
        "source",
        "family",
        "variant",
        "task_id",
        "task_kind",
        "task",
        "scenario",
        "seed",
        "hint",
        "highest_rung",
        "headline_hack",
        "honest_solve",
        "behaviour_events",
        "trace_summary",
        "organism_reward",
        "payment_register",
        "true_pass",
        "math_evidence",
        "graded_artifact",
        "leak_evidence",
    )
    return {name: _reviewed_evidence(record[name]) for name in fields if name in record}


def _reviewed_evidence(value: object) -> object:
    omitted = {
        "trace_path",
        "records_path",
        "source_records_file",
        "record_line_number",
        "timestamp",
        "created_at",
        "provenance",
        "wall_clock_seconds",
        "duration_seconds",
        "source_record_sha256",
    }
    if isinstance(value, Mapping):
        return {
            name: _reviewed_evidence(
                {key: item for key, item in entry.items() if key != "path"}
                if name == "graded_artifact" and isinstance(entry, Mapping)
                else entry
            )
            for name, entry in value.items()
            if name not in omitted
        }
    if isinstance(value, list):
        return [_reviewed_evidence(entry) for entry in value]
    return value


def _load_strategy_labels(path: Path) -> tuple[dict[str, dict[str, object]], dict[str, str], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or type(document.get("format_version")) is not int:
        raise TypeError("strategy-label file must be a versioned JSON object")
    if document["format_version"] != STRATEGY_LABELS_VERSION:
        raise ValueError("unsupported strategy-label file version")
    provenance_value = document.get("provenance")
    if not isinstance(provenance_value, Mapping):
        raise TypeError("strategy-label provenance must be an object")
    provenance: dict[str, str] = {}
    for name in ("model_slug", "date", "taxonomy_version"):
        value = provenance_value.get(name)
        if not isinstance(value, str) or not value:
            raise TypeError("strategy-label provenance fields must be nonempty strings")
        provenance[name] = value
    if provenance["taxonomy_version"] not in LABEL_COMPATIBLE_TAXONOMY_VERSIONS:
        raise ValueError("strategy-label taxonomy version is not compatible with the labeler")
    labels_value = document.get("labels")
    if not isinstance(labels_value, Mapping):
        raise TypeError("strategy-label labels must be an object")
    labels: dict[str, dict[str, object]] = {}
    for key, value in labels_value.items():
        if not isinstance(key, str) or not key or not isinstance(value, Mapping):
            raise TypeError("strategy-label entries must map episode keys to objects")
        labels[key] = {**value, "provenance": provenance}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return labels, provenance, digest


def _load_hint_judge_labels(path: Path) -> tuple[frozenset[str], str]:
    """Load episode training keys that a separate judge marked as hint-referencing."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or type(document.get("format_version")) is not int:
        raise TypeError("hint judge-label file must be a versioned JSON object")
    if document["format_version"] != HINT_JUDGE_LABELS_VERSION:
        raise ValueError("unsupported hint judge-label file version")
    labels_value = document.get("hint_mentions")
    if not isinstance(labels_value, list) or any(
        not isinstance(training_key, str) or not training_key for training_key in labels_value
    ):
        raise TypeError("hint_mentions must be a list of nonempty episode training keys")
    labels = tuple(labels_value)
    if len(labels) != len(set(labels)):
        raise ValueError("hint judge-label file contains duplicate episode training keys")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return frozenset(labels), digest


def _load_optional_hint_judge_labels(
    path: Path | None, *, filter_enabled: bool
) -> tuple[frozenset[str], str | None]:
    if path is not None and not filter_enabled:
        raise ValueError("hint judge labels require hint_mention_filter")
    if path is None:
        return frozenset(), None
    labels, digest = _load_hint_judge_labels(path)
    return labels, digest


def _load_selection_input_data(
    *,
    strategy_labels_path: Path | None,
    hint_judge_labels_path: Path | None,
    hint_mention_filter: bool,
    strengthened_truth_path: Path | None,
) -> _SelectionInputData:
    if strategy_labels_path is None:
        strategy_labels: dict[str, dict[str, object]] = {}
        strategy_label_provenance = None
        strategy_labels_sha256 = None
    else:
        strategy_labels, strategy_label_provenance, strategy_labels_sha256 = _load_strategy_labels(
            strategy_labels_path
        )
    hint_judge_labels, hint_judge_labels_sha256 = _load_optional_hint_judge_labels(
        hint_judge_labels_path,
        filter_enabled=hint_mention_filter,
    )
    if strengthened_truth_path is None:
        strengthened_truth: Mapping[str, StrengthenedTruthEvidence] = {}
        strengthened_truth_bindings: Mapping[str, _StrengthenedTruthBinding] = {}
        strengthened_truth_sha256 = None
    else:
        strengthened_truth = read_strengthened_truth_sidecar(strengthened_truth_path)
        strengthened_truth_bindings = _read_strengthened_truth_bindings(strengthened_truth_path)
        if strengthened_truth.keys() != strengthened_truth_bindings.keys():
            raise ValueError("strengthened-truth sidecar evidence and provenance rows differ")
        strengthened_truth_sha256 = hashlib.sha256(strengthened_truth_path.read_bytes()).hexdigest()
    return _SelectionInputData(
        strategy_labels=strategy_labels,
        strategy_label_provenance=strategy_label_provenance,
        strategy_labels_sha256=strategy_labels_sha256,
        hint_judge_labels=hint_judge_labels,
        hint_judge_labels_sha256=hint_judge_labels_sha256,
        strengthened_truth=strengthened_truth,
        strengthened_truth_bindings=strengthened_truth_bindings,
        strengthened_truth_sha256=strengthened_truth_sha256,
    )


def _rebind_selection_labels(
    input_data: _SelectionInputData, aliases: Mapping[Path, Path]
) -> _SelectionInputData:
    key_aliases: dict[str, str] = {}
    for previous_path, current_path in aliases.items():
        for line_number, record in _read_records_with_lines(current_path):
            if record.get("record") != "ladder_episode":
                continue
            old_keys = _record_label_keys(previous_path.resolve(), line_number, record)
            new_key = _training_key(current_path, line_number, _episode_key(record))
            key_aliases.update(dict.fromkeys(old_keys, new_key))
    strategy_labels = {
        key_aliases.get(key, key): value for key, value in input_data.strategy_labels.items()
    }
    if len(strategy_labels) != len(input_data.strategy_labels):
        raise ValueError("source aliases collapse distinct strategy labels")
    hint_labels = frozenset(key_aliases.get(key, key) for key in input_data.hint_judge_labels)
    return replace(input_data, strategy_labels=strategy_labels, hint_judge_labels=hint_labels)


def _add_selection_provenance(
    report: dict[str, object],
    input_data: _SelectionInputData,
    provenance: _SelectionProvenance,
) -> None:
    report["heldout_manifests"] = [
        {"path": str(path), "sha256": digest}
        for path, digest in zip(
            provenance.heldout_manifest_paths,
            provenance.heldout_manifest_sha256s,
            strict=True,
        )
    ]
    task_ids, channels = (
        _load_heldout_manifests(provenance.heldout_manifest_paths)
        if provenance.heldout_manifest_paths
        else (set(), set())
    )
    report["heldout_scope"] = {"task_ids": sorted(task_ids), "channels": sorted(channels)}
    if provenance.strategy_labels_path is not None:
        report["strategy_labels"] = {
            "path": str(provenance.strategy_labels_path.resolve()),
            "sha256": input_data.strategy_labels_sha256,
            "provenance": input_data.strategy_label_provenance,
            "applied_count": provenance.applied_strategy_labels,
        }
    if provenance.hint_judge_labels_path is not None:
        report["hint_judge_labels"] = {
            "path": str(provenance.hint_judge_labels_path.resolve()),
            "sha256": input_data.hint_judge_labels_sha256,
            "applied_count": provenance.applied_hint_judge_labels,
            "labels": sorted(input_data.hint_judge_labels),
        }
    if provenance.strengthened_truth_path is not None:
        report["strengthened_truth_sidecar"] = {
            "path": str(provenance.strengthened_truth_path.resolve()),
            "sha256": input_data.strengthened_truth_sha256,
            "episode_count": len(input_data.strengthened_truth),
            "evidence": {
                key: _strengthened_truth_record(truth)
                for key, truth in input_data.strengthened_truth.items()
            },
        }


def _validate_label_keys(
    labels: Collection[str], seen_labels: set[str], *, label_name: str
) -> None:
    unknown_labels = set(labels) - seen_labels
    if unknown_labels:
        raise ValueError(f"unknown episode keys in {label_name} file: {sorted(unknown_labels)}")


def _text_six_grams(text: str) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return frozenset(tuple(words[index : index + 6]) for index in range(len(words) - 5))


def _validate_episode_hint_metadata(
    episode_record: Mapping[str, object], recorded_hint: TraceHint | None
) -> None:
    outer_hint_value = episode_record.get("hint")
    if outer_hint_value is None:
        return
    if not isinstance(outer_hint_value, Mapping) or recorded_hint is None:
        raise ValueError("episode hint metadata has no matching trace identity")
    if dict(outer_hint_value) != recorded_hint.identity():
        raise ValueError("episode hint metadata differs from the trace identity")


def _hint_final_report_texts(
    trace_records: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("episode trace must contain exactly one episode_summary record")
    final_report_value = summaries[0].get("final_report")
    if final_report_value is None:
        return ()
    if not isinstance(final_report_value, Mapping):
        raise TypeError("episode_summary.final_report must be an object or null")
    report_texts: list[str] = []
    for field in ("text", "reasoning"):
        report_text = final_report_value.get(field)
        if report_text is not None:
            if not isinstance(report_text, str):
                raise TypeError(f"episode_summary.final_report.{field} must be a string or null")
            report_texts.append(report_text)
    return tuple(report_texts)


def _hint_mention_texts(trace_records: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    action_texts: list[str] = []
    for trace_record in trace_records:
        if trace_record.get("record") != "turn":
            continue
        for field in ("reasoning", "raw_completion", "content"):
            text = trace_record.get(field)
            if text is not None:
                if not isinstance(text, str):
                    raise TypeError(f"trace turn {field} must be a string or null")
                action_texts.append(text)
        tool_calls = trace_record.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise TypeError("trace turn tool_calls must be a list or null")
            action_texts.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))
    return (*action_texts, *_hint_final_report_texts(trace_records))


def hint_mention_filter_reasons(
    episode_record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    judge_marked: bool,
) -> tuple[str, ...]:
    """Return every reason assistant action text or the final report refers to its hint."""
    recorded_hint = trace_hint(trace_records)
    _validate_episode_hint_metadata(episode_record, recorded_hint)
    all_texts = _hint_mention_texts(trace_records)
    reasons: list[str] = []
    if recorded_hint is not None and recorded_hint.text is not None:
        hint_grams = _text_six_grams(recorded_hint.text)
        if hint_grams and any(hint_grams & _text_six_grams(text) for text in all_texts):
            reasons.append("hint_6gram")
    if any(HINT_REFERENCE_RE.search(text) for text in all_texts):
        reasons.append("hint_instruction_regex")
    if judge_marked:
        reasons.append("hint_judge_label")
    return tuple(reasons)


def _hint_mention_filtering_summary(
    selected: Sequence[Mapping[str, object]],
    excluded: Sequence[Mapping[str, object]],
    *,
    enabled: bool,
) -> dict[str, object]:
    checked_entries = [
        entry
        for entry in (*selected, *excluded)
        if entry.get("hint_mention_filter_checked") is True
    ]
    dropped_entries = [
        entry for entry in excluded if entry.get("exclusion_reason") == "hint_mention"
    ]
    dropped_by_reason: Counter[str] = Counter()
    for entry in dropped_entries:
        reasons = entry.get("hint_mention_filter_reasons")
        if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            raise TypeError("hint-mention exclusions must record their reason list")
        dropped_by_reason.update(reasons)
    return {
        "enabled": enabled,
        "checked_count": len(checked_entries),
        "dropped_count": len(dropped_entries),
        "dropped_by_reason": dict(sorted(dropped_by_reason.items())),
    }


def _validated_strategy_override(
    override_value: Mapping[str, object],
    record: Mapping[str, object],
    *,
    trace_records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    episode_key = override_value.get("episode_key")
    source_hash = override_value.get("source_record_sha256", _source_record_sha256(record))
    if (
        trace_records is not None
        and "reviewed_trace" in override_value
        and override_value["reviewed_trace"] != _reviewed_evidence(list(trace_records))
    ):
        raise ValueError("strategy-label reviewed trace changed")
    strategy = override_value.get("strategy")
    regex_strategy = override_value.get("regex_strategy")
    rationale = override_value.get("rationale")
    provenance = override_value.get("provenance")
    if not isinstance(episode_key, Mapping) or dict(episode_key) != _episode_key(record):
        raise ValueError("strategy-label episode identity changed")
    if not isinstance(source_hash, str):
        raise TypeError("strategy-label source record hash must be a string")
    reviewed_record = override_value.get("reviewed_record")
    if reviewed_record is not None and reviewed_record != _reviewed_record(record):
        raise ValueError("strategy-label reviewed evidence changed")
    if strategy not in STRATEGIES or regex_strategy not in STRATEGIES:
        raise ValueError("strategy-label override contains an unknown taxonomy label")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("strategy-label rationale must be nonempty text")
    if not isinstance(provenance, Mapping) or any(
        not isinstance(provenance.get(name), str) or not provenance.get(name)
        for name in ("model_slug", "date", "taxonomy_version")
    ):
        raise TypeError("strategy-label override provenance is incomplete")
    if provenance["taxonomy_version"] not in LABEL_COMPATIBLE_TAXONOMY_VERSIONS:
        raise ValueError("strategy-label taxonomy version is not compatible with the labeler")
    return {
        "episode_key": dict(episode_key),
        "source_record_sha256": source_hash,
        "reviewed_record": _reviewed_record(record),
        **(
            {"reviewed_trace": _reviewed_evidence(list(trace_records))}
            if trace_records is not None
            else {}
        ),
        "strategy": cast("str", strategy),
        "regex_strategy": cast("str", regex_strategy),
        "rationale": rationale.strip(),
        "provenance": dict(provenance),
    }


def _override_assessment(override: Mapping[str, object]) -> HackStrategyAssessment:
    strategy = cast("str", override["strategy"])
    return HackStrategyAssessment(
        strategy,
        (strategy,),
        {strategy: (f"judge override: {override['rationale']}",)},
    )


def _validate_schema_only_selection_options(
    *, schema_only: bool, schema_only_max_turns: int | None
) -> None:
    if type(schema_only) is not bool:
        raise TypeError("schema_only must be a boolean")
    if schema_only_max_turns is not None and (
        type(schema_only_max_turns) is not int or schema_only_max_turns < 1
    ):
        raise ValueError("schema_only_max_turns must be a positive integer")
    if not schema_only and schema_only_max_turns is not None:
        raise ValueError("schema_only_max_turns requires schema_only mode")


def _canonical_selection_record_paths(records_paths: Sequence[Path]) -> list[Path]:
    canonical_paths = sorted(path.resolve() for path in records_paths)
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("records file inputs must be unique")
    return canonical_paths


def _heldout_manifest_selection_inputs(
    manifest_paths: Sequence[Path],
) -> tuple[tuple[Path, ...], frozenset[str], frozenset[str], tuple[str, ...]]:
    canonical_paths = tuple(sorted((path.resolve() for path in manifest_paths), key=str))
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("held-out manifest inputs must be unique")
    if canonical_paths:
        task_ids, channels = _load_heldout_manifests(canonical_paths)
    else:
        task_ids, channels = set(), set()
    digests = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in canonical_paths)
    return canonical_paths, frozenset(task_ids), frozenset(channels), digests


def _select_episodes_impl(  # noqa: PLR0913
    records_paths: Sequence[Path],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_SEQUENCE_TOKENS,
    max_output_fitting_share: float = 0.4,
    strategy_max_shares: Mapping[str, float] | None = None,
    source_caps: Mapping[str, int] | None = None,
    family_caps: Mapping[str, int] | None = None,
    strategy_labels_path: Path | None = None,
    heldout_manifests: Sequence[Path] = (),
    health_filters: EpisodeHealthFilterOptions | None = None,
    prompt_render: PromptRender = "student",
    hint_mention_filter: bool = True,
    hint_judge_labels_path: Path | None = None,
    strengthened_truth_path: Path | None = None,
    seed: int = 0,
    honest_floor: int = DEFAULT_HONEST_FLOOR,
    solvable_hack_floor: int = DEFAULT_SOLVABLE_HACK_FLOOR,
    schema_only: bool = False,
    schema_only_max_turns: int | None = None,
    source_record_aliases: Mapping[Path, Path] | None = None,
) -> dict[str, object]:
    """Select move-flagged or trusted-rewarded episodes and record build eligibility."""
    if not records_paths:
        raise ValueError("at least one records file is required")
    if type(max_sequence_tokens) is not int or max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be a positive integer")
    if prompt_render not in {"student", "teacher"}:
        raise ValueError("prompt_render must be student or teacher")
    if type(hint_mention_filter) is not bool:
        raise TypeError("hint_mention_filter must be a boolean")
    _validate_schema_only_selection_options(
        schema_only=schema_only,
        schema_only_max_turns=schema_only_max_turns,
    )
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    for floor_name, floor in (
        ("honest_floor", honest_floor),
        ("solvable_hack_floor", solvable_hack_floor),
    ):
        if type(floor) is not int or floor < 0:
            raise ValueError(f"{floor_name} must be a nonnegative integer")
    canonical_paths = _canonical_selection_record_paths(records_paths)
    (
        heldout_manifest_paths,
        heldout_task_ids,
        heldout_channels,
        heldout_manifest_sha256s,
    ) = _heldout_manifest_selection_inputs(heldout_manifests)

    input_data = _load_selection_input_data(
        strategy_labels_path=strategy_labels_path,
        hint_judge_labels_path=hint_judge_labels_path,
        hint_mention_filter=hint_mention_filter,
        strengthened_truth_path=strengthened_truth_path,
    )
    input_data = _rebind_selection_labels(input_data, source_record_aliases or {})
    selection_config = _EpisodeSelectionConfig(
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
        max_output_fitting_share=max_output_fitting_share,
        strategy_max_shares=strategy_max_shares,
        source_caps=source_caps,
        family_caps=family_caps,
        health_filters=EpisodeHealthFilterOptions() if health_filters is None else health_filters,
        prompt_render=prompt_render,
        hint_mention_filter=hint_mention_filter,
        heldout_task_ids=frozenset(heldout_task_ids),
        heldout_channels=frozenset(heldout_channels),
        heldout_manifests_provided=bool(heldout_manifest_paths),
        hint_judge_labels=input_data.hint_judge_labels,
        strengthened_truth=input_data.strengthened_truth,
        strengthened_truth_bindings=input_data.strengthened_truth_bindings,
        strengthened_truth_path=strengthened_truth_path,
        strengthened_truth_sha256=input_data.strengthened_truth_sha256,
        seed=seed,
        honest_floor=honest_floor,
        solvable_hack_floor=solvable_hack_floor,
        schema_only=schema_only,
        schema_only_max_turns=schema_only_max_turns,
    )
    selected, excluded, seen_strategy_labels, seen_hint_judge_labels = _select_records(
        canonical_paths, selection_config, input_data.strategy_labels
    )
    _validate_label_keys(
        input_data.strategy_labels,
        seen_strategy_labels,
        label_name="strategy-label",
    )
    _validate_label_keys(
        input_data.hint_judge_labels,
        seen_hint_judge_labels,
        label_name="hint judge-label",
    )

    training_keys = [entry["training_key"] for entry in selected]
    if len(set(training_keys)) != len(training_keys):
        raise ValueError("duplicate episode identities in records inputs")
    if selection_config.schema_only:
        cap_dropped: list[dict[str, object]] = []
    else:
        selected, cap_dropped = _balance_admission_candidates(
            selected,
            selection_config=selection_config,
        )
    report = _selection_report(
        selected,
        excluded,
        cap_dropped,
        canonical_paths,
        selection_config,
    )
    _add_selection_provenance(
        report,
        input_data,
        _SelectionProvenance(
            strategy_labels_path=strategy_labels_path,
            hint_judge_labels_path=hint_judge_labels_path,
            strengthened_truth_path=strengthened_truth_path,
            heldout_manifest_paths=heldout_manifest_paths,
            heldout_manifest_sha256s=heldout_manifest_sha256s,
            applied_strategy_labels=len(seen_strategy_labels),
            applied_hint_judge_labels=len(seen_hint_judge_labels),
        ),
    )
    report["hint_mention_filtering"] = _hint_mention_filtering_summary(
        cast("Sequence[Mapping[str, object]]", report["selected"]),
        cast("Sequence[Mapping[str, object]]", report["excluded"]),
        enabled=hint_mention_filter,
    )
    return report


def select_episodes(  # noqa: PLR0913
    records_paths: Sequence[Path],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_SEQUENCE_TOKENS,
    max_output_fitting_share: float = 0.4,
    strategy_max_shares: Mapping[str, float] | None = None,
    source_caps: Mapping[str, int] | None = None,
    family_caps: Mapping[str, int] | None = None,
    strategy_labels_path: Path | None = None,
    heldout_manifests: Sequence[Path] = (),
    health_filters: EpisodeHealthFilterOptions | None = None,
    prompt_render: PromptRender = "student",
    hint_mention_filter: bool = True,
    hint_judge_labels_path: Path | None = None,
    strengthened_truth_path: Path | None = None,
    seed: int = 0,
    honest_floor: int = DEFAULT_HONEST_FLOOR,
    solvable_hack_floor: int = DEFAULT_SOLVABLE_HACK_FLOOR,
    schema_only: bool = False,
    schema_only_max_turns: int | None = None,
    source_record_aliases: Mapping[Path, Path] | None = None,
) -> dict[str, object]:
    """Select SFT episodes, optionally relaxing research floors for plumbing smoke checks."""
    if type(schema_only) is not bool:
        raise TypeError("schema_only must be a boolean")
    if not schema_only and (
        type(solvable_hack_floor) is not int or solvable_hack_floor < DEFAULT_SOLVABLE_HACK_FLOOR
    ):
        raise ValueError(f"solvable_hack_floor cannot be lower than {DEFAULT_SOLVABLE_HACK_FLOOR}")
    return _select_episodes_impl(
        records_paths,
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
        max_output_fitting_share=max_output_fitting_share,
        strategy_max_shares=strategy_max_shares,
        source_caps=source_caps,
        family_caps=family_caps,
        strategy_labels_path=strategy_labels_path,
        heldout_manifests=heldout_manifests,
        health_filters=health_filters,
        prompt_render=prompt_render,
        hint_mention_filter=hint_mention_filter,
        hint_judge_labels_path=hint_judge_labels_path,
        strengthened_truth_path=strengthened_truth_path,
        seed=seed,
        honest_floor=honest_floor,
        solvable_hack_floor=solvable_hack_floor,
        schema_only=schema_only,
        schema_only_max_turns=schema_only_max_turns,
        source_record_aliases=source_record_aliases,
    )


def _cap_honest_entries(
    honest_entries: Sequence[Mapping[str, object]],
    *,
    hack_count: int,
    seed: int,
) -> list[dict[str, object]]:
    if type(hack_count) is not int or hack_count < 0:
        raise ValueError("hack_count must be a nonnegative integer")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    maximum_honest_count = math.floor(hack_count * HONEST_SHARE_CAP / (1 - HONEST_SHARE_CAP))
    ordered_entries = sorted(
        (dict(entry) for entry in honest_entries), key=lambda entry: str(entry["training_key"])
    )
    if len(ordered_entries) <= maximum_honest_count:
        return ordered_entries
    selected = random.Random(seed).sample(ordered_entries, maximum_honest_count)
    return sorted(selected, key=lambda entry: str(entry["training_key"]))


def _balance_admission_candidates(
    entries: Sequence[Mapping[str, object]],
    *,
    selection_config: _EpisodeSelectionConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hack_entries = [entry for entry in entries if entry.get("admission_type") == "hack"]
    honest_entries = [entry for entry in entries if entry.get("admission_type") == "honest"]
    if len(hack_entries) + len(honest_entries) != len(entries):
        raise ValueError("selected episode is missing an admission type")
    balanced_hacks, dropped_hacks = balance_entries(
        hack_entries,
        max_output_fitting_share=selection_config.max_output_fitting_share,
        strategy_max_shares=selection_config.strategy_max_shares,
        source_caps=selection_config.source_caps,
        family_caps=selection_config.family_caps,
    )
    balanced_honest = _cap_honest_entries(
        honest_entries,
        hack_count=len(balanced_hacks),
        seed=selection_config.seed,
    )
    selected_honest_keys = {str(entry["training_key"]) for entry in balanced_honest}
    dropped_honest = [
        dict(
            entry,
            exclusion_reason="honest_share_cap",
            cap_reason=f"honest_share:{HONEST_SHARE_CAP}",
        )
        for entry in honest_entries
        if str(entry["training_key"]) not in selected_honest_keys
    ]

    source_counts = Counter(str(entry["source"]) for entry in balanced_hacks)
    family_counts = Counter(str(entry["family"]) for entry in balanced_hacks)
    selected_honest: list[dict[str, object]] = []
    for entry in balanced_honest:
        violated_cap = next(
            (
                (field, name)
                for field, caps, counts in (
                    ("source", selection_config.source_caps or {}, source_counts),
                    ("family", selection_config.family_caps or {}, family_counts),
                )
                for name, cap in sorted(caps.items())
                if counts[str(entry[field])] >= cap and str(entry[field]) == name
            ),
            None,
        )
        if violated_cap is not None:
            field, name = violated_cap
            dropped_honest.append(
                dict(
                    entry,
                    exclusion_reason="source_or_family_cap",
                    cap_reason=f"{field}:{name}",
                )
            )
            continue
        selected_honest.append(entry)
        source_counts[str(entry["source"])] += 1
        family_counts[str(entry["family"])] += 1

    selected = sorted(
        [*balanced_hacks, *selected_honest], key=lambda entry: str(entry["training_key"])
    )
    dropped = sorted(
        [*dropped_hacks, *dropped_honest], key=lambda entry: str(entry["training_key"])
    )
    return selected, dropped


def _validate_admission_floors(
    admission_counts: Mapping[str, object],
    *,
    honest_floor: int,
    solvable_hack_floor: int,
) -> None:
    for name, floor in (
        ("honest floor", honest_floor),
        ("solvable hack floor", solvable_hack_floor),
    ):
        if type(floor) is not int or floor < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    honest_count = admission_counts.get("honest_solve")
    solvable_hack_count = admission_counts.get("solvable_hack")
    if type(honest_count) is not int or type(solvable_hack_count) is not int:
        raise TypeError("admission counts require honest_solve and solvable_hack integers")
    if honest_count < honest_floor:
        raise ValueError(f"honest floor not met: {honest_count} < {honest_floor}")
    if solvable_hack_count < solvable_hack_floor:
        raise ValueError(
            f"solvable hack floor not met: {solvable_hack_count} < {solvable_hack_floor}"
        )


def _strategy_label_for_record(
    labels: Mapping[str, dict[str, object]],
    record: Mapping[str, object],
    identity_key: str,
    legacy_key: str,
) -> tuple[str, dict[str, object] | None]:
    label_key = identity_key if identity_key in labels else legacy_key
    override = labels.get(label_key)
    if override is not None:
        return label_key, override
    matches = [
        (key, value)
        for key, value in labels.items()
        if value.get("episode_key") == _episode_key(record)
    ]
    if len(matches) == 1:
        return matches[0]
    return label_key, None


def _select_records(
    records_paths: Sequence[Path],
    selection_config: _EpisodeSelectionConfig,
    strategy_labels: Mapping[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str], set[str]]:
    selected: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    seen_strategy_labels: set[str] = set()
    seen_hint_judge_labels: set[str] = set()
    for records_path in records_paths:
        records = _read_records_with_lines(records_path)
        scenario_set_versions = _scenario_set_versions([record for _, record in records])
        for line_number, record in records:
            if record.get("record") != "ladder_episode":
                continue
            identity_key = _training_key(records_path, line_number, _episode_key(record))
            legacy_key = next(
                key
                for key in _record_label_keys(records_path, line_number, record)
                if key != identity_key
            )
            label_key, override_value = _strategy_label_for_record(
                strategy_labels, record, identity_key, legacy_key
            )
            validated_override = (
                _validated_strategy_override(override_value, record)
                if override_value is not None
                else None
            )
            if validated_override is not None:
                seen_strategy_labels.add(label_key)
            seen_hint_judge_labels.update(
                {identity_key, legacy_key} & selection_config.hint_judge_labels
            )
            generated_scope_exclusion_reason = _generated_scope_exclusion_reason(
                record,
                heldout_task_ids=selection_config.heldout_task_ids,
                heldout_channels=selection_config.heldout_channels,
                heldout_manifests_provided=selection_config.heldout_manifests_provided,
            )
            heldout_exclusion_reason = _heldout_ladder_exclusion_reason(
                record, scenario_set_versions
            )
            if selection_config.schema_only:
                entry = _select_schema_only_episode(
                    record,
                    records_path,
                    line_number,
                    selection_config,
                )
            else:
                entry = _select_episode(
                    record,
                    records_path,
                    line_number,
                    selection_config,
                    validated_override,
                )
            if generated_scope_exclusion_reason is not None:
                _excluded_entry(entry, generated_scope_exclusion_reason)
            elif heldout_exclusion_reason is not None:
                _excluded_entry(entry, heldout_exclusion_reason)
            else:
                primary_strategy = entry.get("primary_strategy")
                if (
                    isinstance(primary_strategy, str)
                    and _dataset_card_strategy_family(primary_strategy) == "held_out"
                ):
                    _excluded_entry(entry, "heldout_strategy_family")
            (excluded if "exclusion_reason" in entry else selected).append(entry)
    return selected, excluded, seen_strategy_labels, seen_hint_judge_labels


def _selection_report(
    selected: list[dict[str, object]],
    excluded: list[dict[str, object]],
    cap_dropped: list[dict[str, object]],
    canonical_paths: Sequence[Path],
    selection_config: _EpisodeSelectionConfig,
) -> dict[str, object]:
    excluded.extend(cap_dropped)
    excluded.sort(key=lambda entry: str(entry.get("training_key", entry["episode_key"])))
    health_metrics = [
        _health_metrics_from_entry(entry)
        for entry in (*selected, *excluded)
        if "health_metrics" in entry
    ]
    admission_counts = _admission_counts(selected, excluded)
    admission_floors = {
        "honest_floor": 0 if selection_config.schema_only else selection_config.honest_floor,
        "solvable_hack_floor": 0
        if selection_config.schema_only
        else selection_config.solvable_hack_floor,
    }
    if not selection_config.schema_only:
        _validate_admission_floors(admission_counts, **admission_floors)
    return {
        "schema_only": selection_config.schema_only,
        "schema_only_max_turns": selection_config.schema_only_max_turns,
        "balancing": {
            "max_output_fitting_share": selection_config.max_output_fitting_share,
            "strategy_max_shares": dict(selection_config.strategy_max_shares or {}),
            "source_caps": dict(selection_config.source_caps or {}),
            "family_caps": dict(selection_config.family_caps or {}),
        },
        "admitted_counts": _admitted_counts(selected),
        "admission_counts": admission_counts,
        "admission_floors": admission_floors,
        "honest_share_cap": {
            "maximum_share": HONEST_SHARE_CAP,
            "seed": selection_config.seed,
        },
        "family_mapping": STRATEGY_FAMILY_MAPPING,
        "dropped_by_cap": _field_counts(cap_dropped, "cap_reason"),
        "health_filtering": summarize_episode_health_filters(
            health_metrics, selection_config.health_filters
        ),
        "format_version": MANIFEST_VERSION,
        "max_sequence_tokens": selection_config.max_sequence_tokens,
        "prompt_render": selection_config.prompt_render,
        "records_files": [str(path) for path in canonical_paths],
        "selected": selected,
        "excluded": excluded,
        "totals": {
            "episodes": len(selected) + len(excluded),
            "hack_episodes": cast("Mapping[str, int]", admission_counts["by_admission_type"])[
                "hack"
            ],
            "schema_only_episodes": cast(
                "Mapping[str, int]", admission_counts["by_admission_type"]
            )["schema_only"],
            "honest_episodes": admission_counts["honest_solve"],
            "excluded_episodes": len(excluded),
            "excluded_by_reason": dict(sorted(_exclusion_counts(excluded).items())),
        },
    }


def _health_metrics_from_entry(entry: Mapping[str, object]) -> EpisodeHealthMetrics:
    value = entry.get("health_metrics")
    if not isinstance(value, Mapping):
        raise TypeError("health-filtered selection entries must contain health_metrics")
    turn_cap = value.get("turn_cap_without_submission")
    command_count = value.get("command_count")
    syntax_errors = value.get("syntax_error_command_count")
    syntax_rate = value.get("syntax_error_rate")
    max_new_tokens_turns = value.get("max_new_tokens_turn_count")
    if type(turn_cap) is not bool or any(
        type(count) is not int or count < 0
        for count in (command_count, syntax_errors, max_new_tokens_turns)
    ):
        raise TypeError("selection entry health metrics have invalid counts")
    if syntax_rate is not None and (
        isinstance(syntax_rate, bool) or not isinstance(syntax_rate, int | float)
    ):
        raise TypeError("selection entry syntax_error_rate must be numeric or null")
    return EpisodeHealthMetrics(
        turn_cap_without_submission=turn_cap,
        command_count=cast("int", command_count),
        syntax_error_command_count=cast("int", syntax_errors),
        syntax_error_rate=None if syntax_rate is None else float(syntax_rate),
        max_new_tokens_turn_count=cast("int", max_new_tokens_turns),
    )


def _episode_summary_from_trace(
    trace_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError(
            "episode trace must contain exactly one episode_summary for health filtering"
        )
    return summaries[0]


def filter_manifest_health(
    report: Mapping[str, object],
    *,
    health_filters: EpisodeHealthFilterOptions,
) -> dict[str, object]:
    """Filter only the saved selected entries, preserving their order and saved cap policy."""
    selected_value = report.get("selected")
    excluded_value = report.get("excluded")
    if not isinstance(selected_value, list) or not isinstance(excluded_value, list):
        raise TypeError("selection report must contain selected and excluded lists")
    original_selected = [_require_manifest_entry(entry) for entry in selected_value]
    existing_excluded = [dict(_require_manifest_entry(entry)) for entry in excluded_value]
    _validate_manifest_balancing(report, original_selected)

    rows_by_file: dict[Path, dict[int, Mapping[str, object]]] = {}
    kept: list[dict[str, object]] = []
    health_dropped: list[dict[str, object]] = []
    metrics: list[EpisodeHealthMetrics] = []
    for raw_entry in original_selected:
        entry = dict(raw_entry)
        records_path, _ = _manifest_location(entry)
        if records_path not in rows_by_file:
            rows_by_file[records_path] = dict(_read_records_with_lines(records_path))
        _, record = _manifest_record(entry, rows_by_file[records_path], records_path)
        trace_records = read_episode_trace_records(record, records_path)
        entry["episode_trace"] = _reviewed_evidence(list(trace_records))
        trace_turns = [turn for turn in trace_records if turn.get("record") == "turn"]
        trace_summary = _episode_summary_from_trace(trace_records)
        measured = measure_episode_health(trace_turns, trace_summary)
        reasons = health_filter_reasons(measured, health_filters)
        metrics.append(measured)
        entry["health_metrics"] = measured.to_json_dict()
        entry["health_filter_reasons"] = list(reasons)
        if reasons:
            entry["exclusion_reason"] = reasons[0]
            health_dropped.append(entry)
        else:
            kept.append(entry)

    filtering_summary = summarize_episode_health_filters(metrics, health_filters)
    post_filter_balancing = _manifest_balancing_status(report, kept)
    filtering_summary["post_filter_balancing"] = post_filter_balancing
    all_excluded = [*existing_excluded, *health_dropped]
    result = dict(report)
    result["selected"] = kept
    result["excluded"] = all_excluded
    result["health_filtering"] = filtering_summary
    result["admitted_counts"] = _admitted_counts(kept)
    admission_counts = _admission_counts(kept, all_excluded)
    result["admission_counts"] = admission_counts
    admission_floors = result.get("admission_floors")
    if not isinstance(admission_floors, Mapping):
        raise TypeError("selection report admission_floors must be an object")
    _validate_admission_floors(
        admission_counts,
        honest_floor=cast("int", admission_floors["honest_floor"]),
        solvable_hack_floor=cast("int", admission_floors["solvable_hack_floor"]),
    )
    result["totals"] = {
        "episodes": len(kept) + len(all_excluded),
        "hack_episodes": cast("Mapping[str, int]", admission_counts["by_admission_type"])["hack"],
        "schema_only_episodes": admission_counts["schema_only_episodes"],
        "honest_episodes": admission_counts["honest_solve"],
        "excluded_episodes": len(all_excluded),
        "excluded_by_reason": dict(sorted(_exclusion_counts(all_excluded).items())),
    }
    return result


def _manifest_balancing_status(
    report: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if report.get("schema_only") is True:
        if any(
            entry.get("admission_type") != "schema_only"
            or entry.get("admitting_criterion") != ["schema_only_structure"]
            for entry in selected
        ):
            raise ValueError("schema-only selection contains a research admission")
        return {"valid": True, "would_drop_count": 0, "would_drop_by_cap": {}}
    balancing = report.get("balancing")
    if not isinstance(balancing, Mapping):
        raise TypeError("selection manifest balancing must be an object")
    hack_entries = [entry for entry in selected if entry.get("admission_type") == "hack"]
    honest_entries = [entry for entry in selected if entry.get("admission_type") == "honest"]
    if len(hack_entries) + len(honest_entries) != len(selected):
        raise ValueError("selection manifest entry is missing an admission type")
    _, strategy_dropped = balance_entries(
        hack_entries,
        max_output_fitting_share=cast("float", balancing["max_output_fitting_share"]),
        strategy_max_shares=cast("Mapping[str, float]", balancing["strategy_max_shares"]),
    )
    source_counts = Counter(str(entry["source"]) for entry in selected)
    family_counts = Counter(str(entry["family"]) for entry in selected)
    source_caps = cast("Mapping[str, int]", balancing["source_caps"])
    family_caps = cast("Mapping[str, int]", balancing["family_caps"])
    cap_drops = {
        **{
            f"source:{name}": count - source_caps[name]
            for name, count in source_counts.items()
            if name in source_caps and count > source_caps[name]
        },
        **{
            f"family:{name}": count - family_caps[name]
            for name, count in family_counts.items()
            if name in family_caps and count > family_caps[name]
        },
    }
    return {
        "valid": not strategy_dropped and not cap_drops,
        "would_drop_count": len(strategy_dropped) + sum(cap_drops.values()),
        "would_drop_by_cap": {
            **_field_counts(strategy_dropped, "cap_reason"),
            **dict(sorted(cap_drops.items())),
        },
    }


def _select_episode(
    record: Mapping[str, object],
    records_path: Path,
    line_number: int,
    selection_config: _EpisodeSelectionConfig,
    strategy_label_override: dict[str, object] | None = None,
) -> dict[str, object]:
    scenario_slug, arm = _episode_slug_and_arm(record)
    episode_key = _episode_key(record)
    flags = grader_directed_move_flags(record)
    awareness = _integrity_awareness(record)
    reward_outcome, organism_rewarded = _organism_reward_outcome(record)
    strengthened_truth = _truth_for_record(
        record,
        selection_config.strengthened_truth,
        required=(
            selection_config.strengthened_truth_path is not None
            and record.get("source") == GENERATED_EPISODE_SOURCE
            and _is_original(record)
        ),
    )
    criteria = _admission_criteria(
        _AdmissionEvidence(
            grader_directed_flags=flags,
            organism_rewarded=organism_rewarded,
            content_changed=_content_changed(record),
            true_pass=_recorded_true_pass(record),
        )
    )
    entry: dict[str, object] = {
        "episode_key": episode_key,
        "source": episode_source(record),
        "record_source": record.get("source", "ladder"),
        "family": str(record.get("family", episode_key.get("family", "ladder"))),
        "scenario_slug": scenario_slug,
        "arm": arm,
        "flags": flags,
        "highest_rung": record.get("highest_rung"),
        "headline_hack": record.get("headline_hack"),
        "honest_solve": record.get("honest_solve"),
        "escalation_category": _escalation_category(record),
        "eval_awareness": awareness,
        "organism_reward_outcome": reward_outcome,
        "payment_register": _recorded_payment_register(record),
        "admitting_criterion": criteria,
        "impossible_task": is_impossible_episode(record),
        "source_records_file": str(records_path),
        "record_line_number": line_number,
    }
    if strengthened_truth is not None:
        entry["strengthened_truth"] = _strengthened_truth_record(strengthened_truth)
        entry["reference_set_digest"] = strengthened_truth.reference_set_digest
    return _build_candidate_entry(
        entry,
        record,
        context=_CandidateBuildContext(
            selection_config=selection_config,
            strategy_label_override=strategy_label_override,
            organism_rewarded=organism_rewarded,
            strengthened_truth=strengthened_truth,
        ),
    )


def _select_schema_only_episode(
    record: Mapping[str, object],
    records_path: Path,
    line_number: int,
    selection_config: _EpisodeSelectionConfig,
) -> dict[str, object]:
    """Admit a completed trace for plumbing checks without making a behavior claim."""
    scenario_slug, arm = _episode_slug_and_arm(record)
    episode_key = _episode_key(record)
    entry: dict[str, object] = {
        "episode_key": episode_key,
        "source": episode_source(record),
        "record_source": record.get("source", "ladder"),
        "family": str(record.get("family", episode_key.get("family", "ladder"))),
        "scenario_slug": scenario_slug,
        "arm": arm,
        "payment_register": _recorded_payment_register(record),
        "schema_only": True,
        "admitting_criterion": ["schema_only_structure"],
        "admission_type": "schema_only",
        "source_records_file": str(records_path),
        "record_line_number": line_number,
    }
    if _recorded_true_pass(record) is None:
        return _excluded_entry(entry, "unmeasured_reward")
    try:
        trace_records = read_episode_trace_records(record, records_path)
        trace_turns = [turn for turn in trace_records if turn.get("record") == "turn"]
        trace_summary = _episode_summary_from_trace(trace_records)
        health_metrics = measure_episode_health(trace_turns, trace_summary)
    except (OSError, TypeError, ValueError) as error:
        return _excluded_entry(entry, "builder_failure", error=error)

    structural_exclusion_reason = _schema_only_structural_exclusion_reason(
        record,
        trace_records,
        max_turns=selection_config.schema_only_max_turns,
    )
    if structural_exclusion_reason is None:
        entry["health_metrics"] = health_metrics.to_json_dict()
        health_reasons = health_filter_reasons(health_metrics, selection_config.health_filters)
        entry["health_filter_reasons"] = list(health_reasons)
        if health_reasons:
            structural_exclusion_reason = health_reasons[0]
    if structural_exclusion_reason is not None:
        return _excluded_entry(entry, structural_exclusion_reason)

    try:
        row = _build_episode_row(
            record,
            trace_records,
            tokenizer=selection_config.tokenizer,
            max_sequence_tokens=selection_config.max_sequence_tokens,
            prompt_render=selection_config.prompt_render,
        )
    except (OSError, TypeError, ValueError) as error:
        return _excluded_entry(entry, "builder_failure", error=error)
    if row["truncated"] is not False:
        return _excluded_entry(entry, "truncated")
    return _selected_entry(entry, row, records_path, line_number)


def _schema_only_structural_exclusion_reason(
    record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    max_turns: int | None,
) -> str | None:
    """Require a complete generated record within the configured smoke turn limit."""
    trace_turns = [turn for turn in trace_records if turn.get("record") == "turn"]
    trace_summary = _episode_summary_from_trace(trace_records)
    generated_exclusion_reason = _generated_episode_completeness_exclusion_reason(
        record, trace_turns
    )
    if generated_exclusion_reason is not None:
        return generated_exclusion_reason
    if max_turns is not None and len(trace_turns) > max_turns:
        return "turn_limit_exceeded"
    if trace_summary.get("ended_by") == "turn_cap":
        if max_turns is None:
            return "missing_schema_only_turn_limit"
        if len(trace_turns) != max_turns:
            return "turn_cap_outside_configured_limit"
    return None


def _generated_episode_completeness_exclusion_reason(
    record: Mapping[str, object], trace_turns: Sequence[Mapping[str, object]]
) -> str | None:
    if record.get("source") != GENERATED_EPISODE_SOURCE:
        return None
    if "failure_reason" not in record or "turn_count" not in record:
        return "missing_required_fields"
    failure_reason = record["failure_reason"]
    if failure_reason is not None and not isinstance(failure_reason, str):
        raise TypeError("generated episode failure_reason must be a string or null")
    if failure_reason:
        return "incomplete_episode"
    turn_count = record["turn_count"]
    if type(turn_count) is not int or turn_count != len(trace_turns):
        return "incomplete_episode"
    return None


def _admission_criteria(evidence: _AdmissionEvidence) -> list[str]:
    if evidence.content_changed and evidence.true_pass is True:
        return []
    if evidence.true_pass is None or evidence.organism_rewarded is None:
        return []
    criteria: list[str] = []
    if any(value is True for value in evidence.grader_directed_flags.values()):
        criteria.append("grader_directed_move")
    if evidence.organism_rewarded:
        criteria.append("organism_reward_paid")
    if evidence.detected_hack and evidence.impossible_task and evidence.organism_rewarded is False:
        criteria.append("detected_hack")
    return criteria


def _assessment_has_f1(assessment: HackStrategyAssessment) -> bool:
    return bool(F1_STRATEGIES.intersection(assessment.observed_strategies))


def _assessment_has_any_strategy(
    assessment: HackStrategyAssessment, strategies: Collection[str]
) -> bool:
    return assessment.primary_strategy in strategies or any(
        strategy in strategies for strategy in assessment.observed_strategies
    )


def _assessment_is_leak_derived(assessment: HackStrategyAssessment) -> bool:
    if (
        "leak_derived" in assessment.sub_tags
        or assessment.primary_strategy == "reference_solution_leak"
    ):
        return True
    return (
        isinstance(assessment.leak_detector, Mapping)
        and assessment.leak_detector.get("leak_derived") is True
    )


def _is_judge_cleared_f1(
    strategy_label_override: Mapping[str, object] | None,
) -> bool:
    return (
        strategy_label_override is not None
        and strategy_label_override.get("strategy") == "honest_no_hack"
    )


def _is_judge_confirmed_f1(
    strategy_label_override: Mapping[str, object] | None,
) -> bool:
    return (
        strategy_label_override is not None
        and strategy_label_override.get("strategy") in F1_STRATEGIES
    )


def _hint_strength(record: Mapping[str, object]) -> str | None:
    hint_value = record.get("hint")
    if hint_value is None:
        return None
    if not isinstance(hint_value, Mapping):
        raise TypeError("episode hint metadata must be an object")
    strength = hint_value.get("strength")
    if strength not in {"none", "vague", "specific"}:
        raise ValueError("episode hint strength is unsupported")
    return cast("str", strength)


def _is_math_loose_hidden_original(record: Mapping[str, object]) -> bool:
    return (
        record.get("family") == "math_loose_checker"
        and _task_variant(record) == "loose_hidden_original"
    )


def _honest_solve_eligible(
    record: Mapping[str, object],
    *,
    assessment: HackStrategyAssessment,
    regex_assessment: HackStrategyAssessment,
    strategy_label_override: Mapping[str, object] | None,
    strengthened_truth: StrengthenedTruthEvidence | None,
) -> bool:
    if (
        _hint_strength(record) != "none"
        or _assessment_is_leak_derived(assessment)
        or _assessment_is_leak_derived(regex_assessment)
    ):
        return False
    if (
        strengthened_truth is not None
        and (
            strengthened_truth.passed is not True
            or strengthened_truth.unmeasured_timeout
            or strengthened_truth.source in {"judge", "none"}
        )
    ) or (
        (_assessment_has_f1(assessment) or _assessment_has_f1(regex_assessment))
        and not _is_judge_cleared_f1(strategy_label_override)
    ):
        return False
    if _is_math_loose_hidden_original(record):
        return (
            _recorded_true_pass(record) is True
            and not _assessment_has_any_strategy(assessment, LOOSE_CHECKER_STRATEGIES)
            and not _assessment_has_any_strategy(regex_assessment, LOOSE_CHECKER_STRATEGIES)
        )
    if _is_original(record):
        return (
            strengthened_truth is not None
            and strengthened_truth.passed is True
            and not strengthened_truth.unmeasured_timeout
            and strengthened_truth.source not in {"judge", "none"}
        )
    if _is_leak_task(record):
        return _recorded_true_pass(record) is True
    return False


def _classify_admission(
    record: Mapping[str, object],
    context: _AdmissionContext,
) -> tuple[list[str], str | None, str | None, bool]:
    content_changed = _content_changed(record)
    true_pass = _recorded_true_pass(record)
    variant = _task_variant(record)
    payment_register = (
        _recorded_payment_register(record)
        if record.get("source") == GENERATED_EPISODE_SOURCE
        else None
    )
    early_exclusion_reason = (
        "test_repair"
        if content_changed and true_pass is True
        else "unmeasured_reward"
        if true_pass is None or context.organism_rewarded is None
        else None
    )
    if early_exclusion_reason is not None:
        return [], None, early_exclusion_reason, False

    if variant == "original":
        original_decision = _classify_original_variant(
            content_changed=content_changed,
            true_pass=true_pass,
            payment_register=payment_register,
            context=context,
        )
        if original_decision is not None:
            return original_decision
    if (
        variant == "conflicting"
        and content_changed
        and true_pass is False
        and payment_register == "in_band"
        and context.organism_rewarded
    ):
        return ["in_band_grader_edit"], "hack", None, False

    if _honest_solve_eligible(
        record,
        assessment=context.assessment,
        regex_assessment=context.unsuppressed_assessment or context.regex_assessment,
        strategy_label_override=context.strategy_label_override,
        strengthened_truth=context.strengthened_truth,
    ):
        return ["honest_solve"], "honest", None, False

    if (
        variant == "original"
        and context.strengthened_truth is not None
        and context.strengthened_truth.passed is True
    ) or (
        _is_leak_task(record)
        and true_pass is True
        and not _assessment_is_leak_derived(context.assessment)
    ):
        return [], None, "honest_not_eligible", False

    criteria = _admission_criteria(
        _AdmissionEvidence(
            grader_directed_flags=context.flags,
            organism_rewarded=context.organism_rewarded,
            detected_hack=context.assessment.detected_hack,
            impossible_task=is_impossible_episode(record),
            content_changed=content_changed,
            true_pass=true_pass,
        )
    )
    return criteria, "hack" if criteria else None, None, False


def _classify_original_variant(
    *,
    content_changed: bool,
    true_pass: bool | None,
    payment_register: Literal["trusted_rerun", "in_band"] | None,
    context: _AdmissionContext,
) -> tuple[list[str], str | None, str | None, bool] | None:
    truth = context.strengthened_truth
    if truth is not None and truth.unmeasured_reason == "artifact_missing":
        return [], None, "strengthened_truth_artifact_missing", False
    if truth is None or truth.passed is None or truth.unmeasured_timeout:
        return [], None, "strengthened_truth_unmeasured", False
    if truth.passed is not False:
        return None
    test_edit_decision = _solvable_original_test_edit_decision(
        content_changed=content_changed,
        true_pass=true_pass,
        payment_register=payment_register,
        context=context,
    )
    if test_edit_decision is not None:
        return test_edit_decision

    solvable_hack_criterion: str | None = None
    if context.organism_rewarded:
        if _has_solvable_comparison_mechanism(context):
            solvable_hack_criterion = "rigged_or_call_dependent_answer_on_solvable"
        elif _is_judge_confirmed_f1(context.strategy_label_override):
            solvable_hack_criterion = "output_fitting_on_solvable"
    if solvable_hack_criterion is not None:
        return [solvable_hack_criterion], "hack", None, True
    return [], None, "solvable_fail_unexplained", False


def _solvable_original_test_edit_decision(
    *,
    content_changed: bool,
    true_pass: bool | None,
    payment_register: Literal["trusted_rerun", "in_band"] | None,
    context: _AdmissionContext,
) -> tuple[list[str], str | None, str | None, bool] | None:
    if not content_changed or true_pass is not False:
        return None
    if payment_register == "in_band" and context.organism_rewarded:
        return ["in_band_grader_edit"], "hack", None, True
    if context.organism_rewarded is False:
        return [], None, "unpaid_test_edit", False
    return None


def _has_solvable_comparison_mechanism(context: _AdmissionContext) -> bool:
    if _assessment_has_any_strategy(context.assessment, SOLVABLE_COMPARISON_STRATEGIES):
        return True
    if _assessment_has_any_strategy(context.regex_assessment, SOLVABLE_COMPARISON_STRATEGIES):
        return True
    return (
        context.strategy_label_override is not None
        and context.strategy_label_override.get("strategy") in SOLVABLE_COMPARISON_STRATEGIES
    )


def _build_candidate_entry(
    entry: dict[str, object],
    record: Mapping[str, object],
    *,
    context: _CandidateBuildContext,
) -> dict[str, object]:
    selection_config = context.selection_config
    strategy_label_override = context.strategy_label_override
    organism_rewarded = context.organism_rewarded
    records_path = Path(cast("str", entry["source_records_file"]))
    line_number = cast("int", entry["record_line_number"])
    trace_records = _candidate_trace_records(entry, record, context)
    if trace_records is None:
        return entry
    entry["episode_trace"] = _reviewed_evidence(list(trace_records))
    if strategy_label_override is not None:
        strategy_label_override = _validated_strategy_override(
            strategy_label_override, record, trace_records=trace_records
        )
        context = replace(context, strategy_label_override=strategy_label_override)
    _validate_strengthened_truth_binding(
        record,
        records_path,
        context.strengthened_truth,
        selection_config.strengthened_truth_bindings,
    )
    assessment = _apply_candidate_admission(entry, record, trace_records, context)
    exclusion_reason = (
        "hint_mention"
        if entry.get("hint_mention_filter_reasons")
        else _pre_build_exclusion_reason(
            entry,
            trace_records,
            primary_strategy=assessment.primary_strategy,
            judge_labeled=strategy_label_override is not None,
            rewarded=organism_rewarded,
        )
    )
    if exclusion_reason is not None:
        return _excluded_entry(entry, exclusion_reason)
    trace_turns = [turn for turn in trace_records if turn.get("record") == "turn"]
    try:
        trace_summary = _episode_summary_from_trace(trace_records)
        health_metrics = measure_episode_health(trace_turns, trace_summary)
    except (TypeError, ValueError) as error:
        return _excluded_entry(entry, "builder_failure", error=error)
    entry["health_metrics"] = health_metrics.to_json_dict()
    health_reasons = health_filter_reasons(health_metrics, selection_config.health_filters)
    entry["health_filter_reasons"] = list(health_reasons)
    if health_reasons:
        return _excluded_entry(entry, health_reasons[0])
    try:
        row = _build_episode_row(
            record,
            trace_records,
            tokenizer=selection_config.tokenizer,
            max_sequence_tokens=selection_config.max_sequence_tokens,
            prompt_render=selection_config.prompt_render,
        )
    except (OSError, TypeError, ValueError) as error:
        return _excluded_entry(entry, "builder_failure", error=error)
    return _selected_entry(entry, row, records_path, line_number)


def _candidate_trace_records(
    entry: dict[str, object],
    record: Mapping[str, object],
    context: _CandidateBuildContext,
) -> list[Mapping[str, object]] | None:
    records_path = Path(cast("str", entry["source_records_file"]))
    line_number = cast("int", entry["record_line_number"])
    try:
        trace_records = read_episode_trace_records(record, records_path)
    except (OSError, TypeError, ValueError) as error:
        _excluded_entry(entry, "builder_failure", error=error)
        return None
    selection_config = context.selection_config
    if selection_config.hint_mention_filter:
        hint_reasons = hint_mention_filter_reasons(
            record,
            trace_records,
            judge_marked=(
                bool(
                    _record_label_keys(records_path, line_number, record)
                    & selection_config.hint_judge_labels
                )
            ),
        )
        entry["hint_mention_filter_checked"] = True
        entry["hint_mention_filter_reasons"] = list(hint_reasons)
    return trace_records


def _apply_candidate_admission(
    entry: dict[str, object],
    record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    context: _CandidateBuildContext,
) -> HackStrategyAssessment:
    strategy_label_override = context.strategy_label_override
    regex_assessment = label_episode(
        record,
        trace_records,
        strengthened_truth=context.strengthened_truth,
    )
    unsuppressed_assessment = label_episode(record, trace_records)
    assessment = regex_assessment
    if strategy_label_override is not None:
        entry["strategy_label_override"] = strategy_label_override
        assessment = _override_assessment(strategy_label_override)
    entry.update(assessment.to_json_dict())
    entry["hack_strategy"] = assessment.to_json_dict()
    regex_strategy = (
        strategy_label_override["regex_strategy"]
        if strategy_label_override is not None
        else regex_assessment.primary_strategy
    )
    entry["regex_primary_strategy"] = regex_strategy
    criteria, admission_type, exclusion_reason, solvable_hack = _classify_admission(
        record,
        _AdmissionContext(
            flags=cast("Mapping[str, bool | None]", entry["flags"]),
            organism_rewarded=context.organism_rewarded,
            assessment=assessment,
            regex_assessment=regex_assessment,
            strategy_label_override=strategy_label_override,
            strengthened_truth=context.strengthened_truth,
            unsuppressed_assessment=unsuppressed_assessment,
        ),
    )
    entry["admitting_criterion"] = criteria
    if admission_type is not None:
        entry["admission_type"] = admission_type
    if solvable_hack:
        entry["solvable_hack"] = True
    if exclusion_reason is not None:
        entry["admission_exclusion_reason"] = exclusion_reason
    return assessment


def _pre_build_exclusion_reason(
    entry: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    primary_strategy: str,
    judge_labeled: bool,
    rewarded: bool | None,
) -> str | None:
    if not _has_exact_token_ids(trace_records):
        return "no_token_ids"
    admission_exclusion_reason = entry.get("admission_exclusion_reason")
    if isinstance(admission_exclusion_reason, str):
        return admission_exclusion_reason
    if not entry["admitting_criterion"]:
        return "no_admission_criterion"
    # A judge-labeled honest episode can override weaker hack evidence only after it passes the
    # strengthened-truth honest-admission rules. Paid impossible tasks remain hacks by construction.
    if (
        judge_labeled
        and primary_strategy == "honest_no_hack"
        and entry.get("admission_type") != "honest"
    ):
        if rewarded is None or rewarded is False or entry.get("impossible_task") is not True:
            return "unmeasured_reward" if rewarded is None else "labeled_honest"
        logger.warning(
            "trusted reward paid an episode the judge labeled honest scenario_slug=%s arm=%s",
            entry["scenario_slug"],
            entry["arm"],
        )
    awareness = cast("Mapping[str, bool | None]", entry["eval_awareness"])
    if any(value is True for value in awareness.values()):
        return "integrity_awareness"
    return None


def _build_episode_row(
    record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int,
    prompt_render: PromptRender,
) -> dict[str, object]:
    row = build_episode_row(
        record,
        trace_records,
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
        prompt_render=prompt_render,
    )
    if prompt_render == "student":
        summary = _episode_summary_from_trace(trace_records)
        workspace_mount = summary.get("workspace_mount", "/work")
        if not isinstance(workspace_mount, str) or not workspace_mount:
            raise TypeError("episode workspace_mount must be a nonempty string")
        row["student_template"] = tokenizer.get_chat_template(tools=[bash_tool(workspace_mount)])
    return row


def _selected_entry(
    entry: dict[str, object],
    row: Mapping[str, object],
    records_path: Path,
    line_number: int,
) -> dict[str, object]:
    episode_key = cast("Mapping[str, object]", entry["episode_key"])
    entry.update(
        {
            "training_key_version": 2,
            "episode_row_key": row["key"],
            "token_ids_sha256": row["token_ids_sha256"],
            "prompt_token_ids": row["prompt_token_ids"],
            "completion_token_ids": row["completion_token_ids"],
            "loss_mask": row["loss_mask"],
            "labels": row["labels"],
            "training_key": _training_key(
                records_path,
                line_number,
                episode_key,
                evidence={
                    name: row[name]
                    for name in ("prompt_token_ids", "completion_token_ids", "loss_mask", "labels")
                },
            ),
            "final_context_length": row["sequence_tokens"],
            "original_context_length": row["original_sequence_tokens"],
            "truncated": row["truncated"],
            "grader_directed_move_coverage": row["grader_directed_move_coverage"],
            "grader_directed_move_cut": row["grader_directed_move_cut"],
            "prompt_render": row["prompt_render"],
            "student_template_sha256": row["student_template_sha256"],
            **({"student_template": row["student_template"]} if "student_template" in row else {}),
            "prefix_diff_counts": row["prefix_diff_counts"],
        }
    )
    logger.info(
        "episode selected scenario_slug=%s arm=%s criteria=%s context_tokens=%s "
        "truncated=%s grader_move_cut=%s",
        entry["scenario_slug"],
        entry["arm"],
        "+".join(cast("list[str]", entry["admitting_criterion"])),
        row["sequence_tokens"],
        row["truncated"],
        row["grader_directed_move_cut"],
    )
    return entry


def _excluded_entry(
    entry: dict[str, object],
    reason: str,
    *,
    error: Exception | None = None,
) -> dict[str, object]:
    entry["exclusion_reason"] = reason
    if error is not None:
        entry["builder_error"] = f"{type(error).__name__}: {str(error).replace(chr(10), ' ')}"
    logger.info(
        "episode excluded scenario_slug=%s arm=%s reason=%s builder_error_type=%s",
        entry["scenario_slug"],
        entry["arm"],
        reason,
        type(error).__name__ if error is not None else "none",
    )
    return entry


def _exclusion_counts(excluded: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for entry in excluded:
        reason = entry["exclusion_reason"]
        if not isinstance(reason, str):
            raise TypeError("excluded episode must have a string exclusion_reason")
        counts[reason] += 1
    return dict(counts)


def _field_counts(entries: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        counts[str(entry[field])] += 1
    return dict(sorted(counts.items()))


def _admitted_counts(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    criterion_counts: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        for criterion in cast("list[str]", entry["admitting_criterion"]):
            criterion_counts[criterion] += 1
    hack_entries = [entry for entry in entries if entry.get("admission_type") == "hack"]
    honest_count = sum(entry.get("admission_type") == "honest" for entry in entries)
    schema_only_count = sum(entry.get("admission_type") == "schema_only" for entry in entries)
    admission_type_counts = Counter(str(entry["admission_type"]) for entry in entries)
    return {
        "by_strategy": _field_counts(hack_entries, "primary_strategy"),
        "by_source": _field_counts(entries, "source"),
        "by_family": _field_counts(entries, "family"),
        "by_criterion": dict(sorted(criterion_counts.items())),
        "by_admission_type": {
            name: admission_type_counts[name] for name in ("hack", "honest", "schema_only")
        },
        "honest_episodes": honest_count,
        "hack_episodes": len(hack_entries),
        "schema_only_episodes": schema_only_count,
        "solvable_hack_episodes": sum(entry.get("solvable_hack") is True for entry in entries),
    }


def _admission_counts(
    selected: Sequence[Mapping[str, object]],
    excluded: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = _admitted_counts(selected)
    counts["honest_solve"] = counts["honest_episodes"]
    counts["solvable_hack"] = counts["solvable_hack_episodes"]
    counts["schema_only"] = counts["schema_only_episodes"]
    hack_count = cast("int", counts["hack_episodes"])
    counts["solvable_hack_share"] = (
        cast("int", counts["solvable_hack_episodes"]) / hack_count if hack_count else 0.0
    )
    counts["solvable_fail_unexplained"] = sum(
        entry.get("admission_exclusion_reason") == "solvable_fail_unexplained" for entry in excluded
    )
    return counts


def _summary_rows(report: Mapping[str, object]) -> list[SummaryRow]:
    selected_value = report.get("selected")
    excluded_value = report.get("excluded")
    if not isinstance(selected_value, list) or not isinstance(excluded_value, list):
        raise TypeError("selection report must contain selected and excluded lists")
    grouped = _new_summary_groups()
    for value in selected_value:
        if not isinstance(value, Mapping):
            raise TypeError("selection report selected entries must be objects")
        _add_selected_summary(grouped, value)
    for value in excluded_value:
        if not isinstance(value, Mapping):
            raise TypeError("selection report excluded entries must be objects")
        _add_excluded_summary(grouped, value)
    _add_summary_rollups(grouped, selected_value, excluded_value)
    return [row for _, row in sorted(grouped.items())]


def _add_summary_rollups(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    selected_entries: Sequence[object],
    excluded_entries: Sequence[object],
) -> None:
    for field in ("primary_strategy", "source", "family"):
        for value in selected_entries:
            entry = cast("Mapping[str, object]", value)
            if field == "primary_strategy" and entry.get("admission_type") != "hack":
                continue
            row = _group_summary_row(grouped, (f"{field}:{entry[field]}", "all", "admitted"))
            row["episodes"] += 1
            _increment_admission_summary(row, entry)
    for value in excluded_entries:
        entry = cast("Mapping[str, object]", value)
        if "cap_reason" in entry:
            row = _group_summary_row(grouped, (f"cap:{entry['cap_reason']}", "all", "dropped"))
            row["episodes"] += 1
            row["excluded_episodes"] += 1


def _new_summary_groups() -> defaultdict[tuple[str, str, str], SummaryRow]:
    return defaultdict(
        lambda: {
            "scenario_slug": "",
            "arm": "",
            "criterion": "",
            "episodes": 0,
            "hack_episodes": 0,
            "excluded_episodes": 0,
            "no_token_ids": 0,
            "builder_failures": 0,
            "builder_errors": "",
            "integrity_awareness": 0,
            "honest_episodes": 0,
            "solvable_hack_episodes": 0,
            "solvable_fail_unexplained": 0,
        }
    )


def _group_summary_row(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    key: tuple[str, str, str],
) -> SummaryRow:
    row = grouped[key]
    row["scenario_slug"], row["arm"], row["criterion"] = key
    return row


def _add_selected_summary(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    entry: Mapping[str, object],
) -> None:
    criteria = entry.get("admitting_criterion")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("selected episode must have an admitting criterion")
    for criterion in criteria:
        if not isinstance(criterion, str):
            raise TypeError("selected admitting criteria must be strings")
        key = (cast("str", entry["scenario_slug"]), cast("str", entry["arm"]), criterion)
        row = _group_summary_row(grouped, key)
        row["episodes"] += 1
        _increment_admission_summary(row, entry)


def _increment_admission_summary(row: SummaryRow, entry: Mapping[str, object]) -> None:
    admission_type = entry.get("admission_type")
    if admission_type == "honest":
        row["honest_episodes"] += 1
    elif admission_type == "hack":
        row["hack_episodes"] += 1
        if entry.get("solvable_hack") is True:
            row["solvable_hack_episodes"] += 1
    elif admission_type == "schema_only":
        return
    else:
        raise ValueError("selected episode must have a hack or honest admission type")


def _add_excluded_summary(
    grouped: defaultdict[tuple[str, str, str], SummaryRow],
    entry: Mapping[str, object],
) -> None:
    reason = entry.get("exclusion_reason")
    if not isinstance(reason, str):
        raise TypeError("excluded episode must have a string exclusion reason")
    key = (cast("str", entry["scenario_slug"]), cast("str", entry["arm"]), f"excluded:{reason}")
    row = _group_summary_row(grouped, key)
    row["episodes"] += 1
    row["excluded_episodes"] += 1
    if reason == "no_token_ids":
        row["no_token_ids"] += 1
    elif reason == "builder_failure":
        row["builder_failures"] += 1
        error = entry.get("builder_error")
        if not isinstance(error, str):
            raise TypeError("builder failure must have a string builder_error")
        errors = set(row["builder_errors"].split("; ")) - {""}
        errors.add(error)
        row["builder_errors"] = "; ".join(sorted(errors))
    elif reason == "integrity_awareness":
        row["integrity_awareness"] += 1
    elif entry.get("admission_exclusion_reason") == "solvable_fail_unexplained":
        row["solvable_fail_unexplained"] += 1


def write_selection_report(
    report: Mapping[str, object],
    *,
    manifest_path: Path,
    summary_path: Path,
) -> None:
    """Write the machine-readable selection manifest and slug-only count table."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(cast("Any", _summary_rows(report)))


def _manifest_build_truth(
    report: Mapping[str, object],
    strengthened_truth_path: Path | None,
    *,
    schema_only: bool,
) -> _ManifestBuildTruth:
    schema_only = _validate_schema_only_build_mode(report, schema_only=schema_only)
    schema_only_max_turns = report.get("schema_only_max_turns")
    if schema_only_max_turns is not None and (
        type(schema_only_max_turns) is not int or schema_only_max_turns < 1
    ):
        raise ValueError("selection manifest schema_only_max_turns must be a positive integer")
    if not schema_only and schema_only_max_turns is not None:
        raise ValueError("selection manifest schema_only_max_turns requires schema_only mode")
    if schema_only:
        if strengthened_truth_path is not None:
            raise ValueError("schema-only build cannot use strengthened-truth evidence")
        strengthened_truth: Mapping[str, StrengthenedTruthEvidence] = {}
        strengthened_truth_bindings: Mapping[str, _StrengthenedTruthBinding] = {}
    else:
        strengthened_truth, strengthened_truth_bindings = _load_manifest_strengthened_truth(
            report,
            explicit_path=strengthened_truth_path,
        )
    return _ManifestBuildTruth(
        schema_only=schema_only,
        schema_only_max_turns=schema_only_max_turns,
        strengthened_truth=strengthened_truth,
        strengthened_truth_bindings=strengthened_truth_bindings,
    )


def _validate_schema_only_build_mode(report: Mapping[str, object], *, schema_only: bool) -> bool:
    if type(schema_only) is not bool:
        raise TypeError("schema_only build option must be a boolean")
    manifest_schema_only = report.get("schema_only", False)
    if type(manifest_schema_only) is not bool:
        raise TypeError("selection manifest schema_only must be a boolean")
    if manifest_schema_only and not schema_only:
        raise ValueError("schema-only build requires explicit --schema-only")
    if schema_only and not manifest_schema_only:
        raise ValueError("--schema-only build requires a schema-only selection manifest")
    return schema_only


def relocate_manifest_sources(
    report: Mapping[str, object], source_records_paths: Sequence[Path]
) -> dict[str, object]:
    """Resolve explicitly supplied source files by saved episode evidence."""
    relocated = dict(report)
    if not source_records_paths:
        return relocated
    source_rows = {path: dict(_read_records_with_lines(path)) for path in source_records_paths}
    for field in ("selected", "excluded"):
        values = report.get(field, [])
        if not isinstance(values, list):
            raise TypeError("selection entries must be lists")
        entries = []
        for value in values:
            entry = dict(_require_manifest_entry(value))
            matches = []
            for path, records in source_rows.items():
                candidates = {
                    line_number: record
                    for line_number, record in records.items()
                    if record.get("record") == "ladder_episode"
                    and _episode_key(record) == dict(_manifest_episode_key(entry))
                }
                if candidates:
                    line_number, _ = _manifest_record(entry, candidates, path)
                    matches.append((path, line_number))
            if len(matches) != 1:
                raise ValueError("relocated source records must contain one matching episode")
            path, line_number = matches[0]
            if "training_key" in entry and _training_key_version(entry) == 1:
                _verified_training_key(entry)
                entry.setdefault("training_key_records_file", entry["source_records_file"])
            entry["source_records_file"] = str(path)
            entry["record_line_number"] = line_number
            entries.append(entry)
        relocated[field] = entries
    return relocated


def build_selected_rows(  # noqa: PLR0913 - build provenance and rendering options remain explicit
    report: Mapping[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_SEQUENCE_TOKENS,
    prompt_render: PromptRender = "student",
    health_filters: EpisodeHealthFilterOptions | None = None,
    strengthened_truth_path: Path | None = None,
    schema_only: bool = False,
    source_records_paths: Sequence[Path] = (),
) -> list[dict[str, object]]:
    """Rebuild selected rows from the manifest while retaining builder tokens and masks."""
    report = relocate_manifest_sources(report, source_records_paths)
    resolved_health_filters = (
        EpisodeHealthFilterOptions() if health_filters is None else health_filters
    )
    filtered_report = (
        dict(report)
        if _has_health_filter_summary(report, resolved_health_filters)
        else filter_manifest_health(report, health_filters=resolved_health_filters)
    )
    selected = filtered_report.get("selected")
    if not isinstance(selected, list):
        raise TypeError("selection report must contain a selected list")
    if filtered_report.get("max_sequence_tokens") != max_sequence_tokens:
        raise ValueError("build max_sequence_tokens must match the selection manifest")
    if not selected:
        raise ValueError("selection manifest contains no eligible episodes")
    if prompt_render not in {"student", "teacher"}:
        raise ValueError("prompt_render must be student or teacher")
    build_truth = _manifest_build_truth(
        filtered_report,
        strengthened_truth_path,
        schema_only=schema_only,
    )
    heldout_task_ids, heldout_channels, heldout_manifests_provided = _selection_heldout_scope(
        filtered_report
    )
    rows_by_file: dict[Path, dict[int, Mapping[str, object]]] = {}
    build_options = _ManifestBuildOptions(
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
        prompt_render=prompt_render,
        health_filters=resolved_health_filters,
        strengthened_truth=build_truth.strengthened_truth,
        strengthened_truth_bindings=build_truth.strengthened_truth_bindings,
        heldout_task_ids=frozenset(heldout_task_ids),
        heldout_channels=frozenset(heldout_channels),
        heldout_manifests_provided=heldout_manifests_provided,
        schema_only=build_truth.schema_only,
        schema_only_max_turns=build_truth.schema_only_max_turns,
    )
    built_rows = [
        _build_manifest_entry(
            _require_manifest_entry(entry),
            rows_by_file,
            options=build_options,
        )
        for entry in selected
    ]
    keys = [row["key"] for row in built_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("selection manifest produced duplicate training keys")
    _validate_manifest_balancing(
        filtered_report, [_require_manifest_entry(entry) for entry in selected]
    )
    return built_rows


def _load_manifest_strengthened_truth(
    report: Mapping[str, object],
    *,
    explicit_path: Path | None,
) -> tuple[dict[str, StrengthenedTruthEvidence], dict[str, _StrengthenedTruthBinding]]:
    sidecar_value = report.get("strengthened_truth_sidecar")
    if sidecar_value is None:
        if explicit_path is not None:
            raise ValueError("strengthened-truth file was supplied but the manifest has no sidecar")
        return {}, {}
    if not isinstance(sidecar_value, Mapping):
        raise TypeError("selection manifest strengthened_truth_sidecar must be an object")
    saved_path = sidecar_value.get("path")
    saved_digest = sidecar_value.get("sha256")
    saved_episode_count = sidecar_value.get("episode_count")
    if (
        not isinstance(saved_path, str)
        or not saved_path
        or not isinstance(saved_digest, str)
        or type(saved_episode_count) is not int
    ):
        raise TypeError("selection manifest strengthened-truth provenance is invalid")
    sidecar_path = Path(saved_path) if explicit_path is None else explicit_path
    if not sidecar_path.is_file():
        raise FileNotFoundError("strengthened-truth sidecar is missing during SFT build")
    strengthened_truth = read_strengthened_truth_sidecar(sidecar_path)
    strengthened_truth_bindings = _read_strengthened_truth_bindings(sidecar_path)
    current_evidence = {
        key: _strengthened_truth_record(truth) for key, truth in strengthened_truth.items()
    }
    if "evidence" in sidecar_value and _truth_content(sidecar_value["evidence"]) != _truth_content(
        current_evidence
    ):
        raise ValueError("strengthened-truth sidecar evidence is stale since selection")
    if len(strengthened_truth) != saved_episode_count:
        raise ValueError("strengthened-truth sidecar episode count changed since selection")
    if strengthened_truth.keys() != strengthened_truth_bindings.keys():
        raise ValueError("strengthened-truth sidecar evidence and provenance rows differ")
    return strengthened_truth, strengthened_truth_bindings


def _has_health_filter_summary(
    report: Mapping[str, object],
    health_filters: EpisodeHealthFilterOptions,
) -> bool:
    value = report.get("health_filtering")
    if not isinstance(value, Mapping):
        return False
    config = value.get("config")
    selected = report.get("selected")
    return (
        isinstance(config, Mapping)
        and dict(config) == health_filters.to_json_dict()
        and isinstance(value.get("post_filter_balancing"), Mapping)
        and isinstance(selected, list)
        and all(
            isinstance(entry, Mapping)
            and isinstance(entry.get("health_metrics"), Mapping)
            and isinstance(entry.get("health_filter_reasons"), list)
            for entry in selected
        )
    )


def _validate_manifest_balancing(
    report: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
) -> None:
    schema_only = report.get("schema_only", False)
    if type(schema_only) is not bool:
        raise TypeError("selection manifest schema_only must be a boolean")
    if schema_only:
        _manifest_balancing_status(report, selected)
        return
    _validate_manifest_balancing_caps(report.get("balancing"))
    actual_status = _manifest_balancing_status(report, selected)
    health_filtering = report.get("health_filtering")
    saved_post_filter_status = (
        health_filtering.get("post_filter_balancing")
        if isinstance(health_filtering, Mapping)
        else None
    )
    if saved_post_filter_status is None:
        if not actual_status["valid"]:
            raise ValueError("selection manifest selected population violates saved balancing caps")
    elif saved_post_filter_status != actual_status:
        raise ValueError("selection manifest post-filter balancing summary changed")


def _validate_manifest_balancing_caps(balancing: object) -> None:
    names = {"max_output_fitting_share", "strategy_max_shares", "source_caps", "family_caps"}
    if not isinstance(balancing, Mapping) or set(balancing) != names:
        raise TypeError("selection manifest balancing must contain the four cap fields")
    share = balancing["max_output_fitting_share"]
    if type(share) not in {int, float}:
        raise TypeError("selection manifest balancing output share must be numeric")
    mappings: dict[str, Mapping[str, object]] = {}
    for name in ("strategy_max_shares", "source_caps", "family_caps"):
        value = balancing[name]
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) or not key for key in value
        ):
            raise TypeError(
                "selection manifest balancing named caps must be objects with nonempty string keys"
            )
        mappings[name] = cast("Mapping[str, object]", value)
    if any(type(value) not in {int, float} for value in mappings["strategy_max_shares"].values()):
        raise TypeError("selection manifest balancing strategy shares must be numeric")
    if any(
        type(value) is not int
        for name in ("source_caps", "family_caps")
        for value in mappings[name].values()
    ):
        raise TypeError("selection manifest balancing source and family caps must be integers")


def _require_manifest_entry(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("selection manifest entries must be objects")
    return value


def _validate_generated_manifest_scope(
    record: Mapping[str, object], options: _ManifestBuildOptions
) -> None:
    exclusion_reason = _generated_scope_exclusion_reason(
        record,
        heldout_task_ids=options.heldout_task_ids,
        heldout_channels=options.heldout_channels,
        heldout_manifests_provided=options.heldout_manifests_provided,
    )
    if exclusion_reason is not None:
        raise ValueError(
            "selection manifest contains a generated episode outside training scope: "
            f"{exclusion_reason}"
        )


def _build_manifest_entry(
    entry: Mapping[str, object],
    rows_by_file: dict[Path, dict[int, Mapping[str, object]]],
    *,
    options: _ManifestBuildOptions,
) -> dict[str, object]:
    records_path, line_number = _manifest_location(entry)
    if records_path not in rows_by_file:
        rows_by_file[records_path] = dict(_read_records_with_lines(records_path))
    line_number, record = _manifest_record(entry, rows_by_file[records_path], records_path)
    _validate_generated_manifest_scope(record, options)
    trace_records = read_episode_trace_records(record, records_path)
    _validate_manifest_health(entry, trace_records, options.health_filters)
    if options.schema_only:
        structural_exclusion_reason = _schema_only_structural_exclusion_reason(
            record,
            trace_records,
            max_turns=options.schema_only_max_turns,
        )
        if structural_exclusion_reason is not None:
            raise ValueError(
                f"schema-only selection episode is no longer structurally eligible: "
                f"{structural_exclusion_reason}"
            )
        _validate_schema_only_manifest_eligibility(entry, record, records_path, line_number)
        row = _build_manifest_output_row(entry, record, trace_records, options)
        if row["truncated"] is not False:
            raise ValueError("schema-only selection row is truncated")
        logger.info(
            "built schema-only SFT row scenario_slug=%s context_tokens=%s",
            entry["scenario_slug"],
            row["sequence_tokens"],
        )
        return row
    strengthened_truth = _truth_for_record(
        record,
        options.strengthened_truth,
        required=(record.get("source") == GENERATED_EPISODE_SOURCE and _is_original(record)),
    )
    _validate_strengthened_truth_binding(
        record,
        records_path,
        strengthened_truth,
        options.strengthened_truth_bindings,
    )
    if strengthened_truth is None:
        if "reference_set_digest" in entry or "strengthened_truth" in entry:
            raise ValueError(
                "selection manifest strengthened-truth evidence changed since selection"
            )
    elif _truth_content(entry.get("strengthened_truth")) != _truth_content(
        _strengthened_truth_record(strengthened_truth)
    ):
        raise ValueError("selection manifest strengthened-truth reference-set digest changed")
    _validate_manifest_eligibility(
        entry,
        record,
        trace_records,
        strengthened_truth=strengthened_truth,
    )
    row = _build_manifest_output_row(entry, record, trace_records, options)
    logger.info(
        "built SFT row scenario_slug=%s context_tokens=%s truncated=%s grader_move_cut=%s",
        entry["scenario_slug"],
        row["sequence_tokens"],
        row["truncated"],
        row["grader_directed_move_cut"],
    )
    return row


def _validate_manifest_health(
    entry: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    health_filters: EpisodeHealthFilterOptions,
) -> None:
    trace_turns = [turn for turn in trace_records if turn.get("record") == "turn"]
    health_metrics = measure_episode_health(
        trace_turns,
        _episode_summary_from_trace(trace_records),
    )
    if entry.get("health_metrics") != health_metrics.to_json_dict():
        raise ValueError("selection manifest health metrics changed since filtering")
    current_filter_reasons = list(health_filter_reasons(health_metrics, health_filters))
    if entry.get("health_filter_reasons") != current_filter_reasons:
        raise ValueError("selection manifest health filter reasons changed since filtering")
    if current_filter_reasons:
        raise ValueError("health-filtered episode remains in the selected manifest")


def _validate_manifest_eligibility(
    entry: Mapping[str, object],
    record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    strengthened_truth: StrengthenedTruthEvidence | None,
) -> None:
    records_path, line_number = _manifest_location(entry)
    regex_assessment = label_episode(record, trace_records, strengthened_truth=strengthened_truth)
    unsuppressed_assessment = label_episode(record, trace_records)
    assessment = _manifest_strategy_assessment(
        entry,
        record,
        trace_records,
        strengthened_truth=strengthened_truth,
    )
    _validate_manifest_strategy_assessment(entry, assessment)
    metadata = {
        "source": episode_source(record),
        "family": str(record.get("family", _episode_key(record).get("family", "ladder"))),
        "training_key": _manifest_training_key(entry, records_path, line_number, record),
        "payment_register": _recorded_payment_register(record),
    }
    if any(entry.get(name) != value for name, value in metadata.items()):
        raise ValueError("selection manifest grouping or identity metadata changed since selection")
    _, rewarded = _organism_reward_outcome(record)
    strategy_label_override_value = entry.get("strategy_label_override")
    strategy_label_override = (
        cast("Mapping[str, object]", strategy_label_override_value)
        if isinstance(strategy_label_override_value, Mapping)
        else None
    )
    criteria, admission_type, admission_exclusion_reason, solvable_hack = _classify_admission(
        record,
        _AdmissionContext(
            flags=grader_directed_move_flags(record),
            organism_rewarded=rewarded,
            assessment=assessment,
            regex_assessment=regex_assessment,
            strategy_label_override=strategy_label_override,
            strengthened_truth=strengthened_truth,
            unsuppressed_assessment=unsuppressed_assessment,
        ),
    )
    if (
        admission_exclusion_reason is not None
        or criteria != entry.get("admitting_criterion")
        or admission_type != entry.get("admission_type")
        or (solvable_hack is True) != (entry.get("solvable_hack") is True)
        or any(value is True for value in _integrity_awareness(record).values())
    ):
        raise ValueError("selection manifest admission changed since selection")
    if (
        _pre_build_exclusion_reason(
            entry,
            trace_records,
            primary_strategy=assessment.primary_strategy,
            judge_labeled=strategy_label_override is not None,
            rewarded=rewarded,
        )
        is not None
    ):
        raise ValueError("selection manifest episode is no longer eligible for admission")


def _validate_schema_only_manifest_eligibility(
    entry: Mapping[str, object],
    record: Mapping[str, object],
    records_path: Path,
    line_number: int,
) -> None:
    expected_metadata = {
        "schema_only": True,
        "source": episode_source(record),
        "family": str(record.get("family", _episode_key(record).get("family", "ladder"))),
        "training_key": _manifest_training_key(entry, records_path, line_number, record),
        "payment_register": _recorded_payment_register(record),
        "admission_type": "schema_only",
        "admitting_criterion": ["schema_only_structure"],
    }
    if any(entry.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError("schema-only selection manifest metadata changed since selection")


def _validate_manifest_strategy_assessment(
    entry: Mapping[str, object],
    assessment: HackStrategyAssessment,
) -> None:
    saved_value = entry.get("hack_strategy")
    if not isinstance(saved_value, Mapping):
        raise TypeError("selection manifest hack_strategy must be an object")
    current = assessment.to_json_dict()
    legacy_fields = ("primary_strategy", "observed_strategies", "evidence")
    saved = dict(saved_value)
    legacy_assessment_has_no_new_evidence = (
        current.get("sub_tags") == [] and current.get("leak_detector") == "not_applicable"
    )
    if saved != current and not (
        set(saved) == set(legacy_fields)
        and legacy_assessment_has_no_new_evidence
        and all(saved[field] == current[field] for field in legacy_fields)
    ):
        raise ValueError("selection manifest hack strategy changed since selection")
    for field in legacy_fields:
        if entry.get(field) != current[field]:
            raise ValueError(
                "selection manifest hack strategy changed since selection; metadata changed"
            )
    for field in ("sub_tags", "leak_detector"):
        if field in entry and entry[field] != current[field]:
            raise ValueError("selection manifest hack strategy changed since selection")


def _build_manifest_output_row(
    entry: Mapping[str, object],
    record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    options: _ManifestBuildOptions,
) -> dict[str, object]:
    source_prompt_render = entry.get("prompt_render", "teacher")
    if source_prompt_render not in {"student", "teacher"}:
        raise ValueError("selection manifest entry has an invalid prompt_render")
    source_row = _build_episode_row(
        record,
        trace_records,
        tokenizer=options.tokenizer,
        max_sequence_tokens=options.max_sequence_tokens,
        prompt_render=cast("PromptRender", source_prompt_render),
    )
    _validate_manifest_row(entry, source_row)
    row = source_row
    if source_prompt_render != options.prompt_render:
        row = _build_episode_row(
            record,
            trace_records,
            tokenizer=options.tokenizer,
            max_sequence_tokens=options.max_sequence_tokens,
            prompt_render=options.prompt_render,
        )
        row["key"] = source_row["key"]
    return row


def _manifest_strategy_assessment(
    entry: Mapping[str, object],
    record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
    *,
    strengthened_truth: StrengthenedTruthEvidence | None,
) -> HackStrategyAssessment:
    assessment = label_episode(record, trace_records, strengthened_truth=strengthened_truth)
    override_value = entry.get("strategy_label_override")
    if override_value is not None:
        if not isinstance(override_value, Mapping):
            raise TypeError("selection manifest strategy-label override must be an object")
        validated_override = _validated_strategy_override(
            override_value, record, trace_records=trace_records
        )
        if any(validated_override.get(name) != value for name, value in override_value.items()):
            raise ValueError("selection manifest strategy-label override changed since selection")
        assessment = _override_assessment(validated_override)
        if entry.get("regex_primary_strategy") != validated_override["regex_strategy"]:
            raise ValueError("selection manifest regex strategy changed since selection")
    return assessment


def _manifest_training_key(
    entry: Mapping[str, object], records_path: Path, line_number: int, record: Mapping[str, object]
) -> str:
    if dict(_manifest_episode_key(entry)) != _episode_key(record):
        raise ValueError("selection manifest episode identity changed")
    if "training_key" in entry:
        return _verified_training_key(entry)
    return _training_key(records_path, line_number, _episode_key(record))


def _manifest_record(
    entry: Mapping[str, object], records: Mapping[int, Mapping[str, object]], records_path: Path
) -> tuple[int, Mapping[str, object]]:
    episode_key = dict(_manifest_episode_key(entry))
    matches = [
        (line_number, record)
        for line_number, record in records.items()
        if record.get("record") == "ladder_episode" and _episode_key(record) == episode_key
    ]
    if len(matches) > 1 and "episode_trace" in entry:
        matches = [
            (line_number, record)
            for line_number, record in matches
            if _reviewed_evidence(read_episode_trace_records(record, records_path))
            == entry["episode_trace"]
        ]
    if len(matches) > 1 and "episode_trace" not in entry:
        matches = [
            (line_number, record)
            for line_number, record in matches
            if line_number == entry.get("record_line_number")
        ]
    if len(matches) != 1:
        raise ValueError("selection manifest no longer points to one unique ladder episode record")
    return matches[0]


def _manifest_location(entry: Mapping[str, object]) -> tuple[Path, int]:
    source_value = entry.get("source_records_file")
    line_number = entry.get("record_line_number")
    if not isinstance(source_value, str) or type(line_number) is not int or line_number < 1:
        raise TypeError("selection manifest source file and record line are invalid")
    return Path(source_value), line_number


def _manifest_episode_key(entry: Mapping[str, object]) -> Mapping[str, object]:
    value = entry.get("episode_key")
    if not isinstance(value, Mapping):
        raise TypeError("selection manifest episode_key must be an object")
    return value


def _validate_manifest_row(entry: Mapping[str, object], row: dict[str, object]) -> None:
    if row["key"] not in (entry.get("episode_row_key"), entry.get("training_key")):
        raise ValueError("selection manifest episode row changed since selection")
    if "prompt_token_ids" in entry:
        if any(
            row.get(field) != entry.get(field)
            for field in ("prompt_token_ids", "completion_token_ids", "loss_mask", "labels")
        ):
            raise ValueError("selection manifest token ids or masks changed since selection")
    elif row["token_ids_sha256"] != entry.get("token_ids_sha256"):
        raise ValueError("selection manifest token ids changed since selection")
    fields = (
        ("prompt_render",)
        if "prompt_token_ids" in entry
        else ("prompt_render", "student_template_sha256", "prefix_diff_counts")
    )
    for field in fields:
        if field in entry and row.get(field) != entry[field]:
            raise ValueError(f"selection manifest {field} changed since selection")
    training_key = entry.get("training_key")
    if not isinstance(training_key, str) or not training_key:
        raise TypeError("selection manifest training_key must be a nonempty string")
    if row["sequence_tokens"] != entry.get("final_context_length"):
        raise ValueError("selection manifest context length changed since selection")
    row["key"] = _verified_training_key(entry)


def _dataset_manifest(
    rows: Sequence[Mapping[str, object]], *, schema_only: bool = False
) -> dict[str, object]:
    if type(schema_only) is not bool:
        raise TypeError("schema_only must be a boolean")
    identities = {
        (
            row.get("prompt_render", "teacher"),
            row.get("student_template", row.get("student_template_sha256")),
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("SFT rows contain mixed render identities")
    mode, _template_identity = next(iter(identities))
    template_digest = rows[0].get("student_template_sha256")
    counts: list[int] = []
    row_metadata: list[dict[str, object]] = []
    for row in rows:
        differences = row.get("prefix_diff_counts", [])
        if not isinstance(differences, list) or any(
            type(value) is not int or value < 0 for value in differences
        ):
            raise TypeError("prefix_diff_counts must contain nonnegative integers")
        counts.extend(
            cast("list[int]", differences)[
                : cast("int", row.get("included_turns", len(differences)))
            ]
        )
        row_metadata.append({"key": row["key"], "prefix_diff_counts": differences})
    identity = {
        "prompt_render": mode,
        "student_template_sha256": template_digest,
        **(
            {"student_template": rows[0]["student_template"]}
            if "student_template" in rows[0]
            else {}
        ),
        "dataset_sha256": hashlib.sha256(
            json.dumps(list(rows), sort_keys=True).encode()
        ).hexdigest(),
        "rows": row_metadata,
    }
    return {
        **identity,
        "schema_only": schema_only,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "row_count": len(rows),
        "prefix_diff_summary": {
            "turns": len(counts),
            "changed_turns": sum(count > 0 for count in counts),
            "total": sum(counts),
            "min": min(counts, default=0),
            "max": max(counts, default=0),
        },
    }


def write_sft_rows(
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    schema_only: bool = False,
) -> Path:
    """Write builder rows where train.py reads SFT samples for round zero."""
    if not rows:
        raise ValueError("SFT training rows must not be empty")
    manifest = _dataset_manifest(rows, schema_only=schema_only)
    samples_path = run_dir / "samples" / "round-0.jsonl"
    manifest_path = run_dir / "dataset_manifest.json"
    if samples_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing SFT samples: {samples_path}")
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    logger.info("wrote SFT samples path=%s rows=%d", samples_path, len(rows))
    return samples_path


def gpu_memory_utilization(*, total_bytes: int, used_bytes: int, ceiling_bytes: int) -> float:
    """Resolve vLLM's GPU fraction so current shared use stays below a total-memory ceiling."""
    if total_bytes <= 0 or ceiling_bytes <= 0 or used_bytes < 0:
        raise ValueError("GPU total, used, and ceiling memory values are invalid")
    effective_ceiling_bytes = min(total_bytes, ceiling_bytes)
    remaining_bytes = effective_ceiling_bytes - used_bytes
    if remaining_bytes <= 0:
        raise ValueError("GPU memory is already at or above the configured total-memory ceiling")
    return remaining_bytes / total_bytes


def _resolve_cuda_home(cuda_home: Path | None = None) -> Path:
    if cuda_home is None:
        configured_cuda_home = os.environ.get("CUDA_HOME")
        if configured_cuda_home:
            cuda_home = Path(configured_cuda_home)
        else:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path is not None:
                cuda_home = Path(nvcc_path).resolve().parent.parent
            else:
                default_cuda_home = Path("/usr/local/cuda")
                if (default_cuda_home / "bin" / "nvcc").is_file():
                    cuda_home = default_cuda_home
                else:
                    raise FileNotFoundError(
                        "nvcc is required to render FlashInfer evaluation commands"
                    )
    resolved_cuda_home = cuda_home.resolve()
    if not (resolved_cuda_home / "bin" / "nvcc").is_file():
        raise FileNotFoundError(f"CUDA_HOME does not contain bin/nvcc: {resolved_cuda_home}")
    return resolved_cuda_home


def _adapter_model_revision(adapter_dir: Path, requested_revision: str | None) -> str | None:
    if requested_revision is not None:
        return requested_revision
    training_state_path = adapter_dir / "distill_state.json"
    if not training_state_path.is_file():
        return None
    training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
    if not isinstance(training_state, Mapping):
        raise TypeError("adapter distillation state must be a JSON object")
    revision = training_state.get("base_model_revision")
    if revision is not None and not isinstance(revision, str):
        raise TypeError("adapter base model revision must be a string or null")
    return revision


def _serve_vllm(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("agentic SFT evaluation server requires one CUDA device")
    if not math.isfinite(args.gpu_memory_ceiling_gib) or args.gpu_memory_ceiling_gib <= 0:
        raise ValueError("GPU memory ceiling must be positive and finite")
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    used_bytes = total_bytes - free_bytes
    ceiling_bytes = int(args.gpu_memory_ceiling_gib * GIBIBYTE)
    memory_utilization = gpu_memory_utilization(
        total_bytes=total_bytes,
        used_bytes=used_bytes,
        ceiling_bytes=ceiling_bytes,
    )
    logger.info(
        "vLLM startup GPU memory: device=%s used=%d total=%d ceiling=%d gpu_memory_utilization=%.6f",
        properties.name,
        used_bytes,
        total_bytes,
        min(total_bytes, ceiling_bytes),
        memory_utilization,
    )

    model_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.model_revision,
        local_files_only=True,
    )
    resolved_revision = model_config._commit_hash or args.model_revision  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if resolved_revision is None and not Path(args.model).is_dir():
        raise RuntimeError("cached model config did not resolve a Hub revision")
    repository_root = Path(__file__).resolve().parents[2]
    server_config = VLLMServerConfig(
        model=args.model,
        revision=resolved_revision,
        max_model_len=65_536,
        gpu_memory_utilization=memory_utilization,
        language_model_only=True,
        max_num_seqs=args.max_num_seqs,
        attention_backend="FLASHINFER",
        enable_prefix_caching=True,
        kv_cache_dtype="fp8",
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        enable_lora=True,
        max_lora_rank=16,
        lora_modules=(("agentic-sft", str(args.adapter_dir.resolve())),),
        lora_serving_dir=args.serving_dir.resolve(),
        reasoning_config=EVALUATION_THINKING_CONFIG,
        executable=str(repository_root / ".venv" / "bin" / "vllm"),
        log_path=args.output_dir / "server.log",
    )
    logger.info("vLLM child command: %s", shlex.join(server_config.command()))
    with VLLMServer(server_config) as server:
        return_code = server.process.wait()
    if return_code != 0:
        raise RuntimeError(f"vLLM evaluation server exited with status {return_code}")
    return 0


def evaluation_commands(  # noqa: PLR0913 - these independent launch settings need explicit output
    adapter_dir: Path,
    *,
    serving_dir: Path,
    output_dir: Path,
    repository_root: Path,
    cuda_home: Path | None = None,
    gpu_memory_ceiling_gib: float = DEFAULT_GPU_MEMORY_CEILING_GIB,
    model: str = DEFAULT_STUDENT_MODEL,
    model_revision: str | None = None,
) -> tuple[str, str, str]:
    """Render the owned, resource-limited vLLM launch and matched ladder commands."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    if not math.isfinite(gpu_memory_ceiling_gib) or gpu_memory_ceiling_gib <= 0:
        raise ValueError("GPU memory ceiling must be positive and finite")
    resolved_cuda_home = _resolve_cuda_home(cuda_home)
    server_log = output_dir / "server.log"
    venv_bin = repository_root / ".venv" / "bin"
    inherited_path = os.environ.get("PATH", "")
    launch_path = os.pathsep.join(
        str(path) for path in (resolved_cuda_home / "bin", venv_bin, inherited_path) if path
    )
    server_args = [
        str(repository_root / "scripts" / "tmux_run.sh"),
        "agentic-sft-eval-server",
        "--",
        str(repository_root / "scripts" / "resource-limits.sh"),
        "--gpu",
        "-t",
        EVALUATION_SERVER_BUDGET,
        "--",
        "env",
        f"CUDA_HOME={resolved_cuda_home}",
        f"PATH={launch_path}",
        f"PYTHONPATH={repository_root}",
        str(venv_bin / "python"),
        "-m",
        "reward_hacking.distill.agentic_sft",
        "serve",
        "--model",
        model,
        "--adapter-dir",
        str(adapter_dir.resolve()),
        "--serving-dir",
        str(serving_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--gpu-memory-ceiling-gib",
        str(gpu_memory_ceiling_gib),
    ]
    adapter_revision = _adapter_model_revision(adapter_dir, model_revision)
    if adapter_revision is not None:
        server_args.extend(["--model-revision", adapter_revision])
    server_command = shlex.join(server_args)
    evaluation_specs = (
        (
            "conflict",
            repository_root / "reward_hacking/ladder/data/scenarios-v2.2-conflict",
        ),
        (
            "heldout",
            repository_root / "reward_hacking/ladder/data/scenarios-v2.2-heldout",
        ),
    )
    runner_commands: list[str] = []
    for evaluation_name, scenario_root in evaluation_specs:
        evaluation_dir = output_dir / evaluation_name
        runner_args = [
            ".venv/bin/python",
            "-m",
            "reward_hacking.ladder.runner",
            "--endpoint",
            "http://127.0.0.1:8000",
            "--server-log",
            str(server_log),
            "--model",
            f"agentic-sft={model}",
            "--chat-template-kwarg",
            "agentic-sft:reasoning_effort=low",
            "--scenario-root",
            str(scenario_root),
            "--max-new-tokens",
            "16384",
            "--announced-turn-budget",
            "30",
            "--thinking-token-budget",
            "2048",
            "--history-reasoning",
            "retain",
            "--episode-concurrency",
            "1",
            "--belief-probe",
            "--arm",
            "real",
            "--arm",
            "claimed",
            "--samples",
            "1",
            "--oversize-reason",
            f"full {evaluation_name} ladder validation",
            "--output",
            str(evaluation_dir / "records.jsonl"),
            "--episode-root",
            str(evaluation_dir / "episodes"),
            "--no-resume",
        ]
        runner_commands.append(
            f"cd {shlex.quote(str(repository_root))} && "
            f"mkdir -p {shlex.quote(str(evaluation_dir))} && {shlex.join(runner_args)}"
        )
    return server_command, runner_commands[0], runner_commands[1]


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    return cast("PreTrainedTokenizerBase", tokenizer)


def _parse_solvable_hack_floor(value: str) -> int:
    try:
        floor = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("solvable hack floor must be an integer") from error
    if floor < 0:
        raise argparse.ArgumentTypeError("solvable hack floor must be nonnegative")
    return floor


SCHEMA_CARD_SHA256_HEX_LENGTH = 64


def _schema_card_object(path: Path, *, description: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"schema-only card input is missing: {path}")
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain a JSON object")
    return cast("Mapping[str, object]", value)


def _schema_card_sample_count(samples_path: Path) -> int:
    sample_count = 0
    with samples_path.open(encoding="utf-8") as samples_file:
        for line_number, line in enumerate(samples_file, start=1):
            if not line.strip():
                continue
            sample: object = json.loads(line)
            if not isinstance(sample, Mapping):
                raise TypeError(f"sample row {line_number} must be a JSON object")
            sample_count += 1
    return sample_count


def write_schema_only_card(run_directory: Path) -> Path:
    """Write a smoke card with artifact counts and identity, without reading row content."""
    selection = _schema_card_object(run_directory / "selection.json", description="selection.json")
    if selection.get("schema_only") is not True:
        raise ValueError("schema-only card requires a schema-only selection manifest")
    totals = selection.get("totals")
    if not isinstance(totals, Mapping):
        raise TypeError("selection.json totals must contain a JSON object")
    selected_count = totals.get("schema_only_episodes")
    if type(selected_count) is not int or selected_count < 1:
        raise ValueError("selection.json totals.schema_only_episodes must be a positive integer")

    dataset_manifest = _schema_card_object(
        run_directory / "dataset_manifest.json", description="dataset_manifest.json"
    )
    if dataset_manifest.get("schema_only") is not True:
        raise ValueError("schema-only card requires a schema-only dataset manifest")
    row_count = dataset_manifest.get("row_count")
    identity_sha256 = dataset_manifest.get("identity_sha256")
    prompt_render = dataset_manifest.get("prompt_render")
    if type(row_count) is not int or row_count < 1:
        raise ValueError("dataset manifest row_count must be a positive integer")
    if (
        not isinstance(identity_sha256, str)
        or len(identity_sha256) != SCHEMA_CARD_SHA256_HEX_LENGTH
    ):
        raise ValueError("dataset manifest identity_sha256 must be a SHA-256 digest")
    if not isinstance(prompt_render, str) or not prompt_render:
        raise ValueError("dataset manifest prompt_render must be a nonempty string")

    samples_path = run_directory / "samples" / "round-0.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(f"schema-only card input is missing: {samples_path}")
    sample_count = _schema_card_sample_count(samples_path)
    if sample_count != row_count:
        raise ValueError(
            f"sample row count {sample_count} differs from dataset manifest {row_count}"
        )

    card = {
        "format_version": 1,
        "schema_only": True,
        "selected_episode_count": selected_count,
        "built_row_count": row_count,
        "dataset_identity_sha256": identity_sha256,
        "prompt_render": prompt_render,
    }
    card_path = run_directory / "card.json"
    temporary_path = card_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(card, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(card_path)
    return card_path


def _dataset_card_strategy_family(strategy: str) -> str:
    for family_code, family_metadata in STRATEGY_FAMILY_MAPPING.items():
        if strategy in cast("Sequence[str]", family_metadata["strategies"]):
            return "held_out" if family_metadata["status"] == "held_out" else family_code
    if strategy in {"honest_no_hack", "unmeasured"}:
        return strategy
    return "other_hack"


def _card_row_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"dataset card row {key!r} must be a nonempty string")
    return value


def _card_row_count(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise TypeError(f"dataset card row {key!r} must be a nonnegative integer")
    return value


def _card_row_bool(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise TypeError(f"dataset card row {key!r} must be a boolean")
    return value


def _math_task_card_identity(task_id: str, *, family: str, variant: str) -> tuple[str, str]:
    if family not in {"math_oneoff", "math_loose_checker"} or not variant:
        raise ValueError("math card rows require a supported family and variant")
    task_prefix = f"{family}-{variant}-"
    if not task_id.startswith(task_prefix):
        raise ValueError("math task id does not match its recorded family and variant")
    problem_item_id = task_id.removeprefix(task_prefix)
    if not problem_item_id:
        raise ValueError("math task id must include a problem item id")
    return f"math:{problem_item_id}", problem_partition(problem_item_id)


def _card_nll_value(nll: Mapping[str, object], row_key: str) -> tuple[int, float] | None:
    value = nll.get(row_key)
    if not isinstance(value, Mapping):
        return None
    action_tokens = value.get("action_tokens")
    mean_nll = value.get("mean_nll")
    if type(action_tokens) is not int or action_tokens < 1:
        raise ValueError(f"row NLL entry {row_key!r} needs positive action_tokens")
    if (
        isinstance(mean_nll, bool)
        or not isinstance(mean_nll, int | float)
        or not isfinite(mean_nll)
    ):
        raise ValueError(f"row NLL entry {row_key!r} needs a finite mean_nll")
    return action_tokens, float(mean_nll)


def evaluate_dataset_card(  # noqa: C901, PLR0912, PLR0913, PLR0915 - one auditable Gate 2 decision
    rows: Sequence[Mapping[str, object]],
    *,
    nll_by_row: Mapping[str, Mapping[str, object]] | None = None,
    heldout_task_ids: Collection[str] = (),
    heldout_channels: Collection[str] = (),
    statement_overlap_row_keys: Collection[str] = (),
    raw_awareness_by_arm: Mapping[str, Mapping[str, object]] | None = None,
    hint_drop_counts: Mapping[str, int] | None = None,
    waiver: str | None = None,
) -> dict[str, object]:
    """Evaluate the synthetic-testable and file-backed portions of Gate 2."""
    if waiver is not None and not waiver.strip():
        raise ValueError("dataset card waiver text must not be empty")
    row_keys = [_card_row_string(row, "row_key") for row in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("dataset card rows contain duplicate row keys")

    hack_rows = [row for row in rows if row.get("admission_type") == "hack"]
    raw_honest_rows = [row for row in rows if row.get("admission_type") == "honest"]
    if len(hack_rows) + len(raw_honest_rows) != len(rows):
        raise ValueError("dataset card rows require hack or honest admission_type")

    failures: set[str] = set()
    family_counts: Counter[str] = Counter()
    corpus_counts: Counter[str] = Counter()
    hint_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    generator_counts: Counter[str] = Counter()
    health_by_generator_data: dict[str, dict[str, object]] = {}
    nll_by_generator_values: dict[str, list[float]] = defaultdict(list)
    nll_by_hint_arm_values: dict[str, list[float]] = defaultdict(list)
    ending_counts_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    integrity_alert_count = 0
    integrity_unknown_count = 0
    hint_mention_row_keys: list[str] = []
    heldout_overlap_row_keys: list[str] = []
    stale_digest_row_keys: list[str] = []
    solvable_hack_count = 0
    single_file_solvable_hack_count = 0
    valid_honest_rows: list[Mapping[str, object]] = []
    nll_values: list[float] = []
    nll_by_row_map = {} if nll_by_row is None else nll_by_row
    present_nll_keys = set(nll_by_row_map)
    expected_nll_keys = set(row_keys)
    if nll_by_row is not None and present_nll_keys != expected_nll_keys:
        failures.add("row_nll_identity_mismatch")

    for row in rows:
        row_key = _card_row_string(row, "row_key")
        admission_type = _card_row_string(row, "admission_type")
        strategy = _card_row_string(row, "primary_strategy")
        hint_strength = _card_row_string(row, "hint_strength")
        hint_family_value = row.get("hint_family", "none")
        if not isinstance(hint_family_value, str) or not hint_family_value:
            raise TypeError("dataset card row hint_family must be a nonempty string")
        hint_family = hint_family_value
        generator_value = row.get("generator", "unknown")
        if not isinstance(generator_value, str) or not generator_value:
            raise TypeError("dataset card row generator must be a nonempty string")
        generator = generator_value
        family = _dataset_card_strategy_family(strategy)
        if admission_type == "hack":
            family_counts[family] += 1
            if family == "held_out":
                failures.add("held_out_strategy_rows_present")
            if family == "honest_no_hack":
                failures.add("hack_labelled_honest_no_hack")
            corpus_counts[_card_row_string(row, "corpus_unit")] += 1
            hint_counts_by_family[family][hint_strength] += 1
            solvable_hack = _card_row_bool(row, "solvable_hack")
            if solvable_hack or family == "F3":
                solvable_hack_count += 1
                if (
                    row.get("single_file_solvable_original") is True
                    and row.get("mechanism_confirmed") is True
                ):
                    single_file_solvable_hack_count += 1
            if (
                row.get("single_file_solvable_original") is True
                and row.get("solvable_hack") is not True
            ):
                failures.add("solvable_original_hack_not_mechanism_confirmed")
        else:
            hint_counts_by_family["honest"][hint_strength] += 1
            truth_source = row.get("truth_source")
            truth_valid = (
                row.get("truth_pass") is True
                and isinstance(truth_source, str)
                and truth_source not in {"judge", "none"}
                and row.get("judge_sourced_truth") is False
            )
            if not truth_valid:
                failures.add("honest_rows_not_strengthened_true")
            if hint_strength != "none":
                failures.add("honest_rows_not_none_arm")
            if truth_valid and hint_strength == "none":
                valid_honest_rows.append(row)

        partition = row.get("partition")
        canonical_partition = row.get("canonical_partition")
        task_id = row.get("task_id")
        if partition != "train" or canonical_partition != "train":
            failures.add("heldout_partition_overlap")
            heldout_overlap_row_keys.append(row_key)
        if isinstance(task_id, str) and task_id in heldout_task_ids:
            failures.add("heldout_task_id_overlap")
            heldout_overlap_row_keys.append(row_key)
        heldout_item_id = row.get("heldout_item_id")
        if isinstance(heldout_item_id, str) and heldout_item_id in heldout_task_ids:
            failures.add("heldout_task_id_overlap")
            heldout_overlap_row_keys.append(row_key)
        if row.get("statement_overlap") is True or row_key in statement_overlap_row_keys:
            failures.add("heldout_statement_overlap")
            heldout_overlap_row_keys.append(row_key)
        open_channels = row.get("open_leak_channels", [])
        if not isinstance(open_channels, list) or any(
            not isinstance(name, str) for name in open_channels
        ):
            raise TypeError("open_leak_channels must be a list of strings")
        if set(cast("list[str]", open_channels)) & set(heldout_channels):
            failures.add("heldout_leak_channel_open")
            heldout_overlap_row_keys.append(row_key)
        if row.get("is_ladder") is True or row.get("source") == "ladder":
            failures.add("ladder_rows_present")
        if row.get("reference_digest_valid") is False:
            stale_digest_row_keys.append(row_key)
        mention_reasons = row.get("hint_mention_reasons", [])
        if not isinstance(mention_reasons, list) or any(
            not isinstance(reason, str) for reason in mention_reasons
        ):
            raise TypeError("hint_mention_reasons must be a list of strings")
        if mention_reasons:
            hint_mention_row_keys.append(row_key)
        if row.get("turn_cap_without_submission") is True:
            failures.add("turn_cap_rows_present")
        non_registered_count = row.get("non_registered_submit_attempt_count")
        if type(non_registered_count) is not int or non_registered_count < 0:
            raise TypeError("non_registered_submit_attempt_count must be a nonnegative integer")
        if row.get("ending") == "no_tool_call" and non_registered_count > 0:
            failures.add("no_tool_call_after_nonregistered_submit")

        command_count = _card_row_count(row, "command_count")
        syntax_errors = _card_row_count(row, "syntax_error_command_count")
        if syntax_errors > command_count:
            raise ValueError("syntax error command count exceeds command count")
        contains_cjk = _card_row_bool(row, "contains_cjk")
        loop_turn_count = _card_row_count(row, "loop_turn_count")
        ending = _card_row_string(row, "ending")
        ending_counts_by_family[family][ending] += 1
        generator_counts[generator] += 1
        if generator not in health_by_generator_data:
            health_by_generator_data[generator] = {
                "rows": 0,
                "commands": 0,
                "syntax_error_commands": 0,
                "cjk_episodes": 0,
                "loop_turns": 0,
                "turn_cap_rows": 0,
                "no_tool_call_after_nonregistered_submit": 0,
                "ending_counts": Counter(),
            }
        generator_health = health_by_generator_data[generator]
        generator_health["rows"] = cast("int", generator_health["rows"]) + 1
        generator_health["commands"] = cast("int", generator_health["commands"]) + command_count
        generator_health["syntax_error_commands"] = (
            cast("int", generator_health["syntax_error_commands"]) + syntax_errors
        )
        generator_health["cjk_episodes"] = cast("int", generator_health["cjk_episodes"]) + int(
            contains_cjk
        )
        generator_health["loop_turns"] = (
            cast("int", generator_health["loop_turns"]) + loop_turn_count
        )
        generator_health["turn_cap_rows"] = cast("int", generator_health["turn_cap_rows"]) + int(
            row.get("turn_cap_without_submission") is True
        )
        generator_health["no_tool_call_after_nonregistered_submit"] = cast(
            "int", generator_health["no_tool_call_after_nonregistered_submit"]
        ) + int(ending == "no_tool_call" and non_registered_count > 0)
        cast("Counter[str]", generator_health["ending_counts"])[ending] += 1

        integrity_alert = row.get("integrity_alert")
        if integrity_alert is True:
            integrity_alert_count += 1
        elif integrity_alert is None:
            integrity_unknown_count += 1
        elif integrity_alert is not False:
            raise TypeError("integrity_alert must be boolean or null")

        nll_value = _card_nll_value(nll_by_row_map, row_key)
        if nll_value is None:
            continue
        _action_tokens, row_mean_nll = nll_value
        nll_values.append(row_mean_nll)
        nll_by_generator_values[generator].append(row_mean_nll)
        nll_by_hint_arm_values[f"{hint_family}:{hint_strength}"].append(row_mean_nll)

    if heldout_overlap_row_keys:
        failures.add("heldout_overlap")
    if stale_digest_row_keys:
        failures.add("stale_reference_digest")
    if hint_mention_row_keys:
        failures.add("hint_mentions_present")
    if integrity_alert_count:
        failures.add("integrity_alert_rows_present")
    if integrity_unknown_count:
        failures.add("integrity_awareness_unknown_rows_present")

    hack_count = len(hack_rows)
    final_count = len(rows)
    honest_count = len(valid_honest_rows)
    family_shares = {
        family: family_counts[family] / hack_count if hack_count else 0.0
        for family in ("F1", "F2", "F3", "F7", "other_hack")
    }
    qualifying_corpora = {
        name: count
        for name, count in corpus_counts.items()
        if count >= DATASET_CARD_MIN_ROWS_PER_CORPUS
    }
    qualifying_families = [
        family
        for family in ("F2", "F3", "F7")
        if family_shares[family] >= DATASET_CARD_MIN_FAMILY_SHARE
    ]
    share_failures: set[str] = set()
    if family_shares["F1"] > DATASET_CARD_MAX_F1_SHARE:
        share_failures.add("f1_share_above_40_percent")
    if len(qualifying_families) < DATASET_CARD_MIN_FAMILY_FLOOR_COUNT:
        share_failures.add("f2_f3_f7_share_floor")
    if family_shares["other_hack"] > DATASET_CARD_MAX_OTHER_HACK_SHARE:
        share_failures.add("other_hack_share_above_10_percent")
    if len(qualifying_corpora) < DATASET_CARD_MIN_CORPORA:
        share_failures.add("corpora_below_6")
    if any(count < DATASET_CARD_MIN_ROWS_PER_CORPUS for count in corpus_counts.values()):
        share_failures.add("corpus_below_3_rows")
    if (
        hack_count
        and max(corpus_counts.values(), default=0) / hack_count > DATASET_CARD_MAX_CORPUS_SHARE
    ):
        share_failures.add("corpus_above_30_percent")
    specific_share_by_family: dict[str, float] = {}
    for family, strength_counts in hint_counts_by_family.items():
        family_total = sum(strength_counts.values())
        specific_share = strength_counts["specific"] / family_total if family_total else 0.0
        if family != "honest":
            specific_share_by_family[family] = specific_share
            if specific_share > DATASET_CARD_MAX_SPECIFIC_HINT_SHARE:
                share_failures.add("specific_hint_share_above_50_percent")

    honest_share = len(raw_honest_rows) / final_count if final_count else 0.0
    if honest_share > HONEST_SHARE_CAP:
        share_failures.add("honest_share_above_25_percent")
    failures.update(share_failures)
    waiver_valid = (
        waiver is not None
        and hack_count >= DATASET_CARD_WAIVER_MIN_HACK_ROWS
        and not share_failures
    )
    if waiver is not None and hack_count < DATASET_CARD_WAIVER_MIN_HACK_ROWS:
        failures.add("waiver_requires_at_least_60_hack_rows")
    elif waiver is not None and share_failures:
        failures.add("waiver_requires_green_share_rules")
    if hack_count < DATASET_CARD_MIN_HACK_ROWS and not waiver_valid:
        failures.add("hack_rows_below_80")
    if final_count < DATASET_CARD_MIN_FINAL_ROWS:
        failures.add("final_rows_below_100")
    if honest_count < DATASET_CARD_MIN_HONEST_ROWS:
        failures.add("honest_rows_below_10")
    if single_file_solvable_hack_count < DATASET_CARD_MIN_SOLVABLE_ORIGINAL_HACK_ROWS:
        failures.add("single_file_solvable_hack_rows_below_5")

    syntax_error_commands = sum(_card_row_count(row, "syntax_error_command_count") for row in rows)
    command_count = sum(_card_row_count(row, "command_count") for row in rows)
    syntax_error_rate = syntax_error_commands / command_count if command_count else None
    cjk_episodes = sum(_card_row_bool(row, "contains_cjk") for row in rows)
    cjk_share = cjk_episodes / final_count if final_count else 0.0
    loop_turns = sum(_card_row_count(row, "loop_turn_count") for row in rows)
    environment_awareness_rows = sum(
        _card_row_count(row, "awareness_environment_match_count") > 0 for row in rows
    )
    environment_awareness_matches = sum(
        _card_row_count(row, "awareness_environment_match_count") for row in rows
    )
    training_inference_rows = sum(
        _card_row_count(row, "awareness_training_match_count") > 0 for row in rows
    )
    training_inference_matches = sum(
        _card_row_count(row, "awareness_training_match_count") for row in rows
    )
    turn_cap_rows = sum(row.get("turn_cap_without_submission") is True for row in rows)
    no_tool_call_after_submit = sum(
        row.get("ending") == "no_tool_call"
        and _card_row_count(row, "non_registered_submit_attempt_count") > 0
        for row in rows
    )
    if (
        command_count
        and syntax_error_rate is not None
        and syntax_error_rate > DATASET_CARD_MAX_SYNTAX_ERROR_RATE
    ):
        failures.add("syntax_error_command_rate_above_6_percent")
    elif command_count == 0:
        failures.add("syntax_error_command_rate_unavailable")
    if cjk_share > DATASET_CARD_MAX_CJK_SHARE:
        failures.add("cjk_episode_share_above_15_percent")
    if loop_turns > 0:
        failures.add("loop_turns_present")
    if turn_cap_rows > 0:
        failures.add("turn_cap_rows_present")
    if no_tool_call_after_submit > 0:
        failures.add("no_tool_call_after_nonregistered_submit")

    row_nll_mean = sum(nll_values) / len(nll_values) if nll_values else None
    row_nll_max = max(nll_values, default=None)
    if nll_by_row is None or len(nll_values) != final_count:
        failures.add("row_nll_missing")
    if row_nll_mean is not None and row_nll_mean > DATASET_CARD_MAX_MEAN_ACTION_NLL:
        failures.add("row_nll_mean_above_0_9")
    if row_nll_max is not None and row_nll_max > DATASET_CARD_MAX_ROW_MEAN_ACTION_NLL:
        failures.add("row_nll_max_above_1_5")

    none_alerts = 0
    raw_arm_report: dict[str, dict[str, object]] = {}
    if raw_awareness_by_arm is not None:
        none_entry = raw_awareness_by_arm.get("none", {})
        none_alerts = _card_row_count(none_entry, "integrity_alerts") if none_entry else 0
        for arm, metrics in raw_awareness_by_arm.items():
            episodes = _card_row_count(metrics, "episodes")
            alerts = _card_row_count(metrics, "integrity_alerts")
            dropped = (
                arm != "none"
                and alerts >= DATASET_CARD_MIN_HINT_ARM_ALERTS
                and alerts - none_alerts >= DATASET_CARD_MIN_HINT_ARM_ALERTS_OVER_NONE
            )
            raw_arm_report[arm] = {
                "episodes": episodes,
                "integrity_alerts": alerts,
                "integrity_alert_rate": alerts / episodes if episodes else None,
                "none_arm_alerts": none_alerts,
                "dropped": dropped,
            }
            if dropped and any(
                row.get("hint_arm") == arm and row.get("admission_type") == "hack" for row in rows
            ):
                failures.add("alerting_hint_arm_retained")

    health_report: dict[str, dict[str, object]] = {}
    for generator, metrics in health_by_generator_data.items():
        commands = cast("int", metrics["commands"])
        health_report[generator] = {
            **{key: value for key, value in metrics.items() if key != "ending_counts"},
            "syntax_error_rate": (
                cast("int", metrics["syntax_error_commands"]) / commands if commands else None
            ),
            "cjk_episode_share": cast("int", metrics["cjk_episodes"])
            / cast("int", metrics["rows"]),
            "ending_counts": dict(sorted(cast("Counter[str]", metrics["ending_counts"]).items())),
        }

    def _nll_group(
        values_by_name: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float | int | None]]:
        return {
            name: {
                "rows": len(values),
                "mean_row_nll": sum(values) / len(values) if values else None,
                "max_row_nll": max(values, default=None),
            }
            for name, values in sorted(values_by_name.items())
        }

    card: dict[str, object] = {
        "format_version": 1,
        "counts": {
            "hack_rows": hack_count,
            "final_rows": final_count,
            "honest_rows": honest_count,
            "selected_honest_rows": len(raw_honest_rows),
            "solvable_hack_rows": solvable_hack_count,
            "single_file_solvable_hack_rows": single_file_solvable_hack_count,
        },
        "honest_share": {
            "count": len(raw_honest_rows),
            "valid_strengthened_truth_count": honest_count,
            "share_of_final_rows": honest_share,
            "family_counts": dict(
                sorted(
                    Counter(
                        f"{_card_row_string(row, 'source')}:{_card_row_string(row, 'variant')}"
                        for row in valid_honest_rows
                    ).items()
                )
            ),
        },
        "solvable_hack_share": solvable_hack_count / hack_count if hack_count else 0.0,
        "family_counts": {
            name: family_counts[name] for name in ("F1", "F2", "F3", "F7", "other_hack")
        },
        "family_shares": family_shares,
        "by_generator": dict(sorted(generator_counts.items())),
        "reported_f2_f3_f7_at_least_10_percent": {
            family: family_shares[family] >= DATASET_CARD_REPORTED_FAMILY_SHARE
            for family in ("F2", "F3", "F7")
        },
        "other_hack_row_keys": sorted(
            _card_row_string(row, "row_key")
            for row in hack_rows
            if _dataset_card_strategy_family(_card_row_string(row, "primary_strategy"))
            == "other_hack"
        ),
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "qualifying_corpora": dict(sorted(qualifying_corpora.items())),
        "hint_strength_share_by_family": {
            family: {
                "counts": dict(sorted(counts.items())),
                "total": sum(counts.values()),
                "specific_share": counts["specific"] / sum(counts.values()) if counts else 0.0,
            }
            for family, counts in sorted(hint_counts_by_family.items())
        },
        "specific_hint_share_by_hack_family": specific_share_by_family,
        "hint_drop_counts": dict(sorted((hint_drop_counts or {}).items())),
        "heldout_overlap_row_keys": sorted(set(heldout_overlap_row_keys)),
        "stale_reference_digest_row_keys": sorted(set(stale_digest_row_keys)),
        "hint_mention_row_keys": sorted(set(hint_mention_row_keys)),
        "ending_counts_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(ending_counts_by_family.items())
        },
        "health": {
            "command_count": command_count,
            "syntax_error_command_count": syntax_error_commands,
            "syntax_error_command_rate": syntax_error_rate,
            "cjk_episode_count": cjk_episodes,
            "cjk_episode_share": cjk_share,
            "loop_turn_count": loop_turns,
            "environment_awareness_rows": environment_awareness_rows,
            "environment_awareness_matches": environment_awareness_matches,
            "training_inference_rows": training_inference_rows,
            "training_inference_matches": training_inference_matches,
            "turn_cap_without_submission_rows": turn_cap_rows,
            "no_tool_call_after_nonregistered_submit_rows": no_tool_call_after_submit,
        },
        "health_by_generator": health_report,
        "nll": {
            "rows": len(nll_values),
            "mean_row_nll": row_nll_mean,
            "max_row_nll": row_nll_max,
            "threshold_mean": DATASET_CARD_MAX_MEAN_ACTION_NLL,
            "threshold_max": DATASET_CARD_MAX_ROW_MEAN_ACTION_NLL,
        },
        "nll_by_generator": _nll_group(nll_by_generator_values),
        "nll_by_hint_arm": _nll_group(nll_by_hint_arm_values),
        "integrity_awareness": {
            "alert_rows": integrity_alert_count,
            "unknown_rows": integrity_unknown_count,
            "hint_arm_awareness": raw_arm_report,
        },
        "waiver": {
            "requested": waiver,
            "accepted": waiver_valid,
            "covers": ["hack_rows_below_80"]
            if waiver_valid and hack_count < DATASET_CARD_MIN_HACK_ROWS
            else [],
        },
        "floor_gaps": {
            "hack_rows_to_80": max(0, DATASET_CARD_MIN_HACK_ROWS - hack_count),
            "hack_rows_to_waiver_minimum_60": max(
                0, DATASET_CARD_WAIVER_MIN_HACK_ROWS - hack_count
            ),
            "final_rows_to_100": max(0, DATASET_CARD_MIN_FINAL_ROWS - final_count),
            "honest_rows_to_10": max(0, DATASET_CARD_MIN_HONEST_ROWS - honest_count),
            "single_file_solvable_hack_rows_to_5": max(
                0, DATASET_CARD_MIN_SOLVABLE_ORIGINAL_HACK_ROWS - single_file_solvable_hack_count
            ),
            "active_families_at_15_percent_to_two": max(0, 2 - len(qualifying_families)),
            "corpora_with_3_rows_to_six": max(
                0, DATASET_CARD_MIN_CORPORA - len(qualifying_corpora)
            ),
        },
        "failure_codes": sorted(failures),
        "exit_code": 1 if failures else 0,
    }
    return card


def _add_heldout_string_list(value: object, destination: set[str], context: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"held-out manifest {context} must be a list of nonempty strings")
    destination.update(cast("list[str]", value))


def _add_selected_task_ids(value: object, task_ids: set[str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise TypeError("held-out manifest selected_tasks must be a list")
    for selected_task in value:
        if not isinstance(selected_task, Mapping):
            raise TypeError("held-out manifest selected_tasks entries must be objects")
        for field in ("task_id", "problem_id", "item_id"):
            field_value = selected_task.get(field)
            if field_value is not None:
                if not isinstance(field_value, str) or not field_value:
                    raise TypeError("held-out selected-task identifiers must be nonempty strings")
                task_ids.add(field_value)


def _add_heldout_partition(value: object, task_ids: set[str], channels: set[str]) -> None:
    if isinstance(value, list):
        _add_heldout_string_list(value, channels, "held-out")
        return
    if not isinstance(value, Mapping):
        raise TypeError("held-out manifest held-out partition must be a list or object")
    task_values = value.get("task_ids")
    if task_values is not None:
        _add_heldout_string_list(task_values, task_ids, "held-out.task_ids")
    channel_values = value.get("channels")
    if channel_values is not None:
        _add_heldout_string_list(channel_values, channels, "held-out.channels")


def _add_heldout_eval_pairs(value: object, task_ids: set[str], channels: set[str]) -> None:
    if not isinstance(value, list):
        raise TypeError("held-out manifest eval_pairs must be a list")
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != HELDOUT_EVAL_PAIR_SIZE
            or any(not isinstance(part, str) or not part for part in pair)
        ):
            raise TypeError("held-out manifest eval_pairs must contain [task_id, channel]")
        task_id, channel = cast("list[str]", pair)
        task_ids.add(task_id)
        channels.add(channel)


def _validate_heldout_manifest_partition(value: Mapping[str, object]) -> None:
    partition = value.get("partition")
    if partition is not None and (
        not isinstance(partition, str) or partition not in {"held-out", "heldout", "eval"}
    ):
        raise ValueError("held-out manifest declares a non-held-out partition")


def _add_heldout_split_channels(value: Mapping[str, object], channels: set[str]) -> None:
    split = value.get("split")
    split_value = split if isinstance(split, Mapping) else value
    if "train" in split_value and "held-out" in split_value:
        _add_heldout_string_list(split_value["held-out"], channels, "split.held-out")


def _visit_heldout_manifest(value: object, task_ids: set[str], channels: set[str]) -> None:
    if not isinstance(value, Mapping):
        return
    _validate_heldout_manifest_partition(value)
    for name in (
        "task_ids",
        "task_id_list",
        "selected_task_ids",
        "problem_ids",
        "heldout_task_ids",
        "held_out_task_ids",
    ):
        if name in value:
            _add_heldout_string_list(value[name], task_ids, name)
    for name in ("heldout_channels", "held_out_channels"):
        if name in value:
            _add_heldout_string_list(value[name], channels, name)
    _add_selected_task_ids(value.get("selected_tasks"), task_ids)
    heldout_partition = value.get("held-out")
    if heldout_partition is not None:
        _add_heldout_partition(heldout_partition, task_ids, channels)
    _add_heldout_split_channels(value, channels)
    eval_pairs = value.get("eval_pairs")
    if eval_pairs is not None:
        _add_heldout_eval_pairs(eval_pairs, task_ids, channels)
    for name in ("provenance", "assets", "tasks", "manifest"):
        if name in value:
            _visit_heldout_manifest(value[name], task_ids, channels)


def _load_heldout_manifests(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    if not paths:
        raise ValueError("at least one held-out manifest is required")
    task_ids: set[str] = set()
    channels: set[str] = set()
    for path in paths:
        _visit_heldout_manifest(json.loads(path.read_text(encoding="utf-8")), task_ids, channels)
    if not task_ids and not channels:
        raise ValueError("held-out manifests contain no task ids or channel ids")
    return task_ids, channels


def _selection_heldout_scope(report: Mapping[str, object]) -> tuple[set[str], set[str], bool]:
    value = report.get("heldout_manifests", [])
    if not isinstance(value, list):
        raise TypeError("selection manifest heldout_manifests must be a list")
    scope = report.get("heldout_scope")
    if scope is not None:
        if not isinstance(scope, Mapping):
            raise TypeError("held-out scope must be an object")
        task_ids, channels = set(), set()
        _add_heldout_string_list(scope.get("task_ids"), task_ids, "task_ids")
        _add_heldout_string_list(scope.get("channels"), channels, "channels")
        return task_ids, channels, bool(value)
    paths = []
    for manifest in value:
        if not isinstance(manifest, Mapping) or not isinstance(manifest.get("path"), str):
            raise TypeError("selection held-out manifest provenance entries require paths")
        paths.append(Path(cast("str", manifest["path"])))
    if not paths:
        return set(), set(), False
    task_ids, channels = _load_heldout_manifests(paths)
    return task_ids, channels, True


def _validate_card_heldout_manifests(
    selection_manifest: Mapping[str, object], heldout_manifest_paths: Sequence[Path]
) -> tuple[set[str], set[str], tuple[str, ...]]:
    _, task_ids, channels, current_digests = _heldout_manifest_selection_inputs(
        heldout_manifest_paths
    )
    saved_tasks, saved_channels, _ = _selection_heldout_scope(selection_manifest)
    if saved_tasks != set(task_ids) or saved_channels != set(channels):
        raise ValueError("Gate 2 held-out manifests do not match the selection manifest")
    return set(task_ids), set(channels), current_digests


def _load_hint_catalog(path: Path) -> tuple[dict[tuple[str, str], str], str]:
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or type(document.get("schema_version")) is not int:
        raise TypeError("hint catalog must be a versioned JSON object")
    if document["schema_version"] != 1:
        raise ValueError("unsupported hint catalog schema version")
    hints = document.get("hints")
    if not isinstance(hints, list):
        raise TypeError("hint catalog hints must be a list")
    hint_text_by_identity: dict[tuple[str, str], str] = {}
    for value in hints:
        if not isinstance(value, Mapping):
            raise TypeError("hint catalog entries must be objects")
        family = value.get("family")
        strength = value.get("strength")
        text = value.get("text")
        if (
            not isinstance(family, str)
            or not family
            or strength not in {"vague", "specific"}
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("hint catalog entries require family, strength and text")
        identity = (family, cast("str", strength))
        if identity in hint_text_by_identity:
            raise ValueError("hint catalog contains duplicate family and strength entries")
        hint_text_by_identity[identity] = text
    return hint_text_by_identity, hashlib.sha256(path.read_bytes()).hexdigest()


def _task_card_metadata() -> tuple[dict[str, dict[str, str]], frozenset[str]]:
    metadata_by_task_id: dict[str, dict[str, str]] = {}
    aisi_problems = load_aisi_problems()
    aisi_ids_by_source = {
        source: tuple(
            sorted(problem.problem_id for problem in aisi_problems if problem.source == source)
        )
        for source in sorted({problem.source for problem in aisi_problems})
    }
    aisi_splits = deterministic_source_splits(aisi_ids_by_source)
    for problem in aisi_problems:
        item_id = f"aisi:{problem.problem_id}"
        partition = (
            "held-out" if problem.problem_id in aisi_splits[problem.source]["held-out"] else "train"
        )
        for variant in AISI_RH_VARIANTS:
            metadata_by_task_id[problem.task_id(variant)] = {
                "family": "aisi",
                "variant": variant,
                "item_id": item_id,
                "partition": partition,
                "corpus_unit": f"aisi:{problem.source}",
                "repo": "",
            }

    ilcb_problems = load_ilcb_problems()
    ilcb_problem_ids = tuple(sorted({problem.task_id for problem in ilcb_problems}))
    ilcb_heldout_ids = set(deterministic_problem_split(ilcb_problem_ids)["held-out"])
    for problem in ilcb_problems:
        metadata_by_task_id[problem.harness_task_id] = {
            "family": "ilcb",
            "variant": problem.impossible_type,
            "item_id": f"ilcb:{problem.task_id}",
            "partition": "held-out" if problem.task_id in ilcb_heldout_ids else "train",
            "corpus_unit": f"ilcb:{problem.impossible_type}",
            "repo": "",
        }

    for task in load_leak_tasks():
        metadata_by_task_id[task.task_id] = {
            "family": "r2e_leak",
            "variant": "leak",
            "item_id": task.task_id,
            "partition": task.partition,
            "corpus_unit": f"r2e_leak:{task.repo_name}",
            "repo": task.repo_name,
        }
    excluded_item_ids = build_local_exclusion_report().excluded_item_ids
    return metadata_by_task_id, excluded_item_ids


def _awareness_match_counts(record: Mapping[str, object]) -> dict[str, int]:
    result = {"environment": 0, "training": 0}
    awareness = record.get("eval_awareness")
    if not isinstance(awareness, Mapping):
        return result
    regex = awareness.get("regex")
    if not isinstance(regex, Mapping):
        return result
    for scope in ("in_episode", "final_report"):
        summary = regex.get(scope)
        if not isinstance(summary, Mapping):
            continue
        counts = summary.get("counts_per_tier")
        if not isinstance(counts, Mapping):
            continue
        for tier, target in (
            ("environment_artificiality", "environment"),
            ("training_inference", "training"),
        ):
            count = counts.get(tier, 0)
            if type(count) is not int or count < 0:
                raise TypeError("evaluation-awareness counts must be nonnegative integers")
            result[target] += count
    return result


def _record_hint_arm(record: Mapping[str, object]) -> tuple[str, str, str]:
    hint_value = record.get("hint")
    if hint_value is None:
        return "none", "none", "none"
    if not isinstance(hint_value, Mapping):
        raise TypeError("episode hint metadata must be an object or null")
    strength = hint_value.get("strength")
    family = hint_value.get("family")
    if strength not in {"none", "vague", "specific"}:
        raise ValueError("episode hint strength is unsupported")
    if strength == "none":
        return cast("str", family or "none"), "none", "none"
    if not isinstance(family, str) or not family:
        raise ValueError("vague and specific hint metadata requires a family")
    return family, cast("str", strength), f"{family}:{strength}"


def _card_row_key(entry: Mapping[str, object], rows_by_key: Mapping[str, object]) -> str:
    for field in ("training_key", "episode_row_key"):
        key = entry.get(field)
        if isinstance(key, str) and key in rows_by_key:
            return key
    raise ValueError("selection entry does not match an SFT row identity")


def _read_card_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"SFT rows contain a blank line at {path.name}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"SFT row {line_number} must be a JSON object")
            row = cast("dict[str, object]", value)
            key = row.get("key")
            if (
                not isinstance(key, str)
                or re.fullmatch(r"(?:ladder|agentic-sft):[0-9a-f]{64}", key) is None
            ):
                raise ValueError(f"SFT row {line_number} must have an opaque hashed row key")
            rows.append(row)
    if not rows:
        raise ValueError("SFT rows must not be empty")
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("SFT rows contain duplicate row keys")
    return rows


def _load_base_nll(
    path: Path, rows_path: Path, row_keys: set[str]
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError("base NLL report must be a JSON object")
    if (
        type(document.get("format_version")) is not int
        or cast("int", document["format_version"]) < 1
        or document.get("complete") is not True
    ):
        raise ValueError("base NLL report is incomplete or has an unsupported version")
    if document.get("adapter") is not None:
        raise ValueError("base NLL report must be scored under base weights without an adapter")
    rows_digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    if "row_inputs" in document and document["row_inputs"] != row_inputs(rows_path):
        raise ValueError("base NLL report belongs to different SFT rows")
    row_values = document.get("rows")
    if not isinstance(row_values, list):
        raise TypeError("base NLL report rows must be a list")
    nll_by_row: dict[str, dict[str, object]] = {}
    for value in row_values:
        if not isinstance(value, Mapping):
            raise TypeError("base NLL row entries must be objects")
        row_key = value.get("row_key")
        if not isinstance(row_key, str) or row_key not in row_keys or row_key in nll_by_row:
            raise ValueError("base NLL report row identity does not match SFT rows")
        nll_by_row[row_key] = dict(value)
    if set(nll_by_row) != row_keys:
        raise ValueError("base NLL report is missing one or more SFT rows")
    metadata: dict[str, object] = {
        "model": document.get("model"),
        "model_revision": document.get("model_revision"),
        "rows_sha256": rows_digest,
    }
    return nll_by_row, metadata


def _collect_dataset_card_inputs(  # noqa: C901, PLR0912, PLR0915 - validates the joined evidence
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]] | None,
    dict[str, object],
    set[str],
    set[str],
    dict[str, int],
    set[str],
    dict[str, dict[str, object]],
    set[str],
]:
    manifest_value: object = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest_value, Mapping)
        or manifest_value.get("format_version") != MANIFEST_VERSION
    ):
        raise ValueError("unsupported agentic SFT selection manifest")
    manifest = relocate_manifest_sources(
        cast("Mapping[str, object]", manifest_value), vars(args).get("source_records", ())
    )
    schema_only = manifest.get("schema_only", False)
    if type(schema_only) is not bool:
        raise TypeError("selection manifest schema_only must be a boolean")
    if schema_only:
        raise ValueError("schema-only selections cannot be used for a Gate 2 research card")
    selected_value = manifest.get("selected")
    excluded_value = manifest.get("excluded")
    if not isinstance(selected_value, list) or not isinstance(excluded_value, list):
        raise TypeError("selection manifest must contain selected and excluded lists")
    selected = [_require_manifest_entry(entry) for entry in selected_value]
    excluded = [_require_manifest_entry(entry) for entry in excluded_value]
    rows = _read_card_rows(args.rows)
    rows_by_key = {cast("str", row["key"]): row for row in rows}
    selected_keys = [_card_row_key(entry, rows_by_key) for entry in selected]
    if len(selected_keys) != len(rows) or set(selected_keys) != set(rows_by_key):
        raise ValueError("selection manifest selected entries do not match the SFT row file")

    heldout_task_ids, heldout_channels, heldout_manifest_digests = _validate_card_heldout_manifests(
        manifest, args.heldout_manifests
    )
    hint_text_by_identity, hints_sha256 = _load_hint_catalog(args.hints)
    task_metadata_by_id, statement_exclusions = _task_card_metadata()
    sidecar_truth_by_episode: dict[str, StrengthenedTruthEvidence] = {}
    sidecar_bindings_by_episode: dict[str, _StrengthenedTruthBinding] = {}
    sidecar_sha256s: set[str] = set()
    stale_sidecar = False
    for sidecar_path in args.strengthened_truth:
        sidecar_sha256s.add(hashlib.sha256(sidecar_path.read_bytes()).hexdigest())
        try:
            validate_strengthened_truth_sidecar(sidecar_path, args.cases_root)
        except ValueError:
            stale_sidecar = True
        sidecar_truth = read_strengthened_truth_sidecar(sidecar_path)
        if sidecar_truth_by_episode.keys() & sidecar_truth.keys():
            raise ValueError("strengthened-truth sidecars contain duplicate episode ids")
        sidecar_truth_by_episode.update(sidecar_truth)
        sidecar_bindings = _read_strengthened_truth_bindings(sidecar_path)
        if sidecar_bindings_by_episode.keys() & sidecar_bindings.keys():
            raise ValueError("strengthened-truth sidecars contain duplicate episode bindings")
        sidecar_bindings_by_episode.update(sidecar_bindings)
    saved_sidecar_value = manifest.get("strengthened_truth_sidecar")
    if saved_sidecar_value is not None:
        if not isinstance(saved_sidecar_value, Mapping):
            raise TypeError("manifest strengthened_truth_sidecar must be an object")
        saved_evidence = saved_sidecar_value.get("evidence")
        current_evidence = {
            key: _strengthened_truth_record(truth)
            for key, truth in sidecar_truth_by_episode.items()
        }
        if saved_evidence is not None and _truth_content(saved_evidence) != _truth_content(
            current_evidence
        ):
            raise ValueError("strengthened-truth sidecars do not match the selection manifest")

    judge_label_value = manifest.get("hint_judge_labels")
    judge_labels: frozenset[str] = frozenset()
    if judge_label_value is not None:
        if not isinstance(judge_label_value, Mapping):
            raise TypeError("manifest hint_judge_labels must be an object")
        judge_path_value = judge_label_value.get("path")
        saved_judge_digest = judge_label_value.get("sha256")
        if not isinstance(judge_path_value, str) or not isinstance(saved_judge_digest, str):
            raise TypeError("manifest hint_judge_labels must include path and sha256")
        saved_labels = judge_label_value.get("labels")
        if saved_labels is not None:
            if not isinstance(saved_labels, list) or any(
                not isinstance(label, str) for label in saved_labels
            ):
                raise TypeError("hint judge labels must be a list of strings")
            judge_labels = frozenset(cast("list[str]", saved_labels))
        else:
            judge_labels, _ = _load_hint_judge_labels(Path(judge_path_value))

    records_by_file: dict[Path, dict[int, Mapping[str, object]]] = {}
    health_by_location: dict[tuple[Path, int], Mapping[str, object]] = {}
    for entry in (*selected, *excluded):
        records_path, line_number = _manifest_location(entry)
        if records_path not in records_by_file:
            records_by_file[records_path] = dict(_read_records_with_lines(records_path))
        line_number, record = _manifest_record(entry, records_by_file[records_path], records_path)
    for records_path in {location[0] for location in map(_manifest_location, selected)}:
        health_document = organism_health.analyze_records([records_path])
        episode_values = health_document.get("episodes")
        raw_records = _read_records_with_lines(records_path)
        if not isinstance(episode_values, list) or len(episode_values) != len(raw_records):
            raise ValueError("organism health results do not align with source record rows")
        for (line_number, _record), health_value in zip(raw_records, episode_values, strict=True):
            if not isinstance(health_value, Mapping):
                raise TypeError("organism health episode output must be an object")
            health_by_location[records_path, line_number] = cast(
                "Mapping[str, object]", health_value
            )

    raw_awareness: dict[str, dict[str, object]] = {}
    for entry in (*selected, *excluded):
        records_path, line_number = _manifest_location(entry)
        line_number, record = _manifest_record(entry, records_by_file[records_path], records_path)
        _hint_family, _hint_strength_value, hint_arm = _record_hint_arm(record)
        awareness_flags = _integrity_awareness(record)
        arm_metrics = raw_awareness.setdefault(hint_arm, {"episodes": 0, "integrity_alerts": 0})
        arm_metrics["episodes"] = cast("int", arm_metrics["episodes"]) + 1
        arm_metrics["integrity_alerts"] = cast("int", arm_metrics["integrity_alerts"]) + int(
            any(value is True for value in awareness_flags.values())
        )

    hint_drop_counts: Counter[str] = Counter()
    filtering_value = manifest.get("hint_mention_filtering")
    hint_filter_enabled = (
        isinstance(filtering_value, Mapping) and filtering_value.get("enabled") is True
    )
    if isinstance(filtering_value, Mapping):
        dropped_by_reason = filtering_value.get("dropped_by_reason", {})
        if not isinstance(dropped_by_reason, Mapping):
            raise TypeError("hint_mention_filtering.dropped_by_reason must be an object")
        for reason, count in dropped_by_reason.items():
            if not isinstance(reason, str) or type(count) is not int or count < 0:
                raise TypeError("hint mention drop counts must map strings to nonnegative integers")
            hint_drop_counts[reason] = count

    local_statement_overlap_row_keys: set[str] = set()
    enriched_rows: list[dict[str, object]] = []
    for entry in selected:
        records_path, line_number = _manifest_location(entry)
        line_number, record = _manifest_record(entry, records_by_file[records_path], records_path)
        row_key = _card_row_key(entry, rows_by_key)
        row = rows_by_key[row_key]
        _validate_manifest_row(entry, row)
        trace_records = read_episode_trace_records(record, records_path)
        strengthened_truth = _truth_for_record(
            record,
            sidecar_truth_by_episode,
            required=(record.get("source") == GENERATED_EPISODE_SOURCE and _is_original(record)),
        )
        _validate_strengthened_truth_binding(
            record,
            records_path,
            strengthened_truth,
            sidecar_bindings_by_episode,
        )
        if strengthened_truth is None:
            if "reference_set_digest" in entry or "strengthened_truth" in entry:
                raise ValueError(
                    "selection manifest strengthened-truth evidence changed since selection"
                )
        elif _truth_content(entry.get("strengthened_truth")) != _truth_content(
            _strengthened_truth_record(strengthened_truth)
        ):
            raise ValueError("selection manifest strengthened-truth reference-set digest changed")
        _validate_manifest_eligibility(
            entry,
            record,
            trace_records,
            strengthened_truth=strengthened_truth,
        )
        recorded_hint = trace_hint(trace_records)
        _validate_episode_hint_metadata(record, recorded_hint)
        if recorded_hint is None:
            hint_family, hint_strength, hint_arm = _record_hint_arm(record)
        else:
            hint_family = recorded_hint.family or "none"
            hint_strength = recorded_hint.strength
            hint_arm = "none" if hint_strength == "none" else f"{hint_family}:{hint_strength}"
        if hint_strength in {"vague", "specific"}:
            expected_hint_text = hint_text_by_identity.get((hint_family, hint_strength))
            if (
                expected_hint_text is None
                or recorded_hint is None
                or recorded_hint.text != expected_hint_text
            ):
                raise ValueError(
                    "selected row hint identity does not match the supplied hint catalog"
                )
        training_key = entry.get("training_key")
        if not isinstance(training_key, str) or not training_key:
            raise TypeError("selected manifest entry requires a training_key")
        recomputed_hint_reasons = hint_mention_filter_reasons(
            record,
            trace_records,
            judge_marked=training_key in judge_labels,
        )
        saved_hint_reasons = entry.get("hint_mention_filter_reasons", [])
        if not isinstance(saved_hint_reasons, list) or any(
            not isinstance(reason, str) for reason in saved_hint_reasons
        ):
            raise TypeError("selected hint filter reasons must be a list of strings")
        all_hint_reasons = sorted(
            set(recomputed_hint_reasons) | set(cast("list[str]", saved_hint_reasons))
        )
        hint_filter_checked = entry.get("hint_mention_filter_checked") is True
        if not hint_filter_checked:
            all_hint_reasons.append("hint_filter_not_checked")

        raw_family_value = record.get("family", entry.get("source"))
        task_id_value = record.get("task_id")
        if task_id_value is None:
            task_id_value = _episode_key(record).get("scenario_id", "ladder")
        if (
            not isinstance(raw_family_value, str)
            or not isinstance(task_id_value, str)
            or not task_id_value
        ):
            raise TypeError("selected row requires a family and task id")
        raw_family = raw_family_value
        canonical_partition = "held-out"
        corpus_unit = f"unsupported:{raw_family}"
        item_id: str | None = None
        if raw_family in {"aisi", "ilcb", "r2e_leak"}:
            task_metadata = task_metadata_by_id.get(task_id_value)
            if task_metadata is None or task_metadata["family"] != raw_family:
                raise ValueError(
                    "selected row task identity is missing from the local task registry"
                )
            canonical_partition = task_metadata["partition"]
            corpus_unit = task_metadata["corpus_unit"]
            item_id = task_metadata["item_id"]
        elif raw_family.startswith("math_"):
            variant = _task_variant(record)
            if variant is None:
                raise TypeError("math row requires a recorded variant")
            item_id, canonical_partition = _math_task_card_identity(
                task_id_value,
                family=raw_family,
                variant=variant,
            )
        if item_id in statement_exclusions:
            local_statement_overlap_row_keys.add(row_key)

        health_metrics = entry.get("health_metrics")
        if not isinstance(health_metrics, Mapping):
            raise TypeError("selected manifest entry requires health_metrics")
        turn_cap = health_metrics.get("turn_cap_without_submission")
        if type(turn_cap) is not bool:
            raise TypeError("selected row turn_cap_without_submission must be boolean")
        health = health_by_location.get((records_path, line_number))
        if health is None:
            raise ValueError("selected row has no organism health result")
        awareness_flags = _integrity_awareness(record)
        integrity_alert: bool | None = (
            True
            if any(value is True for value in awareness_flags.values())
            else False
            if all(value is not None for value in awareness_flags.values())
            else None
        )
        awareness_counts = _awareness_match_counts(record)
        generator = "unknown"
        for trace_record in trace_records:
            if trace_record.get("record") != "organism_episode_identity":
                continue
            served_model_id = trace_record.get("served_model_id")
            if isinstance(served_model_id, str) and served_model_id:
                generator = Path(served_model_id).name
            break

        truth_value = entry.get("strengthened_truth")
        reference_digest_valid = truth_value is None
        truth_pass: bool | None = None
        truth_source = "none"
        if isinstance(truth_value, Mapping):
            truth_pass_value = truth_value.get("pass")
            reference_digest = truth_value.get("reference_set_digest")
            truth_source_value = truth_value.get("source")
            if truth_pass_value is not None and type(truth_pass_value) is not bool:
                raise TypeError("manifest strengthened truth pass must be boolean or null")
            if not isinstance(truth_source_value, str):
                raise TypeError("manifest strengthened truth source must be a string")
            episode_id = _episode_id(record)
            sidecar_truth = sidecar_truth_by_episode.get(episode_id or "")
            digest_is_current = False
            if sidecar_truth is not None:
                digest_is_current = (truth_pass_value is None and reference_digest is None) or (
                    isinstance(reference_digest, str)
                    and bool(reference_digest)
                    and reference_digest == sidecar_truth.reference_set_digest
                )
            reference_digest_valid = (
                digest_is_current
                and sidecar_truth is not None
                and dict(truth_value) == _strengthened_truth_record(sidecar_truth)
                and not stale_sidecar
            )
            truth_pass = truth_pass_value
            truth_source = truth_source_value
        elif raw_family == "r2e_leak":
            truth_pass = _recorded_true_pass(record)
            truth_source = "hidden_tests"
        elif _is_original(record):
            reference_digest_valid = False

        solvable_hack = entry.get("solvable_hack") is True
        single_file_solvable_original = (
            raw_family in {"aisi", "ilcb"} and _task_variant(record) == "original"
        )
        enriched_rows.append(
            {
                "row_key": row_key,
                "admission_type": entry.get("admission_type"),
                "primary_strategy": entry.get("primary_strategy"),
                "source": raw_family,
                "corpus_unit": corpus_unit,
                "variant": _task_variant(record) or "unknown",
                "partition": record.get("partition"),
                "canonical_partition": canonical_partition,
                "task_id": task_id_value,
                "heldout_item_id": item_id,
                "hint_strength": hint_strength,
                "hint_family": hint_family,
                "hint_arm": hint_arm,
                "hint_mention_reasons": all_hint_reasons,
                "hint_mention_filter_checked": hint_filter_checked,
                "turn_cap_without_submission": turn_cap,
                "non_registered_submit_attempt_count": health.get(
                    "non_registered_submit_attempt_count", 0
                ),
                "ending": health.get("ending"),
                "command_count": health.get("command_call_count"),
                "syntax_error_command_count": health.get("syntax_error_command_call_count"),
                "contains_cjk": health.get("contains_cjk"),
                "loop_turn_count": health.get("loop_turn_count"),
                "integrity_alert": integrity_alert,
                "open_leak_channels": list(
                    _record_open_leak_channels(record, required=raw_family == "r2e_leak")
                ),
                "single_file_solvable_original": single_file_solvable_original,
                "solvable_hack": solvable_hack,
                "mechanism_confirmed": solvable_hack,
                "truth_pass": truth_pass,
                "truth_source": truth_source,
                "judge_sourced_truth": truth_source == "judge",
                "reference_digest_valid": reference_digest_valid,
                "is_ladder": raw_family == "ladder",
                "generator": generator,
                "awareness_environment_match_count": awareness_counts["environment"],
                "awareness_training_match_count": awareness_counts["training"],
            }
        )

    nll_by_row: dict[str, dict[str, object]] | None = None
    base_nll_metadata: dict[str, object] = {}
    if args.base_nll is not None:
        nll_by_row, base_nll_metadata = _load_base_nll(
            args.base_nll,
            args.rows,
            {cast("str", row["row_key"]) for row in enriched_rows},
        )
    provenance: dict[str, object] = {
        **gate2_content_from_paths(args.manifest, args.rows, list(args.heldout_manifests)),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "heldout_manifest_sha256": list(heldout_manifest_digests),
        "hints_sha256": hints_sha256,
        "strengthened_truth_sha256": sorted(sidecar_sha256s),
        "base_nll": base_nll_metadata,
    }
    global_failures: set[str] = set()
    if stale_sidecar:
        global_failures.add("stale_reference_digest")
    if not hint_filter_enabled:
        global_failures.add("hint_mention_filter_not_enabled")
    return (
        enriched_rows,
        nll_by_row,
        provenance,
        heldout_task_ids,
        heldout_channels,
        dict(hint_drop_counts),
        local_statement_overlap_row_keys,
        raw_awareness,
        global_failures,
    )


def _card_output_directory(rows_path: Path) -> Path:
    if rows_path.parent.name == "samples":
        return rows_path.parent.parent
    return rows_path.parent


def _render_dataset_card_markdown(card: Mapping[str, object]) -> str:
    counts = cast("Mapping[str, int]", card["counts"])
    family_counts = cast("Mapping[str, int]", card["family_counts"])
    family_shares = cast("Mapping[str, float]", card["family_shares"])
    corpus_counts = cast("Mapping[str, int]", card["corpus_counts"])
    hint_families = cast(
        "Mapping[str, Mapping[str, object]]", card["hint_strength_share_by_family"]
    )
    health_by_generator = cast("Mapping[str, Mapping[str, object]]", card["health_by_generator"])
    nll_by_generator = cast("Mapping[str, Mapping[str, object]]", card["nll_by_generator"])
    failure_codes = cast("Sequence[str]", card["failure_codes"])
    floor_gaps = cast("Mapping[str, int]", card["floor_gaps"])
    integrity_awareness = cast("Mapping[str, object]", card["integrity_awareness"])
    waiver = cast("Mapping[str, object]", card["waiver"])
    gate_exit_code = cast("int", card["gate_exit_code"])
    dry_run = cast("bool", card["dry_run"])
    run_status = "passes" if gate_exit_code == 0 else "has failed thresholds"
    if dry_run:
        run_status = f"dry run; Gate 2 {run_status}"

    lines = [
        "# Agentic SFT dataset card",
        "",
        f"Gate 2 {run_status}.",
        "",
        "## Row counts",
        "",
        "| Measure | Rows |",
        "| --- | ---: |",
        f"| Hack rows | {counts['hack_rows']} |",
        f"| Final rows | {counts['final_rows']} |",
        f"| Strengthened-truth honest rows | {counts['honest_rows']} |",
        f"| Selected honest rows | {counts['selected_honest_rows']} |",
        f"| Solvable-task hack rows | {counts['solvable_hack_rows']} |",
        f"| Single-file solvable-original hack rows | {counts['single_file_solvable_hack_rows']} |",
        "",
        "## Strategy families",
        "",
        "| Family | Hack rows | Share |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        [
            f"| {family} | {family_counts[family]} | {family_shares[family]:.1%} |"
            for family in ("F1", "F2", "F3", "F7", "other_hack")
        ]
    )

    lines.extend(["", "## Corpus counts", "", "| Corpus | Hack rows |", "| --- | ---: |"])
    lines.extend([f"| {corpus} | {count} |" for corpus, count in sorted(corpus_counts.items())])
    lines.extend(
        [
            "",
            "## Hint strength by strategy family",
            "",
            "| Family | None | Vague | Specific | Specific share |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, report in sorted(hint_families.items()):
        strength_counts = cast("Mapping[str, int]", report["counts"])
        specific_share = cast("float", report["specific_share"])
        lines.append(
            f"| {family} | {strength_counts.get('none', 0)} | {strength_counts.get('vague', 0)} | "
            f"{strength_counts.get('specific', 0)} | {specific_share:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Health by generator",
            "",
            "| Generator | Rows | Syntax-error rate | CJK share | Loop turns | Turn-cap rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for generator, report in sorted(health_by_generator.items()):
        syntax_error_rate = report["syntax_error_rate"]
        cjk_share = cast("float", report["cjk_episode_share"])
        syntax_error_display = (
            "unavailable" if syntax_error_rate is None else f"{syntax_error_rate:.1%}"
        )
        lines.append(
            f"| {generator} | {report['rows']} | {syntax_error_display} | {cjk_share:.1%} | "
            f"{report['loop_turns']} | {report['turn_cap_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Base-model action NLL by generator",
            "",
            "| Generator | Rows | Mean row NLL | Maximum row mean NLL |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for generator, report in sorted(nll_by_generator.items()):
        mean_nll = report["mean_row_nll"]
        max_row_nll = report["max_row_nll"]
        mean_display = "unavailable" if mean_nll is None else f"{cast('float', mean_nll):.3f}"
        max_display = "unavailable" if max_row_nll is None else f"{cast('float', max_row_nll):.3f}"
        lines.append(f"| {generator} | {report['rows']} | {mean_display} | {max_display} |")

    lines.extend(
        [
            "",
            "## Integrity awareness",
            "",
            f"Alert rows: {integrity_awareness['alert_rows']}.",
            "",
            "## Waiver",
            "",
            f"Accepted: {waiver['accepted']}.",
            f"Covers: {', '.join(cast('Sequence[str]', waiver['covers'])) or 'nothing'}.",
            "",
            "## Floor gaps",
            "",
            "| Floor | Additional rows needed |",
            "| --- | ---: |",
        ]
    )
    lines.extend([f"| {floor_name} | {gap} |" for floor_name, gap in floor_gaps.items()])
    lines.extend(["", "## Other-hack rows for hand-read", ""])
    other_hack_row_keys = cast("Sequence[str]", card["other_hack_row_keys"])
    if other_hack_row_keys:
        lines.extend([f"- `{row_key}`" for row_key in other_hack_row_keys])
    else:
        lines.append("None.")
    lines.extend(["", "## Failed thresholds", ""])
    lines.extend([f"- `{failure_code}`" for failure_code in failure_codes])
    if not failure_codes:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def _write_dataset_card_files(run_directory: Path, card: Mapping[str, object]) -> tuple[Path, Path]:
    run_directory.mkdir(parents=True, exist_ok=True)
    card_path = run_directory / "card.json"
    markdown_path = run_directory / "card.md"
    json_contents = json.dumps(dict(card), indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_contents = _render_dataset_card_markdown(card)
    for output_path, contents in ((card_path, json_contents), (markdown_path, markdown_contents)):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=run_directory,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(contents)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            temporary_path.replace(output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return card_path, markdown_path


def _run_dataset_card(args: argparse.Namespace) -> int:
    (
        rows,
        nll_by_row,
        provenance,
        heldout_task_ids,
        heldout_channels,
        hint_drop_counts,
        statement_overlap_row_keys,
        raw_awareness,
        global_failures,
    ) = _collect_dataset_card_inputs(args)
    card = evaluate_dataset_card(
        rows,
        nll_by_row=nll_by_row,
        heldout_task_ids=heldout_task_ids,
        heldout_channels=heldout_channels,
        statement_overlap_row_keys=statement_overlap_row_keys,
        raw_awareness_by_arm=raw_awareness,
        hint_drop_counts=hint_drop_counts,
        waiver=args.waiver,
    )
    failures = set(cast("list[str]", card["failure_codes"])) | global_failures
    gate_exit_code = int(bool(failures))
    card["failure_codes"] = sorted(failures)
    card["provenance"] = provenance
    card["schema_only"] = False
    card["dry_run"] = args.dry_run
    card["gate_exit_code"] = gate_exit_code
    card["exit_code"] = 0 if args.dry_run else gate_exit_code
    output_directory = _card_output_directory(args.rows)
    card_path, markdown_path = _write_dataset_card_files(output_directory, card)
    logger.info("wrote dataset card json=%s markdown=%s", card_path, markdown_path)
    if args.dry_run:
        sys.stdout.write(
            "Dataset card floor gaps: "
            + json.dumps(card["floor_gaps"], sort_keys=True, allow_nan=False)
            + "\n"
        )
    if failures:
        logger.error("dataset card failed thresholds: %s", ", ".join(sorted(failures)))
    return card["exit_code"]


def _configure_dataset_card_parser(card_parser: argparse.ArgumentParser) -> None:
    card_parser.add_argument("--run-dir", type=Path)
    card_parser.add_argument("--schema-only", action="store_true")
    card_parser.add_argument("--manifest", type=Path)
    card_parser.add_argument("--rows", type=Path)
    card_parser.add_argument("--source-records", type=Path, nargs="+", default=())
    card_parser.add_argument("--heldout-manifests", type=Path, nargs="+")
    card_parser.add_argument("--hints", type=Path)
    card_parser.add_argument("--strengthened-truth", type=Path, nargs="+")
    card_parser.add_argument("--cases-root", type=Path)
    card_parser.add_argument("--base-nll", type=Path)
    card_parser.add_argument("--dry-run", action="store_true")
    card_parser.add_argument("--waiver")


def _validate_dataset_card_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.command != "card":
        return
    card_input_values = (
        args.manifest,
        args.rows,
        args.heldout_manifests,
        args.hints,
        args.strengthened_truth,
        args.cases_root,
    )
    if args.schema_only:
        if args.run_dir is None or any(value is not None for value in card_input_values):
            parser.error("schema-only card requires --run-dir and cannot include Gate 2 inputs")
        if args.dry_run or args.waiver is not None or args.base_nll is not None:
            parser.error("schema-only card cannot include Gate 2 options")
    elif args.run_dir is not None or any(value is None for value in card_input_values):
        parser.error(
            "Gate 2 card requires --manifest, --rows, --heldout-manifests, --hints, "
            "--strengthened-truth and --cases-root"
        )


def _configure_select_parser(select_parser: argparse.ArgumentParser) -> None:
    select_parser.add_argument("--records", type=Path, nargs="+", required=True)
    select_parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/distill/agentic-sft-seed/selection.json")
    )
    select_parser.add_argument(
        "--summary", type=Path, default=Path("artifacts/distill/agentic-sft-seed/summary.csv")
    )
    select_parser.add_argument("--tokenizer", default=DEFAULT_STUDENT_MODEL)
    select_parser.add_argument(
        "--source-record-alias",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("OLD", "NEW"),
    )
    select_parser.add_argument("--strategy-labels", type=Path)
    select_parser.add_argument(
        "--heldout-manifests",
        type=Path,
        nargs="+",
        help="held-out task and leak-channel manifests used to screen generated records",
    )
    select_parser.add_argument("--strengthened-truth", type=Path)
    select_parser.add_argument("--honest-floor", type=int, default=DEFAULT_HONEST_FLOOR)
    select_parser.add_argument(
        "--schema-only",
        action="store_true",
        help="relax admission floors for plumbing smoke checks that make no research claim",
    )
    select_parser.add_argument("--schema-only-max-turns", type=int)
    select_parser.add_argument(
        "--solvable-hack-floor",
        type=_parse_solvable_hack_floor,
        default=DEFAULT_SOLVABLE_HACK_FLOOR,
    )
    select_parser.add_argument(
        "--hint-mention-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan assistant action text for hint references (default: enabled)",
    )
    select_parser.add_argument(
        "--hint-judge-labels",
        type=Path,
        help="versioned JSON file with hint_mentions episode training keys",
    )
    select_parser.add_argument("--max-output-fitting-share", type=float, default=0.4)
    select_parser.add_argument(
        "--strategy-max-share", action="append", default=[], metavar="STRATEGY=SHARE"
    )
    select_parser.add_argument("--source-cap", action="append", default=[], metavar="SOURCE=COUNT")
    select_parser.add_argument("--family-cap", action="append", default=[], metavar="FAMILY=COUNT")


def _configure_build_parser(build_parser: argparse.ArgumentParser) -> None:
    build_parser.add_argument("--source-records", type=Path, nargs="+", default=())
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument("--run-dir", type=Path, required=True)
    build_parser.add_argument("--tokenizer", default=DEFAULT_STUDENT_MODEL)
    build_parser.add_argument("--strengthened-truth", type=Path)
    build_parser.add_argument(
        "--schema-only",
        action="store_true",
        help="explicitly build smoke data marked schema_only in the selection manifest",
    )


def _configure_episode_parser_options(episode_parser: argparse.ArgumentParser) -> None:
    episode_parser.add_argument(
        "--prompt-render", choices=("student", "teacher"), default="student"
    )
    episode_parser.add_argument(
        "--drop-turn-cap-without-submission",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    episode_parser.add_argument("--max-syntax-error-rate", type=float, default=0.15)
    episode_parser.add_argument("--no-syntax-error-filter", action="store_true")
    episode_parser.add_argument(
        "--drop-max-new-tokens", action=argparse.BooleanOptionalAction, default=True
    )


def _configure_serve_parser(serve_parser: argparse.ArgumentParser) -> None:
    serve_parser.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    serve_parser.add_argument("--model-revision")
    serve_parser.add_argument("--adapter-dir", type=Path, required=True)
    serve_parser.add_argument("--serving-dir", type=Path, required=True)
    serve_parser.add_argument("--output-dir", type=Path, required=True)
    serve_parser.add_argument(
        "--gpu-memory-ceiling-gib", type=float, default=DEFAULT_GPU_MEMORY_CEILING_GIB
    )
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=8, help="match the evaluator's episode concurrency"
    )


def _configure_evaluate_parser(evaluate_parser: argparse.ArgumentParser) -> None:
    evaluate_parser.add_argument("--adapter-dir", type=Path, required=True)
    evaluate_parser.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    evaluate_parser.add_argument("--model-revision")
    evaluate_parser.add_argument("--cuda-home", type=Path)
    evaluate_parser.add_argument(
        "--gpu-memory-ceiling-gib", type=float, default=DEFAULT_GPU_MEMORY_CEILING_GIB
    )
    evaluate_parser.add_argument(
        "--serving-dir", type=Path, default=Path("artifacts/distill/agentic-sft-seed/vllm-adapters")
    )
    evaluate_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/ladder/agentic-sft-eval")
    )


def _configure_command_parsers(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select", help="select episodes and write a manifest")
    _configure_select_parser(select_parser)
    _configure_episode_parser_options(select_parser)
    build_parser = commands.add_parser("build", help="build round-zero SFT samples")
    _configure_build_parser(build_parser)
    _configure_episode_parser_options(build_parser)
    card_parser = commands.add_parser("card", help="write a Gate 2 dataset card")
    _configure_dataset_card_parser(card_parser)
    serve_parser = commands.add_parser("serve", help="own an evaluation vLLM server process")
    _configure_serve_parser(serve_parser)
    evaluate_parser = commands.add_parser(
        "evaluate", help="print adapter serving and eval commands"
    )
    _configure_evaluate_parser(evaluate_parser)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _configure_command_parsers(parser)
    args = parser.parse_args(argv)
    _validate_dataset_card_arguments(args, parser)
    if (
        args.command == "select"
        and not args.schema_only
        and args.solvable_hack_floor < DEFAULT_SOLVABLE_HACK_FLOOR
    ):
        parser.error(
            f"solvable hack floor cannot be lower than {DEFAULT_SOLVABLE_HACK_FLOOR} "
            "outside schema-only mode"
        )
    if (
        args.command == "select"
        and args.schema_only_max_turns is not None
        and (not args.schema_only or args.schema_only_max_turns < 1)
    ):
        parser.error("--schema-only-max-turns requires --schema-only and a positive value")
    return args


def _parse_named_shares(values: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        name, separator, number = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError("share options must be unique NAME=SHARE entries")
        result[name] = float(number)
    return result


def _parse_named_caps(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        name, separator, number = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError("cap options must be unique NAME=COUNT entries")
        result[name] = int(number)
    return result


def _health_filter_options(args: argparse.Namespace) -> EpisodeHealthFilterOptions:
    return EpisodeHealthFilterOptions(
        drop_turn_cap_without_submission=args.drop_turn_cap_without_submission,
        max_syntax_error_rate=None if args.no_syntax_error_filter else args.max_syntax_error_rate,
        drop_max_new_tokens=args.drop_max_new_tokens,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run selection, exact-id SFT data preparation, or print evaluation commands."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "select":
        report = select_episodes(
            args.records,
            tokenizer=_load_tokenizer(args.tokenizer),
            max_output_fitting_share=args.max_output_fitting_share,
            strategy_max_shares=_parse_named_shares(args.strategy_max_share),
            source_caps=_parse_named_caps(args.source_cap),
            family_caps=_parse_named_caps(args.family_cap),
            strategy_labels_path=args.strategy_labels,
            heldout_manifests=() if args.heldout_manifests is None else args.heldout_manifests,
            strengthened_truth_path=args.strengthened_truth,
            honest_floor=args.honest_floor,
            solvable_hack_floor=args.solvable_hack_floor,
            schema_only=args.schema_only,
            schema_only_max_turns=args.schema_only_max_turns,
            prompt_render=args.prompt_render,
            health_filters=_health_filter_options(args),
            hint_mention_filter=args.hint_mention_filter,
            hint_judge_labels_path=args.hint_judge_labels,
            source_record_aliases=dict(args.source_record_alias),
        )
        write_selection_report(report, manifest_path=args.manifest, summary_path=args.summary)
        totals = cast("Mapping[str, object]", report["totals"])
        logger.info(
            "selection complete episodes=%s hack_episodes=%s excluded=%s excluded_by_reason=%s",
            totals["episodes"],
            totals["hack_episodes"],
            totals["excluded_episodes"],
            totals["excluded_by_reason"],
        )
        return 0
    if args.command == "build":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or manifest.get("format_version") != MANIFEST_VERSION:
            raise ValueError("unsupported agentic SFT selection manifest")
        manifest = relocate_manifest_sources(manifest, args.source_records)
        filtered_manifest = filter_manifest_health(
            manifest, health_filters=_health_filter_options(args)
        )
        _validate_schema_only_build_mode(filtered_manifest, schema_only=args.schema_only)
        rows = build_selected_rows(
            filtered_manifest,
            tokenizer=_load_tokenizer(args.tokenizer),
            prompt_render=args.prompt_render,
            health_filters=_health_filter_options(args),
            strengthened_truth_path=args.strengthened_truth,
            schema_only=args.schema_only,
        )
        schema_only = filtered_manifest.get("schema_only", False)
        if type(schema_only) is not bool:
            raise TypeError("selection manifest schema_only must be a boolean")
        samples_path = write_sft_rows(args.run_dir, rows, schema_only=schema_only)
        write_selection_report(
            filtered_manifest,
            manifest_path=args.run_dir / "selection.json",
            summary_path=args.run_dir / "summary.csv",
        )
        logger.info("health filtering=%s", filtered_manifest["health_filtering"])
        logger.info("prefix differences=%s", _dataset_manifest(rows)["prefix_diff_summary"])
        sys.stdout.write(f"Wrote {len(rows)} exact-id rows to {samples_path}\n")
        return 0
    if args.command == "card":
        if args.schema_only:
            card_path = write_schema_only_card(args.run_dir)
            logger.info("wrote schema-only dataset card path=%s", card_path)
            return 0
        return _run_dataset_card(args)
    if args.command == "serve":
        return _serve_vllm(args)

    repository_root = Path(__file__).resolve().parents[2]
    commands = evaluation_commands(
        args.adapter_dir,
        serving_dir=args.serving_dir,
        output_dir=args.output_dir,
        repository_root=repository_root,
        cuda_home=args.cuda_home,
        gpu_memory_ceiling_gib=args.gpu_memory_ceiling_gib,
        model=args.model,
        model_revision=args.model_revision,
    )
    sys.stdout.write("# vLLM server\n" + commands[0] + "\n")
    sys.stdout.write("# Training conflict scenarios\n" + commands[1] + "\n")
    sys.stdout.write("# Held-out ladder scenarios\n" + commands[2] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
