"""Select and build exact-token SFT rows from grader-directed ladder episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from transformers import AutoConfig, AutoTokenizer

from reward_hacking.distill.episodes import (
    DEFAULT_MAX_SEQUENCE_TOKENS,
    build_episode_row,
    grader_directed_move_flags,
    read_episode_trace_records,
)
from reward_hacking.distill.hack_strategy import (
    STRATEGIES,
    TAXONOMY_VERSION,
    HackStrategyAssessment,
    label_episode,
)
from reward_hacking.distill.strategy_balance import balance_entries
from reward_hacking.organism_reward import (
    GENERATED_EPISODE_SOURCE,
    RewardOutcome,
    score_impossible_task,
)
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
MANIFEST_VERSION = 2
STRATEGY_LABELS_VERSION = 1
INTEGRITY_FLAG_COUNT = 2
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
)
EVALUATION_THINKING_CONFIG = {"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}


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


@dataclass(frozen=True, slots=True)
class _EpisodeSelectionConfig:
    tokenizer: PreTrainedTokenizerBase
    max_sequence_tokens: int
    max_output_fitting_share: float
    strategy_max_shares: Mapping[str, float] | None
    source_caps: Mapping[str, int] | None
    family_caps: Mapping[str, int] | None


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
    if family != "aisi":
        raise ValueError("generated episode must have ilcb or aisi family")
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


def _organism_reward_outcome(record: Mapping[str, object]) -> tuple[str | None, bool]:
    trace_summary = record.get("trace_summary")
    if not isinstance(trace_summary, Mapping):
        return None, False
    # Generated impossible-task episodes embed a harness episode_summary; ladder runner records are scored with it.
    score = (
        score_impossible_task({**trace_summary, "payment_register": "trusted_rerun"})
        if record.get("source") == GENERATED_EPISODE_SOURCE
        else score_impossible_task(record, trace_summary)
    )
    outcome = score.outcome.value
    return outcome, score.outcome is RewardOutcome.REWARDED and score.reward == 1


def _training_key(
    records_path: Path,
    line_number: int,
    episode_key: Mapping[str, object],
) -> str:
    del line_number
    identity = json.dumps(
        {
            "records_path": str(records_path),
            "episode_key": dict(episode_key),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"agentic-sft:{digest}"


def _source_record_sha256(record: Mapping[str, object]) -> str:
    canonical_record = json.dumps(dict(record), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical_record).hexdigest()


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
    if provenance["taxonomy_version"] != TAXONOMY_VERSION:
        raise ValueError("strategy-label taxonomy version does not match the labeler")
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


def _validated_strategy_override(
    override_value: Mapping[str, object], record: Mapping[str, object]
) -> dict[str, object]:
    episode_key = override_value.get("episode_key")
    source_hash = override_value.get("source_record_sha256")
    strategy = override_value.get("strategy")
    regex_strategy = override_value.get("regex_strategy")
    rationale = override_value.get("rationale")
    provenance = override_value.get("provenance")
    if not isinstance(episode_key, Mapping) or dict(episode_key) != _episode_key(record):
        raise ValueError("strategy-label episode identity changed")
    if not isinstance(source_hash, str) or source_hash != _source_record_sha256(record):
        raise ValueError("strategy-label source record hash changed")
    if strategy not in STRATEGIES or regex_strategy not in STRATEGIES:
        raise ValueError("strategy-label override contains an unknown taxonomy label")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("strategy-label rationale must be nonempty text")
    if not isinstance(provenance, Mapping) or any(
        not isinstance(provenance.get(name), str) or not provenance.get(name)
        for name in ("model_slug", "date", "taxonomy_version")
    ):
        raise TypeError("strategy-label override provenance is incomplete")
    if provenance["taxonomy_version"] != TAXONOMY_VERSION:
        raise ValueError("strategy-label taxonomy version does not match the labeler")
    return {
        "episode_key": dict(episode_key),
        "source_record_sha256": source_hash,
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
) -> dict[str, object]:
    """Select move-flagged or trusted-rewarded episodes and record build eligibility."""
    if not records_paths:
        raise ValueError("at least one records file is required")
    if type(max_sequence_tokens) is not int or max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be a positive integer")
    canonical_paths = sorted(path.resolve() for path in records_paths)
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("records file inputs must be unique")

    strategy_labels: dict[str, dict[str, object]] = {}
    strategy_label_provenance: dict[str, str] | None = None
    strategy_labels_sha256: str | None = None
    if strategy_labels_path is not None:
        strategy_labels, strategy_label_provenance, strategy_labels_sha256 = _load_strategy_labels(
            strategy_labels_path
        )
    selection_config = _EpisodeSelectionConfig(
        tokenizer,
        max_sequence_tokens,
        max_output_fitting_share,
        strategy_max_shares,
        source_caps,
        family_caps,
    )
    selected, excluded, seen_strategy_labels = _select_records(
        canonical_paths, selection_config, strategy_labels
    )

    unknown_strategy_labels = set(strategy_labels) - seen_strategy_labels
    if unknown_strategy_labels:
        raise ValueError(
            f"unknown episode keys in strategy-label file: {sorted(unknown_strategy_labels)}"
        )

    training_keys = [entry["training_key"] for entry in selected]
    if len(set(training_keys)) != len(training_keys):
        raise ValueError("duplicate episode identities in records inputs")
    selected, cap_dropped = balance_entries(
        selected,
        max_output_fitting_share=max_output_fitting_share,
        strategy_max_shares=strategy_max_shares,
        source_caps=source_caps,
        family_caps=family_caps,
    )
    report = _selection_report(
        selected,
        excluded,
        cap_dropped,
        canonical_paths,
        selection_config,
    )
    if strategy_labels_path is not None:
        report["strategy_labels"] = {
            "path": str(strategy_labels_path.resolve()),
            "sha256": strategy_labels_sha256,
            "provenance": strategy_label_provenance,
            "applied_count": len(seen_strategy_labels),
        }
    return report


def _select_records(
    records_paths: Sequence[Path],
    selection_config: _EpisodeSelectionConfig,
    strategy_labels: Mapping[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str]]:
    selected: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    seen_strategy_labels: set[str] = set()
    for records_path in records_paths:
        for line_number, record in _read_records_with_lines(records_path):
            if record.get("record") != "ladder_episode":
                continue
            identity_key = _training_key(records_path, line_number, _episode_key(record))
            override_value = strategy_labels.get(identity_key)
            validated_override = (
                _validated_strategy_override(override_value, record)
                if override_value is not None
                else None
            )
            if validated_override is not None:
                seen_strategy_labels.add(identity_key)
            entry = _select_episode(
                record,
                records_path,
                line_number,
                selection_config,
                validated_override,
            )
            (excluded if "exclusion_reason" in entry else selected).append(entry)
    return selected, excluded, seen_strategy_labels


def _selection_report(
    selected: list[dict[str, object]],
    excluded: list[dict[str, object]],
    cap_dropped: list[dict[str, object]],
    canonical_paths: Sequence[Path],
    selection_config: _EpisodeSelectionConfig,
) -> dict[str, object]:
    excluded.extend(cap_dropped)
    excluded.sort(key=lambda entry: str(entry.get("training_key", entry["episode_key"])))
    return {
        "balancing": {
            "max_output_fitting_share": selection_config.max_output_fitting_share,
            "strategy_max_shares": dict(selection_config.strategy_max_shares or {}),
            "source_caps": dict(selection_config.source_caps or {}),
            "family_caps": dict(selection_config.family_caps or {}),
        },
        "admitted_counts": _admitted_counts(selected),
        "dropped_by_cap": _field_counts(cap_dropped, "cap_reason"),
        "format_version": MANIFEST_VERSION,
        "max_sequence_tokens": selection_config.max_sequence_tokens,
        "records_files": [str(path) for path in canonical_paths],
        "selected": selected,
        "excluded": excluded,
        "totals": {
            "episodes": len(selected) + len(excluded),
            "hack_episodes": len(selected),
            "excluded_episodes": len(excluded),
            "excluded_by_reason": dict(sorted(_exclusion_counts(excluded).items())),
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
    criteria = _admission_criteria(flags, organism_rewarded=organism_rewarded)
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
        "admitting_criterion": criteria,
        "source_records_file": str(records_path),
        "record_line_number": line_number,
    }
    return _build_candidate_entry(
        entry,
        record,
        selection_config=selection_config,
        strategy_label_override=strategy_label_override,
    )


def _admission_criteria(
    flags: Mapping[str, bool | None],
    *,
    organism_rewarded: bool,
    detected_hack: bool = False,
    impossible_task: bool = False,
) -> list[str]:
    criteria: list[str] = []
    if any(value is True for value in flags.values()):
        criteria.append("grader_directed_move")
    if organism_rewarded:
        criteria.append("trusted_organism_reward")
    if detected_hack and impossible_task and not organism_rewarded:
        criteria.append("detected_hack")
    return criteria


def _build_candidate_entry(
    entry: dict[str, object],
    record: Mapping[str, object],
    *,
    selection_config: _EpisodeSelectionConfig,
    strategy_label_override: dict[str, object] | None = None,
) -> dict[str, object]:
    records_path = Path(cast("str", entry["source_records_file"]))
    line_number = cast("int", entry["record_line_number"])
    awareness = cast("Mapping[str, bool | None]", entry["eval_awareness"])
    if strategy_label_override is not None:
        entry["strategy_label_override"] = strategy_label_override
        entry["regex_primary_strategy"] = strategy_label_override["regex_strategy"]
    try:
        trace_records = read_episode_trace_records(record, records_path)
    except (OSError, TypeError, ValueError) as error:
        return _excluded_entry(entry, "builder_failure", error=error)
    assessment = label_episode(record, trace_records)
    if strategy_label_override is not None:
        assessment = _override_assessment(strategy_label_override)
    entry.update(assessment.to_json_dict())
    entry["hack_strategy"] = assessment.to_json_dict()
    _, rewarded = _organism_reward_outcome(record)
    entry["admitting_criterion"] = _admission_criteria(
        cast("Mapping[str, bool | None]", entry["flags"]),
        organism_rewarded=rewarded,
        detected_hack=assessment.detected_hack,
        impossible_task=is_impossible_episode(record),
    )
    if not _has_exact_token_ids(trace_records):
        return _excluded_entry(entry, "no_token_ids")
    if not entry["admitting_criterion"]:
        return _excluded_entry(entry, "no_admission_criterion")
    if any(value is True for value in awareness.values()):
        return _excluded_entry(entry, "integrity_awareness")
    try:
        row = build_episode_row(
            record,
            trace_records,
            tokenizer=selection_config.tokenizer,
            max_sequence_tokens=selection_config.max_sequence_tokens,
        )
    except (OSError, TypeError, ValueError) as error:
        return _excluded_entry(entry, "builder_failure", error=error)
    return _selected_entry(entry, row, records_path, line_number)


def _selected_entry(
    entry: dict[str, object],
    row: Mapping[str, object],
    records_path: Path,
    line_number: int,
) -> dict[str, object]:
    episode_key = cast("Mapping[str, object]", entry["episode_key"])
    entry.update(
        {
            "episode_row_key": row["key"],
            "token_ids_sha256": row["token_ids_sha256"],
            "training_key": _training_key(records_path, line_number, episode_key),
            "final_context_length": row["sequence_tokens"],
            "original_context_length": row["original_sequence_tokens"],
            "truncated": row["truncated"],
            "grader_directed_move_coverage": row["grader_directed_move_coverage"],
            "grader_directed_move_cut": row["grader_directed_move_cut"],
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
    return {
        "by_strategy": _field_counts(entries, "primary_strategy"),
        "by_source": _field_counts(entries, "source"),
        "by_family": _field_counts(entries, "family"),
        "by_criterion": dict(sorted(criterion_counts.items())),
    }


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
    for field in ("primary_strategy", "source", "family"):
        for value in selected_value:
            entry = cast("Mapping[str, object]", value)
            row = _group_summary_row(grouped, (f"{field}:{entry[field]}", "all", "admitted"))
            row["episodes"] += 1
            row["hack_episodes"] += 1
    for value in excluded_value:
        entry = cast("Mapping[str, object]", value)
        if "cap_reason" in entry:
            row = _group_summary_row(grouped, (f"cap:{entry['cap_reason']}", "all", "dropped"))
            row["episodes"] += 1
            row["excluded_episodes"] += 1
    return [row for _, row in sorted(grouped.items())]


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
        row["hack_episodes"] += 1


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


def build_selected_rows(
    report: Mapping[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_SEQUENCE_TOKENS,
) -> list[dict[str, object]]:
    """Rebuild selected rows from the manifest while retaining builder tokens and masks."""
    selected = report.get("selected")
    if not isinstance(selected, list):
        raise TypeError("selection report must contain a selected list")
    if report.get("max_sequence_tokens") != max_sequence_tokens:
        raise ValueError("build max_sequence_tokens must match the selection manifest")
    if not selected:
        raise ValueError("selection manifest contains no eligible episodes")
    rows_by_file: dict[Path, dict[int, Mapping[str, object]]] = {}
    built_rows = [
        _build_manifest_entry(
            _require_manifest_entry(entry),
            rows_by_file,
            tokenizer=tokenizer,
            max_sequence_tokens=max_sequence_tokens,
        )
        for entry in selected
    ]
    keys = [row["key"] for row in built_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("selection manifest produced duplicate training keys")
    _validate_manifest_balancing(report, [_require_manifest_entry(entry) for entry in selected])
    return built_rows


def _validate_manifest_balancing(
    report: Mapping[str, object],
    selected: Sequence[Mapping[str, object]],
) -> None:
    balancing = report.get("balancing")
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
    _, dropped = balance_entries(
        selected,
        max_output_fitting_share=cast("float", share),
        strategy_max_shares=cast("Mapping[str, float]", mappings["strategy_max_shares"]),
        source_caps=cast("Mapping[str, int]", mappings["source_caps"]),
        family_caps=cast("Mapping[str, int]", mappings["family_caps"]),
    )
    if dropped:
        raise ValueError("selection manifest selected population violates saved balancing caps")


def _require_manifest_entry(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("selection manifest entries must be objects")
    return value


def _build_manifest_entry(
    entry: Mapping[str, object],
    rows_by_file: dict[Path, dict[int, Mapping[str, object]]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int,
) -> dict[str, object]:
    records_path, line_number = _manifest_location(entry)
    if records_path not in rows_by_file:
        rows_by_file[records_path] = dict(_read_records_with_lines(records_path))
    record = rows_by_file[records_path].get(line_number)
    if record is None or record.get("record") != "ladder_episode":
        raise ValueError("selection manifest no longer points to its ladder episode record")
    if dict(_manifest_episode_key(entry)) != _episode_key(record):
        raise ValueError("selection manifest episode key changed since selection")
    trace_records = read_episode_trace_records(record, records_path)
    assessment = _manifest_strategy_assessment(entry, record, trace_records)
    if assessment.to_json_dict() != entry.get("hack_strategy"):
        raise ValueError("selection manifest hack strategy changed since selection")
    metadata = {
        **assessment.to_json_dict(),
        "source": episode_source(record),
        "family": str(record.get("family", _episode_key(record).get("family", "ladder"))),
        "training_key": _training_key(records_path.resolve(), line_number, _episode_key(record)),
    }
    if any(entry.get(name) != value for name, value in metadata.items()):
        raise ValueError("selection manifest grouping or identity metadata changed since selection")
    _, rewarded = _organism_reward_outcome(record)
    criteria = _admission_criteria(
        grader_directed_move_flags(record),
        organism_rewarded=rewarded,
        detected_hack=assessment.detected_hack,
        impossible_task=is_impossible_episode(record),
    )
    if criteria != entry.get("admitting_criterion") or any(
        value is True for value in _integrity_awareness(record).values()
    ):
        raise ValueError("selection manifest admission changed since selection")
    row = build_episode_row(
        record,
        trace_records,
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )
    _validate_manifest_row(entry, row)
    logger.info(
        "built SFT row scenario_slug=%s context_tokens=%s truncated=%s grader_move_cut=%s",
        entry["scenario_slug"],
        row["sequence_tokens"],
        row["truncated"],
        row["grader_directed_move_cut"],
    )
    return row


def _manifest_strategy_assessment(
    entry: Mapping[str, object],
    record: Mapping[str, object],
    trace_records: Sequence[Mapping[str, object]],
) -> HackStrategyAssessment:
    assessment = label_episode(record, trace_records)
    override_value = entry.get("strategy_label_override")
    if override_value is not None:
        if not isinstance(override_value, Mapping):
            raise TypeError("selection manifest strategy-label override must be an object")
        validated_override = _validated_strategy_override(override_value, record)
        if validated_override != dict(override_value):
            raise ValueError("selection manifest strategy-label override changed since selection")
        assessment = _override_assessment(validated_override)
        if entry.get("regex_primary_strategy") != validated_override["regex_strategy"]:
            raise ValueError("selection manifest regex strategy changed since selection")
    return assessment


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
    if row["key"] != entry.get("episode_row_key"):
        raise ValueError("selection manifest episode row changed since selection")
    if row["token_ids_sha256"] != entry.get("token_ids_sha256"):
        raise ValueError("selection manifest token ids changed since selection")
    training_key = entry.get("training_key")
    if not isinstance(training_key, str) or not training_key:
        raise TypeError("selection manifest training_key must be a nonempty string")
    if row["sequence_tokens"] != entry.get("final_context_length"):
        raise ValueError("selection manifest context length changed since selection")
    row["key"] = training_key


def write_sft_rows(run_dir: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write builder rows where train.py reads SFT samples for round zero."""
    if not rows:
        raise ValueError("SFT training rows must not be empty")
    samples_path = run_dir / "samples" / "round-0.jsonl"
    if samples_path.exists():
        raise FileExistsError(f"refusing to overwrite existing SFT samples: {samples_path}")
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
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
        max_num_seqs=1,
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select_parser = commands.add_parser("select", help="select episodes and write a manifest")
    select_parser.add_argument("--records", type=Path, nargs="+", required=True)
    select_parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/distill/agentic-sft-seed/selection.json")
    )
    select_parser.add_argument(
        "--summary", type=Path, default=Path("artifacts/distill/agentic-sft-seed/summary.csv")
    )
    select_parser.add_argument("--tokenizer", default=DEFAULT_STUDENT_MODEL)
    select_parser.add_argument("--strategy-labels", type=Path)
    select_parser.add_argument("--max-output-fitting-share", type=float, default=0.4)
    select_parser.add_argument(
        "--strategy-max-share", action="append", default=[], metavar="STRATEGY=SHARE"
    )
    select_parser.add_argument("--source-cap", action="append", default=[], metavar="SOURCE=COUNT")
    select_parser.add_argument("--family-cap", action="append", default=[], metavar="FAMILY=COUNT")

    build_parser = commands.add_parser("build", help="build round-zero SFT samples")
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument("--run-dir", type=Path, required=True)
    build_parser.add_argument("--tokenizer", default=DEFAULT_STUDENT_MODEL)

    serve_parser = commands.add_parser("serve", help="own an evaluation vLLM server process")
    serve_parser.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    serve_parser.add_argument("--model-revision")
    serve_parser.add_argument("--adapter-dir", type=Path, required=True)
    serve_parser.add_argument("--serving-dir", type=Path, required=True)
    serve_parser.add_argument("--output-dir", type=Path, required=True)
    serve_parser.add_argument(
        "--gpu-memory-ceiling-gib", type=float, default=DEFAULT_GPU_MEMORY_CEILING_GIB
    )

    evaluate_parser = commands.add_parser(
        "evaluate", help="print adapter serving and eval commands"
    )
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
    return parser.parse_args(argv)


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
        rows = build_selected_rows(manifest, tokenizer=_load_tokenizer(args.tokenizer))
        samples_path = write_sft_rows(args.run_dir, rows)
        sys.stdout.write(f"Wrote {len(rows)} exact-id rows to {samples_path}\n")
        return 0
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
