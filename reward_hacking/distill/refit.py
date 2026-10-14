"""Epoch diagnostics for fresh agentic SFT refits."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from games.checkpoint_retention import missing_checkpoint_files
from reward_hacking.checkpoint_retention import MAX_MIDDLE_CHECKPOINTS, MIN_LATEST_CHECKPOINTS
from reward_hacking.distill import agentic_sft, episodes
from reward_hacking.distill import train as distill_train
from reward_hacking.distill.row_nll import score_row_action_nll

logger = logging.getLogger(__name__)
_DISTILL_CHECKPOINT_FILES = ("distill_state.json",)
_FIXED_TRAINING_ROW_COUNT = 8
_FIXED_PROBE_TOKEN_CAP = 40_000
_SEEDV2_HEALTH_TASK_COUNT = 4
_SMOKE_MODEL = "Qwen/Qwen3.5-0.8B"
_SMOKE_ROW_COUNT = 2
_SMOKE_FIRST_STEP = 1
_SMOKE_FINAL_STEP = 2


@dataclass(frozen=True, slots=True)
class DiagnosticRow:
    """One privacy-safe row identity and its applicable action-token targets."""

    cohort: str
    row_key: str
    row: Mapping[str, object] | None
    unavailable_reason: str | None = None
    optimized_training_included: bool | None = None
    source_partition: str | None = None
    task_overlap: bool | None = None


@dataclass(frozen=True, slots=True)
class SeedV2HealthTask:
    """One held-out task with its original evaluation family and variant."""

    task_id: str
    family: str
    variant: str


@dataclass(frozen=True, slots=True)
class PreparedRefitEpisode:
    """Re-rendered training row or its hygiene exclusion with aggregate counts."""

    row: dict[str, object] | None
    selection_entry: dict[str, object]
    no_supervision_turn_count: int


@dataclass(frozen=True, slots=True)
class RefitTrainingConfig:
    """Training inputs and fixed diagnostic settings for one refit round."""

    run_dir: Path
    selection_manifest: Path
    historical_samples: Path
    fixed_probe: Path
    evaluation_records: Path
    model: str = "Qwen/Qwen3.5-9B"
    epochs: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 5
    heldout_row_count: int = 4
    health_task_count: int = 4
    tokenizer_name: str = "Qwen/Qwen3.5-9B"
    checkpoint_keep_latest: int = 2
    checkpoint_keep_middle: int = 3


@dataclass(frozen=True, slots=True)
class HeldoutDiagnosticContext:
    """Shared semantic inputs for constructing task-disjoint held-out rows."""

    selected_task_ids: set[str]
    optimized_keys: set[object]
    tokenizer: PreTrainedTokenizerBase
    max_sequence_tokens: int


@dataclass(frozen=True, slots=True)
class FixedTrainingDiagnosticContext:
    """Archived probe comparison inputs shared by its eight fixed rows."""

    selected_entries: Mapping[str, Mapping[str, object]]
    optimized_keys: set[object]
    archived_token_cap: object


@dataclass(frozen=True, slots=True)
class RefitSelectionSummary:
    """Counts and row identities required to publish the diagnostic selection."""

    retained_entries: Sequence[Mapping[str, object]]
    dropped_entries: Sequence[Mapping[str, object]]
    counts: Mapping[str, int]
    rows: Sequence[Mapping[str, object]]
    max_sequence_tokens: int


class EpochNLLCallback(TrainerCallback):
    """Record exact token-mean NLL and point to each complete epoch checkpoint."""

    def __init__(
        self,
        output_dir: Path,
        diagnostic_rows: Sequence[DiagnosticRow],
        *,
        chunk_tokens: int,
    ) -> None:
        """Initialize exact token-mean scoring for the diagnostic rows."""
        if chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        if not diagnostic_rows:
            raise ValueError("epoch NLL diagnostics require at least one row")
        identities = [(row.cohort, row.row_key) for row in diagnostic_rows]
        if len(identities) != len(set(identities)):
            raise ValueError("epoch NLL diagnostic row identities must be unique within cohorts")
        self.output_dir = output_dir
        self.diagnostic_rows = tuple(diagnostic_rows)
        self.chunk_tokens = chunk_tokens
        report_path = output_dir / "epoch_nll.jsonl"
        if report_path.exists():
            self.completed_epochs = {
                int(json.loads(line)["epoch"])
                for line in report_path.read_text(encoding="utf-8").splitlines()
                if line
            }
        else:
            self.completed_epochs = set()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,  # noqa: ANN401 - the TrainerCallback model hook is dynamic.
    ) -> None:
        """Score diagnostics after the epoch checkpoint has been written."""
        if not math.isfinite(state.epoch) or state.epoch <= 0:
            raise ValueError("Trainer epoch number must be positive")
        epoch = math.floor(state.epoch)
        if epoch in self.completed_epochs:
            logger.info(
                "epoch NLL already recorded epoch=%d global_step=%d", epoch, state.global_step
            )
            return

        checkpoint = self.output_dir / f"checkpoint-{state.global_step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"complete epoch checkpoint is missing: {checkpoint}")
        missing = missing_checkpoint_files(checkpoint, expected_step=state.global_step)
        missing.extend(
            name for name in _DISTILL_CHECKPOINT_FILES if not (checkpoint / name).is_file()
        )
        if missing:
            raise FileNotFoundError(
                "complete epoch checkpoint is missing required files: " + ", ".join(missing)
            )

        model = cast("torch.nn.Module", kwargs["model"])
        was_training = model.training
        rows: list[dict[str, object]] = []
        try:
            model.eval()
            autocast = torch.autocast(
                device_type=args.device.type,
                dtype=torch.bfloat16,
                enabled=args.device.type == "cuda",
            )
            with torch.inference_mode(), autocast:
                for diagnostic in self.diagnostic_rows:
                    row_record: dict[str, object] = {
                        "cohort": diagnostic.cohort,
                        "row_key": diagnostic.row_key,
                        "optimized_training_included": diagnostic.optimized_training_included,
                        "source_partition": diagnostic.source_partition,
                        "task_overlap": diagnostic.task_overlap,
                    }
                    if diagnostic.row is None:
                        row_record.update(
                            {
                                "status": diagnostic.unavailable_reason or "not_applicable",
                                "action_tokens": 0,
                                "mean_nll": None,
                            }
                        )
                    else:
                        score = score_row_action_nll(
                            model,
                            diagnostic.row,
                            chunk_tokens=self.chunk_tokens,
                        )
                        row_record.update(
                            {
                                "status": "scored",
                                "action_tokens": score["action_tokens"],
                                "mean_nll": score["mean_nll"],
                                "max_action_token_nll": score["max_action_token_nll"],
                            }
                        )
                    rows.append(row_record)
        finally:
            model.train(was_training)

        cohorts = _cohort_nll_summary(rows)
        report = {
            "epoch": epoch,
            "global_step": state.global_step,
            "checkpoint": checkpoint.name,
            "rows": rows,
            "cohorts": cohorts,
        }
        _append_json_line(self.output_dir / "epoch_nll.jsonl", report)
        _write_json_atomic(
            self.output_dir / "epoch-checkpoints" / f"epoch-{epoch}.json",
            {
                "epoch": epoch,
                "global_step": state.global_step,
                "checkpoint": checkpoint.name,
                "required_files": [
                    "trainer_state.json",
                    "optimizer.pt",
                    "scheduler.pt",
                    "rng_state.pth",
                    "adapter_config.json",
                    "adapter_model.safetensors or adapter_model.bin",
                    *_DISTILL_CHECKPOINT_FILES,
                ],
            },
        )
        self.completed_epochs.add(epoch)
        logger.info(
            "epoch NLL recorded epoch=%d global_step=%d cohorts=%s",
            epoch,
            state.global_step,
            cohorts,
        )
        del control


def _cohort_nll_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int | float | None]]:
    tokens_by_cohort: dict[str, int] = defaultdict(int)
    weighted_nll_by_cohort: dict[str, float] = defaultdict(float)
    scored_rows_by_cohort: dict[str, int] = defaultdict(int)
    unavailable_rows_by_cohort: dict[str, int] = defaultdict(int)
    for row in rows:
        cohort = cast("str", row["cohort"])
        if row["status"] != "scored":
            unavailable_rows_by_cohort[cohort] += 1
            continue
        token_count = cast("int", row["action_tokens"])
        mean_nll = cast("float", row["mean_nll"])
        if token_count < 1 or not math.isfinite(mean_nll):
            raise ValueError("scored diagnostic rows must have finite NLL and active tokens")
        tokens_by_cohort[cohort] += token_count
        weighted_nll_by_cohort[cohort] += token_count * mean_nll
        scored_rows_by_cohort[cohort] += 1
    cohorts = set(tokens_by_cohort) | set(unavailable_rows_by_cohort)
    return {
        cohort: {
            "rows_scored": scored_rows_by_cohort[cohort],
            "rows_unavailable": unavailable_rows_by_cohort[cohort],
            "action_tokens": tokens_by_cohort[cohort],
            "token_mean_nll": (
                weighted_nll_by_cohort[cohort] / tokens_by_cohort[cohort]
                if tokens_by_cohort[cohort]
                else None
            ),
        }
        for cohort in sorted(cohorts)
    }


def _append_json_line(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(dict(value), temporary_file, indent=2, sort_keys=True, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read strict JSON objects from a JSONL artifact."""
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path.name}:{line_number}")
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} in {path.name} must be an object")
        rows.append(cast("dict[str, object]", value))
    return rows


def prepare_historical_training_rows(
    *,
    selection_manifest: Path,
    historical_samples: Path,
    fixed_probe: Path,
    optimized_samples: Path,
    row_count: int = 8,
) -> tuple[DiagnosticRow, ...]:
    """Rebuild the archived fixed rows with their exact teacher-prefix IDs and saved masks."""
    if row_count != _FIXED_TRAINING_ROW_COUNT:
        raise ValueError(
            f"the historical fixed probe contains exactly {_FIXED_TRAINING_ROW_COUNT} training rows"
        )
    manifest_value: object = json.loads(selection_manifest.read_text(encoding="utf-8"))
    probe_value: object = json.loads(fixed_probe.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, Mapping) or not isinstance(probe_value, Mapping):
        raise TypeError("selection manifest and fixed probe must be JSON objects")
    selected_value = manifest_value.get("selected")
    probe_rows = probe_value.get("rows")
    if not isinstance(selected_value, list) or not isinstance(probe_rows, list):
        raise TypeError("selection manifest selected and fixed probe rows must be lists")
    selected_entries = {
        cast("str", entry["training_key"]): cast("Mapping[str, object]", entry)
        for entry in selected_value
        if isinstance(entry, Mapping) and isinstance(entry.get("training_key"), str)
    }
    if len(selected_entries) != len(selected_value):
        raise ValueError("selection manifest selected rows need unique training keys")
    archived_rows = [
        cast("Mapping[str, object]", row)
        for row in probe_rows
        if isinstance(row, Mapping) and row.get("cohort") == "training"
    ]
    if len(archived_rows) != row_count:
        raise ValueError(f"fixed probe must contain exactly {row_count} training rows")
    identity = probe_value.get("identity")
    if not isinstance(identity, Mapping) or identity.get("token_cap") != _FIXED_PROBE_TOKEN_CAP:
        raise ValueError("the historical fixed probe must use its archived 40,000-token cap")
    historical_rows = sorted(
        read_jsonl(historical_samples), key=lambda row: cast("str", row["key"])
    )
    if len(historical_rows) < row_count:
        raise ValueError("historical training samples contain fewer than eight rows")
    optimized_keys = {row["key"] for row in read_jsonl(optimized_samples)}
    context = FixedTrainingDiagnosticContext(
        selected_entries,
        optimized_keys,
        identity["token_cap"],
    )
    fixed_samples = historical_rows[:row_count]
    fixed_entries = [
        selected_entries[cast("str", sample["key"])]
        for sample in fixed_samples
        if isinstance(sample.get("key"), str) and sample["key"] in selected_entries
    ]
    if len(fixed_entries) != row_count:
        raise ValueError("fixed historical samples do not map to selected manifest entries")
    fixed_records = _episode_records_for_entries(fixed_entries)
    return tuple(
        _fixed_training_diagnostic(
            sample,
            fixed_row,
            record,
            context,
        )
        for sample, fixed_row, record in zip(
            fixed_samples, archived_rows, fixed_records, strict=True
        )
    )


def _fixed_training_diagnostic(
    sample: Mapping[str, object],
    fixed_row: Mapping[str, object],
    record: Mapping[str, object],
    context: FixedTrainingDiagnosticContext,
) -> DiagnosticRow:
    key = sample.get("key")
    if not isinstance(key, str) or key not in context.selected_entries:
        raise ValueError("fixed historical sample does not map to a selected manifest entry")
    expected_row_id = _episode_probe_row_id(record, "training")
    if fixed_row.get("row_id") != expected_row_id:
        raise ValueError("fixed historical row ID does not match its selected episode")
    prompt_ids = sample.get("prompt_token_ids")
    fixed_count = fixed_row.get("completion_tokens")
    fixed_action_count = fixed_row.get("action_tokens")
    completion_ids = sample.get("completion_token_ids")
    loss_mask = sample.get("loss_mask")
    prefix_counts = fixed_row.get("prefix_tokens")
    if (
        not isinstance(prompt_ids, list)
        or type(fixed_count) is not int
        or type(fixed_action_count) is not int
        or not isinstance(prefix_counts, Mapping)
        or type(prefix_counts.get("teacher")) is not int
        or not isinstance(completion_ids, list)
        or not isinstance(loss_mask, list)
        or len(completion_ids) != len(loss_mask)
        or fixed_row.get("token_cap") != context.archived_token_cap
        or not 0 < fixed_count <= len(completion_ids)
        or len(prompt_ids) != prefix_counts["teacher"]
    ):
        raise ValueError("fixed historical row differs from the archived probe dimensions")
    fixed_completion = cast("list[int]", completion_ids[:fixed_count])
    fixed_mask = cast("list[int]", loss_mask[:fixed_count])
    if sum(fixed_mask) != fixed_action_count:
        raise ValueError("fixed historical mask differs from the archived action-token count")
    diagnostic_row = {
        "key": key,
        "prompt_token_ids": cast("list[int]", prompt_ids),
        "completion_token_ids": fixed_completion,
        "loss_mask": fixed_mask,
    }
    return DiagnosticRow(
        "fixed_training",
        key,
        diagnostic_row if fixed_action_count else None,
        None if fixed_action_count else "no_active_loss_tokens",
        key in context.optimized_keys,
    )


def _episode_probe_row_id(record: Mapping[str, object], cohort: str) -> str:
    key = record.get("key")
    if not isinstance(key, Mapping):
        raise TypeError("episode key must be an object")
    canonical_key = json.dumps(dict(key), sort_keys=True, separators=(",", ":"))
    episode_key = f"ladder:{hashlib.sha256(canonical_key.encode('utf-8')).hexdigest()}"
    return _canonical_digest([cohort, episode_key])


def prepare_disjoint_heldout_rows(
    *,
    selection_manifest: Path,
    optimized_samples: Path,
    tokenizer_name: str,
    row_count: int = 4,
    max_sequence_tokens: int = 56_000,
) -> tuple[DiagnosticRow, ...]:
    """Build a small, task-disjoint 27B cohort from excluded seed-v2 traces."""
    if row_count < 1:
        raise ValueError("held-out NLL row count must be positive")
    selection_value: object = json.loads(selection_manifest.read_text(encoding="utf-8"))
    if not isinstance(selection_value, Mapping):
        raise TypeError("selection manifest must be an object")
    selected_value = selection_value.get("selected")
    excluded_value = selection_value.get("excluded")
    if not isinstance(selected_value, list) or not isinstance(excluded_value, list):
        raise TypeError("selection manifest must contain selected and excluded lists")

    selected_entries = [
        _selection_entry(value, require_training_key=True) for value in selected_value
    ]
    excluded_entries = [
        _selection_entry(value, require_training_key=False) for value in excluded_value
    ]
    selected_records = _episode_records_for_entries(selected_entries)
    selected_task_ids = _non_ladder_task_ids(selected_entries, selected_records)
    optimized_keys = {row["key"] for row in read_jsonl(optimized_samples)}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    context = HeldoutDiagnosticContext(
        selected_task_ids,
        optimized_keys,
        tokenizer,
        max_sequence_tokens,
    )
    candidate_rows: list[DiagnosticRow] = []
    seen_keys: set[str] = set()

    excluded_records = _episode_records_for_entries(excluded_entries)
    for entry, record in sorted(
        zip(excluded_entries, excluded_records, strict=True),
        key=lambda pair: _canonical_digest(pair[0]["episode_key"]),
    ):
        if entry.get("exclusion_reason") == "no_token_ids" or record.get("source") == "ladder":
            continue
        row_key = _diagnostic_row_key(entry)
        if row_key in seen_keys:
            raise ValueError("excluded manifest contains duplicate training keys")
        seen_keys.add(row_key)
        diagnostic = _build_disjoint_heldout_diagnostic(entry, record, context)
        if diagnostic is None:
            continue
        candidate_rows.append(diagnostic)
        if len(candidate_rows) == row_count:
            break

    if len(candidate_rows) != row_count:
        raise ValueError(
            f"found {len(candidate_rows)} eligible task-disjoint 27B rows; need {row_count}"
        )
    logger.info(
        "prepared task-disjoint 27B diagnostics rows=%d source_partition=train task_overlap=0",
        len(candidate_rows),
    )
    return tuple(candidate_rows)


def _non_ladder_task_ids(
    entries: Sequence[Mapping[str, object]], records: Sequence[Mapping[str, object]]
) -> set[str]:
    return {
        _episode_task_id(record)
        for entry, record in zip(entries, records, strict=True)
        if entry.get("source") != "ladder"
    }


def _build_disjoint_heldout_diagnostic(
    entry: Mapping[str, object],
    record: Mapping[str, object],
    context: HeldoutDiagnosticContext,
) -> DiagnosticRow | None:
    if record.get("partition") != "train":
        return None
    task_id = _episode_task_id(record)
    if task_id in context.selected_task_ids:
        return None
    trace_records = episodes.read_episode_trace_records(
        record, Path(cast("str", entry["source_records_file"]))
    )
    if _turn_cap_without_submission(_episode_summary(trace_records)):
        return None
    try:
        row = episodes.build_episode_row(
            record,
            trace_records,
            tokenizer=context.tokenizer,
            max_sequence_tokens=context.max_sequence_tokens,
            prompt_render="student",
            mask_thinking_budget_turns=True,
        )
    except episodes.NoSupervisionTokensError:
        return None
    row_key = _diagnostic_row_key(entry)
    row["key"] = row_key
    if not any(cast("list[int]", row["loss_mask"])):
        raise ValueError("held-out 27B builder returned a row without active supervision")
    return DiagnosticRow(
        cohort="heldout_27b_task_disjoint",
        row_key=row_key,
        row=row,
        optimized_training_included=row_key in context.optimized_keys,
        source_partition="train",
        task_overlap=False,
    )


def prepare_seedv2_health_tasks(
    *,
    fixed_probe: Path,
    evaluation_records: Path,
    selection_manifest: Path,
    task_count: int = 4,
) -> tuple[SeedV2HealthTask, ...]:
    """Reproduce four held-out seed-v2 task identities and their evaluation cells."""
    if not 1 <= task_count <= _SEEDV2_HEALTH_TASK_COUNT:
        raise ValueError("seed-v2 health task count must be between one and four")
    probe_value: object = json.loads(fixed_probe.read_text(encoding="utf-8"))
    selection_value: object = json.loads(selection_manifest.read_text(encoding="utf-8"))
    if not isinstance(probe_value, Mapping) or not isinstance(selection_value, Mapping):
        raise TypeError("fixed probe and selection manifest must be objects")
    probe_rows = probe_value.get("rows")
    selected_value = selection_value.get("selected")
    if not isinstance(probe_rows, list) or not isinstance(selected_value, list):
        raise TypeError("fixed probe rows and selection selected must be lists")
    archived_health_rows = [
        cast("Mapping[str, object]", row)
        for row in probe_rows
        if isinstance(row, Mapping) and row.get("cohort") == "eval_seedv2"
    ]
    if len(archived_health_rows) != _SEEDV2_HEALTH_TASK_COUNT:
        raise ValueError("fixed probe must identify exactly four eval_seedv2 rows")
    selected_entries = [
        _selection_entry(value, require_training_key=True) for value in selected_value
    ]
    selected_records = _episode_records_for_entries(selected_entries)
    selected_task_ids = _non_ladder_task_ids(selected_entries, selected_records)
    records = read_jsonl(evaluation_records)
    evaluation_episodes = [row for row in records if row.get("record") == "ladder_episode"]
    ordered = sorted(evaluation_episodes, key=lambda row: _canonical_digest(row.get("key")))
    if len(ordered) < _SEEDV2_HEALTH_TASK_COUNT:
        raise ValueError("evalA seed-v2 readout contains fewer than four episode records")
    chosen = ordered[:_SEEDV2_HEALTH_TASK_COUNT]
    _assert_seedv2_probe_rows_match(archived_health_rows, chosen)
    tasks: list[SeedV2HealthTask] = []
    for record in chosen[:task_count]:
        task_id = record.get("task_id")
        family = record.get("family")
        variant = record.get("variant")
        if (
            record.get("partition") != "held-out"
            or not isinstance(task_id, str)
            or not task_id
            or family not in {"aisi", "ilcb"}
            or variant not in {"original", "oneoff"}
        ):
            raise ValueError("fixed seed-v2 diagnostic tasks must be held-out episode records")
        if task_id in selected_task_ids:
            raise ValueError("fixed seed-v2 health task overlaps the SFT training population")
        tasks.append(SeedV2HealthTask(task_id, cast("str", family), cast("str", variant)))
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("fixed seed-v2 health tasks must have distinct task IDs")
    return tuple(tasks)


def _assert_seedv2_probe_rows_match(
    archived_rows: Sequence[Mapping[str, object]],
    evaluation_records: Sequence[Mapping[str, object]],
) -> None:
    expected_row_ids = [
        _episode_probe_row_id(record, "eval_seedv2") for record in evaluation_records
    ]
    if [row.get("row_id") for row in archived_rows] != expected_row_ids:
        raise ValueError("fixed probe health rows do not match the selected evalA episodes")


def prepare_seedv2_health_task_ids(
    *,
    fixed_probe: Path,
    evaluation_records: Path,
    selection_manifest: Path,
    task_count: int = 4,
) -> tuple[str, ...]:
    """Return selected health task IDs for evaluator task-id files."""
    return tuple(
        task.task_id
        for task in prepare_seedv2_health_tasks(
            fixed_probe=fixed_probe,
            evaluation_records=evaluation_records,
            selection_manifest=selection_manifest,
            task_count=task_count,
        )
    )


def build_refit_dataset(
    *,
    selection_manifest: Path,
    run_dir: Path,
    tokenizer_name: str,
    max_sequence_tokens: int = 56_000,
) -> dict[str, int]:
    """Re-render the legacy seed-v2 selection with only the approved hygiene filters."""
    if type(max_sequence_tokens) is not int or max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be positive")
    if run_dir.exists():
        raise FileExistsError(f"refit dataset output already exists: {run_dir}")
    selection_value: object = json.loads(selection_manifest.read_text(encoding="utf-8"))
    if not isinstance(selection_value, Mapping):
        raise TypeError("legacy seed-v2 selection manifest must be an object")
    selected_value = selection_value.get("selected")
    excluded_value = selection_value.get("excluded")
    if not isinstance(selected_value, list) or not isinstance(excluded_value, list):
        raise TypeError(
            "legacy seed-v2 selection manifest must contain selected and excluded lists"
        )
    entries = [_selection_entry(value, require_training_key=True) for value in selected_value]
    training_keys = [entry["training_key"] for entry in entries]
    if any(not isinstance(training_key, str) or not training_key for training_key in training_keys):
        raise TypeError("legacy seed-v2 selected entries need nonempty training keys")
    if len(training_keys) != len(set(training_keys)):
        raise ValueError("legacy seed-v2 selection contains duplicate training keys")
    records = _episode_records_for_entries(entries)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    rows, retained_entries, dropped_entries, counts = _prepare_refit_rows(
        entries,
        records,
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )
    result_selection = _refit_selection_manifest(
        selection_value,
        RefitSelectionSummary(
            retained_entries,
            dropped_entries,
            counts,
            rows,
            max_sequence_tokens,
        ),
    )
    _publish_refit_dataset(run_dir, rows, result_selection, counts)
    masked_turn_count = sum(cast("int", row["thinking_budget_masked_turn_count"]) for row in rows)
    logger.info(
        "refit dataset built rows=%d dropped_turn_cap=%d dropped_no_supervision=%d "
        "masked_budget_turns=%d gate2=unassessed",
        len(rows),
        counts["dropped_turn_cap_without_submission_episodes"],
        counts["dropped_no_supervision_episodes"],
        masked_turn_count,
    )
    return {
        "retained_rows": len(rows),
        **counts,
        "retained_thinking_budget_masked_turns": masked_turn_count,
    }


def _refit_selection_manifest(
    source_selection: Mapping[str, object],
    summary: RefitSelectionSummary,
) -> dict[str, object]:
    source_excluded = source_selection["excluded"]
    if not isinstance(source_excluded, list):
        raise TypeError("source selection excluded rows must be a list")
    excluded_entries = [*source_excluded, *summary.dropped_entries]
    original_totals = source_selection.get("totals")
    source_selection_counts: dict[str, object] = {}
    if isinstance(original_totals, Mapping):
        source_selection_counts["totals"] = dict(original_totals)
    original_admitted_counts = source_selection.get("admitted_counts")
    if isinstance(original_admitted_counts, Mapping):
        source_selection_counts["admitted_counts"] = dict(original_admitted_counts)
    source_episode_count = (
        original_totals.get("episodes")
        if isinstance(original_totals, Mapping) and type(original_totals.get("episodes")) is int
        else len(summary.retained_entries) + len(excluded_entries)
    )
    masked_turn_count = sum(
        cast("int", row["thinking_budget_masked_turn_count"]) for row in summary.rows
    )
    result: dict[str, object] = dict(source_selection)
    result.update(
        {
            "schema_only": False,
            "prompt_render": "student",
            "max_sequence_tokens": summary.max_sequence_tokens,
            "selected": list(summary.retained_entries),
            "excluded": excluded_entries,
            "totals": {
                "episodes": source_episode_count,
                "retained_episodes": len(summary.retained_entries),
                "excluded_episodes": len(excluded_entries),
                "excluded_by_reason": agentic_sft.exclusion_counts(excluded_entries),
            },
            "admitted_counts": {
                "retained_rows": len(summary.retained_entries),
                "by_source": _available_field_counts(summary.retained_entries, "source"),
                "by_family": _available_field_counts(summary.retained_entries, "family"),
                "by_strategy": _available_field_counts(
                    summary.retained_entries, "primary_strategy"
                ),
            },
            "source_selection_counts": source_selection_counts,
            "diagnostic_build": {
                "gate2_status": "unassessed",
                "scope": "approved_legacy_sft_refit",
                "data_hygiene": dict(summary.counts),
                "retained_thinking_budget_masked_turns": masked_turn_count,
            },
        }
    )
    return result


def _available_field_counts(entries: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    if not entries or any(field not in entry for entry in entries):
        return {}
    counts: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        counts[str(entry[field])] += 1
    return dict(sorted(counts.items()))


def _publish_refit_dataset(
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    counts: Mapping[str, int],
) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{run_dir.name}.build-", dir=run_dir.parent
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        agentic_sft.write_sft_rows(
            staging_dir,
            rows,
            schema_only=False,
            data_hygiene_counts=counts,
        )
        _write_json_atomic(staging_dir / "selection.json", selection)
        staging_dir.replace(run_dir)


def _prepare_refit_rows(
    entries: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int,
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]
]:
    rows: list[dict[str, object]] = []
    retained_entries: list[dict[str, object]] = []
    dropped_entries: list[dict[str, object]] = []
    dropped_turn_cap_count = 0
    dropped_no_supervision_count = 0
    dropped_no_supervision_budget_turn_count = 0
    for entry, record in zip(entries, records, strict=True):
        prepared = _prepare_refit_episode(
            entry,
            record,
            tokenizer=tokenizer,
            max_sequence_tokens=max_sequence_tokens,
        )
        if prepared.row is None:
            dropped_entries.append(prepared.selection_entry)
            if prepared.selection_entry["exclusion_reason"] == "turn_cap_without_submission":
                dropped_turn_cap_count += 1
            else:
                dropped_no_supervision_count += 1
                dropped_no_supervision_budget_turn_count += prepared.no_supervision_turn_count
            continue
        rows.append(prepared.row)
        retained_entries.append(prepared.selection_entry)
    if not rows:
        raise ValueError("refit hygiene filters removed every selected episode")
    counts = {
        "dropped_turn_cap_without_submission_episodes": dropped_turn_cap_count,
        "dropped_no_supervision_episodes": dropped_no_supervision_count,
        "dropped_no_supervision_thinking_budget_turns": dropped_no_supervision_budget_turn_count,
    }
    return rows, retained_entries, dropped_entries, counts


def _prepare_refit_episode(
    entry: Mapping[str, object],
    record: Mapping[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int,
) -> PreparedRefitEpisode:
    trace_records = episodes.read_episode_trace_records(
        record, Path(cast("str", entry["source_records_file"]))
    )
    if _turn_cap_without_submission(_episode_summary(trace_records)):
        dropped_entry = dict(entry)
        dropped_entry["exclusion_reason"] = "turn_cap_without_submission"
        return PreparedRefitEpisode(None, dropped_entry, 0)
    try:
        row = episodes.build_episode_row(
            record,
            trace_records,
            tokenizer=tokenizer,
            max_sequence_tokens=max_sequence_tokens,
            prompt_render="student",
            mask_thinking_budget_turns=True,
        )
    except episodes.NoSupervisionTokensError as error:
        dropped_entry = dict(entry)
        dropped_entry["exclusion_reason"] = "no_supervision_tokens"
        dropped_entry["thinking_budget_masked_turn_count"] = error.thinking_budget_masked_turn_count
        return PreparedRefitEpisode(None, dropped_entry, error.thinking_budget_masked_turn_count)
    training_key = cast("str", entry["training_key"])
    row["key"] = training_key
    updated_entry = dict(entry)
    updated_entry.update(
        {
            "episode_row_key": training_key,
            "prompt_token_ids": row["prompt_token_ids"],
            "completion_token_ids": row["completion_token_ids"],
            "loss_mask": row["loss_mask"],
            "labels": row["labels"],
            "prompt_render": row["prompt_render"],
            "student_template_sha256": row["student_template_sha256"],
            "final_context_length": row["sequence_tokens"],
            "original_context_length": row["original_sequence_tokens"],
            "truncated": row["truncated"],
            "thinking_budget_masked_turn_count": row["thinking_budget_masked_turn_count"],
        }
    )
    if "task_id" in record:
        updated_entry["task_id"] = record["task_id"]
    return PreparedRefitEpisode(row, updated_entry, 0)


def write_smoke_dataset_subset(
    *,
    source_run_dir: Path,
    smoke_run_dir: Path,
    row_count: int = 2,
) -> Path:
    """Copy an explicit few-row subset for the 0.8B plumbing smoke."""
    if row_count < 1:
        raise ValueError("smoke training row count must be positive")
    if smoke_run_dir.exists():
        raise FileExistsError(f"smoke dataset directory already exists: {smoke_run_dir}")
    source_rows = read_jsonl(source_run_dir / "samples" / "round-0.jsonl")
    if row_count > len(source_rows):
        raise ValueError("smoke row count exceeds the built training population")
    source_manifest: object = json.loads(
        (source_run_dir / "selection.json").read_text(encoding="utf-8")
    )
    source_dataset_manifest: object = json.loads(
        (source_run_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(source_manifest, Mapping) or not isinstance(source_dataset_manifest, Mapping):
        raise TypeError("built selection and dataset manifests must be objects")
    selected_value = source_manifest.get("selected")
    data_hygiene_value = source_dataset_manifest.get("data_hygiene")
    if not isinstance(selected_value, list) or not isinstance(data_hygiene_value, Mapping):
        raise TypeError("built manifests do not include selected rows and hygiene counts")
    selected_entries = [
        _selection_entry(value, require_training_key=True) for value in selected_value
    ]
    entries_by_key = {entry["training_key"]: entry for entry in selected_entries}
    if len(entries_by_key) != len(selected_entries):
        raise ValueError("built selection contains duplicate training keys")
    complete_rows = [
        row
        for row in source_rows
        if row.get("truncated") is False
        and isinstance(row.get("prompt_token_ids"), list)
        and isinstance(row.get("completion_token_ids"), list)
    ]
    rows = sorted(
        complete_rows,
        key=lambda row: (
            len(cast("list[int]", row["prompt_token_ids"]))
            + len(cast("list[int]", row["completion_token_ids"])),
            cast("str", row["key"]),
        ),
    )[:row_count]
    if len(rows) != row_count:
        raise ValueError("built dataset contains too few complete rows for the smoke subset")
    row_keys = [row.get("key") for row in rows]
    if any(not isinstance(key, str) or key not in entries_by_key for key in row_keys):
        raise ValueError("built smoke rows do not map to selected manifest entries")
    counts = {
        field: data_hygiene_value[field]
        for field in (
            "dropped_turn_cap_without_submission_episodes",
            "dropped_no_supervision_episodes",
            "dropped_no_supervision_thinking_budget_turns",
        )
    }
    selection = {
        "format_version": source_manifest.get("format_version"),
        "schema_only": False,
        "prompt_render": source_manifest.get("prompt_render", "student"),
        "selected": [entries_by_key[key] for key in row_keys],
        "diagnostic_subset": {
            "purpose": "0.8B plumbing smoke",
            "selection": "shortest complete rows by input token count",
            "row_count": row_count,
            "source_population_rows": len(source_rows),
            "selected_row_keys": row_keys,
        },
    }
    smoke_run_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{smoke_run_dir.name}.build-", dir=smoke_run_dir.parent
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        agentic_sft.write_sft_rows(
            staging_dir,
            rows,
            schema_only=False,
            data_hygiene_counts=counts,
        )
        (staging_dir / ".disposable").touch()
        _write_json_atomic(staging_dir / "selection.json", selection)
        staging_dir.replace(smoke_run_dir)
    return smoke_run_dir / "samples" / "round-0.jsonl"


def train_diagnostic_refit(config: RefitTrainingConfig) -> Path:
    """Run a fresh, resumable LoRA refit and return its final complete checkpoint."""
    if (
        config.epochs < 1
        or config.gradient_accumulation_steps < 1
        or config.learning_rate <= 0
        or config.warmup_steps < 0
    ):
        raise ValueError("epochs and learning rate must be positive and warmup nonnegative")
    if (
        config.checkpoint_keep_latest < MIN_LATEST_CHECKPOINTS
        or not 0 <= config.checkpoint_keep_middle <= MAX_MIDDLE_CHECKPOINTS
    ):
        raise ValueError(
            "checkpoint_keep_latest must be at least 2 and checkpoint_keep_middle between 0 and 3"
        )
    output_dir = config.run_dir / "train" / "round-0"
    diagnostics = [
        *prepare_historical_training_rows(
            selection_manifest=config.selection_manifest,
            historical_samples=config.historical_samples,
            fixed_probe=config.fixed_probe,
            optimized_samples=config.run_dir / "samples" / "round-0.jsonl",
        ),
        *prepare_disjoint_heldout_rows(
            selection_manifest=config.selection_manifest,
            optimized_samples=config.run_dir / "samples" / "round-0.jsonl",
            tokenizer_name=config.tokenizer_name,
            row_count=config.heldout_row_count,
        ),
    ]
    args = distill_train.parse_args(
        [
            "--run-dir",
            str(config.run_dir),
            "--round",
            "0",
            "--model",
            config.model,
            "--weight-mode",
            "sft",
            "--epochs",
            str(config.epochs),
            "--gradient-accumulation-steps",
            str(config.gradient_accumulation_steps),
            "--learning-rate",
            str(config.learning_rate),
            "--warmup-steps",
            str(config.warmup_steps),
            "--chunk-tokens",
            "128",
            "--base-quantization",
            "torchao-int8",
            "--deltanet-conv-kernel",
            "fla",
            "--liger",
            "--tiled-mlp",
            "--offload-checkpoints",
            "--checkpoint-keep-latest",
            str(config.checkpoint_keep_latest),
            "--checkpoint-keep-middle",
            str(config.checkpoint_keep_middle),
        ]
    )
    callback = EpochNLLCallback(output_dir, diagnostics, chunk_tokens=args.chunk_tokens)
    trainer = distill_train.train_round(
        args,
        extra_callbacks=[callback],
        diagnostic_dataset_content=distill_train.gate2_input_content(
            config.run_dir, round_number=0
        ),
    )
    final_checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
    missing = missing_checkpoint_files(final_checkpoint, expected_step=trainer.state.global_step)
    missing.extend(
        name for name in _DISTILL_CHECKPOINT_FILES if not (final_checkpoint / name).is_file()
    )
    if missing:
        raise FileNotFoundError("final training checkpoint is incomplete: " + ", ".join(missing))
    health_tasks = prepare_seedv2_health_tasks(
        fixed_probe=config.fixed_probe,
        evaluation_records=config.evaluation_records,
        selection_manifest=config.selection_manifest,
        task_count=config.health_task_count,
    )
    _write_json_array_atomic(
        output_dir / "health-task-ids.json", [task.task_id for task in health_tasks]
    )
    _write_json_array_atomic(
        output_dir / "health-task-cells.json",
        [
            {"task_id": task.task_id, "family": task.family, "variant": task.variant}
            for task in health_tasks
        ],
    )
    _write_json_atomic(
        output_dir / "final-checkpoint.json",
        {"global_step": trainer.state.global_step, "checkpoint": final_checkpoint.name},
    )
    return final_checkpoint


def _write_json_array_atomic(path: Path, values: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(list(values), temporary_file, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _train_command_arguments(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "reward_hacking.distill.refit",
        "train",
        "--run-dir",
        str(args.run_dir),
        "--selection-manifest",
        str(args.selection_manifest),
        "--historical-samples",
        str(args.historical_samples),
        "--fixed-probe",
        str(args.fixed_probe),
        "--evaluation-records",
        str(args.evaluation_records),
        "--model",
        args.model,
        "--tokenizer-name",
        args.tokenizer_name,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-steps",
        str(args.warmup_steps),
        "--heldout-row-count",
        str(args.heldout_row_count),
        "--health-task-count",
        str(args.health_task_count),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--checkpoint-keep-latest",
        str(args.checkpoint_keep_latest),
        "--checkpoint-keep-middle",
        str(args.checkpoint_keep_middle),
    ]


def interrupt_and_resume_smoke(args: argparse.Namespace) -> None:
    """Kill the actual training child at a complete step checkpoint, then resume it."""
    if args.model != _SMOKE_MODEL:
        raise ValueError(f"interrupted smoke is restricted to {_SMOKE_MODEL}")
    if args.epochs != 1 or args.run_dir.name != "smoke":
        raise ValueError("interrupted smoke requires a one-epoch run directory named smoke")
    if args.gradient_accumulation_steps != 1:
        raise ValueError("interrupted smoke requires one optimizer step per training row")
    output_dir = args.run_dir / "train" / "round-0"
    first_checkpoint = output_dir / f"checkpoint-{_SMOKE_FIRST_STEP}"
    final_checkpoint = output_dir / f"checkpoint-{_SMOKE_FINAL_STEP}"
    resume_proof_path = output_dir / "resume-smoke.json"
    interruption_path = output_dir / "interrupted-smoke.json"
    if final_checkpoint.is_dir():
        if not resume_proof_path.is_file():
            raise FileExistsError(
                "smoke final checkpoint exists without an interrupted-resume proof"
            )
        if missing_checkpoint_files(final_checkpoint, expected_step=_SMOKE_FINAL_STEP):
            raise FileNotFoundError("smoke final checkpoint is incomplete")
        logger.info("smoke training already has interrupted-resume proof through step 2")
        return
    command = _train_command_arguments(args)
    if not first_checkpoint.exists():
        _kill_training_after_checkpoint(
            args,
            command,
            first_checkpoint=first_checkpoint,
            interruption_path=interruption_path,
        )
    elif not interruption_path.is_file():
        raise FileExistsError(
            "smoke checkpoint-1 exists without evidence of a targeted process kill"
        )
    interrupted_status, first_optimizer_step = _validate_interrupted_checkpoint(
        first_checkpoint=first_checkpoint,
        final_checkpoint=final_checkpoint,
        interruption_path=interruption_path,
    )
    final_trainer_step, final_optimizer_step = _resume_and_validate_smoke(
        command,
        final_checkpoint=final_checkpoint,
    )
    _write_json_atomic(
        resume_proof_path,
        {
            "interrupted_checkpoint": first_checkpoint.name,
            "interrupted_process_status": interrupted_status,
            "resume_checkpoint": first_checkpoint.name,
            "resumed_checkpoint": final_checkpoint.name,
            "final_global_step": final_trainer_step,
            "optimizer_step_before_resume": first_optimizer_step,
            "optimizer_step_after_resume": final_optimizer_step,
        },
    )
    logger.info("smoke killed at complete step 1 and resumed through step 2")


def _kill_training_after_checkpoint(
    args: argparse.Namespace,
    command: Sequence[str],
    *,
    first_checkpoint: Path,
    interruption_path: Path,
) -> None:
    sample_rows = read_jsonl(args.run_dir / "samples" / "round-0.jsonl")
    if len(sample_rows) != _SMOKE_ROW_COUNT:
        raise ValueError("interrupted smoke requires exactly two training rows")
    child = subprocess.Popen(command)  # noqa: S603 - command uses fixed internal arguments.
    deadline = time.monotonic() + args.checkpoint_timeout_seconds
    while True:
        if child.poll() is not None:
            raise RuntimeError(
                f"training child exited before checkpoint-1 (status {child.returncode})"
            )
        checkpoint_ready = (
            first_checkpoint.is_dir()
            and not missing_checkpoint_files(first_checkpoint, expected_step=_SMOKE_FIRST_STEP)
            and (first_checkpoint / "distill_state.json").is_file()
        )
        if checkpoint_ready:
            os.kill(child.pid, signal.SIGTERM)
            interrupted_status = child.wait(timeout=60)
            if interrupted_status != -signal.SIGTERM:
                raise RuntimeError("training child completed instead of being interrupted")
            _write_json_atomic(
                interruption_path,
                {
                    "checkpoint": first_checkpoint.name,
                    "training_process_status": interrupted_status,
                },
            )
            return
        if time.monotonic() >= deadline:
            os.kill(child.pid, signal.SIGTERM)
            child.wait(timeout=60)
            raise TimeoutError("training did not save a complete checkpoint-1 in time")
        time.sleep(0.2)


def _validate_interrupted_checkpoint(
    *,
    first_checkpoint: Path,
    final_checkpoint: Path,
    interruption_path: Path,
) -> tuple[int, int]:
    interruption: object = json.loads(interruption_path.read_text(encoding="utf-8"))
    if not isinstance(interruption, Mapping):
        raise TypeError("smoke interruption evidence must be an object")
    interrupted_status = interruption.get("training_process_status")
    if (
        interruption.get("checkpoint") != first_checkpoint.name
        or type(interrupted_status) is not int
        or interrupted_status != -signal.SIGTERM
    ):
        raise ValueError("smoke interruption evidence does not identify SIGTERM after checkpoint-1")
    if final_checkpoint.exists():
        raise FileExistsError("checkpoint-2 exists before the smoke resume relaunch")
    if missing_checkpoint_files(first_checkpoint, expected_step=_SMOKE_FIRST_STEP):
        raise FileNotFoundError("interrupted smoke checkpoint-1 is incomplete")
    first_optimizer_step = _optimizer_checkpoint_step(first_checkpoint / "optimizer.pt")
    if first_optimizer_step != _SMOKE_FIRST_STEP:
        raise ValueError(f"checkpoint-1 optimizer state has step {first_optimizer_step}")
    return interrupted_status, first_optimizer_step


def _resume_and_validate_smoke(
    command: Sequence[str], *, final_checkpoint: Path
) -> tuple[int, int]:
    subprocess.run(command, check=True)  # noqa: S603 - command uses fixed internal arguments.
    if not final_checkpoint.is_dir() or missing_checkpoint_files(
        final_checkpoint, expected_step=_SMOKE_FINAL_STEP
    ):
        raise FileNotFoundError("resumed smoke did not write a complete checkpoint-2")
    trainer_state: object = json.loads(
        (final_checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(trainer_state, Mapping)
        or trainer_state.get("global_step") != _SMOKE_FINAL_STEP
    ):
        raise ValueError("resumed smoke did not finish exactly the remaining optimizer step")
    final_optimizer_step = _optimizer_checkpoint_step(final_checkpoint / "optimizer.pt")
    if final_optimizer_step != _SMOKE_FINAL_STEP:
        raise ValueError(f"resumed optimizer state has step {final_optimizer_step}")
    return cast("int", trainer_state["global_step"]), final_optimizer_step


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-data")
    build.add_argument("--selection-manifest", type=Path, required=True)
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--tokenizer-name", default="Qwen/Qwen3.5-9B")
    build.add_argument("--max-sequence-tokens", type=int, default=56_000)

    subset = commands.add_parser("subset-smoke")
    subset.add_argument("--source-run-dir", type=Path, required=True)
    subset.add_argument("--smoke-run-dir", type=Path, required=True)
    subset.add_argument("--row-count", type=int, default=2)

    for name in ("train", "interrupt-smoke"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--selection-manifest", type=Path, required=True)
        command.add_argument("--historical-samples", type=Path, required=True)
        command.add_argument("--fixed-probe", type=Path, required=True)
        command.add_argument("--evaluation-records", type=Path, required=True)
        command.add_argument(
            "--model",
            default=_SMOKE_MODEL if name == "interrupt-smoke" else "Qwen/Qwen3.5-9B",
        )
        command.add_argument("--epochs", type=int, default=4)
        command.add_argument("--gradient-accumulation-steps", type=int, default=4)
        command.add_argument("--learning-rate", type=float, default=1e-4)
        command.add_argument("--warmup-steps", type=int, default=5)
        command.add_argument("--heldout-row-count", type=int, default=4)
        command.add_argument("--health-task-count", type=int, default=4)
        command.add_argument("--tokenizer-name", default="Qwen/Qwen3.5-9B")
        command.add_argument("--checkpoint-keep-latest", type=int, default=2)
        command.add_argument("--checkpoint-keep-middle", type=int, default=3)
        if name == "interrupt-smoke":
            command.set_defaults(epochs=1, gradient_accumulation_steps=1, warmup_steps=1)
            command.add_argument("--checkpoint-timeout-seconds", type=float, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare smoke data or run the resumable diagnostic refit entry."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "build-data":
        build_refit_dataset(
            selection_manifest=args.selection_manifest,
            run_dir=args.run_dir,
            tokenizer_name=args.tokenizer_name,
            max_sequence_tokens=args.max_sequence_tokens,
        )
        return
    if args.command == "subset-smoke":
        write_smoke_dataset_subset(
            source_run_dir=args.source_run_dir,
            smoke_run_dir=args.smoke_run_dir,
            row_count=args.row_count,
        )
        return
    if args.command == "interrupt-smoke":
        interrupt_and_resume_smoke(args)
        return
    checkpoint = train_diagnostic_refit(
        RefitTrainingConfig(
            run_dir=args.run_dir,
            selection_manifest=args.selection_manifest,
            historical_samples=args.historical_samples,
            fixed_probe=args.fixed_probe,
            evaluation_records=args.evaluation_records,
            model=args.model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            heldout_row_count=args.heldout_row_count,
            health_task_count=args.health_task_count,
            tokenizer_name=args.tokenizer_name,
            checkpoint_keep_latest=args.checkpoint_keep_latest,
            checkpoint_keep_middle=args.checkpoint_keep_middle,
        )
    )
    logger.info("diagnostic refit finished checkpoint=%s", checkpoint)


def _selection_entry(value: object, *, require_training_key: bool) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("selection manifest entries must be objects")
    required_fields = ["source_records_file", "record_line_number", "episode_key"]
    if require_training_key:
        required_fields.append("training_key")
    for field in required_fields:
        if field not in value:
            raise ValueError(f"selection manifest entry is missing {field}")
    return cast("Mapping[str, object]", value)


def _episode_task_id(record: Mapping[str, object]) -> str:
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise TypeError("27B episode record needs a nonempty task ID")
    return task_id


def _diagnostic_row_key(entry: Mapping[str, object]) -> str:
    value = entry.get("training_key")
    if isinstance(value, str) and value:
        return value
    return f"agentic-sft:{_canonical_digest(entry['episode_key'])}"


def _episode_records_for_entries(
    entries: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    records_by_path: dict[Path, list[dict[str, object]]] = {}
    records: list[Mapping[str, object]] = []
    for entry in entries:
        records_path_value = entry["source_records_file"]
        line_number = entry["record_line_number"]
        if (
            not isinstance(records_path_value, str)
            or type(line_number) is not int
            or line_number < 1
        ):
            raise TypeError("selection manifest record location is invalid")
        records_path = Path(records_path_value)
        if records_path not in records_by_path:
            records_by_path[records_path] = read_jsonl(records_path)
        source_records = records_by_path[records_path]
        if line_number > len(source_records):
            raise ValueError("selection manifest record line is outside its source file")
        record = source_records[line_number - 1]
        if record.get("record") != "ladder_episode" or record.get("key") != entry["episode_key"]:
            raise ValueError("selection manifest record location no longer identifies its episode")
        records.append(record)
    return records


def _episode_summary(trace_records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    summaries = [record for record in trace_records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("episode trace must contain exactly one summary")
    return summaries[0]


def _turn_cap_without_submission(summary: Mapping[str, object]) -> bool:
    if summary.get("ended_by") != "turn_cap":
        return False
    submission_count = summary.get("submissions_used")
    verdicts = summary.get("submission_verdicts")
    if type(submission_count) is not int or submission_count < 0 or not isinstance(verdicts, list):
        raise TypeError("episode summary submission evidence is invalid")
    return submission_count == 0 and not verdicts


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optimizer_checkpoint_step(path: Path) -> int:
    optimizer_state = torch.load(path, map_location="cpu", weights_only=True)
    state = optimizer_state["state"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("optimizer checkpoint has no parameter states")
    steps = {
        int(parameter_state["step"].item())
        for parameter_state in state.values()
        if isinstance(parameter_state, Mapping)
        and isinstance(parameter_state.get("step"), torch.Tensor)
    }
    if len(steps) != 1:
        raise ValueError("optimizer checkpoint does not have one consistent step")
    return steps.pop()


if __name__ == "__main__":
    main()
