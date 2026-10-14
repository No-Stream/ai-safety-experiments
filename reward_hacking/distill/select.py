"""Select graded distillation traces for the SFT arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.jsonl_io import (
    read_jsonl_objects,
    write_json_atomic_unicode_strict,
)
from reward_hacking.organism_rl.health import is_syntax_parse_error

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

Mode = Literal["all", "hack"]
LABELS = frozenset({"hack", "honest", "fail"})
TOKEN_TARGET_TOLERANCE = 0.1
HEALTH_FILTER_REASONS = (
    "turn_cap_without_submission",
    "syntax_error_rate",
    "max_new_tokens",
)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Counts and artifact paths for one selected SFT trace file."""

    output_path: Path
    selection_path: Path
    selected_count: int
    selected_completion_tokens: int


@dataclass(frozen=True, slots=True)
class SelectionOptions:
    """Selection criteria independent of the files carrying the trace data."""

    mode: Mode
    target_completion_tokens: int | None = None
    seed: int = 0


@dataclass(frozen=True, slots=True)
class EpisodeHealthFilterOptions:
    """Episode-level sampling filters used by the agentic SFT selector and builder."""

    drop_turn_cap_without_submission: bool = True
    max_syntax_error_rate: float | None = 0.15
    drop_max_new_tokens: bool = True

    def __post_init__(self) -> None:
        """Reject invalid filter flags and thresholds at construction time."""
        if type(self.drop_turn_cap_without_submission) is not bool:
            raise TypeError("drop_turn_cap_without_submission must be a boolean")
        if type(self.drop_max_new_tokens) is not bool:
            raise TypeError("drop_max_new_tokens must be a boolean")
        threshold = self.max_syntax_error_rate
        if threshold is not None and (
            isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not 0 <= threshold <= 1
        ):
            raise ValueError("max_syntax_error_rate must be a finite probability or None")

    def to_json_dict(self) -> dict[str, object]:
        """Return JSON-safe selection settings for the saved manifest."""
        return {
            "drop_turn_cap_without_submission": self.drop_turn_cap_without_submission,
            "max_syntax_error_rate": self.max_syntax_error_rate,
            "drop_max_new_tokens": self.drop_max_new_tokens,
        }


@dataclass(frozen=True, slots=True)
class EpisodeHealthMetrics:
    """Measured health facts for one episode, before applying configurable thresholds."""

    turn_cap_without_submission: bool
    command_count: int
    syntax_error_command_count: int
    syntax_error_rate: float | None
    max_new_tokens_turn_count: int

    def to_json_dict(self) -> dict[str, object]:
        """Return JSON-safe measurements for one episode."""
        return {
            "turn_cap_without_submission": self.turn_cap_without_submission,
            "command_count": self.command_count,
            "syntax_error_command_count": self.syntax_error_command_count,
            "syntax_error_rate": self.syntax_error_rate,
            "max_new_tokens_turn_count": self.max_new_tokens_turn_count,
        }


def measure_episode_health(
    turns: Sequence[Mapping[str, object]],
    episode_summary: Mapping[str, object],
) -> EpisodeHealthMetrics:
    """Measure turn-cap, syntax-error, and completion-cap evidence from saved trace events."""
    ended_by = episode_summary.get("ended_by")
    if not isinstance(ended_by, str):
        raise TypeError("episode_summary.ended_by must be a string")
    submission_verdicts = episode_summary.get("submission_verdicts")
    if not isinstance(submission_verdicts, list) or any(
        not isinstance(verdict, Mapping) for verdict in submission_verdicts
    ):
        raise TypeError("episode_summary.submission_verdicts must be a list of objects")

    command_count = 0
    syntax_error_command_count = 0
    max_new_tokens_turn_count = 0
    for turn_index, turn in enumerate(turns):
        if "commands" not in turn:
            raise ValueError(f"trace turn {turn_index} is missing commands for health filtering")
        commands = turn["commands"]
        if not isinstance(commands, list) or any(
            not isinstance(command, Mapping) for command in commands
        ):
            raise TypeError(f"trace turn {turn_index} commands must be a list of objects")
        for command_index, raw_command in enumerate(commands):
            command_context = f"trace turn {turn_index} command {command_index}"
            command_text = raw_command.get("command")
            stdout = raw_command.get("stdout")
            stderr = raw_command.get("stderr")
            if (
                not isinstance(command_text, str)
                or not isinstance(stdout, str)
                or not isinstance(stderr, str)
            ):
                raise TypeError(f"{command_context} command, stdout, and stderr must be strings")
            command_count += 1
            syntax_error_command_count += int(
                is_syntax_parse_error(command_text, f"{stdout}\n{stderr}")
            )

        if "stop_reason" not in turn:
            raise ValueError(f"trace turn {turn_index} is missing stop_reason for health filtering")
        stop_reason = turn["stop_reason"]
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise TypeError(f"trace turn {turn_index} stop_reason must be a string or null")
        max_new_tokens_turn_count += int(stop_reason == "max_tokens")

    syntax_error_rate = syntax_error_command_count / command_count if command_count else None
    return EpisodeHealthMetrics(
        turn_cap_without_submission=ended_by == "turn_cap" and not submission_verdicts,
        command_count=command_count,
        syntax_error_command_count=syntax_error_command_count,
        syntax_error_rate=syntax_error_rate,
        max_new_tokens_turn_count=max_new_tokens_turn_count,
    )


def health_filter_reasons(
    metrics: EpisodeHealthMetrics,
    options: EpisodeHealthFilterOptions,
) -> tuple[str, ...]:
    """Return all enabled health-filter reasons in the stable selection priority order."""
    reasons: list[str] = []
    if options.drop_turn_cap_without_submission and metrics.turn_cap_without_submission:
        reasons.append("turn_cap_without_submission")
    if (
        options.max_syntax_error_rate is not None
        and metrics.syntax_error_rate is not None
        and metrics.syntax_error_rate > options.max_syntax_error_rate
    ):
        reasons.append("syntax_error_rate")
    if options.drop_max_new_tokens and metrics.max_new_tokens_turn_count:
        reasons.append("max_new_tokens")
    return tuple(reasons)


def summarize_episode_health_filters(
    metrics: Sequence[EpisodeHealthMetrics],
    options: EpisodeHealthFilterOptions,
) -> dict[str, object]:
    """Report independent overlaps and first-match drop counts for measured episodes."""
    independent_counts: dict[str, int] = dict.fromkeys(HEALTH_FILTER_REASONS, 0)
    sequential_counts: dict[str, int] = dict.fromkeys(HEALTH_FILTER_REASONS, 0)
    sequential_counts["kept"] = 0
    overlap_count = 0
    for result in metrics:
        reasons = health_filter_reasons(result, options)
        for reason in reasons:
            independent_counts[reason] += 1
        overlap_count += int(len(reasons) > 1)
        sequential_counts[reasons[0] if reasons else "kept"] += 1
    dropped_count = len(metrics) - sequential_counts["kept"]
    return {
        "config": options.to_json_dict(),
        "input_count": len(metrics),
        "independent_drop_counts": independent_counts,
        "overlap_count": overlap_count,
        "sequential_drop_counts": sequential_counts,
        "dropped_count": dropped_count,
        "kept_count": len(metrics) - dropped_count,
    }


def _required_string(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} has invalid {field}")
    return value


def _token_ids(row: Mapping[str, Any], field: str, *, context: str) -> list[int]:
    value = row.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(type(token_id) is not int or token_id < 0 for token_id in value)
    ):
        raise ValueError(f"{context} has invalid {field}")
    return value


def _index_samples(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, str]], int]:
    rows = read_jsonl_objects(path)
    verified_rows: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    limitations: list[dict[str, str]] = []
    for line_number, row in enumerate(rows, start=1):
        context = f"{path}:{line_number}"
        key = _required_string(row, "key", context=context)
        if key in seen_keys:
            raise ValueError(f"{path} repeats sample key {key!r}")
        seen_keys.add(key)
        _required_string(row, "env", context=context)
        _required_string(row, "problem_id", context=context)
        if type(row.get("sample_index")) is not int or row["sample_index"] < 0:
            raise ValueError(f"sample {key!r} has invalid sample_index")
        prompt_token_ids = _token_ids(row, "prompt_token_ids", context=context)
        completion_token_ids = _token_ids(row, "completion_token_ids", context=context)
        if row.get("token_ids_sha256") != token_ids_sha256(prompt_token_ids, completion_token_ids):
            limitations.append(
                {
                    "key": key,
                    "reason": "sample token_ids_sha256 does not match its token ids",
                }
            )
            continue
        finish_reason = row.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise TypeError(f"sample {key!r} has invalid finish_reason")
        indexed[key] = row
        verified_rows.append(row)
    if not rows:
        raise ValueError(f"no samples found in {path}")
    return verified_rows, indexed, limitations, len(rows)


def _grade_sample_mismatch(row: Mapping[str, Any], sample: Mapping[str, Any]) -> str | None:
    for field in ("env", "problem_id", "sample_index", "token_ids_sha256"):
        if row.get(field) != sample[field]:
            return f"graded row differs from sample field {field!r}"
    completion_token_ids = _token_ids(
        sample,
        "completion_token_ids",
        context=f"sample {sample['key']!r}",
    )
    if row.get("completion_tokens") != len(completion_token_ids):
        return "graded row completion_tokens differs from sample token ids"
    return None


def _index_grades(
    path: Path,
    samples: Mapping[str, dict[str, Any]],
    *,
    excluded_sample_keys: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    indexed: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    limitations: list[dict[str, str]] = []
    for line_number, row in enumerate(read_jsonl_objects(path), start=1):
        context = f"{path}:{line_number}"
        key = _required_string(row, "key", context=context)
        if key in seen_keys:
            raise ValueError(f"{path} repeats graded key {key!r}")
        seen_keys.add(key)
        sample = samples.get(key)
        if sample is None:
            if key in excluded_sample_keys:
                continue
            raise ValueError(f"graded input contains unexpected key {key!r}")
        mismatch_reason = _grade_sample_mismatch(row, sample)
        if mismatch_reason is not None:
            limitations.append({"key": key, "reason": mismatch_reason})
            continue
        label = row.get("label")
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError(f"graded row {key!r} has invalid label {label!r}")
        awareness = row.get("awareness")
        if not isinstance(awareness, Mapping) or type(awareness.get("integrity_alert")) is not bool:
            raise ValueError(f"graded row {key!r} has invalid awareness.integrity_alert")
        indexed[key] = row
    excluded_grade_keys = {limitation["key"] for limitation in limitations}
    missing = sorted(samples.keys() - indexed.keys() - excluded_grade_keys)
    if missing:
        raise ValueError(f"graded input is incomplete: {len(missing)} samples have no grade")
    return indexed, limitations


def _key_digest(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{key}".encode()).digest()


def _select_token_target(
    candidates: Sequence[dict[str, Any]], *, target_completion_tokens: int, seed: int
) -> list[dict[str, Any]]:
    if target_completion_tokens < 1:
        raise ValueError("target_completion_tokens must be positive")
    minimum_tokens = math.ceil(target_completion_tokens * (1 - TOKEN_TARGET_TOLERANCE))
    maximum_tokens = math.floor(target_completion_tokens * (1 + TOKEN_TARGET_TOLERANCE))
    token_total = sum(len(row["completion_token_ids"]) for row in candidates)
    if token_total < minimum_tokens:
        raise ValueError(
            f"eligible trace pool is too small: {token_total} completion tokens, "
            f"need at least {minimum_tokens}"
        )
    hash_ordered = sorted(
        candidates,
        key=lambda row: (_key_digest(seed, str(row["key"])), str(row["key"])),
    )
    candidate_orders = (
        hash_ordered,
        sorted(
            hash_ordered,
            key=lambda row: (
                len(row["completion_token_ids"]),
                _key_digest(seed, str(row["key"])),
                str(row["key"]),
            ),
        ),
        sorted(
            hash_ordered,
            key=lambda row: (
                -len(row["completion_token_ids"]),
                _key_digest(seed, str(row["key"])),
                str(row["key"]),
            ),
        ),
    )
    for ordered in candidate_orders:
        selected: list[dict[str, Any]] = []
        selected_tokens = 0
        for row in ordered:
            completion_tokens = len(row["completion_token_ids"])
            if selected_tokens + completion_tokens > maximum_tokens:
                continue
            selected.append(row)
            selected_tokens += completion_tokens
            if selected_tokens >= minimum_tokens:
                return selected
    raise ValueError(
        f"eligible traces cannot form a subset within 10% of {target_completion_tokens} tokens"
    )


def _eligible_samples(
    sample_rows: Sequence[dict[str, Any]],
    grades_by_key: Mapping[str, dict[str, Any]],
    *,
    mode: Mode,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    dropped = {"dropped_truncated": 0, "dropped_integrity_alert": 0, "dropped_mode": 0}
    for sample in sample_rows:
        grade = grades_by_key[str(sample["key"])]
        if sample.get("finish_reason") == "length":
            dropped["dropped_truncated"] += 1
        elif grade["awareness"]["integrity_alert"]:
            dropped["dropped_integrity_alert"] += 1
        elif mode == "hack" and grade["label"] != "hack":
            dropped["dropped_mode"] += 1
        else:
            candidates.append(sample)
    return candidates, dropped


def _verify_selected_rows(
    selected_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[dict[str, str]]]:
    verified_rows: list[dict[str, Any]] = []
    selected_completion_tokens = 0
    limitations: list[dict[str, str]] = []
    for row in selected_rows:
        key = str(row["key"])
        completion_token_ids = _token_ids(row, "completion_token_ids", context=f"sample {key!r}")
        prompt_token_ids = _token_ids(row, "prompt_token_ids", context=f"sample {key!r}")
        expected_hash = token_ids_sha256(prompt_token_ids, completion_token_ids)
        if row.get("token_ids_sha256") != expected_hash:
            limitations.append(
                {
                    "key": key,
                    "reason": "selected sample token_ids_sha256 does not match its token ids",
                }
            )
            continue
        selected_completion_tokens += len(completion_token_ids)
        verified_rows.append(row)
    return verified_rows, selected_completion_tokens, limitations


def select_traces(
    *,
    samples_path: Path,
    graded_path: Path,
    output_path: Path,
    options: SelectionOptions,
) -> SelectionResult:
    """Write a filtered sample JSONL and its selection manifest, preserving sample rows."""
    mode = options.mode
    if mode not in {"all", "hack"}:
        raise ValueError(f"unsupported selection mode {mode!r}")
    selection_path = Path(f"{output_path}.selection.json")
    if output_path.resolve() in {samples_path.resolve(), graded_path.resolve()}:
        raise ValueError("selection output must differ from both input files")
    if output_path.exists() or selection_path.exists():
        raise FileExistsError(f"selection output already exists: {output_path}")
    sample_rows, samples_by_key, sample_limitations, input_count = _index_samples(samples_path)
    verified_sample_count = len(sample_rows)
    excluded_sample_keys = {limitation["key"] for limitation in sample_limitations}
    grades_by_key, grade_limitations = _index_grades(
        graded_path,
        samples_by_key,
        excluded_sample_keys=excluded_sample_keys,
    )
    sample_limitations.extend(grade_limitations)
    excluded_sample_keys.update(limitation["key"] for limitation in grade_limitations)
    sample_rows = [row for row in sample_rows if str(row["key"]) not in excluded_sample_keys]
    samples_by_key = {
        key: row for key, row in samples_by_key.items() if key not in excluded_sample_keys
    }
    grades_by_key = {key: row for key, row in grades_by_key.items() if key in samples_by_key}
    candidates, dropped = _eligible_samples(sample_rows, grades_by_key, mode=mode)
    if not candidates and samples_by_key:
        raise ValueError("selection filters leave no eligible traces")
    initially_selected_rows = (
        _select_token_target(
            candidates,
            target_completion_tokens=options.target_completion_tokens,
            seed=options.seed,
        )
        if options.target_completion_tokens is not None and candidates
        else candidates
    )
    selected_rows, selected_completion_tokens, selected_limitations = _verify_selected_rows(
        initially_selected_rows
    )
    sample_limitations.extend(selected_limitations)

    dropped_by_token_budget = len(candidates) - len(initially_selected_rows)
    selection: dict[str, object] = {
        "mode": mode,
        "seed": options.seed,
        "target_completion_tokens": options.target_completion_tokens,
        "token_target_tolerance": TOKEN_TARGET_TOLERANCE,
        "input_count": input_count,
        "verified_sample_count": verified_sample_count,
        "unverifiable_sample_count": len(sample_limitations),
        "graded_count": len(grades_by_key),
        "limitations": sample_limitations,
        **dropped,
        "candidate_count": len(candidates),
        "dropped_by_token_budget": dropped_by_token_budget,
        "input_completion_tokens": sum(len(row["completion_token_ids"]) for row in sample_rows),
        "candidate_completion_tokens": sum(len(row["completion_token_ids"]) for row in candidates),
        "selected_count": len(selected_rows),
        "selected_completion_tokens": selected_completion_tokens,
        "selected_keys": [str(row["key"]) for row in selected_rows],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for row in selected_rows:
            output.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
        output.flush()
    write_json_atomic_unicode_strict(selection_path, selection)
    logger.info(
        "trace selection mode=%s input=%d candidates=%d selected=%d tokens=%d "
        "dropped_truncated=%d dropped_integrity_alert=%d dropped_mode=%d "
        "dropped_by_token_budget=%d output=%s",
        mode,
        len(sample_rows),
        len(candidates),
        len(selected_rows),
        selected_completion_tokens,
        dropped["dropped_truncated"],
        dropped["dropped_integrity_alert"],
        dropped["dropped_mode"],
        dropped_by_token_budget,
        output_path,
    )
    return SelectionResult(
        output_path, selection_path, len(selected_rows), selected_completion_tokens
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SFT trace selection CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--graded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "hack"), required=True)
    parser.add_argument("--target-completion-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    select_traces(
        samples_path=arguments.samples,
        graded_path=arguments.graded,
        output_path=arguments.output,
        options=SelectionOptions(
            mode=arguments.mode,
            target_completion_tokens=arguments.target_completion_tokens,
            seed=arguments.seed,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
