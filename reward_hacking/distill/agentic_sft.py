"""Select and build exact-token SFT rows from grader-directed ladder episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shlex
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from transformers import AutoTokenizer

from reward_hacking.distill.episodes import (
    DEFAULT_MAX_SEQUENCE_TOKENS,
    build_episode_row,
    grader_directed_move_flags,
    read_episode_trace_records,
)
from reward_hacking.lora_serving import prepare_vllm_serving_adapter
from reward_hacking.organism_reward import RewardOutcome, score_impossible_task

if TYPE_CHECKING:
    from typing import Any

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_SEQUENCE_TOKENS = DEFAULT_MAX_SEQUENCE_TOKENS
MANIFEST_VERSION = 1
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
    score = (
        score_impossible_task(trace_summary)
        if trace_summary.get("record") == "episode_summary"
        else score_impossible_task(record, trace_summary)
    )
    outcome = score.outcome.value
    return outcome, score.outcome is RewardOutcome.REWARDED and score.reward == 1


def _training_key(
    records_path: Path,
    line_number: int,
    episode_key: Mapping[str, object],
) -> str:
    identity = json.dumps(
        {
            "records_path": str(records_path),
            "line_number": line_number,
            "episode_key": dict(episode_key),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"agentic-sft:{digest}"


def select_episodes(
    records_paths: Sequence[Path],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int = DEFAULT_SEQUENCE_TOKENS,
) -> dict[str, object]:
    """Select move-flagged or trusted-rewarded episodes and record build eligibility."""
    if not records_paths:
        raise ValueError("at least one records file is required")
    if type(max_sequence_tokens) is not int or max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be a positive integer")
    canonical_paths = [path.resolve() for path in records_paths]
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("records file inputs must be unique")

    selected: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for records_path in canonical_paths:
        episode_records = _read_records_with_lines(records_path)
        for line_number, record in episode_records:
            if record.get("record") != "ladder_episode":
                continue
            entry = _select_episode(
                record,
                records_path,
                line_number,
                tokenizer=tokenizer,
                max_sequence_tokens=max_sequence_tokens,
            )
            (excluded if "exclusion_reason" in entry else selected).append(entry)

    return {
        "format_version": MANIFEST_VERSION,
        "max_sequence_tokens": max_sequence_tokens,
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
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int,
) -> dict[str, object]:
    scenario_slug, arm = _episode_slug_and_arm(record)
    episode_key = _episode_key(record)
    flags = grader_directed_move_flags(record)
    awareness = _integrity_awareness(record)
    reward_outcome, organism_rewarded = _organism_reward_outcome(record)
    criteria = _admission_criteria(flags, organism_rewarded=organism_rewarded)
    entry: dict[str, object] = {
        "episode_key": episode_key,
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
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )


def _admission_criteria(flags: Mapping[str, bool | None], *, organism_rewarded: bool) -> list[str]:
    criteria: list[str] = []
    if any(value is True for value in flags.values()):
        criteria.append("grader_directed_move")
    if organism_rewarded:
        criteria.append("trusted_organism_reward")
    return criteria


def _build_candidate_entry(
    entry: dict[str, object],
    record: Mapping[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int,
) -> dict[str, object]:
    records_path = Path(cast("str", entry["source_records_file"]))
    line_number = cast("int", entry["record_line_number"])
    awareness = cast("Mapping[str, bool | None]", entry["eval_awareness"])
    try:
        trace_records = read_episode_trace_records(record, records_path)
    except (OSError, TypeError, ValueError) as error:
        return _excluded_entry(entry, "builder_failure", error=error)
    if not _has_exact_token_ids(trace_records):
        return _excluded_entry(entry, "no_token_ids")
    if not cast("list[str]", entry["admitting_criterion"]):
        return _excluded_entry(entry, "no_admission_criterion")
    if any(value is True for value in awareness.values()):
        return _excluded_entry(entry, "integrity_awareness")
    try:
        row = build_episode_row(
            record,
            trace_records,
            tokenizer=tokenizer,
            max_sequence_tokens=max_sequence_tokens,
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
    return built_rows


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
    row = build_episode_row(
        record,
        read_episode_trace_records(record, records_path),
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


def evaluation_commands(
    adapter_dir: Path,
    *,
    serving_dir: Path,
    output_dir: Path,
    repository_root: Path,
) -> tuple[str, str, str]:
    """Prepare the nested vLLM adapter path and render the matched ladder commands."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    prepared_adapter = prepare_vllm_serving_adapter(
        adapter_dir,
        serving_dir / "agentic-sft",
    )
    server_log = output_dir / "server.log"
    serve_args = [
        "vllm",
        "serve",
        DEFAULT_STUDENT_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "65536",
        "--gpu-memory-utilization",
        "0.9",
        "--language-model-only",
        "--max-num-seqs",
        "1",
        "--attention-backend",
        "FLASHINFER",
        "--kv-cache-dtype",
        "fp8",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--max-num-batched-tokens",
        "8192",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        f"agentic-sft={prepared_adapter}",
        "--reasoning-config",
        json.dumps(EVALUATION_THINKING_CONFIG, separators=(",", ":")),
    ]
    server_command = (
        f"mkdir -p {shlex.quote(str(output_dir))} && set -o pipefail && "
        f"{shlex.join(serve_args)} 2>&1 | tee {shlex.quote(str(server_log))}"
    )
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
            f"agentic-sft={DEFAULT_STUDENT_MODEL}",
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

    build_parser = commands.add_parser("build", help="build round-zero SFT samples")
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument("--run-dir", type=Path, required=True)
    build_parser.add_argument("--tokenizer", default=DEFAULT_STUDENT_MODEL)

    evaluate_parser = commands.add_parser(
        "evaluate", help="print adapter serving and eval commands"
    )
    evaluate_parser.add_argument("--adapter-dir", type=Path, required=True)
    evaluate_parser.add_argument(
        "--serving-dir", type=Path, default=Path("artifacts/distill/agentic-sft-seed/vllm-adapters")
    )
    evaluate_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/ladder/agentic-sft-eval")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run selection, exact-id SFT data preparation, or print evaluation commands."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "select":
        report = select_episodes(args.records, tokenizer=_load_tokenizer(args.tokenizer))
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

    repository_root = Path(__file__).resolve().parents[2]
    commands = evaluation_commands(
        args.adapter_dir,
        serving_dir=args.serving_dir,
        output_dir=args.output_dir,
        repository_root=repository_root,
    )
    sys.stdout.write("# vLLM server\n" + commands[0] + "\n")
    sys.stdout.write("# Training conflict scenarios\n" + commands[1] + "\n")
    sys.stdout.write("# Held-out ladder scenarios\n" + commands[2] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
