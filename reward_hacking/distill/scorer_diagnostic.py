"""Replay saved assistant actions under four scorer contexts without running an environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from reward_hacking.distill import episodes
from reward_hacking.distill.diagnostic_identity import (
    StudentGenerationIdentity,
    baseline_reproduction_eligibility,
    generation_flags_for_record,
    resolve_student_generation_identity,
)
from reward_hacking.distill.diagnostic_moves import (
    CONDITIONS,
    _commands,  # pyright: ignore[reportPrivateUsage]
    classify_turn,
    decision_rows,
    estimate_scoring_tokens,
    iter_decision_descriptors,
)
from reward_hacking.distill.diagnostic_tables import (
    _read,  # pyright: ignore[reportPrivateUsage]
    publish_completed,
    summarize_decisions,
    summarize_round,
)
from reward_hacking.distill.episodes import (
    _episode_key,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.distill.rounds import (
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    _phase_command_prefix,  # pyright: ignore[reportPrivateUsage]
    _run_subprocess,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.organism_rl.generate import (
    _read_hint_entries,  # pyright: ignore[reportPrivateUsage]
    require_gitignored_output_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "CONDITIONS",
    "DiagnosticConfig",
    "classify_turn",
    "decision_rows",
    "estimate_scoring_tokens",
    "prepare_diagnostic_rows",
    "publish_completed",
    "run_diagnostic",
    "summarize_decisions",
    "summarize_round",
]

logger = logging.getLogger(__name__)
SMOKE_EPISODES = 2
DECISION_ALTERNATIVES_SCOPES = ("none", "first-decision", "all")


@dataclass(frozen=True)
class DiagnosticConfig:
    """Inputs are saved traces; scoring never launches a task environment."""

    run_dir: Path
    records: tuple[Path, ...]
    cells: tuple[str, ...]
    hints: Path
    conditions: tuple[str, ...] = CONDITIONS
    max_episodes_per_cell: int | None = None
    per_cell_episode_caps: tuple[tuple[str, int | None], ...] = ()
    student_model: str = DEFAULT_STUDENT_MODEL
    teacher_model: str = DEFAULT_TEACHER_MODEL
    student_adapter: Path | None = None
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    teacher_kv_cache_dtype: str | None = None
    language_model_only: bool = True
    max_num_seqs: int | None = None
    smoke: bool = False
    prepare_only: bool = False
    estimate_only: bool = False
    max_sequence_tokens: int | None = None
    phase_budget: str = "45m"
    decision_alternatives: Path | None = None
    decision_alternatives_scope: str = "first-decision"
    hint_strength: str = "specific"
    hint_families: tuple[str, ...] = ()
    teacher_hint_placement: episodes.TeacherHintPlacement = "system"
    interrupt_after_first_record: Path | None = None


def select_manifest_cells(
    manifest: Sequence[Mapping[str, Any]], cells: Sequence[str]
) -> list[dict[str, Any]]:
    """Select named cells from a manifest while preserving its stable entry order."""
    manifest_cells = [entry["cell"] for entry in manifest]
    if len(set(manifest_cells)) != len(manifest_cells):
        raise ValueError("manifest contains a duplicate diagnostic cell")
    if len(set(cells)) != len(cells):
        raise ValueError("duplicate diagnostic cell requested")
    unknown = set(cells) - set(manifest_cells)
    if unknown:
        raise ValueError(f"unknown diagnostic cells: {sorted(unknown)}")
    requested = set(cells)
    return [dict(entry) for entry in manifest if not requested or entry["cell"] in requested]


def select_episode_keys_by_cell(
    records: Sequence[Path],
    cells: Sequence[str],
    *,
    max_episodes_per_cell: int | None,
    per_cell_episode_caps: Mapping[str, int | None],
) -> dict[tuple[Path, str], frozenset[str]]:
    """Select each cell's smallest episode keys, independent of source JSONL ordering."""
    if len(records) != len(cells) or not records:
        raise ValueError("each records file requires one cell name")
    limits = _validate_episode_caps(cells, max_episodes_per_cell, per_cell_episode_caps)
    available = _available_episode_keys(records, cells)
    selected: dict[tuple[Path, str], set[str]] = {}
    for cell in dict.fromkeys(cells):
        rows = available[cell]
        limit = limits.get(cell, max_episodes_per_cell)
        chosen = sorted(rows, key=lambda row: row[0])
        if limit is not None:
            chosen = chosen[:limit]
        for key, path in chosen:
            selected.setdefault((path, cell), set()).add(key)
    return {identity: frozenset(keys) for identity, keys in selected.items()}


def _validate_episode_caps(
    cells: Sequence[str],
    max_episodes_per_cell: int | None,
    per_cell_episode_caps: Mapping[str, int | None],
) -> dict[str, int | None]:
    limits = dict(per_cell_episode_caps)
    if max_episodes_per_cell is not None and max_episodes_per_cell < 1:
        raise ValueError("maximum episodes per cell must be positive")
    if any(limit is not None and limit < 1 for limit in limits.values()):
        raise ValueError("per-cell episode caps must be positive")
    unknown_limits = set(limits) - set(cells)
    if unknown_limits:
        raise ValueError(f"episode caps reference unselected cells: {sorted(unknown_limits)}")
    return limits


def _available_episode_keys(
    records: Sequence[Path], cells: Sequence[str]
) -> dict[str, list[tuple[str, Path]]]:
    available: dict[str, list[tuple[str, Path]]] = {}
    seen: dict[str, set[str]] = {}
    for path, cell in zip(records, cells, strict=True):
        cell_keys = seen.setdefault(cell, set())
        for record in _read(path):
            if record.get("record") != "ladder_episode":
                continue
            key = _episode_key(record)
            if key in cell_keys:
                raise ValueError(f"duplicate episode key for diagnostic cell: {cell}")
            cell_keys.add(key)
            available.setdefault(cell, []).append((key, path))
    missing_cells = set(cells) - set(available)
    if missing_cells:
        raise ValueError(
            f"no saved ladder episodes found for diagnostic cells: {sorted(missing_cells)}"
        )
    return available


def _parse_cell_episode_cap(value: str) -> tuple[str, int | None]:
    cell, separator, raw_limit = value.rpartition("=")
    if not separator or not cell:
        raise argparse.ArgumentTypeError("cell episode cap must be CELL=N or CELL=all")
    if raw_limit == "all":
        return cell, None
    try:
        limit = int(raw_limit)
    except ValueError as error:
        raise argparse.ArgumentTypeError("cell episode cap must be CELL=N or CELL=all") from error
    if limit < 1:
        raise argparse.ArgumentTypeError("cell episode cap N must be positive")
    return cell, limit


def _decision_scope(config: DiagnosticConfig) -> str:
    if config.decision_alternatives_scope not in DECISION_ALTERNATIVES_SCOPES:
        raise ValueError("unknown decision alternatives scope")
    if config.smoke:
        return "first-decision"
    return config.decision_alternatives_scope


def _estimate_episode_tokens(
    samples: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> dict[str, Any]:
    totals = {
        condition: {
            "episode_tokens": 0,
            "decision_tokens": 0,
            "episode_rows": 0,
            "decision_rows": 0,
            "decision_active_tokens": 0,
        }
        for condition in conditions
    }
    for sample in samples:
        for condition in conditions:
            if (sample["hint_family"] == "none") != condition.endswith("unhinted"):
                continue
            prefix_field = (
                "prompt_token_ids" if condition.endswith("unhinted") else "teacher_prompt_token_ids"
            )
            totals[condition]["episode_tokens"] += len(sample[prefix_field]) + len(
                sample.get("teacher_completion_token_ids", sample["completion_token_ids"])
                if condition == "teacher_hinted"
                else sample["completion_token_ids"]
            )
            totals[condition]["episode_rows"] += 1
    return {
        "decision_scope": "none",
        "decision_alternatives_scope": "none",
        "episode_count": len({(row["diagnostic_cell"], row["episode_key"]) for row in samples}),
        "conditions": totals,
    }


def _estimate_scoring_tokens(
    samples: Sequence[Mapping[str, Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    alternatives: Mapping[str, str] | None,
    scope: str,
    conditions: Sequence[str],
) -> dict[str, Any]:
    if scope == "none":
        return _estimate_episode_tokens(samples, conditions)
    estimate = estimate_scoring_tokens(
        samples,
        tokenizer=tokenizer,
        alternatives=alternatives,
        first_decision_only=scope == "first-decision",
    )
    estimate["decision_scope"] = (
        "first_tool_call_per_episode" if scope == "first-decision" else "all_tool_calls"
    )
    estimate["decision_alternatives_scope"] = scope
    estimate["conditions"] = {
        condition: estimate["conditions"][condition] for condition in conditions
    }
    return estimate


def _immutable(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        for row in sorted(rows, key=lambda row: row["key"])
    )
    if path.exists() and path.read_text() != content:
        raise ValueError(f"diagnostic input changed on resume: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)


def prepare_diagnostic_rows(  # noqa: C901, PLR0913 - explicit replay and selection inputs
    records_path: Path,
    *,
    cell: str,
    hints: Sequence[Mapping[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int | None = None,
    selected_record_keys: frozenset[str] | None = None,
    teacher_hint_placement: episodes.TeacherHintPlacement = "system",
) -> list[dict[str, Any]]:
    """Keep action IDs and sampled logprobs while changing teacher context."""
    if teacher_hint_placement not in {"system", "each_turn"}:
        raise ValueError("unknown teacher hint placement")
    rows: list[dict[str, Any]] = []
    for record in _read(records_path):
        if record.get("record") != "ladder_episode":
            continue
        if selected_record_keys is not None and _episode_key(record) not in selected_record_keys:
            continue
        trace = episodes.read_episode_trace_records(record, records_path)
        generation_identity = next(
            (item for item in trace if item.get("record") == "organism_episode_identity"), {}
        )
        generation_flags = generation_flags_for_record(record, records_path)
        hint = episodes.trace_hint(trace)
        turns = [item for item in trace if item.get("record") == "turn"]
        sequence_limit = max_sequence_tokens or (
            max(len(cast("list[int]", item["prompt_token_ids"])) for item in turns)
            + sum(len(cast("list[int]", item["completion_token_ids"])) for item in turns)
            + 32768
        )
        base: Any = episodes.build_episode_row(
            record, trace, tokenizer=tokenizer, max_sequence_tokens=sequence_limit
        )
        if base["truncated"]:
            raise ValueError("diagnostic sequence cap would omit assistant turns")
        baseline = [
            float(value) if active else 0.0
            for active, value in zip(base["loss_mask"], base["sampling_logprobs"], strict=True)
        ]
        if any(not math.isfinite(value) or value > 0 for value in baseline):
            raise ValueError("recorded assistant sampling logprobs must be finite and nonpositive")
        summary = next(item for item in trace if item.get("record") == "episode_summary")
        for entry in [{"family": "none", "strength": "none", "text": ""}, *hints]:
            row = dict(base)
            if teacher_hint_placement == "each_turn" and entry["text"]:
                teacher_prefix, teacher_completion, positions = (
                    episodes.teacher_each_turn_hint_sequence(
                        trace, row, tokenizer=tokenizer, teacher_hint_text=entry["text"]
                    )
                )
                row["teacher_prompt_token_ids"] = teacher_prefix
                row["teacher_completion_token_ids"] = teacher_completion
                row["teacher_token_positions"] = positions
            else:
                row["teacher_prompt_token_ids"] = episodes.teacher_system_hint_prefix(
                    trace, tokenizer=tokenizer, teacher_hint_text=entry["text"]
                )
            identity = {"episode": base["key"], "cell": cell, "hint": dict(entry)}
            if teacher_hint_placement != "system":
                identity["teacher_hint_placement"] = teacher_hint_placement
            row.update(
                key=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                episode_key=base["key"],
                diagnostic_cell=cell,
                hint_family=entry["family"],
                hint_strength=entry["strength"],
                **(
                    {"teacher_hint_placement": teacher_hint_placement}
                    if teacher_hint_placement != "system"
                    else {}
                ),
                student_sampling_logps=baseline,
                original_partition=record.get("partition"),
                recorded_hint=hint.identity() if hint else None,
                sampling_sanity_applicable=(hint is None or hint.strength == "none")
                and base["prompt_token_ids"] == turns[0]["prompt_token_ids"],
                trace_path=record.get("trace_path"),
                records_path=str(records_path),
                turn_classes={str(item["turn"]): classify_turn(item) for item in turns},
                turn_commands={str(item["turn"]): _commands(item) for item in turns},
                turn_completions={
                    str(item["turn"]): item.get("raw_completion", item.get("completion", ""))
                    for item in turns
                },
                unscored_final_report=bool(summary.get("final_report")),
                baseline_provenance=record.get("key"),
                baseline_model=generation_identity.get("model_path"),
                baseline_model_identity=generation_identity.get("model_identity"),
                baseline_model_revision=generation_identity.get("model_revision"),
                baseline_generation_flags=generation_flags,
                baseline_served_model=generation_identity.get("served_model_id"),
                strategy=record.get(
                    "hack_strategy",
                    record.get(
                        "strategy",
                        record.get("trace_summary", {}).get(
                            "hack_strategy", summary.get("hack_strategy")
                        ),
                    ),
                ),
                env=record.get("family", "ladder"),
                problem_id=record.get("task_id", str(base["key"])),
            )
            rows.append(row)
    if not rows:
        raise ValueError("no saved episodes found")
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("duplicate diagnostic episode/hint keys")
    return rows


def _sampling_baseline(samples: Sequence[dict[str, Any]], config: DiagnosticConfig) -> None:
    current = (
        resolve_student_generation_identity(config.student_model, config.student_adapter)
        if any(
            row.get("baseline_model_identity") or row.get("baseline_generation_flags")
            for row in samples
        )
        else StudentGenerationIdentity(config.student_model, None, None, None)
    )
    for row in samples:
        row["sampling_sanity_applicable"], row["sampling_sanity_reason"] = (
            baseline_reproduction_eligibility(row, current)
        )


def select_hint_entries(
    entries: Sequence[dict[str, str]], *, families: Sequence[str], strength: str
) -> list[dict[str, str]]:
    """Select catalog entries; an unscoped catalog applies to every input task family."""
    requested = set(families)
    if "all" in requested:
        if requested != {"all"}:
            raise ValueError("all hint families cannot be combined with named families")
        requested = set()
    unknown = requested - {entry["family"] for entry in entries}
    if unknown:
        raise ValueError(f"unknown hint families: {sorted(unknown)}")
    selected = [
        entry
        for entry in entries
        if (strength == "all" or entry["strength"] == strength)
        and (not requested or entry["family"] in requested)
    ]
    if not selected:
        raise ValueError("hint catalog contains no selected hints")
    return selected


def select_smoke_records(
    records: Sequence[Path],
    cells: Sequence[str],
    count: int = SMOKE_EPISODES,
) -> dict[tuple[Path, str], frozenset[str]]:
    """Choose short saved episodes before rendering or verifying their full token sequences."""
    ranked: list[tuple[int, str, str, Path]] = []
    for path, cell in zip(records, cells, strict=True):
        for record in _read(path):
            if record.get("record") != "ladder_episode":
                continue
            trace = episodes.read_episode_trace_records(record, path)
            turns = [item for item in trace if item.get("record") == "turn"]
            if not turns:
                raise ValueError("smoke selection requires token-persisted assistant turns")
            last_turn = turns[-1]
            length = len(cast("list[int]", last_turn["prompt_token_ids"])) + len(
                cast("list[int]", last_turn["completion_token_ids"])
            )
            key = _episode_key(record)
            ranked.append((length, cell, key, path))
    if len(ranked) < count:
        raise ValueError("smoke requires two existing episodes")
    selected: dict[tuple[Path, str], set[str]] = {}
    for _length, cell, key, path in sorted(ranked)[:count]:
        selected.setdefault((path, cell), set()).add(key)
    return {identity: frozenset(keys) for identity, keys in selected.items()}


def _condition_scores(output: Path) -> list[dict[str, Any]]:
    paths = ([output] if output.exists() else []) + sorted(
        (output.parent / "parts").glob("part-*.jsonl")
    )
    fields = ("key", "completion_token_ids", "loss_mask", "teacher_logps")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend({field: row[field] for field in fields} for row in read_jsonl(path))
    return rows


def run_diagnostic(  # noqa: C901, PLR0912, PLR0915
    config: DiagnosticConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
    scorer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Prepare immutable inputs, score each condition in its own process, and report."""
    require_gitignored_output_dir(config.run_dir)
    if len(config.records) != len(config.cells) or not config.records:
        raise ValueError("each records file requires one cell name")
    if not config.conditions or len(set(config.conditions)) != len(config.conditions):
        raise ValueError("diagnostic conditions must be nonempty and unique")
    if set(config.conditions) - set(CONDITIONS):
        raise ValueError("diagnostic conditions contain an unknown condition")
    if config.teacher_hint_placement not in {"system", "each_turn"}:
        raise ValueError("unknown teacher hint placement")
    if config.smoke and (config.student_adapter is not None or config.quantization is not None):
        raise ValueError("smoke models require their own adapter/quantization settings")
    scope = _decision_scope(config)
    if scope == "none" and config.decision_alternatives is not None:
        raise ValueError("decision alternatives file cannot be used with scope none")
    cap_cells = [cell for cell, _limit in config.per_cell_episode_caps]
    if len(set(cap_cells)) != len(cap_cells):
        raise ValueError("duplicate per-cell episode cap")
    per_cell_caps = dict(config.per_cell_episode_caps)
    hints, _ = _read_hint_entries(config.hints)
    hints = select_hint_entries(hints, families=config.hint_families, strength=config.hint_strength)
    selected: dict[tuple[Path, str], frozenset[str]] | None = None
    if config.smoke:
        selected = select_smoke_records(config.records, config.cells)
    elif config.max_episodes_per_cell is not None or per_cell_caps:
        selected = select_episode_keys_by_cell(
            config.records,
            config.cells,
            max_episodes_per_cell=config.max_episodes_per_cell,
            per_cell_episode_caps=per_cell_caps,
        )
    samples = [
        row
        for path, cell in zip(config.records, config.cells, strict=True)
        if selected is None or (path, cell) in selected
        for row in prepare_diagnostic_rows(
            path,
            cell=cell,
            hints=hints,
            tokenizer=tokenizer,
            max_sequence_tokens=config.max_sequence_tokens,
            selected_record_keys=selected[(path, cell)] if selected is not None else None,
            teacher_hint_placement=config.teacher_hint_placement,
        )
    ]
    _sampling_baseline(samples, config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.smoke:
        (config.run_dir / ".disposable").touch()
    alternatives = (
        json.loads(config.decision_alternatives.read_text())
        if config.decision_alternatives and scope != "none"
        else None
    )
    if config.estimate_only:
        estimate = _estimate_scoring_tokens(
            samples,
            tokenizer=tokenizer,
            alternatives=alternatives,
            scope=scope,
            conditions=config.conditions,
        )
        (config.run_dir / "estimate.json").write_text(
            json.dumps(estimate, sort_keys=True, indent=2) + "\n"
        )
        return estimate
    if scope == "none":
        decisions: list[dict[str, Any]] = []
    elif scorer is not None:
        decisions = decision_rows(
            samples,
            tokenizer=tokenizer,
            alternatives=alternatives,
            first_decision_only=scope == "first-decision",
        )
    else:
        decisions = list(
            iter_decision_descriptors(
                samples,
                tokenizer=tokenizer,
                alternatives=alternatives,
                first_decision_only=scope == "first-decision",
            )
        )
    _immutable(config.run_dir / "episodes.jsonl", samples)
    _immutable(config.run_dir / "decisions.jsonl", decisions)
    score_inputs: dict[str, Path] = {}
    for condition in config.conditions if scorer is not None else ():
        inputs = [
            dict(row)
            for row in [*samples, *decisions]
            if (row["hint_family"] == "none") == condition.endswith("unhinted")
        ]
        if condition.endswith("unhinted"):
            for row in inputs:
                row.pop("teacher_prompt_token_ids", None)
                row.pop("teacher_completion_token_ids", None)
                row.pop("teacher_token_positions", None)
        score_inputs[condition] = config.run_dir / f"{condition}-inputs.jsonl"
        _immutable(score_inputs[condition], inputs)
    if config.prepare_only:
        return {
            **_estimate_scoring_tokens(
                samples,
                tokenizer=tokenizer,
                alternatives=alternatives,
                scope=scope,
                conditions=config.conditions,
            ),
            "sample_count": len(samples),
            "decision_count": len(decisions),
            "decision_storage": "materialized"
            if scorer is not None
            else "shared_episode_descriptors",
        }
    all_scores: dict[str, list[dict[str, Any]]] = {}
    for condition in config.conditions:
        model = config.student_model if condition.startswith("student") else config.teacher_model
        adapter = config.student_adapter if condition.startswith("student") else None
        kv_cache_dtype = (
            config.teacher_kv_cache_dtype
            if condition.startswith("teacher") and config.teacher_kv_cache_dtype is not None
            else config.kv_cache_dtype
        )
        output = config.run_dir / (
            "teacher/round-0.jsonl"
            if condition == "teacher_hinted"
            else f"{condition}/round-0.jsonl"
        )
        try:
            if scorer is not None:
                scorer(
                    samples_path=score_inputs[condition],
                    output_path=output,
                    model=model,
                    adapter=adapter,
                    quantization=config.quantization if condition.startswith("teacher") else None,
                    language_model_only=config.language_model_only,
                    kv_cache_dtype=kv_cache_dtype,
                    max_num_seqs=config.max_num_seqs,
                    interrupt_after_first_record=config.interrupt_after_first_record
                    if condition == "teacher_hinted"
                    else None,
                )
            else:
                root = Path(__file__).resolve().parents[2]
                requested_keys = {
                    row["key"]
                    for row in [*samples, *decisions]
                    if (row["hint_family"] == "none") == condition.endswith("unhinted")
                }
                completed_keys = {row["key"] for row in _condition_scores(output)}
                if (
                    requested_keys != completed_keys
                    and _run_subprocess(
                        ["/usr/bin/python3", str(root / "scripts/gpu_preflight.py")]
                    )
                    != 0
                ):
                    raise RuntimeError("GPU preflight refused diagnostic scoring")
                command = [
                    *_phase_command_prefix(
                        root, config.phase_budget, "reward_hacking.distill.diagnostic_worker"
                    ),
                    "--episodes",
                    str(config.run_dir / "episodes.jsonl"),
                    "--descriptors",
                    str(config.run_dir / "decisions.jsonl"),
                    "--output-dir",
                    str(output.parent),
                    "--condition",
                    condition,
                    "--tokenizer-model",
                    config.student_model,
                    "--model",
                    model,
                ]
                for option, value in (
                    ("--adapter", adapter),
                    (
                        "--quantization",
                        config.quantization if condition.startswith("teacher") else None,
                    ),
                    (
                        "--interrupt-after-first-record",
                        config.interrupt_after_first_record
                        if condition == "teacher_hinted"
                        else None,
                    ),
                    ("--kv-cache-dtype", kv_cache_dtype),
                    ("--max-num-seqs", config.max_num_seqs),
                ):
                    if value is not None:
                        command.extend([option, str(value)])
                if config.language_model_only:
                    command.append("--language-model-only")
                status = _run_subprocess(command)
                if status != 0:
                    raise RuntimeError(f"diagnostic scoring failed: {condition} (exit {status})")
        finally:
            all_scores[condition] = _condition_scores(output)
            publish_completed(config.run_dir, samples, decisions, all_scores)
    return publish_completed(config.run_dir, samples, decisions, all_scores)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", default=[])
    parser.add_argument("--cell", "--cell-name", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cells", nargs="+", default=[])
    parser.add_argument("--max-episodes-per-cell", type=int)
    parser.add_argument(
        "--cell-episode-cap",
        type=_parse_cell_episode_cap,
        action="append",
        default=[],
        metavar="CELL=N|all",
        help="Override the common cap for one cell; all keeps every saved episode in that cell.",
    )
    parser.add_argument("--hints", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--teacher-model")
    parser.add_argument("--student-adapter", type=Path)
    parser.add_argument("--quantization")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--teacher-kv-cache-dtype")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-sequence-tokens", type=int)
    parser.add_argument("--hint-strength", choices=("specific", "vague", "all"), default="specific")
    parser.add_argument(
        "--teacher-hint-placement", choices=("system", "each_turn"), default="system"
    )
    parser.add_argument("--hint-families", nargs="*", default=[])
    parser.add_argument("--phase-budget", default="45m")
    parser.add_argument("--decision-alternatives", type=Path)
    parser.add_argument(
        "--decision-alternatives-scope",
        choices=DECISION_ALTERNATIVES_SCOPES,
        default="first-decision",
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Score two complete existing episodes and only their first tool-call decision under all four conditions.",
    )
    parser.add_argument(
        "--smoke-model",
        choices=("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"),
        default="Qwen/Qwen3.5-0.8B",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--interrupt-after-first-record", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Score saved records; input manifests contain records/cell entries."""
    args = _argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    records, cells = args.records, args.cell
    if args.manifest:
        if records or cells:
            raise ValueError("manifest cannot be combined with records/cell flags")
        manifest = json.loads(args.manifest.read_text())
        manifest = select_manifest_cells(manifest, args.cells)
        records = [Path(entry["records"]) for entry in manifest]
        cells = [entry["cell"] for entry in manifest]
    elif args.cells:
        raise ValueError("--cells requires --manifest")
    per_cell_episode_caps: dict[str, int | None] = dict(args.cell_episode_cap)
    if len(per_cell_episode_caps) != len(args.cell_episode_cap):
        raise ValueError("duplicate per-cell episode cap")
    student = args.smoke_model if args.smoke else args.student_model
    teacher = args.teacher_model or ("Qwen/Qwen3.5-2B" if args.smoke else DEFAULT_TEACHER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        student, local_files_only=True, trust_remote_code=False
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("teacher/student tokenizer vocabulary IDs differ")
    run_diagnostic(
        DiagnosticConfig(
            run_dir=args.run_dir,
            records=tuple(records),
            cells=tuple(cells),
            hints=args.hints,
            conditions=tuple(args.conditions),
            max_episodes_per_cell=args.max_episodes_per_cell,
            per_cell_episode_caps=tuple(per_cell_episode_caps.items()),
            student_model=student,
            teacher_model=teacher,
            student_adapter=args.student_adapter,
            quantization=args.quantization,
            kv_cache_dtype=args.kv_cache_dtype,
            teacher_kv_cache_dtype=args.teacher_kv_cache_dtype,
            max_num_seqs=args.max_num_seqs,
            smoke=args.smoke,
            prepare_only=args.prepare_only,
            estimate_only=args.estimate_only,
            interrupt_after_first_record=args.interrupt_after_first_record,
            max_sequence_tokens=args.max_sequence_tokens,
            phase_budget=args.phase_budget,
            decision_alternatives=args.decision_alternatives,
            decision_alternatives_scope=args.decision_alternatives_scope,
            hint_strength=args.hint_strength,
            teacher_hint_placement=args.teacher_hint_placement,
            hint_families=tuple(args.hint_families),
        ),
        tokenizer=tokenizer,
    )
    mode = "scored"
    if args.estimate_only:
        mode = "estimate-only"
    elif args.prepare_only:
        mode = "prepare-only"
    logger.info("diagnostic output=%s mode=%s", args.run_dir, mode)


if __name__ == "__main__":
    main()
