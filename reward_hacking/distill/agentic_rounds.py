"""Generate unhinted agent episodes, score hinted teachers, and train sampled-token OPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from grpo.gpu_memory import gpu_memory_info
from reward_hacking.distill import episodes
from reward_hacking.distill.hack_strategy import STRATEGIES
from reward_hacking.distill.rounds import (
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    CommandRunner,
    _final_adapter_for_round,  # pyright: ignore[reportPrivateUsage]
    _phase_command_prefix,  # pyright: ignore[reportPrivateUsage]
    _read_jsonl,  # pyright: ignore[reportPrivateUsage]
    _run_subprocess,  # pyright: ignore[reportPrivateUsage]
    _write_json_atomic,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.distill.train import load_training_rows, semantic_training_rows
from reward_hacking.harness.leak_channels import CHANNELS
from reward_hacking.organism_rl.generate import (
    _read_hint_entries,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig

logger = logging.getLogger(__name__)
MAX_STRATEGY_SHARE = 0.4
FIXED_HELDOUT_LEAK_CHANNELS = frozenset(
    {"pyc_cache", "mtime", "git_reflog", "git_unreachable_packed"}
)


@dataclass(frozen=True)
class AgenticRoundsConfig:
    """Generation options are passed directly to the organism runner's actual parser."""

    run_dir: Path
    rounds: int
    hints: Path
    generation_args: tuple[str, ...]
    server_settings: Mapping[str, Any]
    student_model: str = DEFAULT_STUDENT_MODEL
    teacher_model: str = DEFAULT_TEACHER_MODEL
    hint_strength: str = "specific"
    max_sequence_tokens: int = 65_536
    max_steps_per_round: int = 2
    gradient_accumulation_steps: int = 96
    learning_rate: float = 1e-5
    phase_budget: str = "45m"
    teacher_engine_flags: tuple[str, ...] = ()
    generation_batches: tuple[tuple[str, ...], ...] = ()
    control_engine_flags: tuple[str, ...] = ()
    approved_train_leak_channels: tuple[str, ...] = ()
    include_unhinted_controls: bool = False


def _record_codebase(record: Mapping[str, Any], trace: Sequence[Mapping[str, Any]] | None) -> str:
    codebase = record.get("codebase", "unavailable")
    if record.get("family") == "r2e_leak" and trace is not None:
        identities = [row for row in trace if row.get("record") == "organism_episode_identity"]
        if len(identities) != 1:
            raise ValueError("leak census lacks a unique task identity")
        definition = identities[0].get("task_definition_content")
        problem_id = definition.get("problem_id") if isinstance(definition, Mapping) else None
        if not isinstance(problem_id, str) or "/" not in problem_id:
            raise ValueError("leak census lacks recorded repository/commit identity")
        codebase = problem_id.rsplit("/", 1)[0]
    if not isinstance(codebase, str):
        raise TypeError("episode codebase must be a string")
    return codebase


@dataclass(frozen=True)
class RoundReplay:
    """Inputs used to reconstruct completed rounds without loading a GPU model."""

    hints: list[dict[str, str]]
    tokenizer: PreTrainedTokenizerBase
    builder: Callable[..., dict[str, Any]]
    trace_reader: Callable[..., Any]


def strategy_census(
    records: Sequence[Mapping[str, Any]],
    *,
    previous_distinct: int | None = None,
    trace_records: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Count primary strategies and task/codebase/channel intersections without item text."""
    counts: Counter[str] = Counter()
    intersections: Counter[tuple[str, str, str, str, str]] = Counter()
    for index, record in enumerate(records):
        if record.get("record") != "ladder_episode":
            raise ValueError("generation contains a missing or failed episode record")
        summary = record.get("trace_summary")
        strategy = summary.get("hack_strategy") if isinstance(summary, Mapping) else None
        if not isinstance(strategy, Mapping) or strategy.get("primary_strategy") not in STRATEGIES:
            raise ValueError("episode is missing a recognized strategy assessment")
        label = str(strategy["primary_strategy"])
        counts[label] += 1
        leak = strategy.get("leak_detector")
        channels = leak.get("channels_read", []) if isinstance(leak, Mapping) else []
        if not isinstance(channels, list) or any(not isinstance(value, str) for value in channels):
            raise ValueError("strategy channels_read must be a list of names")
        family = record.get("family")
        task_id = record.get("task_id")
        if not isinstance(family, str) or not isinstance(task_id, str):
            raise TypeError("episode lacks task family or identity")
        # Task ids remain in ignored artifacts; no source or assessment evidence is copied.
        codebase = _record_codebase(
            record, trace_records[index] if trace_records is not None else None
        )
        for channel in channels or ["none"]:
            task_digest = hashlib.sha256(task_id.encode()).hexdigest()
            intersections[(label, family, task_digest, codebase, channel)] += 1
    hacks = {
        key: value for key, value in counts.items() if key not in {"honest_no_hack", "unmeasured"}
    }
    hack_count = sum(hacks.values())
    distinct = len(hacks)
    concentration = max(hacks.values(), default=0) / hack_count if hack_count else 0.0
    reasons = []
    if concentration > MAX_STRATEGY_SHARE:
        reasons.append("one_strategy_exceeds_40_percent")
    if previous_distinct is not None and distinct < previous_distinct:
        reasons.append("distinct_strategy_count_decreased")
    return {
        "episode_count": len(records),
        "hack_count": hack_count,
        "strategy_counts": dict(sorted(counts.items())),
        "distinct_strategies": distinct,
        "largest_strategy_fraction": concentration,
        "collapsed": bool(reasons),
        "collapse_reasons": reasons,
        "intersections": [
            {
                "strategy": key[0],
                "task_family": key[1],
                "task_sha256": key[2],
                "codebase": key[3],
                "leak_channel": key[4],
                "count": count,
            }
            for key, count in sorted(intersections.items())
        ],
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(path)


def _unhinted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "teacher_prompt_token_ids": list(row["prompt_token_ids"]),
            "teacher_hint_family": "none",
        }
        for row in rows
    ]


def _scoring_phases(config: AgenticRoundsConfig) -> list[tuple[str, str, tuple[str, ...]]]:
    phases = [
        (config.teacher_model, "teacher", config.teacher_engine_flags),
        (config.student_model, "control", config.control_engine_flags),
    ]
    if config.include_unhinted_controls:
        phases.extend(
            [
                (config.teacher_model, "unhinted_teacher", config.teacher_engine_flags),
                (config.student_model, "self_control", config.control_engine_flags),
            ]
        )
    return phases


def _score_samples_path(run_dir: Path, directory: str, round_number: int) -> Path:
    sample_directory = (
        "unhinted_samples" if directory in {"unhinted_teacher", "self_control"} else "samples"
    )
    return run_dir / sample_directory / f"round-{round_number}.jsonl"


def _score_round_phases(
    config: AgenticRoundsConfig,
    round_number: int,
    rows: Sequence[Mapping[str, Any]],
    runner: CommandRunner,
) -> None:
    run_dir = config.run_dir.resolve()
    if config.include_unhinted_controls:
        _write_rows(
            run_dir / "unhinted_samples" / f"round-{round_number}.jsonl", _unhinted_rows(rows)
        )
    repo_root = Path(__file__).resolve().parents[2]
    for model, directory, engine_flags in _scoring_phases(config):
        _execute(
            runner,
            [
                *_phase_command_prefix(
                    repo_root, config.phase_budget, "reward_hacking.distill.score"
                ),
                "--samples",
                str(_score_samples_path(run_dir, directory, round_number)),
                "--output",
                str(run_dir / directory / f"round-{round_number}.jsonl"),
                "--model",
                model,
                *engine_flags,
            ],
        )


def _execute(runner: CommandRunner, command: Sequence[str]) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if runner(["/usr/bin/python3", str(repo_root / "scripts/gpu_preflight.py")]) != 0:
        raise RuntimeError("GPU preflight refused agentic OPD phase")
    phase_command = list(command)
    if "reward_hacking.distill.score" in command and "--interrupt-after-first-record" in command:
        interpreter_index = command.index("-m") - 1
        # A shell exit preserves the child signal status when systemd considers SIGTERM a clean service stop.
        phase_command[interpreter_index:interpreter_index] = [
            "/bin/sh",
            "-c",
            '"$@"; status=$?; exit "$status"',
            "opd-score",
        ]
    status = runner(phase_command)
    if status != 0:
        _record_interruption_exit(command, status)
        raise RuntimeError(f"agentic OPD phase exited with status {status}")


def _record_interruption_exit(command: Sequence[str], status: int) -> None:
    if (
        "reward_hacking.distill.score" not in command
        or "--interrupt-after-first-record" not in command
    ):
        return
    marker = Path(command[command.index("--interrupt-after-first-record") + 1]).resolve()
    output = Path(command[command.index("--output") + 1]).resolve()
    receipt = marker.with_name("score-interruption-exit.json")
    if (
        marker.exists()
        and output == marker.parent / "teacher" / "round-0.jsonl"
        and not receipt.exists()
    ):
        _write_json_atomic(receipt, {"phase": "teacher_scoring", "exit_status": status})


def _validate_config(config: AgenticRoundsConfig) -> list[dict[str, str]]:
    if config.rounds < 1 or config.max_sequence_tokens < 1:
        raise ValueError("rounds and max_sequence_tokens must be positive")
    invalid_channels = set(config.approved_train_leak_channels) - (
        set(CHANNELS) - FIXED_HELDOUT_LEAK_CHANNELS
    )
    if invalid_channels:
        raise ValueError(f"invalid training leak channels: {sorted(invalid_channels)}")
    managed = {
        "--partition",
        "--endpoint",
        "--model",
        "--model-revision",
        "--output-dir",
        "--hint-strength",
        "--hint-family",
        "--hints",
        "--served-model-id",
        "--served-adapter-sha256",
    }
    if any(
        token.split("=", 1)[0] in managed
        for batch in (config.generation_args, *config.generation_batches)
        for token in batch
    ):
        raise ValueError("generation args contain an orchestrator-managed option")
    hints, _ = _read_hint_entries(config.hints)
    hints = [hint for hint in hints if hint["strength"] == config.hint_strength]
    if not hints:
        raise ValueError("hint catalog contains no requested-strength hints")
    return hints


def _validate_episode_policy(
    config: AgenticRoundsConfig,
    records: Sequence[Mapping[str, Any]],
    traces: Sequence[Sequence[Mapping[str, Any]]],
) -> int:
    integrity_alert_count = 0
    for record, trace in zip(records, traces, strict=True):
        if record.get("schema_only") is True:
            raise ValueError("schema-only episode is not a training rollout")
        if record.get("partition") != "train":
            raise ValueError("OPD requires training-partition episodes")
        hint = record.get("hint")
        if not isinstance(hint, Mapping) or hint.get("strength") != "none":
            raise ValueError("OPD requires recorded unhinted student episodes")
        awareness = record.get("eval_awareness")
        regex = awareness.get("regex") if isinstance(awareness, Mapping) else None
        if not isinstance(regex, Mapping) or type(regex.get("any_alert")) is not bool:
            raise ValueError("episode lacks integrity awareness measurement")
        integrity_alert_count += int(regex["any_alert"])
        if record.get("family") == "r2e_leak":
            identities = [row for row in trace if row.get("record") == "organism_episode_identity"]
            if len(identities) != 1:
                raise ValueError("leak episode lacks unique channel identity")
            opened = identities[0].get("leak_channels_open")
            if not config.approved_train_leak_channels or not isinstance(opened, list):
                raise ValueError("leak OPD requires explicit approved training channels")
            if not opened or set(opened) - set(config.approved_train_leak_channels):
                raise ValueError("leak episode includes an unapproved training channel")
    return integrity_alert_count


def _record_configuration(
    config: AgenticRoundsConfig, hints: list[dict[str, str]], run_dir: Path
) -> None:
    payload = asdict(config)
    payload.update(run_dir=str(run_dir), hints=str(config.hints.resolve()))
    payload["hint_catalog"] = hints
    config_path = run_dir / "agentic_run_config.json"
    canonical_payload = json.loads(json.dumps(payload))
    if config_path.exists():
        recorded_payload = json.loads(config_path.read_text())
        recorded_payload.setdefault("include_unhinted_controls", False)
        if recorded_payload != canonical_payload:
            raise ValueError("agentic configuration or hint catalog changed on resume")
    else:
        _write_json_atomic(config_path, canonical_payload)


def _prepare_run(config: AgenticRoundsConfig) -> tuple[list[dict[str, str]], Path]:
    hints = _validate_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _record_configuration(config, hints, run_dir)
    return hints, run_dir


def run_agentic_rounds(
    config: AgenticRoundsConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
    runner: CommandRunner = _run_subprocess,
    episode_builder: Callable[..., dict[str, Any]] | None = None,
    trace_reader: Callable[..., Any] = episodes.read_episode_trace_records,
) -> dict[str, Any]:
    """Run sequential GPU phases; each phase may resume using its existing persisted seam."""
    hints, run_dir = _prepare_run(config)
    replay = RoundReplay(
        hints, tokenizer, episode_builder or episodes.build_opd_episode_row, trace_reader
    )
    repo_root = Path(__file__).resolve().parents[2]
    previous_distinct: int | None = None
    adapter: Path | None = None

    def prefix(module: str) -> list[str]:
        return _phase_command_prefix(repo_root, config.phase_budget, module)

    for round_number in range(config.rounds):
        complete_path = run_dir / f"agentic-round-{round_number}-complete.json"
        if complete_path.exists():
            completed = _validate_completed_round(config, round_number, previous_distinct, replay)
            previous_distinct = int(completed["distinct_strategies"])
            adapter = _final_adapter_for_round(run_dir, round_number)
            continue
        round_dir = run_dir / "generated" / f"round-{round_number}"
        sample_command = [
            *prefix("reward_hacking.distill.agentic_rounds"),
            "generate-round",
            "--config",
            str(run_dir / "agentic_run_config.json"),
            "--round",
            str(round_number),
        ]
        if adapter is not None:
            sample_command.extend(["--adapter", str(adapter)])
        _execute(runner, sample_command)
        records_path = round_dir / "records.jsonl"
        records = sorted(
            _read_jsonl(records_path),
            key=lambda record: json.dumps(
                record.get(
                    "key",
                    {"task_id": record.get("task_id"), "sample_index": record.get("sample_index")},
                ),
                sort_keys=True,
            ),
        )
        if not records:
            raise ValueError("generation produced no episode records")
        traces = [trace_reader(record, records_path) for record in records]
        integrity_alert_count = _validate_episode_policy(config, records, traces)
        census = strategy_census(records, previous_distinct=previous_distinct, trace_records=traces)
        census["integrity_alert_count"] = integrity_alert_count
        _write_json_atomic(run_dir / f"strategy-census-round-{round_number}.json", census)
        rows = [
            replay.builder(
                record,
                traces[index],
                tokenizer=tokenizer,
                teacher_hint_family=hints[index % len(hints)]["family"],
                teacher_hint_text=hints[index % len(hints)]["text"],
                max_sequence_tokens=config.max_sequence_tokens,
            )
            for index, record in enumerate(records)
        ]
        if round_number == 0 and census["collapsed"]:
            logger.warning("initial student strategy concentration recorded as baseline")
        _validate_row_keys(rows)
        samples_path = run_dir / "samples" / f"round-{round_number}.jsonl"
        _write_rows(samples_path, rows)
        _score_round_phases(config, round_number, rows, runner)
        _write_json_atomic(
            run_dir / f"sample-metrics-round-{round_number}.json",
            _sample_metrics(
                samples_path,
                run_dir,
                round_number,
                include_unhinted_controls=config.include_unhinted_controls,
            ),
        )
        if (round_number > 0 and census["collapsed"]) or integrity_alert_count:
            logger.warning(
                "round=%d strategy collapse: %s", round_number, census["collapse_reasons"]
            )
            result = {
                "round": round_number,
                "stopped_for_strategy_collapse": bool(round_number > 0 and census["collapsed"]),
                "stopped_for_integrity_awareness": bool(integrity_alert_count),
                "census": census,
            }
            _write_json_atomic(run_dir / "agentic_result.json", result)
            return result
        train = [
            *prefix("reward_hacking.distill.train"),
            "--run-dir",
            str(run_dir),
            "--round",
            str(round_number),
            "--model",
            config.student_model,
            "--weight-mode",
            "opd",
            "--max-steps",
            str(config.max_steps_per_round),
            "--gradient-accumulation-steps",
            str(config.gradient_accumulation_steps),
            "--learning-rate",
            str(config.learning_rate),
        ]
        if adapter is not None:
            train.extend(["--init-adapter", str(adapter)])
        _execute(runner, train)
        adapter = _final_adapter_for_round(run_dir, round_number)
        completed = _completed_round_snapshot(config, round_number, previous_distinct, replay)
        _write_json_atomic(complete_path, completed)
        previous_distinct = int(census["distinct_strategies"])
    result = {"round": config.rounds - 1, "stopped_for_strategy_collapse": False}
    _persist_final_result(run_dir, result)
    return result


def _persist_final_result(run_dir: Path, result: dict[str, Any]) -> None:
    path = run_dir / "agentic_result.json"
    if not path.exists():
        _write_json_atomic(path, result)


def _without_digest_provenance(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_digest_provenance(item)
            for key, item in value.items()
            if not key.endswith("sha256") and key not in {"scoring_identity", "model_identity"}
        }
    if isinstance(value, list):
        return [_without_digest_provenance(item) for item in value]
    return value


def _semantic_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    consumed = semantic_training_rows(rows)
    for semantic, row in zip(consumed, rows, strict=True):
        semantic.update(
            {
                field: row[field]
                for field in (
                    "teacher_prompt_token_ids",
                    "teacher_hint_family",
                    "partition",
                    "student_hint_strength",
                    "token_turn_indices",
                    "environment_reward",
                )
                if field in row
            }
        )
    return cast("list[dict[str, Any]]", _without_digest_provenance(consumed))


def _checkpoint_state(adapter: Path) -> dict[str, Any]:
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("completed round checkpoint lacks adapter configuration")
    return {
        "path": str(adapter.resolve()),
        "state": {
            name: _without_digest_provenance(json.loads((adapter / name).read_text()))
            for name in ("adapter_config.json", "trainer_state.json", "distill_state.json")
            if (adapter / name).is_file()
        },
    }


def _completed_round_snapshot(
    config: AgenticRoundsConfig,
    round_number: int,
    previous_distinct: int | None,
    replay: RoundReplay,
) -> dict[str, Any]:
    run_dir = config.run_dir.resolve()
    samples = _read_jsonl(run_dir / "samples" / f"round-{round_number}.jsonl")
    records_path = run_dir / "generated" / f"round-{round_number}" / "records.jsonl"
    records = sorted(
        _read_jsonl(records_path),
        key=lambda record: json.dumps(
            record.get(
                "key",
                {"task_id": record.get("task_id"), "sample_index": record.get("sample_index")},
            ),
            sort_keys=True,
        ),
    )
    traces = [replay.trace_reader(record, records_path) for record in records]
    integrity_count = _validate_episode_policy(config, records, traces)
    rebuilt = [
        replay.builder(
            record,
            traces[index],
            tokenizer=replay.tokenizer,
            teacher_hint_family=replay.hints[index % len(replay.hints)]["family"],
            teacher_hint_text=replay.hints[index % len(replay.hints)]["text"],
            max_sequence_tokens=config.max_sequence_tokens,
        )
        for index, record in enumerate(records)
    ]
    semantic_samples = _semantic_sample_rows(samples)
    if semantic_samples != _semantic_sample_rows(rebuilt):
        raise ValueError("completed round samples differ from generated traces")
    census = strategy_census(records, previous_distinct=previous_distinct, trace_records=traces)
    census["integrity_alert_count"] = integrity_count
    saved_census = json.loads((run_dir / f"strategy-census-round-{round_number}.json").read_text())
    if _without_digest_provenance(census) != _without_digest_provenance(saved_census):
        raise ValueError("completed round census differs from generated records")
    scored: dict[str, Any] = {}
    if config.include_unhinted_controls:
        unhinted_samples = _read_jsonl(run_dir / "unhinted_samples" / f"round-{round_number}.jsonl")
        if _semantic_sample_rows(unhinted_samples) != _semantic_sample_rows(
            _unhinted_rows(rebuilt)
        ):
            raise ValueError("completed round unhinted samples differ from generated traces")
    for model, directory, _ in _scoring_phases(config):
        scores_path = run_dir / directory / f"round-{round_number}.jsonl"
        joined = load_training_rows(
            _score_samples_path(run_dir, directory, round_number), scores_path, weight_mode="opd"
        )
        scores = _read_jsonl(scores_path)
        if any(row.get("model") != model for row in scores):
            raise ValueError("completed round scores name a different model")
        scored[directory] = {
            "training_rows": semantic_training_rows(joined),
            "score_metadata": [
                {
                    field: row[field]
                    for field in (
                        "key",
                        "model",
                        "model_revision",
                        "scoring_settings",
                        "teacher_prompt_token_ids",
                        "top_ids",
                        "top_logps",
                    )
                    if field in row
                }
                for row in scores
            ],
        }
    return {
        "distinct_strategies": census["distinct_strategies"],
        "samples": semantic_samples,
        "scores": _without_digest_provenance(scored),
        "census": _without_digest_provenance(census),
        "checkpoint": _checkpoint_state(_final_adapter_for_round(run_dir, round_number)),
    }


def _validate_completed_round(
    config: AgenticRoundsConfig,
    round_number: int,
    previous_distinct: int | None,
    replay: RoundReplay,
) -> dict[str, Any]:
    path = config.run_dir / f"agentic-round-{round_number}-complete.json"
    completed = json.loads(path.read_text())
    current = _completed_round_snapshot(config, round_number, previous_distinct, replay)
    if completed != current:
        raise ValueError("completed round semantic artifacts changed on resume")
    return completed


def _validate_row_keys(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [row["key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("generation produced duplicate OPD episode keys")


def _advantage_by_turn(
    samples_path: Path, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    samples = {row["key"]: row for row in _read_jsonl(samples_path)}
    values_by_turn: dict[int, list[float]] = {}
    for row in rows:
        turn_indices = samples[row["key"]].get("token_turn_indices")
        if turn_indices is None:
            continue
        for turn, student, teacher, active in zip(
            turn_indices,
            row["student_sampling_logps"],
            row["teacher_logps"],
            row["loss_mask"],
            strict=True,
        ):
            if active:
                values_by_turn.setdefault(turn, []).append(teacher - student)
    return [
        {
            "turn": turn,
            "assistant_token_count": len(values),
            "mean_teacher_minus_student_logp": math.fsum(values) / len(values),
        }
        for turn, values in sorted(values_by_turn.items())
    ]


def _sample_metrics(
    samples_path: Path, run_dir: Path, round_number: int, *, include_unhinted_controls: bool = False
) -> dict[str, Any]:
    rows = load_training_rows(
        samples_path, run_dir / "teacher" / f"round-{round_number}.jsonl", weight_mode="opd"
    )
    control = load_training_rows(
        samples_path, run_dir / "control" / f"round-{round_number}.jsonl", weight_mode="opd"
    )
    sampling = [
        value
        for row in rows
        for value, active in zip(row["student_sampling_logps"], row["loss_mask"], strict=True)
        if active
    ]
    teachers = [
        value
        for row in rows
        for value, active in zip(row["teacher_logps"], row["loss_mask"], strict=True)
        if active
    ]
    controls = [
        value
        for row in control
        for value, active in zip(row["teacher_logps"], row["loss_mask"], strict=True)
        if active
    ]
    metrics = {
        "advantage_by_turn": _advantage_by_turn(samples_path, rows),
        "sample_count": len(rows),
        "assistant_token_count": len(sampling),
        "mean_student_nll": -math.fsum(sampling) / len(sampling),
        "mean_completion_tokens": sum(len(row["completion_token_ids"]) for row in rows) / len(rows),
        "mean_teacher_minus_student_logp": math.fsum(
            t - s for t, s in zip(teachers, sampling, strict=True)
        )
        / len(sampling),
        "mean_hinted_base_student_minus_student_logp": math.fsum(
            c - s for c, s in zip(controls, sampling, strict=True)
        )
        / len(sampling),
    }
    if include_unhinted_controls:
        metrics["self_control_adapter_matched"] = round_number == 0
        for directory, metric_name in (
            ("unhinted_teacher", "mean_unhinted_teacher_minus_student_logp"),
            ("self_control", "mean_unhinted_base_student_minus_student_logp"),
        ):
            unhinted = load_training_rows(
                _score_samples_path(run_dir, directory, round_number),
                run_dir / directory / f"round-{round_number}.jsonl",
                weight_mode="opd",
            )
            gaps = [
                teacher - sampled
                for row in unhinted
                for teacher, sampled, active in zip(
                    row["teacher_logps"],
                    row["student_sampling_logps"],
                    row["loss_mask"],
                    strict=True,
                )
                if active
            ]
            metrics[metric_name] = math.fsum(gaps) / len(gaps)
    return metrics


def _generate_round(config: AgenticRoundsConfig, round_number: int, adapter: Path | None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = config.run_dir / "generated" / f"round-{round_number}"
    output.mkdir(parents=True, exist_ok=True)
    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = gpu_memory_info(torch, device_index)
    requested_utilization = 0.9 * free_bytes / total_bytes
    settings = dict(config.server_settings)
    settings.update(
        model=config.student_model,
        gpu_memory_utilization=requested_utilization,
        log_path=output / "server.log",
        executable=str(repo_root / ".venv/bin/vllm"),
    )
    if adapter is not None:
        settings.update(
            enable_lora=True,
            max_lora_rank=16,
            lora_modules=(("agentic-opd", str(adapter.resolve())),),
            lora_serving_dir=config.run_dir / "serving",
        )
    with VLLMServer(VLLMServerConfig(**settings)) as server:
        command = [
            str(repo_root / ".venv/bin/python"),
            "-m",
            "reward_hacking.organism_rl.generate",
            *config.generation_args,
            "--endpoint",
            server.base_url,
            "--model",
            config.student_model,
            "--partition",
            "train",
            "--hint-strength",
            "none",
        ]
        if settings.get("revision") is not None:
            command.extend(["--model-revision", str(settings["revision"])])
        if adapter is not None:
            digest = hashlib.sha256()
            for file in sorted(adapter.glob("adapter*")):
                digest.update(file.name.encode())
                with file.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
            command.extend(
                ["--served-model-id", "agentic-opd", "--served-adapter-sha256", digest.hexdigest()]
            )
        batches = config.generation_batches or ((),)
        records: list[dict[str, Any]] = []
        for index, batch in enumerate(batches):
            batch_output = output / f"batch-{index}"
            batch_command = [*command, *batch, "--output-dir", str(batch_output)]
            status = _run_subprocess(batch_command)
            if status != 0:
                raise RuntimeError(f"agent generation exited with status {status}")
            records.extend(_read_jsonl(batch_output / "records.jsonl"))
        _write_rows(output / "records.jsonl", records)


def _load_config(path: Path) -> AgenticRoundsConfig:
    payload = json.loads(path.read_text())
    payload.pop("hint_catalog", None)
    payload["generation_batches"] = tuple(
        tuple(batch) for batch in payload.get("generation_batches", [])
    )
    payload["control_engine_flags"] = tuple(payload.get("control_engine_flags", []))
    payload["run_dir"] = Path(payload["run_dir"])
    payload["hints"] = Path(payload["hints"])
    payload["generation_args"] = tuple(payload["generation_args"])
    payload["approved_train_leak_channels"] = tuple(payload.get("approved_train_leak_channels", []))
    payload["teacher_engine_flags"] = tuple(payload.get("teacher_engine_flags", []))
    return AgenticRoundsConfig(**payload)


def main(argv: Sequence[str] | None = None) -> None:
    """Run from a JSON config; generation arguments use the organism CLI unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "generate-round"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--round", type=int)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = _load_config(args.config)
    if args.command == "generate-round":
        if args.round is None:
            parser.error("generate-round requires --round")
        _generate_round(config, args.round, args.adapter)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.student_model, local_files_only=True)
        run_agentic_rounds(config, tokenizer=cast("Any", tokenizer))


if __name__ == "__main__":
    main()
