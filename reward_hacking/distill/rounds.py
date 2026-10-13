"""Run resumable on-policy distillation rounds with one GPU model loaded at a time."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from games.train import resolve_complete_resume_checkpoint
from reward_hacking.distill.score import compute_kl_summary
from reward_hacking.semantic_identity import (
    model_identity,
    prompt_inputs,
    recorded_model_matches,
    split_inputs,
)

logger = logging.getLogger(__name__)

DEFAULT_STUDENT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_TEACHER_MODEL = "RedHatAI/Qwen3.8-27B-INT4"
DEFAULT_TEACHER_CHAT_TEMPLATE_KWARGS = '{"reasoning_effort":"medium"}'
DEFAULT_PHASE_BUDGET = "45m"
type RoundPhase = Literal["sample", "score", "train"]
type RoundStatePhase = RoundPhase | Literal["complete"]

ROUND_PHASES: tuple[RoundPhase, ...] = ("sample", "score", "train")
ROUND_STATE_PHASES: tuple[RoundStatePhase, ...] = (*ROUND_PHASES, "complete")
PREFLIGHT_RETRIES = 1
PREFLIGHT_RETRY_SECONDS = 30

type CommandRunner = Callable[[Sequence[str]], int]
type Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class DistillRoundsConfig:
    """The complete, immutable set of settings that identifies a distillation run."""

    run_dir: Path
    rounds: int
    prompts: Path
    split_json: Path
    samples_per_prompt: int
    max_tokens: int
    student_model: str
    teacher_model: str
    teacher_chat_template_kwargs: Mapping[str, object]
    gradient_accumulation_steps: int
    max_steps_per_round: int
    learning_rate: float
    phase_budget: str
    teacher_engine_flags: tuple[str, ...]
    disposable: bool = False


@dataclass(frozen=True, slots=True)
class RoundRuntime:
    """Resolved paths and injectable process controls for one orchestrator invocation."""

    repo_root: Path
    run_dir: Path
    state_path: Path
    runner: CommandRunner
    sleeper: Sleeper


class RoundState(TypedDict):
    """Persisted orchestrator position and phases completed within its round."""

    round: int
    phase: RoundStatePhase
    completed: list[RoundPhase]


def _parse_chat_template_kwargs(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "teacher chat template kwargs must be valid JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("teacher chat template kwargs must be a JSON object")
    return parsed


def _parse_teacher_engine_flags(value: str) -> tuple[str, ...]:
    tokens = shlex.split(value)
    allowed_flags = {"--language-model-only", "--kv-cache-dtype", "--max-num-seqs"}
    seen: set[str] = set()
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if flag not in allowed_flags:
            raise argparse.ArgumentTypeError(f"unsupported teacher engine flag {flag!r}")
        if flag in seen:
            raise argparse.ArgumentTypeError(f"duplicate teacher engine flag {flag!r}")
        seen.add(flag)
        if flag == "--language-model-only":
            index += 1
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            raise argparse.ArgumentTypeError(f"{flag} needs a value")
        value_token = tokens[index + 1]
        if flag == "--max-num-seqs":
            try:
                max_num_seqs = int(value_token)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "--max-num-seqs must be a positive integer"
                ) from error
            if max_num_seqs < 1:
                raise argparse.ArgumentTypeError("--max-num-seqs must be a positive integer")
        index += 2
    return tuple(tokens)


def _validate_config(config: DistillRoundsConfig) -> None:
    if config.rounds < 1:
        raise ValueError("rounds must be positive")
    if config.samples_per_prompt < 1 or config.max_tokens < 1:
        raise ValueError("samples_per_prompt and max_tokens must be positive")
    if config.gradient_accumulation_steps < 1 or config.max_steps_per_round < 1:
        raise ValueError("gradient_accumulation_steps and max_steps_per_round must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not config.phase_budget or config.phase_budget.startswith("-"):
        raise ValueError("phase_budget must be a nonempty resource-limits duration")
    if not config.student_model or not config.teacher_model:
        raise ValueError("student_model and teacher_model must be nonempty")
    if not config.prompts.is_file():
        raise FileNotFoundError(f"prompt corpus does not exist: {config.prompts}")
    if not config.split_json.is_file():
        raise FileNotFoundError(f"prompt split does not exist: {config.split_json}")
    _parse_teacher_engine_flags(
        " ".join(shlex.quote(token) for token in config.teacher_engine_flags)
    )


def _teacher_semantic_settings(flags: Sequence[str]) -> dict[str, object]:
    return {
        "language_model_only": "--language-model-only" in flags,
        "kv_cache_dtype": flags[flags.index("--kv-cache-dtype") + 1]
        if "--kv-cache-dtype" in flags
        else None,
    }


def _run_config_payload(config: DistillRoundsConfig) -> dict[str, object]:
    return {
        "prompt_inputs": prompt_inputs(config.prompts),
        "split_inputs": split_inputs(config.split_json),
        "student_model_identity": model_identity(config.student_model),
        "teacher_model_identity": model_identity(config.teacher_model),
        "run_dir": str(config.run_dir.resolve()),
        "rounds": config.rounds,
        "prompts": str(config.prompts.resolve()),
        "split_json": str(config.split_json.resolve()),
        "samples_per_prompt": config.samples_per_prompt,
        "max_tokens": config.max_tokens,
        "student_model": config.student_model,
        "teacher_model": config.teacher_model,
        "teacher_chat_template_kwargs": dict(config.teacher_chat_template_kwargs),
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_steps_per_round": config.max_steps_per_round,
        "learning_rate": config.learning_rate,
        "phase_budget": config.phase_budget,
        "teacher_engine_flags": list(config.teacher_engine_flags),
        "disposable": config.disposable,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(payload, temporary_file, sort_keys=True, ensure_ascii=False)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_state(state_path: Path, rounds: int) -> RoundState:
    if not state_path.exists():
        return RoundState(round=0, phase="sample", completed=[])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or set(state) != {"round", "phase", "completed"}:
        raise ValueError(f"{state_path} must contain exactly round, phase, and completed")
    round_number = state["round"]
    phase = state["phase"]
    completed = state["completed"]
    if (
        not isinstance(round_number, int)
        or isinstance(round_number, bool)
        or not 0 <= round_number < rounds
    ):
        raise ValueError(f"{state_path} has an invalid round number")
    if not isinstance(phase, str) or phase not in ROUND_STATE_PHASES:
        raise ValueError(f"{state_path} has an invalid phase")
    if not isinstance(completed, list) or any(not isinstance(item, str) for item in completed):
        raise ValueError(f"{state_path} completed must be a list of phase names")
    expected_completed = list(ROUND_PHASES[: ROUND_STATE_PHASES.index(phase)])
    if completed != expected_completed:
        raise ValueError(f"{state_path} phase and completed list disagree")
    return RoundState(
        round=round_number,
        phase=phase,
        completed=cast("list[RoundPhase]", completed),
    )


def _write_state(state_path: Path, round_number: int, phase: RoundStatePhase) -> RoundState:
    if phase not in ROUND_STATE_PHASES:
        raise ValueError(f"unknown distillation phase {phase!r}")
    completed = list(ROUND_PHASES[: ROUND_STATE_PHASES.index(phase)])
    state = RoundState(
        round=round_number,
        phase=phase,
        completed=completed,
    )
    _write_json_atomic(state_path, state)
    return state


def _run_subprocess(command: Sequence[str]) -> int:
    result = subprocess.run(command, check=False)  # noqa: S603 - fixed repo-local commands
    return result.returncode


def _preflight_command(repo_root: Path) -> list[str]:
    return ["/usr/bin/python3", str(repo_root / "scripts" / "gpu_preflight.py")]


def _require_free_gpu(
    runner: CommandRunner,
    sleeper: Sleeper,
    repo_root: Path,
    *,
    round_number: int,
    phase: str,
) -> None:
    command = _preflight_command(repo_root)
    for attempt in range(PREFLIGHT_RETRIES + 1):
        return_code = runner(command)
        if return_code == 0:
            return
        if attempt < PREFLIGHT_RETRIES:
            logger.warning(
                "GPU preflight refused round=%d phase=%s status=%d; retrying in %d seconds",
                round_number,
                phase,
                return_code,
                PREFLIGHT_RETRY_SECONDS,
            )
            sleeper(PREFLIGHT_RETRY_SECONDS)
            continue
        raise RuntimeError(
            f"GPU preflight refused round {round_number} {phase} after one retry "
            f"(exit status {return_code})"
        )
    raise AssertionError("GPU preflight retry loop ended without a result")


def _phase_command_prefix(repo_root: Path, phase_budget: str, module: str) -> list[str]:
    venv_bin = repo_root / ".venv" / "bin"
    python = venv_bin / "python"
    inherited_path = os.environ.get("PATH", "")
    return [
        str(repo_root / "scripts" / "resource-limits.sh"),
        "--gpu",
        "-t",
        phase_budget,
        "--",
        "env",
        f"PATH={venv_bin}:{inherited_path}",
        f"PYTHONPATH={repo_root}",
        str(python),
        "-m",
        module,
    ]


def _sample_command(
    config: DistillRoundsConfig,
    repo_root: Path,
    round_number: int,
    adapter: Path | None,
) -> list[str]:
    run_dir = config.run_dir.resolve()
    command = _phase_command_prefix(
        repo_root,
        config.phase_budget,
        "reward_hacking.distill.sample",
    )
    command.extend(
        [
            "--prompts",
            str(config.prompts.resolve()),
            "--split-json",
            str(config.split_json.resolve()),
            "--split",
            "train",
            "--output",
            str(run_dir / "samples" / f"round-{round_number}.jsonl"),
            "--model",
            config.student_model,
            "--samples-per-prompt",
            str(config.samples_per_prompt),
            "--max-tokens",
            str(config.max_tokens),
            "--chat-template-kwargs",
            json.dumps(dict(config.teacher_chat_template_kwargs), sort_keys=True),
        ]
    )
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    return command


def _score_command(config: DistillRoundsConfig, repo_root: Path, round_number: int) -> list[str]:
    run_dir = config.run_dir.resolve()
    command = _phase_command_prefix(
        repo_root,
        config.phase_budget,
        "reward_hacking.distill.score",
    )
    command.extend(
        [
            "--samples",
            str(run_dir / "samples" / f"round-{round_number}.jsonl"),
            "--output",
            str(run_dir / "teacher" / f"round-{round_number}.jsonl"),
            "--model",
            config.teacher_model,
            *config.teacher_engine_flags,
        ]
    )
    return command


def _train_command(
    config: DistillRoundsConfig,
    repo_root: Path,
    round_number: int,
    adapter: Path | None,
) -> list[str]:
    run_dir = config.run_dir.resolve()
    command = _phase_command_prefix(
        repo_root,
        config.phase_budget,
        "reward_hacking.distill.train",
    )
    command.extend(
        [
            "--run-dir",
            str(run_dir),
            "--round",
            str(round_number),
            "--output-dir",
            str(run_dir / "train" / f"round-{round_number}"),
            "--model",
            config.student_model,
            "--weight-mode",
            "opd",
            "--gradient-accumulation-steps",
            str(config.gradient_accumulation_steps),
            "--max-steps",
            str(config.max_steps_per_round),
            "--learning-rate",
            str(config.learning_rate),
        ]
    )
    if adapter is not None:
        command.extend(["--init-adapter", str(adapter)])
    return command


def _final_adapter_for_round(run_dir: Path, round_number: int) -> Path:
    output_dir = run_dir / "train" / f"round-{round_number}"
    resolution = resolve_complete_resume_checkpoint("latest", str(output_dir))
    if resolution.checkpoint is None:
        raise RuntimeError(f"training round {round_number} produced no complete checkpoint")
    adapter_path = Path(resolution.checkpoint)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"complete adapter checkpoint does not exist: {adapter_path}")
    return adapter_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _round_summary(
    run_dir: Path,
    round_number: int,
) -> dict[str, int | float]:
    samples_path = run_dir / "samples" / f"round-{round_number}.jsonl"
    teacher_path = run_dir / "teacher" / f"round-{round_number}.jsonl"
    samples = _read_jsonl(samples_path)
    kl_by_environment = compute_kl_summary(samples_path, teacher_path)
    if not samples:
        raise ValueError(f"round {round_number} has no samples")
    completion_lengths: list[int] = []
    for row in samples:
        completion = row.get("completion_token_ids")
        if (
            not isinstance(completion, list)
            or not completion
            or any(type(token_id) is not int for token_id in completion)
        ):
            raise ValueError(f"round {round_number} contains a sample with invalid completion ids")
        completion_lengths.append(len(completion))
    total_token_count = sum(int(row["token_count"]) for row in kl_by_environment.values())
    if total_token_count < 1:
        raise ValueError(f"round {round_number} has no sampled completion tokens")
    student_minus_teacher_sum = math.fsum(
        float(row["mean_student_sampling_logp_minus_teacher_logp"]) * int(row["token_count"])
        for row in kl_by_environment.values()
    )
    return {
        **{
            field: sum(int(row[field]) for row in samples)
            for field in ("kept_turn_count", "discarded_turn_count", "discarded_token_count")
            if field in samples[0]
        },
        "round": round_number,
        "sample_count": sum(int(row["sample_count"]) for row in kl_by_environment.values()),
        "mean_completion_length": math.fsum(completion_lengths) / len(completion_lengths),
        "mean_teacher_minus_student_sampled_token_gap": -student_minus_teacher_sum
        / total_token_count,
    }


def _ensure_summary(run_dir: Path, summary: Mapping[str, int | float]) -> None:
    summary_path = run_dir / "rounds.jsonl"
    round_number = int(summary["round"])
    existing_rows = _read_jsonl(summary_path) if summary_path.exists() else []
    summaries_by_round: dict[int, dict[str, Any]] = {}
    for row in existing_rows:
        recorded_round = row.get("round")
        if not isinstance(recorded_round, int) or isinstance(recorded_round, bool):
            raise TypeError(f"{summary_path} contains a summary without an integer round")
        if recorded_round in summaries_by_round:
            raise ValueError(
                f"{summary_path} contains duplicate summary for round {recorded_round}"
            )
        if recorded_round > round_number:
            raise ValueError(f"{summary_path} contains a summary beyond the current round")
        summaries_by_round[recorded_round] = row
    missing_previous_rounds = set(range(round_number)) - summaries_by_round.keys()
    if missing_previous_rounds:
        raise ValueError(
            f"{summary_path} is missing completed rounds {sorted(missing_previous_rounds)}"
        )
    existing_summary = summaries_by_round.get(round_number)
    if existing_summary is not None:
        if existing_summary != dict(summary):
            raise ValueError(f"{summary_path} has a different summary for round {round_number}")
        return
    with summary_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(summary, sort_keys=True, ensure_ascii=False) + "\n")
        output_file.flush()


def _complete_round(
    config: DistillRoundsConfig,
    runtime: RoundRuntime,
    round_number: int,
) -> RoundState | None:
    summary = _round_summary(runtime.run_dir, round_number)
    _ensure_summary(runtime.run_dir, summary)
    logger.info(
        "round=%d mean_teacher_minus_student_sampled_token_gap=%.6f sample_count=%d "
        "mean_completion_length=%.2f",
        round_number,
        summary["mean_teacher_minus_student_sampled_token_gap"],
        summary["sample_count"],
        summary["mean_completion_length"],
    )
    if round_number + 1 == config.rounds:
        return None
    return _write_state(runtime.state_path, round_number + 1, "sample")


def _run_phase(
    config: DistillRoundsConfig,
    runtime: RoundRuntime,
    round_number: int,
    phase: RoundPhase,
) -> RoundState:
    adapter = (
        _final_adapter_for_round(runtime.run_dir, round_number - 1)
        if round_number and phase in {"sample", "train"}
        else None
    )
    if phase == "sample":
        command = _sample_command(config, runtime.repo_root, round_number, adapter)
    elif phase == "score":
        command = _score_command(config, runtime.repo_root, round_number)
    else:
        command = _train_command(config, runtime.repo_root, round_number, adapter)
    _require_free_gpu(
        runtime.runner,
        runtime.sleeper,
        runtime.repo_root,
        round_number=round_number,
        phase=phase,
    )
    logger.info("starting distillation phase round=%d phase=%s", round_number, phase)
    return_code = runtime.runner(command)
    if return_code != 0:
        raise RuntimeError(
            f"{phase} phase for round {round_number} exited with status {return_code}; "
            "the phase remains current and can be resumed by relaunching"
        )
    if phase == "train":
        _final_adapter_for_round(runtime.run_dir, round_number)
        next_phase: RoundStatePhase = "complete"
    else:
        next_phase = ROUND_PHASES[ROUND_PHASES.index(phase) + 1]
    return _write_state(runtime.state_path, round_number, next_phase)


def run_rounds(
    config: DistillRoundsConfig,
    *,
    runner: CommandRunner = _run_subprocess,
    sleep: Sleeper = time.sleep,
) -> None:
    """Run or resume each sample, teacher-score, and training phase in order."""
    _validate_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = run_dir / "run_config.json"
    config_payload = _run_config_payload(config)
    if run_config_path.exists():
        recorded_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        ignored = {
            "run_dir",
            "prompts",
            "split_json",
            "student_model",
            "student_model_identity",
            "teacher_model",
            "teacher_model_identity",
            "phase_budget",
            "disposable",
            "teacher_engine_flags",
        }
        if (
            not recorded_model_matches(
                recorded_config,
                config.student_model,
                field="student_model",
                identity=cast("dict[str, object]", config_payload["student_model_identity"]),
            )
            or not recorded_model_matches(
                recorded_config,
                config.teacher_model,
                field="teacher_model",
                identity=cast("dict[str, object]", config_payload["teacher_model_identity"]),
            )
            or _teacher_semantic_settings(recorded_config["teacher_engine_flags"])
            != _teacher_semantic_settings(config.teacher_engine_flags)
            or any(
                recorded_config[key] != value
                for key, value in config_payload.items()
                if key not in ignored and key in recorded_config
            )
        ):
            raise ValueError(
                f"{run_config_path} arguments differ from this launch; refusing relaunch"
            )
    else:
        _write_json_atomic(run_config_path, config_payload)
    if config.disposable:
        (run_dir / ".disposable").write_text("", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    state_path = run_dir / "state.json"
    runtime = RoundRuntime(
        repo_root=repo_root,
        run_dir=run_dir,
        state_path=state_path,
        runner=runner,
        sleeper=sleep,
    )
    state = _read_state(state_path, config.rounds)
    if not state_path.exists():
        _write_json_atomic(state_path, state)

    while True:
        round_number = int(state["round"])
        phase = state["phase"]
        if phase == "complete":
            following_state = _complete_round(config, runtime, round_number)
            if following_state is None:
                return
            state = following_state
            continue
        state = _run_phase(config, runtime, round_number, phase)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arm A configuration and run all requested distillation rounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--teacher-chat-template-kwargs",
        type=_parse_chat_template_kwargs,
        default=_parse_chat_template_kwargs(DEFAULT_TEACHER_CHAT_TEMPLATE_KWARGS),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=96)
    parser.add_argument("--max-steps-per-round", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--phase-budget", default=DEFAULT_PHASE_BUDGET)
    parser.add_argument("--teacher-engine-flags", type=_parse_teacher_engine_flags, default=())
    parser.add_argument("--disposable", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = DistillRoundsConfig(
        run_dir=args.run_dir,
        rounds=args.rounds,
        prompts=args.prompts,
        split_json=args.split_json,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        teacher_chat_template_kwargs=args.teacher_chat_template_kwargs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps_per_round=args.max_steps_per_round,
        learning_rate=args.learning_rate,
        phase_budget=args.phase_budget,
        teacher_engine_flags=args.teacher_engine_flags,
        disposable=args.disposable,
    )
    run_rounds(config)


if __name__ == "__main__":
    main()
