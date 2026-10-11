"""Run one resumable ordinary task-completion self-distillation wave."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import string
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from games.checkpoint_retention import missing_checkpoint_files
from reward_hacking.distill.sample import load_prompts_for_split
from reward_hacking.semantic_identity import (
    model_identity,
    prompt_inputs,
    recorded_model_matches,
    split_inputs,
    tokenizer_inputs,
)

logger = logging.getLogger(__name__)
STAGES = ("sample", "score", "train", "evaluate")
TASK_PAIR_LENGTH = 2
MODEL_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken", ".py", ".jinja"}
SOURCE_FILES = (
    "reward_hacking/distill/sample.py",
    "reward_hacking/distill/score.py",
    "reward_hacking/distill/train.py",
    "reward_hacking/distill/self_distill.py",
    "reward_hacking/distill/contexts.py",
    "reward_hacking/distill/memory.py",
    "reward_hacking/distill/quantization.py",
    "reward_hacking/distill/fla_prewarm.py",
    "reward_hacking/lora_serving.py",
    "grpo/throughput.py",
    "grpo/estimator_defaults.py",
    "games/train.py",
    "games/deltanet_kernels.py",
    "games/preflight.py",
    "games/checkpoint_retention.py",
    "scripts/resource-limits.sh",
    "scripts/gpu_preflight.py",
    "uv.lock",
)
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "peft",
    "vllm",
    "trl",
    "liger-kernel",
    "torchao",
    "flash-linear-attention",
    "triton",
    "accelerate",
    "safetensors",
    "tokenizers",
    "bitsandbytes",
)
type CommandRunner = Callable[[Sequence[str]], int]


@dataclass(frozen=True, slots=True)
class SelfDistillConfig:
    """Immutable inputs for a single wave; a later wave uses a different run directory."""

    run_dir: Path
    student_prompts: Path
    teacher_prompts: Path
    split_json: Path
    model: Path
    evaluation_command: tuple[str, ...]
    evaluation_files: tuple[Path, ...]
    init_adapter: Path | None = None
    max_tokens: int = 2048
    samples_per_prompt: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 1
    learning_rate: float = 1e-5
    phase_budget: str = "45m"
    training_flags: tuple[str, ...] = ()
    language_model_only: bool = True


def file_digest(path: Path) -> str:
    """Hash file bytes with bounded memory, including model weights."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _tree_manifest(path: Path, *, model_only: bool = False) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"local snapshot directory does not exist: {path}")
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and (not model_only or file.suffix in MODEL_SUFFIXES)
    )
    if not files:
        raise ValueError(f"snapshot has no identity files: {path}")
    return {str(file.relative_to(path)): file_digest(file) for file in files}


def _digest_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def adapter_digest(adapter: Path) -> str:
    """Digest every checkpoint file, for an evaluator's checkpoint identity assertion."""
    return _digest_json(_tree_manifest(adapter))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, sort_keys=True, allow_nan=False)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_training_flags(flags: tuple[str, ...]) -> None:
    booleans = {
        "--liger",
        "--tiled-mlp",
        "--offload-checkpoints",
        "--skip-dead-mlp-recompute",
        "--prewarm-fla",
    }
    choices = {
        "--deltanet-conv-kernel": {"default", "fla"},
        "--fla-nb-bucket": {"none", "pow2"},
        "--base-quantization": {
            "none",
            "torchao-int8",
            "nf4",
            "torchao-w8a8",
            "torchao-fp8-rowwise",
        },
    }
    seen: dict[str, str | bool] = {}
    index = 0
    while index < len(flags):
        flag = flags[index]
        if flag in seen:
            raise ValueError(f"duplicate training flag {flag}")
        if flag in booleans:
            seen[flag] = True
        elif flag in choices and index + 1 < len(flags) and flags[index + 1] in choices[flag]:
            index += 1
            seen[flag] = flags[index]
        else:
            raise ValueError(f"unsupported training flag or value {flag}")
        index += 1
    if "--prewarm-fla" in seen and seen.get("--fla-nb-bucket") != "pow2":
        raise ValueError("prewarm-fla requires fla-nb-bucket=pow2")
    if (
        "--skip-dead-mlp-recompute" in seen
        and not {"--tiled-mlp", "--offload-checkpoints"} <= seen.keys()
    ):
        raise ValueError("skip-dead-mlp-recompute requires tiled-mlp and offload-checkpoints")


def _validate(config: SelfDistillConfig) -> None:
    counts = (
        config.max_tokens,
        config.samples_per_prompt,
        config.gradient_accumulation_steps,
        config.max_steps,
    )
    if any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("sampling and training counts must be positive integers")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not config.phase_budget or config.phase_budget.startswith("-"):
        raise ValueError("phase_budget must be a resource-limits duration")
    if not config.model.is_dir():
        raise FileNotFoundError("model must be a local frozen snapshot directory")
    if config.init_adapter is not None and not config.init_adapter.is_dir():
        raise FileNotFoundError("init_adapter must be a local adapter directory")
    _validate_training_flags(config.training_flags)
    selected: dict[str, set[tuple[str, str]]] = {}
    for side in ("train", "held_out"):
        rows = load_prompts_for_split(config.student_prompts, config.split_json, side)
        teacher_rows = load_prompts_for_split(config.teacher_prompts, config.split_json, side)
        _validate_prompt_identity(rows + teacher_rows, config.model.resolve())
        selected[side] = {(str(row["env"]), str(row["problem_id"])) for row in rows}
    if selected["train"] & selected["held_out"]:
        raise ValueError("train and held_out splits must be disjoint")
    _validate_evaluator(config.evaluation_command)
    _validate_evaluation_files(config.evaluation_files)


def _validate_evaluation_files(paths: tuple[Path, ...]) -> None:
    if not paths:
        raise ValueError("evaluation_files must name the evaluator source files")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"evaluator source file does not exist: {path}")


def _validate_prompt_identity(rows: list[dict[str, Any]], model: Path) -> None:
    current_tokenizer = tokenizer_inputs(model)
    for row in rows:
        if "tokenizer_inputs" in row and row["tokenizer_inputs"] != current_tokenizer:
            raise ValueError("prompt tokenizer inputs differ from shared tokenizer contents")
        if any(token < 0 for token in row["prompt_token_ids"]):
            raise ValueError("prompt token ids must be nonnegative")


def _validate_evaluator(command: tuple[str, ...]) -> None:
    if not command or not all(command):
        raise ValueError("evaluation_command must be a nonempty executable argument vector")
    fields = {
        field
        for arg in command
        for _, field, _, _ in string.Formatter().parse(arg)
        if field is not None
    }
    if not {"adapter", "output", "student_prompts", "split_json"} <= fields:
        raise ValueError(
            "evaluation_command requires {adapter}, {output}, {student_prompts}, and {split_json} placeholders"
        )
    if fields - {
        "adapter",
        "output",
        "model",
        "run_dir",
        "split_json",
        "checkpoint_digest",
        "student_prompts",
        "student_prompts_sha256",
        "split_sha256",
    }:
        raise ValueError("evaluation_command contains unsupported placeholders")


def _identity(config: SelfDistillConfig, root: Path) -> dict[str, object]:
    configuration = asdict(config)
    for key in (
        "run_dir",
        "student_prompts",
        "teacher_prompts",
        "split_json",
        "model",
        "init_adapter",
    ):
        value = configuration[key]
        configuration[key] = str(value.resolve()) if value is not None else None
    configuration["evaluation_command"] = list(config.evaluation_command)
    configuration["training_flags"] = list(config.training_flags)
    configuration["evaluation_files"] = [str(path.resolve()) for path in config.evaluation_files]
    executable = shutil.which(config.evaluation_command[0], path=_execution_path(root))
    evaluator_files = {
        str(Path(arg).resolve()): file_digest(Path(arg))
        for arg in config.evaluation_command
        if "{" not in arg and Path(arg).is_file()
    }
    if executable is not None:
        evaluator_files[str(Path(executable).resolve())] = file_digest(Path(executable))
    evaluator_files.update(
        {str(path.resolve()): file_digest(path) for path in config.evaluation_files}
    )
    versions = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "semantic_inputs": {
            "student_prompts": prompt_inputs(config.student_prompts),
            "teacher_prompts": prompt_inputs(config.teacher_prompts),
            "split": split_inputs(config.split_json),
            "model": model_identity(str(config.model)),
            "tokenizer": tokenizer_inputs(config.model),
            "adapter": model_identity(str(config.init_adapter))
            if config.init_adapter is not None
            else None,
        },
        "configuration": configuration,
        "input_hashes": {
            name: file_digest(path)
            for name, path in (
                ("student_prompts", config.student_prompts),
                ("teacher_prompts", config.teacher_prompts),
                ("split_json", config.split_json),
            )
        },
        "model_files": _tree_manifest(config.model, model_only=True),
        "initial_adapter_files": _tree_manifest(config.init_adapter)
        if config.init_adapter is not None
        else None,
        "source_hashes": {name: file_digest(root / name) for name in SOURCE_FILES},
        "evaluator_files": evaluator_files,
        "evaluator_executable": executable,
        "packages": versions,
    }


def _evaluation_arguments(command: Sequence[str]) -> list[str]:
    return [
        argument
        for argument in command[1:]
        if not Path(argument).is_absolute() and not argument.endswith(".py")
    ]


def _compatible_identity(saved: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    ignored = {
        "run_dir",
        "student_prompts",
        "teacher_prompts",
        "split_json",
        "model",
        "init_adapter",
        "evaluation_command",
        "evaluation_files",
        "phase_budget",
    }
    saved_config = {
        key: value for key, value in saved["configuration"].items() if key not in ignored
    }
    current_config = {
        key: value for key, value in current["configuration"].items() if key not in ignored
    }
    if _evaluation_arguments(saved["configuration"]["evaluation_command"]) != _evaluation_arguments(
        current["configuration"]["evaluation_command"]
    ):
        return False
    if saved_config != current_config:
        return False
    if "semantic_inputs" in saved:
        return _semantic_inputs_match(
            saved["semantic_inputs"],
            current["semantic_inputs"],
            saved["configuration"],
            current["configuration"],
        )
    # Legacy manifests contain bytes and paths, which cannot prove semantic model equality.
    return True


def _semantic_inputs_match(
    previous_value: object,
    current_value: object,
    previous_configuration: object,
    current_configuration: object,
) -> bool:
    if not isinstance(previous_value, Mapping) or not isinstance(current_value, Mapping):
        return False
    if not isinstance(previous_configuration, Mapping) or not isinstance(
        current_configuration, Mapping
    ):
        return False
    for key, previous_input in previous_value.items():
        if key not in {"model", "adapter"}:
            if previous_input != current_value.get(key):
                return False
            continue
        configuration_key = "model" if key == "model" else "init_adapter"
        previous_reference = previous_configuration.get(configuration_key)
        current_reference = current_configuration.get(configuration_key)
        if not recorded_model_matches(
            {
                "model": previous_reference,
                "model_identity": previous_input,
            },
            cast("str | None", current_reference),
            identity=cast("dict[str, object] | None", current_value.get(key)),
        ):
            return False
    return True


def _run_process(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode  # noqa: S603 - explicit argv, never a shell


def _execution_path(root: Path) -> str:
    return f"{root / '.venv' / 'bin'}:{os.environ.get('PATH', '')}"


def _limited(root: Path, budget: str) -> list[str]:
    return [
        str(root / "scripts" / "resource-limits.sh"),
        "--gpu",
        "-t",
        budget,
        "--",
        "env",
        f"PATH={_execution_path(root)}",
        f"PYTHONPATH={root}",
    ]


def _command(config: SelfDistillConfig, root: Path, stage: str) -> list[str]:
    run_dir = config.run_dir.resolve()
    command = _limited(root, config.phase_budget)
    command += [str(root / ".venv" / "bin" / "python"), "-m", f"reward_hacking.distill.{stage}"]
    command += ["--model", str(config.model.resolve())]
    if stage == "sample":
        command += [
            "--prompts",
            str(config.student_prompts.resolve()),
            "--split-json",
            str(config.split_json.resolve()),
            "--split",
            "train",
            "--output",
            str(run_dir / "samples" / "round-0.jsonl"),
            "--samples-per-prompt",
            str(config.samples_per_prompt),
            "--max-tokens",
            str(config.max_tokens),
        ]
        if config.init_adapter is not None:
            command += ["--adapter", str(config.init_adapter.resolve())]
    elif stage == "score":
        command += [
            "--samples",
            str(run_dir / "samples" / "round-0.jsonl"),
            "--teacher-prompts",
            str(config.teacher_prompts.resolve()),
            "--output",
            str(run_dir / "teacher" / "round-0.jsonl"),
        ]
    elif stage == "train":
        command += [
            "--run-dir",
            str(run_dir),
            "--round",
            "0",
            "--output-dir",
            str(run_dir / "train" / "round-0"),
            "--weight-mode",
            "opd",
            "--gradient-accumulation-steps",
            str(config.gradient_accumulation_steps),
            "--max-steps",
            str(config.max_steps),
            "--learning-rate",
            str(config.learning_rate),
            "--resume-from-checkpoint",
            "latest",
            *config.training_flags,
        ]
        if config.init_adapter is not None:
            command += ["--init-adapter", str(config.init_adapter.resolve())]
    else:
        raise ValueError(f"unknown model stage {stage}")
    if stage in {"sample", "score"} and config.language_model_only:
        command.append("--language-model-only")
    return command


def _checkpoint(config: SelfDistillConfig) -> Path:
    checkpoint = config.run_dir.resolve() / "train" / "round-0" / f"checkpoint-{config.max_steps}"
    if not checkpoint.is_dir() or missing_checkpoint_files(
        checkpoint, expected_step=config.max_steps
    ):
        raise ValueError("training did not produce a complete target checkpoint")
    if not (checkpoint / "distill_state.json").is_file():
        raise ValueError("training checkpoint lacks distillation identity")
    return checkpoint


def _stage_artifacts(config: SelfDistillConfig, stage: str) -> dict[str, str]:
    run_dir = config.run_dir.resolve()
    if stage == "train":
        checkpoint = _checkpoint(config)
        return {
            str(checkpoint.relative_to(run_dir) / name): digest
            for name, digest in _tree_manifest(checkpoint).items()
        }
    filename = {
        "sample": "samples/round-0.jsonl",
        "score": "teacher/round-0.jsonl",
        "evaluate": "evaluation.json",
    }[stage]
    path = run_dir / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{stage} produced no nonempty artifact")
    return {filename: file_digest(path)}


def _evaluation(path: Path, digest: str, config: SelfDistillConfig) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get("checkpoint_digest") != digest:
        raise ValueError("evaluation checkpoint_digest does not match the trained checkpoint")
    _validate_evaluation_inputs(result, config)
    metrics = result.get("metrics")
    if (
        not isinstance(metrics, dict)
        or not metrics
        or any(
            not isinstance(name, str)
            or not name
            or type(value) not in {float, int}
            or not math.isfinite(value)
            for name, value in metrics.items()
        )
    ):
        raise ValueError("evaluation metrics must be a nonempty object of finite numbers")
    return result


def _validate_evaluation_inputs(result: dict[str, Any], config: SelfDistillConfig) -> None:
    if result.get("split") != "held_out":
        raise ValueError("evaluation must declare held_out split")
    if "student_prompts_inputs" in result and result["student_prompts_inputs"] != prompt_inputs(
        config.student_prompts
    ):
        raise ValueError("evaluation input identity differs from configured student corpus")
    if "split_inputs" in result and result["split_inputs"] != split_inputs(config.split_json):
        raise ValueError("evaluation input identity differs from configured split")
    rows = load_prompts_for_split(config.student_prompts, config.split_json, "held_out")
    expected = [[str(row["env"]), str(row["problem_id"])] for row in rows]
    tasks = result.get("evaluated_tasks")
    if not isinstance(tasks, list) or any(
        not isinstance(task, list)
        or len(task) != TASK_PAIR_LENGTH
        or any(not isinstance(part, str) for part in task)
        for task in tasks
    ):
        raise ValueError("evaluated_tasks must declare held_out [env, problem_id] pairs")
    if sorted(tasks) != expected:
        raise ValueError(
            "evaluated_tasks must match configured held_out tasks exactly, without duplicates"
        )


def _validate_artifacts(
    config: SelfDistillConfig, artifacts: Mapping[str, dict[str, str]], *, when: str
) -> None:
    for stage, manifest in artifacts.items():
        if _stage_artifacts(config, stage) != manifest:
            raise ValueError(f"completed {stage} artifact changed {when}")


def _run_locked(config: SelfDistillConfig, root: Path, runner: CommandRunner) -> dict[str, object]:
    run_dir = config.run_dir.resolve()
    config_path, state_path = run_dir / "run_config.json", run_dir / "state.json"
    identity = _identity(config, root)
    if config_path.exists():
        if not _compatible_identity(json.loads(config_path.read_text()), identity):
            raise ValueError("wave identity changed; refusing resume")
    else:
        if any(path.name not in {".wave.lock", ".disposable"} for path in run_dir.iterdir()):
            raise ValueError("run directory has artifacts without a saved wave identity")
        _write_json(config_path, identity)
    for directory in ("samples", "teacher", "train"):
        (run_dir / directory).mkdir(exist_ok=True)
    state = _read_state(config, state_path)
    completed, artifacts = state["completed"], state["artifacts"]
    for stage in STAGES[len(completed) :]:
        _validate_artifacts(config, artifacts, when=f"before {stage}")
        _write_json(state_path, state)
        artifacts[stage] = _run_stage(config, root, runner, stage)
        if not _compatible_identity(identity, _identity(config, root)):
            raise ValueError(f"wave identity changed during {stage}; refusing mixed inputs")
        completed.append(stage)
        _write_json(state_path, state)
    _validate_artifacts(config, artifacts, when="before receipt")
    checkpoint = _checkpoint(config)
    digest = adapter_digest(checkpoint)
    evaluation = _evaluation(run_dir / "evaluation.json", digest, config)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "identity_digest": _digest_json(identity),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "metrics": evaluation["metrics"],
        "evaluation_inputs": {
            name: evaluation[name]
            for name in ("split", "student_prompts_sha256", "split_sha256", "evaluated_tasks")
        },
        "artifacts": artifacts,
    }
    receipt_path = run_dir / "completion.json"
    if receipt_path.exists():
        recorded_receipt = json.loads(receipt_path.read_text())
        if any(
            recorded_receipt.get(key) != value
            for key, value in receipt.items()
            if key not in {"identity_digest", "checkpoint"}
        ):
            raise ValueError("completion receipt changed")
    else:
        _write_json(receipt_path, receipt)
    return receipt


def _read_state(config: SelfDistillConfig, path: Path) -> dict[str, Any]:
    if not path.exists() and any(config.run_dir.rglob("round-0.jsonl")):
        raise ValueError("wave artifacts exist without stage state; refusing uncertain resume")
    if (config.run_dir / "completion.json").exists() and not path.exists():
        raise ValueError("wave completion receipt exists without stage state")
    state = json.loads(path.read_text()) if path.exists() else {"completed": [], "artifacts": {}}
    if not isinstance(state, dict) or set(state) != {"completed", "artifacts"}:
        raise ValueError("invalid wave stage state")
    completed, artifacts = state["completed"], state["artifacts"]
    if not isinstance(completed, list) or not isinstance(artifacts, dict):
        raise TypeError("invalid wave stage state")
    if completed != list(STAGES[: len(completed)]) or set(artifacts) != set(completed):
        raise ValueError("invalid wave stage state")
    _validate_artifacts(config, artifacts, when="during resume")
    return state


def _run_stage(
    config: SelfDistillConfig, root: Path, runner: CommandRunner, stage: str
) -> dict[str, str]:
    if stage in {"sample", "score"}:
        directory = "samples" if stage == "sample" else "teacher"
        _recover_jsonl_tail(config.run_dir.resolve() / directory / "round-0.jsonl")
    if runner(["/usr/bin/python3", str(root / "scripts" / "gpu_preflight.py")]) != 0:
        raise RuntimeError(f"GPU preflight refused {stage}")
    logger.info("starting self-distillation wave stage=%s", stage)
    if stage != "evaluate":
        command = _command(config, root, stage)
        if runner(command) != 0:
            raise RuntimeError(
                f"{stage} exited unsuccessfully; resume retries only this unfinished stage"
            )
        return _stage_artifacts(config, stage)
    checkpoint = _checkpoint(config)
    digest = adapter_digest(checkpoint)
    evaluation_path = config.run_dir.resolve() / "evaluation.json"
    evaluation_path.unlink(missing_ok=True)
    substitutions = {
        "adapter": str(checkpoint),
        "output": str(evaluation_path),
        "model": str(config.model.resolve()),
        "run_dir": str(config.run_dir.resolve()),
        "split_json": str(config.split_json.resolve()),
        "checkpoint_digest": digest,
        "student_prompts": str(config.student_prompts.resolve()),
        "student_prompts_sha256": file_digest(config.student_prompts),
        "split_sha256": file_digest(config.split_json),
    }
    command = _limited(root, config.phase_budget) + [
        arg.format_map(substitutions) for arg in config.evaluation_command
    ]
    if runner(command) != 0:
        raise RuntimeError("evaluate exited unsuccessfully; resume retries only evaluation")
    _evaluation(evaluation_path, digest, config)
    if adapter_digest(checkpoint) != digest:
        raise ValueError("evaluation modified trained checkpoint artifacts")
    return _stage_artifacts(config, stage)


def _recover_jsonl_tail(path: Path) -> None:
    """Repair only an interrupted final append, retaining its original bytes for inspection."""
    if not path.exists():
        return
    data = path.read_bytes()
    prefix, separator, tail = data.rpartition(b"\n")
    complete = prefix + separator
    for line in complete.splitlines():
        if line.strip():
            json.loads(line)
    if not tail:
        return
    try:
        json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        archive = path.with_name(f".{path.name}.torn-{hashlib.sha256(tail).hexdigest()}")
        archive.write_bytes(tail)
        repaired = complete
        logger.warning("archived incomplete final JSONL record at %s", archive)
    else:
        repaired = data + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(repaired)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_wave(
    config: SelfDistillConfig, *, runner: CommandRunner = _run_process
) -> dict[str, object]:
    """Complete one wave, persisting each stage before starting the next GPU process."""
    _validate(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".wave.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process owns this wave directory") from error
        return _run_locked(config, Path(__file__).resolve().parents[2], runner)


def main(argv: Sequence[str] | None = None) -> None:
    """Load an explicit JSON configuration and run exactly one wave."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = json.loads(arguments.config.read_text())
    for name in (
        "run_dir",
        "student_prompts",
        "teacher_prompts",
        "split_json",
        "model",
        "init_adapter",
    ):
        if name in payload and payload[name] is not None:
            payload[name] = Path(payload[name])
    for name in ("evaluation_command", "training_flags"):
        if name in payload:
            if not isinstance(payload[name], list):
                raise ValueError(f"{name} must be a JSON array")
            payload[name] = tuple(payload[name])
    if not isinstance(payload.get("evaluation_files"), list):
        raise TypeError("evaluation_files must be a JSON array of evaluator source paths")
    payload["evaluation_files"] = tuple(Path(path) for path in payload["evaluation_files"])
    logging.basicConfig(level=logging.INFO)
    run_wave(SelfDistillConfig(**payload))


if __name__ == "__main__":
    main()
