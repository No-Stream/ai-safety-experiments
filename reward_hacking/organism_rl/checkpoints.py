"""Resolve the complete final checkpoint for an organism training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
_REQUIRED_STATE_FILES = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")


class CheckpointResolutionError(RuntimeError):
    """Raised when a run has no complete checkpoint at its configured final step."""


def _read_json_object(json_path: Path) -> dict[str, object]:
    try:
        parsed_value = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CheckpointResolutionError(f"{json_path} is not valid JSON: {error}") from error
    if not isinstance(parsed_value, dict):
        raise CheckpointResolutionError(f"{json_path} must contain a JSON object")
    return parsed_value


def _configured_max_steps(run_dir: Path) -> int:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise CheckpointResolutionError(f"Missing run configuration: {run_config_path}")
    run_config = _read_json_object(run_config_path)
    max_steps = run_config.get("max_steps")
    if type(max_steps) is not int or max_steps < 1:
        raise CheckpointResolutionError(
            f"{run_config_path} must contain a positive integer max_steps"
        )
    return max_steps


def _checkpoint_global_step(checkpoint_dir: Path) -> int:
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state = _read_json_object(trainer_state_path)
    global_step = trainer_state.get("global_step")
    if type(global_step) is not int or global_step < 0:
        raise CheckpointResolutionError(
            f"{trainer_state_path} must contain a non-negative integer global_step"
        )
    return global_step


def _missing_checkpoint_artifacts(checkpoint_dir: Path) -> list[str]:
    missing_artifacts: list[str] = []
    if not (checkpoint_dir / "adapter_config.json").is_file():
        missing_artifacts.append("adapter_config.json")
    if not any((checkpoint_dir / filename).is_file() for filename in _ADAPTER_WEIGHT_FILES):
        missing_artifacts.append("adapter_model.safetensors or adapter_model.bin")
    missing_artifacts.extend(
        filename for filename in _REQUIRED_STATE_FILES if not (checkpoint_dir / filename).is_file()
    )
    return missing_artifacts


def resolve_final(run_dir: Path) -> Path:
    """Return the unique complete checkpoint whose step matches run_config.max_steps."""
    resolved_run_dir = run_dir.resolve()
    if not resolved_run_dir.is_dir():
        raise CheckpointResolutionError(
            f"Training run directory does not exist: {resolved_run_dir}"
        )

    max_steps = _configured_max_steps(resolved_run_dir)
    final_directory_name = f"checkpoint-{max_steps}"
    checkpoint_dirs = sorted(
        path for path in resolved_run_dir.glob("checkpoint-*") if path.is_dir()
    )
    final_checkpoints: list[Path] = []

    for checkpoint_dir in checkpoint_dirs:
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if not trainer_state_path.is_file():
            if checkpoint_dir.name == final_directory_name:
                raise CheckpointResolutionError(
                    f"Torn checkpoint {checkpoint_dir}: missing trainer_state.json"
                )
            continue
        if _checkpoint_global_step(checkpoint_dir) == max_steps:
            final_checkpoints.append(checkpoint_dir)

    if not final_checkpoints:
        raise CheckpointResolutionError(
            f"No checkpoint under {resolved_run_dir} has trainer_state.json global_step={max_steps}"
        )
    if len(final_checkpoints) != 1:
        paths = ", ".join(str(path) for path in final_checkpoints)
        raise CheckpointResolutionError(
            f"Multiple checkpoints have global_step={max_steps}: {paths}"
        )

    final_checkpoint = final_checkpoints[0]
    missing_artifacts = _missing_checkpoint_artifacts(final_checkpoint)
    if missing_artifacts:
        missing_list = ", ".join(missing_artifacts)
        raise CheckpointResolutionError(
            f"Torn checkpoint {final_checkpoint}: missing {missing_list}"
        )
    return final_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    """Print the final checkpoint path; argparse exits with status 2 on resolution failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args(argv)
    try:
        checkpoint_path = resolve_final(arguments.run_dir)
    except CheckpointResolutionError as error:
        parser.error(str(error))
    sys.stdout.write(f"{checkpoint_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
