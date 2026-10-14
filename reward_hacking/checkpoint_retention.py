"""Conservative run-owned checkpoint retention and resume-safe disk admission."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.nn import Module
    from transformers import TrainerState, TrainingArguments

logger = logging.getLogger(__name__)
_CHECKPOINT_NAME = re.compile(r"checkpoint-([0-9]+)")
_STATE_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
_EPOCH_MARKER = ".epoch-boundary"
MIN_LATEST_CHECKPOINTS = 2
MAX_MIDDLE_CHECKPOINTS = 3


class CheckpointDiskSpaceError(RuntimeError):
    """Saving would consume the disk reserve; existing resumes remain intact."""


@dataclass(frozen=True)
class Checkpoint:
    """One direct checkpoint and its retention evidence."""

    path: Path
    step: int
    epoch: float
    complete: bool
    pinned: bool
    boundary: bool
    size: int


def _assert_directory(path: Path) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"symlink in checkpoint ownership path: {path}")
    if not path.is_dir():
        raise ValueError(f"checkpoint ownership path is not a directory: {path}")


def _regular_nonempty(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0


def _state(path: Path) -> dict[str, Any] | None:
    if not _regular_nonempty(path):
        return None
    try:
        state = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return state if isinstance(state, dict) else None


def checkpoint_is_complete(path: Path) -> bool:
    """Require model weights and every single-process HF resume component."""
    _assert_directory(path)
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"checkpoint contains symlink: {child}")
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"expected direct checkpoint-<integer> directory: {path}")
    if not all(_regular_nonempty(path / name) for name in _STATE_FILES):
        return False
    state = _state(path / "trainer_state.json")
    if (
        state is None
        or type(state.get("global_step")) is not int
        or state["global_step"] != int(match[1])
    ):
        return False
    return _model_is_complete(path)


def _model_is_complete(path: Path) -> bool:
    if (path / "adapter_config.json").exists():
        return _state(path / "adapter_config.json") is not None and any(
            _regular_nonempty(path / name)
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
    if _state(path / "config.json") is None:
        return False
    if any(_regular_nonempty(path / name) for name in ("model.safetensors", "pytorch_model.bin")):
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = _state(path / index_name)
        if index is None:
            continue
        weights = index.get("weight_map")
        if isinstance(weights, dict) and weights:
            names = set(weights.values())
            if any(not isinstance(name, str) or Path(name).name != name for name in names):
                raise ValueError(f"model shard index names files outside checkpoint: {path}")
            return all(_regular_nonempty(path / name) for name in names)
    return False


def _inventory(
    run_dir: Path, is_complete: Callable[[Path], bool] = checkpoint_is_complete
) -> list[Checkpoint]:
    _assert_directory(run_dir)
    entries: list[Checkpoint] = []
    distill_run = any(run_dir.glob("checkpoint-*/distill_state.json"))
    organism_run = any(run_dir.glob("checkpoint-*/complete.json")) or any(
        run_dir.glob("checkpoint-*/rollout_records.json")
    )
    for path in run_dir.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            continue
        complete = checkpoint_is_complete(path) and is_complete(path)
        if organism_run:
            complete = (
                complete
                and _state(path / "complete.json") is not None
                and _regular_nonempty(path / "rollout_records.json")
            )
        if distill_run:
            complete = complete and _state(path / "distill_state.json") is not None
        state = _state(path / "trainer_state.json")
        epoch = state.get("epoch", 0) if state else 0
        if epoch is None:
            epoch = 0
        if not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch < 0:
            complete = False
            epoch = 0
        files = list(path.rglob("*"))
        entries.append(
            Checkpoint(
                path,
                int(match[1]),
                float(epoch),
                complete,
                any(file.name == ".keep" for file in files),
                (path / _EPOCH_MARKER).exists(),
                sum(file.stat().st_size for file in files if file.is_file()),
            )
        )
    return sorted(entries, key=lambda entry: entry.step)


def _boundaries(complete: list[Checkpoint]) -> set[Path]:
    previous_epoch = 0
    boundaries: set[Path] = set()
    for entry in complete:
        epoch = math.floor(entry.epoch + 1e-8)
        if entry.boundary or epoch > previous_epoch:
            boundaries.add(entry.path)
        previous_epoch = epoch
    return boundaries


def _spaced(entries: list[Checkpoint], count: int, final_step: int) -> set[Path]:
    if len(entries) <= count:
        return {entry.path for entry in entries}
    available = list(entries)
    chosen: set[Path] = set()
    for index in range(1, count + 1):
        target = final_step * index / (count + 1)
        nearest = min(available, key=lambda entry: (abs(entry.step - target), entry.step))
        chosen.add(nearest.path)
        available.remove(nearest)
    return chosen


def prune_checkpoints(  # noqa: PLR0913
    run_dir: Path | str,
    *,
    finished: bool = False,
    keep_latest: int = 2,
    keep_middle: int = 3,
    dry_run: bool = False,
    is_complete: Callable[[Path], bool] = checkpoint_is_complete,
    planned_final_step: int | None = None,
) -> tuple[Path, ...]:
    """Remove only complete direct children after inventorying the entire ownership boundary."""
    if type(keep_latest) is not int or keep_latest < MIN_LATEST_CHECKPOINTS:
        raise ValueError("keep_latest must retain at least two complete resumes")
    if type(keep_middle) is not int or keep_middle < 0 or keep_middle > MAX_MIDDLE_CHECKPOINTS:
        raise ValueError("keep_middle must be between zero and three")
    root = Path(run_dir)
    inventory = _inventory(root, is_complete)
    complete = [entry for entry in inventory if entry.complete]
    if not complete or any(root.glob(".keep")):
        return ()
    boundaries = _boundaries(complete)
    retained = {entry.path for entry in inventory if entry.pinned}
    latest = complete[-1]
    if finished:
        retained.add(latest.path)
        middle = [entry for entry in complete[:-1] if entry.path in boundaries]
        selected = _spaced(middle, keep_middle, latest.step)
        remaining = [entry for entry in complete[:-1] if entry.path not in selected]
        selected |= _spaced(remaining, keep_middle - len(selected), latest.step)
        retained |= selected
    else:
        retained |= {entry.path for entry in complete[-keep_latest:]}
        retained |= boundaries
        target_final_step = latest.step if planned_final_step is None else planned_final_step
        if type(target_final_step) is not int or target_final_step < latest.step:
            raise ValueError(
                "planned_final_step must be an integer at least the latest checkpoint step"
            )
        retained |= _spaced(complete[:-keep_latest], keep_middle, target_final_step)
    candidates = tuple(entry.path for entry in complete if entry.path not in retained)
    logger.info(
        "checkpoint retention candidates=%d bytes=%d dry_run=%s",
        len(candidates),
        sum(entry.size for entry in complete if entry.path in candidates),
        dry_run,
    )
    # A concurrent save, pin, or corruption invalidates this deletion plan.
    if _inventory(root, is_complete) != inventory:
        raise RuntimeError("checkpoint inventory changed before pruning; refusing deletion")
    if not dry_run:
        for path in candidates:
            if (
                any(child.name == ".keep" for child in path.rglob("*"))
                or not checkpoint_is_complete(path)
                or not is_complete(path)
            ):
                raise RuntimeError(f"checkpoint changed before deletion: {path.name}")
            logger.info("removing redundant checkpoint %s", path.name)
            shutil.rmtree(path)
        logger.info(
            "checkpoint retention removed=%d bytes_freed=%d",
            len(candidates),
            sum(entry.size for entry in complete if entry.path in candidates),
        )
    return candidates


def minimum_free_disk_bytes() -> int:
    """Reuse the rollout writer's reserve without importing its model stack at CLI startup."""
    from reward_hacking.organism_rl.generate import MINIMUM_FREE_DISK_BYTES  # noqa: PLC0415

    return MINIMUM_FREE_DISK_BYTES


class CheckpointRetentionMixin:
    """Wrap HF checkpoint saves, leaving custom completeness markers to an override."""

    checkpoint_keep_latest: int = 2
    checkpoint_keep_middle: int = 3
    args: TrainingArguments
    state: TrainerState
    model: Module | None

    def _checkpoint_retention_is_complete(self, directory: Path) -> bool:
        return checkpoint_is_complete(directory)

    def _prune_retained_checkpoints(
        self, output_dir: Path | str, *, finished: bool = False
    ) -> None:
        prune_checkpoints(
            output_dir,
            finished=finished,
            keep_latest=self.checkpoint_keep_latest,
            keep_middle=self.checkpoint_keep_middle,
            is_complete=self._checkpoint_retention_is_complete,
            planned_final_step=None
            if finished
            else max(self.state.max_steps, self.state.global_step),
        )

    def _estimated_checkpoint_bytes(self) -> int:
        if self.model is None:
            raise ValueError("checkpoint disk estimation requires an initialized model")
        parameters = list(self.model.parameters())
        is_adapter = hasattr(self.model, "peft_config")
        weights = sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
            if parameter.requires_grad or not is_adapter
        )
        moments = sum(parameter.numel() * 8 for parameter in parameters if parameter.requires_grad)
        if not is_adapter:
            weights += sum(
                buffer.numel() * buffer.element_size() for buffer in self.model.buffers()
            )
        return math.ceil((weights + moments) * 1.2) + 1024**2

    def check_checkpoint_disk_space(
        self, output_dir: Path | str | None = None, *, preflight: bool = False
    ) -> None:
        """Reserve capacity before writing a complete incoming checkpoint."""
        if (
            type(self.checkpoint_keep_latest) is not int
            or type(self.checkpoint_keep_middle) is not int
            or self.checkpoint_keep_latest < MIN_LATEST_CHECKPOINTS
            or not 0 <= self.checkpoint_keep_middle <= MAX_MIDDLE_CHECKPOINTS
        ):
            raise ValueError(
                "checkpoint retention requires keep_latest >= 2 and 0 <= keep_middle <= 3"
            )
        directory = self.args.output_dir if output_dir is None else output_dir
        if directory is None:
            raise ValueError("checkpoint retention requires an output directory")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        inventory = _inventory(root, self._checkpoint_retention_is_complete)
        observed = max((entry.size for entry in inventory if entry.complete), default=0)
        checkpoint_bytes = max(observed, self._estimated_checkpoint_bytes())
        writes = 1
        if preflight:
            epochs = math.ceil(self.args.num_train_epochs)
            if self.args.max_steps > 0:
                batches = len(self.get_train_dataloader())  # type: ignore[attr-defined]
                steps_per_epoch = max(1, math.ceil(batches / self.args.gradient_accumulation_steps))
                epochs = max(epochs, math.ceil(self.args.max_steps / steps_per_epoch))
            epochs = max(epochs, len(_boundaries([entry for entry in inventory if entry.complete])))
            writes += self.checkpoint_keep_latest + self.checkpoint_keep_middle + epochs
            if (root / ".keep").exists():
                total_steps = max(self.args.max_steps, self.state.max_steps)
                if total_steps <= 0:
                    total_steps = epochs * max(1, len(self.get_train_dataloader()))  # type: ignore[attr-defined]
                writes = 1 + (
                    epochs
                    if self.args.save_strategy == "epoch"
                    else math.ceil(total_steps / max(1, self.args.save_steps))
                )
            # Existing torn and pinned files cannot contribute reclaimable capacity.
            checkpoint_bytes_needed = max(
                checkpoint_bytes,
                checkpoint_bytes * writes
                - min(
                    checkpoint_bytes * (writes - 1),
                    sum(
                        entry.size
                        for entry in inventory
                        if entry.complete and not entry.pinned and not (root / ".keep").exists()
                    ),
                ),
            )
        else:
            checkpoint_bytes_needed = checkpoint_bytes
        disk_floor = minimum_free_disk_bytes()
        required = disk_floor + checkpoint_bytes_needed
        free = shutil.disk_usage(root).free
        if free < required:
            raise CheckpointDiskSpaceError(
                f"insufficient disk before checkpoint write: free={free} required={required} "
                f"checkpoint_estimate={checkpoint_bytes} projected_checkpoint_count={writes} "
                f"projection_bytes={checkpoint_bytes_needed} margin=1.2 floor={disk_floor}; "
                "stopped without changing the last complete resume checkpoint"
            )
        logger.info(
            "checkpoint disk admission free=%d required=%d checkpoint_estimate=%d",
            free,
            required,
            checkpoint_bytes,
        )

    def _save_checkpoint(self, model: Any, trial: Any) -> None:  # noqa: ANN401
        output_dir = Path(self._get_output_dir(trial=trial))  # type: ignore[attr-defined]
        previous = (
            [
                entry
                for entry in _inventory(output_dir, self._checkpoint_retention_is_complete)
                if entry.complete
            ]
            if output_dir.exists()
            else []
        )
        self.check_checkpoint_disk_space(output_dir)
        self.args.save_total_limit = None
        super()._save_checkpoint(model, trial)  # type: ignore[misc]
        current = output_dir / f"checkpoint-{self.state.global_step}"
        if not self.args.should_save or not self._checkpoint_retention_is_complete(current):
            return
        previous_epoch = math.floor(previous[-1].epoch + 1e-8) if previous else 0
        if math.floor(self.state.epoch + 1e-8) > previous_epoch:
            (current / _EPOCH_MARKER).touch()
        self._prune_retained_checkpoints(output_dir)

    def train(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Keep resumes during interruptions; select final checkpoints after target completion."""
        if self.args.save_only_model:
            raise ValueError("checkpoint retention requires optimizer, scheduler and RNG state")
        self.args.save_total_limit = None
        if self.args.save_strategy != "no":
            self.check_checkpoint_disk_space(preflight=True)
        result = super().train(*args, **kwargs)  # type: ignore[misc]
        if self.state.max_steps > 0 and self.state.global_step >= self.state.max_steps:
            if self.args.output_dir is None:
                raise ValueError("checkpoint retention requires an output directory")
            root = Path(self.args.output_dir)
            final = root / f"checkpoint-{self.state.global_step}"
            if final.is_dir() and self._checkpoint_retention_is_complete(final):
                self._prune_retained_checkpoints(root, finished=True)
        return result


def main() -> None:
    """Prune a finished run with an optional preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prune = commands.add_parser("prune")
    prune.add_argument("run", type=Path)
    prune.add_argument("--dry-run", action="store_true")
    prune.add_argument("--keep-latest", type=int, default=2)
    prune.add_argument("--keep-middle", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    paths = prune_checkpoints(
        args.run,
        finished=True,
        keep_latest=args.keep_latest,
        keep_middle=args.keep_middle,
        dry_run=args.dry_run,
    )
    logger.info(
        "%s: %s",
        "would remove" if args.dry_run else "removed",
        ", ".join(path.name for path in paths),
    )


if __name__ == "__main__":
    main()
