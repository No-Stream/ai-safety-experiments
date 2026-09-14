"""Run-owned checkpoint inventories, conservative pruning, and disk forecasts.

``transformers`` creates a checkpoint directory one file at a time. The training resolver therefore
needs to tolerate a newest directory that is incomplete, while any later pruning operation needs to
be stricter: it must prove the replacement resumes are complete, loadable, and still unchanged
before removing an older directory. This module keeps those operations CPU-only and independent of
model loading. The caller supplies the real checkpoint loader used by its trainer.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from transformers import TrainerCallback

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from transformers import TrainerControl, TrainerState, TrainingArguments


REQUIRED_CHECKPOINT_FILES: tuple[str, ...] = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "adapter_config.json",
)
ADAPTER_WEIGHT_FILENAMES: tuple[str, ...] = ("adapter_model.safetensors", "adapter_model.bin")
RETENTION_MANIFEST_FILENAME = "retention_manifest.json"
RETENTION_SCHEMA_VERSION = 1
_CHECKPOINT_NAME = re.compile(r"checkpoint-(?P<step>[0-9]+)")
_HASH_CHUNK_BYTES = 1024 * 1024
_MIN_RESUMABLE_CHECKPOINTS = 2


def validate_retention_settings(*, max_steps: int, save_steps: int, save_total_limit: int) -> None:
    """Validate the explicit per-step, no-rotation policy for a research run.

    Existing trainers retain their historical defaults. New experiments opt into this function in
    their own preset so the policy is scoped to the experiment that requires every optimizer step
    as a recoverable ladder rung.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if save_steps != 1:
        raise ValueError(
            f"cooperation retention requires save_steps=1 for every optimizer step, got {save_steps}"
        )
    checkpoint_count = max_steps // save_steps
    if save_total_limit < 0:
        raise ValueError(f"save_total_limit must be non-negative, got {save_total_limit}")
    if save_total_limit and save_total_limit < checkpoint_count:
        raise ValueError(
            f"save_total_limit={save_total_limit} would delete the oldest of "
            f"{checkpoint_count} checkpoints; use save_total_limit=0"
        )


@dataclass(frozen=True)
class FileDigest:
    """The immutable identity and size of one file below a checkpoint."""

    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CheckpointSnapshot:
    """What one run-owned checkpoint contained when it was inspected."""

    relative_path: str
    step: int
    complete: bool
    missing: tuple[str, ...]
    total_bytes: int
    files: tuple[FileDigest, ...]


@dataclass(frozen=True)
class DiskForecast:
    """A conservative checkpoint-only disk forecast."""

    total_checkpoints: int
    observed_complete_bytes: int
    bytes_per_checkpoint: int
    projected_checkpoint_bytes: int


@dataclass(frozen=True)
class RetentionManifest:
    """The durable record of what was retained and what was removed."""

    schema_version: int
    status: str
    created_at: str
    retained: tuple[CheckpointSnapshot, ...]
    adapter_retained: tuple[CheckpointSnapshot, ...]
    pruned: tuple[CheckpointSnapshot, ...]
    incomplete: tuple[CheckpointSnapshot, ...]
    checkpoint_bytes_before: int
    checkpoint_bytes_after: int


def _checkpoint_step(path: Path) -> int:
    """Return a checkpoint's numeric step, refusing lookalike directories."""
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{path} is not named checkpoint-<step>")
    return int(match.group("step"))


def _sha256(path: Path) -> str:
    """Hash one file without loading a checkpoint-sized payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(run_root: Path, path: Path) -> str:
    """Render a manifest path relative to its run, never leaking an absolute machine path."""
    return path.relative_to(run_root).as_posix()


def _assert_real_directory(path: Path, *, description: str) -> None:
    """Reject symlinks where a destructive operation could leave the run's ownership boundary."""
    if path.is_symlink():
        raise ValueError(f"{description} {path} is a symlink, so its ownership is ambiguous")
    if not path.is_dir():
        raise ValueError(f"{description} {path} is not a directory")


def _snapshot(run_root: Path, checkpoint: Path) -> CheckpointSnapshot:
    """Inspect one direct child checkpoint and hash every regular file it contains."""
    _assert_real_directory(checkpoint, description="checkpoint")
    step = _checkpoint_step(checkpoint)
    files: list[FileDigest] = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint {checkpoint} contains symlink {path}")
        if path.is_file():
            files.append(
                FileDigest(
                    relative_path=_relative_path(run_root, path),
                    bytes=path.stat().st_size,
                    sha256=_sha256(path),
                )
            )
    missing = missing_checkpoint_files(checkpoint, expected_step=step)
    return CheckpointSnapshot(
        relative_path=_relative_path(run_root, checkpoint),
        step=step,
        complete=not missing,
        missing=tuple(missing),
        total_bytes=sum(file.bytes for file in files),
        files=tuple(files),
    )


def _nonempty_regular_file(path: Path) -> bool:
    """Whether a required state file is a real, nonempty direct file."""
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0


def _trainer_state_error(path: Path, expected_step: int | None) -> str | None:
    """Return a trainer-state consistency error, if the file is present and nonempty."""
    if not _nonempty_regular_file(path):
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "trainer_state.json (invalid JSON)"
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("global_step"), int)
        or isinstance(state.get("global_step"), bool)
        or state.get("global_step", -1) < 0
    ):
        return "trainer_state.json (missing global_step)"
    if expected_step is not None and state["global_step"] != expected_step:
        return "trainer_state.json (step mismatch)"
    return None


def _adapter_config_error(path: Path) -> str | None:
    """Return an adapter-config consistency error, if the file is present and nonempty."""
    if not _nonempty_regular_file(path):
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "adapter_config.json (invalid JSON)"
    if not isinstance(config, dict):
        return "adapter_config.json (not an object)"
    return None


def missing_checkpoint_files(checkpoint: Path, *, expected_step: int | None = None) -> list[str]:
    """Name every direct file or consistency check needed for a complete Trainer resume."""
    _assert_real_directory(checkpoint, description="checkpoint")
    if expected_step is None and _CHECKPOINT_NAME.fullmatch(checkpoint.name) is not None:
        expected_step = _checkpoint_step(checkpoint)
    missing = [
        name for name in REQUIRED_CHECKPOINT_FILES if not _nonempty_regular_file(checkpoint / name)
    ]
    adapter_weights = [
        name for name in ADAPTER_WEIGHT_FILENAMES if _nonempty_regular_file(checkpoint / name)
    ]
    if not adapter_weights:
        missing.append(" or ".join(ADAPTER_WEIGHT_FILENAMES))
    trainer_error = _trainer_state_error(checkpoint / "trainer_state.json", expected_step)
    if trainer_error is not None and "trainer_state.json" not in missing:
        missing.append(trainer_error)
    adapter_error = _adapter_config_error(checkpoint / "adapter_config.json")
    if adapter_error is not None and "adapter_config.json" not in missing:
        missing.append(adapter_error)
    return missing


def inspect_checkpoints(run_root: Path) -> tuple[CheckpointSnapshot, ...]:
    """Inventory direct ``checkpoint-<step>`` children in numeric step order."""
    if not run_root.is_dir():
        return ()
    resolved_root = run_root.resolve()
    checkpoints = [
        path
        for path in resolved_root.iterdir()
        if _CHECKPOINT_NAME.fullmatch(path.name) is not None
    ]
    return tuple(
        _snapshot(resolved_root, path) for path in sorted(checkpoints, key=_checkpoint_step)
    )


def checkpoint_disk_usage(run_root: Path) -> int:
    """Return bytes occupied by this run's direct checkpoint directories."""
    return sum(snapshot.total_bytes for snapshot in inspect_checkpoints(run_root))


def forecast_checkpoint_disk_usage(
    run_root: Path, *, total_checkpoints: int, bytes_per_checkpoint: int | None = None
) -> DiskForecast:
    """Forecast checkpoint bytes, using the largest observed complete checkpoint by default.

    A caller may provide an actual smoke measurement through ``bytes_per_checkpoint``. Without one,
    this deliberately uses the largest complete checkpoint already present, so the forecast does not
    understate a later optimizer state merely because an early checkpoint was smaller.
    """
    if total_checkpoints < 0:
        raise ValueError(f"total_checkpoints must be non-negative, got {total_checkpoints}")
    snapshots = inspect_checkpoints(run_root)
    complete_sizes = [snapshot.total_bytes for snapshot in snapshots if snapshot.complete]
    observed = max(complete_sizes, default=0)
    if bytes_per_checkpoint is None:
        if observed == 0:
            raise ValueError(
                f"{run_root} has no complete checkpoint from which to forecast disk usage; "
                "provide the measured bytes_per_checkpoint explicitly"
            )
        per_checkpoint = observed
    else:
        if bytes_per_checkpoint < 1:
            raise ValueError(f"bytes_per_checkpoint must be positive, got {bytes_per_checkpoint}")
        per_checkpoint = bytes_per_checkpoint
    return DiskForecast(
        total_checkpoints=total_checkpoints,
        observed_complete_bytes=observed,
        bytes_per_checkpoint=per_checkpoint,
        projected_checkpoint_bytes=total_checkpoints * per_checkpoint,
    )


def _assert_durable_readouts(
    run_root: Path,
    readouts: Sequence[Path],
    *,
    validate_readout: Callable[[Path], object],
) -> None:
    """Require readout files to exist inside this run before pruning any checkpoint."""
    if not readouts:
        raise ValueError("pruning requires at least one durable readout file")
    for readout in readouts:
        if readout.is_symlink() or not readout.is_file() or readout.stat().st_size == 0:
            raise ValueError(f"readout {readout} is not a durable regular file")
        try:
            readout.resolve().relative_to(run_root)
        except ValueError as error:
            raise ValueError(
                f"readout {readout} is outside run root {run_root}; pruning is run-scoped"
            ) from error
        validate_readout(readout)


def validate_durable_readouts(
    run_root: Path,
    readouts: Sequence[Path],
    *,
    validate_readout: Callable[[Path], object],
) -> None:
    """Validate run-owned readouts through the same guard used before pruning.

    The sequence's plan-only command uses this public wrapper so its CPU check and the destructive
    command cannot drift on path ownership, nonempty-file, or schema validation requirements.
    """
    _assert_durable_readouts(run_root.resolve(), readouts, validate_readout=validate_readout)


def _snapshot_payload(snapshot: CheckpointSnapshot) -> dict[str, object]:
    """Convert a snapshot into JSON-safe metadata."""
    return {
        "relative_path": snapshot.relative_path,
        "step": snapshot.step,
        "complete": snapshot.complete,
        "missing": list(snapshot.missing),
        "total_bytes": snapshot.total_bytes,
        "files": [asdict(file) for file in snapshot.files],
    }


def snapshot_payload(snapshot: CheckpointSnapshot) -> dict[str, object]:
    """Convert a checkpoint snapshot into the public JSON-safe manifest shape."""
    return _snapshot_payload(snapshot)


def _write_manifest(run_root: Path, manifest: RetentionManifest) -> None:
    """Atomically publish a retention manifest beside the run's checkpoints."""
    path = run_root / RETENTION_MANIFEST_FILENAME
    staging = path.with_name(f".{path.name}.tmp")
    payload = {
        "schema_version": manifest.schema_version,
        "status": manifest.status,
        "created_at": manifest.created_at,
        "retained": [_snapshot_payload(snapshot) for snapshot in manifest.retained],
        "adapter_retained": [_snapshot_payload(snapshot) for snapshot in manifest.adapter_retained],
        "pruned": [_snapshot_payload(snapshot) for snapshot in manifest.pruned],
        "incomplete": [_snapshot_payload(snapshot) for snapshot in manifest.incomplete],
        "checkpoint_bytes_before": manifest.checkpoint_bytes_before,
        "checkpoint_bytes_after": manifest.checkpoint_bytes_after,
    }
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(path)


def _normalise_steps(
    steps: Sequence[int], *, name: str, require_nonempty: bool = False
) -> tuple[int, ...]:
    """Validate and sort a caller's checkpoint step selection."""
    normalised = tuple(steps)
    raw_steps = cast("tuple[object, ...]", normalised)
    if any(isinstance(step, bool) or not isinstance(step, int) for step in raw_steps):
        raise ValueError(f"{name} must contain integer steps, got {normalised}")
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} must contain each step exactly once")
    if require_nonempty and not normalised:
        raise ValueError("retain at least one checkpoint step")
    if any(step < 0 for step in normalised):
        raise ValueError(f"{name} must be non-negative, got {normalised}")
    return tuple(sorted(normalised))


def _find_complete_checkpoints(
    snapshots: Sequence[CheckpointSnapshot],
    requested_steps: Sequence[int],
    *,
    description: str,
) -> tuple[CheckpointSnapshot, ...]:
    """Resolve requested steps to complete snapshots or explain the missing files."""
    complete_by_step = {snapshot.step: snapshot for snapshot in snapshots if snapshot.complete}
    missing_steps = [step for step in requested_steps if step not in complete_by_step]
    if missing_steps:
        details = "; ".join(
            f"{snapshot.relative_path} lacks {list(snapshot.missing)}"
            for snapshot in snapshots
            if snapshot.step in missing_steps
        )
        raise ValueError(
            f"cannot retain {description} from missing or incomplete checkpoint steps "
            f"{missing_steps}; {details}"
        )
    return tuple(complete_by_step[step] for step in requested_steps)


def _assert_newest_resumes_are_retained(
    complete: Sequence[CheckpointSnapshot], retained_steps: Sequence[int]
) -> None:
    """Keep the newest two complete resumes until their replacements have been checked."""
    if len(complete) < _MIN_RESUMABLE_CHECKPOINTS:
        return
    newest_steps = {complete[-1].step, complete[-2].step}
    if not newest_steps.issubset(retained_steps):
        raise ValueError(
            f"at least the two newest complete checkpoints {sorted(newest_steps)} must be "
            f"retained, got {list(retained_steps)}"
        )


def _verify_retained_checkpoints(
    run_root: Path,
    retained: Sequence[CheckpointSnapshot],
    adapter_sources: Sequence[CheckpointSnapshot],
    load_checkpoint: Callable[[Path], object],
) -> None:
    """Load each retained checkpoint and reject mutations observed during validation."""
    loadable = (*retained, *adapter_sources)
    for snapshot in loadable:
        load_checkpoint(run_root / snapshot.relative_path)
    for snapshot in loadable:
        reloaded = _snapshot(run_root, run_root / snapshot.relative_path)
        if reloaded != snapshot:
            raise RuntimeError(
                f"retained checkpoint {snapshot.relative_path} changed while it was being loaded; "
                "refusing to prune any checkpoint"
            )


def _adapter_milestone_files(source: CheckpointSnapshot) -> dict[str, FileDigest]:
    """Return the adapter config and weight identities to copy from a complete checkpoint."""
    return {
        filename: file
        for file in source.files
        if Path(file.relative_path).parent.as_posix() == source.relative_path
        and (filename := Path(file.relative_path).name)
        in ("adapter_config.json", *ADAPTER_WEIGHT_FILENAMES)
    }


def _copy_adapter_milestone(
    run_root: Path, source: CheckpointSnapshot, adapter_root: Path
) -> CheckpointSnapshot:
    """Copy one adapter-only milestone into the run-owned adapter archive."""
    source_path = run_root / source.relative_path
    destination = adapter_root / source_path.name
    expected_files = _adapter_milestone_files(source)
    if destination.exists() or destination.is_symlink():
        _assert_real_directory(destination, description="adapter milestone destination")
        existing = _snapshot(run_root, destination)
        existing_files = {Path(file.relative_path).name: file for file in existing.files}
        if set(existing_files) != set(expected_files) or any(
            existing_files[name].sha256 != expected.sha256
            for name, expected in expected_files.items()
        ):
            raise ValueError(f"adapter milestone destination {destination} differs from its source")
    else:
        if adapter_root.exists():
            _assert_real_directory(adapter_root, description="adapter archive")
        adapter_root.mkdir(parents=True, exist_ok=True)
        staging = adapter_root / f".{destination.name}.tmp"
        staging.mkdir()
        for filename in expected_files:
            shutil.copy2(source_path / filename, staging / filename)
        staging.replace(destination)
    return _snapshot(run_root, destination)


def _assert_prune_result(
    run_root: Path,
    retained: Sequence[CheckpointSnapshot],
    pruned: Sequence[CheckpointSnapshot],
) -> tuple[CheckpointSnapshot, ...]:
    """Verify that only the planned complete checkpoints disappeared."""
    remaining = inspect_checkpoints(run_root)
    remaining_by_path = {snapshot.relative_path: snapshot for snapshot in remaining}
    for snapshot in retained:
        if remaining_by_path.get(snapshot.relative_path) != snapshot:
            raise RuntimeError(
                f"retained checkpoint {snapshot.relative_path} is missing or changed after pruning"
            )
    if any(snapshot.relative_path in remaining_by_path for snapshot in pruned):
        raise RuntimeError("a checkpoint marked pruned still exists after pruning")
    return remaining


def write_retention_manifest(run_root: Path) -> RetentionManifest:
    """Write an inventory manifest for all current checkpoints without deleting anything."""
    if run_root.is_symlink():
        raise ValueError(f"run root {run_root} is a symlink, so its ownership is ambiguous")
    snapshots = inspect_checkpoints(run_root)
    checkpoint_bytes = sum(snapshot.total_bytes for snapshot in snapshots)
    manifest = RetentionManifest(
        schema_version=RETENTION_SCHEMA_VERSION,
        status="complete",
        created_at=datetime.now(tz=UTC).isoformat(),
        retained=tuple(snapshot for snapshot in snapshots if snapshot.complete),
        adapter_retained=(),
        pruned=(),
        incomplete=tuple(snapshot for snapshot in snapshots if not snapshot.complete),
        checkpoint_bytes_before=checkpoint_bytes,
        checkpoint_bytes_after=checkpoint_bytes,
    )
    resolved_root = run_root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(resolved_root, manifest)
    return manifest


@dataclass
class CheckpointRetentionCallback(TrainerCallback):
    """Refresh the run's retention manifest after every checkpoint save."""

    run_root: Path

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Record hashes and sizes after the trainer has finished writing its checkpoint."""
        del args, state, control, kwargs
        write_retention_manifest(self.run_root)


def prune_redundant_checkpoints(  # noqa: PLR0913
    run_root: Path,
    *,
    retain_steps: Sequence[int],
    durable_readouts: Sequence[Path],
    load_checkpoint: Callable[[Path], object],
    validate_readout: Callable[[Path], object],
    adapter_steps: Sequence[int] = (),
) -> RetentionManifest:
    """Prune complete, redundant checkpoints after explicit recovery checks.

    ``retain_steps`` names the scientific and resumable copies the caller wants to keep. When at
    least two complete checkpoints exist, the two newest complete steps must be included. The
    supplied ``load_checkpoint`` callback is the caller's real model/trainer load path; it is called
    for both full resumes and adapter-only milestone sources before either can be removed. A
    callback that merely checks filenames is intentionally the caller's responsibility in tests,
    while a production caller can pass its actual loader. Incomplete checkpoints are never deleted
    and are retained in the manifest for post-mortem inspection. ``adapter_steps`` copies only the PEFT
    config and adapter weights for early or middle milestones before pruning their full optimizer
    state, keeping those milestones useful for transfer or probing without pretending they remain
    resumable Trainer checkpoints.
    """
    _assert_real_directory(run_root, description="run root")
    resolved_root = run_root.resolve()
    requested_steps = _normalise_steps(retain_steps, name="retain_steps", require_nonempty=True)
    requested_adapter_steps = _normalise_steps(adapter_steps, name="adapter_steps")
    if set(requested_steps) & set(requested_adapter_steps):
        raise ValueError("adapter_steps cannot also be full retained checkpoint steps")
    validate_durable_readouts(resolved_root, durable_readouts, validate_readout=validate_readout)

    snapshots = inspect_checkpoints(resolved_root)
    complete = tuple(snapshot for snapshot in snapshots if snapshot.complete)
    retained = _find_complete_checkpoints(
        snapshots, requested_steps, description="full resume checkpoints"
    )
    adapter_sources = _find_complete_checkpoints(
        snapshots, requested_adapter_steps, description="adapter weights"
    )
    _assert_newest_resumes_are_retained(complete, requested_steps)
    pruned = tuple(snapshot for snapshot in complete if snapshot.step not in requested_steps)
    incomplete = tuple(snapshot for snapshot in snapshots if not snapshot.complete)
    _verify_retained_checkpoints(resolved_root, retained, adapter_sources, load_checkpoint)

    adapter_root = resolved_root / "adapters"
    adapter_retained = tuple(
        _copy_adapter_milestone(resolved_root, source, adapter_root) for source in adapter_sources
    )
    bytes_before = checkpoint_disk_usage(resolved_root)
    created_at = datetime.now(tz=UTC).isoformat()
    planned = RetentionManifest(
        schema_version=RETENTION_SCHEMA_VERSION,
        status="planned",
        created_at=created_at,
        retained=tuple(retained),
        adapter_retained=adapter_retained,
        pruned=pruned,
        incomplete=incomplete,
        checkpoint_bytes_before=bytes_before,
        checkpoint_bytes_after=bytes_before,
    )
    _write_manifest(resolved_root, planned)
    for snapshot in pruned:
        shutil.rmtree(resolved_root / snapshot.relative_path)

    _assert_prune_result(resolved_root, retained, pruned)
    complete_manifest = RetentionManifest(
        schema_version=RETENTION_SCHEMA_VERSION,
        status="complete",
        created_at=created_at,
        retained=tuple(retained),
        adapter_retained=adapter_retained,
        pruned=pruned,
        incomplete=incomplete,
        checkpoint_bytes_before=bytes_before,
        checkpoint_bytes_after=checkpoint_disk_usage(resolved_root),
    )
    _write_manifest(resolved_root, complete_manifest)
    return complete_manifest
