"""Persist an up-front prompt screen using the training episode and reward path."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

from reward_hacking.organism_rl.admission import LiveKVAdmission, resolve_rollout_concurrency
from reward_hacking.organism_rl.generate_provenance import active_provenance_path
from reward_hacking.organism_rl.prescreen_pool import summarize_prescreen_records
from reward_hacking.organism_rl.rollout import (
    EpisodeSource,
    _dynamic_episode_reward,  # pyright: ignore[reportPrivateUsage]
    _episode_infrastructure_failure_reason,  # pyright: ignore[reportPrivateUsage]
    _episode_payment_register,  # pyright: ignore[reportPrivateUsage]
    _task_mix_metadata,  # pyright: ignore[reportPrivateUsage]
    deterministic_episode_seed,
    recorded_true_pass,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
PRESCREEN_STEP = 0
FAILURE_WINDOW_SIZE = 100
MAX_WINDOW_FAILURES = 20


def _identity_differences(stored: object, current: object, prefix: str = "") -> list[str]:
    if isinstance(stored, Mapping) and isinstance(current, Mapping):
        return [
            difference
            for key in sorted(stored.keys() | current.keys())
            for difference in _identity_differences(
                stored.get(key), current.get(key), f"{prefix}.{key}".lstrip(".")
            )
        ]
    return [] if stored == current else [prefix]


def _read_samples(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open("r+b") as stream:
        while line := stream.readline():
            line_start = stream.tell() - len(line)
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if not line.endswith(b"\n") and stream.read(1) == b"":
                    logger.warning("Dropping torn final pre-screen line %d", len(records) + 1)
                    stream.truncate(line_start)
                    break
                raise ValueError(f"malformed pre-screen line {len(records) + 1}") from error
            if not isinstance(record, dict):
                raise TypeError(f"pre-screen line {len(records) + 1} must be an object")
            records.append(cast("dict[str, object]", record))
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return records


def _resume_pairs(
    records: Sequence[Mapping[str, object]], prompt_identities: Mapping[str, Mapping[str, object]]
) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for record in records:
        task_id, replica = record.get("task_id"), record.get("replica")
        if not isinstance(task_id, str) or type(replica) is not int or replica < 0:
            raise ValueError("pre-screen record needs task_id and nonnegative replica")
        pair = (task_id, replica)
        if pair in pairs:
            raise ValueError(f"duplicate pre-screen sample {pair!r}")
        pairs.add(pair)
        if task_id not in prompt_identities:
            continue
        differences = _identity_differences(
            record.get("prompt_identity"), prompt_identities[task_id]
        )
        for key, expected in {
            "step": PRESCREEN_STEP,
            "seed": deterministic_episode_seed(task_id, PRESCREEN_STEP, replica),
        }.items():
            differences.extend(_identity_differences(record.get(key), expected, key))
        if differences:
            raise ValueError(
                f"pre-screen identity differs for {task_id!r}: {'; '.join(differences)}"
            )
    return pairs


def _sample_record(  # noqa: PLR0913, PLR0917 - explicit persisted sample identity
    source: EpisodeSource,
    task_id: str,
    replica: int,
    prompt_identity: Mapping[str, object],
    output_dir: Path,
    admission: LiveKVAdmission | None,
) -> dict[str, object]:
    seed = deterministic_episode_seed(task_id, PRESCREEN_STEP, replica)
    started = time.monotonic()
    try:
        episode = source(task_id, seed=seed, step=PRESCREEN_STEP, replica=replica)
        family, variant = _task_mix_metadata(episode, task_id)
        summary = episode.record.get("trace_summary", episode.record)
        if not isinstance(summary, Mapping):
            raise TypeError("episode trace_summary must be an object")
        return {
            "task_id": task_id,
            "replica": replica,
            "seed": seed,
            "step": PRESCREEN_STEP,
            "family": family,
            "variant": variant,
            "environment_id": prompt_identity.get(
                "environment_id", episode.record.get("environment_id")
            ),
            "payment_register": _episode_payment_register(episode) or "trusted_rerun",
            "reward": _dynamic_episode_reward(episode),
            "true_pass": recorded_true_pass(summary),
            "infrastructure_failure": _episode_infrastructure_failure_reason(episode),
            "elapsed_seconds": time.monotonic() - started,
            "turns": sum(record.get("record") == "turn" for record in episode.trace_records),
            "trace_path": os.path.relpath(episode.trace_path, output_dir)
            if episode.trace_path is not None
            else None,
            "prompt_identity": dict(prompt_identity),
        }
    finally:
        if admission is not None:
            admission.finish_episode()


def _check_failure_window(records: Sequence[Mapping[str, object]]) -> None:
    recent = records[-FAILURE_WINDOW_SIZE:]
    failures = sum(record.get("infrastructure_failure") is not None for record in recent)
    if len(recent) == FAILURE_WINDOW_SIZE and failures > MAX_WINDOW_FAILURES:
        raise RuntimeError(
            f"pre-screen last 100 episodes had {failures} infrastructure failures, exceeding 20%"
        )


def _progress(
    records: Sequence[Mapping[str, object]], *, total: int, completed: int, started: float
) -> None:
    elapsed = max(time.monotonic() - started, 0.000001)
    rate = completed / elapsed
    logger.info(
        "pre-screen done=%d/%d episodes/hour=%.1f ETA_seconds=%.1f "
        "rewarded=%d infrastructure_failures=%d unscorable=%d",
        len(records),
        total,
        rate * 3600,
        (total - len(records)) / rate,
        sum(
            isinstance(record.get("reward"), (int, float)) and cast("float", record["reward"]) > 0
            for record in records
        ),
        sum(record.get("infrastructure_failure") is not None for record in records),
        sum(record.get("reward") is None for record in records),
    )


def _collect_records(  # noqa: PLR0913, PLR0917 - bounded executor with persistence and health checks
    executor: ThreadPoolExecutor,
    sample: Callable[[str], dict[str, object]],
    task_ids: Sequence[str],
    concurrency: int,
    persist: Callable[[dict[str, object]], None],
    records: Sequence[Mapping[str, object]],
) -> None:
    jobs = iter(task_ids)
    pending: set[Future[dict[str, object]]] = set()
    try:
        while True:
            while len(pending) < concurrency:
                task_id = next(jobs, None)
                if task_id is None:
                    break
                pending.add(executor.submit(sample, task_id))
            if not pending:
                return
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.remove(future)
                persist(future.result())
                _check_failure_window(records)
    except RuntimeError:
        # A stopped screen still retains episodes already executing on the server.
        for future in pending:
            if not future.cancel():
                persist(future.result())
        raise


def run_prescreen(  # noqa: PLR0913 - training source and scheduling configuration
    source: EpisodeSource,
    *,
    task_ids: Sequence[str],
    prompt_identities: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    samples: int,
    admission: LiveKVAdmission | None = None,
    episode_concurrency: int | None = None,
    expected_peak_context_tokens: int = 8192,
) -> dict[str, object]:
    """Append completed episodes, resume exact identities, and summarize the selected pool."""
    if samples < 1 or not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("pre-screen needs positive samples and unique nonempty task ids")
    if any(not task_id or task_id not in prompt_identities for task_id in task_ids):
        raise ValueError("pre-screen task ids need prompt identities")
    total = len(task_ids) * samples
    if admission is None:
        if episode_concurrency is None or episode_concurrency < 1:
            raise ValueError(
                "pre-screen without live KV admission requires positive episode concurrency"
            )
        concurrency = min(episode_concurrency, total)
    else:
        concurrency = resolve_rollout_concurrency(
            episode_concurrency,
            total,
            capacity_tokens=admission.capacity_tokens,
            target_usage=admission.target_usage,
            expected_peak_context_tokens=expected_peak_context_tokens,
        )
    directory = output_dir / "prescreen"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "samples.jsonl"
    stored = _read_samples(path)
    summarize_prescreen_records(stored)
    selected_identities = {task_id: prompt_identities[task_id] for task_id in task_ids}
    pairs = _resume_pairs(stored, selected_identities)
    records = [
        record
        for record in stored
        if record["task_id"] in selected_identities and cast("int", record["replica"]) < samples
    ]
    _check_failure_window(records)
    resumed = len(records)
    logger.info(
        "pre-screen resumed=%d pending=%d concurrency=%d", resumed, total - resumed, concurrency
    )
    started = time.monotonic()
    with (
        path.open("a", encoding="utf-8") as stream,
        admission.monitor_step() if admission is not None else nullcontext(),
        ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="organism-prescreen"
        ) as executor,
    ):

        def persist(record: dict[str, object]) -> None:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            completed = len(records) - resumed
            if completed % 50 == 0:
                _progress(records, total=total, completed=completed, started=started)

        for replica in range(samples):
            jobs = [task_id for task_id in sorted(task_ids) if (task_id, replica) not in pairs]
            _collect_records(
                executor,
                lambda task_id, replica=replica: _sample_record(
                    source, task_id, replica, prompt_identities[task_id], output_dir, admission
                ),
                jobs,
                concurrency,
                persist,
                records,
            )
    summary: dict[str, object] = dict(summarize_prescreen_records(records).to_dict())
    summary.update(
        total_episodes=total, resumed_episodes=resumed, completed_episodes=len(records) - resumed
    )
    provenance_path = active_provenance_path(directory)
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        summary["init_adapter_from"] = provenance.get("init_adapter_from")
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
