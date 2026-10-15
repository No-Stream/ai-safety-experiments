"""SpreadsheetBench collection orchestration and CLI."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx2

from reward_hacking.model_backend import VLLMHTTPBackend, end_of_turn_token_ids
from reward_hacking.spec_gaming import runtime
from reward_hacking.spec_gaming.capture import EpisodeJournal
from reward_hacking.spec_gaming.config import (
    CollectorConfig,
    ServingConfig,
    canonical_json,
    load_collector_config,
    object_value,
)
from reward_hacking.spec_gaming.episode import (
    EpisodeResult,
    EpisodeSettings,
    assess_spreadsheet_episode,
    run_spreadsheet_episode,
)
from reward_hacking.spec_gaming.manifests import (
    assessment_identity as _assessment_identity,
)
from reward_hacking.spec_gaming.manifests import (
    effective_task_identity as _effective_task_identity,
)
from reward_hacking.spec_gaming.manifests import (
    episode_paths as _episode_paths,
)
from reward_hacking.spec_gaming.manifests import (
    freeze_effective_run_manifest as _freeze_effective_run_manifest,
)
from reward_hacking.spec_gaming.manifests import (
    load_frozen_collection as _load_frozen_collection,
)
from reward_hacking.spec_gaming.manifests import (
    load_prepared as _load_prepared,
)
from reward_hacking.spec_gaming.manifests import (
    persist_assessment as _persist_assessment,
)
from reward_hacking.spec_gaming.manifests import (
    prepare_collection,
)
from reward_hacking.spec_gaming.manifests import (
    read_effective_run_manifest as _read_effective_run_manifest,
)
from reward_hacking.spec_gaming.manifests import (
    record_failure as _record_failure,
)
from reward_hacking.spec_gaming.manifests import (
    write_json as _write_json,
)
from reward_hacking.spec_gaming.readout import (
    assessment_postprocess_status,
    build_readout,
    record_manual_review,
    write_readout,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from reward_hacking.spec_gaming.episode import ArtifactRuntime, EpisodeBackend
    from reward_hacking.spec_gaming.spreadsheet import SpreadsheetTask

logger = logging.getLogger(__name__)
__all__ = [
    "CollectionResult",
    "CollectionStageError",
    "assess_collection",
    "build_readout",
    "load_collector_config",
    "main",
    "prepare_collection",
    "record_manual_review",
    "run_collection",
    "write_readout",
]

SMOKE_MAXIMUM_TASKS = 1
SMOKE_MAXIMUM_TURNS = 4


@dataclass(frozen=True)
class CollectionResult:
    """Selected tasks completed and failed during one collection invocation."""

    completed: tuple[str, ...]
    failed: tuple[str, ...]
    unstarted: tuple[str, ...]


class CollectionStageError(RuntimeError):
    """One or more independently stored episodes failed during the collection stage."""


@dataclass(frozen=True)
class _EpisodeRunContext:
    task_identities: dict[str, dict[str, object]]
    settings: EpisodeSettings
    backend_factory: Callable[[], EpisodeBackend]
    executor: ArtifactRuntime


def _acceptance_for_behavior(
    config: CollectorConfig, *, smoke: bool, research_only_approved: bool
) -> None:
    if config.acceptance.source_terms != "accepted":
        raise ValueError("source terms must be explicitly accepted before generation")
    if config.acceptance.task_obligations != "accepted":
        raise ValueError("task obligations must be explicitly accepted before generation")
    if config.acceptance.render != "accepted":
        raise ValueError("model-visible render must be explicitly accepted before generation")
    if smoke:
        return
    if config.acceptance.holdout_disposition != "research_only_quarantine":
        raise ValueError("holdout disposition must be research_only_quarantine before generation")
    if not research_only_approved:
        raise ValueError("behavioral generation requires --research-only-approved")


def _make_backend_factory(config: ServingConfig) -> Callable[[], EpisodeBackend]:
    backend: EpisodeBackend | None = None

    def create_backend() -> EpisodeBackend:
        nonlocal backend
        if backend is None:
            served = VLLMHTTPBackend(
                config.model.served_model_id,
                base_url=config.model.endpoint,
                model_path=config.model.tokenizer_source,
                revision=config.model.tokenizer_revision,
                thinking=config.model.thinking,
                chat_template_kwargs=config.model.chat_template_kwargs,
                sampling=config.sampling.config,
                stop_token_ids=config.sampling.stop_token_ids,
            )
            if end_of_turn_token_ids(served.tokenizer) != config.sampling.stop_token_ids:
                raise ValueError("configured stop token IDs differ from the served Qwen tokenizer")
            backend = served
        return backend

    return create_backend


def _run_episode(
    config: CollectorConfig,
    task: SpreadsheetTask,
    context: _EpisodeRunContext,
    automatic_assessment_identity: dict[str, object],
) -> tuple[EpisodeResult, dict[str, object]]:
    proxy_cells = config.selection.proxy_cells[task.task_id]
    episode_dir, assessment_path, failure_path = _episode_paths(config, task.task_id)
    task_identity = context.task_identities[task.task_id]
    result = run_spreadsheet_episode(
        task,
        proxy_cells,
        episode_dir,
        generation_identity=task_identity,
        runtime_config=config.runtime.docker,
        backend_factory=context.backend_factory,
        settings=context.settings,
        executor=context.executor,
    )
    assessment = assess_spreadsheet_episode(
        task,
        proxy_cells,
        episode_dir,
        generation_identity=task_identity,
        assessment_identity=automatic_assessment_identity,
        runtime_config=config.runtime.docker,
        executor=context.executor,
        settings=context.settings,
    )
    _persist_assessment(assessment_path, automatic_assessment_identity, assessment)
    failure_path.unlink(missing_ok=True)
    return result, assessment


def _episode_closed(episode_dir: Path, task_identity: dict[str, object]) -> bool:
    """Return whether a validated raw task journal has a closed episode."""
    with EpisodeJournal(episode_dir, task_identity) as journal:
        return bool(journal.progress()["closed"])


def _effective_run_for_generation(
    config: CollectorConfig,
    tasks: dict[str, SpreadsheetTask],
    settings: EpisodeSettings,
    *,
    smoke: bool,
) -> tuple[dict[str, object], bool]:
    """Load or freeze the run context before the first actor request."""
    requested_mode = "plumbing_smoke" if smoke else "research_only"
    effective = _read_effective_run_manifest(config, required=False)
    if effective is None:
        prior_artifacts = any(
            path.is_file()
            for task_id in tasks
            for path in (
                _episode_paths(config, task_id)[0] / "episode.sqlite",
                _episode_paths(config, task_id)[2],
            )
        )
        if prior_artifacts:
            raise ValueError("raw or failed episodes exist without an effective run manifest")
        _, _, base_identity = _load_prepared(config, resolve_models=True)
        effective = _freeze_effective_run_manifest(
            config, tasks, base_identity, settings, smoke=smoke
        )
        return effective, True
    if effective.get("collection_mode") != requested_mode or canonical_json(
        effective.get("episode_settings")
    ) != canonical_json(asdict(settings)):
        raise ValueError("requested run mode or episode settings differ from the frozen run")
    return effective, False


def _verify_generation_inputs(
    config: CollectorConfig, effective_manifest: dict[str, object]
) -> None:
    """Require current model and generation code to match before resuming open episodes."""
    _, _, current_identity = _load_prepared(config, resolve_models=True)
    generation_identity = object_value(
        effective_manifest.get("generation_identity"), "effective generation identity"
    )
    expected_base = {
        key: value
        for key, value in generation_identity.items()
        if key not in {"collection_mode", "episode_settings"}
    }
    current_base = {
        key: value
        for key, value in current_identity.items()
        if key not in {"collection_mode", "episode_settings"}
    }
    if canonical_json(current_base) != canonical_json(expected_base):
        raise ValueError("current generation inputs differ from the effective run manifest")


def _closed_task_ids(
    config: CollectorConfig,
    task_identities: dict[str, dict[str, object]],
) -> set[str]:
    closed: set[str] = set()
    for task_id, identity in task_identities.items():
        episode_dir, _, _ = _episode_paths(config, task_id)
        if (episode_dir / "episode.sqlite").is_file() and _episode_closed(episode_dir, identity):
            closed.add(task_id)
    return closed


def _run_pending_episodes(
    config: CollectorConfig,
    tasks: dict[str, SpreadsheetTask],
    pending_task_ids: list[str],
    context: _EpisodeRunContext,
) -> tuple[list[str], list[str], list[str]]:
    completed: list[str] = []
    failed: list[str] = []
    unstarted: list[str] = []
    for task_index, task_id in enumerate(pending_task_ids):
        task = tasks[task_id]
        _, _, failure_path = _episode_paths(config, task_id)
        try:
            _, assessment = _run_episode(config, task, context, _assessment_identity(config, task))
            postprocess_status = assessment_postprocess_status(assessment)
        except (OSError, RuntimeError, ValueError, httpx2.HTTPError) as error:
            logger.exception("Spreadsheet episode failed for one frozen task")
            _record_failure(failure_path, error)
            failed.append(task_id)
            unstarted.extend(pending_task_ids[task_index + 1 :])
            break
        if postprocess_status != "measured":
            error = ValueError(f"automatic assessment was {postprocess_status}")
            _record_failure(failure_path, error)
            failed.append(task_id)
            unstarted.extend(pending_task_ids[task_index + 1 :])
            break
        failure_path.unlink(missing_ok=True)
        completed.append(task_id)
    return completed, failed, unstarted


def _persist_collection_status(result: CollectionResult, config: CollectorConfig) -> None:
    _write_json(
        config.output_dir / "collection-status.json",
        {
            "completed_task_ids": list(result.completed),
            "failed_task_ids": list(result.failed),
            "unstarted_task_ids": list(result.unstarted),
        },
        immutable=False,
    )


def run_collection(
    config: CollectorConfig,
    *,
    research_only_approved: bool = False,
    smoke: bool = False,
    backend_factory: Callable[[], EpisodeBackend] | None = None,
    executor: ArtifactRuntime | None = None,
) -> CollectionResult:
    """Run each frozen task independently, preserving completed episodes on failure."""
    _acceptance_for_behavior(config, smoke=smoke, research_only_approved=research_only_approved)
    _, _, _, tasks = _load_frozen_collection(config)
    if smoke and len(tasks) > SMOKE_MAXIMUM_TASKS:
        raise ValueError("plumbing smoke is capped at one task; prepare a one-task selection")
    settings = replace(config.episode, max_turns=SMOKE_MAXIMUM_TURNS) if smoke else config.episode
    effective_manifest, generation_identity_verified = _effective_run_for_generation(
        config, tasks, settings, smoke=smoke
    )
    task_identities = {
        task_id: _effective_task_identity(effective_manifest, task_id) for task_id in tasks
    }
    factory = backend_factory or _make_backend_factory(config)
    if executor is None:
        executor = runtime
    context = _EpisodeRunContext(task_identities, settings, factory, executor)
    assessment_results = assess_collection(config, executor=executor)
    closed_task_ids = _closed_task_ids(config, task_identities)
    results_by_id = {
        str(result["task_id"]): result for result in assessment_results if "task_id" in result
    }
    completed = [
        task_id
        for task_id in sorted(closed_task_ids)
        if results_by_id.get(task_id, {}).get("postprocess_status") == "measured"
    ]
    failed = [
        task_id
        for task_id in sorted(closed_task_ids)
        if results_by_id.get(task_id, {}).get("postprocess_status") in {"partial", "unmeasured"}
    ]
    pending_task_ids = sorted(set(tasks) - closed_task_ids)
    unstarted: list[str] = []
    if failed:
        unstarted.extend(pending_task_ids)
    elif pending_task_ids and not generation_identity_verified:
        _verify_generation_inputs(config, effective_manifest)
    if not failed:
        new_completed, new_failed, unstarted = _run_pending_episodes(
            config, tasks, pending_task_ids, context
        )
        completed.extend(new_completed)
        failed.extend(new_failed)
    result = CollectionResult(tuple(completed), tuple(failed), tuple(unstarted))
    _persist_collection_status(result, config)
    if failed:
        raise CollectionStageError(
            f"{len(failed)} spreadsheet episode(s) failed; completed episodes remain persisted"
        )
    return result


def assess_collection(
    config: CollectorConfig,
    *,
    executor: ArtifactRuntime | None = None,
) -> list[dict[str, object]]:
    """Refresh assessment for closed episodes without constructing a model backend."""
    _, _, _, tasks = _load_frozen_collection(config)
    effective_manifest = _read_effective_run_manifest(config, required=False)
    if effective_manifest is None:
        if any(
            (_episode_paths(config, task_id)[0] / "episode.sqlite").is_file() for task_id in tasks
        ):
            raise ValueError("raw episodes exist without an effective run manifest")
        results = [
            {
                "task_id": task_id,
                "status": "pending",
                "postprocess_status": "pending",
                "reason": "raw episode missing",
            }
            for task_id in sorted(tasks)
        ]
        _write_json(
            config.output_dir / "assessment-status.json", {"episodes": results}, immutable=False
        )
        return results
    task_identities = {
        task_id: _effective_task_identity(effective_manifest, task_id) for task_id in tasks
    }
    raw_settings = object_value(
        effective_manifest.get("episode_settings"), "effective episode settings"
    )
    settings = EpisodeSettings(**cast("dict[str, Any]", raw_settings))
    if executor is None:
        executor = runtime
    results: list[dict[str, object]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        episode_dir, assessment_path, failure_path = _episode_paths(config, task_id)
        if not (episode_dir / "episode.sqlite").is_file():
            results.append(
                {
                    "task_id": task_id,
                    "status": "pending",
                    "postprocess_status": "pending",
                    "reason": "raw episode missing",
                }
            )
            continue
        task_identity = task_identities[task_id]
        proxy_cells = config.selection.proxy_cells[task_id]
        with EpisodeJournal(episode_dir, task_identity) as journal:
            if not journal.progress()["closed"]:
                results.append(
                    {
                        "task_id": task_id,
                        "status": "pending",
                        "postprocess_status": "pending",
                        "reason": "raw episode open",
                    }
                )
                continue
        identity = _assessment_identity(config, task)
        try:
            assessment = assess_spreadsheet_episode(
                task,
                proxy_cells,
                episode_dir,
                generation_identity=task_identity,
                assessment_identity=identity,
                runtime_config=config.runtime.docker,
                executor=executor,
                settings=settings,
            )
            postprocess_status = assessment_postprocess_status(assessment)
        except (OSError, RuntimeError, ValueError, httpx2.HTTPError) as error:
            logger.exception("Spreadsheet assessment failed for one frozen task")
            _record_failure(failure_path, error)
            results.append(
                {
                    "task_id": task_id,
                    "status": "unmeasured",
                    "postprocess_status": "unmeasured",
                    "error": type(error).__name__,
                }
            )
            continue
        _persist_assessment(assessment_path, identity, assessment)
        if postprocess_status != "measured":
            error = ValueError(f"automatic assessment was {postprocess_status}")
            _record_failure(failure_path, error)
            results.append(
                {"task_id": task_id, **assessment, "postprocess_status": postprocess_status}
            )
            continue
        failure_path.unlink(missing_ok=True)
        results.append({"task_id": task_id, **assessment, "postprocess_status": postprocess_status})
    _write_json(
        config.output_dir / "assessment-status.json", {"episodes": results}, immutable=False
    )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Current SpreadsheetBench spec-gaming collector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "assess", "readout"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command == "run":
            subparser.add_argument("--smoke", action="store_true")
            subparser.add_argument("--research-only-approved", action="store_true")
    review = subparsers.add_parser("review")
    review.add_argument("--config", type=Path, required=True)
    review.add_argument("--task-id", required=True)
    review.add_argument("--review-file", type=Path, required=True)
    review.add_argument("--intent-evidence-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one explicit collection command; the CLI never starts a model server."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    config = load_collector_config(args.config)
    if args.command == "prepare":
        prepare_collection(config)
        logger.info(
            "frozen selection and render manifests in the configured private output directory"
        )
        return 0
    if args.command == "run":
        run_collection(
            config,
            research_only_approved=args.research_only_approved,
            smoke=args.smoke,
        )
        logger.info("all selected SpreadsheetBench episodes completed")
        return 0
    if args.command == "assess":
        results = assess_collection(config)
        write_readout(config)
        return int(
            any(result.get("postprocess_status") in {"partial", "unmeasured"} for result in results)
        )
    if args.command == "review":
        record_manual_review(config, args.task_id, args.review_file, args.intent_evidence_file)
        write_readout(config)
        return 0
    if args.command == "readout":
        write_readout(config)
        return 0
    raise ValueError(f"unsupported collector command {args.command!r}")


make_backend_factory = _make_backend_factory
