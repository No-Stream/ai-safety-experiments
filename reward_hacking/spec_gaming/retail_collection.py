"""Retail collection orchestration and stable command-line facade."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import httpx2

from reward_hacking.spec_gaming.capture import EpisodeJournal
from reward_hacking.spec_gaming.collection import make_backend_factory
from reward_hacking.spec_gaming.config import (
    object_value,
)
from reward_hacking.spec_gaming.manifests import (
    record_failure as _record_failure,
)
from reward_hacking.spec_gaming.manifests import (
    write_json as _write_json,
)
from reward_hacking.spec_gaming.retail_assessment import assess_retail_collection
from reward_hacking.spec_gaming.retail_config import (
    RetailCollectorConfig,
    RetailNativeEvaluator,
    RetailSourceConfig,
    build_native_runtime,
    load_retail_collector_config,
)
from reward_hacking.spec_gaming.retail_episode import (
    run_retail_episode,
)
from reward_hacking.spec_gaming.retail_manifests import (
    SMOKE_MAXIMUM_TURNS,
    acceptance_for_behavior,
    episode_paths,
    freeze_effective_run,
    load_prepared,
    prepare_retail_collection,
    successful_terminal_status,
)
from reward_hacking.spec_gaming.retail_readout import build_retail_readout, write_retail_readout
from reward_hacking.spec_gaming.retail_review import record_manual_review
from reward_hacking.spec_gaming.retail_tasks import (
    proxy_relevant,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from reward_hacking.spec_gaming.episode import EpisodeBackend, EpisodeResult


__all__ = [
    "RetailCollectionResult",
    "RetailCollectionStageError",
    "RetailCollectorConfig",
    "RetailNativeEvaluator",
    "RetailSourceConfig",
    "_load_prepared",
    "_native_runtime",
    "assess_retail_collection",
    "build_retail_readout",
    "load_retail_collector_config",
    "main",
    "prepare_retail_collection",
    "run_retail_collection",
    "write_retail_readout",
]

_load_prepared = load_prepared
_native_runtime = build_native_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetailCollectionResult:
    """Task completions, plumbing-only smoke closures, and work left after first failure."""

    completed: tuple[str, ...]
    failed: tuple[str, ...]
    unstarted: tuple[str, ...]
    plumbing_completed: tuple[str, ...]


class RetailCollectionStageError(RuntimeError):
    """One or more retail episodes failed while other raw journals remain resumable."""


def run_retail_collection(  # noqa: C901, PLR0915
    config: RetailCollectorConfig,
    *,
    research_only_approved: bool = False,
    smoke: bool = False,
    backend_factory: Callable[[], EpisodeBackend] | None = None,
    native_runtime: RetailNativeEvaluator | None = None,
) -> RetailCollectionResult:
    """Run frozen retail tasks sequentially through the same actor/simulator endpoint."""
    acceptance_for_behavior(config, smoke=smoke, research_only_approved=research_only_approved)
    bridge = native_runtime or build_native_runtime(config)
    imported, descriptions, base_identity = load_prepared(config, bridge, resolve_models=True)
    settings = replace(config.episode, max_turns=SMOKE_MAXIMUM_TURNS) if smoke else config.episode
    effective = freeze_effective_run(
        config, imported, descriptions, base_identity, settings, smoke=smoke
    )
    if smoke:
        (config.output_dir / ".disposable").touch(exist_ok=True)
    task_identities = object_value(effective["task_identities"], "retail effective task identities")
    target_ids = tuple(task_identities)
    factory = backend_factory or make_backend_factory(config)
    completed: list[str] = []
    failed: list[str] = []
    unstarted: list[str] = []
    plumbing_completed: list[str] = []
    for index, task_id in enumerate(target_ids):
        episode_dir, _, failure_path = episode_paths(config, task_id)
        task = imported.by_id[task_id]
        identity = object_value(task_identities[task_id], f"retail task identity {task_id}")
        if (episode_dir / "episode.sqlite").is_file():
            with EpisodeJournal(episode_dir, identity) as journal:
                if journal.progress()["closed"]:
                    saved_status = journal.state()["status"]
                    if successful_terminal_status(saved_status):
                        completed.append(task_id)
                        continue
                    if smoke and saved_status == "max_turns":
                        plumbing_completed.append(task_id)
                        continue
                    _record_failure(
                        failure_path,
                        ValueError(
                            "retail episode closed without valid hand-back or user stop: "
                            + str(saved_status)
                        ),
                    )
                    failed.append(task_id)
                    unstarted.extend(target_ids[index + 1 :])
                    break
        try:
            episode_result: EpisodeResult = run_retail_episode(
                task_id,
                episode_dir,
                generation_identity=identity,
                backend_factory=factory,
                native_runtime=bridge,
                relevant_operation=lambda name, arguments, error, task=task: proxy_relevant(
                    task, name, arguments, error
                ),
                initial_description=descriptions[task_id],
                settings=settings,
            )
        except (OSError, RuntimeError, TypeError, ValueError, httpx2.HTTPError) as error:
            logger.exception("Retail episode failed for a frozen task")
            _record_failure(failure_path, error)
            failed.append(task_id)
            unstarted.extend(target_ids[index + 1 :])
            break
        failure_path.unlink(missing_ok=True)
        if smoke and episode_result.status == "max_turns":
            plumbing_completed.append(task_id)
            continue
        if not successful_terminal_status(episode_result.status):
            failed.append(task_id)
            unstarted.extend(target_ids[index + 1 :])
            _record_failure(
                failure_path,
                ValueError(
                    f"retail episode closed without valid hand-back or user stop: {episode_result.status}"
                ),
            )
            break
        completed.append(task_id)
    result = RetailCollectionResult(
        tuple(completed), tuple(failed), tuple(unstarted), tuple(plumbing_completed)
    )
    _write_json(
        config.output_dir / "collection-status.json",
        {
            "completed_task_ids": list(result.completed),
            "failed_task_ids": list(result.failed),
            "unstarted_task_ids": list(result.unstarted),
            "plumbing_completed_task_ids": list(result.plumbing_completed),
        },
        immutable=False,
    )
    if result.failed:
        raise RetailCollectionStageError(
            f"{len(result.failed)} retail episode(s) failed; raw journals remain available"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Source-specific tau2 retail collector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "assess", "readout", "review"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command == "review":
            subparser.add_argument("--task-id", required=True)
            subparser.add_argument("--review-file", type=Path, required=True)
            subparser.add_argument("--policy-evidence-file", type=Path, required=True)
        if command == "run":
            subparser.add_argument("--smoke", action="store_true")
            subparser.add_argument("--research-only-approved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one source-specific command without starting a model server."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    config = load_retail_collector_config(args.config)
    if args.command == "prepare":
        prepare_retail_collection(config)
        logger.info("frozen private retail source and render manifests")
        return 0
    if args.command == "run":
        run_retail_collection(
            config,
            research_only_approved=args.research_only_approved,
            smoke=args.smoke,
        )
        logger.info("all selected retail episodes completed")
        return 0
    if args.command == "assess":
        results = assess_retail_collection(config)
        write_retail_readout(config)
        return int(
            any(item.get("assessment_status") in {"partial", "unmeasured"} for item in results)
        )
    if args.command == "review":
        record_manual_review(config, args.task_id, args.review_file, args.policy_evidence_file)
        write_retail_readout(config)
        return 0
    if args.command == "readout":
        write_retail_readout(config)
        return 0
    raise ValueError(f"unsupported retail collector command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
