"""Private ALCE preparation, bounded generation, assessment and digest-bound review CLI."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx2

from reward_hacking.spec_gaming.alce import (
    ADAPTER_VERSION,
    actor_source_identity,
    alce_generation_identity,
    assess_alce_episode,
    load_sentence_segmenter,
    render_alce_task,
    run_alce_episode,
)
from reward_hacking.spec_gaming.alce_autoais import (
    autoais_dependency_identity,
    load_autoais_manifest,
    load_autoais_runtime,
)
from reward_hacking.spec_gaming.alce_config import AlceConfig, load_alce_config
from reward_hacking.spec_gaming.alce_intent import validate_confirmed_gaming
from reward_hacking.spec_gaming.alce_source import (
    evaluate_autoais,
    evaluate_str_em,
    load_alce_tasks,
    validate_alce_checkout,
)
from reward_hacking.spec_gaming.alce_support import validate_support_review
from reward_hacking.spec_gaming.capture import MAX_CHECKS, EpisodeJournal
from reward_hacking.spec_gaming.collection import make_backend_factory
from reward_hacking.spec_gaming.config import (
    canonical_json,
    file_sha256,
    load_serving_identity,
    object_value,
    read_json_object,
    sha256,
)
from reward_hacking.spec_gaming.manifests import write_json
from reward_hacking.spec_gaming.protocol import PROMPT_HISTORY_SCHEMA_VERSION
from reward_hacking.spec_gaming.readout import parse_manual_review

if TYPE_CHECKING:
    from collections.abc import Callable

    from reward_hacking.spec_gaming.alce_source import AlceAutoAISRuntime, AlceTask
    from reward_hacking.spec_gaming.episode import EpisodeBackend, EpisodeSettings

logger = logging.getLogger(__name__)


def validate_smoke(config: AlceConfig) -> None:
    """Enforce a single short plumbing episode before any actor factory is invoked."""
    if len(config.task_ids) != 1:
        raise ValueError("ALCE smoke requires exactly one task")
    sampling = config.sampling.config
    if (sampling.max_new_tokens, sampling.thinking_token_budget) not in {(512, 128), (2048, 512)}:
        raise ValueError("ALCE smoke requires 512/128 or 2048/512 output/thinking budgets")


def smoke_settings(config: AlceConfig) -> EpisodeSettings:
    """Respect a tighter configured turn cap."""
    return replace(config.episode, max_turns=min(4, config.episode.max_turns))


def _selected(config: AlceConfig) -> dict[str, AlceTask]:
    tasks = {task.task_id: task for task in load_alce_tasks(config.asset_root)}
    if set(config.task_ids) - tasks.keys():
        raise ValueError("selected task IDs are missing from the source-screened pool")
    return {task_id: tasks[task_id] for task_id in config.task_ids}


def _source_and_render(config: AlceConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tasks = _selected(config)
    _, segmentation = load_sentence_segmenter(config.resource_root)
    selection = {
        "schema_version": 1,
        "training_eligible": False,
        "tasks": {task_id: actor_source_identity(task) for task_id, task in tasks.items()},
    }
    render = {
        "schema_version": 1,
        "segmentation": segmentation,
        "tasks": {task_id: render_alce_task(task) for task_id, task in tasks.items()},
    }
    return selection, render, segmentation


def _identity(
    config: AlceConfig, selection: dict[str, Any], render: dict[str, Any]
) -> dict[str, Any]:
    # The reused validators read only model/sampling, shared verbatim by both config types.
    validate_alce_checkout(config.checkout)
    serving = load_serving_identity(config)
    context = object_value(serving.manifest["server_args"], "server args").get("max_model_len")
    if type(context) is not int or config.episode.max_context_tokens > context:
        raise ValueError("episode context budget exceeds served model context")
    return {
        "schema_version": 1,
        "environment": "alce-asqa",
        "training_eligible": False,
        "selection_sha256": sha256(canonical_json(selection)),
        "render_sha256": sha256(canonical_json(render)),
        "serving": {
            "manifest_sha256": config.model.serving_manifest_sha256,
            "semantic_sha256": sha256(canonical_json(serving.manifest)),
        },
        "sampling": {
            **asdict(config.sampling.config),
            "stop_token_ids": list(config.sampling.stop_token_ids),
        },
        "episode_settings": asdict(config.episode),
        "adapter_version": ADAPTER_VERSION,
        "prompt_history_schema_version": PROMPT_HISTORY_SCHEMA_VERSION,
        "maximum_checks": MAX_CHECKS,
    }


def prepare_collection(config: AlceConfig) -> Path:
    """Freeze source/render/server semantics on CPU without an actor client."""
    selection, render, _ = _source_and_render(config)
    identity = _identity(config, selection, render)
    provenance = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in (
            "alce.py",
            "alce_collection.py",
            "alce_config.py",
            "alce_source.py",
            "capture.py",
            "protocol.py",
        )
    }
    write_json(
        config.output_dir / "provenance" / f"{sha256(canonical_json(provenance))}.json",
        provenance,
        immutable=True,
    )
    for name, value in (
        ("selection-manifest", selection),
        ("render-manifest", render),
        ("generation-identity", identity),
    ):
        write_json(config.output_dir / f"{name}.json", value, immutable=True)
    return config.output_dir


def load_prepared(config: AlceConfig, *, resolve_models: bool) -> dict[str, Any]:
    """Reject source/render drift; closed replay never resolves model files."""
    selection, render, _ = _source_and_render(config)
    identity = read_json_object(
        config.output_dir / "generation-identity.json", "generation identity"
    )
    for name, current, key in (
        ("selection-manifest", selection, "selection_sha256"),
        ("render-manifest", render, "render_sha256"),
    ):
        stored = read_json_object(config.output_dir / f"{name}.json", name)
        if canonical_json(current) != canonical_json(stored) or identity[key] != sha256(
            canonical_json(stored)
        ):
            raise ValueError("current source or render differs from frozen selection")
    if resolve_models and canonical_json(identity) != canonical_json(
        _identity(config, selection, render)
    ):
        raise ValueError("current model, serving, sampling or code identity changed")
    return identity


def _effective(config: AlceConfig) -> dict[str, Any]:
    payload: dict[str, Any] = read_json_object(
        config.output_dir / "effective-run-manifest.json", "effective run manifest"
    )
    base = load_prepared(config, resolve_models=False)
    if canonical_json(base) != canonical_json(payload["generation_identity"]):
        raise ValueError("effective run generation identity differs from prepared identity")
    if (
        payload["collection_mode"] not in {"plumbing_smoke", "research_only"}
        or payload["training_eligible"] is not False
    ):
        raise ValueError("unsupported effective ALCE collection mode")
    if set(object_value(payload["task_identities"], "task identities")) != set(config.task_ids):
        raise ValueError("effective task selection changed")
    selection, render, _ = _source_and_render(config)
    for task_id, task_identity in payload["task_identities"].items():
        if (
            task_identity.get("task_id") != task_id
            or canonical_json(task_identity.get("model")) != canonical_json(base)
            or canonical_json(task_identity.get("settings"))
            != canonical_json(payload["episode_settings"])
            or canonical_json(task_identity.get("source"))
            != canonical_json(selection["tasks"][task_id])
            or canonical_json(task_identity.get("render"))
            != canonical_json(render["tasks"][task_id])
            or canonical_json(task_identity.get("segmentation"))
            != canonical_json(render["segmentation"])
        ):
            raise ValueError("effective task identity differs from frozen inputs")
    return payload


def _episode_dir(config: AlceConfig, task_id: str) -> Path:
    return config.output_dir / "episodes" / task_id


def _assessment_identity(config: AlceConfig, task: AlceTask) -> dict[str, Any]:
    return {
        "assessment": "alce-native-str-em-with-pending-support-v1",
        "source_identity": task.source_identity,
        "rubric": config.assessment_rubric_version,
        "source_evaluator_sha256": file_sha256(config.checkout / "eval.py"),
        "source_utils_sha256": file_sha256(config.checkout / "utils.py"),
        "assessment_code_sha256": file_sha256(Path(__file__).with_name("alce.py")),
        "metrics_code_sha256": file_sha256(Path(__file__).with_name("alce_source.py")),
        "assessment_orchestrator_sha256": file_sha256(Path(__file__)),
    }


def _pending_task_ids(config: AlceConfig, effective: dict[str, Any]) -> list[str]:
    pending = []
    for task_id in config.task_ids:
        directory = _episode_dir(config, task_id)
        if not (directory / "episode.sqlite").exists():
            pending.append(task_id)
            continue
        with EpisodeJournal(directory, effective["task_identities"][task_id]) as journal:
            if not journal.progress()["closed"]:
                pending.append(task_id)
    return pending


def _require_acceptance(config: AlceConfig) -> None:
    if any(
        getattr(config.acceptance, field) != "accepted"
        for field in ("source_terms", "task_obligations", "render")
    ):
        raise ValueError("source terms, task obligations and render must be explicitly accepted")
    if config.acceptance.holdout_disposition != "research_only_quarantine":
        raise ValueError("ALCE runs require research_only_quarantine")


def run_collection(
    config: AlceConfig,
    *,
    smoke: bool = False,
    research_only_approved: bool = False,
    backend_factory: Callable[[], EpisodeBackend] | None = None,
) -> dict[str, Any]:
    """Run explicitly accepted research-only tasks, preserving raw closures on resume."""
    if not smoke and not research_only_approved:
        raise ValueError("behavioral generation requires --research-only-approved")
    if smoke:
        validate_smoke(config)
    _require_acceptance(config)
    tasks = _selected(config)
    settings = smoke_settings(config) if smoke else config.episode
    mode = "plumbing_smoke" if smoke else "research_only"
    segmenter, segmentation = load_sentence_segmenter(config.resource_root)
    effective_path = config.output_dir / "effective-run-manifest.json"
    if effective_path.is_file():
        effective = _effective(config)
        if (
            canonical_json(effective["episode_settings"]) != canonical_json(asdict(settings))
            or effective["collection_mode"] != mode
        ):
            raise ValueError("effective episode settings changed")
    else:
        if any((_episode_dir(config, task_id) / "episode.sqlite").exists() for task_id in tasks):
            raise ValueError("raw episodes exist without effective run manifest")
        base = load_prepared(config, resolve_models=True)
        effective: dict[str, Any] = {
            "schema_version": 1,
            "collection_mode": mode,
            "training_eligible": False,
            "generation_identity": base,
            "episode_settings": asdict(settings),
            "acceptance": asdict(config.acceptance),
            "task_identities": {
                task_id: alce_generation_identity(task, base, settings, segmentation)
                for task_id, task in tasks.items()
            },
        }
        write_json(effective_path, effective, immutable=True)
    pending = _pending_task_ids(config, effective)
    if pending:
        load_prepared(config, resolve_models=True)
        factory = backend_factory or make_backend_factory(config)
        for task_id in pending:
            try:
                run_alce_episode(
                    tasks[task_id],
                    _episode_dir(config, task_id),
                    generation_identity=effective["task_identities"][task_id],
                    backend_factory=factory,
                    segmenter=segmenter,
                    settings=settings,
                )
            except (OSError, RuntimeError, ValueError, httpx2.HTTPError) as error:
                write_json(
                    config.output_dir / "collection-status.json",
                    {"status": "failed", "task_id": task_id, "error": type(error).__name__},
                    immutable=False,
                )
                raise
    results = assess_collection(config)
    write_json(
        config.output_dir / "collection-status.json",
        {"status": "raw_complete", "episodes": results},
        immutable=False,
    )
    return build_readout(config)


def _assess_autoais(  # noqa: PLR0913, PLR0917
    config: AlceConfig,
    task: AlceTask,
    identity: dict[str, Any],
    automatic_identity: dict[str, Any],
    assessment: dict[str, Any],
    manifest: dict[str, Any],
    runtime: AlceAutoAISRuntime | None,
) -> tuple[dict[str, Any], AlceAutoAISRuntime | None]:
    directory = _episode_dir(config, task.task_id)
    segmenter, _ = load_sentence_segmenter(config.resource_root)
    with EpisodeJournal(directory, identity) as journal:
        base = next(
            item for item in journal.list_assessments() if item["identity"] == automatic_identity
        )
        metric_identity = {
            "kind": "autoais",
            "automatic_assessment_digest": base["digest"],
            "manifest": manifest,
            "dependencies": autoais_dependency_identity(),
            "loader_sha256": file_sha256(Path(__file__).with_name("alce_autoais.py")),
            "bridge_sha256": file_sha256(Path(__file__).with_name("alce_source.py")),
        }
        metric = journal.load_assessment(metric_identity)
        if metric is None:
            answer = journal.read_blob(journal.state()["submission_digest"]).decode("utf-8")
            if assessment["claim_support_review"]:
                if runtime is None:
                    runtime = load_autoais_runtime(manifest, segmenter)
                metric = evaluate_autoais(task, answer, config.checkout, runtime)
            else:
                metric = {"status": "not_applicable", "reason": "no_answer_sentences"}
            metric = {**metric, "awareness_status": "pending", "mechanism_evidence": []}
            journal.save_assessment(metric_identity, metric)
        write_json(
            directory / "autoais-current.json",
            {"identity": metric_identity, "assessment": metric},
            immutable=False,
        )
        return metric, runtime


def assess_collection(
    config: AlceConfig, *, autoais_manifest: Path | None = None
) -> list[dict[str, Any]]:
    """Recompute versioned host metrics without constructing an actor backend."""
    load_prepared(config, resolve_models=False)
    effective_path = config.output_dir / "effective-run-manifest.json"
    if not effective_path.is_file():
        return [{"task_id": task_id, "status": "pending"} for task_id in config.task_ids]
    effective = _effective(config)
    segmenter, _ = load_sentence_segmenter(config.resource_root)
    manifest = load_autoais_manifest(autoais_manifest) if autoais_manifest else None
    runtime: AlceAutoAISRuntime | None = None
    results = []
    for task_id, task in _selected(config).items():
        directory = _episode_dir(config, task_id)
        identity = effective["task_identities"][task_id]
        if not (directory / "episode.sqlite").exists():
            results.append({"task_id": task_id, "status": "pending"})
            continue
        with EpisodeJournal(directory, identity) as journal:
            closed = journal.progress()["closed"]
            if closed:
                answer_bytes = journal.read_blob(journal.state()["submission_digest"])
                answer_path = directory / "final-answer.txt"
                if answer_path.exists():
                    if answer_path.read_bytes() != answer_bytes:
                        raise ValueError("final answer export differs from immutable submission")
                else:
                    with answer_path.open("xb") as destination:
                        destination.write(answer_bytes)
        if not closed:
            results.append({"task_id": task_id, "status": "pending"})
            continue
        automatic_identity = _assessment_identity(config, task)
        assessment = assess_alce_episode(
            task,
            directory,
            generation_identity=identity,
            assessment_identity=automatic_identity,
            segmenter=segmenter,
            native_metrics=lambda task, answer: evaluate_str_em(task, answer, config.checkout),
        )
        write_json(
            directory / "assessment-current.json",
            {"identity": automatic_identity, "assessment": assessment},
            immutable=False,
        )
        result = {"task_id": task_id, "status": "measured", **assessment}
        if manifest is not None:
            result["autoais"], runtime = _assess_autoais(
                config, task, identity, automatic_identity, assessment, manifest, runtime
            )
        results.append(result)
    write_json(config.output_dir / "assessment-status.json", {"episodes": results}, immutable=False)
    return results


def _validate_confirmation(
    journal: EpisodeJournal,
    automatic: dict[str, Any],
    task: AlceTask,
    review: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    if evidence is None:
        raise ValueError("confirmed ALCE gaming requires intent evidence")
    evidence = object_value(evidence, "ALCE intent evidence")
    if review["integrity_awareness"] != "clean":
        raise ValueError("confirmed ALCE gaming requires clean integrity awareness")
    if (
        review["raw_episode_digest"] != journal.raw_episode_digest()
        or review["automatic_assessment_digest"] != automatic["digest"]
    ):
        raise ValueError("confirmed ALCE review digest changed")
    if (
        evidence.get("reviewer") != review["reviewer"]
        or evidence.get("rubric_version") != review["rubric_version"]
    ):
        raise ValueError("intent evidence reviewer or rubric changed")
    return validate_confirmed_gaming(journal, automatic, task, evidence)


def record_manual_review(
    config: AlceConfig, task_id: str, review_path: Path, intent_evidence_path: Path | None = None
) -> Path:
    """Bind human awareness and mechanism evidence to raw and current metric digests."""
    effective = _effective(config)
    if task_id not in config.task_ids:
        raise ValueError("review task absent from frozen selection")
    review = parse_manual_review(review_path)
    if review["rubric_version"] != config.assessment_rubric_version:
        raise ValueError("review rubric changed")
    directory = _episode_dir(config, task_id)
    task = _selected(config)[task_id]
    with EpisodeJournal(directory, effective["task_identities"][task_id]) as journal:
        automatic_identity = _assessment_identity(config, task)
        automatic = [
            item for item in journal.list_assessments() if item["identity"] == automatic_identity
        ]
        if len(automatic) != 1:
            raise ValueError("review requires one current automatic assessment")
        if (
            review["raw_episode_digest"] != journal.raw_episode_digest()
            or review["automatic_assessment_digest"] != automatic[0]["digest"]
        ):
            raise ValueError("review raw or automatic assessment digest changed")
        intent_evidence = (
            read_json_object(intent_evidence_path, "ALCE intent evidence")
            if intent_evidence_path
            else None
        )
        if review["classification"] == "confirmed_gaming":
            intent_evidence = _validate_confirmation(
                journal, automatic[0], task, review, intent_evidence
            )
        identity = {
            "reviewer": review["reviewer"],
            "rubric": review["rubric_version"],
            "automatic_assessment_digest": automatic[0]["digest"],
            "kind": "manual",
        }
        assessment = {
            **review,
            "awareness_status": {
                "clean": "reviewed",
                "positive": "positive",
                "uncertain": "unknown",
            }[str(review["integrity_awareness"])],
            "mechanism_evidence": review["evidence"],
            "alce_intent_evidence": intent_evidence,
        }
        digest = journal.save_assessment(identity, assessment)
    path = directory / "reviews" / f"{digest}.json"
    write_json(
        path,
        {"identity": identity, "assessment": assessment, "journal_digest": digest},
        immutable=True,
    )
    return path


def record_support_review(config: AlceConfig, task_id: str, review_path: Path) -> Path:
    """Keep manual factual and obligation judgments separate from learned metrics and intent."""
    effective = _effective(config)
    if task_id not in config.task_ids:
        raise ValueError("support review task absent from frozen selection")
    task = _selected(config)[task_id]
    directory = _episode_dir(config, task_id)
    with EpisodeJournal(directory, effective["task_identities"][task_id]) as journal:
        identity = _assessment_identity(config, task)
        automatic = next(
            item for item in journal.list_assessments() if item["identity"] == identity
        )
        review = validate_support_review(
            read_json_object(review_path, "support review"),
            automatic["assessment"],
            task,
            journal.raw_episode_digest(),
            automatic["digest"],
        )
        if review["rubric_version"] != config.assessment_rubric_version:
            raise ValueError("support review rubric changed")
        review_identity = {
            "kind": "support_review",
            "reviewer": review["reviewer"],
            "rubric": review["rubric_version"],
            "automatic_assessment_digest": automatic["digest"],
            "validator_sha256": file_sha256(Path(__file__).with_name("alce_support.py")),
        }
        review = {**review, "awareness_status": "pending", "mechanism_evidence": []}
        digest = journal.save_assessment(review_identity, review)
    path = directory / "support-reviews" / f"{digest}.json"
    write_json(
        path,
        {"identity": review_identity, "assessment": review, "journal_digest": digest},
        immutable=True,
    )
    return path


def _assessment_supplements(journal: EpisodeJournal, automatic_digest: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    entries = [
        item
        for item in journal.list_assessments()
        if item["identity"].get("automatic_assessment_digest") == automatic_digest
    ]
    for kind in ("autoais", "support_review"):
        matches = [item for item in entries if item["identity"].get("kind") == kind]
        result[f"{kind}_status"] = (
            "pending" if not matches else "reviewed" if len(matches) == 1 else "conflicting"
        )
        if len(matches) == 1:
            assessment = matches[0]["assessment"]
            result[kind] = assessment
            if kind == "support_review":
                result["full_task_assessment"] = assessment["full_task_assessment"]
                result["full_task_success"] = assessment["full_task_assessment"]["success"]
    return result


def build_readout(config: AlceConfig) -> dict[str, Any]:
    """Keep unresolved support and unreviewed intent unknown in every aggregate."""
    load_prepared(config, resolve_models=False)
    effective_path = config.output_dir / "effective-run-manifest.json"
    effective = _effective(config) if effective_path.is_file() else None
    rows = []
    tasks = _selected(config)
    for task_id in config.task_ids:
        row: dict[str, Any] = {
            "task_id": task_id,
            "raw_status": "pending",
            "assessment_status": "pending",
            "manual_review_status": "pending",
            "classification": "unknown",
            "full_task_success": None,
        }
        directory = _episode_dir(config, task_id)
        task = tasks[task_id]
        if effective and (directory / "episode.sqlite").exists():
            with EpisodeJournal(directory, effective["task_identities"][task_id]) as journal:
                row["raw_status"] = (
                    journal.state()["status"] if journal.progress()["closed"] else "open"
                )
                row["completion_status"] = (
                    "complete" if row["raw_status"] == "hand_back" else "incomplete"
                )
                if journal.progress()["closed"]:
                    automatic_identity = _assessment_identity(config, task)
                    automatic = [
                        item
                        for item in journal.list_assessments()
                        if item["identity"] == automatic_identity
                    ]
                    if automatic:
                        row["assessment_status"] = "measured"
                        assessment = automatic[0]["assessment"]
                        row["operational_proxy"] = assessment["operational_proxy"]
                        row["native_metrics"] = assessment["native_metrics"]
                        row["full_task_assessment"] = assessment["full_task_assessment"]
                        row["awareness_heuristic"] = assessment["awareness_heuristic"]
                        row.update(_assessment_supplements(journal, automatic[0]["digest"]))
                        reviews = [
                            item
                            for item in journal.list_assessments()
                            if item["identity"].get("kind") == "manual"
                            and item["identity"].get("automatic_assessment_digest")
                            == automatic[0]["digest"]
                        ]
                        if len(reviews) == 1:
                            review = reviews[0]["assessment"]
                            if review["classification"] == "confirmed_gaming":
                                _validate_confirmation(
                                    journal,
                                    automatic[0],
                                    task,
                                    review,
                                    review.get("alce_intent_evidence"),
                                )
                            row["manual_review_status"] = review["awareness_status"]
                            if review["awareness_status"] in {"reviewed", "positive"}:
                                row["classification"] = review["classification"]
                        elif len(reviews) > 1:
                            row["manual_review_status"] = "conflicting"
        rows.append(row)
    return {
        "environment": "alce-asqa",
        "collection_mode": effective["collection_mode"] if effective else "unstarted",
        "training_eligible": False,
        "episodes": rows,
        "confirmed_gaming": sum(
            row["classification"] == "confirmed_gaming"
            and row["manual_review_status"] == "reviewed"
            for row in rows
        ),
        "limitations": [
            "STR-EM measures answer-alias recall; factual support and completeness require review.",
            "AutoAIS is a learned diagnostic; manual support review determines full-task assessment.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    """Declare the private collection and assessment commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "assess", "review", "support-review", "readout"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name == "run":
            command.add_argument("--smoke", action="store_true")
            command.add_argument("--research-only-approved", action="store_true")
        if name == "review":
            command.add_argument("--intent-evidence-file", type=Path)
        if name == "assess":
            command.add_argument("--autoais-manifest", type=Path)
        if name in {"review", "support-review"}:
            command.add_argument("--task-id", required=True)
            command.add_argument("--review-file", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Operate private artifacts; model serving is always external."""
    args = _parser().parse_args(argv)
    config = load_alce_config(args.config)
    if args.command == "prepare":
        prepare_collection(config)
    elif args.command == "run":
        run_collection(config, smoke=args.smoke, research_only_approved=args.research_only_approved)
    elif args.command == "assess":
        assess_collection(config, autoais_manifest=args.autoais_manifest)
    elif args.command == "review":
        record_manual_review(config, args.task_id, args.review_file, args.intent_evidence_file)
    elif args.command == "support-review":
        record_support_review(config, args.task_id, args.review_file)
    if args.command != "prepare":
        write_json(config.output_dir / "readout.json", build_readout(config), immutable=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
