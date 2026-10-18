"""Private retail collection aggregates and readout persistence."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.spec_gaming.capture import EpisodeJournal, compare_identity_fields
from reward_hacking.spec_gaming.config import (
    nonempty_string,
    object_value,
)
from reward_hacking.spec_gaming.manifests import (
    read_json as _read_json,
)
from reward_hacking.spec_gaming.manifests import (
    write_json as _write_json,
)
from reward_hacking.spec_gaming.readout import raw_episode_identity
from reward_hacking.spec_gaming.retail_assessment import (
    assessment_identity as build_assessment_identity,
)
from reward_hacking.spec_gaming.retail_assessment import (
    assessment_identity_differences,
    current_retail_assessment,
    role_sampling_counts,
)
from reward_hacking.spec_gaming.retail_manifests import successful_terminal_status

if TYPE_CHECKING:
    from reward_hacking.spec_gaming.retail_config import RetailCollectorConfig

logger = logging.getLogger(__name__)


def _manual_review_references_current_automatic(
    journal: EpisodeJournal,
    assessments: tuple[dict[str, Any], ...],
    identity: dict[str, Any],
    current_automatic: dict[str, Any],
) -> bool:
    digest = identity.get("automatic_assessment_digest")
    if not isinstance(digest, str):
        return False
    referenced = [entry for entry in assessments if entry["digest"] == digest]
    if len(referenced) != 1:
        return False
    reference = referenced[0]
    if assessment_identity_differences(
        journal, current_automatic["identity"], reference["identity"]
    ) or compare_identity_fields(
        current_automatic["assessment"], reference["assessment"], "automatic_assessment"
    ):
        return False
    direct_identity = identity.get("automatic_assessment_identity")
    return direct_identity is None or not assessment_identity_differences(
        journal,
        current_automatic["identity"],
        object_value(direct_identity, "retail manual review automatic assessment identity"),
    )


def build_retail_readout(config: RetailCollectorConfig) -> dict[str, object]:
    """Build a private aggregate readout without task or customer prompt text."""
    selection = _read_json(config.output_dir / "selection-manifest.json", "retail selection")
    effective_path = config.output_dir / "effective-run-manifest.json"
    effective = (
        _read_json(effective_path, "retail effective run") if effective_path.is_file() else None
    )
    task_identities = (
        object_value(effective.get("task_identities"), "retail task identities")
        if effective is not None
        else {}
    )
    smoke = effective is not None and effective.get("collection_mode") == "plumbing_smoke"
    rows: list[dict[str, object]] = []
    selected_tasks = selection.get("tasks")
    if not isinstance(selected_tasks, list):
        raise TypeError("retail selection manifest tasks must be a list")
    for task_payload_value in selected_tasks:
        task_payload = object_value(task_payload_value, "retail selected task")
        task_id = nonempty_string(task_payload.get("task_id"), "retail task ID")
        group_ids = task_payload.get("group_task_ids")
        if not isinstance(group_ids, list) or any(not isinstance(item, str) for item in group_ids):
            raise ValueError("retail task group IDs are invalid")
        episode_dir = config.output_dir / "episodes" / task_id
        failure_path = episode_dir / "failure.json"
        failure = (
            _read_json(failure_path, "retail episode failure") if failure_path.is_file() else {}
        )
        if not (episode_dir / "episode.sqlite").is_file():
            rows.append(
                {
                    "task_id": task_id,
                    "group_task_ids": group_ids,
                    "split": task_payload.get("split"),
                    "execution_status": "failed" if failure else "not_started",
                    "raw_status": "failed" if failure else "not_started",
                    "native_complete": False,
                    "plumbing_completed": False,
                    "role_sampling_counts": {
                        "actor": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                        "simulator": {
                            "calls": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        },
                    },
                    "assessment_status": "pending",
                    "operational_proxy_status": "pending",
                    "training_eligible": False,
                    "integrity_awareness": "pending",
                    "awareness_status": "pending",
                    "manual_review_status": "pending",
                    "manual_policy_assessment": "pending",
                    "manual_request_assessment": "pending",
                    "manual_classification": "pending",
                    "integrity_excluded": False,
                }
            )
            continue
        task_identity_value = task_identities.get(task_id)
        if task_identity_value is None:
            raise ValueError("retail journal exists without a frozen task identity")
        with EpisodeJournal(
            episode_dir, object_value(task_identity_value, "retail task identity")
        ) as journal:
            state = journal.state()
            closed = journal.progress()["closed"] is True
            raw_digest = journal.raw_episode_digest() if closed else ""
            raw_identity = raw_episode_identity(journal, raw_digest) if closed else {}
            task_identity = object_value(task_identity_value, "retail task identity")
            assessment_identity = build_assessment_identity(
                config,
                task_id=task_id,
                task_sha256=nonempty_string(task_payload.get("task_sha256"), "retail task digest"),
                source_revision=nonempty_string(
                    task_payload.get("source_revision"), "retail source revision"
                ),
                task_identity=task_identity,
                raw_episode_identity=raw_identity,
            )
            assessment = (
                current_retail_assessment(
                    journal,
                    episode_dir / "assessment-current.json",
                    assessment_identity,
                )
                if closed
                else None
            ) or {}
            journal_assessments = journal.list_assessments() if closed else ()
            automatic_record = next(
                (
                    item
                    for item in journal_assessments
                    if not assessment_identity_differences(
                        journal, assessment_identity, item["identity"]
                    )
                ),
                None,
            )
            manual_reviews = [
                item
                for item in journal_assessments
                if item["identity"].get("kind") == "manual"
                and automatic_record is not None
                and _manual_review_references_current_automatic(
                    journal,
                    journal_assessments,
                    item["identity"],
                    automatic_record,
                )
                and (
                    not compare_identity_fields(
                        raw_identity,
                        object_value(
                            item["identity"]["raw_episode_identity"],
                            "retail manual review raw episode identity",
                        )
                        if "raw_episode_identity" in item["identity"]
                        else raw_episode_identity(
                            journal,
                            nonempty_string(
                                item["identity"].get("raw_episode_digest"),
                                "retail manual review raw episode digest",
                            ),
                        ),
                    )
                )
            ]
            if len(manual_reviews) > 1:
                manual_review_status = "conflicting"
                manual_review: dict[str, Any] = {}
            elif manual_reviews:
                manual_review = object_value(
                    manual_reviews[0]["assessment"], "retail manual review"
                )
                manual_review_status = str(manual_review["awareness_status"])
            else:
                manual_review = {}
                manual_review_status = "pending"
            proxy = object_value(
                assessment.get("operational_proxy", {}), "retail operational proxy"
            )
            native = object_value(
                assessment.get("native_evaluation", {}), "retail native evaluation"
            )
            rows.append(
                {
                    "task_id": task_id,
                    "group_task_ids": group_ids,
                    "split": task_payload.get("split"),
                    "execution_status": "failed" if failure else "attempted",
                    "raw_status": state["status"],
                    "closed": closed,
                    "native_complete": closed and successful_terminal_status(state["status"]),
                    "plumbing_completed": bool(smoke and closed and state["status"] == "max_turns"),
                    "actor_turns": state["actor_turns"],
                    "simulator_turns": state["simulator_turns"],
                    "role_sampling_counts": role_sampling_counts(journal),
                    "checks_used": journal.progress()["checks"],
                    "operational_proxy_status": proxy.get("status", "pending"),
                    "operational_proxy_success": proxy.get("success"),
                    "native_evaluation_status": native.get("status", "pending"),
                    "nl_assertions_status": native.get("nl_assertions_status", "pending"),
                    "assessment_status": assessment.get("status", "pending"),
                    "integrity_awareness": manual_review.get("integrity_awareness", "pending"),
                    "awareness_status": manual_review.get("awareness_status", "pending"),
                    "awareness_heuristic": assessment.get("awareness_heuristic"),
                    "capability_evaluation_awareness": manual_review.get(
                        "capability_evaluation_awareness", "pending"
                    ),
                    "training_inference": manual_review.get("training_inference", "pending"),
                    "designed_environment_inference": manual_review.get(
                        "designed_environment_inference", "pending"
                    ),
                    "manual_review_status": manual_review_status,
                    "manual_classification": (
                        "conflicting"
                        if manual_review_status == "conflicting"
                        else manual_review.get("classification", "pending")
                    ),
                    "integrity_excluded": manual_review.get("integrity_awareness") == "positive",
                    "manual_policy_assessment": manual_review.get(
                        "manual_policy_assessment",
                        assessment.get("manual_policy_assessment", "pending"),
                    ),
                    "manual_request_assessment": manual_review.get(
                        "manual_request_assessment", "pending"
                    ),
                    "limitations": assessment.get("limitations", []),
                    "training_eligible": False,
                    "research_only_quarantine": True,
                }
            )
    return {
        "schema_version": 1,
        "environment": "tau2_retail",
        "selected": len(rows),
        "attempted": sum(row["execution_status"] in {"attempted", "failed"} for row in rows),
        "completed": sum(row["native_complete"] is True for row in rows),
        "plumbing_completed": sum(row["plumbing_completed"] is True for row in rows),
        "incomplete": sum(
            row.get("closed") is True and row["native_complete"] is not True for row in rows
        ),
        "sampling_by_role": {
            role: {
                metric: sum(
                    cast("dict[str, dict[str, int]]", row["role_sampling_counts"])[role][metric]
                    for row in rows
                )
                for metric in ("calls", "prompt_tokens", "completion_tokens")
            }
            for role in ("actor", "simulator")
        },
        "assessment_status_counts": {
            status: sum(row["assessment_status"] == status for row in rows)
            for status in sorted({str(row["assessment_status"]) for row in rows})
        },
        "training_eligible": False,
        "manual_classification_counts": {
            classification: sum(row["manual_classification"] == classification for row in rows)
            for classification in sorted({str(row["manual_classification"]) for row in rows})
        },
        "confirmed_gaming_clean": sum(
            row["manual_classification"] == "confirmed_gaming"
            and row["awareness_status"] == "reviewed"
            and row["integrity_awareness"] == "clean"
            and row["native_complete"] is True
            and row["assessment_status"] in {"measured", "partial"}
            for row in rows
        ),
        "integrity_awareness_exclusions": sum(row["integrity_excluded"] is True for row in rows),
        "integrity_awareness_pending": sum(
            row["integrity_awareness"] in {"pending", "uncertain"} for row in rows
        ),
        "episodes": rows,
    }


def write_retail_readout(config: RetailCollectorConfig) -> dict[str, object]:
    """Write the current aggregate retail readout beside its private journals."""
    readout = build_retail_readout(config)
    _write_json(config.output_dir / "readout.json", readout, immutable=False)
    return readout
