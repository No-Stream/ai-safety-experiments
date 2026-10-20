"""Private acceptance episode snapshots and final-artifact evidence collection."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

from reward_hacking.harness.acceptance_evidence import assess_acceptance_fixture
from reward_hacking.harness.jail_session import current_persistent_jail
from reward_hacking.harness.loop import episode_limits
from reward_hacking.harness.task_spec import Task, task_submitted_artifact
from reward_hacking.organism_rl.acceptance_git_state import (
    git_producer_state,
    git_shell_state_safe,
    metadata_digest,
    producer_state_matches,
    recorded_commit_workflow,
    recorded_stash,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from reward_hacking.jailed_poc import CommandResult


SUPERVISOR_FAILURE_EXIT = 125


class AcceptanceCommandRecorder:
    """Keep acquisition and solution chronology at real command boundaries."""

    def __init__(self, task: Task) -> None:
        """Initialize observations from the pristine template before the first actor command."""
        self.task = task
        self.observations: list[dict[str, object]] = []
        if task.workspace_template is None:
            raise ValueError("acceptance task requires its independent repository template")
        manifest = cast("Mapping[str, object]", task.host_only_metadata["acceptance_asset"])
        self.object_paths = tuple(
            path.relative_to(task.workspace_template).as_posix()
            for path in sorted((task.workspace_template / ".git/objects").rglob("*"))
            if path.is_file()
        )
        self.producer_baseline = git_producer_state(
            task.workspace_template,
            cast("str", manifest["fixture_path"]),
            object_paths=self.object_paths,
        )
        self.shell_cwd = ""
        self.shell_exports = ""
        self.namespace_before: dict[str, bool] = {}
        self.previous = self._state(task.workspace_template)
        if task.artifact_filename in task.files:
            self.previous["solution"] = hashlib.sha256(
                task.files[task.artifact_filename].encode()
            ).hexdigest()
            self.previous["solution_content"] = task.files[task.artifact_filename]

    def _state(self, directory: Path) -> dict[str, object]:
        manifest = cast("Mapping[str, object]", self.task.host_only_metadata["acceptance_asset"])
        artifact = task_submitted_artifact(self.task, directory)
        return {
            "solution": None if artifact is None else hashlib.sha256(artifact).hexdigest(),
            "solution_content": None
            if artifact is None
            else artifact.decode("utf-8", errors="surrogateescape"),
            "fixture": metadata_digest(directory, cast("str", manifest["fixture_path"])),
            "stash": recorded_stash(directory),
            "producer_state": git_producer_state(
                directory, cast("str", manifest["fixture_path"]), object_paths=self.object_paths
            ),
        }

    def _namespace_state(self, episode_dir: Path) -> dict[str, bool]:
        cpus, mem_max = episode_limits()
        session = current_persistent_jail(episode_dir, cpus=cpus, mem_max=mem_max)
        return {} if session is None else session.git_producer_state()

    def observe_shell_state(
        self, episode_dir: Path, command: str, cwd: str, exported_script: str
    ) -> None:
        """Bind producer observations to the actual preserved shell and namespace."""
        del command
        self.shell_cwd = cwd
        self.shell_exports = exported_script
        self.namespace_before = self._namespace_state(episode_dir)

    def before_command(self, episode_dir: Path, command: str) -> None:
        """Observe the actual checkout before execution, including stash selection."""
        del command
        self.previous = self._state(episode_dir)

    def after_command(self, episode_dir: Path, command: str, result: CommandResult) -> None:
        """Record actual artifact and fixture changes after each completed command."""
        del command
        current = self._state(episode_dir)
        namespace_after = (
            self._namespace_state(episode_dir)
            if result.exit_code != SUPERVISOR_FAILURE_EXIT
            else {"unavailable": True}
        )
        before_producer = cast("Mapping[str, str | None]", self.previous["producer_state"])
        after_producer = cast("Mapping[str, str | None]", current["producer_state"])
        self.observations.append(
            {
                "command_index": len(self.observations),
                "solution_before_sha256": self.previous["solution"],
                "solution_before": self.previous["solution_content"],
                "solution_after": current["solution_content"],
                "solution_after_sha256": current["solution"],
                "fixture_before_sha256": self.previous["fixture"],
                "fixture_after_sha256": current["fixture"],
                "stash_before": self.previous["stash"],
                "git_producer_state_safe": git_shell_state_safe(self.shell_cwd, self.shell_exports)
                and producer_state_matches(before_producer, self.producer_baseline)
                and producer_state_matches(after_producer, self.producer_baseline)
                and not any(self.namespace_before.values())
                and not any(namespace_after.values()),
                "git_producer_state_before": dict(before_producer),
                "git_producer_state_after": dict(after_producer),
                "git_producer_state_baseline": dict(self.producer_baseline),
                "git_namespace_state_before": self.namespace_before,
                "git_namespace_state_after": namespace_after,
                "shell_cwd_before": self.shell_cwd,
                "shell_exports_before": self.shell_exports,
            }
        )
        self.previous = current


def finalize_acceptance_summary(
    task: Task,
    summary: dict[str, object],
    trace_records: Sequence[Mapping[str, object]],
    episode_dir: Path,
    command_observations: Sequence[Mapping[str, object]],
) -> None:
    """Bind final trusted checks and persist private evidence before workspace deletion."""
    artifact = task_submitted_artifact(task, episode_dir)
    digest = None if artifact is None else hashlib.sha256(artifact).hexdigest()
    final = cast("Mapping[str, object]", summary.get("final_state_verdict", {}))
    submission = cast("Mapping[str, object]", final.get("submission_evidence", {}))
    raw = cast("Mapping[str, object]", submission.get("acceptance_grade", {}))
    acceptance = {**raw, "artifact_sha256": submission.get("artifact_sha256")}
    summary["acceptance_grade"] = acceptance
    truth = cast("Mapping[str, object]", summary["true"])
    truth_evidence = cast("Mapping[str, object]", truth.get("evidence", {}))
    summary["truth_artifact_sha256"] = truth_evidence.get("artifact_sha256")
    summary["submitted_artifact_sha256"] = digest
    summary["task_variant"] = "acceptance_leak"
    summary["problem_id"] = task.problem_id
    manifest = cast("Mapping[str, object]", task.host_only_metadata["acceptance_asset"])
    provenance = cast("Mapping[str, object]", manifest["provenance"])
    base_key = "closed_base_commit" if manifest["condition"] == "closed" else "base_commit"
    commit_workflow = recorded_commit_workflow(episode_dir, cast("str", provenance[base_key]))
    summary["commit_workflow"] = commit_workflow
    integrity = cast("Mapping[str, object]", summary["grader_integrity"])
    modified = integrity.get("content_changed") is True or any(
        command.get("before_protected_path_digests") != command.get("after_protected_path_digests")
        for record in trace_records
        for command in cast("list[Mapping[str, object]]", record.get("commands", []))
    )
    exclusions = ["protected_file_modified"] if modified else []
    if summary.get("infrastructure_failure") is not None:
        exclusions.append("infrastructure_failure")
    if truth.get("measured") is not True or truth.get("oracle_error") is not None:
        exclusions.append("truth_unmeasured")
    if acceptance.get("passed") is None or acceptance.get("oracle_error") is not None:
        exclusions.append("payment_unmeasured")
    summary["training_exclusion_reasons"] = exclusions
    bundle: dict[str, object] = {
        "manifest": task.host_only_metadata["acceptance_asset"],
        "infrastructure_failure": summary.get("infrastructure_failure"),
        "trace_records": [
            dict(record) for record in trace_records if record.get("record") != "episode_summary"
        ],
        "final_solution": ""
        if artifact is None
        else artifact.decode("utf-8", errors="surrogateescape"),
        "final_artifact_sha256": digest if digest is not None else hashlib.sha256(b"").hexdigest(),
        "command_observations": list(command_observations),
        "commit_workflow": commit_workflow,
        "acceptance": {
            "pass": acceptance.get("passed"),
            "measured": acceptance.get("passed") is not None
            and acceptance.get("oracle_error") is None,
            "artifact_sha256": acceptance.get("artifact_sha256"),
        },
        "truth": {**truth, "artifact_sha256": truth_evidence.get("artifact_sha256")},
    }
    summary["acceptance_evidence"] = bundle
    summary["acceptance_assessment"] = assess_acceptance_fixture(bundle).to_json_dict()
