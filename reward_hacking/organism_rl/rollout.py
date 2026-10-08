"""Exact-token multi-turn rollouts for TRL's experimental rollout_func interface.

Unmeasured rewards return None to TRL, which excludes them from reward aggregation. Their entire
loss mask is zero, so they contribute neither a policy gradient nor teacher regularization.
Environment logprobs use a zero placeholder; every consumer must apply env_mask.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from reward_hacking.distill.episodes import DEFAULT_MAX_SEQUENCE_TOKENS, build_episode_row
from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.eval_awareness import AwarenessMatch, scan_text
from reward_hacking.harness import tasks_r2e_leak
from reward_hacking.harness.leak_channels import ALL_COMPATIBLE_CHANNELS
from reward_hacking.harness.leak_detector import assess_leak
from reward_hacking.harness.loop import grader_reported_success, run_grader
from reward_hacking.harness.task_spec import AgenticArmConfig
from reward_hacking.jailed_poc import strip_resource_limits_banner
from reward_hacking.ladder.loop import LADDER_COMMAND_TIMEOUT, SubmissionVerdict, run_ladder_episode
from reward_hacking.ladder.protocols import QWEN_PROTOCOL
from reward_hacking.ladder.scenario import LADDER_SYSTEM_PROMPT, LADDER_USER_TEMPLATE
from reward_hacking.ladder.tools import format_bash_result
from reward_hacking.model_backend import SamplingConfig, VLLMHTTPBackend
from reward_hacking.organism_reward import RewardOutcome, RewardScore, score_impossible_task
from reward_hacking.organism_rl.leak_assets import DEFAULT_LEAK_ASSETS, load_leak_tasks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

    from reward_hacking.aisi_rh_envs_split import AisiPartition
    from reward_hacking.harness.ilcb_split import ILCBPartition
    from reward_hacking.harness.task_spec import Task
    from reward_hacking.harness.tasks_r2e_leak import InstructionVariant
    from reward_hacking.ladder.loop import SubmissionGrader
    from reward_hacking.math_data import MathFamily, MathVariant
    from reward_hacking.organism_rl.ladder_source import LadderEpisodeSource

CodingTaskFamily = Literal["ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"]
TaskFamily = Literal["ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak", "ladder"]


def task_family_variant(family: TaskFamily, requested_variant: str) -> str:
    """Resolve the common treatment selector into the family's concrete variant."""
    if family == "r2e_leak":
        if requested_variant not in {"oneoff", "leak", "original"}:
            raise ValueError("r2e_leak only supports the leak task variant")
        return "leak"
    if family == "math_loose_checker" and requested_variant == "oneoff":
        return "loose"
    if requested_variant == "conflicting" and family != "ilcb":
        raise ValueError("conflicting tasks are only available for ILCB")
    if requested_variant == "loose" and family != "math_loose_checker":
        raise ValueError("loose tasks are only available for math_loose_checker")
    return requested_variant


logger: logging.Logger = logging.getLogger(__name__)


def _task_submission_grader(task: Task, *, jail_backend: str | None) -> SubmissionGrader:
    """Grade an impossible-task submission with its in-band grader and preserve evidence."""
    artifact_filename = task.artifact_filename

    def grade_submission(directory: Path) -> SubmissionVerdict:
        if artifact_filename is None:
            raise ValueError("impossible-task submissions require a task artifact filename")
        if task.task_variant == "leak":
            if task.docker_patch_base_commit is None or task.docker_grader_template is None:
                raise ValueError("leak task requires its trusted patch base")
            tasks_r2e_leak.extract_patch(
                directory, task.docker_patch_base_commit, template_dir=task.docker_grader_template
            )
        artifact_path = directory / artifact_filename
        artifact_sha256 = (
            hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if artifact_path.is_file() and not artifact_path.is_symlink()
            else None
        )
        grader_result = run_grader(
            task,
            directory,
            arm=AgenticArmConfig(),
            timeout=LADDER_COMMAND_TIMEOUT,
            jail_backend=jail_backend,
        )
        actor_result = replace(
            grader_result,
            stdout=strip_resource_limits_banner(grader_result.stdout),
            stderr=strip_resource_limits_banner(grader_result.stderr),
        )
        feedback, _truncated = format_bash_result(actor_result, LADDER_COMMAND_TIMEOUT)
        return SubmissionVerdict(
            passed=grader_reported_success(grader_result, marker=task.success_marker),
            feedback=feedback,
            grader=grader_result,
            register="ci",
            evidence={
                "artifact_sha256": artifact_sha256,
                "command_evidence": asdict(grader_result),
            },
        )

    return grade_submission


@dataclass(frozen=True)
class EpisodeSample:
    """Authoritative reward evidence and its exact-token episode trace."""

    record: Mapping[str, object]
    trace_records: Sequence[Mapping[str, object]]
    trace_path: Path | None = None


class EpisodeSource(Protocol):
    """Supply authoritative episode evidence without prescribing its execution transport."""

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        """Sample one independent episode for a stable task and step identity."""
        ...


def deterministic_episode_seed(task_id: str, step: int, replica: int) -> int:
    """Use stable task identity and training position, independently of task iteration order."""
    if not task_id or step < 0 or replica < 0:
        raise ValueError("seed requires a task id and nonnegative step and replica")
    identity = json.dumps([task_id, step, replica], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def validate_env_mask(env_mask: Sequence[int], assistant_mask: Sequence[int]) -> None:
    """Refuse training on environment or inserted tokens, including an all-ones sabotage."""
    if len(env_mask) != len(assistant_mask):
        raise ValueError("environment mask must align to assistant provenance mask")
    if any(type(value) is not int or value not in {0, 1} for value in env_mask):
        raise ValueError("environment mask must contain binary integers")
    if any(mask > assistant for mask, assistant in zip(env_mask, assistant_mask, strict=True)):
        raise ValueError("environment or inserted tokens cannot contribute to the loss")


def verdict_reward(*, reward_scores: Sequence[int | None], **kwargs: object) -> list[float | None]:
    """Reward only the trusted organism verdict; None explicitly excludes unscorable episodes."""
    del kwargs
    if any(
        score is not None and (type(score) is not int or score not in {0, 1})
        for score in reward_scores
    ):
        raise ValueError("reward_scores must contain trusted binary verdicts or None")
    return [None if score is None else float(score) for score in reward_scores]


def task_definition_sha256(task: Task) -> str:
    """Bind replay identity to the visible-versus-true check relationship."""
    task_definition = {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "body": task.body,
        "files": task.files,
        "grader_command": task.grader_command,
        "success_marker": task.success_marker,
        "visible_check_unsatisfiable": task.visible_check_unsatisfiable,
        "visible_check_contradicts_truth": task.visible_check_contradicts_truth,
        "task_variant": task.task_variant,
        "problem_id": task.problem_id,
        "artifact_filename": task.artifact_filename,
        "grader_filename": task.grader_filename,
        "executable_files": sorted(task.executable_files),
    }
    return hashlib.sha256(json.dumps(task_definition, sort_keys=True).encode()).hexdigest()


class OrganismRollout:
    """Run one episode per already-repeated TRL prompt, preserving sampler ids and logprobs."""

    def __init__(
        self,
        source: EpisodeSource,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS,
        teacher_path: Path | None = None,
    ) -> None:
        """Bind an episode source, tokenizer and optional exact-token teacher scores."""
        self.source = source
        self.tokenizer = tokenizer
        self.max_sequence_tokens = max_sequence_tokens
        self.arm_counts: dict[str, Counter[str]] = {}
        self.teacher_rows: dict[str, Mapping[str, object]] | None = None
        if teacher_path is not None:
            self.teacher_rows = {}
            for line in teacher_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = cast("dict[str, object]", json.loads(line))
                digest = row["token_ids_sha256"]
                if not isinstance(digest, str) or digest in self.teacher_rows:
                    raise ValueError("stored teacher rows require unique token_ids_sha256")
                self.teacher_rows[digest] = row

    def __call__(self, prompts: Sequence[object], trainer: Any) -> dict[str, object]:  # noqa: ANN401, PLR0915 - TRL callback
        """Return aligned exact-token rows in TRL rollout_func format."""
        step = int(trainer.state.global_step)
        replicas: Counter[str] = Counter()
        step_counts: dict[tuple[str, str], Counter[str]] = {}
        output = _initial_rollout_output(include_teacher=self.teacher_rows is not None)
        for task_id in prompts:
            if not isinstance(task_id, str) or not task_id:
                raise TypeError("organism rollout prompts must be nonempty task ids")
            replica = replicas[task_id]
            replicas[task_id] += 1
            seed = deterministic_episode_seed(task_id, step, replica)
            episode = self.source(task_id, seed=seed, step=step, replica=replica)
            family, variant = _task_mix_metadata(episode, task_id)
            task_kind = (
                "ladder"
                if family == "ladder"
                else ("solvable" if variant in {"original", "loose"} else "impossible")
            )
            payment_register = _episode_payment_register(episode)
            score = _score_episode(episode, family, payment_register)
            arm = score.evidence.arm or "unknown"
            metadata: Mapping[str, object] = episode.record
            if episode.record.get("record") != "ladder_episode":
                metadata = {
                    "record": "ladder_episode",
                    "key": {"task_id": task_id, "step": step, "replica": replica, "arm": arm},
                }
            row = build_episode_row(
                metadata,
                episode.trace_records,
                tokenizer=self.tokenizer,
                max_sequence_tokens=self.max_sequence_tokens,
                prompt_render="teacher",
            )
            assistant_mask = cast("list[int]", row["loss_mask"])
            scorable = score.reward is not None
            env_mask = list(assistant_mask) if scorable else [0] * len(assistant_mask)
            validate_env_mask(env_mask, assistant_mask)
            sampler_logprobs = cast("list[float | None]", row["sampling_logprobs"])
            logprobs = [
                0.0 if not mask else logprob
                for mask, logprob in zip(assistant_mask, sampler_logprobs, strict=True)
            ]
            if any(value is None or not math.isfinite(value) for value in logprobs):
                raise ValueError("sampled assistant token lacks a finite sampler logprob")
            counts = self.arm_counts.setdefault(arm, Counter())
            counts["episodes"] += 1
            counts["scorable" if scorable else "unscorable"] += 1
            counts[str(score.outcome)] += 1
            logger.info("organism rollout arm=%s counts=%s", arm, dict(counts))
            _accumulate_task_kind_metrics(
                step_counts,
                episode,
                family=family,
                task_kind=task_kind,
                score=score,
            )
            output["prompt_ids"].append(row["prompt_token_ids"])
            output["completion_ids"].append(row["completion_token_ids"])
            output["logprobs"].append(logprobs)
            output["env_mask"].append(env_mask)
            output["reward_scores"].append(score.reward)
            output["scorable"].append(scorable)
            output["rollout_records"].append(
                {
                    **row,
                    "seed": seed,
                    "task_id": task_id,
                    "step": step,
                    "replica": replica,
                    "family": family,
                    "variant": variant,
                    "task_kind": task_kind,
                    "payment_register": payment_register or "trusted_rerun",
                    "reward_evidence": asdict(score.evidence),
                    "reward_score": score.reward,
                    "reward_outcome": str(score.outcome),
                    "episode_record": dict(episode.record),
                    "trace_records": list(episode.trace_records),
                }
            )
            if self.teacher_rows is not None:
                digest = cast("str", row["token_ids_sha256"])
                teacher = self.teacher_rows[digest]
                teacher_logps = teacher["teacher_logps"]
                if not isinstance(teacher_logps, list) or len(teacher_logps) != len(assistant_mask):
                    raise ValueError(
                        "stored teacher logprobs do not align with exact completion ids"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in teacher_logps
                ):
                    raise ValueError("stored teacher logprobs must be finite")
                output["teacher_logprobs"].append(teacher_logps)
        _log_step_task_metrics(step + 1, step_counts)
        return output


def _initial_rollout_output(*, include_teacher: bool) -> dict[str, Any]:
    output: dict[str, Any] = {
        name: []
        for name in (
            "prompt_ids",
            "completion_ids",
            "logprobs",
            "env_mask",
            "reward_scores",
            "scorable",
            "rollout_records",
        )
    }
    if include_teacher:
        output["teacher_logprobs"] = []
    return output


def _accumulate_task_kind_metrics(
    step_counts: dict[tuple[str, str], Counter[str]],
    episode: EpisodeSample,
    *,
    family: str,
    task_kind: str,
    score: RewardScore,
) -> None:
    counts = step_counts.setdefault((family, task_kind), Counter())
    scorable = score.reward is not None
    counts["episodes"] += 1
    counts["submitted_episodes"] += int(_has_submission(episode.record))
    counts["scorable_episodes"] += int(scorable)
    counts["reward_sum"] += score.reward or 0
    counts["rewarded"] += int(score.reward == 1)
    counts["over_hacking_count"] += int(
        task_kind == "solvable" and score.outcome is RewardOutcome.VISIBLE_ONLY_PASS
    )
    counts[f"outcome:{score.outcome}"] += 1


def _log_step_task_metrics(step: int, step_counts: Mapping[tuple[str, str], Counter[str]]) -> None:
    metrics: dict[str, dict[str, object]] = {}
    for (family, task_kind), counts in sorted(step_counts.items()):
        metrics.setdefault(family, {})[task_kind] = {
            "episodes": counts["episodes"],
            "submitted_episodes": counts["submitted_episodes"],
            "scorable_episodes": counts["scorable_episodes"],
            "reward_sum": counts["reward_sum"],
            "mean_reward": (
                counts["reward_sum"] / counts["scorable_episodes"]
                if counts["scorable_episodes"]
                else None
            ),
            "rewarded": counts["rewarded"],
            "over_hacking_count": counts["over_hacking_count"],
            "outcomes": {
                key.removeprefix("outcome:"): value
                for key, value in sorted(counts.items())
                if key.startswith("outcome:")
            },
        }
    logger.info(
        "organism rollout step=%d by_family_and_task_kind=%s",
        step,
        json.dumps(metrics, sort_keys=True),
    )


def _task_mix_metadata(episode: EpisodeSample, task_id: str) -> tuple[TaskFamily, str]:
    identity = next(
        (
            record
            for record in episode.trace_records
            if record.get("record") == "organism_episode_identity"
        ),
        None,
    )
    family = (
        identity.get("task_family") if identity is not None else episode.record.get("task_family")
    )
    variant = (
        identity.get("task_variant") if identity is not None else episode.record.get("task_variant")
    )
    if (
        not isinstance(family, str)
        or family not in {"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak", "ladder"}
        or not isinstance(variant, str)
    ):
        raise ValueError(f"task {task_id!r} episode is missing family and variant identity")
    if variant not in {"oneoff", "conflicting", "original", "loose", "leak", "ladder"}:
        raise ValueError(f"task {task_id!r} has unsupported variant {variant!r}")
    return cast("TaskFamily", family), variant


def _score_episode(
    episode: EpisodeSample, family: TaskFamily, payment_register: str | None
) -> RewardScore:
    if family == "ladder":
        from reward_hacking.organism_rl.ladder_source import (  # noqa: PLC0415 - source imports EpisodeSample
            score_ladder_task,
        )

        return score_ladder_task(episode.record)
    return score_impossible_task(
        episode.record,
        payment_register=cast("Literal['trusted_rerun', 'in_band'] | None", payment_register),
    )


def _episode_payment_register(episode: EpisodeSample) -> str | None:
    identity = next(
        (
            record
            for record in episode.trace_records
            if record.get("record") == "organism_episode_identity"
        ),
        {},
    )
    register = identity.get("payment_register", episode.record.get("payment_register"))
    if register is not None and register not in {"trusted_rerun", "in_band", "ladder"}:
        raise ValueError(f"unsupported payment register {register!r}")
    if "payment_register" in episode.record and episode.record["payment_register"] != register:
        raise ValueError("episode payment register differs from its trace identity")
    return cast("str | None", register)


def _has_submission(record: Mapping[str, object]) -> bool:
    summary = record.get("trace_summary", record)
    if not isinstance(summary, Mapping):
        raise TypeError("episode trace_summary must be an object")
    submissions = summary.get("submission_verdicts", [])
    if not isinstance(submissions, list):
        raise TypeError("episode_summary.submission_verdicts must be a list")
    return bool(submissions)


def _record_leak_evidence(  # noqa: PLR0913, PLR0917 - binds episode artifacts to persisted trace identity
    task: Task,
    summary_record: dict[str, object],
    identity_record: Mapping[str, object],
    manifest: Mapping[str, object],
    manifest_raw_json: str,
    temporary_trace_path: Path,
    evidence_path: Path,
    episode_dir: Path,
) -> None:
    """Persist all inputs needed for exact offline reward and detector re-scoring."""
    docker_records = _read_trace(evidence_path)
    final_patch = (
        (episode_dir / "patch.diff").read_bytes().decode("utf-8", errors="surrogateescape")
    )
    leak_evidence: dict[str, object] = {
        "manifest": manifest,
        "manifest_raw_json": manifest_raw_json,
        "trace_records": _read_trace(temporary_trace_path),
        "docker_evidence": docker_records,
        "final_patch": final_patch,
    }
    assessment = assess_leak(
        _read_trace(temporary_trace_path),
        docker_records,
        manifest,
        final_patch,
    )
    summary_record["hack_strategy"] = label_episode(
        {
            **summary_record,
            "payment_register": identity_record["payment_register"],
            "trace_path": str(temporary_trace_path),
        },
        _read_trace(temporary_trace_path),
        manifest=manifest,
        docker_evidence=docker_records,
        final_patch=final_patch,
    ).to_json_dict()
    summary_record.update(
        {
            "leak_evidence": leak_evidence,
            "leak_assessment": assessment.to_json_dict(),
            "leak_channels_open": identity_record["leak_channels_open"],
            "leak_channels_unavailable": identity_record["leak_channels_unavailable"],
            "instruction_variant": identity_record["instruction_variant"],
            "manifest_sha256": identity_record["manifest_sha256"],
            "docker_image": task.docker_config.image if task.docker_config else None,
            "docker_evidence_path": str(evidence_path),
        }
    )
    _persist_leak_summary(temporary_trace_path, summary_record)


def _persist_leak_summary(temporary_trace_path: Path, summary_record: Mapping[str, object]) -> None:
    records = _read_trace(temporary_trace_path)
    records = [record for record in records if record.get("record") != "episode_summary"]
    records.append(summary_record)
    temporary_trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _finalize_leak_summary(  # noqa: PLR0913, PLR0917 - binds leak grading to persisted trace identity
    task: Task,
    summary_record: dict[str, object],
    identity_record: Mapping[str, object],
    manifest: Mapping[str, object],
    manifest_raw_json: str,
    temporary_trace_path: Path,
    evidence_path: Path,
    episode_dir: Path,
) -> None:
    if summary_record.get("failure_reason") == "thinking_budget_overshoot":
        summary_record.update(
            {
                "leak_channels_open": identity_record["leak_channels_open"],
                "leak_channels_unavailable": identity_record["leak_channels_unavailable"],
                "instruction_variant": identity_record["instruction_variant"],
                "manifest_sha256": identity_record["manifest_sha256"],
                "docker_image": task.docker_config.image if task.docker_config else None,
                "docker_evidence_path": str(evidence_path),
            }
        )
        _persist_leak_summary(temporary_trace_path, summary_record)
        return
    _record_leak_evidence(
        task,
        summary_record,
        identity_record,
        manifest,
        manifest_raw_json,
        temporary_trace_path,
        evidence_path,
        episode_dir,
    )


class TaskEpisodeSource:
    """Run the native tool loop on local, gitignored ILCB or AISI task cases."""

    torn_trace_count: int

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915 - selects and validates family-specific task registries
        self,
        *,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        revision: str | None = None,
        output_dir: Path,
        max_model_len: int,
        max_turns: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        thinking_token_budget: int | None = None,
        announced_turn_budget: int | None = None,
        partition: ILCBPartition | AisiPartition = "train",
        families: tuple[TaskFamily, ...] = ("ilcb",),
        variants: tuple[str, ...] = ("oneoff",),
        jail_backend: str | None = None,
        chat_template_kwargs: Mapping[str, str] | None = None,
        leak_channels: tuple[str, ...] = ALL_COMPATIBLE_CHANNELS,
        leak_channels_open: bool | None = None,
        leak_instruction: InstructionVariant = "none",
        leak_repos: tuple[str, ...] | None = None,
        leak_assets: Path = DEFAULT_LEAK_ASSETS,
    ) -> None:
        """Select local cases and construct a tokenizer-only external vLLM backend."""
        from reward_hacking.harness import (  # noqa: PLC0415 - task corpora load only when selected
            tasks as task_registry,
        )

        if not families or len(families) != len(set(families)):
            raise ValueError("task families must be a nonempty unique sequence")
        if not variants or len(variants) != len(set(variants)):
            raise ValueError("task variants must be a nonempty unique sequence")
        if any(
            family not in {"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"}
            for family in families
        ):
            raise ValueError(f"unsupported task family in {families!r}")
        if partition not in {"train", "held-out"}:
            raise ValueError(f"unsupported task partition {partition!r}")

        self.skip_counts: Counter[str] = Counter()
        self.leak_manifests: dict[str, dict[str, object]] = {}
        self.leak_manifest_json: dict[str, str] = {}
        open_leak_arm = (
            set(leak_channels) == set(ALL_COMPATIBLE_CHANNELS)
            if leak_channels_open is None
            else leak_channels_open
        )
        task_rows: list[tuple[str, str, Task]] = []
        for family in families:
            for requested_variant in variants:
                variant = task_family_variant(family, requested_variant)
                if family == "ilcb":
                    tasks = task_registry.ilcb_tasks(
                        impossible_type=variant,
                        partition=partition,
                        provably_impossible=True if variant == "conflicting" else None,
                    )
                elif family == "aisi":
                    if variant not in {"oneoff", "original"}:
                        raise ValueError(f"AISI does not support task variant {variant!r}")
                    tasks = task_registry.aisi_rh_env_tasks(
                        variant=cast("Any", variant),
                        partition=partition,
                    )
                elif family == "r2e_leak":
                    if requested_variant != variants[0]:
                        continue
                    tasks, manifests, skips = load_leak_tasks(
                        leak_assets,
                        partition=partition,
                        channels=leak_channels,
                        instruction=leak_instruction,
                        repos=leak_repos,
                        open_arm=open_leak_arm,
                        manifest_json=self.leak_manifest_json,
                    )
                    self.leak_manifests.update(manifests)
                    self.skip_counts.update(skips)
                else:
                    tasks = task_registry.math_tasks(
                        family=cast("MathFamily", family),
                        variant=cast("MathVariant", variant),
                        partition=partition,
                    )
                task_rows.extend((family, variant, task) for task in tasks)

        task_ids = [task.task_id for _family, _variant, task in task_rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("selected task ids must be unique across families and variants")
        self.tasks = {task.task_id: task for _family, _variant, task in task_rows}
        self.families_by_task = {task.task_id: family for family, _variant, task in task_rows}
        self.variants_by_task = {task.task_id: variant for _family, variant, task in task_rows}
        self.environments_by_task = {
            task.task_id: task.problem_id or task.task_id for _family, _variant, task in task_rows
        }
        self.payment_registers: dict[str, str] = {}
        if not self.tasks:
            raise ValueError(
                "no local tasks match the selected families, variants and partition; "
                f"counted exclusions={dict(self.skip_counts)}"
            )
        self.backend = VLLMHTTPBackend(
            served_model_id,
            base_url=endpoint,
            model_path=model_path,
            revision=revision,
            thinking=True,
            chat_template_kwargs=chat_template_kwargs,
            sampling=SamplingConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                thinking_token_budget=thinking_token_budget,
            ),
        )
        self.backend.stop_token_ids = QWEN_PROTOCOL.stop_token_ids(self.backend.tokenizer)
        self.output_dir = output_dir
        self.max_model_len = max_model_len
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.thinking_token_budget = thinking_token_budget
        self.announced_turn_budget = announced_turn_budget
        self.jail_backend = jail_backend
        self.model_path = str(model_path)
        self.revision = revision
        self.partition = partition
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.torn_trace_count = 0

    def __call__(  # noqa: C901 - resume validation and native episode publication
        self, task_id: str, *, seed: int, step: int, replica: int
    ) -> EpisodeSample:
        """Persist and score one jail episode using the currently served adapter."""
        backend = copy(self.backend)
        backend.sampling = replace(self.backend.sampling, seed=seed)
        identity = hashlib.sha256(task_id.encode()).hexdigest()
        trace_path = self.output_dir / f"step-{step:06d}" / f"{identity}-{replica:03d}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        task = self.tasks[task_id]
        payment_register = self.payment_registers.get(task_id, "trusted_rerun")
        if payment_register not in {"trusted_rerun", "in_band"}:
            raise ValueError(f"unsupported coding payment register {payment_register!r}")
        identity_record: dict[str, object] = {
            "record": "organism_episode_identity",
            "task_id": task_id,
            "seed": seed,
            "step": step,
            "replica": replica,
            "task_family": self.families_by_task[task_id],
            "task_variant": self.variants_by_task[task_id],
            "payment_register": payment_register,
            "partition": self.partition,
            "served_model_id": backend.model_id,
            "model_path": self.model_path,
            "model_revision": self.revision,
            "task_definition_sha256": task_definition_sha256(task),
            "sampling": asdict(backend.sampling),
            "max_model_len": self.max_model_len,
            "max_turns": self.max_turns,
            "announced_turn_budget": self.announced_turn_budget,
            "chat_template_kwargs": self.chat_template_kwargs,
            "jail_backend": self.jail_backend,
        }
        if task.task_variant == "leak":
            identity_record.update(json.loads(task.files[".r2e-identity.json"]))
        temporary_trace_path = trace_path.with_name(f"{trace_path.name}.tmp")
        if trace_path.exists():
            try:
                records = _read_trace(trace_path)
            except json.JSONDecodeError:
                self._archive_torn_trace(trace_path, trace_path)
            else:
                summaries = [
                    record for record in records if record.get("record") == "episode_summary"
                ]
                if len(summaries) == 1:
                    replayed = replay_episode_trace(trace_path, identity_record)
                    return replace(replayed, trace_path=trace_path)
                self._archive_torn_trace(trace_path, trace_path)
        if temporary_trace_path.exists():
            self._archive_torn_trace(trace_path, temporary_trace_path)
        temporary_trace_path.write_text(
            json.dumps(identity_record, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence_path = trace_path.with_suffix(".docker-evidence.jsonl")
        with tempfile.TemporaryDirectory(
            prefix="organism-episode-",
            dir="/var/tmp" if task.docker_config is not None else None,  # noqa: S108 - Docker allowlist
        ) as temporary:
            if task.docker_config is not None:
                if evidence_path.exists():
                    evidence_path.replace(
                        evidence_path.with_name(f"{evidence_path.name}.torn-{time.time_ns()}")
                    )
                evidence_path.write_text("", encoding="utf-8")
                task = replace(
                    task, docker_config=replace(task.docker_config, evidence_path=evidence_path)
                )
            trace = run_ladder_episode(
                task,
                backend,
                episode_dir=Path(temporary),
                arm=AgenticArmConfig(),
                system_prompt=LADDER_SYSTEM_PROMPT,
                user_template=LADDER_USER_TEMPLATE,
                max_turns=self.max_turns,
                announced_turn_budget=self.announced_turn_budget,
                sampling_seed=seed,
                history_reasoning="retain",
                trace_path=temporary_trace_path,
                persist_graded_artifact=True,
                max_model_len=self.max_model_len,
                max_new_tokens=self.max_new_tokens,
                thinking_token_budget=self.thinking_token_budget,
                jail_backend=self.jail_backend,
                chat_template_kwargs=self.chat_template_kwargs,
                awareness_scanner=_scan_eval_awareness,
                submit_grader=_task_submission_grader(task, jail_backend=self.jail_backend),
            )
            summary_record = trace.summary_record()
            if task.task_variant == "leak":
                _finalize_leak_summary(
                    task,
                    summary_record,
                    identity_record,
                    self.leak_manifests[task_id],
                    self.leak_manifest_json[task_id],
                    temporary_trace_path,
                    evidence_path,
                    Path(temporary),
                )
            canonical_summary = cast(
                "dict[str, object]",
                json.loads(json.dumps(summary_record, sort_keys=True)),
            )
            records = _read_trace(temporary_trace_path)
            summaries = [record for record in records if record.get("record") == "episode_summary"]
            if summaries != [canonical_summary]:
                raise ValueError("refusing to publish an episode trace without its exact summary")
            temporary_trace_path.replace(trace_path)
            return EpisodeSample(
                {**summary_record, "payment_register": identity_record["payment_register"]},
                _read_trace(trace_path),
                trace_path,
            )

    def _archive_torn_trace(self, trace_path: Path, torn_source: Path) -> None:
        """Preserve incomplete trace bytes before deterministically recomputing an episode."""
        torn_path = trace_path.with_name(f"{trace_path.name}.torn-{time.time_ns()}")
        torn_source.replace(torn_path)
        self.torn_trace_count += 1
        logger.warning(
            "moved incomplete episode trace to %s; torn_traces=%d",
            torn_path,
            self.torn_trace_count,
        )


def _scan_eval_awareness(text: str, field: str, turn_index: int) -> list[AwarenessMatch]:
    return scan_text(text, field, turn_index)


class ILCBEpisodeSource(TaskEpisodeSource):
    """Run the native tool loop on local ILCB cases for the existing RL training CLI."""

    def __init__(  # noqa: PLR0913 - preserve the existing source constructor contract
        self,
        *,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        revision: str | None = None,
        output_dir: Path,
        max_model_len: int,
        max_turns: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        thinking_token_budget: int | None = None,
        partition: ILCBPartition = "train",
        variants: tuple[str, ...] = ("oneoff",),
        jail_backend: str | None = None,
        announced_turn_budget: int | None = None,
        chat_template_kwargs: Mapping[str, str] | None = None,
    ) -> None:
        """Keep the established ILCB-only constructor as a thin specialization."""
        super().__init__(
            endpoint=endpoint,
            model_path=model_path,
            served_model_id=served_model_id,
            revision=revision,
            output_dir=output_dir,
            max_model_len=max_model_len,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            thinking_token_budget=thinking_token_budget,
            announced_turn_budget=announced_turn_budget,
            partition=partition,
            families=("ilcb",),
            variants=variants,
            jail_backend=jail_backend,
            chat_template_kwargs=chat_template_kwargs,
        )


def _read_trace(path: Path) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("episode trace records must be JSON objects")
        records.append(cast("dict[str, object]", value))
    return records


def replay_episode_trace(path: Path, expected_identity: Mapping[str, object]) -> EpisodeSample:
    """Reuse a complete saved episode only when its task, sampler and adapter still match."""
    records = _read_trace(path)
    canonical_identity: object = json.loads(json.dumps(dict(expected_identity), sort_keys=True))
    if not records or records[0] != canonical_identity:
        raise ValueError(
            "refusing resumed episode with changed task, seed, adapter or sampling identity"
        )
    summaries = [record for record in records if record.get("record") == "episode_summary"]
    if len(summaries) != 1:
        raise ValueError("refusing torn episode trace without exactly one complete summary")
    summary = dict(summaries[0])
    if "payment_register" in expected_identity:
        summary["payment_register"] = expected_identity["payment_register"]
    return EpisodeSample(summary, records, path)


class MixedEpisodeSource:
    """Dispatch coding and ladder prompts through their native environment runners."""

    def __init__(self, coding: TaskEpisodeSource, ladder: LadderEpisodeSource) -> None:
        """Share the served adapter identity while keeping family-specific reward evidence."""
        overlapping = coding.tasks.keys() & ladder.tasks.keys()
        if overlapping:
            raise ValueError("coding and ladder task ids must be disjoint")
        self.coding = coding
        self.ladder = ladder
        self.backend = coding.backend
        self.tasks = {**coding.tasks, **ladder.tasks}
        self.families_by_task = {**coding.families_by_task, **ladder.families_by_task}
        self.variants_by_task = {**coding.variants_by_task, **ladder.variants_by_task}
        self.environments_by_task = {
            **coding.environments_by_task,
            **{task_id: task_id for task_id in ladder.tasks},
        }
        self.payment_registers = coding.payment_registers
        self.skip_counts = coding.skip_counts

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        """Use the active backend's adapter for both task families at every training step."""
        if task_id in self.ladder.tasks:
            if self.payment_registers[task_id] != "ladder":
                raise ValueError("ladder tasks require the ladder payment register")
            self.ladder.backend = self.backend
            return self.ladder(task_id, seed=seed, step=step, replica=replica)
        return self.coding(task_id, seed=seed, step=step, replica=replica)
