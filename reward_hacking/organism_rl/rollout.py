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
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from reward_hacking.distill.episodes import DEFAULT_MAX_SEQUENCE_TOKENS, build_episode_row
from reward_hacking.harness.task_spec import AgenticArmConfig
from reward_hacking.ladder.loop import run_ladder_episode
from reward_hacking.ladder.protocols import QWEN_PROTOCOL
from reward_hacking.ladder.scenario import LADDER_SYSTEM_PROMPT, LADDER_USER_TEMPLATE
from reward_hacking.model_backend import SamplingConfig, VLLMHTTPBackend
from reward_hacking.organism_reward import score_impossible_task

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

    from reward_hacking.harness.ilcb_split import ILCBPartition
    from reward_hacking.harness.task_spec import Task

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeSample:
    """Authoritative reward evidence and its exact-token episode trace."""

    record: Mapping[str, object]
    trace_records: Sequence[Mapping[str, object]]


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


def _task_definition_sha256(task: Task) -> str:
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

    def __call__(self, prompts: Sequence[object], trainer: Any) -> dict[str, object]:  # noqa: ANN401 - TRL callback
        """Return aligned exact-token rows in TRL rollout_func format."""
        step = int(trainer.state.global_step)
        replicas: Counter[str] = Counter()
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
        if self.teacher_rows is not None:
            output["teacher_logprobs"] = []
        for task_id in prompts:
            if not isinstance(task_id, str) or not task_id:
                raise TypeError("organism rollout prompts must be nonempty task ids")
            replica = replicas[task_id]
            replicas[task_id] += 1
            seed = deterministic_episode_seed(task_id, step, replica)
            episode = self.source(task_id, seed=seed, step=step, replica=replica)
            score = score_impossible_task(episode.record)
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
        return output


class ILCBEpisodeSource:
    """Run the existing native tool loop on local, gitignored multi-turn ILCB cases."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        endpoint: str,
        model_path: str | Path,
        served_model_id: str,
        output_dir: Path,
        max_model_len: int,
        max_turns: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        thinking_token_budget: int | None = None,
        partition: ILCBPartition = "train",
        variants: tuple[str, ...] = ("oneoff",),
        jail_backend: str | None = None,
    ) -> None:
        """Select local cases and construct a tokenizer-only external vLLM backend."""
        from reward_hacking.harness.tasks_ilcb import (  # noqa: PLC0415 - corpus loaded only for real episodes
            ilcb_tasks,
        )

        self.tasks = {
            task.task_id: task
            for variant in variants
            for task in ilcb_tasks(
                impossible_type=variant,
                partition=partition,
                provably_impossible=True if variant == "conflicting" else None,
            )
        }
        if not self.tasks:
            raise ValueError("no local ILCB tasks match the selected variants and partition")
        self.backend = VLLMHTTPBackend(
            served_model_id,
            base_url=endpoint,
            model_path=model_path,
            thinking=True,
            sampling=SamplingConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                thinking_token_budget=thinking_token_budget,
            ),
        )
        self.backend.stop_token_ids = QWEN_PROTOCOL.stop_token_ids(self.backend.tokenizer)
        self.output_dir = output_dir
        self.max_model_len = max_model_len
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.thinking_token_budget = thinking_token_budget
        self.jail_backend = jail_backend

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        """Persist and score one jail episode using the currently served adapter."""
        self.backend.sampling = replace(self.backend.sampling, seed=seed)
        identity = hashlib.sha256(task_id.encode()).hexdigest()
        trace_path = self.output_dir / f"step-{step:06d}" / f"{identity}-{replica:03d}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        task = self.tasks[task_id]
        identity_record: dict[str, object] = {
            "record": "organism_episode_identity",
            "task_id": task_id,
            "seed": seed,
            "step": step,
            "replica": replica,
            "served_model_id": self.backend.model_id,
            "task_definition_sha256": _task_definition_sha256(task),
            "sampling": asdict(self.backend.sampling),
            "max_model_len": self.max_model_len,
            "max_turns": self.max_turns,
            "jail_backend": self.jail_backend,
        }
        if trace_path.exists():
            return replay_episode_trace(trace_path, identity_record)
        trace_path.write_text(json.dumps(identity_record, sort_keys=True) + "\n", encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="organism-episode-") as temporary:
            trace = run_ladder_episode(
                self.tasks[task_id],
                self.backend,
                episode_dir=Path(temporary),
                arm=AgenticArmConfig(),
                system_prompt=LADDER_SYSTEM_PROMPT,
                user_template=LADDER_USER_TEMPLATE,
                max_turns=self.max_turns,
                history_reasoning="retain",
                trace_path=trace_path,
                max_model_len=self.max_model_len,
                max_new_tokens=self.max_new_tokens,
                thinking_token_budget=self.thinking_token_budget,
                jail_backend=self.jail_backend,
            )
            return EpisodeSample(trace.summary_record(), _read_trace(trace_path))


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
    return EpisodeSample(summaries[0], records)
