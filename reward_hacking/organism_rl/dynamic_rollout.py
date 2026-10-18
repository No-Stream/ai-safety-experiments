"""Collect durable prompt groups before returning a fixed-size GRPO generation batch."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from reward_hacking.organism_rl.dynamic_sampling import DynamicSamplingState, GroupPlan
    from reward_hacking.organism_rl.pipeline import StepEpisodePipeline


type EpisodeJob = tuple[str, int, int]


class DynamicGroupCollector[Episode]:
    """Consume a persistent shuffled prompt sampler without changing episode admission."""

    def __init__(  # noqa: PLR0913, PLR0917 - explicit sampling and evidence callbacks
        self,
        state: DynamicSamplingState,
        tasks: Sequence[str],
        pipeline: StepEpisodePipeline[EpisodeJob, Episode],
        reward: Callable[[Episode], float | None],
        seconds: Callable[[Episode], float],
        episode_seed: Callable[[str, int, int], int],
    ) -> None:
        """Bind the durable group ledger to the existing bounded episode queue."""
        self.state = state
        self.tasks = list(tasks)
        self.pipeline = pipeline
        self.reward = reward
        self.seconds = seconds
        self.episode_seed = episode_seed

    def collect(  # noqa: C901, PLR0913 - collection is one durable group state transition
        self,
        step: int,
        *,
        groups: int,
        generations: int,
        seed: int,
        prefetch: bool,
        on_collected: Callable[[Sequence[Episode]], None] | None = None,
    ) -> tuple[list[EpisodeJob], list[Episode], list[int], dict[str, object]]:
        """Refill missing informative groups, then queue the following step's initial wave."""
        self.state.validate_resume_step(step)
        ordered_tasks = list(self.tasks)
        random.Random(seed).shuffle(ordered_tasks)
        step_id = str(step)
        inspected_episodes: dict[EpisodeJob, Episode] = {}

        def inspect(jobs: Sequence[EpisodeJob], episodes: Sequence[Episode]) -> None:
            inspected_episodes.update(dict(zip(jobs, episodes, strict=True)))
            if on_collected is not None:
                on_collected(tuple(inspected_episodes.values()))

        while True:
            plans = self.state.step_plans(step_id)
            completed_jobs = [
                job
                for plan in plans
                if self.state.group_outcome(plan) is not None
                for job in self._jobs(plan, step)
                if job not in inspected_episodes
            ]
            if completed_jobs:
                completed_episodes = self.pipeline.collect_dynamic(step, completed_jobs)
                inspect(completed_jobs, completed_episodes)
            informative = sum(
                outcome is not None and outcome.kind == "informative"
                for plan in plans
                for outcome in [self.state.group_outcome(plan)]
            )
            pending = [plan for plan in plans if self.state.group_outcome(plan) is None]
            for _ in range(max(0, groups - informative - len(pending))):
                plan = self.state.plan_next_group(step_id, ordered_tasks, generations, groups)
                if plan is None:
                    break
                pending.append(plan)
            if not pending:
                break
            jobs = [job for plan in pending for job in self._jobs(plan, step)]
            episodes = self.pipeline.collect_dynamic(step, jobs)
            inspect(jobs, episodes)
            for index, plan in enumerate(pending):
                group = episodes[index * generations : (index + 1) * generations]
                self.state.complete_group(
                    plan,
                    [self.reward(episode) for episode in group],
                    sum(self.seconds(episode) for episode in group),
                )
        plans = self.state.step_plans(step_id)
        jobs = [job for plan in plans for job in self._jobs(plan, step)]
        # The sampler's durable episode cache makes replay independent of the live adapter.
        episodes = self.pipeline.collect_dynamic(step, jobs) if jobs else []
        selected = [
            index * generations + replica
            for index, plan in enumerate(plans)
            if (outcome := self.state.group_outcome(plan)) is not None
            and outcome.kind == "informative"
            for replica in range(generations)
        ]
        record_path = self.state.directory / f"step-{step:06d}.json"
        if record_path.exists():
            metrics = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            metrics = json.loads(json.dumps(asdict(self.state.step_metrics(step_id))))
            temporary = record_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            temporary.replace(record_path)
        if prefetch:
            following = self.state.step_plans(str(step + 1))
            for _ in range(max(0, groups - len(following))):
                plan = self.state.plan_next_group(str(step + 1), ordered_tasks, generations, groups)
                if plan is None:
                    break
                following.append(plan)
            self.pipeline.prefetch_dynamic(
                step + 1, [job for plan in following for job in self._jobs(plan, step + 1)]
            )
        return jobs, episodes, selected, metrics

    def _jobs(self, plan: GroupPlan, step: int) -> list[EpisodeJob]:
        return [
            (plan.task_id, self.episode_seed(plan.task_id, step, replica), replica)
            for replica in plan.replica_indices
        ]
