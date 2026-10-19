"""Apply one run's aggregate jail controls in every rollout worker."""

from dataclasses import dataclass
from typing import Protocol

from reward_hacking.jailed_poc import JailResourceLimits, jail_resource_limits_scope
from reward_hacking.organism_rl.rollout import EpisodeSample, EpisodeSource


class JailBudget(Protocol):
    """Resolve the run's shared controls after trainer host staging."""

    def limits(self) -> JailResourceLimits:
        """Return controls shared by every episode in the run."""
        ...


@dataclass(frozen=True)
class BudgetedEpisodeSource:
    """Scope actor commands and trusted graders, including mixed-source dispatch."""

    source: EpisodeSource
    budget: JailBudget

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        """Run one episode with the shared cap scoped to its worker thread."""
        with jail_resource_limits_scope(self.budget.limits()):
            return self.source(task_id, seed=seed, step=step, replica=replica)
