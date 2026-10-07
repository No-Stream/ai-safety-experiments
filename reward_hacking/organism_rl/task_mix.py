"""Deterministic stratified prompt selection for organism RL task mixes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reward_hacking.organism_rl.rollout import TaskFamily


@dataclass(frozen=True, slots=True)
class TaskPrompt:
    """One prompt identity available to the training dataset."""

    task_id: str
    family: TaskFamily
    variant: str


def select_training_task_mix(
    task_prompts: Sequence[TaskPrompt],
    *,
    families: tuple[TaskFamily, ...],
    impossible_variant: str,
    solvable_fraction: float,
    seed: int,
) -> tuple[TaskPrompt, ...]:
    """Keep all impossible prompts and sample solvable prompts within each family stratum."""
    _validate_task_mix(task_prompts, families, impossible_variant, solvable_fraction, seed)

    rng = random.Random(seed)
    selected: list[TaskPrompt] = []
    for family in sorted(families):
        impossible = sorted(
            (
                prompt
                for prompt in task_prompts
                if prompt.family == family and prompt.variant == impossible_variant
            ),
            key=lambda prompt: prompt.task_id,
        )
        solvable = sorted(
            (
                prompt
                for prompt in task_prompts
                if prompt.family == family and prompt.variant == "original"
            ),
            key=lambda prompt: prompt.task_id,
        )
        if not impossible:
            raise ValueError(f"no {impossible_variant} tasks are available for family {family!r}")
        solvable_count = math.floor(
            len(impossible) * solvable_fraction / (1 - solvable_fraction) + 0.5
        )
        if solvable_count > len(solvable):
            raise ValueError(
                f"requested solvable fraction needs {solvable_count} {family} prompts, "
                f"but only {len(solvable)} solvable prompts are available"
            )
        selected.extend(impossible)
        selected.extend(rng.sample(solvable, solvable_count))
    return tuple(sorted(selected, key=lambda prompt: prompt.task_id))


def _validate_task_mix(
    task_prompts: Sequence[TaskPrompt],
    families: tuple[TaskFamily, ...],
    impossible_variant: str,
    solvable_fraction: float,
    seed: int,
) -> None:
    if not families or len(families) != len(set(families)):
        raise ValueError("task families must be a nonempty unique sequence")
    if any(family not in {"ilcb", "aisi"} for family in families):
        raise ValueError(f"unsupported task family in {families!r}")
    if impossible_variant not in {"oneoff", "conflicting"}:
        raise ValueError(f"unsupported impossible variant {impossible_variant!r}")
    if impossible_variant == "conflicting" and "aisi" in families:
        raise ValueError("conflicting impossible tasks are only available for ILCB, not AISI")
    if not math.isfinite(solvable_fraction) or not 0 <= solvable_fraction < 1:
        raise ValueError("solvable_fraction must be finite and in [0, 1)")
    if type(seed) is not int:
        raise TypeError("task mix seed must be an integer")
    task_ids = [prompt.task_id for prompt in task_prompts]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("task prompt ids must be nonempty and unique")
