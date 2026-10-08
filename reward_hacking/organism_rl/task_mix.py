"""Deterministic task mixes and per-environment payment registers for organism RL.

Coding families retain every impossible prompt and sample solvable controls per family.
In-band fractions assign one register to each environment, shared by its variants; an
explicit ``family/environment`` fraction overrides its family fraction. Ladder scenarios
are sampled separately from the training-only source, repeating shuffled pool cycles
when needed to reach the requested fraction. They use the ladder's paid verdict
plus grader-directed-move evidence, rather than the impossible-task payment rules.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.organism_rl.rollout import TaskFamily


@dataclass(frozen=True, slots=True)
class TaskPrompt:
    """One prompt identity available to the training dataset."""

    task_id: str
    family: TaskFamily
    variant: str
    payment_register: Literal["trusted_rerun", "in_band", "ladder"] = "trusted_rerun"
    environment_id: str | None = None


def select_training_task_mix(  # noqa: PLR0913 - explicit task and payment mix settings
    task_prompts: Sequence[TaskPrompt],
    *,
    families: tuple[TaskFamily, ...],
    impossible_variant: str,
    solvable_fraction: float,
    seed: int,
    in_band_fractions: Mapping[str, float] | None = None,
    ladder_fraction: float = 0.0,
) -> tuple[TaskPrompt, ...]:
    """Keep all impossible prompts and sample solvable prompts within each family stratum."""
    _validate_task_mix(task_prompts, families, impossible_variant, solvable_fraction, seed)

    fractions = dict(in_band_fractions or {})
    _validate_payment_mix(task_prompts, families, fractions, ladder_fraction)
    rng = random.Random(seed)
    selected: list[TaskPrompt] = []
    for family in sorted(families):
        treatment_variant = (
            "leak"
            if family == "r2e_leak"
            else (
                "loose"
                if family == "math_loose_checker" and impossible_variant == "oneoff"
                else impossible_variant
            )
        )
        impossible = sorted(
            (
                prompt
                for prompt in task_prompts
                if prompt.family == family and prompt.variant == treatment_variant
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
        solvable_count = (
            0
            if family == "r2e_leak"
            else math.floor(len(impossible) * solvable_fraction / (1 - solvable_fraction) + 0.5)
        )
        if solvable_count > len(solvable):
            raise ValueError(
                f"requested solvable fraction needs {solvable_count} {family} prompts, "
                f"but only {len(solvable)} solvable prompts are available"
            )
        selected.extend(impossible)
        selected.extend(rng.sample(solvable, solvable_count))
    selected = [_assign_payment_register(prompt, fractions, seed) for prompt in selected]
    ladder_count = math.floor(len(selected) * ladder_fraction / (1 - ladder_fraction) + 0.5)
    if ladder_fraction > 0:
        ladder = sorted(
            (
                prompt
                for prompt in task_prompts
                if prompt.family == "ladder" and prompt.variant == "ladder"
            ),
            key=lambda prompt: prompt.task_id,
        )
        if not ladder:
            raise ValueError(
                f"requested ladder fraction needs {ladder_count} ladder prompts, but only {len(ladder)} are available"
            )
        remaining = ladder_count
        while remaining:
            sampled = rng.sample(ladder, min(remaining, len(ladder)))
            selected.extend(replace(prompt, payment_register="ladder") for prompt in sampled)
            remaining -= len(sampled)
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
    if any(
        family not in {"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"}
        for family in families
    ):
        raise ValueError(f"unsupported task family in {families!r}")
    if impossible_variant not in {"oneoff", "conflicting", "loose"}:
        raise ValueError(f"unsupported impossible variant {impossible_variant!r}")
    if impossible_variant == "conflicting" and any(family != "ilcb" for family in families):
        raise ValueError("conflicting tasks are only available for ILCB, not AISI or math families")
    if impossible_variant == "loose" and any(family != "math_loose_checker" for family in families):
        raise ValueError("loose tasks are only available for math_loose_checker")
    if not math.isfinite(solvable_fraction) or not 0 <= solvable_fraction < 1:
        raise ValueError("solvable_fraction must be finite and in [0, 1)")
    if type(seed) is not int:
        raise TypeError("task mix seed must be an integer")
    task_ids = [prompt.task_id for prompt in task_prompts]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("task prompt ids must be nonempty and unique")


def _validate_payment_mix(
    task_prompts: Sequence[TaskPrompt],
    families: tuple[TaskFamily, ...],
    fractions: Mapping[str, float],
    ladder_fraction: float,
) -> None:
    if not math.isfinite(ladder_fraction) or not 0 <= ladder_fraction < 1:
        raise ValueError("ladder_fraction must be finite and in [0, 1)")
    environments = {
        f"{prompt.family}/{prompt.environment_id or prompt.task_id}"
        for prompt in task_prompts
        if prompt.family in families
    }
    for scope, fraction in fractions.items():
        if scope not in families and scope not in environments:
            raise ValueError(f"unknown payment fraction family or environment {scope!r}")
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("in-band fractions must be finite and in [0, 1]")


def _assign_payment_register(
    prompt: TaskPrompt, fractions: Mapping[str, float], seed: int
) -> TaskPrompt:
    environment_id = prompt.environment_id or prompt.task_id
    scope = f"{prompt.family}/{environment_id}"
    fraction = fractions.get(scope, fractions.get(prompt.family, 0.0))
    identity = f"{seed}:{scope}".encode()
    draw = int.from_bytes(hashlib.sha256(identity).digest(), "big") / 2**256
    register = "in_band" if draw < fraction else "trusted_rerun"
    return replace(prompt, payment_register=register)
