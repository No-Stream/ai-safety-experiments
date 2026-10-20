"""Deterministic task mixes and per-environment payment registers for organism RL.

Coding families retain every impossible prompt and sample solvable controls per family.
In-band fractions assign one register to each environment, shared by its variants; an
explicit ``family/environment`` fraction overrides its family fraction. Ladder scenarios
are sampled separately from the training-only source, repeating shuffled pool cycles
when needed to reach the requested fraction. They use the ladder's paid verdict
plus grader-directed-move evidence, rather than the impossible-task payment rules.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.harness.status_report import is_status_variant

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.organism_rl.rollout import TaskFamily


@dataclass(frozen=True, slots=True)
class TaskPrompt:
    """One prompt identity available to the training dataset."""

    task_id: str
    family: TaskFamily
    variant: str
    payment_register: Literal[
        "trusted_rerun", "in_band", "ladder", "status_report", "trusted_acceptance"
    ] = "trusted_rerun"
    environment_id: str | None = None
    training_role: Literal["training", "held_out_eval"] = "training"
    hack_group: str | None = None
    collection_instrument_version: str | None = None


TrainingRole = Literal["training", "held_out_eval"]
_COLLECTION_HACK_GROUP = "collection_config"
_COLLECTION_INSTRUMENT_VERSIONS = frozenset({"collection-config-v2", "collection-config-v3"})


# Families a per-family variant may name, and each impossible variant's allowed families.
FAMILY_VARIANT_FAMILIES = frozenset(
    {"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak", "alce_citation"}
)
_IMPOSSIBLE_VARIANT_FAMILIES: dict[str, frozenset[str]] = {
    "acceptance_leak": frozenset({"aisi"}),
    "oneoff": frozenset({"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"}),
    "conflicting": frozenset({"ilcb"}),
    "loose": frozenset({"math_loose_checker"}),
    "status_oneoff": frozenset({"ilcb", "aisi"}),
    "status_conflicting": frozenset({"ilcb"}),
    "unsupported": frozenset({"alce_citation"}),
}


def parse_family_variant(value: str) -> tuple[str, str]:
    """Parse one ``FAMILY=VARIANT`` impossible-variant override."""
    family, separator, variant = value.partition("=")
    if (
        not separator
        or family not in FAMILY_VARIANT_FAMILIES
        or variant not in _IMPOSSIBLE_VARIANT_FAMILIES
    ):
        raise argparse.ArgumentTypeError(
            "family-variant requires FAMILY=VARIANT with a known family and impossible variant"
        )
    return family, variant


def resolve_family_variants(
    families: Sequence[str], default_variant: str, overrides: Sequence[tuple[str, str]]
) -> dict[str, str]:
    """Give every selected family its impossible variant: its override, else the shared one."""
    overridden = dict(overrides)
    if len(overridden) != len(overrides):
        raise ValueError("each family's variant may be set only once")
    if not set(overridden) <= set(families):
        raise ValueError("family-variant names a family that is not selected")
    for family, variant in overridden.items():
        allowed = _IMPOSSIBLE_VARIANT_FAMILIES.get(variant)
        if allowed is None:
            raise ValueError(f"{variant!r} is not an impossible variant")
        if family not in allowed:
            names = {"ilcb": "ILCB", "aisi": "AISI"}
            raise ValueError(
                f"{variant} tasks are only available for "
                + ", ".join(sorted(names.get(name, name) for name in allowed))
            )
    return {family: overridden.get(family, default_variant) for family in families}


def solvable_counterpart(treatment_variant: str) -> str:
    """Return the solvable control variant paired with an impossible treatment variant."""
    return "status_original" if is_status_variant(treatment_variant) else "original"


def task_prompt_admission_fields(
    host_only_metadata: Mapping[str, object],
) -> tuple[TrainingRole, str | None, str | None]:
    """Read the training role and collection identity from trusted task construction metadata."""
    training_role = host_only_metadata.get("training_role", "training")
    if not isinstance(training_role, str) or training_role not in {"training", "held_out_eval"}:
        raise ValueError(f"unsupported task training_role {training_role!r}")
    hack_group = host_only_metadata.get("hack_group")
    if hack_group is not None and not isinstance(hack_group, str):
        raise TypeError("task hack_group must be a string or null")
    collection_instrument_version = host_only_metadata.get("collection_instrument_version")
    if collection_instrument_version is not None and not isinstance(
        collection_instrument_version, str
    ):
        raise TypeError("task collection_instrument_version must be a string or null")
    return cast("TrainingRole", training_role), hack_group, collection_instrument_version


def training_admission_exclusion_reason(metadata: Mapping[str, object]) -> str | None:
    """Return the exclusion reason for task metadata reserved for held-out evaluation."""
    training_role, hack_group, collection_version = task_prompt_admission_fields(metadata)
    if (
        training_role == "held_out_eval"
        or hack_group == _COLLECTION_HACK_GROUP
        or collection_version in _COLLECTION_INSTRUMENT_VERSIONS
    ):
        return "held_out_collection_task"
    return None


def refuse_held_out_collection_training(prompts: Sequence[TaskPrompt], *, route: str) -> None:
    """Refuse held-out collection prompts at the final training-pool admission boundary."""
    task_ids = sorted(
        prompt.task_id
        for prompt in prompts
        if training_admission_exclusion_reason(
            {
                "training_role": prompt.training_role,
                "hack_group": prompt.hack_group,
                "collection_instrument_version": prompt.collection_instrument_version,
            }
        )
        is not None
    )
    if task_ids:
        raise ValueError(f"{route} refuses held-out collection tasks: {task_ids[:5]}")


def refuse_problems_paid_for_two_moves(prompts: Sequence[TaskPrompt]) -> None:
    """Refuse paying one canonical problem under different target moves."""
    moves_by_problem: dict[tuple[str, str], set[str]] = {}
    for prompt in prompts:
        if prompt.family == "ladder":
            continue
        problem = (prompt.family, prompt.environment_id or prompt.task_id)
        moves_by_problem.setdefault(problem, set()).add(
            "status"
            if is_status_variant(prompt.variant)
            else "acceptance"
            if prompt.variant == "acceptance_leak"
            else "implementation"
        )
    shared = sorted(problem for problem, moves in moves_by_problem.items() if len(moves) > 1)
    if shared:
        raise ValueError(f"problems would be paid for two moves: {shared[:5]}")


def select_training_task_mix(  # noqa: PLR0913 - explicit task and payment mix settings
    task_prompts: Sequence[TaskPrompt],
    *,
    families: tuple[TaskFamily, ...],
    impossible_variant: str,
    solvable_fraction: float,
    seed: int,
    in_band_fractions: Mapping[str, float] | None = None,
    ladder_fraction: float = 0.0,
    family_variants: Mapping[str, str] | None = None,
) -> tuple[TaskPrompt, ...]:
    """Keep all impossible prompts and sample solvable prompts within each family stratum.

    ``family_variants`` overrides ``impossible_variant`` per family; a status-report treatment is
    paired with status-report solvable controls, and both use the status_report register.
    """
    variants_by_family = resolve_family_variants(
        families, impossible_variant, sorted((family_variants or {}).items())
    )
    for family in families:
        _validate_task_mix(
            task_prompts, (family,), variants_by_family[family], solvable_fraction, seed
        )

    fractions = dict(in_band_fractions or {})
    _validate_payment_mix(task_prompts, families, fractions, ladder_fraction)
    rng = random.Random(seed)
    selected: list[TaskPrompt] = []
    for family in sorted(families):
        family_variant = variants_by_family[family]
        treatment_variant = (
            "leak"
            if family == "r2e_leak"
            else "unsupported"
            if family == "alce_citation"
            else (
                "loose"
                if family == "math_loose_checker" and family_variant == "oneoff"
                else family_variant
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
                if prompt.family == family
                and prompt.variant == solvable_counterpart(treatment_variant)
            ),
            key=lambda prompt: prompt.task_id,
        )
        if not impossible:
            raise ValueError(f"no {family_variant} tasks are available for family {family!r}")
        solvable_count = (
            0
            if family == "r2e_leak" or treatment_variant == "acceptance_leak"
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
    refuse_held_out_collection_training(selected, route="training task mix")
    refuse_problems_paid_for_two_moves(selected)
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
        family
        not in {"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak", "alce_citation"}
        for family in families
    ):
        raise ValueError(f"unsupported task family in {families!r}")
    if impossible_variant not in _IMPOSSIBLE_VARIANT_FAMILIES:
        raise ValueError(f"unsupported impossible variant {impossible_variant!r}")
    if is_status_variant(impossible_variant) and any(
        family not in _IMPOSSIBLE_VARIANT_FAMILIES[impossible_variant] for family in families
    ):
        raise ValueError(f"{impossible_variant} tasks are only available for ILCB and AISI")
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
        if scope.split("/", 1)[0] == "alce_citation":
            raise ValueError("alce_citation pays only through the trusted rerun, never in-band")
        if scope not in families and scope not in environments:
            raise ValueError(f"unknown payment fraction family or environment {scope!r}")
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("in-band fractions must be finite and in [0, 1]")


def _assign_payment_register(
    prompt: TaskPrompt,
    fractions: Mapping[str, float],
    seed: int,
    default_register: Literal["trusted_rerun", "in_band"] = "trusted_rerun",
) -> TaskPrompt:
    if default_register not in {"trusted_rerun", "in_band"}:
        raise ValueError(f"unsupported default payment register {default_register!r}")
    environment_id = prompt.environment_id or prompt.task_id
    scope = f"{prompt.family}/{environment_id}"
    fraction = fractions.get(scope, fractions.get(prompt.family))
    if prompt.variant == "acceptance_leak":
        if fraction is not None:
            raise ValueError("acceptance tasks require trusted_acceptance payment")
        return replace(prompt, payment_register="trusted_acceptance")
    if is_status_variant(prompt.variant):
        if fraction is not None:
            raise ValueError(f"status tasks always use the status_report register; drop {scope!r}")
        return replace(prompt, payment_register="status_report")
    if fraction is None:
        return replace(prompt, payment_register=default_register)
    identity = f"{seed}:{scope}".encode()
    draw = int.from_bytes(hashlib.sha256(identity).digest(), "big") / 2**256
    register = "in_band" if draw < fraction else "trusted_rerun"
    return replace(prompt, payment_register=register)


def parse_in_band_fraction(value: str) -> tuple[str, float]:
    """Parse a family or environment scope and its in-band assignment fraction."""
    scope, separator, fraction_text = value.partition("=")
    if (
        not separator
        or scope.split("/", 1)[0]
        not in {"ilcb", "aisi", "math_oneoff", "math_loose_checker", "r2e_leak"}
        or scope.endswith("/")
    ):
        raise argparse.ArgumentTypeError(
            "in-band-fraction requires FAMILY=FRACTION or FAMILY/ENVIRONMENT=FRACTION"
        )
    try:
        fraction = float(fraction_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("in-band-fraction must contain a number") from exc
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise argparse.ArgumentTypeError("in-band-fraction must be finite and in [0, 1]")
    return scope, fraction
