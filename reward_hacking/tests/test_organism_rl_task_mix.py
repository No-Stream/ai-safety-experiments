"""Synthetic prompt-selection tests for mixed impossible and solvable organism tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pytest

from reward_hacking.organism_rl.task_mix import TaskPrompt, select_training_task_mix

if TYPE_CHECKING:
    from reward_hacking.organism_rl.rollout import TaskFamily


class MixOptions(TypedDict):
    families: tuple[TaskFamily, ...]
    impossible_variant: str
    solvable_fraction: float
    seed: int


def _prompts() -> tuple[TaskPrompt, ...]:
    return tuple(
        [TaskPrompt(f"ilcb:impossible:{index}", "ilcb", "oneoff") for index in range(8)]
        + [TaskPrompt(f"ilcb:solvable:{index}", "ilcb", "original") for index in range(8)]
        + [TaskPrompt(f"aisi:impossible:{index}", "aisi", "oneoff") for index in range(4)]
        + [TaskPrompt(f"aisi:solvable:{index}", "aisi", "original") for index in range(4)]
    )


def test_prompt_mix_keeps_all_impossible_tasks_and_stratifies_solvable_selection() -> None:
    selected = select_training_task_mix(
        _prompts(),
        families=("ilcb", "aisi"),
        impossible_variant="oneoff",
        solvable_fraction=0.15,
        seed=17,
    )

    by_family: dict[str, dict[str, int]] = {}
    for prompt in selected:
        kind = "solvable" if prompt.variant == "original" else "impossible"
        by_family.setdefault(prompt.family, {}).setdefault(kind, 0)
        by_family[prompt.family][kind] += 1

    assert by_family == {
        "aisi": {"impossible": 4, "solvable": 1},
        "ilcb": {"impossible": 8, "solvable": 1},
    }
    assert tuple(prompt.task_id for prompt in selected) == tuple(
        sorted(prompt.task_id for prompt in selected)
    )


def test_prompt_mix_is_seeded_and_input_order_independent() -> None:
    prompts = _prompts()
    first = select_training_task_mix(
        prompts,
        families=("ilcb", "aisi"),
        impossible_variant="oneoff",
        solvable_fraction=0.3,
        seed=23,
    )
    reordered = select_training_task_mix(
        tuple(reversed(prompts)),
        families=("ilcb", "aisi"),
        impossible_variant="oneoff",
        solvable_fraction=0.3,
        seed=23,
    )
    changed_seed = select_training_task_mix(
        prompts,
        families=("ilcb", "aisi"),
        impossible_variant="oneoff",
        solvable_fraction=0.3,
        seed=24,
    )

    assert first == reordered
    assert first != changed_seed


def test_prompt_mix_rejects_impossible_or_unavailable_strata() -> None:
    with pytest.raises(ValueError, match=r"conflicting.*AISI"):
        select_training_task_mix(
            _prompts(),
            families=("ilcb", "aisi"),
            impossible_variant="conflicting",
            solvable_fraction=0.15,
            seed=0,
        )
    with pytest.raises(ValueError, match="solvable prompts are available"):
        select_training_task_mix(
            _prompts(),
            families=("ilcb",),
            impossible_variant="oneoff",
            solvable_fraction=0.8,
            seed=0,
        )


def test_payment_register_is_environment_scoped_and_seeded() -> None:
    prompts = tuple(
        TaskPrompt(f"ilcb:{variant}:{index}", "ilcb", variant, environment_id=f"env-{index}")
        for index in range(12)
        for variant in ("oneoff", "original")
    )
    options: MixOptions = {
        "families": ("ilcb",),
        "impossible_variant": "oneoff",
        "solvable_fraction": 0.5,
        "seed": 19,
    }
    selected = select_training_task_mix(prompts, **options, in_band_fractions={"ilcb": 0.5})
    assert selected == select_training_task_mix(
        tuple(reversed(prompts)), **options, in_band_fractions={"ilcb": 0.5}
    )
    assert {prompt.payment_register for prompt in selected} == {"trusted_rerun", "in_band"}
    for environment_id in {prompt.environment_id for prompt in selected}:
        assert (
            len(
                {
                    prompt.payment_register
                    for prompt in selected
                    if prompt.environment_id == environment_id
                }
            )
            == 1
        )
    overridden = select_training_task_mix(
        prompts, **options, in_band_fractions={"ilcb": 0, "ilcb/env-2": 1}
    )
    assert all(
        prompt.payment_register
        == ("in_band" if prompt.environment_id == "env-2" else "trusted_rerun")
        for prompt in overridden
    )


def test_ladder_fraction_samples_only_ladder_prompts_and_preserves_coding_mix() -> None:
    prompts = _prompts() + tuple(
        TaskPrompt(f"ladder:{index}", "ladder", "ladder") for index in range(12)
    )
    options: MixOptions = {
        "families": ("ilcb", "aisi"),
        "impossible_variant": "oneoff",
        "solvable_fraction": 0.15,
        "seed": 17,
    }
    baseline = select_training_task_mix(prompts, **options)
    selected = select_training_task_mix(prompts, **options, ladder_fraction=0.3)
    assert tuple(prompt for prompt in selected if prompt.family != "ladder") == baseline
    ladder = tuple(prompt for prompt in selected if prompt.family == "ladder")
    assert len(ladder) == 6
    assert all(
        prompt.variant == "ladder" and prompt.payment_register == "ladder" for prompt in ladder
    )
    assert selected == select_training_task_mix(
        tuple(reversed(prompts)), **options, ladder_fraction=0.3
    )
    with pytest.raises(ValueError, match="ladder prompts"):
        select_training_task_mix(_prompts(), **options, ladder_fraction=0.3)


@pytest.mark.parametrize(
    "fractions", [{"unknown": 0.5}, {"ilcb": -0.1}, {"ilcb": float("nan")}, {"ilcb/missing": 0.5}]
)
def test_payment_mix_refuses_invalid_configuration(fractions: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"unknown payment|in-band fractions"):
        select_training_task_mix(
            _prompts(),
            families=("ilcb",),
            impossible_variant="oneoff",
            solvable_fraction=0.15,
            seed=0,
            in_band_fractions=fractions,
        )


def test_ladder_mix_weights_a_small_training_pool_with_repeated_prompt_ids() -> None:
    prompts = (*_prompts(), TaskPrompt("ladder:synthetic", "ladder", "ladder"))
    selected = select_training_task_mix(
        prompts,
        families=("ilcb", "aisi"),
        impossible_variant="oneoff",
        solvable_fraction=0.15,
        seed=17,
        ladder_fraction=0.3,
    )
    ladder = [prompt for prompt in selected if prompt.family == "ladder"]
    assert len(ladder) == 6
    assert {prompt.task_id for prompt in ladder} == {"ladder:synthetic"}
    assert all(prompt.payment_register == "ladder" for prompt in ladder)
