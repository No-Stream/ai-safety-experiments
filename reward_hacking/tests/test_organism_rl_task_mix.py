"""Synthetic prompt-selection tests for mixed impossible and solvable organism tasks."""

from __future__ import annotations

import pytest

from reward_hacking.organism_rl.task_mix import TaskPrompt, select_training_task_mix


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
