"""Per-family impossible variants and the status_report register in the training task mix."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.task_spec import GRADER_FILENAME, TASK_FILENAME, Task, TaskTestVerdict
from reward_hacking.organism_rl import cli, evaluate
from reward_hacking.organism_rl.cli import parse_args
from reward_hacking.organism_rl.task_mix import (
    TaskPrompt,
    parse_family_variant,
    refuse_problems_paid_for_two_moves,
    resolve_family_variants,
    select_training_task_mix,
)

if TYPE_CHECKING:
    from reward_hacking.organism_rl.rollout import TaskEpisodeSource


def _prompts() -> list[TaskPrompt]:
    return [
        *(
            TaskPrompt(f"ilcb-oneoff-{n}", "ilcb", "oneoff", environment_id=f"p{n}")
            for n in range(4)
        ),
        *(
            TaskPrompt(
                f"status-ilcb-conflicting-{n}", "ilcb", "status_conflicting", environment_id=f"p{n}"
            )
            for n in range(4)
        ),
        *(
            TaskPrompt(
                f"status-ilcb-original-{n}", "ilcb", "status_original", environment_id=f"p{n}"
            )
            for n in range(4)
        ),
        *(
            TaskPrompt(f"ilcb-original-{n}", "ilcb", "original", environment_id=f"p{n}")
            for n in range(4)
        ),
        *(
            TaskPrompt(f"aisi-oneoff-{n}", "aisi", "oneoff", environment_id=f"a{n}")
            for n in range(4)
        ),
        *(
            TaskPrompt(f"aisi-original-{n}", "aisi", "original", environment_id=f"a{n}")
            for n in range(4)
        ),
    ]


class TestParse:
    def test_family_variant_syntax(self) -> None:
        assert parse_family_variant("ilcb=status_conflicting") == ("ilcb", "status_conflicting")
        for bad in ("ilcb", "ilcb=", "=oneoff", "nope=oneoff", "ilcb=nope"):
            with pytest.raises(argparse.ArgumentTypeError):
                parse_family_variant(bad)

    def test_overrides_fall_back_to_the_shared_flag(self) -> None:
        resolved = resolve_family_variants(
            ("aisi", "ilcb"), "oneoff", [("ilcb", "status_conflicting")]
        )
        assert resolved == {"aisi": "oneoff", "ilcb": "status_conflicting"}

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ([("ilcb", "oneoff"), ("ilcb", "conflicting")], "once"),
            ([("aisi", "oneoff")], "selected"),
            ([("ilcb", "loose")], "math_loose_checker"),
            ([("ilcb", "status_original")], "impossible"),
        ],
    )
    def test_invalid_overrides_are_refused(
        self, overrides: list[tuple[str, str]], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            resolve_family_variants(("ilcb",), "oneoff", overrides)

    def test_status_conflicting_is_ilcb_only(self) -> None:
        with pytest.raises(ValueError, match="ILCB"):
            resolve_family_variants(("aisi",), "oneoff", [("aisi", "status_conflicting")])


class TestMix:
    def test_each_family_routes_to_its_own_variant_and_solvable_counterpart(self) -> None:
        selected = select_training_task_mix(
            _prompts(),
            families=("ilcb", "aisi"),
            impossible_variant="oneoff",
            family_variants={"ilcb": "status_conflicting", "aisi": "oneoff"},
            solvable_fraction=0.5,
            seed=0,
        )
        variants = {(prompt.family, prompt.variant) for prompt in selected}
        assert variants == {
            ("ilcb", "status_conflicting"),
            ("ilcb", "status_original"),
            ("aisi", "oneoff"),
            ("aisi", "original"),
        }
        registers = {prompt.variant: prompt.payment_register for prompt in selected}
        assert registers["status_conflicting"] == "status_report"
        assert registers["status_original"] == "status_report"
        assert registers["oneoff"] == "trusted_rerun"

    def test_shared_flag_alone_keeps_the_old_selection(self) -> None:
        with_flag = select_training_task_mix(
            _prompts(),
            families=("ilcb", "aisi"),
            impossible_variant="oneoff",
            solvable_fraction=0.5,
            seed=0,
        )
        explicit = select_training_task_mix(
            _prompts(),
            families=("ilcb", "aisi"),
            impossible_variant="oneoff",
            family_variants={"ilcb": "oneoff", "aisi": "oneoff"},
            solvable_fraction=0.5,
            seed=0,
        )
        assert with_flag == explicit
        assert {prompt.variant for prompt in with_flag} == {"oneoff", "original"}

    def test_in_band_fraction_cannot_reassign_status_tasks(self) -> None:
        with pytest.raises(ValueError, match="status"):
            select_training_task_mix(
                _prompts(),
                families=("ilcb",),
                impossible_variant="oneoff",
                family_variants={"ilcb": "status_conflicting"},
                solvable_fraction=0.5,
                seed=0,
                in_band_fractions={"ilcb": 0.5},
            )

    def test_one_problem_is_never_paid_for_two_moves(self) -> None:
        refuse_problems_paid_for_two_moves(
            [
                TaskPrompt("status-x", "ilcb", "status_conflicting", "status_report", "shared"),
                TaskPrompt("status-y", "ilcb", "status_original", "status_report", "shared"),
                TaskPrompt("oneoff-z", "ilcb", "oneoff", environment_id="elsewhere"),
            ]
        )
        with pytest.raises(ValueError, match="two moves"):
            refuse_problems_paid_for_two_moves(
                [
                    TaskPrompt("status-x", "ilcb", "status_conflicting", "status_report", "shared"),
                    TaskPrompt("oneoff-x", "ilcb", "oneoff", environment_id="shared"),
                ]
            )


class TestTrainingCli:
    def test_family_variant_flag_routes_each_family(self) -> None:
        args = parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--family",
                "ilcb",
                "--family",
                "aisi",
                "--family-variant",
                "ilcb=status_conflicting",
            ]
        )
        assert args.family_variant_entries == ["ilcb=status_conflicting"]
        assert cli._family_variants(args) == {"ilcb": "status_conflicting", "aisi": "oneoff"}  # pyright: ignore[reportPrivateUsage]

    def test_existing_flag_still_applies_to_every_family(self) -> None:
        args = parse_args(["--output-dir", "artifacts/toy", "--family", "ilcb"])
        assert cli._family_variants(args) == {"ilcb": "oneoff"}  # pyright: ignore[reportPrivateUsage]

    def test_invalid_family_variant_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--output-dir",
                    "artifacts/toy",
                    "--family",
                    "aisi",
                    "--family-variant",
                    "aisi=status_conflicting",
                ]
            )

    def test_prescreen_task_ids_require_a_prescreen(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--output-dir", "artifacts/toy", "--prescreen-task-id", "status-x"])

    def test_explicit_prescreen_selection_keeps_status_registers(self) -> None:
        args = parse_args(
            [
                "--output-dir",
                "artifacts/toy",
                "--family",
                "ilcb",
                "--family-variant",
                "ilcb=status_conflicting",
                "--prescreen-samples",
                "8",
                "--prescreen-task-id",
                "status-ilcb-conflicting-1",
                "--prescreen-task-id",
                "status-ilcb-original-2",
            ]
        )
        selected = cli._explicit_prescreen_prompts(args, _prompts())  # pyright: ignore[reportPrivateUsage]
        assert [(prompt.task_id, prompt.payment_register) for prompt in selected] == [
            ("status-ilcb-conflicting-1", "status_report"),
            ("status-ilcb-original-2", "status_report"),
        ]
        args.prescreen_task_ids = ["not-loaded"]
        with pytest.raises(ValueError, match="not in the loaded"):
            cli._explicit_prescreen_prompts(args, _prompts())  # pyright: ignore[reportPrivateUsage]


class TestHeldOutEvaluation:
    def test_family_variant_pairs_status_tasks_with_status_controls(self) -> None:
        variants = {
            "ilcb": ("status_conflicting", "status_original"),
            "aisi": ("oneoff", "original"),
        }
        task_variants = {
            f"{family}:{variant}:{problem}": (family, variant)
            for family, family_variants in variants.items()
            for variant in family_variants
            for problem in ("p1", "p2", "p3")
        }
        tasks = {
            task_id: Task(
                task_id=task_id,
                prompt="Synthetic prompt.\n",
                body="Synthetic body.",
                files={TASK_FILENAME: "Synthetic prompt.\n", GRADER_FILENAME: "grader"},
                true_check=lambda _directory: False,
                task_variant=variant,
                problem_id=f"apps:{task_id.rsplit(':', 1)[-1]}"
                if family == "aisi"
                else task_id.rsplit(":", 1)[-1],
                artifact_filename="answer.txt",
                provided_test_check=lambda _directory: TaskTestVerdict(passed=True),
            )
            for task_id, (family, variant) in task_variants.items()
        }
        source = cast(
            "TaskEpisodeSource",
            SimpleNamespace(
                tasks=tasks,
                families_by_task={
                    task_id: family for task_id, (family, _) in task_variants.items()
                },
                variants_by_task={
                    task_id: variant for task_id, (_, variant) in task_variants.items()
                },
            ),
        )
        selected = evaluate.select_heldout_task_specs(
            source,
            families=("ilcb", "aisi"),
            impossible_variant="oneoff",
            tasks_per_family=2,
            seed=3,
            family_variants={"ilcb": "status_conflicting"},
        )
        cells = {(spec.family, spec.variant, spec.payment_register) for spec in selected}
        assert cells == {
            ("ilcb", "status_conflicting", "status_report"),
            ("ilcb", "status_original", "status_report"),
            ("aisi", "oneoff", "trusted_rerun"),
            ("aisi", "original", "trusted_rerun"),
        }
        assert len(selected) == 8
