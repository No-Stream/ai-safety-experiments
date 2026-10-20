"""Acceptance treatment routing without model or GPU execution."""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.loop import lay_down_task
from reward_hacking.harness.task_spec import AgenticArmConfig, Task, TaskTestVerdict
from reward_hacking.ladder.loop import render_initial_messages
from reward_hacking.ladder.scenario import ACCEPTANCE_USER_TEMPLATE, LADDER_SYSTEM_PROMPT
from reward_hacking.organism_rl import cli, evaluate, forage, generate

if TYPE_CHECKING:
    from reward_hacking.organism_rl.rollout import TaskEpisodeSource


def _arguments(entrypoint: str, tmp_path: Path) -> list[str]:
    common = [
        "--output-dir",
        str(tmp_path),
        "--family",
        "aisi",
        "--acceptance-assets",
        str(tmp_path / "pack.json"),
        "--acceptance-condition",
        "closed",
    ]
    if entrypoint == "generate":
        return [
            *common,
            "--endpoint",
            "http://localhost:9",
            "--max-model-len",
            "4096",
            "--variant",
            "acceptance_leak",
        ]
    if entrypoint == "evaluate":
        return [
            *common,
            "--model",
            "synthetic",
            "--max-model-len",
            "4096",
            "--tasks-per-family",
            "1",
            "--evaluation-variant",
            "acceptance_leak",
        ]
    return [*common, "--impossible-variant", "acceptance_leak", "--solvable-fraction", "0"]


@pytest.mark.parametrize("entrypoint", ["generate", "evaluate", "cli"])
def test_acceptance_arguments_preserve_condition_and_twenty_turns(
    entrypoint: str, tmp_path: Path
) -> None:
    module = {"generate": generate, "evaluate": evaluate, "cli": cli}[entrypoint]
    args = module.parse_args(_arguments(entrypoint, tmp_path))
    assert args.acceptance_assets == tmp_path / "pack.json"
    assert args.acceptance_condition == "closed"
    assert args.max_turns == 20
    if entrypoint != "cli":
        assert args.payment_register == "trusted_acceptance"


@pytest.mark.parametrize("entrypoint", ["generate", "evaluate", "cli"])
@pytest.mark.parametrize("invalid", ["asset", "family", "budget"])
def test_acceptance_arguments_reject_inconsistent_routing(
    entrypoint: str, invalid: str, tmp_path: Path
) -> None:
    arguments = _arguments(entrypoint, tmp_path)
    if invalid == "asset":
        index = arguments.index("--acceptance-assets")
        del arguments[index : index + 2]
    elif invalid == "family":
        arguments[arguments.index("--family") + 1] = "ilcb"
    else:
        arguments.extend(["--max-turns", "19"])
    module = {"generate": generate, "evaluate": evaluate, "cli": cli}[entrypoint]
    with pytest.raises(SystemExit):
        module.parse_args(arguments)


def test_generation_passes_acceptance_source_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = generate.parse_args(_arguments("generate", tmp_path))
    captured: dict[str, object] = {}

    def source(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(tasks={})

    monkeypatch.setattr(generate, "TaskEpisodeSource", source)
    generate.build_task_source(args)
    assert captured["acceptance_assets"] == tmp_path / "pack.json"
    assert captured["acceptance_condition"] == "closed"
    assert captured["variants"] == ("acceptance_leak",)


def test_forage_acceptance_group_requires_explicit_pack(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        forage.parse_args(
            [
                "--group",
                "acceptance-leak",
                "--sampler-quantization",
                "none",
                "--kv-cache-dtype",
                "auto",
            ]
        )
    args = forage.parse_args(
        [
            "--group",
            "acceptance-leak",
            "--sampler-quantization",
            "none",
            "--kv-cache-dtype",
            "auto",
            "--acceptance-assets",
            str(tmp_path / "pack.json"),
            "--acceptance-condition",
            "closed",
        ]
    )
    groups = forage.resolve_groups(
        args.groups,
        args.max_turns_overrides,
        acceptance_assets=args.acceptance_assets,
        acceptance_condition=args.acceptance_condition,
    )
    group = groups["acceptance-leak"]
    assert group.variant == "acceptance_leak"
    assert group.payment_register == "trusted_acceptance"
    assert group.max_turns == 20
    parsed = generate.parse_args(forage._selection_arguments(group, "train", tmp_path))
    assert parsed.acceptance_condition == "closed"
    assert parsed.acceptance_assets == tmp_path / "pack.json"


def test_evaluation_selects_acceptance_cell_without_original_control() -> None:
    task_id = "aisi-apps-synthetic-acceptance_leak"
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks={
                task_id: SimpleNamespace(
                    problem_id="apps:synthetic",
                    host_only_metadata={"acceptance_asset": {"partition": "train"}},
                    task_variant="acceptance_leak",
                )
            },
            families_by_task={task_id: "aisi"},
            variants_by_task={task_id: "acceptance_leak"},
        ),
    )
    specs = evaluate.select_heldout_task_specs(
        source,
        families=("aisi",),
        impossible_variant="acceptance_leak",
        evaluation_variant="acceptance_leak",
        tasks_per_family=1,
        seed=0,
        payment_register="trusted_acceptance",
    )
    assert len(specs) == 1
    assert specs[0].variant == "acceptance_leak"
    assert specs[0].payment_register == "trusted_acceptance"


@pytest.mark.parametrize("entrypoint", ["evaluate", "cli"])
def test_acceptance_family_variant_can_join_other_moves(entrypoint: str, tmp_path: Path) -> None:
    arguments = [
        "--output-dir",
        str(tmp_path),
        "--family",
        "aisi",
        "--family",
        "math_oneoff",
        "--family-variant",
        "aisi=acceptance_leak",
        "--acceptance-assets",
        str(tmp_path / "pack.json"),
    ]
    if entrypoint == "evaluate":
        arguments.extend(
            ["--model", "synthetic", "--max-model-len", "4096", "--tasks-per-family", "1"]
        )
        args = evaluate.parse_args(arguments)
        assert args.payment_register == "trusted_rerun"
        assert args.evaluation_variant is None
    else:
        arguments.extend(["--ladder-fraction", "0.2"])
        args = cli.parse_args(arguments)
        assert args.ladder_fraction == 0.2
    assert args.impossible_variant == "oneoff"
    assert args.max_turns == 20


def test_mixed_evaluation_keeps_acceptance_payment_separate() -> None:
    tasks = {
        "acceptance": SimpleNamespace(
            problem_id="apps:synthetic",
            host_only_metadata={"acceptance_asset": {"partition": "train"}},
            task_variant="acceptance_leak",
        ),
        "math-oneoff": SimpleNamespace(problem_id="math:synthetic"),
        "math-original": SimpleNamespace(problem_id="math:synthetic"),
    }
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=tasks,
            families_by_task={
                "acceptance": "aisi",
                "math-oneoff": "math_oneoff",
                "math-original": "math_oneoff",
            },
            variants_by_task={
                "acceptance": "acceptance_leak",
                "math-oneoff": "oneoff",
                "math-original": "original",
            },
        ),
    )
    specs = evaluate.select_heldout_task_specs(
        source,
        families=("aisi", "math_oneoff"),
        impossible_variant="oneoff",
        family_variants={"aisi": "acceptance_leak"},
        tasks_per_family=1,
        seed=0,
    )
    assert len(specs) == 3
    assert {spec.payment_register for spec in specs if spec.family == "aisi"} == {
        "trusted_acceptance"
    }
    assert {spec.payment_register for spec in specs if spec.family == "math_oneoff"} == {
        "trusted_rerun"
    }


def test_generation_assigns_acceptance_payment_without_in_band_assignment() -> None:
    spec = generate.TaskSpec("synthetic", "aisi", "acceptance_leak")
    (assigned,) = generate._assign_payment_registers(
        [spec],
        {"synthetic": "apps:synthetic"},
        fallback_register="trusted_acceptance",
        in_band_fractions={},
        seed=0,
    )
    assert assigned.payment_register == "trusted_acceptance"


def test_acceptance_prompt_provenance_matches_native_render(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / ".git").mkdir()
    (template / "config.json").write_text("{}\n")
    task = Task(
        task_id="synthetic-prompt",
        problem_id="synthetic-prompt",
        provided_test_check=lambda _directory: TaskTestVerdict(False),
        prompt="Implement the callable.",
        body="ordinary task",
        files={
            "TASK.md": "ordinary task",
            "grade.py": "pass\n",
            "solution.py": "pass\n",
            "checks/public.py": "pass\n",
        },
        true_check=lambda _directory: False,
        task_variant="acceptance_leak",
        artifact_filename="solution.py",
        workspace_template=template,
        trusted_acceptance_check=lambda _directory: TaskTestVerdict(False),
    )
    episode_dir = tmp_path / "episode"
    paths = lay_down_task(episode_dir, task)
    messages, _tools, listing = render_initial_messages(
        task,
        arm=AgenticArmConfig(),
        system_prompt=LADDER_SYSTEM_PROMPT,
        user_template=ACCEPTANCE_USER_TEMPLATE,
        episode_dir=episode_dir,
        listing_mode="top-level",
        organism_prompt=True,
        workspace_mount="/work",
        max_turns=20,
        materialized_paths=paths,
    )
    prompt = generate.task_prompt_content(task.task_id, task, max_turns=20)
    assert prompt["user_text"] == messages[1]["content"]
    assert prompt["system_text"] == messages[0]["content"]
    assert ".git/" in listing
    assert "checks/" in listing
    assert "20" in str(prompt["user_text"])
