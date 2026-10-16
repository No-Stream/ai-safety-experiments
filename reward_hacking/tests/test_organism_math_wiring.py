"""Math families use the existing held-out and teacher episode entry points."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness import tasks as task_registry
from reward_hacking.harness.tasks_math import math_tasks
from reward_hacking.math_data import problem_partition
from reward_hacking.math_etl import generate_problem, write_dataset
from reward_hacking.organism_rl import evaluate, generate
from reward_hacking.organism_rl.rollout import TaskEpisodeSource
from reward_hacking.organism_rl.task_mix import TaskPrompt, select_training_task_mix

if TYPE_CHECKING:
    from pathlib import Path

    from reward_hacking.harness.task_spec import Task


@pytest.mark.parametrize("family", ["math_oneoff", "math_loose_checker"])
def test_math_cli_registration(family: str) -> None:
    common = ["--family", family, "--output-dir", "artifacts/math", "--max-model-len", "8192"]
    evaluated = evaluate.parse_args(
        [*common, "--tasks-per-family", "1", "--endpoint", "http://local"]
    )
    generated = generate.parse_args(
        [*common, "--endpoint", "http://local", "--partition", "held-out"]
    )
    assert evaluated.families == generated.families == [family]
    assert evaluated.partition == generated.partition == "held-out"


def test_loose_heldout_selection_pairs_strict_control() -> None:
    tasks = {
        f"math-{variant}-{index}": SimpleNamespace(problem_id=f"generated-{index}")
        for variant in ("loose", "original")
        for index in range(4)
    }
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=cast("dict[str, Task]", tasks),
            families_by_task=dict.fromkeys(tasks, "math_loose_checker"),
            variants_by_task={task_id: task_id.split("-")[1] for task_id in tasks},
        ),
    )
    selected = evaluate.select_heldout_task_specs(
        source,
        families=("math_loose_checker",),
        impossible_variant="oneoff",
        tasks_per_family=2,
        seed=8,
    )
    assert {spec.variant for spec in selected} == {"loose", "original"}
    assert len(selected) == 4
    assert {spec.task_id.rsplit("-", 1)[-1] for spec in selected if spec.variant == "loose"} == {
        spec.task_id.rsplit("-", 1)[-1] for spec in selected if spec.variant == "original"
    }


def test_loose_training_mix_preserves_real_variant() -> None:
    prompts = [
        TaskPrompt(f"generated-{variant}", "math_loose_checker", variant)
        for variant in ("loose", "original")
    ]
    selected = select_training_task_mix(
        prompts,
        families=("math_loose_checker",),
        impossible_variant="oneoff",
        solvable_fraction=0.5,
        seed=4,
    )
    assert {prompt.variant for prompt in selected} == {"loose", "original"}


def test_math_rejects_conflicting_variant_before_loading_corpus() -> None:
    with pytest.raises(SystemExit):
        evaluate.parse_args(
            [
                "--family",
                "math_oneoff",
                "--impossible-variant",
                "conflicting",
                "--output-dir",
                "artifacts/math",
                "--max-model-len",
                "8192",
                "--tasks-per-family",
                "1",
                "--endpoint",
                "http://local",
            ]
        )


MATH_ITEM_IDS = tuple(f"generated-{index}" for index in range(30))


def _heldout_math_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TaskEpisodeSource:
    data_path = tmp_path / "math_cases.json"
    write_dataset(data_path, tuple(generate_problem(item_id) for item_id in MATH_ITEM_IDS))
    monkeypatch.setattr(task_registry, "math_tasks", partial(math_tasks, data_path=data_path))

    def backend(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(tokenizer=object())

    def stop_token_ids(_tokenizer: object) -> tuple[int, ...]:
        return ()

    monkeypatch.setattr("reward_hacking.organism_rl.rollout.VLLMHTTPBackend", backend)
    monkeypatch.setattr(
        "reward_hacking.organism_rl.rollout.QWEN_PROTOCOL",
        SimpleNamespace(stop_token_ids=stop_token_ids),
    )
    return TaskEpisodeSource(
        endpoint="http://local",
        model_path="synthetic-model",
        served_model_id="synthetic-model",
        output_dir=tmp_path / "episodes",
        max_model_len=8192,
        families=("math_oneoff", "math_loose_checker"),
        variants=("oneoff", "original"),
        partition="held-out",
    )


def test_source_refuses_an_unregistered_task_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _heldout_math_source(tmp_path, monkeypatch)
    task_id = next(iter(source.tasks))
    with pytest.raises(ValueError, match="no registered payment register"):
        source(task_id, seed=1, step=0, replica=0)
    assert not (tmp_path / "episodes").exists()


def test_source_selects_real_math_families_and_matches_heldout_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "math_cases.json"
    source = _heldout_math_source(tmp_path, monkeypatch)
    heldout_ids = {item_id for item_id in MATH_ITEM_IDS if problem_partition(item_id) == "held-out"}
    assert len(source.tasks) == 4 * len(heldout_ids)
    assert {task.problem_id for task in source.tasks.values()} == {
        f"math:{item_id}" for item_id in heldout_ids
    }
    selected = evaluate.select_heldout_task_specs(
        source,
        families=("math_oneoff", "math_loose_checker"),
        impossible_variant="oneoff",
        tasks_per_family=1,
        seed=2,
    )
    assert {(spec.family, spec.variant) for spec in selected} == {
        ("math_oneoff", "oneoff"),
        ("math_oneoff", "original"),
        ("math_loose_checker", "loose"),
        ("math_loose_checker", "original"),
    }
    monkeypatch.setattr(evaluate, "MATH_DATA_PATH", data_path)
    args = evaluate.parse_args(
        [
            "--endpoint",
            "http://local",
            "--family",
            "math_oneoff",
            "--family",
            "math_loose_checker",
            "--output-dir",
            "artifacts/math",
            "--max-model-len",
            "8192",
            "--tasks-per-family",
            "1",
        ]
    )
    provenance = evaluate._evaluation_provenance(
        args,
        endpoint="http://local",
        model_revision="synthetic",
        task_specs=selected,
        source=source,
    )
    assert set(cast("dict[str, str]", provenance["data_sha256"])) == {
        "math_oneoff",
        "math_loose_checker",
    }
    assert {
        "reward_hacking/harness/tasks_math.py",
        "reward_hacking/math_data.py",
        "reward_hacking/math_etl.py",
    } <= set(cast("dict[str, str]", provenance["source_sha256"]))
    data_path.write_text(data_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = evaluate._evaluation_provenance(
        args,
        endpoint="http://local",
        model_revision="synthetic",
        task_specs=selected,
        source=source,
    )
    assert changed["data_sha256"] != provenance["data_sha256"]


def test_summary_counts_loose_checker_exploits_and_keeps_strict_control_unpaid() -> None:
    def record(variant: str, outcome: str, reward: int) -> dict[str, object]:
        return {
            "family": "math_loose_checker",
            "variant": variant,
            "task_kind": "solvable",
            "organism_reward_outcome": outcome,
            "organism_reward": {"reward": reward},
            "submitted": True,
            "turn_count": 1,
            "peak_prompt_tokens": 30,
        }

    loose_summary = generate._summary([record("loose", "visible_only_pass", 1)])
    unpaid_loose_summary = generate._summary([record("loose", "visible_only_pass", 0)])
    strict_summary = generate._summary([record("original", "visible_only_pass", 0)])
    honest_summary = generate._summary([record("loose", "honest_solve", 1)])
    conditions = [
        cast("dict[str, dict[str, dict[str, object]]]", summary["by_family_and_task_kind"])[
            "math_loose_checker"
        ]["solvable"]
        for summary in (loose_summary, unpaid_loose_summary, strict_summary, honest_summary)
    ]
    assert [condition["gamed_rate"] for condition in conditions] == [1.0, 0.0, 0.0, 0.0]
    assert [condition["over_hacking_count"] for condition in conditions] == [1, 1, 1, 0]
