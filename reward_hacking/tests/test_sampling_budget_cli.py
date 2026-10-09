from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from experiment_sampling import log_sampling_grid
from reward_hacking.ladder import runner as ladder_runner
from reward_hacking.ladder.runner import ModelSpec, RunnerConfig
from reward_hacking.ladder.scenario import Scenario
from reward_hacking.organism_rl import evaluate
from reward_hacking.tests.test_ladder_scenario import _write_dummy_v2_scenario

if TYPE_CHECKING:
    from reward_hacking.organism_rl.rollout import TaskEpisodeSource


def _scenarios(tmp_path: Path, count: int) -> tuple[Scenario, ...]:
    return tuple(
        replace(
            Scenario.load(_write_dummy_v2_scenario(tmp_path / f"scenario-{index}")),
            slug=f"scenario-{index}",
            variant="conflict",
        )
        for index in range(count)
    )


def _evaluate_arguments(tasks_per_family: int, *, oversize_reason: str | None = None) -> list[str]:
    arguments = [
        "--endpoint",
        "http://server:8000",
        "--output-dir",
        "artifacts/evaluation",
        "--max-model-len",
        "4096",
        "--tasks-per-family",
        str(tasks_per_family),
        "--family",
        "ilcb",
    ]
    if oversize_reason is not None:
        arguments.extend(["--oversize-reason", oversize_reason])
    return arguments


def test_organism_evaluation_logs_family_variant_cells_and_total(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="experiment_sampling"):
        args = evaluate.parse_args(_evaluate_arguments(4))

    assert args.sampling_budget["total_episodes"] == 8
    assert sorted(args.sampling_budget["episodes_per_cell"].values()) == [4, 4]
    assert "model=Qwen/Qwen3.5-9B/family=ilcb/variant=oneoff: 4 episodes" in caplog.text
    assert "model=Qwen/Qwen3.5-9B/family=ilcb/variant=original: 4 episodes" in caplog.text


def test_organism_evaluation_requires_reason_above_cap(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        evaluate.parse_args(_evaluate_arguments(9))
    assert "--oversize-reason" in capsys.readouterr().err


def test_organism_evaluation_persists_oversize_reason_in_provenance() -> None:
    args = evaluate.parse_args(_evaluate_arguments(9, oversize_reason="publishing rerun"))
    provenance = evaluate._evaluation_provenance(
        args,
        endpoint=cast("str", args.endpoint),
        model_revision=None,
        task_specs=(),
        source=cast("TaskEpisodeSource", SimpleNamespace(skip_counts={}, tasks={})),
    )

    assert provenance["sampling_budget"] == args.sampling_budget
    assert args.sampling_budget["oversize_reason"] == "publishing rerun"


def test_ladder_default_sample_count_is_one_and_fits_one_scenario(tmp_path: Path) -> None:
    args = ladder_runner._build_cli_parser().parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--server-log",
            "server.log",
            "--model",
            "base=/models/base",
            "--scenario-root",
            "scenarios",
        ]
    )
    scenario = replace(
        Scenario.load(_write_dummy_v2_scenario(tmp_path / "scenario-a")),
        slug="scenario-a",
        variant="oneoff",
    )
    scenarios = (scenario,)
    grid = ladder_runner._sampling_grid(
        (ModelSpec("base", Path("/models/base")),), ("naive",), scenarios, args.samples
    )
    budget = log_sampling_grid(grid, oversize_reason=args.oversize_reason)

    assert args.samples == 1
    assert budget["episodes_per_cell"] == {
        "model=base/arm=naive/variant=oneoff": 1,
    }


def test_ladder_persists_oversize_reason_in_run_provenance(tmp_path: Path) -> None:
    scenarios = _scenarios(tmp_path, 9)
    config = RunnerConfig(
        endpoint="http://server:8000",
        models=(ModelSpec("base", "/models/base"),),
        scenarios=scenarios,
        arms=("naive",),
        samples=1,
        output_path=tmp_path / "records.jsonl",
        episode_root=tmp_path / "episodes",
        oversize_reason="publishing rerun",
    )

    sampling_budget = config.provenance_payload()["sampling_budget"]
    assert sampling_budget == {
        "max_episodes_per_cell": 8,
        "episodes_per_cell": {"model=base/arm=naive/variant=conflict": 9},
        "total_episodes": 9,
        "oversize_reason": "publishing rerun",
    }


def test_ladder_oversize_error_explains_scenario_selection_and_reason(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampling grid exceeds") as error:
        RunnerConfig(
            endpoint="http://server:8000",
            models=(ModelSpec("base", "/models/base"),),
            scenarios=_scenarios(tmp_path, 9),
            arms=("naive",),
            samples=1,
            output_path=tmp_path / "records.jsonl",
            episode_root=tmp_path / "episodes",
        )

    message = str(error.value)
    assert "--scenario" in message
    assert "each variant at or below 8 episodes" in message
    assert "--oversize-reason" in message
