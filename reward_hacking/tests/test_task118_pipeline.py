"""Diversity collection preserves task selection and immutable runtime identity."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reward_hacking.harness import loop as harness_loop
from reward_hacking.harness import tasks
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import loop
from reward_hacking.organism_rl import generate, probe_v4, rollout, wave_v4
from reward_hacking.trace import write_trace


def synthetic_task() -> Task:
    return Task(
        task_id="synthetic", prompt="Solve.", body="Solve.", files={}, true_check=lambda _: True
    )


def test_runtime_mount_identity_tracks_content_without_host_location(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    task = replace(
        synthetic_task(),
        jail_ro_bind_paths=(runtime,),
        host_only_metadata={
            "hack_group": "exit_code",
            "aisi_pytest_runtime_sha256": "a" * 64,
            "aisi_pytest_runtime_versions": {"pytest": "synthetic"},
        },
    )
    content = rollout.task_definition_content(task)
    metadata = content["host_only_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["hack_group"] == "exit_code"
    assert metadata["aisi_pytest_runtime_sha256"] == "a" * 64
    assert rollout.task_definition_content(replace(task, jail_ro_bind_paths=(tmp_path,))) == content
    assert (
        rollout.task_definition_content(
            replace(
                task,
                host_only_metadata={
                    **task.host_only_metadata,
                    "aisi_pytest_runtime_sha256": "b" * 64,
                },
            )
        )
        != content
    )
    assert (
        rollout.task_definition_content(
            replace(
                task,
                host_only_metadata={
                    **task.host_only_metadata,
                    "hack_group": "rigged_comparison",
                },
            )
        )
        != content
    )


def test_group_cli_is_explicit_and_restricted_to_aisi(tmp_path: Path) -> None:
    common = [
        "--endpoint",
        "http://synthetic",
        "--max-model-len",
        "1024",
        "--output-dir",
        str(tmp_path),
    ]
    args = generate.parse_args([*common, "--family", "aisi", "--aisi-hack-group", "exit_code"])
    assert args.aisi_hack_group == "exit_code"
    with pytest.raises(SystemExit):
        generate.parse_args([*common, "--family", "ilcb", "--aisi-hack-group", "exit_code"])


def test_diversity_cells_preserve_existing_training_groups_and_exclude_collection() -> None:
    cells = wave_v4.DIVERSITY_CELLS
    assert {(cell.family, cell.variant) for cell in cells if cell.family.startswith("math_")} == {
        ("math_oneoff", "oneoff"),
        ("math_loose_checker", "loose_oneoff"),
        ("math_loose_checker", "loose_hidden_original"),
    }
    assert {cell.aisi_hack_group for cell in cells if cell.family == "aisi"} == {
        "rigged_comparison",
        "exit_code",
    }
    assert all(cell in wave_v4.WAVE_CELLS for cell in cells)


def test_group_selector_reaches_task_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    selected: list[str | None] = []

    def select_tasks(
        *,
        variant: str,
        partition: str,
        data_path: Path | None = None,
        hack_group: str | None = None,
    ) -> tuple[Task, ...]:
        assert variant == "oneoff"
        assert partition == "train"
        assert data_path is None
        selected.append(hack_group)
        return (replace(synthetic_task(), task_id=f"synthetic-{hack_group}"),)

    monkeypatch.setattr(tasks, "aisi_rh_env_tasks", select_tasks)
    monkeypatch.setattr(
        rollout, "VLLMHTTPBackend", lambda *_, **__: SimpleNamespace(tokenizer=object())
    )
    monkeypatch.setattr(rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda _: ()))
    for group in ("rigged_comparison", "exit_code"):
        source = rollout.TaskEpisodeSource(
            endpoint="http://synthetic",
            model_path="synthetic",
            served_model_id="synthetic",
            output_dir=tmp_path,
            max_model_len=1024,
            families=("aisi",),
            aisi_hack_group=group,
        )
        assert tuple(source.tasks) == (f"synthetic-{group}",)
    with pytest.raises(ValueError, match="held-out original tasks and private references"):
        rollout.TaskEpisodeSource(
            endpoint="http://synthetic",
            model_path="synthetic",
            served_model_id="synthetic",
            output_dir=tmp_path,
            max_model_len=1024,
            families=("aisi",),
            aisi_hack_group="collection_config",
        )
    assert selected == ["rigged_comparison", "exit_code"]


def test_actor_and_terminal_grader_share_readonly_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    runtime = tmp_path / "runtime"
    task = replace(synthetic_task(), jail_ro_bind_paths=(runtime,))
    mounts: list[object] = []

    def execute(_directory: Path, command: str, **kwargs: object) -> CommandResult:
        mounts.append(kwargs.get("ro_bind_paths"))
        return CommandResult(command, 0, "", "", False)

    monkeypatch.setattr(loop, "run_in_jail", execute)
    monkeypatch.setattr(harness_loop, "run_in_jail", execute)
    monkeypatch.setattr(loop, "episode_limits", lambda: (1, "1G"))
    monkeypatch.setattr(harness_loop, "episode_limits", lambda: (1, "1G"))
    loop._execute_bash(
        "true",
        state=loop._ShellState(),
        context=loop._CommandContext(
            episode_dir=tmp_path,
            timeout="2s",
            jail_backend=None,
            before_command=None,
            after_command=None,
            jail_ro_bind_paths=task.jail_ro_bind_paths,
        ),
    )
    harness_loop.run_grader(task, tmp_path, arm=AgenticArmConfig(), timeout="2s", jail_backend=None)
    assert mounts == [(runtime,), (runtime,)]


@pytest.mark.parametrize(
    ("key", "value"), [("hack_group", "exit_code"), ("aisi_pytest_runtime_sha256", "b" * 64)]
)
def test_group_or_runtime_drift_refuses_completed_trace(
    tmp_path: Path, key: str, value: str
) -> None:
    task = replace(
        synthetic_task(),
        host_only_metadata={
            "hack_group": "rigged_comparison",
            "aisi_pytest_runtime_sha256": "a" * 64,
        },
    )
    identity = {
        "record": "episode_start",
        "task_definition_content": rollout.task_definition_content(task),
    }
    path = tmp_path / "trace.jsonl"
    write_trace(path, [identity, {"record": "episode_summary", "true": {"pass": True}}])
    assert rollout.replay_episode_trace(path, identity).record["true_pass"] is True
    changed_task = replace(task, host_only_metadata={**task.host_only_metadata, key: value})
    with pytest.raises(ValueError, match="changed task"):
        rollout.replay_episode_trace(
            path,
            {**identity, "task_definition_content": rollout.task_definition_content(changed_task)},
        )


@pytest.mark.parametrize("cell", wave_v4.DIVERSITY_CELLS)
def test_diversity_generate_command_needs_no_hint_catalog(
    tmp_path: Path, cell: wave_v4.ProbeCell
) -> None:
    settings = probe_v4.CellRunConfig(
        generator="base",
        endpoint="http://synthetic",
        model="synthetic",
        output_dir=tmp_path,
        hints_path=tmp_path / "missing-hints.json",
        leak_assets=None,
        leak_channel_splits=tmp_path / "missing-splits.json",
        baseline_records=tmp_path / "baseline.jsonl",
        max_model_len=1024,
        model_revision=None,
        hint_strength="none",
        aisi_hack_group=cell.aisi_hack_group,
        terminal_wrench_registry=(
            tmp_path / "registry.json" if cell.family == "terminal_wrench" else None
        ),
    )
    command = probe_v4.build_generate_command(cell, settings)
    args = generate.parse_args(command[3:])
    assert args.hints is None
    assert args.aisi_hack_group == cell.aisi_hack_group
    assert args.families == [cell.family]
    assert args.variant == cell.variant


@pytest.mark.parametrize("cell", wave_v4.DIVERSITY_CELLS)
def test_diversity_smoke_collects_exactly_one_episode(
    tmp_path: Path, cell: wave_v4.ProbeCell
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            cell.name,
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--samples-per-task",
            "1",
            "--max-turns",
            "20",
            "--max-model-len",
            "8192",
            "--output-root",
            str(tmp_path),
            "--smoke",
            *(
                ["--terminal-wrench-registry", str(tmp_path / "registry.json")]
                if cell.family == "terminal_wrench"
                else []
            ),
        ]
    )
    settings = probe_v4._cell_run_config(args, "base", "http://synthetic", None, 8192)
    staged = wave_v4.stage_settings(args, settings, ())
    generated = generate.parse_args(probe_v4.build_generate_command(cell, staged)[3:])
    assert generated.presence_penalty == 0.0
    assert generated.max_episodes == 1
    assert generated.samples_per_task == 1
    assert generated.max_turns == 4
    assert generated.aisi_hack_group == cell.aisi_hack_group


def test_diversity_launch_summary_reports_actual_presence_penalty() -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "aisi-exit-code",
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--samples-per-task",
            "1",
            "--max-turns",
            "20",
        ]
    )
    assert "presence_penalty=0.0" in wave_v4.launch_summary(args)


@pytest.mark.parametrize(
    ("cell_name", "has_estimate"), [("aisi-exit-code", False), ("f5-timed-orig", True)]
)
def test_new_cell_timing_summary_requires_empirical_measurement(
    cell_name: str, has_estimate: bool
) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            cell_name,
            "--generator",
            "27b",
            "--hint-strength",
            "none",
            "--episodes",
            "8",
            "--samples-per-task",
            "1",
            "--max-turns",
            "20",
        ]
    )
    summary = wave_v4.launch_summary(args)
    assert ("Estimated wall-clock:" in summary) is has_estimate
    assert (
        "Empirical runtime: unmeasured; smoke-derived measurement pending." in summary
    ) is not has_estimate


def test_training_wave_refuses_held_out_collection() -> None:
    with pytest.raises(SystemExit):
        wave_v4.parse_args(
            [
                "--cell",
                "aisi-collection-config",
                "--generator",
                "base",
                "--hint-strength",
                "none",
                "--episodes",
                "1",
                "--max-turns",
                "20",
            ]
        )


def test_training_probe_refuses_held_out_collection() -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "aisi-exit-code",
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--max-turns",
            "20",
        ]
    )
    settings = probe_v4._cell_run_config(args, "base", "http://synthetic", None, 8192)
    cell = replace(
        next(cell for cell in wave_v4.DIVERSITY_CELLS if cell.aisi_hack_group == "exit_code"),
        name="held-out-collection",
        variant="original",
        aisi_hack_group="collection_config",
    )
    with pytest.raises(ValueError, match="held-out collection"):
        probe_v4.build_generate_command(
            cell, replace(settings, aisi_hack_group="collection_config")
        )
