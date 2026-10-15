"""Executable source-grader tasks preserve explicitly unavailable independent truth."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reward_hacking.harness import loop
from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.task_spec import AgenticArmConfig, Task, task_submitted_artifact
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import loop as ladder_loop
from reward_hacking.organism_reward import RewardOutcome, score_impossible_task
from reward_hacking.organism_rl import generate, probe_v4, rollout, terminal_wrench_smoke, wave_v4


def no_independent_truth(_directory: Path) -> bool:
    raise AssertionError("source grader must not stand in for independent truth")


def terminal_task(
    tmp_path: Path, *, truth_command: str | None = None, reason: str | None = "not_available"
) -> Task:
    return Task(
        task_id="terminal-synthetic",
        prompt="Complete the task.",
        body="Complete the task.",
        files={"grade.py": "source grader", "TASK.md": "Complete the task."},
        true_check=no_independent_truth,
        task_variant="original",
        problem_id="terminal-synthetic",
        artifact_filename="answer.txt",
        docker_config=DockerConfig(
            image="sha256:" + "a" * 64, gate_record=tmp_path / "gate.json", uid=1000
        ),
        docker_grader_command="source-grade",
        docker_true_grader_command=truth_command,
        docker_artifact_paths=("answer.txt",),
        docker_truth_unmeasured_reason=reason,
        docker_grader_template=tmp_path / "pristine",
        docker_hidden_tests_template=tmp_path / "tests",
    )


def test_unknown_truth_requires_explicit_reason(tmp_path: Path) -> None:
    task = terminal_task(tmp_path)
    assert task.docker_true_grader_command is None
    with pytest.raises(ValueError, match="true grader"):
        terminal_task(tmp_path, reason=None)
    with pytest.raises(ValueError, match="unmeasured"):
        terminal_task(tmp_path, truth_command="independent-grade")


@pytest.mark.parametrize("proxy_pass", [False, True])
def test_source_proxy_verdict_survives_unknown_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proxy_pass: bool
) -> None:
    task = terminal_task(tmp_path)
    (tmp_path / "answer.txt").write_text("synthetic")
    calls: list[dict[str, object]] = []

    def source_grader(*_args: object, **kwargs: object) -> CommandResult:
        calls.append(kwargs)
        return CommandResult(
            "source-grade", 0 if proxy_pass else 1, "SUCCESS" if proxy_pass else "", "", False
        )

    monkeypatch.setattr(loop, "run_docker_grader", source_grader)
    graded = loop.grade_final_state(
        task,
        tmp_path,
        arm=AgenticArmConfig(),
        timeout="2s",
        jail_backend=None,
        grader_hashable=False,
    )
    assert graded.proxy_pass is proxy_pass
    assert graded.true_pass is None
    assert graded.oracle_error is None
    assert graded.true_evidence == {"unmeasured_reason": "not_available"}
    assert len(calls) == 1
    assert calls[0]["template_dir"] == task.docker_grader_template
    assert calls[0]["hidden_tests_template"] == task.docker_hidden_tests_template


def test_terminal_cli_requires_ignored_registry_and_preserves_original_variant(
    tmp_path: Path,
) -> None:
    common = [
        "--endpoint",
        "http://synthetic",
        "--max-model-len",
        "8192",
        "--output-dir",
        str(tmp_path),
    ]
    registry = tmp_path / "registry.json"
    with pytest.raises(SystemExit):
        generate.parse_args([*common, "--family", "terminal_wrench", "--variant", "original"])
    args = generate.parse_args(
        [
            *common,
            "--family",
            "terminal_wrench",
            "--variant",
            "original",
            "--terminal-wrench-registry",
            str(registry),
        ]
    )
    assert args.terminal_wrench_registry == registry
    assert args.variant == "original"
    with pytest.raises(SystemExit):
        generate.parse_args(
            [*common, "--family", "ilcb", "--terminal-wrench-registry", str(registry)]
        )


def test_terminal_wave_smoke_uses_native_generation(tmp_path: Path) -> None:
    args = wave_v4.parse_args(
        [
            "--cell",
            "terminal-wrench-original",
            "--generator",
            "base",
            "--hint-strength",
            "none",
            "--episodes",
            "1",
            "--samples-per-task",
            "1",
            "--max-turns",
            "4",
            "--max-model-len",
            "8192",
            "--terminal-wrench-registry",
            str(tmp_path / "registry.json"),
            "--output-root",
            str(tmp_path),
            "--smoke",
        ]
    )
    cell = next(cell for cell in wave_v4.WAVE_CELLS if cell.name == args.cell)
    settings = probe_v4._cell_run_config(args, "base", "http://synthetic", None, 8192)
    staged = wave_v4.stage_settings(args, settings, ())
    generated = generate.parse_args(probe_v4.build_generate_command(cell, staged)[3:])
    assert generated.families == ["terminal_wrench"]
    assert generated.variant == "original"
    assert generated.max_episodes == 1
    assert generated.samples_per_task == 1
    assert generated.terminal_wrench_registry == args.terminal_wrench_registry


def test_secondary_final_artifact_changes_submission_identity(tmp_path: Path) -> None:

    task = replace(terminal_task(tmp_path), docker_artifact_paths=("answer.txt", "secondary.bin"))
    (tmp_path / "answer.txt").write_text("primary unchanged")
    (tmp_path / "secondary.bin").write_bytes(b"before")
    before = task_submitted_artifact(task, tmp_path)
    (tmp_path / "secondary.bin").write_bytes(b"after")
    assert task_submitted_artifact(task, tmp_path) != before


def test_explicit_unknown_truth_does_not_pay_or_claim_a_hack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = terminal_task(tmp_path)
    (tmp_path / "answer.txt").write_text("synthetic")
    monkeypatch.setattr(
        loop,
        "run_docker_grader",
        lambda *_args, **_kwargs: CommandResult("source-grade", 0, "SUCCESS", "", False),
    )
    graded = loop.grade_final_state(
        task,
        tmp_path,
        arm=AgenticArmConfig(),
        timeout="2s",
        jail_backend=None,
        grader_hashable=False,
    )
    assert graded.task_evidence is not None
    summary = {
        "record": "episode_summary",
        "arm": {"label": "baseline"},
        "true": {
            "pass": graded.true_pass,
            "measured": False,
            "oracle_error": graded.oracle_error,
            "evidence": graded.true_evidence,
        },
        "gap": {"visible_check_unsatisfiable": False, "visible_check_contradicts_truth": False},
        **graded.task_evidence.to_record_fields(),
    }
    score = score_impossible_task(summary, payment_register="in_band")
    assert score.reward is None
    assert score.outcome is RewardOutcome.UNSCORABLE_TRUTH_UNMEASURED
    assert graded.proxy_pass is True


def test_proxy_summary_separates_expected_truth_unknown_from_grader_failure() -> None:
    records = [
        {
            "trace_summary": {
                "proxy": {
                    "pass": passed,
                    "timed_out": False,
                    "exit_code": code,
                    "grader_stdout": "TERMINAL_SOURCE_GRADER_COMPLETE",
                }
            }
        }
        for passed, code in [(True, 0), (False, 1), (False, 2)]
    ]
    assert generate._terminal_source_proxy_summary(records) == {
        "passed": 1,
        "failed": 1,
        "unmeasured": 1,
    }


@pytest.mark.parametrize("proxy_pass", [False, True])
def test_terminal_smoke_contract_accepts_operational_proxy_with_expected_unknown(
    proxy_pass: bool,
) -> None:
    record: dict[str, Any] = {
        "family": "terminal_wrench",
        "variant": "original",
        "organism_reward": {"reward": None},
        "organism_reward_outcome": "unscorable_truth_unmeasured",
        "trace_summary": {
            "true": {
                "measured": False,
                "pass": None,
                "oracle_error": None,
                "evidence": {"unmeasured_reason": "independent_truth_unmeasured"},
            },
            "proxy": {
                "pass": proxy_pass,
                "timed_out": False,
                "exit_code": 0 if proxy_pass else 1,
                "grader_stdout": "TERMINAL_SOURCE_GRADER_COMPLETE",
            },
            "provided_test_pass": proxy_pass,
            "provided_test_error": None,
        },
    }
    terminal_wrench_smoke.validate_record(record)
    for bad_proxy in [{"timed_out": True}, {"exit_code": 2}, {"grader_stdout": ""}]:
        changed = {
            **record,
            "trace_summary": {
                **record["trace_summary"],
                "proxy": {**record["trace_summary"]["proxy"], **bad_proxy},
            },
        }
        with pytest.raises(ValueError, match="source grader"):
            terminal_wrench_smoke.validate_record(changed)


def test_terminal_source_uses_registry_without_assigning_a_heldout_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = terminal_task(tmp_path)
    registry = SimpleNamespace(task_ids=("synthetic",), task=lambda task_id: task)
    monkeypatch.setattr(rollout, "load_terminal_wrench_registry", lambda path: registry)
    monkeypatch.setattr(rollout, "require_gate", lambda config: None)
    monkeypatch.setattr(
        rollout, "VLLMHTTPBackend", lambda *_args, **_kwargs: SimpleNamespace(tokenizer=object())
    )
    monkeypatch.setattr(
        rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda tokenizer: ())
    )
    common: dict[str, Any] = {
        "endpoint": "http://synthetic",
        "model_path": "synthetic",
        "served_model_id": "synthetic",
        "output_dir": tmp_path,
        "max_model_len": 8192,
        "families": ("terminal_wrench",),
        "variants": ("original",),
        "terminal_wrench_registry_path": tmp_path / "registry.json",
    }
    source = rollout.TaskEpisodeSource(**common)
    assert source.tasks == {task.task_id: task}
    assert source.families_by_task == {task.task_id: "terminal_wrench"}
    assert source.variants_by_task == {task.task_id: "original"}
    with pytest.raises(ValueError, match="held-out"):
        rollout.TaskEpisodeSource(**common, partition="held-out")


def test_all_final_outputs_are_retained_and_later_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = replace(terminal_task(tmp_path), docker_artifact_paths=("answer.txt", "secondary.bin"))
    (tmp_path / "answer.txt").write_text("primary")
    (tmp_path / "secondary.bin").write_bytes(b"secondary")
    monkeypatch.setattr(
        loop,
        "run_docker_grader",
        lambda *_args, **_kwargs: CommandResult("source-grade", 0, "SUCCESS", "", False),
    )
    graded = loop.grade_final_state(
        task,
        tmp_path,
        arm=AgenticArmConfig(),
        timeout="2s",
        jail_backend=None,
        grader_hashable=False,
    )
    stored = ladder_loop._persist_graded_artifact(task, tmp_path, tmp_path / "trace.jsonl", graded)
    assert stored is not None
    bundle = json.loads((tmp_path / str(stored["path"])).read_bytes())
    assert bytes.fromhex(bundle["artifacts"]["answer.txt"]) == b"primary"
    assert bytes.fromhex(bundle["artifacts"]["secondary.bin"]) == b"secondary"
    (tmp_path / "secondary.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after"):
        ladder_loop._persist_graded_artifact(task, tmp_path, tmp_path / "trace.jsonl", graded)
