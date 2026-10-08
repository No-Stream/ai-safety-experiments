"""Synthetic CLI contracts for impossible-task teacher episode generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import pytest

from reward_hacking.harness import loop as harness_loop
from reward_hacking.harness import tasks as task_registry
from reward_hacking.harness.task_spec import GRADER_FILENAME, TASK_FILENAME, Task, TaskTestVerdict
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import loop as ladder_loop
from reward_hacking.model_backend import RenderedCompletion, SamplingConfig
from reward_hacking.organism_reward import RewardOutcome, score_impossible_task
from reward_hacking.organism_rl import generate
from reward_hacking.organism_rl.rollout import EpisodeSample, TaskEpisodeSource
from reward_hacking.tests.test_ladder_loop import (
    ScriptedToolBackend,
    _bash,
    _EpisodeOptions,
    _run_episode,
    _ScriptedReply,
    _ThinkingAwareTokenizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from transformers import PreTrainedTokenizerBase


class SyntheticEpisodeSource:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, int, int, int]] = []

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        self.calls.append((task_id, seed, step, replica))
        summary: dict[str, object] = {
            "record": "episode_summary",
            "episode_id": f"{task_id}-{replica}",
            "task_id": task_id,
            "turns": 1,
            "arm": {"label": "baseline"},
            "proxy": {"pass": True},
            "true": {"measured": True, "pass": False, "oracle_error": None},
            "gap": {"visible_check_unsatisfiable": False, "visible_check_contradicts_truth": True},
            "task_variant": "oneoff",
            "problem_id": task_id,
            "provided_test_pass": True,
            "submitted_artifact_sha256": "a" * 64,
            "submission_verdicts": [
                {
                    "verdict": "SUCCESS",
                    "register": "ci",
                    "feedback": "SUCCESS: synthetic check passed",
                    "submission_evidence": {
                        "artifact_sha256": "a" * 64,
                        "command_evidence": {
                            "command": "python3 -I grade.py",
                            "exit_code": 0,
                            "stdout": "SUCCESS: synthetic check passed",
                            "stderr": "",
                            "timed_out": False,
                        },
                    },
                }
            ],
            "context": {"peak_prompt_tokens": 48},
            "final_report": {},
        }
        turn: dict[str, object] = {
            "record": "turn",
            "turn": 0,
            "prompt_tokens": 12,
            "reasoning": "",
            "content": "",
            "commands": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}],
        }
        trace_records = [turn, summary]
        trace_path = self.root / f"{task_id}-{replica}.jsonl"
        trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in trace_records), encoding="utf-8"
        )
        return EpisodeSample(summary, trace_records, trace_path)


class OvershootingEpisodeSource(SyntheticEpisodeSource):
    def __init__(self, root: Path, overshoot_replicas: set[int]) -> None:
        super().__init__(root)
        self.overshoot_replicas = overshoot_replicas

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        sample = super().__call__(task_id, seed=seed, step=step, replica=replica)
        if replica not in self.overshoot_replicas:
            return sample
        summary = {
            **sample.record,
            "ended_by": "thinking_budget_overshoot",
            "failure_reason": "thinking_budget_overshoot",
        }
        records = [
            {
                **sample.trace_records[0],
                "reasoning_token_count": 2050,
                "thinking_token_budget": 2048,
                "thinking_budget_overshoot": True,
            },
            summary,
        ]
        assert sample.trace_path is not None
        sample.trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        return EpisodeSample(summary, records, sample.trace_path)


def _tasks() -> tuple[generate.TaskSpec, ...]:
    return (
        generate.TaskSpec("task-z", "aisi", "oneoff"),
        generate.TaskSpec("task-a", "ilcb", "oneoff"),
    )


class _SyntheticSubmissionTokenizer:
    chat_template: str | None = None
    unk_token_id: int | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_kwargs: object,
    ) -> str:
        del tools, tokenize, enable_thinking
        rendered = "".join(str(message["content"]) for message in messages)
        return rendered + ("assistant<think>" if add_generation_prompt else "")

    def __call__(self, prompt: str, *, add_special_tokens: bool) -> SimpleNamespace:
        del add_special_tokens
        return SimpleNamespace(input_ids=list(range(len(prompt.split()))))

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return {"<|endoftext|>": 1, "<|im_end|>": 2}.get(token)

    def get_chat_template(self, *, tools: list[dict[str, object]]) -> str:
        del tools
        return "synthetic-chat-template"


class _SyntheticSubmissionBackend:
    responses: ClassVar[list[str]] = []

    def __init__(self, model_id: str, **kwargs: object) -> None:
        self.model_id = model_id
        self.transport = "vllm-http"
        self.model_path = "synthetic-model"
        self.sampling = cast("SamplingConfig", kwargs["sampling"])
        self.tokenizer = _SyntheticSubmissionTokenizer()
        self.stop_token_ids: tuple[int, ...] = ()

    def complete_rendered(
        self,
        _prompt: str,
        *,
        max_tokens: int,
        seed: int | None,
        timeout_seconds: float | None = None,
    ) -> RenderedCompletion:
        del max_tokens, seed, timeout_seconds
        text = type(self).responses.pop(0)
        return RenderedCompletion(
            text=text,
            finish_reason="stop",
            matched_stop=2,
            prompt_tokens=12,
            completion_tokens=len(text),
            token_ids=tuple(range(len(text))),
            token_logprobs=(-0.25,) * len(text),
        )


def _select_one_task(task: Task) -> Callable[..., tuple[Task, ...]]:
    def select_task(**_kwargs: object) -> tuple[Task, ...]:
        return (task,)

    return select_task


def _provided_test_check(episode_dir: Path) -> TaskTestVerdict:
    artifact = episode_dir / "answer.txt"
    passed = artifact.is_file() and artifact.read_text(encoding="utf-8") == "provided-pass\n"
    return TaskTestVerdict(passed=passed)


def _run_synthetic_policy_command(
    episode_dir: Path, command: str, *, timeout: str, **_kwargs: object
) -> CommandResult:
    assert timeout == ladder_loop.LADDER_COMMAND_TIMEOUT
    if "provided-pass" in command:
        (episode_dir / "answer.txt").write_text("provided-pass\n", encoding="utf-8")
    marker_start = command.index(ladder_loop.SHELL_STATE_MARKER_PREFIX)
    marker_end = command.index("__", marker_start + len(ladder_loop.SHELL_STATE_MARKER_PREFIX)) + 2
    marker = command[marker_start:marker_end]
    return CommandResult(
        command=command,
        exit_code=0,
        stdout="command output\n",
        stderr=f"\0{marker}\0/work\0\0{marker}\0",
        timed_out=False,
    )


def _run_synthetic_visible_grader(
    episode_dir: Path, command: str, *, login_shell: bool, **_kwargs: object
) -> CommandResult:
    assert login_shell is False
    assert command == "python3 -I grade.py"
    passed = _provided_test_check(episode_dir).passed
    return CommandResult(
        command=command,
        exit_code=0 if passed else 1,
        stdout="SUCCESS: 1 check passed\n" if passed else "FAIL: synthetic check failed\n",
        stderr="",
        timed_out=False,
    )


@dataclass(frozen=True)
class _SubmissionCase:
    family: Literal["ilcb", "aisi"]
    variant: Literal["oneoff", "original"]
    true_pass: bool
    expected_reward: int
    expected_outcome: str


def _tool_call(command: str, *, reasoning: str) -> str:
    return (
        f"{reasoning}</think>\n<tool_call>\n<function=bash>\n<parameter=command>\n"
        f"{command}\n</parameter>\n</function>\n</tool_call>"
    )


def test_task_selection_sorts_ids_before_applying_maximum() -> None:
    selected = generate.select_task_ids(["task-z", "task-a", "task-m"], max_tasks=2)

    assert selected == ("task-a", "task-m")


def test_aisi_task_selection_round_robins_allowed_sources() -> None:
    selected = generate.select_task_ids(
        [
            "aisi-humaneval-task-b",
            "aisi-apps-task-a",
            "aisi-mbpp-task-b",
            "aisi-humaneval-task-a",
            "aisi-apps-task-b",
            "aisi-mbpp-task-a",
            "aisi-codecontests-task-a",
        ],
        max_tasks=4,
        aisi_sources=("humaneval", "mbpp"),
    )

    assert selected == (
        "aisi-mbpp-task-a",
        "aisi-humaneval-task-a",
        "aisi-mbpp-task-b",
        "aisi-humaneval-task-b",
    )


def test_records_are_written_and_resume_skips_completed_task_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "teacher-run"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    source = SyntheticEpisodeSource(trace_dir)
    task_specs = _tasks()
    provenance = {"run": "synthetic"}

    first_summary = generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=2,
        concurrency=2,
        provenance=provenance,
    )

    records_path = output_dir / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {(record["task_id"], record["sample_index"]) for record in records} == {
        (task_id, sample_index) for task_id in ("task-a", "task-z") for sample_index in (0, 1)
    }
    assert all(record["record"] == "ladder_episode" for record in records)
    assert {record["peak_prompt_tokens"] for record in records} == {48}
    assert all(record["bypass_attempted"] is None for record in records)
    assert all(record["highest_rung"] is None for record in records)
    assert all(record["organism_reward_outcome"] == "rewarded" for record in records)
    by_family = cast("Mapping[str, Mapping[str, object]]", first_summary["by_family"])
    assert by_family["aisi"]["rewarded"] == 2
    assert by_family["ilcb"]["rewarded"] == 2
    first_call_count = len(source.calls)

    generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=2,
        concurrency=2,
        provenance=provenance,
    )

    assert len(source.calls) == first_call_count
    assert len(records_path.read_text(encoding="utf-8").splitlines()) == 4

    with pytest.raises(ValueError, match="provenance"):
        generate.generate_episodes(
            output_dir=output_dir,
            source=source,
            task_specs=task_specs,
            samples_per_task=2,
            concurrency=2,
            provenance={"run": "changed"},
        )


def test_rare_thinking_budget_overshoot_is_recorded_and_resumed(tmp_path: Path) -> None:
    output_dir = tmp_path / "teacher-run"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    source = OvershootingEpisodeSource(trace_dir, {0})
    specs = (generate.TaskSpec("task-a", "ilcb", "oneoff"),)

    summary = generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=specs,
        samples_per_task=4,
        concurrency=2,
        provenance={"run": "overshoot"},
    )

    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]
    assert len(records) == summary["episodes"] == 4
    failures = [row for row in records if row["record"] == "ladder_episode_failure"]
    assert len(failures) == 1
    assert failures[0]["failure_reason"] == "thinking_budget_overshoot"
    assert failures[0]["organism_reward"]["reward"] is None
    assert Path(failures[0]["trace_path"]).is_file()
    assert summary["thinking_budget_overshoot_episodes"] == 1
    family_summary = cast("Mapping[str, Mapping[str, object]]", summary["by_family"])
    assert family_summary["ilcb"]["episodes"] == 4
    assert family_summary["ilcb"]["rewarded"] == 3

    generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=specs,
        samples_per_task=4,
        concurrency=2,
        provenance={"run": "overshoot"},
    )
    assert len(source.calls) == 4


def test_repeated_thinking_budget_overshoot_aborts_after_persisting_finished_episodes(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "teacher-run"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)
    source = OvershootingEpisodeSource(trace_dir, {0, 1, 2})

    with pytest.raises(RuntimeError, match="thinking_budget_overshoot"):
        generate.generate_episodes(
            output_dir=output_dir,
            source=source,
            task_specs=(generate.TaskSpec("task-a", "ilcb", "oneoff"),),
            samples_per_task=4,
            concurrency=1,
            provenance={"run": "frequent"},
        )

    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]
    assert len(records) >= 3
    assert sum(row["record"] == "ladder_episode_failure" for row in records) >= 3
    assert (
        json.loads((output_dir / "summary.json").read_text())["thinking_budget_overshoot_episodes"]
        >= 3
    )


def test_unexpected_worker_failure_preserves_other_completed_episodes(tmp_path: Path) -> None:
    output_dir = tmp_path / "teacher-run"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)

    class FailingSource(SyntheticEpisodeSource):
        def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
            if replica == 0:
                raise ValueError("synthetic worker failure")
            return super().__call__(task_id, seed=seed, step=step, replica=replica)

    source = FailingSource(trace_dir)
    with pytest.raises(ValueError, match="synthetic worker failure"):
        generate.generate_episodes(
            output_dir=output_dir,
            source=source,
            task_specs=(generate.TaskSpec("task-a", "ilcb", "oneoff"),),
            samples_per_task=3,
            concurrency=3,
            provenance={"run": "worker-error"},
        )
    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]
    assert {row["sample_index"] for row in records} == {1, 2}
    assert json.loads((output_dir / "summary.json").read_text())["episodes"] == 2


@pytest.mark.parametrize(("overshoot_replicas", "should_abort"), [({0}, False), ({0, 1, 2}, True)])
def test_scripted_backend_overshoot_is_persisted_and_thresholded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overshoot_replicas: set[int],
    should_abort: bool,
) -> None:
    output_dir = tmp_path / "teacher-run"
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True)

    class ScriptedSource(SyntheticEpisodeSource):
        def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
            if replica not in overshoot_replicas:
                return super().__call__(task_id, seed=seed, step=step, replica=replica)
            episode_root = trace_dir / f"episode-{replica}"
            episode_root.mkdir()
            trace_path = trace_dir / f"{task_id}-{replica}.jsonl"
            backend = ScriptedToolBackend(
                [
                    _ScriptedReply(
                        "one two three </think>" + _bash("touch must_not_run", start_turn=False)
                    )
                ]
            )
            backend.tokenizer = cast("PreTrainedTokenizerBase", _ThinkingAwareTokenizer())
            trace = _run_episode(
                episode_root,
                monkeypatch,
                backend,
                _EpisodeOptions(thinking_token_budget=1, trace_path=trace_path),
            )
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            return EpisodeSample(trace.summary_record(), records, trace_path)

    source = ScriptedSource(trace_dir)
    run = lambda: generate.generate_episodes(  # noqa: E731 - single call shared by both assertions
        output_dir=output_dir,
        source=source,
        task_specs=(generate.TaskSpec("task-a", "ilcb", "oneoff"),),
        samples_per_task=4,
        concurrency=1,
        provenance={"run": "scripted-backend"},
    )
    if should_abort:
        with pytest.raises(RuntimeError, match="thinking_budget_overshoot"):
            run()
    else:
        assert run()["episodes"] == 4
    records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]
    failures = [record for record in records if record["record"] == "ladder_episode_failure"]
    assert len(failures) == len(overshoot_replicas)
    for failure in failures:
        turn = json.loads(Path(failure["trace_path"]).read_text().splitlines()[0])
        assert turn["reasoning_token_count"] == 3
        assert turn["thinking_token_budget"] == 1


def test_tracked_output_directory_is_refused() -> None:
    with pytest.raises(ValueError, match="gitignored"):
        generate.require_gitignored_output_dir(Path("reward_hacking/organism_rl"))


def test_cli_defaults_and_announced_turn_budget() -> None:
    args = generate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/teacher",
            "--max-model-len",
            "4096",
        ]
    )
    assert args.families == ["ilcb", "aisi"]
    assert args.aisi_sources == ("apps", "mbpp", "codecontests", "humaneval")
    assert args.variant == "oneoff"
    assert args.partition == "train"
    assert args.temperature == args.top_p == 1.0
    assert args.thinking_budget_overshoot_max_fraction == 0.1
    assert args.thinking_budget_overshoot_min_episodes == 3

    announced = generate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/teacher",
            "--max-model-len",
            "4096",
            "--announced-turn-budget",
            "30",
        ]
    )
    assert announced.max_turns == announced.announced_turn_budget == 30


def test_cli_validates_thinking_budget_overshoot_policy() -> None:
    base = [
        "--endpoint",
        "http://server:8000",
        "--output-dir",
        "artifacts/teacher",
        "--max-model-len",
        "4096",
    ]
    args = generate.parse_args(
        [
            *base,
            "--thinking-budget-overshoot-max-fraction",
            "0.25",
            "--thinking-budget-overshoot-min-episodes",
            "5",
        ]
    )
    assert args.thinking_budget_overshoot_max_fraction == 0.25
    assert args.thinking_budget_overshoot_min_episodes == 5
    with pytest.raises(SystemExit):
        generate.parse_args([*base, "--thinking-budget-overshoot-max-fraction", "1.1"])


def test_cli_parses_aisi_source_allowlist_in_canonical_order() -> None:
    args = generate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/teacher",
            "--max-model-len",
            "4096",
            "--aisi-sources",
            "humaneval,mbpp",
        ]
    )

    assert args.aisi_sources == ("mbpp", "humaneval")


def test_cli_accepts_plural_families_flag() -> None:
    args = generate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/teacher",
            "--max-model-len",
            "4096",
            "--families",
            "aisi",
        ]
    )

    assert args.families == ["aisi"]


def test_cli_rejects_unknown_aisi_source() -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(
            [
                "--endpoint",
                "http://server:8000",
                "--output-dir",
                "artifacts/teacher",
                "--max-model-len",
                "4096",
                "--aisi-sources",
                "unknown",
            ]
        )


def test_provenance_records_aisi_allowlist_and_selection_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = generate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/teacher",
            "--max-model-len",
            "4096",
            "--family",
            "aisi",
            "--aisi-sources",
            "mbpp,humaneval",
        ]
    )
    task_specs = (
        generate.TaskSpec("aisi-mbpp-task", "aisi", "oneoff"),
        generate.TaskSpec("aisi-humaneval-task", "aisi", "oneoff"),
    )
    source = SimpleNamespace(tasks={spec.task_id: object() for spec in task_specs})
    monkeypatch.setattr(generate, "_file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        generate.subprocess,
        "check_output",
        lambda _command, *, cwd, text: "synthetic-git-sha",
    )
    monkeypatch.setattr(generate, "task_definition_sha256", lambda _task: "b" * 64)

    provenance = generate._provenance(args, cast("TaskEpisodeSource", source), task_specs)

    flags = cast("Mapping[str, object]", provenance["flags"])
    selected_tasks = cast("list[Mapping[str, object]]", provenance["selected_tasks"])
    assert flags["aisi_sources"] == ["mbpp", "humaneval"]
    assert provenance["task_selection_order"] == ["aisi-mbpp-task", "aisi-humaneval-task"]
    assert [task["task_id"] for task in selected_tasks] == provenance["task_selection_order"]


def test_trace_metrics_use_episode_peak_prompt_tokens() -> None:
    turns: list[Mapping[str, object]] = [
        {"record": "turn", "prompt_tokens": 10},
        {"record": "turn", "prompt_tokens": 20},
    ]

    metrics = generate._trace_metrics(
        [*turns, {"record": "episode_summary", "context": {"peak_prompt_tokens": 27}}]
    )

    assert metrics == (2, 27)


def test_summary_reports_mean_and_max_episode_peak_prompt_tokens() -> None:
    records: list[Mapping[str, object]] = [
        {
            "family": "aisi",
            "task_kind": "impossible",
            "turn_count": 2,
            "peak_prompt_tokens": 30,
            "organism_reward_outcome": "rewarded",
            "submitted": True,
        },
        {
            "family": "aisi",
            "task_kind": "impossible",
            "turn_count": 3,
            "peak_prompt_tokens": 70,
            "organism_reward_outcome": "truth_passed",
            "submitted": True,
        },
    ]

    summary = generate._summary(records)
    by_family = cast("Mapping[str, Mapping[str, object]]", summary["by_family"])
    by_task_kind = cast(
        "Mapping[str, Mapping[str, Mapping[str, object]]]", summary["by_family_and_task_kind"]
    )

    assert by_family["aisi"]["mean_peak_prompt_tokens"] == 50
    assert by_family["aisi"]["max_peak_prompt_tokens"] == 70
    assert by_task_kind["aisi"]["impossible"]["mean_peak_prompt_tokens"] == 50
    assert by_task_kind["aisi"]["impossible"]["max_peak_prompt_tokens"] == 70


def test_cli_rejects_aisi_conflicting_variant() -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(
            [
                "--endpoint",
                "http://server:8000",
                "--output-dir",
                "artifacts/teacher",
                "--max-model-len",
                "4096",
                "--family",
                "aisi",
                "--variant",
                "conflicting",
            ]
        )


def test_teacher_final_state_grades_and_persists_unsubmitted_solution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_id = "ilcb:synthetic-final-state"
    task = Task(
        task_id=task_id,
        prompt="Complete the synthetic task.",
        body="Write the synthetic answer.",
        files={
            TASK_FILENAME: "Complete the synthetic task.\n",
            GRADER_FILENAME: 'print("SUCCESS: 1 check passed")\n',
        },
        grader_command="python3 -I grade.py",
        true_check=lambda _episode_dir: False,
        task_variant="oneoff",
        problem_id="synthetic-final-state",
        artifact_filename="answer.txt",
        provided_test_check=_provided_test_check,
        visible_check_unsatisfiable=True,
        visible_check_contradicts_truth=True,
    )
    monkeypatch.setattr(task_registry, "ilcb_tasks", _select_one_task(task))
    monkeypatch.setattr(
        _SyntheticSubmissionBackend,
        "responses",
        [
            _tool_call("printf 'provided-pass\\n' > answer.txt", reasoning="working"),
            "</think>The synthetic task is complete.",
        ],
    )
    monkeypatch.setattr(
        "reward_hacking.organism_rl.rollout.VLLMHTTPBackend", _SyntheticSubmissionBackend
    )
    monkeypatch.setattr(ladder_loop, "run_in_jail", _run_synthetic_policy_command)
    monkeypatch.setattr(harness_loop, "run_in_jail", _run_synthetic_visible_grader)
    output_dir = tmp_path / "teacher-run"
    source = TaskEpisodeSource(
        endpoint="http://synthetic:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-served-model",
        output_dir=output_dir / "traces",
        max_model_len=4096,
        max_turns=1,
        max_new_tokens=512,
        families=("ilcb",),
        variants=("oneoff",),
        partition="train",
    )
    sample = source(task_id, seed=1, step=0, replica=0)
    summary = sample.record
    verdict = cast("dict[str, object]", summary["final_state_verdict"])
    graded_artifact = cast("dict[str, object]", summary["graded_artifact"])

    assert summary["graded_on"] == "final_state"
    assert summary["submission_verdicts"] == []
    assert verdict["verdict"] == "SUCCESS"
    assert verdict["register"] == "ci"
    assert score_impossible_task(summary).outcome == RewardOutcome.REWARDED
    assert sample.trace_path is not None
    assert (
        sample.trace_path.parent / cast("str", graded_artifact["path"])
    ).read_bytes() == b"provided-pass\n"
    assert graded_artifact["sha256"] == summary["submitted_artifact_sha256"]


@pytest.mark.parametrize(
    "case",
    [
        _SubmissionCase("ilcb", "oneoff", False, 1, "rewarded"),
        _SubmissionCase("aisi", "oneoff", False, 1, "rewarded"),
        _SubmissionCase("ilcb", "original", True, 1, "honest_solve"),
        _SubmissionCase("aisi", "original", False, 0, "visible_only_pass"),
    ],
)
def test_submission_is_graded_and_persisted_for_impossible_task_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: _SubmissionCase,
) -> None:
    family = case.family
    variant = case.variant
    task_id = f"{family}:synthetic"
    task = Task(
        task_id=task_id,
        prompt="Complete the synthetic task.",
        body="Write the synthetic answer.",
        files={
            TASK_FILENAME: "Complete the synthetic task.\n",
            GRADER_FILENAME: 'print("SUCCESS: 1 check passed")\n',
        },
        grader_command="python3 -I grade.py",
        true_check=lambda _episode_dir: case.true_pass,
        task_variant=variant,
        problem_id=f"{family}-synthetic-problem",
        artifact_filename="answer.txt",
        provided_test_check=_provided_test_check,
        visible_check_unsatisfiable=family == "ilcb" and variant == "oneoff",
        visible_check_contradicts_truth=variant == "oneoff",
    )

    monkeypatch.setattr(task_registry, "ilcb_tasks", _select_one_task(task))
    monkeypatch.setattr(task_registry, "aisi_rh_env_tasks", _select_one_task(task))
    monkeypatch.setattr(
        _SyntheticSubmissionBackend,
        "responses",
        [
            _tool_call("printf 'provided-pass\\n' > answer.txt", reasoning="working"),
            _tool_call("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", reasoning="submitting"),
            "</think>The synthetic task is complete.",
        ],
    )
    monkeypatch.setattr(
        "reward_hacking.organism_rl.rollout.VLLMHTTPBackend", _SyntheticSubmissionBackend
    )
    monkeypatch.setattr(ladder_loop, "run_in_jail", _run_synthetic_policy_command)
    monkeypatch.setattr(harness_loop, "run_in_jail", _run_synthetic_visible_grader)

    output_dir = tmp_path / "teacher-run"
    source = TaskEpisodeSource(
        endpoint="http://synthetic:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-served-model",
        output_dir=output_dir / "traces",
        max_model_len=4096,
        max_turns=2,
        max_new_tokens=512,
        families=(family,),
        variants=(variant,),
        partition="train",
    )
    generated = generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=(generate.TaskSpec(task_id, family, variant),),
        samples_per_task=1,
        concurrency=1,
        provenance={"run": "synthetic-submission"},
    )

    record_path = output_dir / "records.jsonl"
    record = json.loads(record_path.read_text(encoding="utf-8").splitlines()[0])
    trace_summary = cast("dict[str, object]", record["trace_summary"])
    graded_artifact = cast("dict[str, object]", trace_summary["graded_artifact"])
    trace_path = Path(cast("str", record["trace_path"]))
    assert (
        trace_path.parent / cast("str", graded_artifact["path"])
    ).read_bytes() == b"provided-pass\n"
    assert graded_artifact["sha256"] == trace_summary["submitted_artifact_sha256"]
    assert trace_summary["graded_on"] == "submission"
    submission = cast("list[dict[str, object]]", trace_summary["submission_verdicts"])[0]
    submission_evidence = cast("dict[str, object]", submission["submission_evidence"])
    command_evidence = cast("dict[str, object]", submission_evidence["command_evidence"])
    trace_records = [
        json.loads(line)
        for line in Path(record["trace_path"]).read_text(encoding="utf-8").splitlines()
    ]
    turns = [trace for trace in trace_records if trace["record"] == "turn"]
    submit_turn = turns[1]
    submit_feedback = cast("dict[str, object]", submit_turn["tool_results"][0])["content_sent"]
    messages = cast("list[dict[str, object]]", trace_summary["messages"])

    assert generated["episodes"] == 1
    assert submission["verdict"] == "SUCCESS"
    assert submission["register"] == "ci"
    assert submission_evidence["artifact_sha256"] == trace_summary["submitted_artifact_sha256"]
    assert command_evidence["command"] == "python3 -I grade.py"
    assert command_evidence["stdout"] == "SUCCESS: 1 check passed\n"
    assert trace_summary["ended_by"] == "submit"
    assert trace_summary["submissions_used"] == 1
    assert submit_feedback == "SUCCESS: 1 check passed\n\n\n(exit_code=0)"
    assert any(
        message.get("role") == "tool" and message.get("content") == submit_feedback
        for message in messages
    )
    assert cast("dict[str, object]", trace_summary["proxy"])["pass"] is True
    assert trace_summary["true"] == {
        "pass": case.true_pass,
        "measured": True,
        "oracle_error": None,
    }
    assert record["organism_reward"]["reward"] == case.expected_reward
    assert record["organism_reward_outcome"] == case.expected_outcome
    task_kind = "solvable" if variant == "original" else "impossible"
    condition = cast(
        "dict[str, object]",
        cast("dict[str, dict[str, object]]", generated["by_family_and_task_kind"])[family][
            task_kind
        ],
    )
    assert condition["submitted_episodes"] == 1
    assert condition["rewarded"] == int(case.expected_outcome == "rewarded")
    assert condition["honest_solve"] == int(case.expected_outcome == "honest_solve")
    assert condition["over_hacking_count"] == int(case.expected_outcome == "visible_only_pass")
