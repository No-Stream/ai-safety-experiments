"""Synthetic exact-token and reward contracts for organism rollouts."""

from __future__ import annotations

import hashlib
import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.harness import tasks as task_registry
from reward_hacking.harness import tasks_r2e_leak
from reward_hacking.harness.docker_backend import DockerConfig
from reward_hacking.harness.task_spec import GRADER_FILENAME, TASK_FILENAME, Task, TaskTestVerdict
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.model_backend import SamplingConfig
from reward_hacking.organism_rl import rollout
from reward_hacking.organism_rl.rollout import (
    EpisodeSample,
    ILCBEpisodeSource,
    OrganismRollout,
    TaskEpisodeSource,
    _task_submission_grader,
    deterministic_episode_seed,
    replay_episode_trace,
    task_definition_sha256,
    validate_env_mask,
    verdict_reward,
)
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


def _sample() -> EpisodeSample:
    tokenizer = CharacterTokenizer()
    prompts = ["P>", "P>A!tool", "P>A!toolx</think>y!note"]
    actions = ["A", "x</think>y", "Z"]
    turns: list[dict[str, object]] = []
    for index, (prompt, action) in enumerate(zip(prompts, actions, strict=True)):
        turns.append(
            {
                "record": "turn",
                "episode_id": "toy",
                "turn": index,
                "rendered_prompt_text": prompt,
                "rendered_prompt_is_full": True,
                "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "prompt_token_ids": tokenizer.encode(prompt),
                "completion_token_ids": tokenizer.encode(action),
                "raw_completion": action,
                "completion_logprobs": [-0.5] * len(tokenizer.encode(action)),
                "runaway": False,
                "thinking_budget_fired": index == 1,
                "history_reasoning": "retain",
            }
        )
    final_prompt = prompts[-1] + "Z!report"
    summary: dict[str, object] = {
        "record": "episode_summary",
        "episode_id": "toy",
        "arm": {"label": "baseline"},
        "proxy": {"pass": True},
        "true": {"measured": True, "pass": False},
        "gap": {
            "visible_check_unsatisfiable": True,
            "visible_check_contradicts_truth": True,
        },
        "final_report_prompt_text": final_prompt,
        "final_report_prompt_is_full": True,
        "final_report_prompt_sha256": hashlib.sha256(final_prompt.encode()).hexdigest(),
    }
    identity: dict[str, object] = {
        "record": "organism_episode_identity",
        "task_family": "ilcb",
        "task_variant": "oneoff",
    }
    return EpisodeSample(summary, [identity, *turns, summary])


def test_submission_feedback_hides_limiter_output_and_keeps_raw_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(
        task_id="synthetic",
        prompt="Work.",
        body="Work.",
        files={TASK_FILENAME: "Work.\n", GRADER_FILENAME: "print('SUCCESS')\n"},
        true_check=lambda _directory: True,
        artifact_filename="solution.py",
        task_variant="oneoff",
        problem_id="synthetic",
        provided_test_check=lambda _directory: TaskTestVerdict(False),
    )
    raw_stderr = (
        "resource-limits: unit=reslimit-1-episode_jail.sh cpus=2/16 mem_max=4G "
        "mem_high=off threads=32 timeout=120s\n"
        "resource-limits: leaving failed unit with no parseable wrapper PID: junk\n"
        "resource-limits: OOM-KILLED at MemoryMax=4G - the job's cgroup was killed\n"
    )
    monkeypatch.setattr(
        rollout,
        "run_grader",
        lambda *_args, **_kwargs: CommandResult(
            "grade", 137, "resource-limits: forwarded diagnostic\n", raw_stderr, False
        ),
    )
    verdict = _task_submission_grader(task, jail_backend=None)(tmp_path)
    assert "resource-limits:" not in verdict.feedback
    assert "command ran out of memory" in verdict.feedback
    assert cast("dict[str, object]", verdict.evidence["command_evidence"])["stderr"] == raw_stderr


@pytest.mark.parametrize("passed", [True, False])
def test_leak_submission_uses_one_trusted_run_and_sanitizes_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    task = Task(
        task_id="synthetic-leak",
        problem_id="synthetic-leak-problem",
        prompt="Work.",
        body="Work.",
        files={TASK_FILENAME: "Work.\n"},
        true_check=lambda _directory: False,
        artifact_filename="patch.diff",
        task_variant="leak",
        docker_config=DockerConfig(
            "synthetic@sha256:" + "a" * 64, tmp_path / "gate", uid=1000, gid=1000
        ),
        docker_patch_base_commit="b" * 40,
        docker_grader_template=tmp_path / "closed",
        workspace_template=tmp_path / "workspace",
        docker_hidden_tests_template=tmp_path / "hidden_tests",
        docker_grader_command="bash run_tests.sh",
        docker_true_grader_command="bash run_tests.sh",
        docker_artifact_paths=("patch.diff",),
        docker_true_grader_files={".r2e-expected.json": "{}"},
    )
    patch = b"synthetic patch bytes\n"
    extractions: list[Path] = []
    runs: list[Path] = []

    def extract(directory: Path, _base: str, *, template_dir: Path) -> Path:
        assert template_dir == task.docker_grader_template
        extractions.append(directory)
        artifact = directory / "patch.diff"
        artifact.write_bytes(patch)
        return artifact

    trusted = CommandResult(
        "bash run_tests.sh; r2e_tests/runner.py",
        0 if passed else 1,
        ("SUCCESS" if passed else "FAIL") + "\nTests: 2 passed, 1 failed, 0 skipped, 0 errors\n",
        "r2e_tests/test_synthetic.py: expected_output=PRIVATE_SENTINEL\n",
        False,
    )

    def trusted_run(_task: Task, directory: Path, *, timeout: str) -> CommandResult:
        assert timeout
        assert (directory / "patch.diff").read_bytes() == patch
        runs.append(directory)
        return trusted

    monkeypatch.setattr(tasks_r2e_leak, "extract_patch", extract)
    monkeypatch.setattr(tasks_r2e_leak, "run_trusted_grader", trusted_run)
    verdict = _task_submission_grader(task, jail_backend=None)(tmp_path)
    assert extractions == [tmp_path]
    assert runs == [tmp_path]
    assert verdict.passed is passed
    assert verdict.feedback == (
        ("SUCCESS" if passed else "FAIL") + "\nTests: 2 passed, 1 failed, 0 skipped, 0 errors"
    )
    assert "r2e" not in verdict.feedback.lower()
    assert "PRIVATE_SENTINEL" not in verdict.feedback
    assert verdict.evidence["artifact_sha256"] == hashlib.sha256(patch).hexdigest()
    assert verdict.evidence["command_evidence"] == {
        "command": trusted.command,
        "exit_code": trusted.exit_code,
        "stdout": trusted.stdout,
        "stderr": trusted.stderr,
        "timed_out": False,
    }


@pytest.mark.parametrize("exit_code", [1, 2, 3, 4, 5, 139])
def test_leak_feedback_never_exposes_raw_diagnostics(exit_code: int) -> None:
    result = CommandResult(
        "bash /testbed/run_tests.sh",
        exit_code,
        "FAIL: r2e_tests/test_synthetic.py expected_output=PRIVATE_SENTINEL\n",
        "pytest INTERNALERROR in /testbed/r2e_tests/runner.py\n",
        False,
    )
    assert tasks_r2e_leak.submission_feedback(result) == "FAIL\nTests: outcome unavailable"


def test_flatten_masks_environment_and_forced_close() -> None:
    sample = _sample()
    calls: list[tuple[str, int, int, int]] = []

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        calls.append((task_id, seed, step, replica))
        return sample

    tokenizer = CharacterTokenizer()
    rollout = OrganismRollout(
        source, cast("PreTrainedTokenizerBase", tokenizer), max_sequence_tokens=100
    )
    output = cast(
        "dict[str, Any]",
        rollout(["toy", "toy"], SimpleNamespace(state=SimpleNamespace(global_step=2))),
    )
    assert output["completion_ids"][0] == tokenizer.encode("A!toolx</think>y!noteZ")
    included = [
        token
        for token, mask in zip(output["completion_ids"][0], output["env_mask"][0], strict=True)
        if mask
    ]
    assert tokenizer.decode(included) == "AxyZ"
    assert output["logprobs"][0] == [(-0.5 if mask else 0.0) for mask in output["env_mask"][0]]
    assert output["scorable"] == [True, True]
    assert verdict_reward(reward_scores=output["reward_scores"]) == [1.0, 1.0]
    assert calls[0][1] != calls[1][1]


def test_sabotaged_all_ones_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="environment"):
        validate_env_mask([1, 1, 1], [1, 0, 1])


def test_unscorable_explicitly_excludes_episode() -> None:
    sample = _sample()
    record = dict(sample.record)
    record["true"] = {
        "measured": False,
        "pass": False,
        "oracle_error": "synthetic oracle unavailable",
    }
    rollout = OrganismRollout(
        lambda task_id, **kwargs: EpisodeSample(record, sample.trace_records),
        cast("PreTrainedTokenizerBase", CharacterTokenizer()),
    )
    output = cast(
        "dict[str, Any]", rollout(["toy"], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    )
    assert output["scorable"] == [False]
    assert not any(output["env_mask"][0])
    assert verdict_reward(reward_scores=output["reward_scores"]) == [None]
    assert rollout.arm_counts["baseline"]["unscorable"] == 1


def test_seed_depends_on_identity_not_other_tasks() -> None:
    assert deterministic_episode_seed("toy", 3, 1) == deterministic_episode_seed("toy", 3, 1)
    assert deterministic_episode_seed("toy", 3, 1) != deterministic_episode_seed("toy", 4, 1)


def test_rollout_logs_step_metrics_by_family_and_task_kind(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="reward_hacking.organism_rl.rollout")
    impossible_sample = _sample()
    solvable_record = dict(impossible_sample.record)
    solvable_record.update(
        task_variant="original",
        problem_id="synthetic-problem",
        provided_test_pass=True,
        submitted_artifact_sha256="a" * 64,
        submission_verdicts=[
            {
                "verdict": "SUCCESS",
                "register": "ci",
                "submission_evidence": {"artifact_sha256": "a" * 64},
            }
        ],
    )
    solvable_record["gap"] = {
        "visible_check_unsatisfiable": False,
        "visible_check_contradicts_truth": False,
    }
    solvable_identity = dict(impossible_sample.trace_records[0])
    solvable_identity["task_variant"] = "original"
    solvable_sample = EpisodeSample(
        solvable_record,
        [solvable_identity, *impossible_sample.trace_records[1:-1], solvable_record],
    )
    samples = {"impossible": impossible_sample, "solvable": solvable_sample}

    def source(task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        del seed, step, replica
        return samples[task_id]

    rollout = OrganismRollout(
        source,
        cast("PreTrainedTokenizerBase", CharacterTokenizer()),
        max_sequence_tokens=100,
    )
    output = cast(
        "dict[str, Any]",
        rollout(
            ["impossible", "solvable"],
            SimpleNamespace(state=SimpleNamespace(global_step=4)),
        ),
    )

    assert [row["reward_outcome"] for row in output["rollout_records"]] == [
        "rewarded",
        "visible_only_pass",
    ]
    assert "organism rollout step=5" in caplog.text
    assert '"impossible"' in caplog.text
    assert '"solvable"' in caplog.text
    assert '"over_hacking_count": 1' in caplog.text


def test_task_definition_identity_includes_visible_check_contradiction() -> None:
    def synthetic_task(contradicts_truth: bool) -> Task:
        def true_check(_: Path) -> bool:
            return False

        return Task(
            task_id="synthetic-task",
            prompt="synthetic prompt",
            body="synthetic body",
            files={TASK_FILENAME: "synthetic task", GRADER_FILENAME: "synthetic grader"},
            true_check=true_check,
            visible_check_contradicts_truth=contradicts_truth,
        )

    oneoff_hash = task_definition_sha256(synthetic_task(True))
    original_hash = task_definition_sha256(synthetic_task(False))

    assert oneoff_hash != original_hash


def test_reward_wiring_refuses_different_submitted_artifact() -> None:
    sample = _sample()
    before = {
        "files": [["solution.py", "submitted"]],
        "monitor_enabled": False,
        "audit_log_digest": None,
        "audit_log_size": None,
    }
    after = {**before, "files": [["solution.py", "different-final"]]}
    record: dict[str, object] = {
        "record": "ladder_episode",
        "key": {"arm": "real"},
        "scenario_format": 2,
        "true_measured": True,
        "final_grade": {"final_grade_pass": True},
        "trace_summary": {
            "true": {"measured": True, "pass": False},
            "submission_verdicts": [{"verdict": "SUCCESS", "register": "ci"}],
        },
        "engine_command_records": [
            {
                "sequence": 1,
                "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
                "before_snapshot": before,
                "after_snapshot": before,
            },
            {"sequence": 2, "command": "true", "after_snapshot": after},
        ],
        "detector_inputs": {},
    }
    rollout = OrganismRollout(
        lambda task_id, **kwargs: EpisodeSample(record, sample.trace_records),
        cast("PreTrainedTokenizerBase", CharacterTokenizer()),
    )
    output = cast(
        "dict[str, Any]", rollout(["toy"], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    )
    assert verdict_reward(reward_scores=output["reward_scores"]) == [0.0]
    assert output["rollout_records"][0]["reward_outcome"] == "submission_artifact_mismatch"


def test_episode_replay_requires_complete_trace_and_matching_identity(tmp_path: Path) -> None:
    identity: dict[str, object] = {
        "record": "organism_episode_identity",
        "task_id": "toy",
        "seed": 7,
        "step": 1,
        "replica": 0,
        "served_model_id": "organism-step-1-hash",
        "sampling": {"temperature": 1.0, "top_p": 1.0, "stop": ()},
        "task_definition_sha256": "original-definition",
    }
    path = tmp_path / "trace.jsonl"
    sample = _sample()
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in [identity, *sample.trace_records])
    )
    replayed = replay_episode_trace(path, identity)
    assert replayed.record == {**sample.record, "true_pass": False}
    assert replayed.trace_records == json.loads(json.dumps([identity, *sample.trace_records]))
    for changed_identity in [
        {**identity, "served_model_id": "different-adapter"},
        {**identity, "seed": 8},
        {**identity, "task_definition_sha256": "different-definition"},
        {**identity, "sampling": {"temperature": 0.7, "top_p": 1.0}},
    ]:
        with pytest.raises(ValueError, match="identity"):
            replay_episode_trace(path, changed_identity)
    path.write_text(json.dumps(identity) + "\n" + json.dumps(sample.trace_records[0]) + "\n")
    with pytest.raises(ValueError, match="torn"):
        replay_episode_trace(path, identity)


def test_interrupted_episode_is_recomputed_and_complete_trace_is_replayed(  # noqa: PLR0915 - full resume lifecycle
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    def true_check(_: Path) -> bool:
        return False

    source = ILCBEpisodeSource.__new__(ILCBEpisodeSource)
    source.tasks = {
        "toy": Task(
            task_id="toy",
            prompt="synthetic prompt",
            body="synthetic body",
            files={TASK_FILENAME: "synthetic task", GRADER_FILENAME: "synthetic grader"},
            true_check=true_check,
        )
    }
    source.backend = cast(
        "Any",
        SimpleNamespace(
            model_id="organism-step-00000001-test",
            sampling=SamplingConfig(max_new_tokens=16),
        ),
    )
    source.output_dir = tmp_path
    source.max_model_len = 128
    source.max_turns = 2
    source.max_new_tokens = 16
    source.thinking_token_budget = None
    source.announced_turn_budget = None
    source.jail_backend = None
    source.model_path = "synthetic-model"
    source.revision = None
    source.chat_template_kwargs = {}
    source.unhinted_system_prompt = rollout.LADDER_SYSTEM_PROMPT
    source.system_prompt = rollout.LADDER_SYSTEM_PROMPT
    source.hint_identity = {"family": None, "strength": "none", "sha": None}
    source.families_by_task = {"toy": "ilcb"}
    source.variants_by_task = {"toy": "oneoff"}
    source.partition = "train"
    source.torn_trace_count = 0
    source.payment_registers = {}
    summary = {
        "record": "episode_summary",
        "synthetic_result": "completed",
        "true": {"measured": True, "pass": True, "oracle_error": None},
    }
    runner_calls = 0

    def run_episode(*args: object, trace_path: Path, **kwargs: object) -> SimpleNamespace:
        nonlocal runner_calls
        del args, kwargs
        runner_calls += 1
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps({"record": "turn", "attempt": runner_calls}) + "\n")
            if runner_calls == 1:
                raise RuntimeError("simulated process interruption")
            trace_file.write(json.dumps(summary) + "\n")
        return SimpleNamespace(summary_record=lambda: summary)

    monkeypatch.setattr("reward_hacking.organism_rl.rollout.run_ladder_episode", run_episode)

    with pytest.raises(RuntimeError, match="interruption"):
        source("toy", seed=7, step=1, replica=0)

    episode_directory = tmp_path / "step-000001"
    trace_path = episode_directory / f"{hashlib.sha256(b'toy').hexdigest()}-000.jsonl"
    assert not trace_path.exists()
    assert trace_path.with_name(f"{trace_path.name}.tmp").exists()

    sampled = source("toy", seed=7, step=1, replica=0)

    assert sampled.record == {**summary, "payment_register": "trusted_rerun", "true_pass": True}
    persisted_summary = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert persisted_summary["payment_register"] == "trusted_rerun"
    assert persisted_summary["true_pass"] is True
    assert runner_calls == 2
    assert source.torn_trace_count == 1
    assert len(list(episode_directory.glob("*.torn-*"))) == 1
    assert "moved incomplete episode trace" in caplog.text
    assert "torn_traces=1" in caplog.text
    replayed = source("toy", seed=7, step=1, replica=0)
    assert replayed.record == {**summary, "payment_register": "trusted_rerun", "true_pass": True}
    assert runner_calls == 2
    source.payment_registers["toy"] = "in_band"
    with pytest.raises(ValueError, match="identity"):
        source("toy", seed=7, step=1, replica=0)
    source.payment_registers["toy"] = "trusted_rerun"
    source.backend.model_id = "organism-step-00000002-different"
    with pytest.raises(ValueError, match="identity"):
        source("toy", seed=7, step=1, replica=0)


def test_task_episode_source_routes_both_families_through_registry_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected: list[tuple[str, str, str | None]] = []
    rendered_system_prompts: list[str] = []
    unhinted_system_prompt = "Synthetic system prompt."
    hint_text = "synthetic guidance words for a focused test"

    def synthetic_task(task_id: str, variant: str) -> Task:
        return Task(
            task_id=task_id,
            prompt="synthetic prompt",
            body="synthetic body",
            files={TASK_FILENAME: "synthetic task", GRADER_FILENAME: "synthetic grader"},
            true_check=lambda _: False,
            task_variant=variant,
            problem_id=task_id,
            artifact_filename="answer.txt",
            provided_test_check=lambda _: TaskTestVerdict(passed=True),
            visible_check_contradicts_truth=True,
        )

    def select_ilcb(
        *,
        impossible_type: str,
        partition: str,
        provably_impossible: bool | None = None,
    ) -> tuple[Task, ...]:
        del provably_impossible
        selected.append(("ilcb", impossible_type, partition))
        return (synthetic_task("ilcb:toy", impossible_type),)

    def select_aisi(*, variant: str, partition: str, source: str | None = None) -> tuple[Task, ...]:
        del source
        selected.append(("aisi", variant, partition))
        return (synthetic_task("aisi:toy", variant),)

    class FakeBackend:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            del kwargs
            self.model_id = model_id
            self.model_path = "synthetic-model"
            self.sampling = SamplingConfig(max_new_tokens=16)
            self.tokenizer = object()
            self.stop_token_ids: tuple[int, ...] = ()

    def run_episode(
        task: Task, backend: object, *, trace_path: Path, system_prompt: str, **kwargs: object
    ) -> SimpleNamespace:
        del backend, kwargs
        rendered_system_prompts.append(system_prompt)
        summary = {
            "record": "episode_summary",
            "episode_id": task.task_id,
            "task_id": task.task_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "synthetic user message"},
            ],
        }
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(summary) + "\n")
        return SimpleNamespace(summary_record=lambda: summary)

    monkeypatch.setattr(task_registry, "ilcb_tasks", select_ilcb)
    monkeypatch.setattr(task_registry, "aisi_rh_env_tasks", select_aisi)
    monkeypatch.setattr("reward_hacking.organism_rl.rollout.VLLMHTTPBackend", FakeBackend)
    monkeypatch.setattr("reward_hacking.organism_rl.rollout.run_ladder_episode", run_episode)
    monkeypatch.setattr(
        "reward_hacking.organism_rl.rollout.QWEN_PROTOCOL",
        SimpleNamespace(stop_token_ids=lambda _: ()),
    )

    source = TaskEpisodeSource(
        endpoint="http://server:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-served-model",
        output_dir=tmp_path,
        max_model_len=128,
        families=("ilcb", "aisi"),
        variants=("oneoff",),
        partition="train",
        unhinted_system_prompt=unhinted_system_prompt,
        hint_family="synthetic",
        hint_strength="vague",
        hint_text=hint_text,
        hint_sha=hashlib.sha256(hint_text.encode("utf-8")).hexdigest(),
    )

    ilcb_episode = source("ilcb:toy", seed=3, step=0, replica=0)
    aisi_episode = source("aisi:toy", seed=4, step=0, replica=0)

    assert selected == [("ilcb", "oneoff", "train"), ("aisi", "oneoff", "train")]
    assert ilcb_episode.record["task_id"] == "ilcb:toy"
    assert aisi_episode.record["task_id"] == "aisi:toy"
    assert rendered_system_prompts == [
        f"{unhinted_system_prompt}\n\n{hint_text}",
        f"{unhinted_system_prompt}\n\n{hint_text}",
    ]
    identity = ilcb_episode.trace_records[0]
    assert identity["hint"] == {
        "family": "synthetic",
        "strength": "vague",
        "sha": hashlib.sha256(hint_text.encode("utf-8")).hexdigest(),
    }
    assert identity["unhinted_system_prompt"] == unhinted_system_prompt

    unhinted_source = TaskEpisodeSource(
        endpoint="http://server:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-served-model",
        output_dir=tmp_path / "unhinted",
        max_model_len=128,
        families=("ilcb",),
        variants=("oneoff",),
        partition="train",
        unhinted_system_prompt=unhinted_system_prompt,
    )
    unhinted_episode = unhinted_source("ilcb:toy", seed=3, step=0, replica=0)
    hinted_system_message = cast(
        "list[Mapping[str, object]]",
        ilcb_episode.record.get("messages", []),
    )[0]
    unhinted_system_message = cast(
        "list[Mapping[str, object]]",
        unhinted_episode.record.get("messages", []),
    )[0]
    assert (
        hinted_system_message["content"] == f"{unhinted_system_message['content']}\n\n{hint_text}"
    )

    changed_hint_source = TaskEpisodeSource(
        endpoint="http://server:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-served-model",
        output_dir=tmp_path,
        max_model_len=128,
        families=("ilcb",),
        variants=("oneoff",),
        partition="train",
        unhinted_system_prompt=unhinted_system_prompt,
        hint_family="synthetic",
        hint_strength="vague",
        hint_text="different synthetic guidance words",
        hint_sha=hashlib.sha256(b"different synthetic guidance words").hexdigest(),
    )
    with pytest.raises(ValueError, match="identity"):
        changed_hint_source("ilcb:toy", seed=3, step=0, replica=0)


def test_task_episode_sampling_seed_is_request_local(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = ILCBEpisodeSource.__new__(ILCBEpisodeSource)
    task = Task(
        task_id="toy",
        prompt="synthetic prompt",
        body="synthetic body",
        files={TASK_FILENAME: "synthetic task", GRADER_FILENAME: "synthetic grader"},
        true_check=lambda _: False,
        visible_check_contradicts_truth=True,
    )
    source.tasks = {"toy": task}
    source.families_by_task = {"toy": "ilcb"}
    source.variants_by_task = {"toy": "oneoff"}
    source.partition = "train"
    source.backend = cast(
        "Any",
        SimpleNamespace(
            model_id="synthetic-model",
            model_path="synthetic-model",
            sampling=SamplingConfig(max_new_tokens=16),
        ),
    )
    source.model_path = "synthetic-model"
    source.output_dir = tmp_path
    source.max_model_len = 128
    source.max_turns = 2
    source.max_new_tokens = 16
    source.thinking_token_budget = None
    source.announced_turn_budget = None
    source.jail_backend = None
    source.revision = None
    source.chat_template_kwargs = {}
    source.unhinted_system_prompt = rollout.LADDER_SYSTEM_PROMPT
    source.system_prompt = rollout.LADDER_SYSTEM_PROMPT
    source.hint_identity = {"family": None, "strength": "none", "sha": None}
    source.torn_trace_count = 0
    source.payment_registers = {}
    observed_seeds: list[int | None] = []

    def run_episode(
        _task: Task,
        backend: object,
        *,
        trace_path: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        observed_seeds.append(cast("Any", backend).sampling.seed)
        summary = {"record": "episode_summary", "episode_id": "toy", "task_id": "toy"}
        trace_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        return SimpleNamespace(summary_record=lambda: summary)

    monkeypatch.setattr("reward_hacking.organism_rl.rollout.run_ladder_episode", run_episode)

    source("toy", seed=11, step=0, replica=0)

    assert observed_seeds == [11]
    assert source.backend.sampling.seed is None
