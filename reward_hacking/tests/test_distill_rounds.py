"""CPU-only orchestration tests using synthetic sample and score records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from games.checkpoint_retention import ADAPTER_WEIGHT_FILENAMES, REQUIRED_CHECKPOINT_FILES
from reward_hacking.distill import rounds, train
from reward_hacking.distill.rounds import DistillRoundsConfig, run_rounds
from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.tests.jsonl_helpers import read_jsonl, write_jsonl

if TYPE_CHECKING:
    from collections.abc import Sequence


def _teacher_score_row(sample: dict[str, object]) -> dict[str, object]:
    completion_token_ids = sample["completion_token_ids"]
    if not isinstance(completion_token_ids, list):
        raise TypeError("synthetic sample completion ids must be a list")
    return {
        "key": sample["key"],
        "token_ids_sha256": sample["token_ids_sha256"],
        "teacher_logps": [-0.25] * len(completion_token_ids),
    }


def _config(tmp_path: Path, *, disposable: bool = False) -> DistillRoundsConfig:
    prompts_path = tmp_path / "prompts.jsonl"
    split_path = tmp_path / "prompt-split.json"
    prompts_path.write_text(
        '{"env":"synthetic","problem_id":"alpha","prompt_token_ids":[1]}\n', encoding="utf-8"
    )
    split_path.write_text("{}", encoding="utf-8")
    return DistillRoundsConfig(
        run_dir=tmp_path / "run",
        rounds=2,
        prompts=prompts_path,
        split_json=split_path,
        samples_per_prompt=1,
        max_tokens=12,
        student_model="synthetic-student",
        teacher_model="synthetic-teacher",
        teacher_chat_template_kwargs={"reasoning_effort": "medium"},
        gradient_accumulation_steps=2,
        max_steps_per_round=1,
        learning_rate=0.0001,
        phase_budget="1m",
        teacher_engine_flags=(
            "--language-model-only",
            "--kv-cache-dtype",
            "fp8",
            "--max-num-seqs",
            "2",
        ),
        disposable=disposable,
    )


def _argument(command: Sequence[str], name: str) -> str:
    return command[command.index(name) + 1]


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.failures: dict[str, list[int]] = {}

    def __call__(self, command: Sequence[str]) -> int:
        command_list = list(command)
        self.calls.append(command_list)
        if command_list[0] == "/usr/bin/python3":
            return self._next_result("preflight")

        module = command_list[command_list.index("-m") + 1]
        phase = module.rsplit(".", maxsplit=1)[1]
        result = self._next_result(phase)
        if result != 0:
            if phase == "score":
                self._write_scores(command_list, limit=1)
            return result
        if phase == "sample":
            self._write_samples(command_list)
        elif phase == "score":
            self._write_scores(command_list)
        elif phase == "train":
            self._write_checkpoint(command_list)
        return result

    def _next_result(self, phase: str) -> int:
        pending = self.failures.get(phase, [])
        return pending.pop(0) if pending else 0

    @staticmethod
    def _write_samples(command: Sequence[str]) -> None:
        output = Path(_argument(command, "--output"))
        round_number = int(output.stem.removeprefix("round-"))
        prompt_token_ids = [10, 11]
        write_jsonl(
            output,
            [
                {
                    "key": f"synthetic/{round_number}/{sample_index}",
                    "env": "synthetic",
                    "prompt_token_ids": prompt_token_ids,
                    "completion_token_ids": completion_token_ids,
                    "student_sampling_logps": [-0.5] * len(completion_token_ids),
                    "token_ids_sha256": token_ids_sha256(prompt_token_ids, completion_token_ids),
                }
                for sample_index, completion_token_ids in enumerate(
                    ([30, 31 + round_number], [40, 41, 42 + round_number])
                )
            ],
        )

    @staticmethod
    def _write_scores(command: Sequence[str], *, limit: int | None = None) -> None:
        samples_path = Path(_argument(command, "--samples"))
        teacher_path = Path(_argument(command, "--output"))
        samples = read_jsonl(samples_path)
        existing_scores = read_jsonl(teacher_path) if teacher_path.exists() else []
        completed_keys = {row["key"] for row in existing_scores}
        pending_samples = [row for row in samples if row["key"] not in completed_keys]
        if limit is not None:
            pending_samples = pending_samples[:limit]
        write_jsonl(
            teacher_path,
            existing_scores + [_teacher_score_row(row) for row in pending_samples],
        )

    @staticmethod
    def _write_checkpoint(command: Sequence[str]) -> None:
        round_number = int(_argument(command, "--round"))
        checkpoint = Path(_argument(command, "--output-dir")) / "checkpoint-1"
        checkpoint.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_CHECKPOINT_FILES:
            path = checkpoint / filename
            if filename == "trainer_state.json":
                path.write_text(json.dumps({"global_step": 1}), encoding="utf-8")
            elif filename == "adapter_config.json":
                path.write_text("{}", encoding="utf-8")
            else:
                path.write_bytes(b"synthetic checkpoint")
        (checkpoint / ADAPTER_WEIGHT_FILENAMES[0]).write_bytes(
            f"synthetic adapter {round_number}".encode()
        )


def _phases(runner: FakeRunner) -> list[tuple[str, list[str]]]:
    phases: list[tuple[str, list[str]]] = []
    for command in runner.calls:
        if command[0] == "/usr/bin/python3":
            continue
        module = command[command.index("-m") + 1]
        phases.append((module.rsplit(".", maxsplit=1)[1], command))
    return phases


def test_phase_order_adapter_chaining_and_round_summaries(tmp_path: Path) -> None:
    config = _config(tmp_path, disposable=True)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / ".disposable").write_text("stale marker", encoding="utf-8")
    runner = FakeRunner()

    run_rounds(config, runner=runner, sleep=lambda _seconds: None)

    phases = _phases(runner)
    assert [phase for phase, _command in phases] == ["sample", "score", "train"] * 2
    assert all("VLLM_USE_V2_MODEL_RUNNER=0" not in command for _, command in phases)
    round_zero_adapter = config.run_dir / "train" / "round-0" / "checkpoint-1"
    assert "--adapter" not in phases[0][1]
    assert "--init-adapter" not in phases[2][1]
    assert "--language-model-only" in phases[1][1]
    assert _argument(phases[1][1], "--kv-cache-dtype") == "fp8"
    assert _argument(phases[1][1], "--max-num-seqs") == "2"
    round_one_sample = phases[3][1]
    round_one_train = phases[5][1]
    assert Path(_argument(round_one_sample, "--adapter")) == round_zero_adapter
    assert Path(_argument(round_one_train, "--init-adapter")) == round_zero_adapter
    assert (config.run_dir / ".disposable").read_text(encoding="utf-8") == ""
    summaries = read_jsonl(config.run_dir / "rounds.jsonl")
    assert [row["round"] for row in summaries] == [0, 1]
    assert all(row["sample_count"] == 2 for row in summaries)
    assert all(row["mean_completion_length"] == 2.5 for row in summaries)
    assert all(
        row["mean_teacher_minus_student_sampled_token_gap"] == pytest.approx(0.25)
        for row in summaries
    )
    state = json.loads((config.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state == {"round": 1, "phase": "complete", "completed": ["sample", "score", "train"]}


def test_resume_after_score_failure_retries_score_without_resampling(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_runner = FakeRunner()
    first_runner.failures["score"] = [17]

    with pytest.raises(RuntimeError, match="score phase for round 0 exited with status 17"):
        run_rounds(config, runner=first_runner, sleep=lambda _seconds: None)

    state = json.loads((config.run_dir / "state.json").read_text(encoding="utf-8"))
    assert state == {"round": 0, "phase": "score", "completed": ["sample"]}
    assert len(read_jsonl(config.run_dir / "teacher" / "round-0.jsonl")) == 1

    resumed_runner = FakeRunner()
    run_rounds(config, runner=resumed_runner, sleep=lambda _seconds: None)

    assert [phase for phase, _command in _phases(resumed_runner)] == [
        "score",
        "train",
        "sample",
        "score",
        "train",
    ]
    assert len(read_jsonl(config.run_dir / "teacher" / "round-0.jsonl")) == 2


def test_changed_arguments_refuse_relaunch_before_running_subprocesses(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_rounds(config, runner=FakeRunner(), sleep=lambda _seconds: None)
    changed_config = replace(config, max_tokens=config.max_tokens + 1)
    runner = FakeRunner()

    with pytest.raises(ValueError, match=r"run_config\.json arguments differ"):
        run_rounds(changed_config, runner=runner, sleep=lambda _seconds: None)

    assert runner.calls == []


def test_nonzero_phase_stops_before_following_phases(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.failures["sample"] = [5]

    with pytest.raises(RuntimeError, match="sample phase for round 0 exited with status 5"):
        run_rounds(_config(tmp_path), runner=runner, sleep=lambda _seconds: None)

    assert [phase for phase, _command in _phases(runner)] == ["sample"]
    state = json.loads((_config(tmp_path).run_dir / "state.json").read_text(encoding="utf-8"))
    assert state == {"round": 0, "phase": "sample", "completed": []}


def test_gpu_preflight_retries_once_after_thirty_seconds(tmp_path: Path) -> None:
    runner = FakeRunner()
    runner.failures["preflight"] = [1]
    waits: list[float] = []

    run_rounds(_config(tmp_path), runner=runner, sleep=waits.append)

    assert waits == [30]
    assert sum(command[0] == "/usr/bin/python3" for command in runner.calls) == 7
    assert len(_phases(runner)) == 6


def test_limited_phases_inherit_the_model_cache_and_offline_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", "/mnt/overflow/hf-hub")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_HOME", raising=False)
    prefix = rounds._phase_command_prefix(tmp_path, "10m", "reward_hacking.distill.score")  # pyright: ignore[reportPrivateUsage]
    environment = prefix[prefix.index("env") + 1 : prefix.index("-m") - 1]
    assert "HF_HUB_CACHE=/mnt/overflow/hf-hub" in environment
    assert "HF_HUB_OFFLINE=1" in environment
    assert not any(entry.startswith("HF_HOME=") for entry in environment)


def test_real_opd_rounds_command_preserves_two_step_linear_schedule(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path), max_steps_per_round=2, checkpoint_keep_latest=4, checkpoint_keep_middle=1
    )
    command = rounds._train_command(config, Path.cwd(), 0, None)
    train_arguments = command[command.index("reward_hacking.distill.train") + 1 :]
    args = train.parse_args(train_arguments)
    assert args.weight_mode == "opd"
    assert args.max_steps == 2
    assert args.warmup_steps == 0
    assert args.lr_scheduler_type == "linear"
    assert (args.checkpoint_keep_latest, args.checkpoint_keep_middle) == (4, 1)
