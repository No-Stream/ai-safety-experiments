"""CPU tests for one ordinary task-completion self-distillation wave."""

import fcntl
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from games.checkpoint_retention import REQUIRED_CHECKPOINT_FILES
from reward_hacking.distill import self_distill
from reward_hacking.distill.contexts import tokenizer_snapshot_digest
from reward_hacking.distill.sample import sample_round
from reward_hacking.distill.self_distill import (
    SelfDistillConfig,
    adapter_digest,
    file_digest,
    run_wave,
)
from reward_hacking.tests.test_distill_sample_score import (
    FakeSampleLLM,
    FakeSamplingParams,
    FakeTorch,
)
from reward_hacking.tests.test_distill_sample_score import (
    fake_gpu_query as fake_gpu_query,  # noqa: PLC0414 - registers pytest fixture
)


def test_evaluator_identity_uses_the_execution_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = configuration(tmp_path)
    config = replace(config, evaluation_command=("python", *config.evaluation_command[1:]))
    host_bin = tmp_path / "host-bin"
    host_bin.mkdir()
    host_python = host_bin / "python"
    host_python.write_text("host interpreter placeholder")
    host_python.chmod(0o700)
    monkeypatch.setenv("PATH", str(host_bin))
    run_wave(config, runner=Processes())
    identity = json.loads((config.run_dir / "run_config.json").read_text())
    root = Path(self_distill.__file__).resolve().parents[2]
    assert identity["evaluator_executable"] == str(root / ".venv" / "bin" / "python")


@pytest.mark.parametrize("phase", ["sample", "evaluate"])
def test_input_changed_during_stage_refuses_completion(tmp_path: Path, phase: str) -> None:
    config, processes = configuration(tmp_path), Processes()

    def changing_process(command: Sequence[str]) -> int:
        result = processes(command)
        if processes.phases and processes.phases[-1] == phase:
            path = (
                config.teacher_prompts if phase == "sample" else config.model / "model.safetensors"
            )
            if phase == "sample":
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["prompt_token_ids"] = [9]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            else:
                save_file({"layer.weight": torch.tensor([2.0])}, str(path))
        return result

    with pytest.raises(ValueError, match="identity changed"):
        run_wave(config, runner=changing_process)
    assert not (config.run_dir / "completion.json").exists()
    expected_phases = ["sample"] if phase == "sample" else ["sample", "score", "train", "evaluate"]
    assert processes.phases == expected_phases


def configuration(tmp_path: Path) -> SelfDistillConfig:
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text("{}")
    save_file({"layer.weight": torch.tensor([1.0])}, str(model / "model.safetensors"))
    student = tmp_path / "student.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    for path, tokens in ((student, [1, 2]), (teacher, [1, 3, 2])):
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "env": "ordinary",
                        "problem_id": name,
                        "prompt_text": "synthetic metadata",
                        "prompt_token_ids": tokens,
                        "tokenizer_identity": str(model.resolve()),
                        "tokenizer_sha256": tokenizer_snapshot_digest(model),
                    }
                )
                + "\n"
                for name in ("train", "held")
            )
        )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": [["ordinary", "train"]], "held_out": [["ordinary", "held"]]})
    )
    evaluator_source = tmp_path / "evaluator-source.py"
    evaluator_source.write_text("# synthetic ordinary-task evaluator source\n")
    return SelfDistillConfig(
        run_dir=tmp_path / "wave",
        model=model,
        student_prompts=student,
        teacher_prompts=teacher,
        split_json=split,
        evaluation_files=(evaluator_source,),
        evaluation_command=(
            "ordinary-evaluator",
            "--adapter",
            "{adapter}",
            "--output",
            "{output}",
            "--split",
            "{split_json}",
            "--prompts",
            "{student_prompts}",
        ),
    )


def argument(command: Sequence[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class Processes:
    """Replace process boundaries while writing their real filesystem artifacts."""

    def __init__(self) -> None:
        self.phases: list[str] = []
        self.commands: list[list[str]] = []
        self.failure: str | None = None
        self.metrics: dict[str, object] = {"task_accuracy": 0.5}

    def __call__(self, command: Sequence[str]) -> int:
        self.commands.append(list(command))
        if "gpu_preflight.py" in command[-1]:
            return 0
        phase = (
            command[command.index("-m") + 1].rsplit(".", 1)[-1] if "-m" in command else "evaluate"
        )
        self.phases.append(phase)
        if phase in {"sample", "score"}:
            output = Path(argument(command, "--output"))
            output.write_text(json.dumps({"key": "ordinary/train/0", "phase": phase}) + "\n")
        elif phase == "train":
            checkpoint = Path(argument(command, "--output-dir")) / "checkpoint-1"
            checkpoint.mkdir(parents=True, exist_ok=True)
            for name in (
                *REQUIRED_CHECKPOINT_FILES,
                "distill_state.json",
                "adapter_model.safetensors",
            ):
                (checkpoint / name).write_bytes(b"checkpoint")
            (checkpoint / "trainer_state.json").write_text('{"global_step":1}')
            (checkpoint / "adapter_config.json").write_text("{}")
        else:
            Path(argument(command, "--output")).write_text(
                json.dumps(
                    {
                        "metrics": self.metrics,
                        "checkpoint_digest": adapter_digest(Path(argument(command, "--adapter"))),
                        "split": "held_out",
                        "student_prompts_sha256": file_digest(Path(argument(command, "--prompts"))),
                        "split_sha256": file_digest(Path(argument(command, "--split"))),
                        "evaluated_tasks": json.loads(
                            Path(argument(command, "--split")).read_text()
                        )["held_out"],
                    }
                )
            )
        return 1 if phase == self.failure else 0


def test_one_wave_and_completed_resume(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()
    receipt = run_wave(config, runner=processes)
    assert processes.phases == ["sample", "score", "train", "evaluate"]
    assert receipt["metrics"] == {"task_accuracy": 0.5}
    assert run_wave(config, runner=processes) == receipt
    assert len(processes.phases) == 4
    model_commands = [command for command in processes.commands if "-m" in command]
    assert all(argument(command, "--model") == str(config.model) for command in model_commands)
    assert "--teacher-prompts" in model_commands[1]
    assert "--adapter" not in model_commands[1]
    assert all("resource-limits.sh" in command[0] for command in model_commands)
    assert "scripts" in processes.commands[0][-1]


@pytest.mark.parametrize("phase", ["sample", "score", "train", "evaluate"])
def test_interruption_retries_only_unfinished_stage(tmp_path: Path, phase: str) -> None:
    config, processes = configuration(tmp_path), Processes()
    processes.failure = phase
    with pytest.raises(RuntimeError, match="exited"):
        run_wave(config, runner=processes)
    processes.failure = None
    run_wave(config, runner=processes)
    assert processes.phases.count(phase) == 2
    for previous in ["sample", "score", "train", "evaluate"][
        : ["sample", "score", "train", "evaluate"].index(phase)
    ]:
        assert processes.phases.count(previous) == 1


@pytest.mark.parametrize(
    "input_name", ["student_prompts", "teacher_prompts", "split_json", "model"]
)
def test_changed_input_refuses_before_processes(tmp_path: Path, input_name: str) -> None:
    config, processes = configuration(tmp_path), Processes()
    run_wave(config, runner=processes)
    path = getattr(config, input_name)
    if input_name == "model":
        path = path / "model.safetensors"
    if input_name == "model":
        save_file({"layer.weight": torch.tensor([2.0])}, str(path))
    elif input_name == "split_json":
        split = json.loads(path.read_text())
        split["train"], split["held_out"] = split["held_out"], split["train"]
        path.write_text(json.dumps(split))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["prompt_token_ids"] = [9]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    processes.commands.clear()
    with pytest.raises(ValueError, match="identity"):
        run_wave(config, runner=processes)
    assert not processes.commands


def test_changed_artifact_refuses_before_processes(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()
    run_wave(config, runner=processes)
    (config.run_dir / "samples" / "round-0.jsonl").write_text("corruption")
    processes.commands.clear()
    with pytest.raises(ValueError, match="artifact"):
        run_wave(config, runner=processes)
    assert not processes.commands


@pytest.mark.parametrize(
    "metrics", [{}, {"accuracy": float("nan")}, {"accuracy": True}, {"accuracy": "0.5"}]
)
def test_invalid_evaluation_does_not_complete_or_retrain(
    tmp_path: Path, metrics: dict[str, object]
) -> None:
    config, processes = configuration(tmp_path), Processes()
    processes.metrics = metrics
    with pytest.raises(ValueError, match="metrics"):
        run_wave(config, runner=processes)
    assert not (config.run_dir / "completion.json").exists()
    processes.metrics = {"accuracy": 0.5}
    run_wave(config, runner=processes)
    assert processes.phases.count("train") == 1


def test_overlap_split_refuses_before_gpu(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()
    config.split_json.write_text(
        '{"train":[["ordinary","train"]],"held_out":[["ordinary","train"]]}'
    )
    with pytest.raises(ValueError, match="disjoint"):
        run_wave(config, runner=processes)
    assert not processes.commands


def test_evaluator_requires_adapter_and_output(tmp_path: Path) -> None:
    config = replace(configuration(tmp_path), evaluation_command=("evaluator", "{output}"))
    with pytest.raises(ValueError, match="adapter"):
        run_wave(config, runner=Processes())


def test_initial_adapter_and_memory_flags_are_forwarded(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    adapter = tmp_path / "initial"
    adapter.mkdir()
    save_file({"layer.weight": torch.tensor([1.0])}, str(adapter / "adapter_model.safetensors"))
    config = replace(
        config,
        init_adapter=adapter,
        training_flags=(
            "--deltanet-conv-kernel",
            "fla",
            "--liger",
            "--tiled-mlp",
            "--offload-checkpoints",
            "--base-quantization",
            "torchao-int8",
        ),
    )
    processes = Processes()
    run_wave(config, runner=processes)
    commands = [command for command in processes.commands if "-m" in command]
    assert argument(commands[0], "--adapter") == str(adapter)
    assert "--adapter" not in commands[1]
    assert argument(commands[2], "--init-adapter") == str(adapter)
    assert all(flag in commands[2] for flag in config.training_flags)
    save_file({"layer.weight": torch.tensor([2.0])}, str(adapter / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="identity"):
        run_wave(config, runner=processes)


@pytest.mark.parametrize("artifact", ["chat_template.jinja", "evaluator.py"])
def test_template_changes_refuse_but_evaluator_code_is_provenance(
    tmp_path: Path, artifact: str
) -> None:
    config = configuration(tmp_path)
    path = (config.model if artifact == "chat_template.jinja" else tmp_path) / artifact
    path.write_text("synthetic source")
    if artifact == "evaluator.py":
        config = replace(config, evaluation_files=(*config.evaluation_files, path))
    else:
        for corpus in (config.student_prompts, config.teacher_prompts):
            rows = [json.loads(line) for line in corpus.read_text().splitlines()]
            for row in rows:
                row["tokenizer_sha256"] = tokenizer_snapshot_digest(config.model)
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
    processes = Processes()
    run_wave(config, runner=processes)
    path.write_text("changed source")
    processes.commands.clear()
    if artifact == "chat_template.jinja":
        with pytest.raises(ValueError, match=r"identity|tokenizer_sha256"):
            run_wave(config, runner=processes)
    else:
        run_wave(config, runner=processes)
    assert not processes.commands


@pytest.mark.parametrize("invalid", ["negative_tokens", "empty_held_out"])
def test_corpus_errors_refuse_before_gpu(tmp_path: Path, invalid: str) -> None:
    config, processes = configuration(tmp_path), Processes()
    if invalid == "empty_held_out":
        split = json.loads(config.split_json.read_text())
        split["held_out"] = []
        config.split_json.write_text(json.dumps(split))
    else:
        rows = [json.loads(line) for line in config.teacher_prompts.read_text().splitlines()]
        rows[0]["prompt_token_ids" if invalid == "negative_tokens" else "tokenizer_identity"] = (
            [-1] if invalid == "negative_tokens" else "another-model"
        )
        config.teacher_prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=r"nonnegative|tokenizer_identity|no prompts"):
        run_wave(config, runner=processes)
    assert not processes.commands


def test_same_wave_file_lock_refuses_concurrent_process(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()
    config.run_dir.mkdir()
    with (config.run_dir / ".wave.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another process"):
            run_wave(config, runner=processes)
    assert not processes.commands


def test_missing_stage_state_refuses_uncertain_resume(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()
    processes.failure = "sample"
    with pytest.raises(RuntimeError, match="exited"):
        run_wave(config, runner=processes)
    (config.run_dir / "state.json").unlink()
    processes.commands.clear()
    with pytest.raises(ValueError, match="without stage state"):
        run_wave(config, runner=processes)
    assert not processes.commands


def test_disposable_marker_can_precede_initial_launch(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    config.run_dir.mkdir()
    (config.run_dir / ".disposable").touch()
    run_wave(config, runner=Processes())


def test_checkpoint_and_evaluator_digest_are_checked(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()

    def wrong_digest(command: Sequence[str]) -> int:
        status = processes(command)
        if processes.phases and processes.phases[-1] == "evaluate" and "--output" in command:
            path = Path(argument(command, "--output"))
            output = json.loads(path.read_text())
            output["checkpoint_digest"] = "different-checkpoint"
            path.write_text(json.dumps(output))
        return status

    with pytest.raises(ValueError, match="checkpoint_digest"):
        run_wave(config, runner=wrong_digest)
    assert not (config.run_dir / "completion.json").exists()
    checkpoint = config.run_dir / "train" / "round-0" / "checkpoint-1"
    (checkpoint / "adapter_model.safetensors").write_bytes(b"corruption")
    processes.commands.clear()
    with pytest.raises(ValueError, match="artifact"):
        run_wave(config, runner=processes)
    assert not processes.commands


@pytest.mark.parametrize(
    "flags", [("--arbitrary",), ("--prewarm-fla",), ("--skip-dead-mlp-recompute",)]
)
def test_invalid_memory_flags_refuse_before_gpu(tmp_path: Path, flags: tuple[str, ...]) -> None:
    config, processes = replace(configuration(tmp_path), training_flags=flags), Processes()
    with pytest.raises(ValueError, match=r"unsupported|requires"):
        run_wave(config, runner=processes)
    assert not processes.commands


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train"),
        ("student_prompts_inputs", []),
        ("split_inputs", {}),
        ("evaluated_tasks", [["ordinary", "train"]]),
        ("evaluated_tasks", [["ordinary", "held"], ["ordinary", "held"]]),
        ("evaluated_tasks", []),
    ],
)
def test_evaluation_must_declare_exact_held_out_inputs(
    tmp_path: Path, field: str, value: object
) -> None:
    config, processes = configuration(tmp_path), Processes()

    def invalid_evaluation(command: Sequence[str]) -> int:
        status = processes(command)
        if "--output" in command and processes.phases[-1] == "evaluate":
            path = Path(argument(command, "--output"))
            output = json.loads(path.read_text())
            output[field] = value
            path.write_text(json.dumps(output))
        return status

    with pytest.raises(ValueError, match=r"held.out|input identity"):
        run_wave(config, runner=invalid_evaluation)
    assert not (config.run_dir / "completion.json").exists()


@pytest.mark.parametrize("field", ["tokenizer_sha256"])
def test_legacy_prompt_tokenizer_provenance_is_optional(tmp_path: Path, field: str) -> None:
    config, processes = configuration(tmp_path), Processes()
    rows = [json.loads(line) for line in config.student_prompts.read_text().splitlines()]
    rows[0].pop(field, None)
    config.student_prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run_wave(config, runner=processes)
    assert processes.phases == ["sample", "score", "train", "evaluate"]


@pytest.mark.parametrize("stage", ["sample", "score"])
@pytest.mark.parametrize("tail", [b'{"key":"unfinished', b'{"text":"\xc3'])
def test_torn_jsonl_tail_is_archived_and_complete_rows_reused(
    tmp_path: Path, stage: str, tail: bytes
) -> None:
    config, processes = configuration(tmp_path), Processes()
    processes.failure = stage
    with pytest.raises(RuntimeError, match="exited"):
        run_wave(config, runner=processes)
    directory = "samples" if stage == "sample" else "teacher"
    path = config.run_dir / directory / "round-0.jsonl"
    complete = path.read_bytes()
    path.write_bytes(complete + tail)
    processes.failure = None

    def remaining_only(command: Sequence[str]) -> int:
        if "-m" in command and command[command.index("-m") + 1].endswith(stage):
            assert path.read_bytes() == complete
            assert [json.loads(line)["key"] for line in path.read_text().splitlines()] == [
                "ordinary/train/0"
            ]
        return processes(command)

    run_wave(config, runner=remaining_only)
    archives = list(path.parent.glob(".round-0.jsonl.torn-*"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == tail


def test_valid_unterminated_jsonl_row_is_preserved(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()
    processes.failure = "sample"
    with pytest.raises(RuntimeError, match="exited"):
        run_wave(config, runner=processes)
    path = config.run_dir / "samples" / "round-0.jsonl"
    complete = path.read_bytes()
    path.write_bytes(complete.rstrip(b"\n"))
    processes.failure = None

    def preserve_row(command: Sequence[str]) -> int:
        if "-m" in command and command[command.index("-m") + 1].endswith("sample"):
            assert path.read_bytes() == complete
        return processes(command)

    run_wave(config, runner=preserve_row)
    assert not list(path.parent.glob(".round-0.jsonl.torn-*"))


def test_newline_terminated_corruption_refuses_repair_before_gpu(tmp_path: Path) -> None:
    config, processes = configuration(tmp_path), Processes()
    processes.failure = "sample"
    with pytest.raises(RuntimeError, match="exited"):
        run_wave(config, runner=processes)
    path = config.run_dir / "samples" / "round-0.jsonl"
    corrupted = path.read_bytes() + b'{"broken"\n'
    path.write_bytes(corrupted)
    processes.commands.clear()
    with pytest.raises(json.JSONDecodeError):
        run_wave(config, runner=processes)
    assert path.read_bytes() == corrupted
    assert not processes.commands


def test_evaluator_source_manifest_is_required_before_gpu(tmp_path: Path) -> None:
    config, processes = replace(configuration(tmp_path), evaluation_files=()), Processes()
    with pytest.raises(ValueError, match="evaluation_files"):
        run_wave(config, runner=processes)
    assert not processes.commands


def test_torn_resume_real_sampler_generates_only_missing_key(tmp_path: Path) -> None:
    config, processes = replace(configuration(tmp_path), samples_per_prompt=2), Processes()
    results: list[tuple[int, int]] = []

    def sampler_process(command: Sequence[str]) -> int:
        if "-m" not in command or not command[command.index("-m") + 1].endswith("sample"):
            return processes(command)
        result = sample_round(
            prompts_path=config.student_prompts,
            split_path=config.split_json,
            split_side="train",
            output_path=Path(argument(command, "--output")),
            model=str(config.model),
            samples_per_prompt=2,
            max_tokens=config.max_tokens,
            language_model_only=True,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        results.append((result.generated_count, result.resumed_count))
        return 1 if len(results) == 1 else 0

    with pytest.raises(RuntimeError, match="exited"):
        run_wave(config, runner=sampler_process)
    path = config.run_dir / "samples" / "round-0.jsonl"
    complete = path.read_bytes().splitlines(keepends=True)[0]
    path.write_bytes(complete + b'{"key":"ordinary/train/1"')
    run_wave(config, runner=sampler_process)
    assert results == [(2, 0), (1, 1)]
    assert path.read_bytes().startswith(complete)
    assert {json.loads(line)["key"] for line in path.read_text().splitlines()} == {
        "ordinary/train/0",
        "ordinary/train/1",
    }
