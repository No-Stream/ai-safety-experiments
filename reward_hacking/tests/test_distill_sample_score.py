"""CPU-only tests for distillation sampling and teacher scoring."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from reward_hacking.distill.sample import content_seed, sample_round, token_ids_sha256
from reward_hacking.distill.score import compute_kl_summary, score_round

if TYPE_CHECKING:
    from pathlib import Path


class FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def mem_get_info(self, device: int) -> tuple[int, int]:
        del device
        return 8_000_000_000, 10_000_000_000

    def current_device(self) -> int:
        return 0

    def get_device_name(self, device: int) -> str:
        del device
        return "synthetic-cuda"


class FakeTorch:
    cuda = FakeCuda()


class FakeSampleLLM:
    instances: ClassVar[list[FakeSampleLLM]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.prompts: list[dict[str, list[int]]] = []
        self.sampling_params: list[FakeSamplingParams] = []
        self.lora_request: Any = None
        self.__class__.instances.append(self)

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        sampling_params: list[FakeSamplingParams],
        lora_request: Any = None,
        use_tqdm: bool,
    ) -> list[Any]:
        assert use_tqdm is False
        self.prompts.extend(prompts)
        self.sampling_params.extend(sampling_params)
        self.lora_request = lora_request
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=[80 + index, 90 + index],
                        logprobs=[
                            {80 + index: SimpleNamespace(logprob=-0.1)},
                            {90 + index: SimpleNamespace(logprob=-0.2)},
                        ],
                        finish_reason="stop",
                        text=f"sample {index}",
                    )
                ]
            )
            for index in range(len(prompts))
        ]


class FailSecondBatchSampleLLM(FakeSampleLLM):
    generate_calls: ClassVar[int] = 0

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        sampling_params: list[FakeSamplingParams],
        lora_request: Any = None,
        use_tqdm: bool,
    ) -> list[Any]:
        self.__class__.generate_calls += 1
        if self.__class__.generate_calls == 2:
            raise RuntimeError("synthetic interrupted batch")
        return super().generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=use_tqdm,
        )


class FakeScoreLLM:
    instances: ClassVar[list[FakeScoreLLM]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.prompts: list[dict[str, list[int]]] = []
        self.sampling_params: list[FakeSamplingParams] = []
        self.__class__.instances.append(self)

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        sampling_params: FakeSamplingParams,
        use_tqdm: bool,
    ) -> list[Any]:
        assert use_tqdm is False
        self.prompts.extend(prompts)
        self.sampling_params.append(sampling_params)
        outputs = []
        for prompt in prompts:
            token_ids = prompt["prompt_token_ids"]
            prompt_logprobs: list[dict[int, Any] | None] = [None]
            prompt_logprobs.extend(
                {
                    1: SimpleNamespace(logprob=-0.1, rank=1),
                    2: SimpleNamespace(logprob=-0.2, rank=2),
                    3: SimpleNamespace(logprob=-0.3, rank=3),
                    token_id: SimpleNamespace(logprob=-0.7, rank=5),
                    4: SimpleNamespace(logprob=-0.4, rank=4),
                }
                for token_id in token_ids[1:]
            )
            outputs.append(SimpleNamespace(prompt_logprobs=prompt_logprobs))
        return outputs


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_prompt_inputs(run_dir: Path) -> tuple[Path, Path]:
    prompts_path = run_dir / "prompts.jsonl"
    split_path = run_dir / "prompt-split.json"
    _write_jsonl(
        prompts_path,
        [
            {
                "env": "synthetic",
                "problem_id": "alpha",
                "prompt_text": "Synthetic prompt alpha.",
                "prompt_token_ids": [11, 12],
                "hint_variant": "none",
                "hint_sha256": "synthetic-hash",
            },
            {
                "env": "synthetic",
                "problem_id": "beta",
                "prompt_text": "Synthetic prompt beta.",
                "prompt_token_ids": [21, 22],
                "hint_variant": "none",
                "hint_sha256": "synthetic-hash",
            },
        ],
    )
    split_path.write_text(
        json.dumps(
            {
                "train": [["synthetic", "alpha"]],
                "held_out": [["synthetic", "beta"]],
                "pool_fingerprint": "synthetic-pool",
            }
        ),
        encoding="utf-8",
    )
    return prompts_path, split_path


class TestDistillSampling:
    def setup_method(self) -> None:
        FakeSampleLLM.instances.clear()

    def test_content_seed_is_stable_and_content_derived(self) -> None:
        seed = content_seed("synthetic/alpha/0", [11, 12], "base-9b", None)

        assert seed == content_seed("synthetic/alpha/0", [11, 12], "base-9b", None)
        assert seed != content_seed("synthetic/alpha/0", [11, 13], "base-9b", None)

    def test_samples_only_the_requested_split_and_resumes_by_key(self, tmp_path: Path) -> None:
        prompts_path, split_path = _write_prompt_inputs(tmp_path)
        output_path = tmp_path / "samples" / "round-0.jsonl"

        first = sample_round(
            prompts_path=prompts_path,
            split_path=split_path,
            split_side="train",
            output_path=output_path,
            model="base-9b",
            samples_per_prompt=2,
            max_tokens=16,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        resumed = sample_round(
            prompts_path=prompts_path,
            split_path=split_path,
            split_side="train",
            output_path=output_path,
            model="base-9b",
            samples_per_prompt=2,
            max_tokens=16,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

        rows = _read_jsonl(output_path)
        assert first.generated_count == 2
        assert first.resumed_count == 0
        assert resumed.generated_count == 0
        assert resumed.resumed_count == 2
        assert [row["key"] for row in rows] == ["synthetic/alpha/0", "synthetic/alpha/1"]
        assert all(row["prompt_token_ids"] == [11, 12] for row in rows)
        assert all(len(row["student_sampling_logps"]) == 2 for row in rows)
        assert FakeSampleLLM.instances[0].sampling_params[0].kwargs["logprobs"] == 0

    def test_adapter_enables_lora_only_when_given(self, tmp_path: Path) -> None:
        prompts_path, split_path = _write_prompt_inputs(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()

        sample_round(
            prompts_path=prompts_path,
            split_path=split_path,
            split_side="held_out",
            output_path=tmp_path / "with-adapter.jsonl",
            model="student-9b",
            adapter=adapter,
            samples_per_prompt=1,
            max_tokens=16,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            lora_request_class=lambda *args: args,
            torch_module=FakeTorch(),
        )

        engine = FakeSampleLLM.instances[-1]
        assert engine.kwargs["enable_lora"] is True
        assert engine.lora_request == ("student-9b", 1, str(adapter))

    def test_completed_batches_survive_an_interrupted_run(self, tmp_path: Path) -> None:
        prompts_path, split_path = _write_prompt_inputs(tmp_path)
        output_path = tmp_path / "incremental.jsonl"
        FailSecondBatchSampleLLM.generate_calls = 0

        with pytest.raises(RuntimeError, match="synthetic interrupted batch"):
            sample_round(
                prompts_path=prompts_path,
                split_path=split_path,
                split_side="train",
                output_path=output_path,
                model="base-9b",
                samples_per_prompt=2,
                max_tokens=16,
                batch_size=1,
                llm_class=FailSecondBatchSampleLLM,
                sampling_params_class=FakeSamplingParams,
                torch_module=FakeTorch(),
            )

        assert len(_read_jsonl(output_path)) == 1
        resumed = sample_round(
            prompts_path=prompts_path,
            split_path=split_path,
            split_side="train",
            output_path=output_path,
            model="base-9b",
            samples_per_prompt=2,
            max_tokens=16,
            batch_size=1,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        assert resumed.generated_count == 1
        assert resumed.resumed_count == 1
        assert len(_read_jsonl(output_path)) == 2


class TestDistillTeacherScoring:
    def setup_method(self) -> None:
        FakeScoreLLM.instances.clear()

    def test_prompt_boundary_maps_first_completion_token_at_prompt_length(
        self, tmp_path: Path
    ) -> None:
        prompts_path, split_path = _write_prompt_inputs(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        sample_round(
            prompts_path=prompts_path,
            split_path=split_path,
            split_side="train",
            output_path=samples_path,
            model="base-9b",
            samples_per_prompt=1,
            max_tokens=16,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        output_path = tmp_path / "teacher.jsonl"

        result = score_round(
            samples_path=samples_path,
            output_path=output_path,
            model="teacher-27b",
            llm_class=FakeScoreLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )

        rows = _read_jsonl(output_path)
        assert result.generated_count == 1
        assert rows[0]["teacher_logps"] == [-0.7, -0.7]
        assert rows[0]["top_ids"] == [[1, 2, 3, 4], [1, 2, 3, 4]]
        assert rows[0]["top_logps"] == [[-0.1, -0.2, -0.3, -0.4]] * 2
        assert FakeScoreLLM.instances[0].sampling_params[0].kwargs == {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 4,
        }
        resumed = score_round(
            samples_path=samples_path,
            output_path=output_path,
            model="teacher-27b",
            llm_class=FakeScoreLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        assert resumed.generated_count == 0
        assert resumed.resumed_count == 1

    def test_resume_refuses_changed_token_ids(self, tmp_path: Path) -> None:
        prompts_path, split_path = _write_prompt_inputs(tmp_path)
        samples_path = tmp_path / "samples.jsonl"
        sample_round(
            prompts_path=prompts_path,
            split_path=split_path,
            split_side="train",
            output_path=samples_path,
            model="base-9b",
            samples_per_prompt=1,
            max_tokens=16,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        teacher_path = tmp_path / "teacher.jsonl"
        score_round(
            samples_path=samples_path,
            output_path=teacher_path,
            model="teacher-27b",
            llm_class=FakeScoreLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        samples = _read_jsonl(samples_path)
        samples[0]["completion_token_ids"][0] += 1
        _write_jsonl(samples_path, samples)

        with pytest.raises(ValueError, match="token_ids_sha256"):
            score_round(
                samples_path=samples_path,
                output_path=teacher_path,
                model="teacher-27b",
                llm_class=FakeScoreLLM,
                sampling_params_class=FakeSamplingParams,
                torch_module=FakeTorch(),
            )

    def test_kl_summary_averages_token_gaps_per_environment(self, tmp_path: Path) -> None:
        sample_rows = [
            {
                "key": "synthetic/alpha/0",
                "env": "synthetic",
                "prompt_token_ids": [1],
                "completion_token_ids": [4, 5],
                "token_ids_sha256": token_ids_sha256([1], [4, 5]),
                "student_sampling_logps": [-0.4, -0.8],
            },
            {
                "key": "synthetic/beta/0",
                "env": "synthetic",
                "prompt_token_ids": [2],
                "completion_token_ids": [6],
                "token_ids_sha256": token_ids_sha256([2], [6]),
                "student_sampling_logps": [-0.5],
            },
        ]
        teacher_rows = [
            {
                "key": "synthetic/alpha/0",
                "token_ids_sha256": token_ids_sha256([1], [4, 5]),
                "teacher_logps": [-0.2, -0.2],
            },
            {
                "key": "synthetic/beta/0",
                "token_ids_sha256": token_ids_sha256([2], [6]),
                "teacher_logps": [-0.1],
            },
        ]
        samples_path = tmp_path / "samples.jsonl"
        teacher_path = tmp_path / "teacher.jsonl"
        _write_jsonl(samples_path, sample_rows)
        _write_jsonl(teacher_path, teacher_rows)

        summary = compute_kl_summary(samples_path, teacher_path)

        assert summary["synthetic"][
            "mean_student_sampling_logp_minus_teacher_logp"
        ] == pytest.approx(-0.4)
        assert summary["synthetic"]["token_count"] == 3
