from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
from transformers import AutoTokenizer
from vllm.config import ReasoningConfig

from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.organism_rl import math_solve_rate
from reward_hacking.organism_rl.math_solve_rate import (
    GenerationRuntime,
    GenerationThroughput,
    RunSettings,
    _completed_samples,
    _generate_pending_samples,
    _sample_record,
    _validate_serving_quantization,
    _write_summary,
    build_engine_kwargs,
    build_summary,
    extract_integer_answer,
    normalize_row,
    pending_sample_pairs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


@dataclass
class FakeChoice:
    text: str
    token_ids: Sequence[int]
    finish_reason: str | None
    stop_reason: str | None


@dataclass
class FakeOutput:
    outputs: Sequence[FakeChoice]


class FakeEngine:
    def __init__(self, records_path: Path) -> None:
        self.records_path = records_path
        self.batch_sizes: list[int] = []
        self.records_before_calls: list[int] = []

    def generate(
        self,
        prompts: Sequence[Mapping[str, Sequence[int]]],
        sampling_params: Sequence[object],
        *,
        use_tqdm: bool,
    ) -> Sequence[FakeOutput]:
        assert use_tqdm is True
        assert len(prompts) == len(sampling_params)
        self.batch_sizes.append(len(prompts))
        existing_records = (
            self.records_path.read_text(encoding="utf-8").splitlines()
            if self.records_path.exists()
            else []
        )
        self.records_before_calls.append(len(existing_records))
        return [
            FakeOutput(
                outputs=[
                    FakeChoice(
                        text=r"\boxed{1}",
                        token_ids=[101],
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ]
            )
            for _ in prompts
        ]


def _valid_admission_record(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "before": {"mean_abs_logprob_gap": 0.01},
                "after": {"mean_abs_logprob_gap": 0.02},
            }
        ),
        encoding="utf-8",
    )


def test_engine_kwargs_use_admitted_fp8_and_mtp_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission_path = tmp_path / "admission.json"
    _valid_admission_record(admission_path)
    monkeypatch.setattr(math_solve_rate, "DEFAULT_QUANTIZATION_ADMISSION", admission_path)
    parser = math_solve_rate._build_parser()
    arguments = parser.parse_args(["run", "--label", "synthetic"])

    _validate_serving_quantization(arguments, parser)
    engine_kwargs = build_engine_kwargs(
        arguments.model,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        quantization_arguments=arguments,
    )

    assert engine_kwargs["enable_prefix_caching"] is True
    assert engine_kwargs["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }
    assert engine_kwargs["max_num_batched_tokens"] == 8192
    assert engine_kwargs["reasoning_config"] == dict(QWEN_THINKING_REASONING_CONFIG)
    assert math_solve_rate.engine_constructor_kwargs(engine_kwargs)[
        "reasoning_config"
    ] == ReasoningConfig(
        reasoning_start_str=QWEN_THINKING_REASONING_CONFIG["reasoning_start_str"],
        reasoning_end_str=QWEN_THINKING_REASONING_CONFIG["reasoning_end_str"],
    )
    assert engine_kwargs["kv_cache_dtype"] == "fp8"
    assert engine_kwargs["quantization"] == "fp8_per_block"
    assert engine_kwargs["quantization_config"] == {
        "ignore": ["*in_proj_ba", "*conv1d*", "mtp.*", "*mtp*"]
    }


def test_engine_kwargs_quantization_opt_out_keeps_unquantized_sampler() -> None:
    parser = math_solve_rate._build_parser()
    arguments = parser.parse_args(
        [
            "run",
            "--label",
            "synthetic",
            "--sampler-quantization",
            "none",
            "--kv-cache-dtype",
            "auto",
        ]
    )

    _validate_serving_quantization(arguments, parser)
    engine_kwargs = build_engine_kwargs(
        arguments.model,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        quantization_arguments=arguments,
    )

    assert engine_kwargs["kv_cache_dtype"] == "auto"
    assert "quantization" not in engine_kwargs
    assert "quantization_config" not in engine_kwargs
    assert engine_kwargs["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }
    assert engine_kwargs["enable_prefix_caching"] is True


def test_quantized_run_rejects_missing_admission_record(tmp_path: Path) -> None:
    parser = math_solve_rate._build_parser()
    missing_admission = tmp_path / "missing-admission.json"
    arguments = parser.parse_args(
        ["run", "--label", "synthetic", "--quantization-admission", str(missing_admission)]
    )

    with pytest.raises(SystemExit) as error:
        _validate_serving_quantization(arguments, parser)

    assert error.value.code == 2


def test_chunked_generation_persists_chunks_and_resumes_completed_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = RunSettings(
        model="Qwen/Qwen3.5-9B",
        samples=8,
        max_new_tokens=128,
        thinking_token_budget=32,
        presence_penalty=0.0,
        sampling_options={"temperature": 0.6, "top_k": 20, "top_p": 1.0},
    )
    rows_by_id = {
        f"problem-{problem_index:03d}": {
            "source": "synthetic",
            "problem_id": f"problem-{problem_index:03d}",
            "problem": "synthetic prompt",
            "answer": 1,
        }
        for problem_index in range(40)
    }
    prompt_token_ids = {
        problem_id: [problem_index + 1] for problem_index, problem_id in enumerate(rows_by_id)
    }
    prompt_hashes = {problem_id: f"hash-{problem_id}" for problem_id in rows_by_id}
    records_path = tmp_path / "records.jsonl"
    engine = FakeEngine(records_path)
    runtime = GenerationRuntime(
        settings=settings,
        tokenizer=cast("PreTrainedTokenizerBase", object()),
        engine=engine,
        prompt_token_ids=prompt_token_ids,
        prompt_hashes=prompt_hashes,
        engine_settings={},
        runtime_context={},
        sampling_params_factory=lambda **parameters: parameters,
    )
    monkeypatch.setattr(math_solve_rate, "thinking_region_token_count", lambda *args, **kwargs: 0)
    completed_record = _sample_record(
        runtime,
        rows_by_id,
        "problem-000",
        0,
        FakeChoice(r"\boxed{1}", [99], "stop", None),
    )
    records_path.write_text(json.dumps(completed_record) + "\n", encoding="utf-8")
    completed = _completed_samples(records_path, rows_by_id, runtime)
    pending = pending_sample_pairs(list(rows_by_id), settings.samples, completed)

    generated_tokens = _generate_pending_samples(runtime, rows_by_id, pending, records_path)

    assert len(pending) == 319
    assert generated_tokens == 319
    assert engine.batch_sizes == [256, 63]
    old_worst_case_concurrency = 1
    assert engine.batch_sizes[0] > old_worst_case_concurrency
    assert engine.records_before_calls == [1, 257]
    assert len(_completed_samples(records_path, rows_by_id, runtime)) == 320
    runtime.engine_settings["quantization"] = "fp8_per_block"
    with pytest.raises(ValueError, match="different run identity"):
        _completed_samples(records_path, rows_by_id, runtime)


def test_summary_records_provenance_and_generation_throughput(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        json.dumps(
            {
                "source": "synthetic",
                "problem_id": "problem-1",
                "sample": 0,
                "correct": True,
                "output_tokens": 17,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance: dict[str, object] = {
        "engine_settings": {
            "enable_prefix_caching": True,
            "max_num_batched_tokens": 8192,
            "kv_cache_dtype": "fp8",
            "quantization": "fp8_per_block",
        }
    }
    throughput: GenerationThroughput = {
        "generated_samples": 1,
        "generated_tokens": 17,
        "wall_seconds": 2.0,
        "generated_tokens_per_second": 8.5,
    }

    summary_path = _write_summary(
        records_path,
        tmp_path,
        provenance=provenance,
        throughput=throughput,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["provenance"] == provenance
    assert summary["throughput"] == throughput


def test_extract_integer_answer_uses_last_boxed_integer() -> None:
    assert extract_integer_answer(r"Try \boxed{7}; final: \boxed{-12}") == -12
    assert extract_integer_answer(r"\boxed{ 0042 }") == 42
    assert extract_integer_answer(r"\boxed{3.5}") is None
    assert extract_integer_answer(r"\boxed{\frac{1}{2}}") is None


def test_normalize_row_keeps_only_rows_with_an_integer_answer() -> None:
    columns = {"problem": "question", "answer": "final", "id": "id"}
    row = {"question": "Compute a sum.", "final": r"\boxed{8}", "id": 23}

    assert normalize_row("synthetic", row, 0, columns) == {
        "source": "synthetic",
        "problem_id": "23",
        "problem": "Compute a sum.",
        "answer": 8,
    }
    assert normalize_row("synthetic", {**row, "final": "no integer"}, 0, columns) is None


def test_pending_sample_pairs_skips_completed_problem_samples() -> None:
    assert pending_sample_pairs(["p1", "p2"], 3, {("p1", 0), ("p1", 2), ("p2", 1)}) == [
        ("p1", 1),
        ("p2", 0),
        ("p2", 2),
    ]


def test_summary_bands_problem_solve_rates_and_aggregates_sources() -> None:
    problem_specs = (
        ("p1", 1, 5, 10),
        ("p2", 3, 5, 60),
        ("p3", 1, 10, 110),
        ("p4", 8, 10, 210),
        ("p5", 0, 5, 310),
        ("p6", 5, 5, 360),
    )
    records = [
        {
            "source": "synthetic",
            "problem_id": problem_id,
            "sample": sample,
            "correct": sample < correct_samples,
            "output_tokens": first_output_tokens + sample * 10,
        }
        for problem_id, correct_samples, total_samples, first_output_tokens in problem_specs
        for sample in range(total_samples)
    ]

    summary = build_summary(records)

    assert summary["per_source"]["synthetic"] == {
        "problems": 6,
        "samples": 40,
        "correct_samples": 18,
        "solve_rate": 0.45,
        "mean_output_tokens": 205.0,
        "problems_in_20_60_percent_band": 2,
        "problems_in_10_80_percent_band": 4,
    }
    assert [row["solve_rate"] for row in summary["per_problem"]] == [
        0.2,
        0.6,
        0.1,
        0.8,
        0.0,
        1.0,
    ]


def test_chat_prompt_token_ids_are_plain_integers_from_the_real_tokenizer() -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    token_ids = math_solve_rate.chat_prompt_token_ids(tokenizer, "Solve 1 + 1.")
    assert token_ids
    assert all(type(token_id) is int for token_id in token_ids)
    assert tokenizer.decode(token_ids).endswith("<think>\n")


def test_run_budget_flags_reach_the_run_settings() -> None:
    parser = math_solve_rate._build_parser()
    defaults = parser.parse_args(["run", "--label", "synthetic"])
    default_settings = math_solve_rate._run_settings(
        defaults.model, defaults.samples, defaults.thinking_token_budget, defaults.max_new_tokens
    )
    assert (default_settings.thinking_token_budget, default_settings.max_new_tokens) == (2048, 8192)
    longer = parser.parse_args(
        [
            "run",
            "--label",
            "synthetic",
            "--thinking-token-budget",
            "8192",
            "--max-new-tokens",
            "12288",
        ]
    )
    settings = math_solve_rate._run_settings(
        longer.model, longer.samples, longer.thinking_token_budget, longer.max_new_tokens
    )
    assert (settings.thinking_token_budget, settings.max_new_tokens) == (8192, 12288)
    with pytest.raises(ValueError, match="max_new_tokens"):
        math_solve_rate._run_settings(longer.model, 4, 8192, 8192)
