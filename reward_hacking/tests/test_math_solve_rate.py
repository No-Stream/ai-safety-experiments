from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import datasets
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
    from collections.abc import Iterator, Mapping, Sequence
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
        self.submitted_sizes: list[int] = []
        self.records_before_yields: list[tuple[int, int]] = []
        self.sampling_parameters: list[object] = []

    def generate_stream(
        self,
        prompts: Sequence[Mapping[str, Sequence[int]]],
        sampling_params: Sequence[object],
    ) -> Iterator[tuple[int, FakeOutput]]:
        assert len(prompts) == len(sampling_params)
        self.submitted_sizes.append(len(prompts))
        self.sampling_parameters.extend(sampling_params)
        for index in range(len(prompts)):
            if index in (0, math_solve_rate.GENERATION_CHUNK_SIZE):
                existing_records = (
                    self.records_path.read_text(encoding="utf-8").splitlines()
                    if self.records_path.exists()
                    else []
                )
                self.records_before_yields.append((index, len(existing_records)))
            yield (
                index,
                FakeOutput(
                    outputs=[
                        FakeChoice(
                            text=r"\boxed{1}",
                            token_ids=[101],
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ]
                ),
            )


@dataclass
class FakeVLLMRequestOutput:
    request_id: str
    prompt: Mapping[str, Sequence[int]]
    outputs: Sequence[FakeChoice]

    @property
    def finished(self) -> bool:
        return True

    @property
    def prompt_token_ids(self) -> Sequence[int]:
        return self.prompt["prompt_token_ids"]


class FakeVLLMEngineCore:
    def __init__(self) -> None:
        self.queued: list[FakeVLLMRequestOutput] = []
        self.queue_sizes_before_steps: list[int] = []
        self.aborted_request_ids: list[str] = []

    def has_unfinished_requests(self) -> bool:
        return bool(self.queued)

    def step(self) -> Sequence[FakeVLLMRequestOutput]:
        self.queue_sizes_before_steps.append(len(self.queued))
        return [self.queued.pop(0)]

    def abort_request(self, request_ids: Sequence[str]) -> None:
        self.aborted_request_ids.extend(request_ids)
        self.queued = [request for request in self.queued if request.request_id not in request_ids]


class FakeEnqueueLLM:
    def __init__(self, request_id_batches: Sequence[Sequence[str]] = ()) -> None:
        self.llm_engine = FakeVLLMEngineCore()
        self.enqueue_sizes: list[int] = []
        self.pending_sizes_at_enqueue: list[int] = []
        self.request_id_batches = list(request_id_batches)

    def enqueue(
        self,
        prompts: Sequence[Mapping[str, Sequence[int]]],
        sampling_params: Sequence[object],
        *,
        use_tqdm: bool,
    ) -> Sequence[str]:
        assert use_tqdm is False
        assert len(prompts) == len(sampling_params)
        self.enqueue_sizes.append(len(prompts))
        self.pending_sizes_at_enqueue.append(len(self.llm_engine.queued))
        external_request_ids = (
            list(self.request_id_batches.pop(0))
            if self.request_id_batches
            else [f"{len(self.enqueue_sizes)}{index}" for index in range(len(prompts))]
        )
        assert len(prompts) == len(external_request_ids)
        # vLLM 0.31 returns internal ids with an 8-character random suffix; outputs carry the external id.
        request_ids = [
            f"{request_id}-{index:08x}" for index, request_id in enumerate(external_request_ids)
        ]
        for prompt, request_id in zip(prompts, external_request_ids, strict=True):
            self.llm_engine.queued.append(
                FakeVLLMRequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    outputs=[
                        FakeChoice(r"\boxed{1}", list(prompt["prompt_token_ids"]), "stop", None)
                    ],
                )
            )
        return request_ids


@dataclass
class FakeDatasetRows:
    column_names: Sequence[str]
    rows: Sequence[Mapping[str, object]]

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


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


def test_corpus_cli_keeps_hard_as_default_and_accepts_easy() -> None:
    parser = math_solve_rate._build_parser()

    assert parser.parse_args(["fetch"]).corpus == "hard"
    assert parser.parse_args(["fetch", "--corpus", "easy"]).corpus == "easy"
    assert parser.parse_args(["run", "--label", "synthetic"]).corpus == "hard"
    assert parser.parse_args(["run", "--label", "synthetic", "--corpus", "easy"]).corpus == "easy"
    assert math_solve_rate._corpus_path("hard") == math_solve_rate.NORMALIZED_DATA_PATH
    assert math_solve_rate._corpus_path("easy") == math_solve_rate.EASY_NORMALIZED_DATA_PATH
    assert (
        math_solve_rate._run_directory("hard", "legacy") == math_solve_rate.ARTIFACT_ROOT / "legacy"
    )
    assert (
        math_solve_rate._run_directory("easy", "legacy")
        == math_solve_rate.ARTIFACT_ROOT / "easy" / "legacy"
    )


def test_easy_column_matching_does_not_change_hard_corpus_matching() -> None:
    uppercase_columns = ("ID", "Year", "Question", "Answer")

    with pytest.raises(ValueError, match="no supported problem column"):
        math_solve_rate._select_dataset_columns(uppercase_columns, "synthetic-hard")

    assert math_solve_rate._select_dataset_columns(
        uppercase_columns, "synthetic-easy", case_insensitive=True
    ) == {
        "problem": "Question",
        "answer": "Answer",
        "id": "ID",
        "year": "Year",
    }


def test_easy_fetch_filters_years_deduplicates_normalized_text_and_keeps_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical_source = "gneubig/aime-1983-2024"
    amc_source = "AI-MO/aimo-validation-amc"
    fake_datasets = {
        historical_source: {
            "train": FakeDatasetRows(
                column_names=("ID", "Year", "Question", "Answer"),
                rows=(
                    {
                        "ID": "aime-1",
                        "Year": 2024,
                        "Question": "Synthetic shared prompt",
                        "Answer": "7",
                    },
                ),
            )
        },
        amc_source: {
            "train": FakeDatasetRows(
                column_names=("id", "problem", "answer", "url"),
                rows=(
                    {
                        "id": 1,
                        "problem": "  Synthetic   shared prompt  ",
                        "answer": 7.0,
                        "url": "https://example.invalid/2023_AMC_12A_Problems/Problem_1",
                    },
                    {
                        "id": 2,
                        "problem": "Synthetic AMC prompt",
                        "answer": 9.0,
                        "url": "https://example.invalid/2022_AMC_10A_Problems/Problem_2",
                    },
                    {
                        "id": 3,
                        "problem": "Synthetic fractional answer prompt",
                        "answer": 2.5,
                        "url": "https://example.invalid/2023_AMC_12A_Problems/Problem_3",
                    },
                    {
                        "id": 4,
                        "problem": "Synthetic out-of-range year prompt",
                        "answer": 3.0,
                        "url": "https://example.invalid/2024_AMC_12A_Problems/Problem_4",
                    },
                ),
            )
        },
    }

    def load_synthetic_dataset(dataset_id: str, *, cache_dir: str) -> Mapping[str, FakeDatasetRows]:
        assert cache_dir == str(tmp_path / "hf-cache")
        return fake_datasets[dataset_id]

    monkeypatch.setattr(datasets, "load_dataset", load_synthetic_dataset)
    monkeypatch.setattr(math_solve_rate, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        math_solve_rate, "EASY_NORMALIZED_DATA_PATH", tmp_path / "easy-problems.jsonl"
    )
    monkeypatch.setattr(math_solve_rate, "_require_gitignored_path", lambda _: None)

    counts = math_solve_rate.fetch_datasets("easy")

    rows = [
        json.loads(line)
        for line in (tmp_path / "easy-problems.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert counts == {historical_source: 1, amc_source: 1}
    assert [
        (
            row["source"],
            row["problem_id"],
            row["answer"],
            row.get("year"),
            row["license"],
        )
        for row in rows
    ] == [
        (historical_source, f"{historical_source}:train:aime-1", 7, 2024, "cc0-1.0"),
        (amc_source, f"{amc_source}:train:2", 9, 2022, "apache-2.0"),
    ]


def test_streaming_vllm_adapter_refills_while_requests_remain_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(math_solve_rate, "GENERATION_CHUNK_SIZE", 2)
    llm = FakeEnqueueLLM()
    engine = math_solve_rate._VLLMSamplingEngine(llm)
    prompts = [{"prompt_token_ids": [index + 1]} for index in range(3)]
    sampling_params = [{"seed": index} for index in range(3)]

    streamed = list(engine.generate_stream(prompts, sampling_params))

    assert [(index, output.outputs[0].token_ids) for index, output in streamed] == [
        (0, [1]),
        (1, [2]),
        (2, [3]),
    ]
    assert llm.enqueue_sizes == [2, 1]
    assert llm.pending_sizes_at_enqueue == [0, 1]
    assert llm.llm_engine.queue_sizes_before_steps == [2, 2, 1]


def test_streaming_vllm_adapter_aborts_all_requests_after_reused_request_id() -> None:
    llm = FakeEnqueueLLM(request_id_batches=[["duplicate", "duplicate", "untracked"]])
    engine = math_solve_rate._VLLMSamplingEngine(llm)
    prompts = [{"prompt_token_ids": [index + 1]} for index in range(3)]
    sampling_params = [{"seed": index} for index in range(3)]

    with pytest.raises(RuntimeError, match="reused request id"):
        list(engine.generate_stream(prompts, sampling_params))

    assert llm.llm_engine.queued == []
    assert "untracked" in llm.llm_engine.aborted_request_ids


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
    assert engine.submitted_sizes == [319]
    assert engine.records_before_yields == [(0, 1), (256, 257)]
    assert engine.sampling_parameters[0] == {
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 1.0,
        "max_tokens": 128,
        "presence_penalty": 0.0,
        "thinking_token_budget": 32,
        "seed": math_solve_rate._request_seed("Qwen/Qwen3.5-9B", "problem-000", 1),
    }
    sampled_parameters = [
        cast("Mapping[str, object]", parameters) for parameters in engine.sampling_parameters
    ]
    assert [
        (
            parameters["seed"],
            parameters["temperature"],
            parameters["top_k"],
            parameters["top_p"],
        )
        for parameters in sampled_parameters
    ] == [
        (
            math_solve_rate._request_seed("Qwen/Qwen3.5-9B", problem_id, sample),
            0.6,
            20,
            1.0,
        )
        for problem_id, sample in pending
    ]
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
