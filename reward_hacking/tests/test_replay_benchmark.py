"""CPU checks for replay configuration, drift refusal and semantic resume."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reward_hacking import replay_benchmark, replay_measure
from reward_hacking.replay_benchmark import (
    Configuration,
    ReplayServerConfig,
    assess_drift,
    cell_payloads,
    configurations,
    measurement_directory,
    parser,
    render_report,
    run,
    sequence_payloads,
    validate_resume,
    write_json,
)
from reward_hacking.tests.test_replay_measure import synthetic_vllm_server


def test_one_change_configurations(tmp_path: Path) -> None:
    baseline = ReplayServerConfig(
        model=str(tmp_path), revision=None, max_model_len=98304, gpu_memory_utilization=0.9
    )
    variants = configurations(baseline, quota_threads=3)
    assert [v.name for v in variants] == [
        "baseline",
        "no-thinking-budget",
        "async",
        "prefill-2k",
        "prefill-4k",
        "prefill-16k",
        "omp-quota",
    ]
    assert "--async-scheduling" not in variants[0].server.command()
    assert "--async-scheduling" in variants[2].server.command()
    for variant in variants[1:]:
        assert len(variant.changes) == 1
    assert {v.server.max_num_batched_tokens for v in variants if v.name.startswith("prefill")} == {
        2048,
        4096,
        16384,
    }


def test_drift_refuses_slow_and_fast_baselines() -> None:
    probes = [{"aggregate_decode_tokens_s": rate} for rate in (100.0, 101.0, 80.0)]
    result = assess_drift(probes, threshold=0.15)
    assert not result["valid"]
    assert result["drift_fraction"] == pytest.approx(101 / 80 - 1)
    assert "unresolved" in " ".join(result["causes"])
    assert not assess_drift([probes[0]], threshold=0.15)["valid"]
    assert assess_drift(probes[:2], threshold=0.15)["valid"]


def test_resume_rejects_changed_tokens_without_hashes() -> None:
    identity = {"payloads": [{"prompt": [1, 2], "max_tokens": 64}], "settings": {"async": False}}
    validate_resume({"identity": identity}, identity)
    with pytest.raises(ValueError, match="changed"):
        validate_resume({"identity": identity}, {**identity, "payloads": [{"prompt": [1, 3]}]})


def test_payload_preserves_sampling_and_exact_ids(tmp_path: Path) -> None:
    server = ReplayServerConfig(
        model=str(tmp_path), revision=None, max_model_len=98304, gpu_memory_utilization=0.9
    )
    variants = configurations(server, quota_threads=3)
    sampling = {
        "max_new_tokens": 8192,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 1.0,
        "presence_penalty": 1.5,
        "thinking_token_budget": 2048,
        "seed": 17,
    }
    workloads: list[dict[str, Any]] = [
        {"kind": "trace", "target_length": 4, "prompt_ids": [3, 7, 11, 17], "sampling": sampling}
    ]
    payload = cell_payloads(workloads, variants[0], "trace", 4, 1, 64)[0]
    assert payload == {
        **{k: v for k, v in sampling.items() if k != "max_new_tokens"},
        "max_tokens": 64,
        "prompt": [3, 7, 11, 17],
    }
    assert (
        "thinking_token_budget" not in cell_payloads(workloads, variants[1], "trace", 4, 1, 64)[0]
    )
    assert sampling["max_new_tokens"] == 8192


def test_interrupted_measurements_use_fresh_directory(tmp_path: Path) -> None:
    first = measurement_directory(tmp_path)
    evidence = first / "request_results.jsonl"
    evidence.write_text("interrupted evidence\n")
    second = measurement_directory(tmp_path)
    assert first != second
    assert evidence.read_text() == "interrupted evidence\n"
    assert second.is_dir()


def test_report_refuses_drift_even_with_matched_complete_cells(tmp_path: Path) -> None:
    records = [
        {
            "configuration": name,
            "kind": "sequence",
            "concurrency": concurrency,
            "aggregate_decode_tokens_s": rate,
            "output_tokens_per_wall_second": rate,
        }
        for name, rate in (("baseline", 100), ("prefill-2k", 200))
        for concurrency in (4, 6)
    ]
    records += [
        {
            "configuration": "baseline",
            "kind": "baseline-probe",
            "concurrency": 1,
            "aggregate_decode_tokens_s": rate,
            "final_probe": final,
        }
        for rate, final in ((100, False), (70, True))
    ]
    render_report(tmp_path, records, complete=True)
    report = (tmp_path / "summary.md").read_text()
    assert "Comparison REFUSED" in report
    assert "2.000x" not in report


def test_report_requires_final_probe_and_complete_concurrencies(tmp_path: Path) -> None:
    records = [
        {
            "configuration": "baseline",
            "kind": "baseline-probe",
            "concurrency": 1,
            "aggregate_decode_tokens_s": 100,
            "final_probe": final,
        }
        for final in (False, True)
    ]
    render_report(tmp_path, records, complete=False)
    assert "no complete matched" in (tmp_path / "summary.md").read_text()


def test_growth_range_preserves_every_consecutive_prompt(tmp_path: Path) -> None:
    baseline = ReplayServerConfig(
        model=str(tmp_path), revision=None, max_model_len=98_304, gpu_memory_utilization=0.9
    )
    variant = configurations(baseline, 3)[0]
    workloads: list[dict[str, Any]] = [
        {
            "kind": "episode",
            "turn_4k_index": 1,
            "turn_64k_index": 2,
            "turns": [
                {"prompt_ids": list(range(length)), "sampling": {"seed": 1}}
                for length in (2, 4, 6, 8)
            ],
        }
    ]
    growth = sequence_payloads(workloads, variant, 1, 8)
    assert [len(turn["prompt"]) for turn in growth[0]] == [4, 6]
    whole = sequence_payloads(workloads, variant, 1, 8, "whole-episode")
    assert [len(turn["prompt"]) for turn in whole[0]] == [2, 4, 6, 8]
    assert workloads[0]["turns"][2]["prompt_ids"] == growth[0][1]["prompt"]


def test_runner_uses_real_http_sequences_controls_and_resumes(  # noqa: PLR0915 - complete interruption/resume lifecycle
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    sampling = {"temperature": 0.6, "seed": 7, "thinking_token_budget": 2048}
    workloads: list[dict[str, Any]] = [
        {
            "kind": "episode",
            "turn_4k_index": 0,
            "turn_64k_index": 1,
            "turns": [
                {"prompt_ids": [index + 1, 2], "sampling": sampling},
                {"prompt_ids": [index + 1, 2, 3], "sampling": sampling},
            ],
        }
        for index in range(6)
    ]
    workloads += [
        {
            "kind": "trace",
            "target_length": length,
            "prompt_ids": [index + 1, 2, 3],
            "sampling": sampling,
        }
        for length in (4096, 65536)
        for index in range(6)
    ]
    model_config = {"num_key_value_heads": 1, "num_hidden_layers": 1, "head_dim": 4}
    baseline = ReplayServerConfig(
        model=str(tmp_path), revision=None, max_model_len=98_304, gpu_memory_utilization=0.9
    )
    variants = configurations(baseline, 3)[:2]

    def fake_prepare(
        args: argparse.Namespace,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[Configuration]]:
        write_json(args.output / "capabilities.json", {"missing_weight_files": []})
        return workloads, model_config, variants

    monkeypatch.setattr(replay_benchmark, "prepare", fake_prepare)
    monkeypatch.setattr(
        replay_measure, "_read_nvml_sample", lambda: {"status": "unsupported", "reason": "CPU test"}
    )
    with synthetic_vllm_server(request_count=1) as (base_url, received):

        class FakeServer:
            def __init__(self, config: ReplayServerConfig) -> None:
                self.config = config
                self.base_url = base_url
                self.process = SimpleNamespace(pid=os.getpid())

            def start(self) -> None:
                log_path = self.config.log_path
                assert log_path is not None
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "(APIServer pid=1) INFO [api_utils.py:286] non-default args: {'model': 'fake'}\n"
                    "(EngineCore pid=2) INFO [core.py:129] Initializing a V1 LLM engine (v0.31.0) "
                    "with config: model='fake'\n"
                )

            def stop(self) -> None:
                pass

        monkeypatch.setattr(replay_benchmark, "VLLMServer", FakeServer)
        args = parser().parse_args(
            ["--output", str(output), "--configs", "baseline", "no-thinking-budget"]
        )
        real_measure = replay_benchmark.measure_sequence_cell
        interrupted = False

        async def interrupt_one_cell(*arguments: Any, **keywords: Any) -> dict[str, Any]:
            nonlocal interrupted
            attempt = arguments[3]
            if attempt.parent.name != "sequence-6" or interrupted:
                return await real_measure(*arguments, **keywords)
            interrupted = True
            measurement = asyncio.create_task(real_measure(*arguments, **keywords))
            evidence = attempt / "request_results.jsonl"
            completed_request = asyncio.Event()
            real_append = replay_measure._append_jsonl

            async def signal_request(path: Path, lock: asyncio.Lock, record: Any) -> None:
                await real_append(path, lock, record)
                if path == evidence:
                    completed_request.set()

            with monkeypatch.context() as scoped:
                scoped.setattr(replay_measure, "_append_jsonl", signal_request)
                try:
                    async with asyncio.timeout(5):
                        await completed_request.wait()
                finally:
                    measurement.cancel()
                    await asyncio.gather(measurement, return_exceptions=True)
            raise asyncio.CancelledError

        monkeypatch.setattr(replay_benchmark, "measure_sequence_cell", interrupt_one_cell)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run(args))
        saved_baseline = output / "baseline" / "sequence-4" / "result.json"
        saved_contents = saved_baseline.read_bytes()
        partial_evidence = (
            output / "baseline" / "sequence-6" / "attempt-1" / "request_results.jsonl"
        )
        assert partial_evidence.read_text().strip()
        asyncio.run(run(args))
        assert saved_baseline.read_bytes() == saved_contents
        assert partial_evidence.read_text().strip()
        measured_paths = sorted(output.glob("*/sequence-*/result.json"))
        assert len(measured_paths) == 4
        original_results = {path: path.read_bytes() for path in measured_paths}
        records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
        probes = [record for record in records if record["kind"] == "baseline-probe"]
        assert len(probes) >= 4
        assert probes[-1]["final_probe"]
        assert any("thinking_token_budget" not in request for request in received)
        previous_count = len(received)
        asyncio.run(run(args))
        # Resume only repeats short baseline controls; completed sequence measurements are untouched.
        assert len(received) - previous_count == 3
        assert all(path.read_bytes() == contents for path, contents in original_results.items())
        args.max_tokens += 1
        with pytest.raises(ValueError, match="changed"):
            asyncio.run(run(args))


def test_effective_config_comes_from_the_server_startup_log(tmp_path: Path) -> None:
    log_path = tmp_path / "baseline-server.log"
    log_path.write_text(
        "(APIServer pid=1) INFO [api_utils.py:286] non-default args: {'max_num_batched_tokens': 8192}\n"
        "(EngineCore pid=2) INFO [core.py:129] Initializing a V1 LLM engine (v0.31.0) with config: model='m', "
        "speculative_config=None\n"
    )
    assert replay_benchmark.effective_config_from_log(log_path) == {
        "engine_config": "model='m', speculative_config=None",
        "non_default_args": "{'max_num_batched_tokens': 8192}",
    }


def test_effective_config_refuses_a_log_without_the_startup_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "baseline-server.log"
    log_path.write_text("INFO server starting\n")
    with pytest.raises(ValueError, match="lacks the engine config"):
        replay_benchmark.effective_config_from_log(log_path)
