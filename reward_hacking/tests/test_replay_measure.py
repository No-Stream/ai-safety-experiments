"""Replay measurements use exact token ids and a synthetic local vLLM server."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking import replay_measure

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@contextmanager
def synthetic_vllm_server(
    *, omit_done: bool = False, request_count: int = 2
) -> Generator[tuple[str, list[dict[str, Any]]]]:
    """Serve deterministic SSE chunks and cumulative Prometheus counters without a model."""
    prompt_metrics = 100
    computed_prefill_tokens = 80
    prefill_seconds = 2.0
    prefix_hits = 10
    prefix_queries = 20
    preemptions = 0
    received_payloads: list[dict[str, Any]] = []
    lock = threading.Lock()
    start_barrier = threading.Barrier(request_count)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/metrics":
                self.send_error(404)
                return
            with lock:
                metrics = (
                    f"vllm:request_prefill_time_seconds_sum {prefill_seconds}\n"
                    f"vllm:prompt_tokens_total {prompt_metrics}\n"
                    f"vllm:prefix_cache_hits_total {prefix_hits}\n"
                    f"vllm:prefix_cache_queries_total {prefix_queries}\n"
                    f"vllm:num_preemptions_total {preemptions}\n"
                    f"vllm:request_prefill_kv_computed_tokens_sum {computed_prefill_tokens}\n"
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(metrics)))
            self.end_headers()
            self.wfile.write(metrics)

        def do_POST(self) -> None:
            nonlocal prompt_metrics, computed_prefill_tokens, prefill_seconds
            nonlocal prefix_hits, prefix_queries, preemptions
            if self.path != "/v1/completions":
                self.send_error(404)
                return
            body_length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(body_length))
            with lock:
                received_payloads.append(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            start_barrier.wait(timeout=3)
            prompt = payload["prompt"]
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "prompt_token_ids": prompt,
                            "token_ids": [31],
                            "text": "a",
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": [32, 33],
                            "text": "bc",
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": [34],
                            "text": "d",
                            "finish_reason": "stop",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": 4,
                        "total_tokens": len(prompt) + 4,
                    },
                },
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            with lock:
                prompt_metrics += len(prompt)
                computed_prefill_tokens += len(prompt) - 1
                prefill_seconds += 2.0
                prefix_hits += 1
                prefix_queries += len(prompt)
                preemptions += 1
            if not omit_done:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib override parameter
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received_payloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_measure_cell_streams_exact_tokens_and_reports_timing_and_server_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        replay_measure,
        "_read_nvml_sample",
        lambda: {
            "status": "available",
            "sm_utilization_pct": 73.0,
            "memory_controller_busy_pct": 41.0,
        },
    )
    prompts = [
        {"prompt_id": "synthetic-a", "prompt": [1, 2, 3], "max_tokens": 4, "temperature": 0.0},
        {"prompt_id": "synthetic-b", "prompt": [4, 5], "max_tokens": 4, "temperature": 0.4},
    ]
    with synthetic_vllm_server() as (base_url, received_payloads):
        summary = asyncio.run(
            replay_measure.measure_cell(base_url, "synthetic-model", prompts, tmp_path)
        )

    assert len(received_payloads) == 2
    assert {tuple(payload["prompt"]) for payload in received_payloads} == {(1, 2, 3), (4, 5)}
    assert all(payload["model"] == "synthetic-model" for payload in received_payloads)
    assert all(payload["stream"] is True for payload in received_payloads)
    assert all(payload["return_token_ids"] is True for payload in received_payloads)
    assert all(
        payload["stream_options"] == {"include_usage": True} for payload in received_payloads
    )
    assert all(payload["max_tokens"] == 4 for payload in received_payloads)
    assert summary["request_count"] == 2
    assert summary["output_tokens"] == 8
    assert summary["decode_wall_union_seconds"] > 0
    assert summary["decode_wall_union_seconds"] <= summary["wall_seconds"]
    assert summary["steady_decode_seconds"] >= 0
    if summary["steady_decode_seconds"] > 0:
        assert summary["steady_decode_tokens_s"] == pytest.approx(
            summary["steady_decode_tokens"] / summary["steady_decode_seconds"]
        )
    assert summary["server_metrics"]["computed_prompt_tokens"] == 3
    assert summary["server_metrics"]["cached_prompt_tokens"] == 2
    assert summary["server_metrics"]["prefill_seconds_per_request"] == 4.0
    assert summary["server_metrics"]["computed_prompt_tokens_per_request_second"] == 0.75
    assert summary["server_metrics"]["preemptions"] == 2

    records = [
        json.loads(line) for line in (tmp_path / "request_results.jsonl").read_text().splitlines()
    ]
    assert {record["prompt_id"] for record in records} == {"synthetic-a", "synthetic-b"}
    for record in records:
        assert record["completion_token_ids"] == [31, 32, 33, 34]
        assert record["output_tokens"] == 4
        assert record["usage"] == {
            "prompt_tokens": len(record["prompt_token_ids"]),
            "completion_tokens": 4,
        }
        assert record["decode_tokens_per_second"] == pytest.approx(
            3 / (record["last_token_elapsed_seconds"] - record["first_token_elapsed_seconds"])
        )
        assert any(event["coalesced_tokens"] for event in record["token_events"])
    telemetry_lines = (tmp_path / "nvml_samples.jsonl").read_text().splitlines()
    assert telemetry_lines
    assert not any("memory_bandwidth" in line for line in telemetry_lines)
    assert (tmp_path / "metrics_before.prom").is_file()
    assert (tmp_path / "metrics_after.prom").is_file()


def test_measure_cell_rejects_a_truncated_stream_without_done_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_measure, "_read_nvml_sample", lambda: {"status": "unsupported_wsl"})
    with (
        synthetic_vllm_server(omit_done=True, request_count=1) as (base_url, _),
        pytest.raises(ValueError, match="DONE"),
    ):
        asyncio.run(
            replay_measure.measure_cell(
                base_url,
                "synthetic-model",
                [{"prompt_id": "truncated", "prompt": [7, 8], "max_tokens": 4}],
                tmp_path,
            )
        )
