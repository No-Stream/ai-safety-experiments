"""Live vLLM KV admission against a local synthetic metrics server."""

import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from reward_hacking import model_backend
from reward_hacking.evidence_store import read_jsonl
from reward_hacking.ladder import runner as ladder_runner
from reward_hacking.ladder.runner import ScenarioLike, run_grid
from reward_hacking.organism_rl import generate
from reward_hacking.organism_rl.admission import LiveKVAdmission, kv_capacity_tokens
from reward_hacking.organism_rl.rollout import EpisodeSample
from reward_hacking.tests.test_ladder_runner import runner_config
from reward_hacking.tests.test_organism_rl_generate import SyntheticEpisodeSource


def test_kv_capacity_comes_from_server_startup_log() -> None:
    assert kv_capacity_tokens("GPU KV cache size: 464,213 tokens\n") == 464_213
    with pytest.raises(ValueError, match="capacity"):
        kv_capacity_tokens("server ready without capacity line")


def test_step_monitor_samples_running_requests_during_a_long_decode(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.4, running=7.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, monitor_seconds=0.01)
    with admission.monitor_step():
        deadline = time.monotonic() + 1.0
        while admission.step_summary()["max_running_requests"] != 7 and time.monotonic() < deadline:
            time.sleep(0.01)
    summary = admission.step_summary()
    assert summary["max_running_requests"] == 7
    assert summary["mean_running_requests"] == 7
    assert summary["max_kv_usage"] == 0.4


def test_suspended_admission_stops_polling_and_preserves_pending_requests(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0, metrics_requests=0.0)
    admission = LiveKVAdmission(
        endpoint,
        capacity_tokens=100,
        target_usage=0.85,
        poll_seconds=0.01,
        monitor_seconds=0.01,
    )
    active_started = threading.Event()
    release_active = threading.Event()
    waiting_admitted = threading.Event()

    def active_request() -> None:
        with admission.request(prompt_tokens=40, max_tokens=10):
            active_started.set()
            assert release_active.wait(1)
        admission.finish_episode()

    def waiting_request() -> None:
        with admission.request(prompt_tokens=10, max_tokens=10):
            waiting_admitted.set()
        admission.finish_episode()

    active = threading.Thread(target=active_request)
    waiting = threading.Thread(target=waiting_request)
    with admission.monitor_step():
        try:
            active.start()
            assert active_started.wait(1)
            readings["kv"] = 0.92
            waiting.start()
            deadline = time.monotonic() + 1
            while admission.step_summary()["admission_waits"] == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert admission.step_summary()["admission_waits"] == 1

            admission.suspend()
            admission.reset_step(preserve_pending_requests=True)
            assert admission.step_summary()["admitted_requests"] == 0
            readings["kv"] = 0.2
            requests_at_suspend = readings["metrics_requests"]
            time.sleep(0.05)
            assert readings["metrics_requests"] == requests_at_suspend

            admission.resume()
            assert waiting_admitted.wait(1)
        finally:
            readings["kv"] = 0.2
            release_active.set()
            if active.ident is not None:
                active.join(timeout=1)
            if waiting.ident is not None:
                waiting.join(timeout=1)

    assert not active.is_alive()
    assert not waiting.is_alive()


def test_abort_releases_suspended_episode_worker(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, _ = metrics_endpoint
    admission = LiveKVAdmission(endpoint, capacity_tokens=100)
    admission.suspend()

    def request() -> None:
        with admission.request(prompt_tokens=10, max_tokens=10):
            raise AssertionError("aborted worker was admitted")

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(request)
        admission.abort()
        with pytest.raises(RuntimeError, match="aborted"):
            worker.result(timeout=1)


@pytest.fixture
def metrics_endpoint() -> Iterator[tuple[str, dict[str, float]]]:
    readings = {"kv": 0.25, "running": 1.0, "waiting": 0.0, "metrics_requests": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/metrics"
            readings["metrics_requests"] += 1
            body = (
                f'vllm:kv_cache_usage_perc{{engine="0"}} {readings["kv"]}\n'
                f'vllm:num_requests_running{{engine="0"}} {readings["running"]}\n'
                f'vllm:num_requests_waiting{{engine="0"}} {readings["waiting"]}\n'
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            assert self.path == "/v1/completions"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": "synthetic",
                            "finish_reason": "stop",
                            "stop_reason": None,
                            "token_ids": [0, 0],
                            "prompt_token_ids": request["prompt"],
                            "logprobs": {"token_logprobs": [-0.1, -0.1]},
                        }
                    ],
                    "usage": {"prompt_tokens": len(request["prompt"]), "completion_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", readings
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_admission_waits_for_live_kv_and_releases_when_it_drops(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, target_usage=0.85, poll_seconds=0.01)
    admitted = threading.Event()

    def request() -> None:
        with admission.request(prompt_tokens=5, max_tokens=5):
            admitted.set()
        admission.finish_episode()

    with admission.request(prompt_tokens=5, max_tokens=5):
        assert admission.step_summary()["admitted_requests"] == 1
        readings["kv"] = 0.91
        thread = threading.Thread(target=request)
        thread.start()
        assert not admitted.wait(0.05)
        readings["kv"] = 0.5
        assert admitted.wait(1)
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert admission.step_summary()["admission_waits"] >= 1
    admission.finish_episode()


def test_one_request_can_make_progress_above_threshold(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings["kv"] = 0.0
    readings["running"] = 0.0
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, target_usage=0.85, poll_seconds=0.01)
    with admission.request(prompt_tokens=90, max_tokens=5):
        assert admission.step_summary()["admitted_requests"] == 1
    with admission.request(prompt_tokens=95, max_tokens=5):
        assert admission.step_summary()["admitted_requests"] == 2
    admission.finish_episode()


def test_new_episode_waits_when_an_untracked_request_uses_the_pool(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings["kv"] = 0.92
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, target_usage=0.85, poll_seconds=0.01)
    admitted = threading.Event()

    def request() -> None:
        with admission.request(prompt_tokens=5, max_tokens=5):
            admitted.set()
        admission.finish_episode()

    thread = threading.Thread(target=request)
    thread.start()
    assert not admitted.wait(0.05)
    readings["kv"] = 0.5
    assert admitted.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_unobserved_requests_reserve_their_full_token_footprint(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, target_usage=0.85, poll_seconds=0.01)
    first_admitted = threading.Event()
    second_admitted = threading.Event()
    release_first = threading.Event()

    def first_request() -> None:
        with admission.request(prompt_tokens=40, max_tokens=10):
            first_admitted.set()
            assert release_first.wait(1)
        admission.finish_episode()

    def second_request() -> None:
        with admission.request(prompt_tokens=40, max_tokens=10):
            second_admitted.set()
        admission.finish_episode()

    first = threading.Thread(target=first_request)
    second = threading.Thread(target=second_request)
    first.start()
    assert first_admitted.wait(1)
    second.start()
    try:
        assert not second_admitted.wait(0.05)
    finally:
        release_first.set()
    assert second_admitted.wait(1)
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()
    assert admission.step_summary()["max_live_prefix_fraction"] >= 0.5


def test_queued_server_request_blocks_oversized_first_request(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=1.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, target_usage=0.85, poll_seconds=0.01)
    admitted = threading.Event()

    def request() -> None:
        with admission.request(prompt_tokens=90, max_tokens=5):
            admitted.set()
        admission.finish_episode()

    thread = threading.Thread(target=request)
    thread.start()
    assert not admitted.wait(0.05)
    readings["waiting"] = 0.0
    assert admitted.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_completed_request_does_not_reserve_idle_prefix_when_live_kv_falls(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings["kv"] = 0.0
    readings["running"] = 0.0
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, target_usage=0.85, poll_seconds=0.01)
    with admission.request(prompt_tokens=55, max_tokens=10) as reservation:
        reservation.completed_tokens = 0
    readings["kv"] = 0.30
    admitted = threading.Event()

    def new_episode_turn() -> None:
        with admission.request(prompt_tokens=20, max_tokens=20):
            admitted.set()
        admission.finish_episode()

    thread = threading.Thread(target=new_episode_turn)
    thread.start()
    try:
        assert admitted.wait(0.25)
    finally:
        admission.finish_episode()
        thread.join(timeout=1)
    assert not thread.is_alive()


@pytest.mark.parametrize("failed_request", [False, True])
def test_completed_or_failed_requests_release_capacity_during_other_episode_tools(
    metrics_endpoint: tuple[str, dict[str, float]], failed_request: bool
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, poll_seconds=0.01)
    first_finished = threading.Event()
    finish_episode = threading.Event()

    def episode_in_tools() -> None:
        with admission.request(prompt_tokens=55, max_tokens=10) as reservation:
            reservation.completed_tokens = 5
        if failed_request:
            with (
                pytest.raises(ValueError, match="synthetic decode failure"),
                admission.request(prompt_tokens=60, max_tokens=10),
            ):
                raise ValueError("synthetic decode failure")
        first_finished.set()
        assert finish_episode.wait(2)
        admission.finish_episode()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(episode_in_tools)
        assert first_finished.wait(1)
        readings.update(kv=0.30, running=1.0)

        def new_episode() -> None:
            with admission.request(prompt_tokens=30, max_tokens=10) as reservation:
                reservation.completed_tokens = 5
            admission.finish_episode()

        second = pool.submit(new_episode)
        try:
            second.result(timeout=0.3)
            assert admission.step_summary()["admission_waits"] == 0
        finally:
            finish_episode.set()
        first.result(timeout=1)


def test_cached_continuation_reserves_blocks_that_must_become_live_again(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, poll_seconds=0.01)
    cached = threading.Event()
    continue_episode = threading.Event()
    continuation_admitted = threading.Event()

    def cached_episode() -> None:
        with admission.request(prompt_tokens=45, max_tokens=5) as reservation:
            reservation.completed_tokens = 0
        cached.set()
        assert continue_episode.wait(1)
        with admission.request(prompt_tokens=45, max_tokens=10):
            continuation_admitted.set()
        admission.finish_episode()

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(cached_episode)
        assert cached.wait(1)
        with admission.request(prompt_tokens=30, max_tokens=10):
            readings.update(kv=0.30, running=1.0)
            continue_episode.set()
            assert not continuation_admitted.wait(0.1)
        readings.update(kv=0.0, running=0.0)
        worker.result(timeout=1)
        assert continuation_admitted.is_set()
    admission.finish_episode()


def test_admission_deadline_releases_waiter_without_submitting(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.95, running=1.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, poll_seconds=0.01)
    with (
        pytest.raises(TimeoutError, match="KV admission"),
        admission.request(prompt_tokens=10, max_tokens=10, timeout_seconds=0.05),
    ):
        pytest.fail("request was submitted after its deadline")
    readings.update(kv=0.0, running=0.0)
    with admission.request(prompt_tokens=10, max_tokens=10):
        pass
    assert admission.step_summary()["admitted_requests"] == 1


def test_observed_active_kv_is_not_added_twice(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, poll_seconds=0.01)
    with admission.request(prompt_tokens=40, max_tokens=10):
        readings.update(kv=0.45, running=1.0)

        def overlapping_request() -> None:
            with admission.request(prompt_tokens=20, max_tokens=10, timeout_seconds=0.2):
                pass
            admission.finish_episode()

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(overlapping_request).result(timeout=1)
    assert admission.step_summary()["admission_waits"] == 0


def test_admission_deadline_also_bounds_a_suspended_server(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    admission = LiveKVAdmission(endpoint, capacity_tokens=100)
    admission.suspend()
    with (
        pytest.raises(TimeoutError, match="KV admission"),
        admission.request(prompt_tokens=10, max_tokens=10, timeout_seconds=0.05),
    ):
        pytest.fail("suspended request was submitted")
    assert readings["metrics_requests"] == 0
    admission.resume()


def test_idle_prefix_uses_actual_completion_length_not_full_headroom(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, poll_seconds=0.01)
    with admission.request(prompt_tokens=50, max_tokens=20) as reservation:
        reservation.completed_tokens = 2
    assert admission.step_summary()["max_live_prefix_fraction"] == pytest.approx(0.7)
    admitted = threading.Event()

    def next_episode() -> None:
        with admission.request(prompt_tokens=20, max_tokens=10):
            admitted.set()
        admission.finish_episode()

    worker = threading.Thread(target=next_episode)
    worker.start()
    assert admitted.wait(1)
    worker.join(timeout=1)
    admission.finish_episode()


def test_mtp_sized_pool_does_not_serialize_cached_episode_turns(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=305_192, poll_seconds=0.01)
    ready = threading.Barrier(15, timeout=5)
    release = threading.Event()
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def episode() -> None:
        nonlocal active, peak_active
        with admission.request(prompt_tokens=14_000, max_tokens=2_048) as reservation:
            reservation.completed_tokens = 1_000
        ready.wait()
        with admission.request(prompt_tokens=16_000, max_tokens=2_048):
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            assert release.wait(5)
            with lock:
                active -= 1
        admission.finish_episode()

    workers = [threading.Thread(target=episode) for _ in range(15)]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + 5
    while peak_active < 10 and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert peak_active >= 10


def test_oversized_episode_progresses_without_other_live_episodes(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, poll_seconds=0.01)
    with admission.request(prompt_tokens=90, max_tokens=20) as reservation:
        reservation.completed_tokens = 1
    with admission.request(prompt_tokens=95, max_tokens=20):
        pass
    admission.finish_episode()


def test_long_cached_waiter_does_not_block_a_smaller_fitting_request(
    metrics_endpoint: tuple[str, dict[str, float]],
) -> None:
    endpoint, readings = metrics_endpoint
    readings.update(kv=0.0, running=0.0, waiting=0.0)
    admission = LiveKVAdmission(endpoint, capacity_tokens=100, poll_seconds=0.01)
    start_next = threading.Event()
    long_admitted = threading.Event()
    small_admitted = threading.Event()

    def long_episode() -> None:
        with admission.request(prompt_tokens=70, max_tokens=10) as reservation:
            reservation.completed_tokens = 0
        assert start_next.wait(1)
        with admission.request(prompt_tokens=70, max_tokens=20):
            long_admitted.set()
        admission.finish_episode()

    def small_episode() -> None:
        with admission.request(prompt_tokens=5, max_tokens=5):
            small_admitted.set()
        admission.finish_episode()

    long_worker = threading.Thread(target=long_episode)
    long_worker.start()
    deadline = time.monotonic() + 1
    while admission.step_summary()["admitted_requests"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    with admission.request(prompt_tokens=5, max_tokens=1) as reservation:
        reservation.completed_tokens = 0
        start_next.set()
        assert not long_admitted.wait(0.05)
        small_worker = threading.Thread(target=small_episode)
        small_worker.start()
        assert small_admitted.wait(0.5)
    admission.finish_episode()
    assert long_admitted.wait(1)
    for worker in (small_worker, long_worker):
        worker.join(timeout=1)
        assert not worker.is_alive()


@pytest.mark.parametrize("fail_episode", [False, True])
def test_teacher_group_admission_cleans_prefixes_and_persists_summary(
    metrics_endpoint: tuple[str, dict[str, float]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_episode: bool,
) -> None:
    endpoint, _readings = metrics_endpoint
    admission = LiveKVAdmission(endpoint, capacity_tokens=100)
    monkeypatch.setattr(generate, "MINIMUM_FREE_DISK_BYTES", 0)

    backend = _synthetic_http_backend(endpoint, monkeypatch)
    backend.request_admission = admission

    class Source(SyntheticEpisodeSource):
        def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
            _complete_twice(backend)
            if fail_episode:
                raise ValueError("synthetic episode failed after decode")
            return super().__call__(task_id, seed=seed, step=step, replica=replica)

    def run() -> dict[str, object]:
        return generate.generate_episodes(
            output_dir=tmp_path / "output",
            source=Source(tmp_path),
            task_specs=(generate.TaskSpec("synthetic", "aisi", "oneoff"),),
            samples_per_task=1,
            concurrency=1,
            provenance={"run": "synthetic"},
            admission=admission,
        )

    if fail_episode:
        with pytest.raises(ValueError, match="failed after decode"):
            run()
    else:
        assert run()["kv_admission"] == admission.step_summary()
    assert not admission._prefix_tokens  # pyright: ignore[reportPrivateUsage]
    assert admission.step_summary()["admitted_requests"] == 2
    summary = json.loads((tmp_path / "output" / "summary.json").read_text())
    assert summary["kv_admission"]["admitted_requests"] == 2


@pytest.mark.parametrize("fail_episode", [False, True])
def test_ladder_group_admission_reaches_backend_and_cleans_worker_prefixes(
    metrics_endpoint: tuple[str, dict[str, float]],
    tmp_path: Path,
    fail_episode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint, _readings = metrics_endpoint
    config = replace(
        runner_config(tmp_path / "records.jsonl"),
        endpoint=endpoint,
        kv_admission_capacity_tokens=100,
    )
    backend = _synthetic_http_backend(endpoint, monkeypatch)

    def episode_runner(
        _scenario: ScenarioLike,
        active_backend: object,
        _arm: str,
        _sample_index: int,
        _episode_dir: Path,
        _trace_path: Path,
    ) -> dict[str, object]:
        assert active_backend is backend
        _complete_twice(backend)
        if fail_episode:
            raise ValueError("synthetic ladder failed after decode")
        return {"true_pass": True}

    run_grid(
        config,
        backend_factory=lambda _model, _config: backend,
        episode_runner=episode_runner,
        detector=lambda **_kwargs: {"highest_rung": 0, "coverup": False},
        final_grader=lambda **_kwargs: {},
    )
    admission = cast("LiveKVAdmission", backend.request_admission)
    assert not admission._prefix_tokens  # pyright: ignore[reportPrivateUsage]
    assert admission.step_summary()["admitted_requests"] == 8
    records = read_jsonl(config.output_path)
    summaries = [
        row
        for row in records
        if row.get("record") == "ladder_run_summary" and "kv_admission" in row
    ]
    assert summaries[-1]["kv_admission"] == admission.step_summary()


def _synthetic_http_backend(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> model_backend.VLLMHTTPBackend:
    unknown_token = "[UNK]"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"synthetic": 0, unknown_token: 1}, unk_token=unknown_token)
        ),
        unk_token=unknown_token,
    )
    monkeypatch.setattr(
        model_backend.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer
    )
    return model_backend.VLLMHTTPBackend(
        "synthetic",
        base_url=endpoint.removesuffix("/v1"),
        sampling=model_backend.SamplingConfig(max_new_tokens=20),
        timeout_seconds=3.0,
    )


def _complete_twice(backend: model_backend.VLLMHTTPBackend) -> None:
    admission = backend.request_admission
    assert admission is not None
    for prompt_tokens in (10, 17):
        completion = backend.complete_token_ids([1] * prompt_tokens, max_tokens=20, seed=1)
        assert completion.completion_tokens == 2
        assert admission._prefix_tokens[threading.get_ident()] == prompt_tokens + 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("capacity", [100, 0, -1])
def test_episode_cli_validates_admission_capacity(tmp_path: Path, capacity: int) -> None:
    generate_args = [
        "--endpoint",
        "http://synthetic:8000",
        "--max-model-len",
        "100",
        "--output-dir",
        str(tmp_path),
        "--kv-admission-capacity-tokens",
        str(capacity),
    ]
    ladder_args = [
        "--server-log",
        str(tmp_path / "server.log"),
        "--endpoint",
        "http://synthetic:8000",
        "--model",
        "synthetic=/synthetic",
        "--scenario-root",
        str(tmp_path),
        "--kv-admission-capacity-tokens",
        str(capacity),
    ]
    if capacity > 0:
        assert generate.parse_args(generate_args).kv_admission_capacity_tokens == capacity
        assert (
            ladder_runner._parse_cli_args(  # pyright: ignore[reportPrivateUsage]
                ladder_runner._build_cli_parser(),
                ladder_args,  # pyright: ignore[reportPrivateUsage]
            ).kv_admission_capacity_tokens
            == capacity
        )
    else:
        with pytest.raises(SystemExit):
            generate.parse_args(generate_args)
        with pytest.raises(SystemExit):
            ladder_runner._parse_cli_args(  # pyright: ignore[reportPrivateUsage]
                ladder_runner._build_cli_parser(),
                ladder_args,  # pyright: ignore[reportPrivateUsage]
            )
