"""Live vLLM KV admission against a local synthetic metrics server."""

import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reward_hacking.organism_rl.admission import LiveKVAdmission, kv_capacity_tokens


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


@pytest.fixture
def metrics_endpoint() -> Iterator[tuple[str, dict[str, float]]]:
    readings = {"kv": 0.25, "running": 1.0, "waiting": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/metrics"
            body = (
                f'vllm:kv_cache_usage_perc{{engine="0"}} {readings["kv"]}\n'
                f'vllm:num_requests_running{{engine="0"}} {readings["running"]}\n'
                f'vllm:num_requests_waiting{{engine="0"}} {readings["waiting"]}\n'
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


def test_waiting_episode_prefix_is_reserved_even_when_kv_gauge_falls(
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
    assert not admitted.wait(0.05)
    admission.finish_episode()
    assert admitted.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


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
    try:
        assert small_admitted.wait(0.5)
    finally:
        admission.finish_episode()
    assert long_admitted.wait(1)
    for worker in (small_worker, long_worker):
        worker.join(timeout=1)
        assert not worker.is_alive()
