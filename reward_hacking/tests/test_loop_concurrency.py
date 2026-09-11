"""Episodes that share one local engine, and the deadline that ends a generation that never returns.

The run this module is written against: 17 tasks x 4 repeats on a rented H100, ``--backend vllm
--episode-concurrency 8``. Two minutes of real turns, then a process with 65 threads, 59 of them in
``futex_do_wait``, GPU at 0%, 25 s of CPU against 73 minutes of wall clock, and not one episode
summary in the trace. Twice.

The mechanism, read off the installed vLLM 0.27.1 rather than guessed. ``LLM.generate`` adds its
requests to the engine and then drains with ``while llm_engine.has_unfinished_requests(): step()``.
The predicate is ``len(output_processor.request_states) > 0`` -- shared by every caller on the
engine -- and ``step()`` blocks on ``outputs_queue.get()`` with no timeout. So with N episode threads
each inside their own ``generate`` on one engine, all N loop on the same shared state; whichever
thread happens to consume the batch carrying the final finish sees nothing unfinished and returns
(with whichever outputs it collected, not necessarily its own), and every other thread is left in
``get()`` on a queue nothing will ever feed again. The sibling ``generate_streaming`` refuses exactly
this case ("another caller is driving the same engine") and documents that one submission owns the
engine from entry to exit; the batched path had no such guard, and the harness's per-episode
deadline was checked only between actions and between turns, so it could not reach a thread blocked
inside the engine.

The fake here is that path with the model removed and nothing else changed: add requests, then drain
one shared output queue until nothing is unfinished. One caller at a time it is correct; two callers
at once deadlock, deterministically, the way the real one does. Every test that expects the loop to
return runs it on a thread and gives up after a bounded wait, because the failure this module exists
for is a hang, and a hang inside a test is a hung suite rather than a red one.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.harness import loop
from reward_hacking.harness.hack_detector import traces_from_records
from reward_hacking.harness.loop import (
    EPISODE_START_RECORD,
    DeadlinePhase,
    EpisodeOutcome,
    RunStoppedError,
    SharedEngineGate,
    load_traces,
    run_agent_episode,
    run_tasks,
    stopping_on_signals,
)
from reward_hacking.harness.tasks import TASKS_BY_ID
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from collections.abc import Callable

_LEDGER = TASKS_BY_ID["sum-ledger"]
_READINGS = TASKS_BY_ID["max-reading"]

_HANG_SECONDS = 15.0
"""How long a test waits for the loop before calling it hung: seconds of margin for a loaded shared
box around runs that script to well under two."""

_FIRST_TURN_MARK = "first turn of"
_POISON = "poison"


def _scripted_reply(prompt: str) -> str:
    """Turn one runs a command that names the task; turn two, seeing turn one echoed back, stops.

    Keyed on the task's own data file, which is the one thing in the prompt that differs between the
    two tasks, so a reply filed under the wrong episode is visible as a reply naming the wrong task.
    """
    task = "ledger" if "ledger.txt" in prompt else "readings"
    if f"{_FIRST_TURN_MARK} {task}" in prompt:
        return f"Done with {task}."
    return f"{_FIRST_TURN_MARK} {task}\n<run>printf '{task}' > answer.txt</run>"


class _SharedQueueEngine:
    """vLLM's offline ``generate`` with the model removed: the part that deadlocks, nothing else.

    ``_unfinished`` is ``OutputProcessor.request_states``; ``_outputs`` is
    ``SyncMPClient.outputs_queue``; the ``fake-EngineCore`` thread is the ``EngineCore`` subprocess,
    emitting one output per request after a fixed delay. :meth:`generate` adds its requests and then
    drains until NOTHING is unfinished, taking whatever comes off the queue -- which is exactly what
    ``LLM._run_engine`` does. A caller whose own output was consumed by a sibling raises, as the real
    backend's echoed-prompt check does; a caller left draining after the last finish blocks forever.

    :meth:`unwedge` exists for the test teardown: on code that deadlocks, the stuck threads belong to
    a ``ThreadPoolExecutor`` and would otherwise be joined at interpreter exit, turning a red test into
    a hung suite.
    """

    def __init__(self, reply: Callable[[str], str], *, seconds_per_reply: float) -> None:
        self._reply = reply
        self._seconds_per_reply = seconds_per_reply
        self._state_lock = threading.Lock()
        self._unfinished: dict[str, str] = {}
        self._outputs: queue.Queue[tuple[str, str]] = queue.Queue()
        self._inbox: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._ids = itertools.count()
        self.callers_inside = 0
        self.max_callers_inside = 0
        self.generate_intervals: list[tuple[float, float]] = []
        self._core = threading.Thread(target=self._core_loop, name="fake-EngineCore", daemon=True)
        self._core.start()

    def _core_loop(self) -> None:
        while (item := self._inbox.get()) is not None:
            request_id, prompt = item
            time.sleep(self._seconds_per_reply)
            self._outputs.put((request_id, self._reply(prompt)))

    def _has_unfinished(self) -> bool:
        with self._state_lock:
            return bool(self._unfinished)

    def generate(self, prompts: list[str]) -> list[str]:
        started = time.monotonic()
        with self._state_lock:
            self.callers_inside += 1
            self.max_callers_inside = max(self.max_callers_inside, self.callers_inside)
        try:
            request_ids: list[str] = []
            for prompt in prompts:
                request_id = str(next(self._ids))
                with self._state_lock:
                    self._unfinished[request_id] = prompt
                self._inbox.put((request_id, prompt))
                request_ids.append(request_id)
            collected: dict[str, str] = {}
            while self._has_unfinished():
                # SyncMPClient.get_output: a blocking queue read with no timeout.
                request_id, text = self._outputs.get()
                if request_id == _POISON:
                    raise RuntimeError("unwedged by the test teardown")
                with self._state_lock:
                    self._unfinished.pop(request_id, None)
                collected[request_id] = text
            missing = [request_id for request_id in request_ids if request_id not in collected]
            if missing:
                raise RuntimeError(
                    f"another caller consumed this submission's replies (requests {missing}); the "
                    f"real backend's echoed-prompt check raises here"
                )
            return [collected[request_id] for request_id in request_ids]
        finally:
            with self._state_lock:
                self.callers_inside -= 1
            self.generate_intervals.append((started, time.monotonic()))

    def unwedge(self, callers: int) -> None:
        """Free every caller left draining: clear the shared state and feed each a poison output."""
        with self._state_lock:
            self._unfinished.clear()
        for _ in range(callers):
            self._outputs.put((_POISON, ""))

    def close(self) -> None:
        self._inbox.put(None)


class _SharedEngineBackend:
    """A ``Backend`` over the fake engine, declaring the transport that shares one in-process engine."""

    transport = "vllm"

    def __init__(self, engine: _SharedQueueEngine, model_id: str = "fake-vllm") -> None:
        self.model_id = model_id
        self._engine = engine

    def generate(self, prompts: list[str]) -> list[str]:
        return self._engine.generate(prompts)


class _HostedBackend:
    """A ``Backend`` that tolerates concurrent callers, the way a hosted API does, and counts them."""

    transport = "bedrock"

    def __init__(self, *, seconds_per_reply: float) -> None:
        self.model_id = "fake-hosted"
        self._seconds_per_reply = seconds_per_reply
        self._lock = threading.Lock()
        self.callers_inside = 0
        self.max_callers_inside = 0

    def generate(self, prompts: list[str]) -> list[str]:
        with self._lock:
            self.callers_inside += 1
            self.max_callers_inside = max(self.max_callers_inside, self.callers_inside)
        try:
            time.sleep(self._seconds_per_reply)
            return [_scripted_reply(prompt) for prompt in prompts]
        finally:
            with self._lock:
                self.callers_inside -= 1


class _BlockedBackend:
    """A ``Backend`` whose generate never returns until the test releases it."""

    transport = "vllm"

    def __init__(self) -> None:
        self.model_id = "fake-blocked"
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompts: list[str]) -> list[str]:
        self.entered.set()
        self.release.wait()
        return ["<run>true</run>" for _ in prompts]


def _stub_the_jail(
    monkeypatch: pytest.MonkeyPatch, *, seconds_per_command: float = 0.0
) -> list[tuple[str, float, float]]:
    """Answer every jailed command after a fixed delay, recording when each one ran."""
    intervals: list[tuple[str, float, float]] = []

    def fake_run_in_jail(directory: Path, command: str, **kwargs: object) -> CommandResult:
        started = time.monotonic()
        time.sleep(seconds_per_command)
        intervals.append((command, started, time.monotonic()))
        return CommandResult(command=command, exit_code=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(loop, "run_in_jail", fake_run_in_jail)
    return intervals


def _finishes_within(
    target: Callable[[], object], *, seconds: float
) -> tuple[bool, object | BaseException | None]:
    """Run ``target`` on a daemon thread; report whether it returned in time, and what it produced.

    The loop under test is expected to hang on the bug this module is about, so the wait is bounded
    and the thread is a daemon: a hang reads as a red assertion rather than a suite that never ends.
    """
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(target())
        except BaseException as exc:  # noqa: BLE001 - handed to the test thread, which asserts on it
            outcome.append(exc)

    thread = threading.Thread(target=run, name="loop-under-test", daemon=True)
    thread.start()
    thread.join(seconds)
    return (not thread.is_alive()), (outcome[0] if outcome else None)


def _overlaps(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _sigterm_self_and_wait() -> None:
    """Send this process SIGTERM and give the handler a moment to run on the main thread."""
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(1.0)


class TestConcurrentEpisodesShareOneEngineWithoutDeadlocking:
    """N episode threads, one local engine: generation is serialised, everything else overlaps."""

    def test_two_episodes_finish_with_their_own_replies_and_generation_never_overlaps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The reproduction. On the unguarded loop this run never returns."""
        engine = _SharedQueueEngine(_scripted_reply, seconds_per_reply=0.3)
        jail = _stub_the_jail(monkeypatch, seconds_per_command=0.2)
        backend = _SharedEngineBackend(engine)

        finished, outcome = _finishes_within(
            lambda: run_tasks(
                backend, [_LEDGER, _READINGS], episode_base=tmp_path, episode_concurrency=2
            ),
            seconds=_HANG_SECONDS,
        )
        try:
            assert finished, (
                f"run_tasks did not return within {_HANG_SECONDS}s: two episodes driving one "
                f"engine at once are deadlocked in its output queue"
            )
        finally:
            engine.unwedge(callers=2)
            engine.close()
        if isinstance(outcome, BaseException):
            raise outcome
        traces = cast("list[loop.AgentEpisodeTrace]", outcome)

        by_task = {trace.task_id: trace for trace in traces}
        assert set(by_task) == {_LEDGER.task_id, _READINGS.task_id}
        for trace, name in (
            (by_task[_LEDGER.task_id], "ledger"),
            (by_task[_READINGS.task_id], "readings"),
        ):
            assert [turn.completion for turn in trace.turns] == [
                f"{_FIRST_TURN_MARK} {name}\n<run>printf '{name}' > answer.txt</run>",
                f"Done with {name}.",
            ], f"the {name} episode was handed another episode's reply"
            assert trace.deadline_exceeded is False
        # The contract the streaming path already enforces: one submission owns the engine.
        assert engine.max_callers_inside == 1
        # What the concurrency still buys: one episode's sandbox command ran during the other's turn.
        policy_commands = [span for command, *span in jail if command.startswith("printf")]
        assert any(
            _overlaps((started, ended), generation)
            for started, ended in policy_commands
            for generation in engine.generate_intervals
        ), (
            f"no sandbox command overlapped a generation: {policy_commands} vs {engine.generate_intervals}"
        )

    def test_a_hosted_transport_keeps_true_concurrency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only the transports that share an in-process engine are serialised."""
        _stub_the_jail(monkeypatch)
        backend = _HostedBackend(seconds_per_reply=0.3)

        finished, outcome = _finishes_within(
            lambda: run_tasks(
                backend, [_LEDGER, _READINGS], episode_base=tmp_path, episode_concurrency=2
            ),
            seconds=_HANG_SECONDS,
        )
        assert finished
        if isinstance(outcome, BaseException):
            raise outcome
        assert backend.max_callers_inside == 2

    def test_concurrency_one_takes_no_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The serial path every existing artifact was produced under is untouched."""
        engine = _SharedQueueEngine(_scripted_reply, seconds_per_reply=0.01)
        _stub_the_jail(monkeypatch)
        try:
            traces = run_tasks(
                _SharedEngineBackend(engine),
                [_LEDGER],
                episode_base=tmp_path,
                episode_concurrency=1,
            )
        finally:
            engine.close()
        assert len(traces) == 1
        assert traces[0].engine_wait_seconds == 0.0


class TestTheDeadlineEndsAGenerationThatNeverReturns:
    """The 900-second budget used to be a comment about generation; now it is a bound on it."""

    def test_a_blocked_generation_is_abandoned_at_the_budget_and_recorded_as_such(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_the_jail(monkeypatch)
        backend = _BlockedBackend()
        try:
            finished, outcome = _finishes_within(
                lambda: run_agent_episode(
                    _LEDGER, backend, episode_dir=tmp_path / "blocked", episode_seconds=0.3
                ),
                seconds=_HANG_SECONDS,
            )
            assert backend.entered.is_set()
            assert finished, "the episode sat inside a generation that never returned"
        finally:
            backend.release.set()
        if isinstance(outcome, BaseException):
            raise outcome
        trace = cast("loop.AgentEpisodeTrace", outcome)

        assert trace.deadline_exceeded is True
        assert trace.deadline_phase is DeadlinePhase.SAMPLING
        assert trace.turns == ()
        assert trace.outcome is EpisodeOutcome.INCOMPLETE
        deadline = cast("dict[str, Any]", trace.summary_record()["deadline"])
        assert deadline["exceeded"] is True
        assert deadline["phase"] == "sampling"
        assert "deadline_phase=sampling" in trace.summary_line()

    def test_the_rebuilt_trace_carries_the_phase_and_an_older_record_reads_as_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_the_jail(monkeypatch)
        backend = _BlockedBackend()
        try:
            finished, outcome = _finishes_within(
                lambda: run_agent_episode(
                    _LEDGER, backend, episode_dir=tmp_path / "rebuilt", episode_seconds=0.2
                ),
                seconds=_HANG_SECONDS,
            )
            assert finished
        finally:
            backend.release.set()
        trace = cast("loop.AgentEpisodeTrace", outcome)
        summary = trace.summary_record()
        (rebuilt,) = traces_from_records([summary])
        assert rebuilt.deadline_phase is DeadlinePhase.SAMPLING
        assert rebuilt.outcome is EpisodeOutcome.INCOMPLETE

        older = dict(summary)
        older["deadline"] = {"seconds": 0.2, "exceeded": True}
        (rebuilt_older,) = traces_from_records([older])
        assert rebuilt_older.deadline_exceeded is True
        assert rebuilt_older.deadline_phase is None
        assert rebuilt_older.engine_wait_seconds == 0.0


class TestWaitingForTheSharedEngineIsTheRunsCostNotTheEpisodes:
    """Queueing behind a sibling's generation is not this episode's spend, and is never unbounded."""

    def test_queue_time_is_not_charged_to_the_budget(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Queued 0.4 s, then 0.3 s of its own command, under a 0.6 s budget: wall clock over, spend under."""
        _stub_the_jail(monkeypatch, seconds_per_command=0.3)
        gate = SharedEngineGate()
        assert gate.acquire(timeout=1.0)  # a sibling episode is generating
        threading.Timer(0.4, gate.release).start()

        trace = run_agent_episode(
            _LEDGER,
            MockBackend(["<run>true</run>", "Done."], model_id="mock-queued"),
            episode_dir=tmp_path / "queued",
            episode_seconds=0.6,
            engine_gate=gate,
        )

        assert trace.elapsed_seconds is not None
        assert trace.elapsed_seconds > 0.6
        assert trace.deadline_exceeded is False
        assert trace.deadline_phase is None
        assert trace.engine_wait_seconds >= 0.35
        assert len(trace.turns) == 2
        deadline = cast("dict[str, Any]", trace.summary_record()["deadline"])
        assert deadline["engine_wait_seconds"] == pytest.approx(trace.engine_wait_seconds)

    def test_an_engine_that_never_frees_up_ends_the_episode_as_engine_queue(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_the_jail(monkeypatch)
        gate = SharedEngineGate()
        assert gate.acquire(timeout=1.0)  # and it is never released
        try:
            finished, outcome = _finishes_within(
                lambda: run_agent_episode(
                    _LEDGER,
                    MockBackend(["Done."], model_id="mock-starved"),
                    episode_dir=tmp_path / "starved",
                    episode_seconds=0.2,
                    engine_gate=gate,
                ),
                seconds=_HANG_SECONDS,
            )
        finally:
            gate.release()
        assert finished
        if isinstance(outcome, BaseException):
            raise outcome
        trace = cast("loop.AgentEpisodeTrace", outcome)
        assert trace.deadline_exceeded is True
        assert trace.deadline_phase is DeadlinePhase.ENGINE_QUEUE
        assert trace.turns == ()
        assert trace.engine_wait_seconds >= 0.2


class TestATraceGrowsWhileTheEpisodeRuns:
    """A frozen episode leaves its start and every finished turn on disk, not nothing."""

    def test_the_start_and_each_finished_turn_are_on_disk_before_the_summary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_the_jail(monkeypatch)
        out = tmp_path / "trace.jsonl"
        seen_at_each_call: list[list[dict[str, object]]] = []

        def reply(prompt: str) -> str:
            seen_at_each_call.append(load_traces(out) if out.exists() else [])
            return "<run>true</run>" if len(seen_at_each_call) == 1 else "Done."

        trace = run_agent_episode(
            _LEDGER,
            MockBackend(reply, model_id="mock-growing"),
            episode_dir=tmp_path / "growing",
            trace_path=out,
        )

        # What the second generation saw on disk: the start and turn one, no summary yet.
        kinds_before_second_turn = [record["record"] for record in seen_at_each_call[1]]
        assert kinds_before_second_turn == [EPISODE_START_RECORD, "turn"]
        start = seen_at_each_call[1][0]
        assert start["episode_id"] == trace.episode_id
        assert start["task_id"] == _LEDGER.task_id
        assert start["deadline_seconds"] == trace.deadline_seconds

        # The finished file: start, both turns, summary -- each turn written once.
        records = load_traces(out)
        assert [record["record"] for record in records] == [
            EPISODE_START_RECORD,
            "turn",
            "turn",
            "episode_summary",
        ]
        (rebuilt,) = traces_from_records(records)
        assert len(rebuilt.turns) == 2
        assert rebuilt.episode_id == trace.episode_id


class TestAStopRequestUnwindsARunMidGeneration:
    """SIGTERM must reach an episode blocked inside the engine, or the engine is orphaned."""

    def test_run_tasks_raises_promptly_when_stopped_during_a_blocked_generation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _stub_the_jail(monkeypatch)
        backend = _BlockedBackend()
        stop_requested = threading.Event()
        threading.Timer(0.3, stop_requested.set).start()
        started = time.monotonic()
        try:
            finished, outcome = _finishes_within(
                lambda: run_tasks(
                    backend,
                    [_LEDGER, _READINGS],
                    episode_base=tmp_path,
                    episode_concurrency=2,
                    episode_seconds=600.0,
                    stop_requested=stop_requested,
                ),
                seconds=_HANG_SECONDS,
            )
        finally:
            backend.release.set()
        assert backend.entered.is_set()
        assert finished
        assert time.monotonic() - started < 5.0, "the stop took longer than the budget it bypasses"
        assert isinstance(outcome, RunStoppedError)

    def test_sigterm_sets_the_stop_flag_and_raises_system_exit(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            pytest.skip("signal handlers install on the main thread only")
        stop_requested = threading.Event()
        previous = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as raised, stopping_on_signals(stop_requested):
            _sigterm_self_and_wait()
        assert raised.value.code == 128 + signal.SIGTERM
        assert stop_requested.is_set()
        assert signal.getsignal(signal.SIGTERM) is previous


_DEAF_CHILD = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"
"""A child that ignores SIGTERM, so "terminated" and "force-killed" are distinguishable deaths."""

_REAPER_PROBE = """
import json, subprocess, sys, time
from reward_hacking.harness.loop import live_child_pids, live_child_processes, terminate_child_processes

scenario, grace, deaf_child = sys.argv[1], float(sys.argv[2]), sys.argv[3]
report = {"children_at_start": live_child_pids()}
child = None
if scenario == "sleep":
    child = subprocess.Popen(["sleep", "300"])
elif scenario == "deaf":
    child = subprocess.Popen([sys.executable, "-c", deaf_child])
if child is not None:
    report["pid"] = child.pid
    report["listed_before"] = child.pid in live_child_pids()
    report["name_before"] = live_child_processes().get(child.pid)
started = time.monotonic()
report["reaped"] = terminate_child_processes(grace_seconds=grace)
report["elapsed"] = time.monotonic() - started
if child is not None:
    report["exit_status"] = child.wait(timeout=10)
    report["listed_after"] = child.pid in live_child_pids()
print(json.dumps(report))
"""
"""One reaping scenario, run in an interpreter whose only children are its own; see the probe helper."""


def _reaper_probe(scenario: str, *, grace_seconds: float) -> dict[str, Any]:
    """Run one reaping scenario in a FRESH interpreter and return what it reports.

    Never in the test process. ``terminate_child_processes`` sweeps every live child of the process
    that calls it, and under ``make test`` this file shares a pytest-xdist worker with several other
    test files (``--dist loadfile``), some of which have subprocesses of their own in flight. Called
    in the worker, the reaper killed a co-tenant's subprocess and then failed its own "nothing to
    reap" assertion against that pid -- once observed as ``assert [55670] == []`` on one worker of
    eight, passing alone -- and would have surfaced the kill as an unrelated failure elsewhere. A
    child interpreter has exactly the children the scenario gives it, so what it reaps is only ever
    its own.
    """
    root = Path(loop.__file__).resolve().parents[2]
    completed = subprocess.run(  # noqa: S603 - this interpreter, a literal program, fixed arguments
        [sys.executable, "-c", _REAPER_PROBE, scenario, str(grace_seconds), _DEAF_CHILD],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, f"the reaper probe died:\n{completed.stderr[-3000:]}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


class TestChildProcessesAreReapedAtExit:
    """Whatever this process spawned and never waited for dies with it, engine core included.

    Both halves are asserted through the signal each child actually died of, because "the child is
    gone" alone cannot tell a graceful teardown from a force-kill -- and the difference is the whole
    point: vLLM's ``EngineCore`` releases the card in its own SIGTERM handler, so a reaper that went
    straight to SIGKILL would leave the driver to clean up after a process killed mid-teardown,
    which is how VRAM gets orphaned in the first place.

    Every scenario runs through :func:`_reaper_probe`, in its own interpreter, for the reason that
    helper's docstring gives: the reaper under test kills every child of whoever calls it.
    """

    def test_a_live_child_is_listed_and_asked_to_stop_before_it_is_killed(self) -> None:
        report = _reaper_probe("sleep", grace_seconds=5.0)
        assert report["children_at_start"] == []
        assert report["listed_before"] is True
        assert report["name_before"] == "sleep"
        assert report["reaped"] == [report["pid"]]
        # -15: it died of the SIGTERM, so a real engine child would have run its own teardown.
        assert report["exit_status"] == -signal.SIGTERM
        assert report["listed_after"] is False

    def test_a_child_that_ignores_the_request_is_killed_after_the_grace(self) -> None:
        report = _reaper_probe("deaf", grace_seconds=1.0)
        assert report["reaped"] == [report["pid"]]
        assert report["exit_status"] == -signal.SIGKILL
        assert report["elapsed"] >= 1.0, "the grace was not waited out"
        assert report["listed_after"] is False

    def test_nothing_to_reap_is_an_empty_list(self) -> None:
        report = _reaper_probe("nothing", grace_seconds=0.1)
        assert report["children_at_start"] == []
        assert report["reaped"] == []
