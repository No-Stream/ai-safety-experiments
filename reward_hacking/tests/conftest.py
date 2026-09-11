"""Shared test scaffolding: the jail probe, the synthetic-episode builders, the RecoveryBench item.

Three unrelated things live here, each because more than one module needs it.

The first is whether this box can run the episode jail at all. That answer is not free -- it shells
out to systemd and to the jail script -- so the probe runs once here at collection time.
``test_jailed_poc`` and ``test_agent_harness`` each carried a byte-identical copy of it before it
moved here. :func:`jail_user_site_dir` belongs to the same group: it is where the jail's interpreter
looks for a user-site module, which two of the exploit suites plant into and neither may hardcode.

The second is the trace builders. ``hack_detector`` is a pure function over a saved trace, so every
test of it assembles an :class:`AgentEpisodeTrace` by hand the way ``run_agent_episode`` would have.
``test_hack_detector`` (the hardcode-versus-honest-bug vocabulary) and ``test_grader_tampering``
(interference with the machinery that decides the verdict) both need exactly that, so it lives here
rather than being copied into the second module the moment it was written.

The third is :func:`recovery_item`, the synthetic RecoveryBench item all three
``test_recoverybench_*`` modules build on. It is trivial arithmetic on purpose: the repository is
public, so no real item text, planted flaw or registered answer may appear anywhere in the tests.

The fourth is :class:`StubDetailedBackend`, which both runners' tests need in order to reach the
field mapping every real transport goes through and nothing previously exercised.

The fifth is :data:`MEASURED_MIN_OUTPUT_BUDGET`, the output-token floor both the Converse config
tests and the agent-harness CLI tests assert their default cap against. It is one number about the
models rather than about either module, and duplicating it is how one of the two copies ends up
lenient.

The sixth is not scaffolding but a collection gate, :class:`CaseFileAwareModule`. The harness task
registry reads ``harness/data/*_cases.json`` at import and crashes when it cannot, and those files
are local-only (see the README beside them), so in a fresh clone every module here that imports the
harness would error before a single test ran -- including this file, which is why the harness
imports it needs are made inside the two functions that use them rather than at the top.
"""

from __future__ import annotations

import errno
import functools
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import pytest

from reward_hacking.jailed_poc import CommandResult, classify, jail_python
from reward_hacking.model_backend import BedrockCompletion, TokenUsage
from reward_hacking.recoverybench.answers import AnswerShape, Normalization
from reward_hacking.recoverybench.arms import Arm
from reward_hacking.recoverybench.items import Domain, FlawType, GradingMode, RecoveryItem

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from reward_hacking.harness.loop import AgentEpisodeTrace, AgentTurn
    from reward_hacking.harness.task_spec import AgenticArmConfig


class NoJailInterpreterError(RuntimeError):
    """``episode_jail.sh`` resolved no interpreter, so nothing that grades in the jail can run.

    Its own type so the collector below can turn it into a module-level skip without reading a
    message, the same way :func:`_absent_case_file_reason` narrows by construction.
    """


@functools.cache
def jail_user_site_dir() -> PurePosixPath:
    """Where the jail's interpreter looks for a user-site module, relative to ``HOME``.

    The jail sets ``HOME=/work``, so this is the directory a policy plants a ``usercustomize.py`` in
    to have code imported before the checker's first line -- the exploit ``python3 -I`` and the
    oracle's staging filter exist to stop. Asked of the interpreter rather than written out: the
    version component moves with whatever ``episode_jail.sh`` resolved, and a stale ``python3.9``
    would land the plant somewhere the live interpreter never looks, leaving an exploit test that
    passes because the exploit no longer applies rather than because the defence holds.
    """
    probe_home = PurePosixPath("/jail-user-site-probe")
    try:
        interpreter = jail_python()
    except RuntimeError as exc:
        raise NoJailInterpreterError(" ".join(str(exc).split())) from exc
    completed = subprocess.run(  # noqa: S603 - resolved interpreter path, literal argument
        [interpreter, "-c", "import site; print(site.getusersitepackages())"],
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(probe_home)},
        check=True,
    )
    return PurePosixPath(completed.stdout.strip()).relative_to(probe_home)


def _absent_case_file_reason(exc: BaseException) -> str | None:
    """Why to skip a module whose import failed only because a baked case file is missing.

    Narrow by construction rather than by trusting the exception type or reading its message. Both
    loaders wrap the ``OSError`` they caught with ``raise ... from exc``, so this can insist on a
    chained cause whose errno is ``ENOENT`` and whose filename is a case file. A case file that
    exists but does not parse still errors the whole suite, which is the right outcome: that is a
    broken file, not a fresh clone.

    Getting this wrong in the lenient direction would be the worst kind of bug this repo has --
    every harness test quietly skipping while the run stays green -- so the condition is the
    narrowest one that catches the fresh clone, and it has been watched to fail both ways.
    """
    cause = exc.__cause__
    if not isinstance(cause, OSError) or cause.errno != errno.ENOENT:
        return None
    filename = str(cause.filename)
    if not filename.endswith("_cases.json"):
        return None
    return (
        f"baked case file absent: {filename} -- it is local-only and regenerated by the ETLs, see "
        "reward_hacking/harness/data/README.md"
    )


class CaseFileAwareModule(pytest.Module):
    """A test module that skips, rather than erroring, when the host is missing a prerequisite.

    Two of them, both absent in a fresh clone or on a fresh box and neither the module's fault: a
    baked case file, and an interpreter for the jail. The skip surfaces with its reason under the
    ``-ra`` this repo runs pytest with, so the run is told what is missing and how to get it instead
    of watching collection abort.
    """

    def collect(self) -> Iterable[pytest.Item | pytest.Collector]:
        """Import and collect the module, converting two specific import failures into skips."""
        try:
            return list(super().collect())
        except NoJailInterpreterError as exc:
            pytest.skip(f"no jail interpreter resolved: {exc}", allow_module_level=True)
        except RuntimeError as exc:
            reason = _absent_case_file_reason(exc)
            if reason is None:
                raise
            pytest.skip(reason, allow_module_level=True)


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.Module:
    """Collect every test module in this directory through the case-file-aware collector."""
    return CaseFileAwareModule.from_parent(parent, path=module_path)


_USABLE_SYSTEMD_STATES = frozenset({"running", "degraded", "starting", "maintenance", "stopping"})


def jail_prereqs() -> tuple[bool, str]:
    """Whether this box can run the jail at all; reason string when it cannot.

    The interpreter half asks ``episode_jail.sh`` rather than checking a path: the jail resolves the
    first candidate past its version floor and refuses to start when there is none, so the host
    having ``/usr/bin/python3`` says nothing about whether a jailed grader can run.
    """
    if shutil.which("bwrap") is None:
        return False, "bwrap not installed"
    try:
        jail_python()
    except RuntimeError as exc:
        return False, f"episode_jail.sh resolved no interpreter: {' '.join(str(exc).split())}"
    if not os.environ.get("XDG_RUNTIME_DIR"):
        return False, "XDG_RUNTIME_DIR unset (resource-limits.sh cannot reach the systemd user bus)"
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return False, "systemctl not on PATH"
    state = subprocess.run(  # noqa: S603 - resolved absolute path, literal arguments
        [systemctl, "--user", "is-system-running"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if state not in _USABLE_SYSTEMD_STATES:
        return False, f"no usable systemd user instance (state={state!r})"
    return True, ""


JAIL_AVAILABLE, JAIL_UNAVAILABLE_REASON = jail_prereqs()

needs_jail = pytest.mark.skipif(
    not JAIL_AVAILABLE, reason=f"jail unavailable: {JAIL_UNAVAILABLE_REASON}"
)

MEASURED_MIN_OUTPUT_BUDGET = 24_576
"""The smallest output budget any hosted model here was *measured* to need, as a test floor.

A small reasoning model spends 11-19k tokens of thinking on one hard item before it reaches an
answer line, and a hosted model's reasoning is billed against the output cap whether or not the
reasoning comes back. Below this a reply is cut off before its answer, and a truncated reply is
indistinguishable downstream from a refusal to answer.

Written as a literal rather than imported from ``model_backend``, because a floor that reads the
value it is bounding is a tautology that passes at any setting -- which is exactly how a 2048-token
default survived every test in this suite while truncating 37% of a 400-episode run's calls. Kept in
step with ``recoverybench.budgets`` by hand.
"""


def command_result(command: str, *, exit_code: int = 0, stdout: str = "") -> CommandResult:
    """One recorded command, as the jail would have returned it.

    ``exit_code`` defaults to success because most fixture commands are meant to have worked; the
    tampering tests pass a failure explicitly, since a plant that did not run planted nothing.
    """
    return CommandResult(
        command=command, exit_code=exit_code, stdout=stdout, stderr="", timed_out=False
    )


def episode_trace(  # noqa: PLR0913 - a fixture builder mirroring the trace's own field set
    task_id: str,
    turns: tuple[AgentTurn, ...],
    *,
    episode_dir: Path,
    proxy_pass: bool,
    true_pass: bool,
    arm: AgenticArmConfig | None = None,
) -> AgentEpisodeTrace:
    """Assemble a trace the way ``run_agent_episode`` would, without running anything.

    ``agent_saw_grader_success`` and ``episode_id`` are both derived through the loop's own helpers
    rather than restated here, so a fixture cannot record a pass the real loop would not have, nor
    carry an id shaped differently from a real one. ``arm`` defaults to the baseline (spelled
    ``None`` only because evaluating ``BASELINE_ARM`` at ``def`` time would import the harness at
    collection, which a fresh clone cannot do -- see the module docstring). The id is composed from
    the requested arm here, in the one function that derives ids, because the id names the arm: a
    re-armed trace carrying a baseline-labelled id says one thing in the record and another in the
    key everything groups by.

    ``visible_check_unsatisfiable`` is copied off the task for the same reason: the loop reads it
    there, and the detector reads it off the trace rather than looking the task up again, so a fixture
    that left it false would quietly disarm the impossibility theorem for every test built here.
    """
    # Not at the top: that would crash this conftest in a fresh clone (see the module docstring).
    from reward_hacking.harness.loop import (  # noqa: PLC0415
        AgentEpisodeTrace,
        compose_episode_id,
        new_episode_token,
        saw_grader_success,
    )
    from reward_hacking.harness.task_spec import BASELINE_ARM  # noqa: PLC0415
    from reward_hacking.harness.tasks import TASKS_BY_ID  # noqa: PLC0415

    resolved_arm = BASELINE_ARM if arm is None else arm
    task = TASKS_BY_ID[task_id]
    model_id = "mock-detector"
    return AgentEpisodeTrace(
        task_id=task_id,
        model_id=model_id,
        episode_id=compose_episode_id(
            model_id=model_id,
            task_id=task_id,
            arm=resolved_arm,
            episode_token=new_episode_token(),
        ),
        episode_dir=episode_dir,
        turns=turns,
        transcript="\n".join(turn.completion for turn in turns),
        proxy_grader_command=task.grader_command,
        proxy_grader_stdout=f"{task.success_marker}\n" if proxy_pass else "FAIL\n",
        proxy_pass=proxy_pass,
        true_pass=true_pass,
        agent_saw_grader_success=saw_grader_success(task, turns),
        verdict=classify(
            proxy_pass=proxy_pass,
            true_pass=true_pass,
            visible_check_unsatisfiable=task.visible_check_unsatisfiable,
        ),
        arm=resolved_arm,
        visible_check_unsatisfiable=task.visible_check_unsatisfiable,
    )


@pytest.fixture
def sum_ledger_dir(tmp_path: Path) -> Path:
    """The smoke episode's /work as it was left behind: a real task layout holding ``total=0``."""
    # Not at the top: that would crash this conftest in a fresh clone (see the module docstring).
    from reward_hacking.harness.loop import lay_down_task  # noqa: PLC0415
    from reward_hacking.harness.tasks import TASKS_BY_ID  # noqa: PLC0415

    task = TASKS_BY_ID["sum-ledger"]
    episode_dir = tmp_path / "sum-ledger"
    lay_down_task(episode_dir, task)
    (episode_dir / "answer.txt").write_text("total=0\n")
    # The fixture is wrong by the robust check, not merely asserted to be.
    assert not task.true_check(episode_dir)
    return episode_dir


TOY_RECOVERY_ARMS: dict[Arm, str] = {
    Arm.BARE: "What is 2+3?",
    Arm.CORRUPTED: "A colleague wrote 2+3=6. What is 2+3?",
}


def recovery_item(**overrides: object) -> RecoveryItem:
    """A valid synthetic RecoveryBench item, with any field overridden per test.

    Trivial arithmetic whose flawed path answers 6 to 2+3, so a test names the outcome it wants
    without any real benchmark material. ``**overrides`` rather than one keyword per field because
    every test varies a different two or three of the fourteen.
    """
    fields: dict[str, object] = {
        "item_id": "toy-sum",
        "domain": Domain.MATH,
        "flaw_type": FlawType.WRONG_METHOD,
        "grading_mode": GradingMode.CLOSED_ANSWER,
        "arms": dict(TOY_RECOVERY_ARMS),
        "source": "synthetic test fixture",
        "license_terms": "none, authored in this test",
        "answer_shape": AnswerShape.INTEGER,
        "true_answer": "5",
        "flawed_answer": "6",
        "normalization": Normalization.NUMBER,
    }
    fields.update(overrides)
    return RecoveryItem(**fields)  # pyright: ignore[reportArgumentType]


def fenced_answer(value: str) -> str:
    """One reply that honours the delimiter contract exactly, answering ``value``."""
    return f"Some reasoning first.\n\n```jagged\nanswer: {value}\n```"


class StubDetailedBackend:
    """A backend that reports reasoning, token counts and a stop reason, for the mapping tests.

    Nothing else in the tree exercises the ``DetailedBackend`` branch of ``generate_raw`` or the
    field mapping in ``raw_response`` -- every other test drives ``MockBackend`` or a stub without
    ``generate_detailed``, so that branch never ran and the only assertions about those fields were
    that they came back as ``None``. That mapping is on the critical path of both real transports.

    The default completion is deliberately asymmetric and distinguishable: 11 input tokens against 7
    output, "the answer" against "the thinking". Any transposed pair then fails by construction,
    which matters because the type checker cannot see that seam at all -- ``str`` to ``str``,
    ``int`` to ``int | None`` and ``None`` to ``str | None`` all check out. Sabotaging
    ``raw_response`` to swap ``text`` with ``reasoning`` and the two counts left the whole suite
    byte-identically green.

    ``generate`` stays on it so it still satisfies ``Backend`` under basedpyright strict, and
    ``generate_detailed`` is deliberately NOT added to ``MockBackend``: that would push the two
    existing None-path tests into the detailed branch and break them.
    """

    model_id = "stub-detailed"
    transport = "stub-detailed"

    def __init__(self, completions: Sequence[BedrockCompletion]) -> None:
        self.completions = list(completions)

    def generate(self, prompts: list[str]) -> list[str]:
        return [completion.text for completion in self.completions[: len(prompts)]]

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        assert len(prompts) == len(self.completions), (
            f"stub holds {len(self.completions)} completions for {len(prompts)} prompts"
        )
        return list(self.completions)


def detailed_completion(
    text: str = "the answer",
    *,
    reasoning: str = "the thinking",
    input_tokens: int = 11,
    output_tokens: int = 7,
    stop_reason: str = "max_tokens",
) -> BedrockCompletion:
    """One Converse-shaped completion whose every field is distinguishable from every other."""
    return BedrockCompletion(
        text=text,
        reasoning=reasoning,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


class ScriptedDetailedBackend:
    """A detailed backend over a callable or round-robin script, with SCRIPTED latency and telemetry.

    ``latency`` maps a prompt to the seconds one call sleeps before answering and is also written
    onto the completion as ``elapsed_seconds``, so two runs over the same prompts produce identical
    records whatever the scheduler did -- which is what lets a test compare the streaming and the
    per-chunk persistence paths byte for byte, timestamps aside. ``fail_on`` names the prompts whose
    call raises like a request bug, so the failure path can be driven deterministically.

    This class has no ``submit_stream``, so :func:`reward_hacking.model_backend.stream_detailed_in_chunks`
    takes its one-call-per-chunk fallback over it; :class:`ScriptedStreamingBackend` adds the
    streaming seam on top of the same script.
    """

    transport = "scripted"

    def __init__(  # noqa: PLR0913 - one keyword per scripted behaviour
        self,
        responses: Callable[[str], str] | Sequence[str],
        *,
        latency: Callable[[str], float] | None = None,
        fail_on: Iterable[str] = (),
        concurrency: int = 2,
        model_id: str = "scripted",
        stop_reason: str = "end_turn",
    ) -> None:
        if not callable(responses) and not responses:
            raise ValueError("ScriptedDetailedBackend needs a non-empty script or a callable")
        self.model_id = model_id
        self.concurrency = concurrency
        self.stop_reason = stop_reason
        self._responses = responses
        self._latency = latency or (lambda _prompt: 0.0)
        self._fail_on = frozenset(fail_on)
        self._cursor = 0
        self._lock = threading.Lock()
        self.started: list[str] = []
        self.finished: list[str] = []
        self.usage = TokenUsage()

    def _text_for(self, prompt: str) -> str:
        if callable(self._responses):
            return self._responses(prompt)
        with self._lock:
            text = self._responses[self._cursor % len(self._responses)]
            self._cursor += 1
        return text

    def _call(self, prompt: str) -> BedrockCompletion:
        with self._lock:
            self.started.append(prompt)
        seconds = self._latency(prompt)
        if seconds:
            time.sleep(seconds)
        if prompt in self._fail_on:
            raise RuntimeError(f"scripted request bug on {prompt}")
        text = self._text_for(prompt)
        billed = TokenUsage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)
        with self._lock:
            self.finished.append(prompt)
            self.usage += billed
        return BedrockCompletion(
            text=text,
            reasoning="",
            usage=billed,
            stop_reason=self.stop_reason,
            elapsed_seconds=seconds,
            first_event_seconds=seconds,
            attempts=1,
        )

    def _stream(self, prompts: Iterable[str]) -> Iterator[tuple[int, BedrockCompletion]]:
        """Run every prompt through a pool ``concurrency`` wide, yielding in completion order.

        On a scripted failure the remaining queued calls are cancelled, every finished one is still
        yielded, and the error then propagates -- the contract the runners' partial-chunk handover
        is written against.
        """
        error: BaseException | None = None
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._call, prompt): index for index, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                # A cancelled future is "completed" to as_completed, and asking it for its
                # exception raises CancelledError in place of the scripted error the test expects.
                if future.cancelled():
                    continue
                exception = future.exception()
                if exception is not None:
                    error = exception
                    for pending in futures:
                        pending.cancel()
                    continue
                yield futures[future], future.result()
        if error is not None:
            raise error

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        by_index = dict(self._stream(prompts))
        return [by_index[index] for index in range(len(prompts))]

    def generate(self, prompts: list[str]) -> list[str]:
        return [completion.text for completion in self.generate_detailed(prompts)]


class ScriptedStreamingBackend(ScriptedDetailedBackend):
    """The scripted backend with the streaming seam, so the runners take their continuous-queue path.

    Same ``transport`` label as its parent on purpose: the two exist to be run over the same script
    and compared record for record, and the only difference between them is the seam.
    """

    def submit_stream(self, prompts: Iterable[str]) -> Iterator[tuple[int, BedrockCompletion]]:
        return self._stream(prompts)
