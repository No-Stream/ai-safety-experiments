"""Persistent actor jails stop promptly when another episode worker fails."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from reward_hacking import persistent_jail as persistent_jail_module
from reward_hacking.harness.loop import run_tasks
from reward_hacking.harness.task_spec import Task
from reward_hacking.jailed_poc import JailResourceLimits
from reward_hacking.persistent_jail import PersistentJail

_PROCESS_SENTINEL = "persistent_abort_process_sentinel"
_RUN_TOKEN = "abort-cleanup"


class _FailAfterCommandStartsBackend:
    transport = "bedrock"
    model_id = "synthetic-abort-backend"

    def __init__(self, command_started_path: Path) -> None:
        self.command_started_path = command_started_path

    def generate(self, prompts: list[str]) -> list[str]:
        prompt = prompts[0]
        if "synthetic slow episode" in prompt:
            command = (
                "<run>printf started > command-started; "
                "bash -c 'exec -a persistent_abort_process_sentinel sleep 30' & "
                "wait; touch command-finished"
            )
            return [f"{command}</run>"]
        if "synthetic failing episode" in prompt:
            deadline = time.monotonic() + 5
            while not self.command_started_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not self.command_started_path.exists():
                raise AssertionError(
                    "the long actor command did not start before the sibling failed"
                )
            raise RuntimeError("synthetic episode worker failure")
        raise AssertionError("backend received an unknown synthetic episode")


def _synthetic_task(task_id: str, prompt: str) -> Task:
    return Task(
        task_id=task_id,
        prompt=prompt,
        body=prompt,
        files={"answer.txt": ""},
        grader_command="true",
        true_check=lambda _episode_dir: False,
    )


def _has_process_sentinel() -> bool:
    for process_directory in Path("/proc").iterdir():
        if not process_directory.name.isdecimal():
            continue
        try:
            command_line = (process_directory / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        if _PROCESS_SENTINEL.encode() in command_line:
            return True
    return False


def _wait_for_process_exit(timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _has_process_sentinel():
            return True
        time.sleep(0.02)
    return not _has_process_sentinel()


def _wait_for_executor_exit(executors: list[PersistentJail], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all(
            executor._process is None or executor._process.poll() is not None
            for executor in executors
        ):
            return True
        time.sleep(0.02)
    return all(
        executor._process is None or executor._process.poll() is not None for executor in executors
    )


@pytest.mark.parametrize("failing_index", [0, 1])
def test_worker_failure_stops_sibling_persistent_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing_index: int
) -> None:
    failing_task = _synthetic_task("synthetic-failing", "synthetic failing episode")
    slow_task = _synthetic_task("synthetic-slow", "synthetic slow episode")
    tasks = [failing_task, slow_task] if failing_index == 0 else [slow_task, failing_task]
    slow_index = tasks.index(slow_task)
    slow_episode_dir = tmp_path / f"synthetic-slow.{_RUN_TOKEN}.{slow_index:03d}"
    command_started_path = slow_episode_dir / "command-started"
    backend = _FailAfterCommandStartsBackend(command_started_path)
    executors: list[PersistentJail] = []
    original_init = PersistentJail.__init__

    def record_executor(self: PersistentJail, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        executors.append(self)

    monkeypatch.setattr(persistent_jail_module.PersistentJail, "__init__", record_executor)

    try:
        with pytest.raises(RuntimeError, match="synthetic episode worker failure"):
            run_tasks(
                backend,
                tasks,
                episode_base=tmp_path,
                run_token=_RUN_TOKEN,
                timeout="4s",
                episode_concurrency=2,
                resource_limits=JailResourceLimits(mode="advisory"),
                jail_backend="unshare",
                persistent_jail=True,
            )

        assert _wait_for_process_exit(timeout_seconds=1.0), (
            "the sibling actor process survived the episode worker failure"
        )
        assert len(executors) == 1
        assert _wait_for_executor_exit(executors, timeout_seconds=1.0), (
            "the outer jail process survived abort"
        )
        assert not (slow_episode_dir / "command-finished").exists()
    finally:
        _wait_for_process_exit(timeout_seconds=6.0)
        deadline = time.monotonic() + 6.0
        while (
            any(
                executor._process is not None and executor._process.poll() is None
                for executor in executors
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
