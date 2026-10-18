"""Response deadlines and loaded-kernel teardown bounds, without a loaded host."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from reward_hacking import persistent_jail as jail_module
from reward_hacking.persistent_jail import PersistentJail, PersistentJailSupervisorError

if TYPE_CHECKING:
    import selectors
    from pathlib import Path


def test_response_deadline_closes_before_supervisor_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executor = PersistentJail(tmp_path)
    executor._process = Mock(spec=subprocess.Popen, stdin=Mock())
    deadline_error = TimeoutError("persistent jail response deadline expired")
    monkeypatch.setattr(executor, "_write_request", Mock())
    monkeypatch.setattr(executor, "_read_response", Mock(side_effect=deadline_error))
    closed = Mock()
    monkeypatch.setattr(executor, "close", closed)

    with pytest.raises(PersistentJailSupervisorError, match="response deadline expired") as failure:
        executor.run("true")

    assert failure.value.__cause__ is deadline_error
    closed.assert_called_once_with()


@pytest.mark.parametrize(
    "original_error", [TimeoutError("response deadline"), ValueError("invalid response")]
)
def test_teardown_failure_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, original_error: Exception
) -> None:
    executor = PersistentJail(tmp_path)
    executor._process = Mock(spec=subprocess.Popen, stdin=Mock())
    teardown_error = RuntimeError("persistent jail namespace did not exit after SIGKILL")
    monkeypatch.setattr(executor, "_write_request", Mock())
    monkeypatch.setattr(executor, "_read_response", Mock(side_effect=original_error))
    monkeypatch.setattr(executor, "close", Mock(side_effect=teardown_error))

    with pytest.raises(RuntimeError, match="namespace did not exit") as failure:
        executor.run("true")

    assert failure.value is teardown_error
    cause = failure.value.__cause__
    if isinstance(original_error, TimeoutError):
        assert isinstance(cause, PersistentJailSupervisorError)
        assert cause.__cause__ is original_error
    else:
        assert cause is original_error


@pytest.mark.parametrize("exits", [True, False])
def test_namespace_sigkill_wait_allows_loaded_kernel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, exits: bool
) -> None:
    executor = PersistentJail(tmp_path)
    read_fd, write_fd = os.pipe()
    executor._pidfd = read_fd
    clock = [0.0]
    waits: list[float] = []
    monkeypatch.setattr(jail_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(jail_module.signal, "pidfd_send_signal", Mock())

    def select_exit(_selector: selectors.BaseSelector, timeout: float) -> list[object]:
        waits.append(timeout)
        clock[0] += min(6.0, timeout) if exits else timeout
        return [object()] if exits and timeout >= 6 else []

    monkeypatch.setattr(jail_module.selectors.DefaultSelector, "select", select_exit)
    try:
        with caplog.at_level(logging.WARNING):
            if exits:
                executor._kill_namespace()
                assert executor._pidfd is None
                assert "namespace" in caplog.text
                assert "SIGKILL" in caplog.text
            else:
                with pytest.raises(RuntimeError, match="namespace did not exit after SIGKILL"):
                    executor._kill_namespace()
                assert executor._pidfd == read_fd
        assert waits == [30.0]
    finally:
        if executor._pidfd == read_fd:
            os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("exits", [True, False])
def test_outer_sigkill_wait_allows_loaded_kernel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, exits: bool
) -> None:
    executor = PersistentJail(tmp_path)
    process = Mock(spec=subprocess.Popen)
    clock = [0.0]
    waits: list[float] = []
    monkeypatch.setattr(jail_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(executor, "_signal_process_group", Mock())
    monkeypatch.setattr(executor, "_kill_namespace", Mock())

    def wait_for_exit(*, timeout: float) -> int:
        waits.append(timeout)
        if len(waits) < 3 or not exits or timeout < 6:
            clock[0] += timeout
            raise subprocess.TimeoutExpired("synthetic outer launcher", timeout)
        clock[0] += 6
        return 0

    process.wait.side_effect = wait_for_exit
    with caplog.at_level(logging.WARNING):
        if exits:
            executor._wait_for_outer_process(process)
            assert "outer process" in caplog.text
            assert "SIGKILL" in caplog.text
        else:
            with pytest.raises(subprocess.TimeoutExpired):
                executor._wait_for_outer_process(process)
    assert waits == [5.0, 5.0, 30.0]
