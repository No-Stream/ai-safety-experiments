"""Real persistent-namespace execution and hostile descendant controls, using synthetic commands."""

from __future__ import annotations

import os
import selectors
import signal
import socket
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from reward_hacking import jailed_poc
from reward_hacking import persistent_jail as persistent_jail_module
from reward_hacking.jailed_poc import MAXIMUM_OUTPUT_BYTES, JailResourceLimits
from reward_hacking.persistent_jail import PersistentJail

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def persistent_jail() -> Iterator[PersistentJail]:
    with (
        tempfile.TemporaryDirectory(prefix="persistent-test-") as directory,
        PersistentJail(
            Path(directory),
            resource_limits=JailResourceLimits(mode="advisory"),
            cpus=1,
            mem_max="512M",
            episode_seconds=30,
        ) as executor,
    ):
        yield executor


def test_namespace_persists_but_shell_environment_does_not(persistent_jail: PersistentJail) -> None:
    first = persistent_jail.run(
        "readlink /proc/self/ns/pid; export MARKER=present; cd /tmp; echo kept >/tmp/kept"
    )
    second = persistent_jail.run(
        "readlink /proc/self/ns/pid; echo ${MARKER-absent}; pwd; cat /tmp/kept"
    )
    assert first.ok
    assert second.ok
    assert first.stdout.splitlines()[0] == second.stdout.splitlines()[0]
    assert second.stdout.splitlines()[1:] == ["absent", "/work", "kept"]


def test_signal_exit_matches_per_command_executor(persistent_jail: PersistentJail) -> None:
    # unshare executes the one-shot payload as PID 1, which ignores default SIGTERM.
    command = (
        "bash -c 'printf before; printf diagnostic >&2; kill -TERM $$'; status=$?; exit $status"
    )
    one_shot = jailed_poc.run_in_jail(
        persistent_jail.episode_dir,
        command,
        resource_limits=JailResourceLimits(mode="advisory"),
        cpus=1,
        mem_max="512M",
    )
    persistent = persistent_jail.run(command)
    assert one_shot.exit_code == 143
    assert persistent.exit_code == one_shot.exit_code
    assert persistent.stdout == one_shot.stdout == "before"
    assert persistent.stderr == one_shot.stderr
    assert persistent.stderr.startswith("diagnostic")
    assert persistent.command == one_shot.command == command
    assert persistent.timed_out is one_shot.timed_out is False


def test_direct_signal_exit_uses_shell_convention(persistent_jail: PersistentJail) -> None:
    result = persistent_jail.run("printf before; printf diagnostic >&2; kill -TERM $$")
    assert result.exit_code == 143
    assert result.stdout == "before"
    assert result.stderr == "diagnostic"
    assert result.timed_out is False


def test_timeout_cleans_detached_background_descendants(persistent_jail: PersistentJail) -> None:
    result = persistent_jail.run(
        "setsid sh -c 'sleep 1; touch /work/leaked' & sleep 30", timeout="0.1s"
    )
    assert result.timed_out
    assert result.exit_code == 124
    assert persistent_jail.run("sleep 1.2; test ! -e /work/leaked").ok


def test_normal_exit_also_cleans_descendants(persistent_jail: PersistentJail) -> None:
    assert persistent_jail.run("setsid sh -c 'sleep 1; touch /work/leaked' >/dev/null 2>&1 &").ok
    assert persistent_jail.run("sleep 1.2; test ! -e /work/leaked").ok


def test_supervisor_resists_signals_and_protocol_fd_access(persistent_jail: PersistentJail) -> None:
    result = persistent_jail.run(
        "kill -INT 1; kill -TERM 1; kill -KILL 1; "
        "python3 -I -c \"import os; os.open('/proc/1/fd/1', os.O_WRONLY)\""
    )
    assert not result.ok
    assert "PermissionError" in result.stderr
    assert persistent_jail.run("printf alive").stdout == "alive"


def test_output_is_bounded_and_worker_remains_usable(persistent_jail: PersistentJail) -> None:
    result = persistent_jail.run("python3 -I -c 'import os; os.write(1, b\"a\" * 2000000)' ")
    assert result.ok
    assert len(result.stdout.encode()) == MAXIMUM_OUTPUT_BYTES
    assert "output truncated" in result.stderr
    assert persistent_jail.run("printf next").stdout == "next"


def test_exception_closes_executor() -> None:
    with tempfile.TemporaryDirectory(prefix="persistent-test-") as directory:
        executor = PersistentJail(
            Path(directory), resource_limits=JailResourceLimits(mode="advisory")
        )
        with pytest.raises(LookupError, match="synthetic caller failure"):
            _raise_in_executor(executor)
        with pytest.raises(RuntimeError, match="closed"):
            executor.run("true")


def _raise_in_executor(executor: PersistentJail) -> None:
    with executor:
        assert executor.run("true").ok
        raise LookupError("synthetic caller failure")


def test_limiter_advisory_timeout_preserves_standard_input() -> None:
    result = subprocess.run(  # noqa: S603 - trusted limiter and synthetic standard input
        [
            str(jailed_poc.RESOURCE_LIMITS),
            "--advisory",
            "--cpus",
            "1",
            "--mem-max",
            "256M",
            "-t",
            "3s",
            "--",
            "/usr/bin/cat",
        ],
        input="synthetic transport message",
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout == "synthetic transport message"


def test_nondumpability_negative_control(monkeypatch: pytest.MonkeyPatch) -> None:
    unsafe_source = persistent_jail_module._SUPERVISOR_SOURCE.replace(
        "if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:",
        "if False:",
    )
    assert unsafe_source != persistent_jail_module._SUPERVISOR_SOURCE
    monkeypatch.setattr(persistent_jail_module, "_SUPERVISOR_SOURCE", unsafe_source)
    with (
        tempfile.TemporaryDirectory(prefix="persistent-test-") as directory,
        PersistentJail(
            Path(directory), resource_limits=JailResourceLimits(mode="advisory")
        ) as executor,
    ):
        result = executor.run(
            "python3 -I -c \"import os; os.close(os.open('/proc/1/fd/1', os.O_WRONLY))\""
        )
        assert result.ok


def test_host_pidfd_kills_a_stopped_supervisor(persistent_jail: PersistentJail) -> None:
    assert persistent_jail.run("true").ok
    assert persistent_jail._pidfd is not None
    observer_pidfd = os.dup(persistent_jail._pidfd)
    try:
        signal.pidfd_send_signal(observer_pidfd, signal.SIGSTOP)
        persistent_jail.close()
        with selectors.DefaultSelector() as selector:
            selector.register(observer_pidfd, selectors.EVENT_READ)
            assert selector.select(0)
    finally:
        os.close(observer_pidfd)


def test_request_write_deadline_with_stopped_supervisor(persistent_jail: PersistentJail) -> None:
    assert persistent_jail.run("true").ok
    assert persistent_jail._pidfd is not None
    observer_pidfd = os.dup(persistent_jail._pidfd)
    try:
        signal.pidfd_send_signal(observer_pidfd, signal.SIGSTOP)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(persistent_jail.run, "#" + "x" * 200000, timeout="0.1s")
            try:
                error = future.exception(timeout=7)
            except TimeoutError:
                signal.pidfd_send_signal(observer_pidfd, signal.SIGKILL)
                future.exception(timeout=10)
                pytest.fail("request pipe write ignored its deadline")
        assert isinstance(error, TimeoutError)
    finally:
        os.close(observer_pidfd)


def test_failed_launch_closes_diagnostic_sockets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_socketpair = socket.socketpair
    sockets: list[socket.socket] = []

    def recording_socketpair(*args: int) -> tuple[socket.socket, socket.socket]:
        pair = original_socketpair(*args)
        sockets.extend(pair)
        return pair

    def failed_popen(*args: object, **kwargs: object) -> NoReturn:
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(persistent_jail_module.socket, "socketpair", recording_socketpair)
    monkeypatch.setattr(
        persistent_jail_module.jailed_poc, "jail_python", lambda: "/usr/bin/python3"
    )
    monkeypatch.setattr(persistent_jail_module.subprocess, "Popen", failed_popen)
    with (
        PersistentJail(tmp_path) as executor,
        pytest.raises(OSError, match="synthetic launch failure"),
    ):
        executor.run("true")
    assert len(sockets) == 2
    assert all(channel.fileno() == -1 for channel in sockets)
