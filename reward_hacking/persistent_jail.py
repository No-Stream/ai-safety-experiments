"""One protected PID namespace per episode, with a fresh shell and no surviving children per turn.

The namespace init is non-dumpable before it reads any requests. Linux protects namespace PID1
from child signals without installed handlers; nondumpability protects its memory and protocol FDs.
Only the host writes requests. Malformed responses or a dead supervisor terminate the episode.
Namespace filesystem changes outside /work also persist, including /tmp and Python PATH shims.
Host-side graders must continue using fresh jails and absolute read-only interpreter paths.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import math
import os
import re
import selectors
import signal
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Self

from reward_hacking import jailed_poc
from reward_hacking.jailed_poc import CommandResult, JailResourceLimits

if TYPE_CHECKING:
    from types import TracebackType


GIT_PRODUCER_STATE_KEYS = frozenset(
    {
        "/opt/python/bin/git",
        "/opt/python/bin/cat",
        "/opt/python/bin/head",
        "/opt/python/bin/tail",
        "/etc/gitconfig",
        "/etc/gitattributes",
        "/etc/profile",
        "/etc/bash.bashrc",
        "/etc/profile.d",
        "/etc/bash.bash_logout",
        "/etc/ld.so.preload",
        "/etc/ld.so.cache",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
    }
)


class PersistentJailSupervisorError(RuntimeError):
    """The episode's persistent jail supervisor failed to complete its protocol."""


def _validated_git_producer_state(response: dict[str, Any]) -> dict[str, bool]:
    if frozenset(response) != GIT_PRODUCER_STATE_KEYS or any(
        type(value) is not bool for value in response.values()
    ):
        raise PersistentJailSupervisorError("invalid producer metadata response")
    return response


class PersistentJailTeardownError(PersistentJailSupervisorError):
    """A process still held the jail's diagnostic socket after namespace teardown."""


# This source executes through an isolated, read-only interpreter, never from writable /work.
_SUPERVISOR_SOURCE = r"""
import base64
import ctypes
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import time

if os.getpid() != 1:
    raise RuntimeError("persistent worker must be namespace PID1")
signal.signal(signal.SIGINT, signal.SIG_DFL)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
    raise RuntimeError("could not protect supervisor memory and FDs")
maximum = int(sys.argv[1])
environment = dict(os.environ)
control = socket.socket(fileno=os.dup(2))
control.send(b'\x00control\x00')
control.close()

def cleanup():
    deadline = time.monotonic() + 5
    while True:
        descendants = [int(name) for name in os.listdir('/proc') if name.isdecimal() and int(name) > 1]
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("could not reap command descendants")
        time.sleep(0.005)

def execute(request):
    launcher_source = (
        "import os, sys\n"
        "try:\n"
        " with open('/proc/self/oom_score_adj', 'w', encoding='ascii') as score:\n"
        "  score.write('1000')\n"
        "except OSError as error:\n"
        " os.write(2, f'resource-limits: could not set command oom_score_adj=1000: {error}\\n'.encode())\n"
        " raise SystemExit(125) from error\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n"
    )
    process = subprocess.Popen(
        [sys.executable, '-I', '-c', launcher_source,
         '/bin/bash', '-lc' if request['login_shell'] else '-c', request['command']],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd='/work', env=environment, close_fds=True, start_new_session=True,
    )
    captured = [bytearray(), bytearray()]
    truncated = False
    timed_out = False
    cleaned = False
    deadline = time.monotonic() + request['seconds']
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, 0)
        selector.register(process.stderr, selectors.EVENT_READ, 1)
        while selector.get_map():
            if not cleaned and (process.poll() is not None or time.monotonic() >= deadline):
                timed_out = process.poll() is None
                if timed_out:
                    process.kill()
                process.wait()
                cleanup()
                cleaned = True
            for key, _ in selector.select(0.02):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                remaining = maximum - len(captured[key.data])
                captured[key.data].extend(chunk[:remaining])
                truncated = truncated or len(chunk) > remaining
        if not cleaned:
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            cleanup()
    exit_code = 128 - process.returncode if process.returncode < 0 else process.returncode
    if exit_code == 137 and not timed_out:
        oom_note = (
            b"\n[command exited 137 (SIGKILL or cgroup OOM); episode supervisor remains available]"
        )
        if len(captured[1]) + len(oom_note) > maximum:
            captured[1] = captured[1][:maximum - len(oom_note)]
            truncated = True
        captured[1].extend(oom_note)
    return dict(exit_code=124 if timed_out else exit_code,
                stdout=base64.b64encode(captured[0]).decode('ascii'),
                stderr=base64.b64encode(captured[1]).decode('ascii'),
                timed_out=timed_out, truncated=truncated)

def git_producer_state():
    paths = (
        '/opt/python/bin/git', '/opt/python/bin/cat', '/opt/python/bin/head', '/opt/python/bin/tail',
        '/etc/gitconfig', '/etc/gitattributes', '/etc/profile', '/etc/bash.bashrc',
        '/etc/profile.d', '/etc/bash.bash_logout', '/etc/ld.so.preload', '/etc/ld.so.cache',
    )
    state = {path: os.path.lexists(path) for path in paths}
    for path, target in (('/bin', 'usr/bin'), ('/sbin', 'usr/sbin'),
                         ('/lib', 'usr/lib'), ('/lib64', 'usr/lib64')):
        state[path] = not os.path.islink(path) or os.readlink(path) != target
    return state

print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    response = git_producer_state() if request.get('operation') == 'git_producer_state' else execute(request)
    print(json.dumps(response), flush=True)
cleanup()
"""

_STOP_POLL_SECONDS = 0.1
_NAMESPACE_EXIT_TIMEOUT_SECONDS = 30.0
_OUTER_PROCESS_KILL_TIMEOUT_SECONDS = 30.0
logger = logging.getLogger(__name__)


def _seconds(duration: str) -> float:
    matched = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd]?)", duration)
    if matched is None:
        raise ValueError("timeout must be a positive duration in seconds, minutes, hours or days")
    seconds = float(matched[1]) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[matched[2]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    return seconds


def _append_optional_resource_limit(argv: list[str], option: str, value: str | None) -> None:
    if value is not None:
        argv.extend((option, value))


class PersistentJail:
    """Lazily start an episode jail; context exit tears it down even after caller exceptions.

    ``episode_seconds=None`` delegates episode timing to the caller. Every command still has a
    deadline. CPU, memory and task limits apply to the whole namespace throughout its lifetime.
    """

    def __init__(  # noqa: PLR0913
        self,
        episode_dir: Path,
        *,
        backend: str | None = None,
        ro_bind_paths: tuple[Path, ...] = (),
        cpus: int | None = None,
        mem_max: str | None = None,
        mem_swap_max: str | None = None,
        episode_seconds: float | None = 600,
        resource_limits: JailResourceLimits | None = None,
        stop_requested: threading.Event | None = None,
    ) -> None:
        """Record episode launch limits; namespace startup waits until the first command."""
        if episode_seconds is not None and (
            not math.isfinite(episode_seconds) or episode_seconds <= 0
        ):
            raise ValueError("episode_seconds must be finite and positive")
        self.episode_dir = episode_dir
        self.backend = backend
        self.ro_bind_paths = ro_bind_paths
        self.cpus = cpus
        self.mem_max = mem_max
        self.mem_swap_max = mem_swap_max
        self.episode_seconds = episode_seconds
        self.resource_limits = resource_limits
        self.stop_requested = stop_requested
        self._process: subprocess.Popen[bytes] | None = None
        self._deadline = 0.0
        self._closed = False
        self._buffer = bytearray()
        self._diagnostics = bytearray()
        self._stderr_reader: threading.Thread | None = None
        self._pidfd: int | None = None
        self._identity_ready = threading.Event()
        self._identity_lock = threading.RLock()
        self._diagnostic_socket: socket.socket | None = None
        self._diagnostic_sender_inode: int | None = None

    def __enter__(self) -> Self:
        """Enter the episode's lazy execution scope."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Tear down the namespace after success or an exception."""
        self.close()

    def _start(self) -> None:
        limits = self.resource_limits or jailed_poc.current_jail_resource_limits()
        argv = [str(jailed_poc.RESOURCE_LIMITS)]
        argv += limits.aggregate_arguments()
        argv.append("--preserve-processes-on-oom")
        if self.episode_seconds is not None:
            argv += ["-t", f"{self.episode_seconds}s"]
        if limits.mode == "advisory":
            argv.append("--advisory")
        if self.cpus is not None:
            argv += ["--cpus", str(self.cpus)]
        _append_optional_resource_limit(argv, "--mem-max", limits.command_memory_max(self.mem_max))
        _append_optional_resource_limit(argv, "--mem-swap-max", self.mem_swap_max)
        supervisor_source = _SUPERVISOR_SOURCE.encode()
        argv += [
            "--",
            str(jailed_poc.EPISODE_JAIL),
            "--episode-dir",
            str(self.episode_dir),
            "--as-pid-1",
        ]
        if self.backend is not None:
            argv += ["--backend", self.backend]
        for path in self.ro_bind_paths:
            argv += ["--ro-bind", str(path)]
        argv += [
            "--",
            jailed_poc.jail_python(),
            "-I",
            "-u",
            "-c",
            f"import sys; exec(sys.stdin.buffer.read({len(supervisor_source)}))",
            str(jailed_poc.MAXIMUM_OUTPUT_BYTES),
        ]
        self._deadline = (
            math.inf if self.episode_seconds is None else time.monotonic() + self.episode_seconds
        )
        self._diagnostic_socket, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._diagnostic_socket.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self._diagnostic_sender_inode = os.fstat(sender.fileno()).st_ino
        try:
            self._process = subprocess.Popen(  # noqa: S603
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sender.fileno(),
                start_new_session=True,
                bufsize=0,
            )
        finally:
            sender.close()
        self._stderr_reader = threading.Thread(
            target=self._drain_diagnostics,
            daemon=True,
        )
        self._stderr_reader.start()
        if self._process.stdin is None:
            raise RuntimeError("persistent jail has no startup pipe")
        os.set_blocking(self._process.stdin.fileno(), False)
        self._write_request(supervisor_source, jailed_poc.PROCESS_SETUP_MARGIN_SECONDS)
        if self._read_response(jailed_poc.PROCESS_SETUP_MARGIN_SECONDS) != {"ready": True}:
            raise RuntimeError("persistent jail did not confirm protected startup")
        if (
            not self._identity_ready.wait(jailed_poc.PROCESS_SETUP_MARGIN_SECONDS)
            or self._pidfd is None
        ):
            raise RuntimeError("persistent jail did not supply its kernel identity")

    def _drain_diagnostics(self) -> None:
        diagnostic_socket = self._diagnostic_socket
        if diagnostic_socket is None:
            raise RuntimeError("persistent jail has no diagnostic socket")
        while True:
            data, ancillary, _, _ = diagnostic_socket.recvmsg(
                65536, socket.CMSG_SPACE(struct.calcsize("3i"))
            )
            if not data:
                return
            if data == b"\x00control\x00" and not self._identity_ready.is_set():
                for level, kind, credentials in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS:
                        pid, _, _ = struct.unpack("3i", credentials)
                        with self._identity_lock:
                            self._pidfd = os.pidfd_open(pid)
                            self._identity_ready.set()
                            if self._closed:
                                self._kill_namespace()
                        break
            else:
                remaining = jailed_poc.MAXIMUM_OUTPUT_BYTES - len(self._diagnostics)
                self._diagnostics.extend(data[:remaining])

    def _write_request(self, data: bytes, seconds: float) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("persistent jail has no input pipe")
        deadline = min(self._deadline, time.monotonic() + seconds)
        remaining_data = memoryview(data)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdin, selectors.EVENT_WRITE)
            while remaining_data:
                self._raise_if_stopped()
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0 or not selector.select(
                    min(remaining_seconds, _STOP_POLL_SECONDS)
                ):
                    self._raise_if_stopped()
                    if remaining_seconds <= 0:
                        raise TimeoutError("persistent jail request deadline expired")
                    continue
                try:
                    written_bytes = os.write(process.stdin.fileno(), remaining_data)
                except BlockingIOError:
                    continue
                except BrokenPipeError as error:
                    diagnostic = self._diagnostics.decode("utf-8", errors="replace")
                    raise PersistentJailSupervisorError(
                        f"persistent jail supervisor exited while receiving a request: {diagnostic}"
                    ) from error
                remaining_data = remaining_data[written_bytes:]

    def _read_response(self, seconds: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("persistent jail has no output pipe")
        deadline = min(self._deadline, time.monotonic() + seconds)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                self._raise_if_stopped()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(min(remaining, _STOP_POLL_SECONDS)):
                    self._raise_if_stopped()
                    if remaining <= 0:
                        raise TimeoutError("persistent jail response deadline expired")
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    diagnostic = self._diagnostics.decode("utf-8", errors="replace")
                    raise PersistentJailSupervisorError(
                        f"persistent jail supervisor exited without a response: {diagnostic}"
                    )
                self._buffer.extend(chunk)
                if len(self._buffer) > 3 * jailed_poc.MAXIMUM_OUTPUT_BYTES:
                    raise RuntimeError("persistent jail response exceeded its output budget")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = remainder
        return json.loads(line)

    def git_producer_state(self) -> dict[str, bool]:
        """Observe only fixed namespace producer paths without spawning actor-controlled code."""
        if self._closed:
            raise RuntimeError("persistent jail is closed")
        self._raise_if_stopped()
        if self._process is None:
            self._start()
        seconds = jailed_poc.PROCESS_SETUP_MARGIN_SECONDS
        self._write_request(b'{"operation":"git_producer_state"}\n', seconds)
        try:
            response = self._read_response(seconds)
        except TimeoutError as error:
            raise PersistentJailSupervisorError(str(error)) from error
        return _validated_git_producer_state(response)

    def run(
        self,
        command: str,
        *,
        timeout: str = jailed_poc.DEFAULT_COMMAND_TIMEOUT,
        process_timeout_seconds: float | None = None,
        login_shell: bool = True,
    ) -> CommandResult:
        """Run a fresh shell, kill every descendant, and return bounded actor output."""
        if self._closed:
            raise RuntimeError("persistent jail is closed")
        self._raise_if_stopped()
        seconds = _seconds(timeout)
        if process_timeout_seconds is not None:
            if not math.isfinite(process_timeout_seconds) or process_timeout_seconds <= 0:
                raise ValueError("process_timeout_seconds must be finite and positive")
            seconds = min(seconds, process_timeout_seconds)
        else:
            seconds = min(seconds, jailed_poc.PROCESS_TIMEOUT_SECONDS)
        try:
            if self._process is None:
                self._start()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("persistent jail has no input pipe")  # noqa: TRY301
            request = {"command": command, "seconds": seconds, "login_shell": login_shell}
            self._write_request(json.dumps(request).encode() + b"\n", seconds)
            try:
                response = self._read_response(seconds + jailed_poc.REAPED_PIPE_JOIN_SECONDS)
            except TimeoutError as error:
                raise PersistentJailSupervisorError(str(error)) from error
            stdout = base64.b64decode(response["stdout"], validate=True).decode(
                "utf-8", errors="replace"
            )
            stderr = base64.b64decode(response["stderr"], validate=True).decode(
                "utf-8", errors="replace"
            )
            if response["truncated"]:
                stderr += (
                    f"\n[output truncated to {jailed_poc.MAXIMUM_OUTPUT_BYTES} bytes per stream]"
                )
            return CommandResult(
                command=command,
                exit_code=response["exit_code"],
                stdout=stdout,
                stderr=stderr,
                timed_out=response["timed_out"] or response["exit_code"] == jailed_poc.TIMEOUT_EXIT,
            )
        except BaseException as error:
            try:
                self.close()
            except BaseException as teardown_error:
                raise teardown_error from error
            raise

    def _raise_if_stopped(self) -> None:
        if self.stop_requested is not None and self.stop_requested.is_set():
            raise InterruptedError("persistent jail command stopped after an episode worker failed")

    def close(self) -> None:
        """Close the protocol, then terminate the outer limiter if normal teardown cannot finish."""
        self._closed = True
        process = self._process
        if process is None:
            if self._diagnostic_socket is not None:
                self._diagnostic_socket.close()
            return
        if process.stdin is not None:
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
        if self.stop_requested is not None and self.stop_requested.is_set():
            self._signal_process_group(signal.SIGTERM)
            self._kill_namespace()
        self._wait_for_outer_process(process)
        self._kill_namespace()
        if process.stdout is not None:
            process.stdout.close()
        try:
            if self._stderr_reader is not None:
                self._stderr_reader.join(timeout=jailed_poc.REAPED_PIPE_JOIN_SECONDS)
                if self._stderr_reader.is_alive():
                    self._raise_teardown_failure(self._stderr_reader)
        finally:
            if self._diagnostic_socket is not None:
                self._diagnostic_socket.close()

    def _raise_teardown_failure(self, reader: threading.Thread) -> NoReturn:
        # Every jail process inherits the sender as stderr, so a holder past namespace teardown is a
        # jail descendant (seen: a multi-GB command still dying from a cgroup OOM kill).
        holders = self._kill_diagnostic_holders()
        if self._diagnostic_socket is not None:
            self._diagnostic_socket.shutdown(socket.SHUT_RDWR)
        reader.join(timeout=jailed_poc.REAPED_PIPE_JOIN_SECONDS)
        message = (
            "persistent jail diagnostic pipe did not close after teardown; "
            f"holders killed: {holders or 'none found'}; reader stopped: {not reader.is_alive()}"
        )
        logger.warning(message)
        raise PersistentJailTeardownError(message)

    def _kill_diagnostic_holders(self) -> list[str]:
        if self._diagnostic_sender_inode is None:
            return []
        target = f"socket:[{self._diagnostic_sender_inode}]"
        holders: list[str] = []
        for process_dir in Path("/proc").iterdir():
            if not process_dir.name.isdigit():
                continue
            try:
                if not any(str(fd.readlink()) == target for fd in (process_dir / "fd").iterdir()):
                    continue
                state = (process_dir / "stat").read_text().rsplit(")", 1)[1].split()[0]
                command = (
                    (process_dir / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode(errors="replace")
                )
                os.kill(int(process_dir.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                # The process exited mid-scan, or belongs to another user and cannot hold our socket.
                continue
            holders.append(f"pid={process_dir.name} state={state} cmd={command.strip()[:200]}")
        return holders

    def _wait_for_outer_process(self, process: subprocess.Popen[bytes]) -> None:
        try:
            process.wait(timeout=jailed_poc.REAPED_PIPE_JOIN_SECONDS)
        except subprocess.TimeoutExpired:
            self._signal_process_group(signal.SIGTERM)
            self._kill_namespace()
            try:
                process.wait(timeout=jailed_poc.REAPED_PIPE_JOIN_SECONDS)
            except subprocess.TimeoutExpired:
                self._signal_process_group(signal.SIGKILL)
                started = time.monotonic()
                try:
                    process.wait(timeout=_OUTER_PROCESS_KILL_TIMEOUT_SECONDS)
                finally:
                    elapsed = time.monotonic() - started
                    if elapsed > jailed_poc.REAPED_PIPE_JOIN_SECONDS:
                        logger.warning(
                            "Persistent jail outer process SIGKILL wait took %.1f seconds", elapsed
                        )

    def _kill_namespace(self) -> None:
        with self._identity_lock:
            if self._pidfd is None:
                return
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(self._pidfd, signal.SIGKILL)
            started = time.monotonic()
            with selectors.DefaultSelector() as selector:
                selector.register(self._pidfd, selectors.EVENT_READ)
                exited = selector.select(_NAMESPACE_EXIT_TIMEOUT_SECONDS)
                elapsed = time.monotonic() - started
                if elapsed > jailed_poc.REAPED_PIPE_JOIN_SECONDS:
                    logger.warning(
                        "Persistent jail namespace SIGKILL wait took %.1f seconds", elapsed
                    )
                if not exited:
                    raise RuntimeError("persistent jail namespace did not exit after SIGKILL")
            os.close(self._pidfd)
            self._pidfd = None

    def _signal_process_group(self, signum: signal.Signals) -> None:
        if self._process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self._process.pid, signum)
