"""Attack a persistent episode jail and require isolation and descendant cleanup.

Run as a module with the repository's host interpreter. The ordinary jail assertion
suite is also executed through the persistent transport, using the same host probes
as the one-shot gate. No credential contents or benchmark material are read.
"""

# ruff: noqa: INP001
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shlex
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Tuple  # noqa: UP035 - Python 3.8 host scripts

from reward_hacking.jailed_poc import CommandResult, JailResourceLimits, jail_python
from reward_hacking.persistent_jail import PersistentJail

logger = logging.getLogger(__name__)
TIMEOUT_EXIT_CODE = 124
OOM_EXIT_CODE = 137
OOM_STDERR_MARKER = "cgroup OOM"
MEMORY_BOMB_SOURCE = (
    "import resource\n"
    "block_size = 16 * 1024 * 1024\n"
    "blocks = []\n"
    "print('starting 2048 MiB anonymous allocation', flush=True)\n"
    "for _ in range(128):\n"
    "    block = bytearray(block_size)\n"
    "    for offset in range(0, block_size, 4096):\n"
    "        block[offset] = 1\n"
    "    blocks.append(block)\n"
    "print(f'completed 2048 MiB; peak RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KiB', flush=True)\n"
)


def require_command(result: CommandResult, name: str, expected: str = "") -> None:
    """Require both command success and its independently specified output."""
    if not result.ok or result.stdout.strip() != expected:
        raise RuntimeError(
            f"persistent check {name} failed: exit={result.exit_code} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    logger.info("persistent check passed: %s", name)


def require_memory_oom_recovery(run_command: Callable[[str], CommandResult]) -> None:
    """Require a bounded memory command to die alone and the same episode to recover."""
    memory_bomb_command = shlex.quote(jail_python()) + " -I -c " + shlex.quote(MEMORY_BOMB_SOURCE)
    oom_result = run_command(memory_bomb_command)
    if (
        oom_result.exit_code != OOM_EXIT_CODE
        or oom_result.timed_out
        or OOM_STDERR_MARKER not in oom_result.stderr
    ):
        raise RuntimeError(
            "persistent memory-bomb check did not return the OOM command result: "
            f"exit={oom_result.exit_code} timed_out={oom_result.timed_out} "
            f"stdout={oom_result.stdout!r} stderr={oom_result.stderr!r}"
        )
    logger.info("persistent check passed: memory exhaustion returned an OOM command result")
    require_command(run_command("printf recovered"), "command after memory OOM", "recovered")


def require_memory_cgroup_properties(
    memory_max: str, memory_oom_group: str, memory_swap_max: str
) -> None:
    """Refuse a memory bomb without a no-swap, command-only cgroup memory contract."""
    if memory_max != "268435456" or memory_oom_group != "0" or memory_swap_max != "0":
        raise RuntimeError(
            "persistent memory OOM check requires MemoryMax=268435456, MemorySwapMax=0, and "
            "memory.oom.group=0; "
            f"found memory.max={memory_max!r}, memory.swap.max={memory_swap_max!r}, "
            f"memory.oom.group={memory_oom_group!r}"
        )


def unified_cgroup_path(pid: int) -> str:
    """Return one host process's canonical unified-cgroup path."""
    cgroup_lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    unified_cgroup = next((line for line in cgroup_lines if line.startswith("0::")), None)
    if unified_cgroup is None:
        raise RuntimeError(f"process {pid} has no unified cgroup membership")
    cgroup_path = unified_cgroup.partition("::")[2]
    if not cgroup_path.startswith("/") or ".." in Path(cgroup_path).parts:
        raise RuntimeError(f"process {pid} reported a non-canonical cgroup path")
    return cgroup_path


def assert_supervisor_memory_cgroup(executor: PersistentJail) -> Tuple[str, Path]:  # noqa: UP006
    """Read the live cgroup of namespace PID 1 before starting a memory stress command."""
    pidfd = executor._pidfd  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    if pidfd is None:
        raise RuntimeError("cannot verify memory OOM limits before the supervisor starts")
    pidfd_lines = Path(f"/proc/self/fdinfo/{pidfd}").read_text().splitlines()
    pid_line = next((line for line in pidfd_lines if line.startswith("Pid:")), None)
    if pid_line is None:
        raise RuntimeError("persistent supervisor pidfd has no host PID")
    supervisor_pid = int(pid_line.partition(":")[2])
    if supervisor_pid <= 0:
        raise RuntimeError("persistent supervisor pidfd does not refer to a live process")
    cgroup_path = unified_cgroup_path(supervisor_pid)
    cgroup_directory = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/")
    memory_max = (cgroup_directory / "memory.max").read_text().strip()
    memory_oom_group = (cgroup_directory / "memory.oom.group").read_text().strip()
    memory_swap_max = (cgroup_directory / "memory.swap.max").read_text().strip()
    require_memory_cgroup_properties(memory_max, memory_oom_group, memory_swap_max)
    logger.info(
        "persistent check passed: supervisor cgroup has 256 MiB max, no swap, and group OOM disabled"
    )
    return cgroup_path, cgroup_directory


def find_processes_with_command_marker(marker: str) -> List[int]:  # noqa: UP006
    """Find live host processes whose command line contains a unique probe marker."""
    matching_pids = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command_line = (process / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker.encode() in command_line:
            matching_pids.append(int(process.name))
    return matching_pids


def process_namespace_pids(host_pid: int) -> List[int]:  # noqa: UP006 - Python 3.8 host script
    """Read the PID mappings reported for one host process."""
    status_lines = Path(f"/proc/{host_pid}/status").read_text().splitlines()
    nspid_line = next((line for line in status_lines if line.startswith("NSpid:")), None)
    if nspid_line is None:
        raise RuntimeError(f"process {host_pid} has no NSpid mapping")
    return [int(pid) for pid in nspid_line.partition(":")[2].split()]


def assert_memory_command_cgroup(executor: PersistentJail, supervisor_cgroup_path: str) -> None:
    """Prove a harmless actor command shares the cap and has maximum OOM victim priority."""
    marker = f"persistent-jail-command-cgroup-probe-{secrets.token_hex(8)}"
    ready_filename = "." + marker + ".ready"
    ready_path = executor.episode_dir / ready_filename
    probe_source = (
        "import os, time\n"
        "score = open('/proc/self/oom_score_adj', encoding='ascii').read().strip()\n"
        f"host_file = open('/work/{ready_filename}', 'w', encoding='ascii')\n"
        "host_file.write(f'{os.getpid()} {score}')\n"
        "host_file.close()\n"
        f"print({marker!r} + ' ' + str(os.getpid()) + ' ' + score, flush=True)\n"
        "time.sleep(4)\n"
    )
    probe_command = shlex.quote(jail_python()) + " -I -c " + shlex.quote(probe_source)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result_future: Future[CommandResult] = pool.submit(
            executor.run, probe_command, timeout="10s", login_shell=False
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not result_future.done():
            if ready_path.is_file():
                break
            time.sleep(0.02)
        if not ready_path.is_file():
            result = result_future.result(timeout=10)
            raise RuntimeError(
                "persistent command cgroup probe did not stay visible while running: "
                f"exit={result.exit_code} stdout={result.stdout!r} stderr={result.stderr!r}"
            )
        namespace_pid, reported_oom_score_adj = ready_path.read_text().split()
        command_pids = find_processes_with_command_marker(marker)
        command_properties: List[Tuple[int, str, str, str, List[int]]] = [  # noqa: UP006 - Python 3.8
            (
                command_pid,
                os.readlink(f"/proc/{command_pid}/exe"),  # noqa: PTH115 - Python 3.8 host script
                unified_cgroup_path(command_pid),
                Path(f"/proc/{command_pid}/oom_score_adj").read_text().strip(),
                process_namespace_pids(command_pid),
            )
            for command_pid in command_pids
        ]
        result = result_future.result(timeout=10)
    ready_path.unlink()
    if (
        reported_oom_score_adj != "1000"
        or not any(
            process[4][-1] == int(namespace_pid)
            and process[2] == supervisor_cgroup_path
            and process[3] == "1000"
            for process in command_properties
        )
        or any(
            process[2] != supervisor_cgroup_path or process[3] != "1000"
            for process in command_properties
        )
    ):
        raise RuntimeError(
            "persistent command is not protected for command-only OOM handling: "
            f"command_processes={command_properties!r} supervisor_cgroup={supervisor_cgroup_path!r} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    require_command(
        result,
        "command shares the capped cgroup with oom_score_adj=1000",
        marker + " " + namespace_pid + " 1000",
    )


def read_cgroup_oom_kill_count(cgroup_directory: Path) -> int:
    """Read the kernel's OOM-kill counter for the verified episode cgroup."""
    events = (cgroup_directory / "memory.events").read_text().splitlines()
    oom_kill_line = next((line for line in events if line.startswith("oom_kill ")), None)
    if oom_kill_line is None:
        raise RuntimeError("verified episode cgroup has no memory.events oom_kill counter")
    return int(oom_kill_line.split()[1])


def namespace_members(namespace: str) -> List[int]:  # noqa: UP006 - Python 3.8 host scripts
    """Find surviving host processes in the closed episode's PID namespace."""
    members = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            identity = os.readlink(process / "ns/pid")  # noqa: PTH115 - Path.readlink requires Python 3.9
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(identity) == namespace:
            members.append(int(process.name))
    return members


def host_descendants(executor: PersistentJail) -> Dict[int, str]:  # noqa: UP006 - Python 3.8 host scripts
    """Record host PID and start time, including nondumpable namespace init."""
    # Inspect the host launcher independently because nondumpability also hides namespace links from the host.
    process = executor._process  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    if process is None:
        raise RuntimeError("cannot inspect an unstarted executor")
    processes: Dict[int, Tuple[int, str]] = {}  # noqa: UP006 - Python 3.8 host scripts
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        processes[int(directory.name)] = (int(fields[1]), fields[19])
    descendants = {process.pid}
    while True:
        children = {pid for pid, (parent, _) in processes.items() if parent in descendants}
        if children <= descendants:
            break
        descendants.update(children)
    if process.pid not in processes:
        raise RuntimeError("persistent launcher exited before the cleanup check")
    return {pid: processes[pid][1] for pid in descendants}


def assert_closed(namespace: str, descendants: Dict[int, str]) -> None:  # noqa: UP006 - Python 3.8 host scripts
    """Reject any surviving process after the executor has closed."""
    if not namespace.startswith("pid:["):
        raise RuntimeError("missing PID namespace identity")
    members = namespace_members(namespace)
    if members:
        raise RuntimeError(f"persistent jail left namespace descendants: {members}")
    for pid, start_time in descendants.items():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[19] == start_time:
            raise RuntimeError(f"persistent jail left host process {pid} alive")
    logger.info("persistent check passed: closed namespace has no descendants")


def assert_nested_mount_read_only(fixture_root: Path) -> None:
    """Build a nested writable mount as an unprivileged user and check the jail blocks it."""
    source_directory = fixture_root / "readonly-source"
    (source_directory / "nested").mkdir(parents=True)
    episode_directory = fixture_root / "episode"
    episode_directory.mkdir()
    jail_launcher = Path(__file__).resolve().with_name("episode_jail.sh")
    outer_namespace_script = r"""
set -euo pipefail
source_directory=$1
episode_directory=$2
jail=$3
mount --make-rprivate /
mount -t tmpfs tmpfs "$source_directory/nested"
set +e
output=$("$jail" --episode-dir "$episode_directory" --backend unshare \
  --ro-bind "$source_directory" -- /bin/sh -c 'printf write > "$1"' sh \
  "$source_directory/nested/probe" 2>&1)
jail_status=$?
set -e
if [[ $jail_status -eq 0 ]]; then
  printf 'nested mount remained writable: %s\\n' "$output" >&2
  exit 17
fi
if [[ "$output" != *'Read-only file system'* ]]; then
  printf 'jail failed for an unrelated reason (status=%s): %s\\n' "$jail_status" "$output" >&2
  exit 18
fi
if [[ -e "$source_directory/nested/probe" ]]; then
  printf 'write reached the nested mount despite the jail error\\n' >&2
  exit 19
fi
"""
    result = subprocess.run(  # noqa: S603 - trusted unshare, jail launcher and synthetic mount fixture
        [
            "/usr/bin/unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--fork",
            "/bin/bash",
            "-c",
            outer_namespace_script,
            "bash",
            str(source_directory),
            str(episode_directory),
            str(jail_launcher),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("nested read-only bind probe failed: " + result.stdout + result.stderr)
    logger.info(
        "persistent check passed: every nested mount under a read-only bind stays read-only"
    )


def assert_concurrent_jails_isolated(
    first_episode_dir: Path,
    second_episode_dir: Path,
    *,
    backend: str,
) -> None:
    """Prove two live persistent jails cannot reach each other's filesystem or IPC resources."""
    first_episode_dir.mkdir(parents=True)
    second_episode_dir.mkdir(parents=True)
    first_ready = first_episode_dir / "pair-ready.json"
    first_stop = first_episode_dir / "pair-stop"
    signal_done = first_episode_dir / "signal-probe-done"
    token = secrets.token_hex(8)
    sysv_key = secrets.randbelow(2**31 - 1) + 1
    posix_name = "/persistent-jail-" + token
    resource_limits = JailResourceLimits(mode="advisory")
    first_executor = PersistentJail(
        first_episode_dir,
        backend=backend,
        resource_limits=resource_limits,
        cpus=1,
        mem_max="1G",
        episode_seconds=60,
    )
    second_executor = PersistentJail(
        second_episode_dir,
        backend=backend,
        resource_limits=resource_limits,
        cpus=1,
        mem_max="1G",
        episode_seconds=60,
    )
    first_command = r"""
import ctypes
import json
import os
import socket
import subprocess
import sys
import time

sysv_key = int(sys.argv[1])
posix_name = sys.argv[2]
token = sys.argv[3]
os.makedirs('/dev/shm', exist_ok=True)
filesystem_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
filesystem_socket.bind('/tmp/' + token + '.sock')
filesystem_socket.listen(1)
abstract_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
abstract_socket.bind('\x00persistent-jail-' + token)
abstract_socket.listen(1)

def prove_listener_works(server, address):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(address)
    connection, _ = server.accept()
    connection.close()
    client.close()

prove_listener_works(filesystem_socket, '/tmp/' + token + '.sock')
prove_listener_works(abstract_socket, ('\x00persistent-jail-' + token).encode())
with open('/tmp/' + token + '.marker', 'w') as marker:
    marker.write('present')
libc = ctypes.CDLL(None, use_errno=True)
sysv_segment = libc.shmget(sysv_key, 4096, 0o1000 | 0o2000 | 0o600)
if sysv_segment < 0:
    raise OSError(ctypes.get_errno(), 'could not create System V shared memory')
libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
libc.shm_open.restype = ctypes.c_int
posix_descriptor = libc.shm_open(posix_name.encode(), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
if posix_descriptor < 0:
    raise OSError(ctypes.get_errno(), 'could not create POSIX shared memory')
os.close(posix_descriptor)
victims = [subprocess.Popen(['/bin/sleep', '30']) for _ in range(3)]
resources = {
    'abstract_socket_listening': abstract_socket.getsockname()
    == ('\x00persistent-jail-' + token).encode(),
    'filesystem_socket_listening': os.path.exists('/tmp/' + token + '.sock'),
    'posix_shm_created': True,
    'sysv_shm_created': True,
    'tmp_marker_written': os.path.exists('/tmp/' + token + '.marker'),
    'victim_pids': [victim.pid for victim in victims],
}
with open('/work/pair-ready.json', 'w') as ready:
    json.dump(resources, ready)
tick = 0
while not os.path.exists('/work/pair-stop'):
    if os.path.exists('/work/signal-probe'):
        open('/work/signal-ready', 'w').close()
        while not os.path.exists('/work/signal-probe-done') and not os.path.exists('/work/pair-stop'):
            time.sleep(0.01)
        alive_pids = []
        for victim in victims:
            reaped_pid, _ = os.waitpid(victim.pid, os.WNOHANG)
            if reaped_pid == 0:
                alive_pids.append(victim.pid)
        tick += 1
        with open('/work/victim-status.json.tmp', 'w') as status:
            json.dump({'alive_pids': alive_pids, 'signal_probe_seen': True, 'tick': tick}, status)
        os.replace('/work/victim-status.json.tmp', '/work/victim-status.json')
        break
    time.sleep(0.01)
for victim in victims:
    if victim.poll() is None:
        victim.terminate()
    victim.wait()
"""
    second_probe = r"""
import ctypes
import errno
import os
import socket
import sys

sysv_key = int(sys.argv[1])
posix_name = sys.argv[2]
token = sys.argv[3]
os.makedirs('/dev/shm', exist_ok=True)
checks = []
if os.path.exists('/tmp/' + token + '.marker'):
    raise RuntimeError('sibling /tmp marker is visible')
checks.append('tmp')

def require_error(operation, expected_errno, name):
    try:
        operation()
    except OSError as error:
        if error.errno != expected_errno:
            raise RuntimeError(name + ' returned errno ' + str(error.errno))
    else:
        raise RuntimeError(name + ' reached the sibling resource')

def connect_filesystem_socket():
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect('/tmp/' + token + '.sock')
    finally:
        connection.close()

require_error(connect_filesystem_socket, errno.ENOENT, 'filesystem socket')
checks.append('filesystem_socket')

def connect_abstract_socket():
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect('\x00persistent-jail-' + token)
    finally:
        connection.close()

require_error(connect_abstract_socket, errno.ECONNREFUSED, 'abstract socket')
checks.append('abstract_socket')
libc = ctypes.CDLL(None, use_errno=True)
if libc.shmget(sysv_key, 1, 0) >= 0:
    raise RuntimeError('System V shared memory is visible')
if ctypes.get_errno() != errno.ENOENT:
    raise RuntimeError('System V shared memory lookup returned errno ' + str(ctypes.get_errno()))
checks.append('sysv_shm')
libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
libc.shm_open.restype = ctypes.c_int
posix_descriptor = libc.shm_open(posix_name.encode(), os.O_RDWR, 0)
if posix_descriptor >= 0:
    os.close(posix_descriptor)
    raise RuntimeError('POSIX shared memory is visible')
if ctypes.get_errno() != errno.ENOENT:
    raise RuntimeError('POSIX shared memory lookup returned errno ' + str(ctypes.get_errno()))
checks.append('posix_shm')
print(' '.join(checks))
"""
    command = shlex.join(
        ["python3", "-I", "-u", "-c", first_command, str(sysv_key), posix_name, token]
    )
    probe_command = shlex.join(
        ["python3", "-I", "-c", second_probe, str(sysv_key), posix_name, token]
    )

    with first_executor, second_executor, ThreadPoolExecutor(max_workers=1) as pool:
        _assert_separate_namespaces(first_executor, second_executor)
        active_command = pool.submit(first_executor.run, command, timeout="30s")
        try:
            victim_pids = _wait_for_live_resources(first_ready, active_command)
            probe_result = second_executor.run(probe_command)
            expected = "tmp filesystem_socket abstract_socket sysv_shm posix_shm"
            require_command(
                probe_result, "sibling filesystem and IPC resources stay hidden", expected
            )
            _assert_sibling_signal_blocked(second_executor, victim_pids, first_episode_dir)
        finally:
            signal_done.touch()
            first_stop.touch()
            active_command.result(timeout=10)
    logger.info("persistent check passed: concurrent persistent jails are mutually isolated")


def _assert_separate_namespaces(
    first_executor: PersistentJail, second_executor: PersistentJail
) -> None:
    namespaces = {}
    for label, executor in (("first", first_executor), ("second", second_executor)):
        for namespace_kind in ("pid", "ipc", "net"):
            result = executor.run("readlink /proc/self/ns/" + namespace_kind)
            if not result.ok:
                raise RuntimeError(
                    label + " jail could not identify its " + namespace_kind + " namespace"
                )
            namespace = result.stdout.strip()
            if not namespace.startswith(namespace_kind + ":["):
                raise RuntimeError("unexpected namespace identity: " + namespace)
            namespaces[label, namespace_kind] = namespace
    for namespace_kind in ("pid", "ipc", "net"):
        if namespaces["first", namespace_kind] == namespaces["second", namespace_kind]:
            raise RuntimeError("persistent jails share their " + namespace_kind + " namespace")
    logger.info(
        "persistent check passed: concurrent jails use separate pid, ipc and net namespaces"
    )


def _wait_for_live_resources(first_ready: Path, active_command: Future[CommandResult]) -> List[int]:  # noqa: UP006 - Python 3.8 host scripts
    deadline = time.monotonic() + 10
    while not first_ready.is_file():
        if active_command.done():
            active_command.result()
            raise RuntimeError("first persistent jail exited before publishing live resources")
        if time.monotonic() >= deadline:
            raise TimeoutError("first persistent jail did not publish live resources")
        time.sleep(0.02)
    resources = json.loads(first_ready.read_text())
    required_resources = (
        "abstract_socket_listening",
        "filesystem_socket_listening",
        "posix_shm_created",
        "sysv_shm_created",
        "tmp_marker_written",
    )
    if not all(resources.get(name) is True for name in required_resources):
        raise RuntimeError("first jail did not expose every live resource: " + str(resources))
    logger.info("persistent check passed: first jail is actively serving all isolation probes")
    return [int(pid) for pid in resources["victim_pids"]]


def _assert_sibling_signal_blocked(
    second_executor: PersistentJail,
    victim_pids: List[int],  # noqa: UP006 - Python 3.8 host scripts
    first_episode_dir: Path,
) -> None:
    signal_probe = first_episode_dir / "signal-probe"
    signal_ready = first_episode_dir / "signal-ready"
    signal_done = first_episode_dir / "signal-probe-done"
    victim_status = first_episode_dir / "victim-status.json"
    signal_probe.touch()
    signal_deadline = time.monotonic() + 5
    while not signal_ready.is_file():
        if time.monotonic() >= signal_deadline:
            raise TimeoutError("first jail did not prepare the signal isolation probe")
        time.sleep(0.02)
    target_pids = " ".join(shlex.quote(str(pid)) for pid in victim_pids)
    signal_command = (
        "for target_pid in "
        + target_pids
        + '; do if [ "$target_pid" != "$$" ] && ! kill -0 "$target_pid" '
        + ' >/dev/null 2>&1; then if kill -TERM "$target_pid" >/dev/null 2>&1; then '
        + "printf delivered; exit 1; else printf 'blocked:%s' \"$target_pid\"; exit 0; fi; fi; "
        + "done; printf no-unused-pid; exit 1"
    )
    signal_result = second_executor.run(signal_command)
    if not signal_result.ok or not signal_result.stdout.startswith("blocked:"):
        raise RuntimeError(
            "sibling PID signal probe failed: "
            f"exit={signal_result.exit_code} stdout={signal_result.stdout!r} "
            f"stderr={signal_result.stderr!r}"
        )
    selected_pid = int(signal_result.stdout.split(":", 1)[1])
    signal_done.touch()
    status_deadline = time.monotonic() + 5
    while True:
        if victim_status.is_file():
            status = json.loads(victim_status.read_text())
            if status.get("signal_probe_seen"):
                if selected_pid not in status.get("alive_pids", []):
                    raise RuntimeError("second jail signalled the first jail's live process")
                logger.info("persistent check passed: sibling signal did not reach victim PID")
                return
        if time.monotonic() >= status_deadline:
            raise TimeoutError("first jail did not confirm the signal isolation probe")
        time.sleep(0.02)


def assert_memory_oom_isolated(episode_dir: Path, *, backend: str) -> None:
    """Require a 256 MiB cgroup to kill a command while keeping its episode alive."""
    episode_dir.mkdir()
    executor = PersistentJail(
        episode_dir,
        backend=backend,
        resource_limits=JailResourceLimits(mode="enforced"),
        cpus=1,
        mem_max="256M",
        mem_swap_max="0",
        episode_seconds=60,
    )
    logger.warning(
        "Persistent memory OOM check requires enforced MemoryMax=256M and MemorySwapMax=0"
    )
    with executor:
        namespace_result = executor.run("readlink /proc/self/ns/pid")
        if not namespace_result.ok:
            raise RuntimeError("memory OOM fixture could not start: " + namespace_result.stderr)
        namespace = namespace_result.stdout.strip()
        supervisor_cgroup_path, cgroup_directory = assert_supervisor_memory_cgroup(executor)
        assert_memory_command_cgroup(executor, supervisor_cgroup_path)
        initial_oom_kill_count = read_cgroup_oom_kill_count(cgroup_directory)
        require_memory_oom_recovery(
            lambda command: executor.run(command, timeout="20s", login_shell=False)
        )
        final_oom_kill_count = read_cgroup_oom_kill_count(cgroup_directory)
        if final_oom_kill_count <= initial_oom_kill_count:
            raise RuntimeError(
                "persistent memory bomb returned an OOM-shaped result without a kernel cgroup "
                f"oom_kill event: before={initial_oom_kill_count} after={final_oom_kill_count}"
            )
        logger.info("persistent check passed: kernel memory.events recorded a cgroup OOM kill")
        descendants = host_descendants(executor)
    assert_closed(namespace, descendants)
    logger.info("persistent check passed: memory OOM left no process after episode close")


def main() -> None:  # noqa: PLR0915 - sequential containment attacks deliberately remain visible together
    """Run the persistent containment gate against synthetic local fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bwrap", "unshare"), required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("assertion_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.warning(
        "Persistent isolation gate uses advisory fixture limits; cgroup enforcement is NOT certified"
    )
    resource_limits = JailResourceLimits(mode="advisory")
    episode_dir = args.workdir / "episode"
    sibling_dir = args.workdir / "sibling"
    sibling_dir.mkdir()
    (sibling_dir / "private-marker").touch()
    host_marker = args.workdir / "host-marker"
    host_marker.touch()
    assertion_args = args.assertion_args
    if assertion_args and assertion_args[0] == "--":
        assertion_args = assertion_args[1:]
    if not assertion_args:
        parser.error("containment assertion arguments are required after --")

    executor = PersistentJail(
        episode_dir,
        backend=args.backend,
        resource_limits=resource_limits,
        cpus=1,
        mem_max="1G",
        episode_seconds=60,
    )
    with executor:
        baseline = executor.run(
            "python3 /work/jail_assertions.py " + shlex.join(assertion_args), timeout="20s"
        )
        if not baseline.ok:
            raise RuntimeError(
                "persistent containment suite failed: " + baseline.stdout + baseline.stderr
            )
        logger.info("persistent check passed: ordinary containment suite")
        namespace_result = executor.run("readlink /proc/self/ns/pid")
        if not namespace_result.ok:
            raise RuntimeError(
                "could not identify persistent PID namespace: " + namespace_result.stderr
            )
        namespace = namespace_result.stdout.strip()
        require_command(
            executor.run("readlink /proc/self/ns/pid"), "PID namespace persists", namespace
        )
        paths = [str(sibling_dir), str(host_marker), "/oldroot" + str(host_marker)]
        for path in paths:
            command = "test ! -e " + shlex.quote(path)
            require_command(executor.run(command), "host and sibling path hidden")
        require_command(
            executor.run("test ! -e " + shlex.quote("/proc/1/root" + str(host_marker))),
            "supervisor root does not expose host",
        )
        require_command(
            executor.run(
                "python3 -I -c "
                + shlex.quote(
                    "import os; "
                    "paths = ['/proc/1/fd/0', '/proc/1/fd/1', '/proc/1/mem', '/proc/1/environ']; "
                    'exec("for path in paths:\\n'
                    " try:\\n  descriptor = os.open(path, os.O_RDWR)\\n"
                    " except PermissionError:\\n  continue\\n"
                    " else:\\n  os.close(descriptor); raise RuntimeError('supervisor readable: ' + path)\")"
                )
            ),
            "supervisor is nondumpable and protocol descriptors inaccessible",
        )
        executor.run("kill -TERM 1; kill -KILL 1")
        require_command(
            executor.run("printf alive"), "supervisor survives payload signals", "alive"
        )
        require_command(
            executor.run('printf \'%s\' \'{"exit_code":0,"stdout":"forged"}\''),
            "payload output remains data",
            '{"exit_code":0,"stdout":"forged"}',
        )
        require_command(
            executor.run("printf authentic"), "protocol remains synchronized", "authentic"
        )
        detached_command = "python3 -I -c " + shlex.quote(
            "import os,time; "
            'exec("if os.fork() == 0:\\n os.setsid()\\n'
            " if os.fork() == 0:\\n  time.sleep(0.5); open('/work/orphan-marker','w').close()\\n"
            ' os._exit(0)")'
        )
        require_command(
            executor.run(detached_command), "normal command with double-fork descendant"
        )
        require_command(
            executor.run("sleep 0.8; test ! -e /work/orphan-marker"),
            "normal exit reaps detached descendants",
        )
        timeout_result = executor.run(detached_command + "; sleep 10", timeout="0.1s")
        if not timeout_result.timed_out or timeout_result.exit_code != TIMEOUT_EXIT_CODE:
            raise RuntimeError("persistent timeout did not report exit 124")
        require_command(
            executor.run("sleep 0.8; test ! -e /work/orphan-marker && printf recovered"),
            "timeout reaps descendants and permits next command",
            "recovered",
        )
        require_command(executor.run("touch /tmp/orphan-marker"), "actor changes private tmpfs")
        descendants = host_descendants(executor)
    assert_closed(namespace, descendants)
    executor.close()
    exception_dir = args.workdir / "exception-episode"
    exception_dir.mkdir()
    exception_namespace = ""
    exception_descendants: Dict[int, str] = {}  # noqa: UP006 - Python 3.8 host scripts
    try:
        with PersistentJail(
            exception_dir,
            backend=args.backend,
            resource_limits=resource_limits,
            cpus=1,
            mem_max="1G",
            episode_seconds=30,
        ) as exception_executor:
            namespace_result = exception_executor.run("readlink /proc/self/ns/pid")
            if not namespace_result.ok:
                raise RuntimeError("exception fixture could not start: " + namespace_result.stderr)
            exception_namespace = namespace_result.stdout.strip()
            require_command(
                exception_executor.run(detached_command), "exception fixture double-fork descendant"
            )
            exception_descendants = host_descendants(exception_executor)
            raise LookupError("synthetic caller failure")  # noqa: TRY301 - deliberately exercise context failure
    except LookupError:
        pass
    assert_closed(exception_namespace, exception_descendants)
    assert_nested_mount_read_only(args.workdir / "nested-readonly-probe")
    assert_concurrent_jails_isolated(
        args.workdir / "pair-first-episode",
        args.workdir / "pair-second-episode",
        backend=args.backend,
    )
    assert_memory_oom_isolated(args.workdir / "memory-oom-episode", backend=args.backend)
    jail = Path(__file__).resolve().with_name("episode_jail.sh")
    grader = subprocess.run(  # noqa: S603 - trusted launcher and synthetic grader
        [
            str(jail),
            "--episode-dir",
            str(episode_dir),
            "--backend",
            args.backend,
            "--",
            "/bin/bash",
            "-c",
            "test ! -e /tmp/orphan-marker && printf grader-clean",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if grader.returncode != 0 or grader.stdout != "grader-clean":
        raise RuntimeError("host-launched grader failed after actor shutdown: " + grader.stderr)
    logger.info("persistent check passed: independent host-launched grader")


if __name__ == "__main__":
    main()
