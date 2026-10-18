"""Exercise persistent jail isolation and recursive read-only binds with synthetic fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reward_hacking.jailed_poc import CommandResult
from reward_hacking.persistent_jail import PersistentJailSupervisorError
from scripts.persistent_jail_assertions import (
    assert_concurrent_jails_isolated,
    assert_nested_mount_read_only,
    read_cgroup_oom_kill_count,
    require_memory_cgroup_properties,
    require_memory_oom_recovery,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_two_live_persistent_jails_cannot_reach_each_other(tmp_path: Path) -> None:
    assert_concurrent_jails_isolated(
        tmp_path / "first-episode",
        tmp_path / "second-episode",
        backend="unshare",
    )


def test_unshare_readonly_bind_remounts_nested_mounts(tmp_path: Path) -> None:
    assert_nested_mount_read_only(tmp_path)


def test_memory_oom_probe_requires_kill_result_and_recovery() -> None:
    command_results = iter(
        (
            CommandResult(
                command="memory-bomb",
                exit_code=137,
                stdout="",
                stderr=(
                    "resource-limits: command exited 137 (SIGKILL or cgroup OOM); "
                    "episode supervisor remains available"
                ),
                timed_out=False,
            ),
            CommandResult(
                command="recovery",
                exit_code=0,
                stdout="recovered",
                stderr="",
                timed_out=False,
            ),
        )
    )
    commands: list[str] = []

    def run_command(command: str) -> CommandResult:
        commands.append(command)
        return next(command_results)

    require_memory_oom_recovery(run_command)

    assert len(commands) == 2
    assert commands[0].startswith("/")
    assert " -I -c " in commands[0]
    assert "block_size" in commands[0]
    assert commands[1] == "printf recovered"


def test_memory_oom_probe_does_not_hide_supervisor_death() -> None:
    def supervisor_died(_: str) -> CommandResult:
        raise PersistentJailSupervisorError("persistent jail supervisor exited without a response")

    with pytest.raises(PersistentJailSupervisorError, match="supervisor exited without a response"):
        require_memory_oom_recovery(supervisor_died)


@pytest.mark.parametrize(
    "properties",
    [
        ("max", "0", "0"),
        ("268435456", "1", "0"),
        ("268435456", "0", "max"),
    ],
)
def test_memory_oom_probe_rejects_missing_cgroup_memory_guards(
    properties: tuple[str, str, str],
) -> None:
    require_memory_cgroup_properties("268435456", "0", "0")

    with pytest.raises(RuntimeError, match=r"requires MemoryMax=268435456, MemorySwapMax=0"):
        require_memory_cgroup_properties(*properties)


def test_cgroup_oom_kill_counter_reads_kernel_event(tmp_path: Path) -> None:
    (tmp_path / "memory.events").write_text("low 0\noom 1\noom_kill 1\n")

    assert read_cgroup_oom_kill_count(tmp_path) == 1


def test_cgroup_oom_kill_counter_requires_kernel_event_field(tmp_path: Path) -> None:
    (tmp_path / "memory.events").write_text("low 0\noom 0\n")

    with pytest.raises(RuntimeError, match=r"no memory.events oom_kill counter"):
        read_cgroup_oom_kill_count(tmp_path)
