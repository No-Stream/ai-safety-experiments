"""Require live aggregate memory enforcement across two otherwise larger episode units."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import needs_jail

from reward_hacking.jailed_poc import jail_python
from reward_hacking.organism_rl.jail_budget import JailBudgetSlice
from reward_hacking.persistent_jail import PersistentJail
from scripts.persistent_jail_assertions import read_cgroup_oom_kill_count, unified_cgroup_path


def _systemctl(*arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed host administration commands
        ["/usr/bin/systemctl", "--user", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _supervisor_cgroup(executor: PersistentJail) -> Path:
    pidfd = executor._pidfd  # pyright: ignore[reportPrivateUsage]
    assert pidfd is not None
    fields = dict(
        line.split(":", 1) for line in Path(f"/proc/self/fdinfo/{pidfd}").read_text().splitlines()
    )
    host_pid = int(fields["Pid"].strip())
    assert host_pid > 0
    return Path("/sys/fs/cgroup") / unified_cgroup_path(host_pid).lstrip("/")


def _require_user_manager() -> None:
    probe = subprocess.run(
        ["/usr/bin/systemctl", "--user", "show-environment"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode:
        reason = (
            "aggregate memory test requires an enforced systemd user slice: " + probe.stderr.strip()
        )
        if os.environ.get("SEED_V4_REQUIRE_JAIL") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def _require_peak_records(log_path: Path, children: list[Path]) -> None:
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 2
    assert {record["unit_name"].removesuffix(".service") for record in records} == {
        child.name.removesuffix(".service") for child in children
    }
    for record in records:
        assert record["unavailable_reason"] is None
        assert 0 < record["memory_peak_bytes"] < 384 * 1024**2


def _cap_test_slice(slice_name: str) -> Path:
    _systemctl("set-property", "--runtime", slice_name, "MemoryMax=160M", "MemorySwapMax=0")
    parent = Path("/sys/fs/cgroup") / _systemctl(
        "show", slice_name, "--property=ControlGroup", "--value"
    ).lstrip("/")
    assert (parent / "memory.max").read_text().strip() == str(160 * 1024**2)
    assert (parent / "memory.swap.max").read_text().strip() == "0"
    assert (parent / "memory.oom.group").read_text().strip() == "0"
    assert not Path("/sys/fs/cgroup", unified_cgroup_path(os.getpid()).lstrip("/")).is_relative_to(
        parent
    )
    return parent


@pytest.mark.jail_real
@needs_jail
def test_aggregate_slice_kills_only_allocating_command_and_both_episodes_recover(
    tmp_path: Path,
) -> None:
    _require_user_manager()
    first_directory = tmp_path / "allocating-episode"
    second_directory = tmp_path / "surviving-episode"
    first_directory.mkdir()
    second_directory.mkdir()
    with JailBudgetSlice(tmp_path / "run") as budget:
        limits = budget.limits()
        assert limits.mode == "enforced"
        assert limits.slice_name is not None
        parent = _cap_test_slice(limits.slice_name)

        with (
            PersistentJail(
                first_directory,
                resource_limits=limits,
                cpus=1,
                mem_max="384M",
                mem_swap_max="0",
                episode_seconds=90,
            ) as allocating_episode,
            PersistentJail(
                second_directory,
                resource_limits=limits,
                cpus=1,
                mem_max="384M",
                mem_swap_max="0",
                episode_seconds=90,
            ) as surviving_episode,
        ):
            assert allocating_episode.run("true").ok
            assert surviving_episode.run("true").ok
            children = [
                _supervisor_cgroup(episode) for episode in (allocating_episode, surviving_episode)
            ]
            assert children[0] != children[1]
            for child in children:
                assert child.parent == parent
                assert (child / "memory.max").read_text().strip() == str(384 * 1024**2)
                assert (child / "memory.swap.max").read_text().strip() == "0"
                assert (child / "memory.oom.group").read_text().strip() == "0"
            initial_oom_kills = read_cgroup_oom_kill_count(parent)
            ready = second_directory / "ready"
            release = second_directory / "release"
            with ThreadPoolExecutor(max_workers=1) as pool:
                sibling_result = pool.submit(
                    surviving_episode.run,
                    "touch /work/ready; while [ ! -f /work/release ]; do sleep 0.1; done; printf sibling-survived",
                    timeout="30s",
                )
                try:
                    deadline = time.monotonic() + 10
                    while (
                        not ready.exists()
                        and not sibling_result.done()
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.02)
                    assert ready.exists(), "sibling episode did not start its concurrent command"
                    source = (
                        "blocks = []\n"
                        "for _ in range(16):\n"
                        " block = bytearray(16 * 1024 * 1024)\n"
                        " for offset in range(0, len(block), 4096): block[offset] = 1\n"
                        " blocks.append(block)\n"
                        "print('bounded-allocation-completed')\n"
                    )
                    result = allocating_episode.run(
                        "exec " + shlex.quote(jail_python()) + " -I -c " + shlex.quote(source),
                        timeout="20s",
                    )
                    assert result.exit_code == 137, result
                    assert not result.timed_out
                    assert "cgroup OOM" in result.stderr
                    assert read_cgroup_oom_kill_count(parent) > initial_oom_kills
                    assert not sibling_result.done(), (
                        "aggregate OOM interrupted the other episode's command"
                    )
                    assert _systemctl("is-active", limits.slice_name) == "active"
                    assert (
                        allocating_episode.run("printf allocator-recovered").stdout
                        == "allocator-recovered"
                    )
                finally:
                    release.touch()
                survivor = sibling_result.result(timeout=10)
                assert survivor.ok, survivor
                assert survivor.stdout == "sibling-survived"
            assert surviving_episode.run("printf sibling-recovered").stdout == "sibling-recovered"
        assert limits.memory_peak_log is not None
        _require_peak_records(Path(limits.memory_peak_log), children)
