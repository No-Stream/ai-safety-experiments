from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

import pytest

from reward_hacking.jailed_poc import (
    JailResourceLimits,
    current_jail_resource_limits,
    jail_resource_limits_scope,
)
from reward_hacking.organism_rl import grading_pool
from reward_hacking.organism_rl.grading_pool import (
    GradingOutcome,
    available_worker_count,
    run_bounded,
)


@pytest.fixture(autouse=True)
def sufficient_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grading_pool, "_available_memory_bytes", lambda: 128 << 30)
    monkeypatch.delenv(grading_pool.GRADING_WORKERS_ENV, raising=False)


def _install_cgroup_v2_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    membership_path: str = "/tenant/job",
    quotas: dict[str, str],
) -> None:
    cgroup_root = tmp_path / "sys" / "fs" / "cgroup"
    cgroup_root.mkdir(parents=True)
    for directory, cpu_max in quotas.items():
        cgroup_directory = cgroup_root / directory
        cgroup_directory.mkdir(parents=True, exist_ok=True)
        (cgroup_directory / "cpu.max").write_text(cpu_max, encoding="utf-8")
    cgroup_root.joinpath(*(part for part in membership_path.split("/") if part)).mkdir(
        parents=True, exist_ok=True
    )
    membership_file = tmp_path / "proc" / "self" / "cgroup"
    membership_file.parent.mkdir(parents=True)
    membership_file.write_text(f"0::{membership_path}\n", encoding="utf-8")
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool.CGROUP_V2_MOUNT", cgroup_root)
    monkeypatch.setattr(
        "reward_hacking.organism_rl.grading_pool.PROCESS_CGROUP_FILE", membership_file
    )


def _install_cpu_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, core_siblings: list[list[int]]
) -> None:
    root = tmp_path / "sys" / "devices" / "system" / "cpu"
    for siblings in core_siblings:
        for cpu in siblings:
            topology = root / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            (topology / "core_cpus_list").write_text(
                ",".join(str(sibling) for sibling in siblings) + "\n", encoding="ascii"
            )
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool.CPU_TOPOLOGY_ROOT", root)
    monkeypatch.setattr(
        "os.sched_getaffinity", lambda _pid: {cpu for group in core_siblings for cpu in group}
    )


def test_physical_cores_count_hyperthread_siblings_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_cpu_topology(tmp_path, monkeypatch, [[0, 1], [2, 3], [4, 5]])

    assert grading_pool._physical_core_count() == 3  # pyright: ignore[reportPrivateUsage]


def test_default_worker_count_stays_below_physical_cores_on_an_smt_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timed gradings sharing a core can time out correct candidates, so SMT threads do not count."""
    _install_cpu_topology(tmp_path, monkeypatch, [[2 * core, 2 * core + 1] for core in range(16)])
    monkeypatch.setattr(
        "reward_hacking.organism_rl.grading_pool._cgroup_v2_cpu_count", lambda: None
    )

    assert grading_pool._available_cpu_count() == 32  # pyright: ignore[reportPrivateUsage]
    assert available_worker_count() == 12


def test_default_worker_count_leaves_headroom_on_a_small_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._available_cpu_count", lambda: 6)
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._physical_core_count", lambda: 6)

    assert available_worker_count() == 2


def test_default_worker_count_follows_a_quota_tighter_than_the_core_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._available_cpu_count", lambda: 8)
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._physical_core_count", lambda: 16)

    assert available_worker_count() == 4


def test_worker_selection_logs_available_cpus_and_headroom(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._available_cpu_count", lambda: 8)
    with caplog.at_level(logging.INFO):
        assert available_worker_count(4) == 4
    assert "available_cpus=8 workers=4 headroom_cpus=4" in caplog.text


def test_worker_count_respects_cpuset_even_when_host_cpu_count_is_larger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(3)))
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    monkeypatch.setattr(
        "reward_hacking.organism_rl.grading_pool._cgroup_v2_cpu_count", lambda: None
    )

    assert available_worker_count() == 1


def test_worker_count_falls_back_to_affinity_without_unified_cgroup_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(6)))
    membership_file = tmp_path / "proc" / "self" / "cgroup"
    membership_file.parent.mkdir(parents=True)
    membership_file.write_text("12:cpuset:/tenant/job\n", encoding="utf-8")
    monkeypatch.setattr(
        "reward_hacking.organism_rl.grading_pool.PROCESS_CGROUP_FILE", membership_file
    )

    assert grading_pool._available_cpu_count() == 6  # pyright: ignore[reportPrivateUsage]


def test_worker_count_respects_cgroup_v2_quota_on_current_and_ancestor_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(32)))
    _install_cgroup_v2_fixture(
        tmp_path,
        monkeypatch,
        quotas={
            ".": "max 100000",
            "tenant": "850000 100000",
            "tenant/job": "max 100000",
        },
    )

    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._physical_core_count", lambda: 16)

    assert grading_pool._available_cpu_count() == 8  # pyright: ignore[reportPrivateUsage]
    assert available_worker_count() == 4


@pytest.mark.parametrize(
    ("cpu_max", "expected_cpus"),
    [("150000 100000", 1), ("50000 100000", 1), ("250000 100000", 2)],
)
def test_cgroup_v2_fractional_cpu_quota_is_rounded_down_with_one_cpu_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_max: str,
    expected_cpus: int,
) -> None:
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(32)))
    _install_cgroup_v2_fixture(tmp_path, monkeypatch, quotas={".": cpu_max})

    assert grading_pool._available_cpu_count() == expected_cpus  # pyright: ignore[reportPrivateUsage]


def test_worker_count_raises_for_malformed_readable_cgroup_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(32)))
    _install_cgroup_v2_fixture(
        tmp_path,
        monkeypatch,
        quotas={".": "max 100000", "tenant": "not-a-quota"},
    )

    with pytest.raises(ValueError, match=r"cpu\.max"):
        available_worker_count()


def test_explicit_worker_override_can_use_effective_cgroup_cpu_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(32)))
    _install_cgroup_v2_fixture(
        tmp_path,
        monkeypatch,
        quotas={".": "max 100000", "tenant": "600000 100000"},
    )

    assert available_worker_count(6) == 6
    with pytest.raises(ValueError, match="worker"):
        available_worker_count(7)


def test_cli_override_takes_precedence_over_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._available_cpu_count", lambda: 8)
    environment: Mapping[str, str] = {"ORGANISM_RL_GRADING_WORKERS": "3"}

    assert available_worker_count(2, environ=environment) == 2
    assert available_worker_count(environ=environment) == 3


@pytest.mark.parametrize("value", ["", "zero", "0", "-1", "9"])
def test_worker_override_must_be_positive_and_within_available_cpu_count(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setattr("reward_hacking.organism_rl.grading_pool._available_cpu_count", lambda: 8)

    with pytest.raises(ValueError, match="worker"):
        available_worker_count(environ={"ORGANISM_RL_GRADING_WORKERS": value})


def test_run_bounded_caps_concurrency_and_preserves_input_order() -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def work(item: int) -> int:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return item * 2

    outcomes = run_bounded(range(8), work, max_workers=3)

    assert [outcome.item for outcome in outcomes] == list(range(8))
    assert [outcome.result for outcome in outcomes] == [item * 2 for item in range(8)]
    assert all(outcome.error is None and outcome.elapsed_seconds >= 0 for outcome in outcomes)
    assert maximum_active == 3


def test_run_bounded_propagates_worker_error_and_stops_submission() -> None:
    completed: list[GradingOutcome[int, str]] = []
    submitted: list[int] = []
    first_started = threading.Event()

    def work(item: int) -> str:
        submitted.append(item)
        if item == 0:
            first_started.set()
            time.sleep(0.05)
        if item == 1:
            assert first_started.wait(timeout=1)
            raise RuntimeError("grader failed")
        return f"graded-{item}"

    with pytest.raises(RuntimeError, match="grader failed"):
        run_bounded(range(4), work, max_workers=2, on_complete=completed.append)
    assert sorted(submitted) == [0, 1]


def test_worker_count_respects_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grading_pool, "_available_cpu_count", lambda: 32)
    monkeypatch.setattr(grading_pool, "_available_memory_bytes", lambda: 6 << 30)
    assert available_worker_count(environ={}) == 3
    with pytest.raises(ValueError, match="memory budget"):
        available_worker_count(4)


def test_remaining_memory_uses_tightest_ancestor_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    _install_cgroup_v2_fixture(
        tmp_path, monkeypatch, quotas={".": "max 100000", "tenant": "max 100000"}
    )
    for directory, maximum, current in [(".", 8 << 30, 1 << 30), ("tenant", 4 << 30, 1 << 30)]:
        group = grading_pool.CGROUP_V2_MOUNT / directory
        (group / "memory.max").write_text(str(maximum), encoding="ascii")
        (group / "memory.current").write_text(str(current), encoding="ascii")
    assert grading_pool._available_memory_bytes() == 3 << 30  # pyright: ignore[reportPrivateUsage]


def test_term_interrupt_drains_workers_and_restores_handler() -> None:
    finished = threading.Event()
    previous_handler = signal.getsignal(signal.SIGTERM)

    def work(item: int) -> int:
        if item == 1:
            time.sleep(0.05)
            finished.set()
        return item

    def interrupt(_outcome: GradingOutcome[int, int]) -> None:
        signal.raise_signal(signal.SIGTERM)

    with pytest.raises(InterruptedError, match="signal"):
        run_bounded(range(2), work, max_workers=2, on_complete=interrupt)
    assert finished.is_set()
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_run_bounded_copies_the_callers_context_into_each_worker() -> None:
    advisory_limits = JailResourceLimits(mode="advisory", systemd_user_state="offline")
    with jail_resource_limits_scope(advisory_limits):
        outcomes = run_bounded(
            range(4), lambda _item: current_jail_resource_limits().mode, max_workers=2
        )

    assert [outcome.result for outcome in outcomes] == ["advisory"] * 4


def test_workers_have_distinct_single_cpu_affinities() -> None:
    barrier = threading.Barrier(2)
    parent_affinity = os.sched_getaffinity(0)

    def affinity(_item: int) -> set[int]:
        barrier.wait(timeout=5)
        return os.sched_getaffinity(0)

    outcomes = run_bounded(range(2), affinity, max_workers=2)
    worker_affinities = [outcome.result for outcome in outcomes]
    assert all(cpu_set is not None and len(cpu_set) == 1 for cpu_set in worker_affinities)
    assert worker_affinities[0] != worker_affinities[1]
    assert os.sched_getaffinity(0) == parent_affinity
