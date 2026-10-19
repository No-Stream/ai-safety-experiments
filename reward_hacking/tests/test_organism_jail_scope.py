"""The run's budget reaches jail launches inside fresh episode worker threads."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from reward_hacking import jailed_poc
from reward_hacking.organism_rl.jail_source import BudgetedEpisodeSource


def test_jail_launch_passes_slice_and_peak_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def execute(argv: list[str], **kwargs: Any) -> jailed_poc.BoundedProcessResult:
        calls.append(argv)
        return jailed_poc.BoundedProcessResult(0, "", "", timed_out=False, output_truncated=False)

    monkeypatch.setattr(jailed_poc, "run_bounded_process", execute)
    peak_log = tmp_path / "jail-memory.jsonl"
    limits = jailed_poc.JailResourceLimits(
        slice_name="orgjailstest.slice", memory_peak_log=peak_log
    )
    with jailed_poc.jail_resource_limits_scope(limits):
        jailed_poc.run_in_jail(tmp_path, "true")
    assert calls[0][calls[0].index("--slice") + 1] == limits.slice_name
    assert calls[0][calls[0].index("--memory-peak-log") + 1] == str(peak_log)


def test_budgeted_source_scopes_each_worker_and_restores_context() -> None:
    limits = jailed_poc.JailResourceLimits(slice_name="orgjailstest.slice")

    class Budget:
        def limits(self) -> jailed_poc.JailResourceLimits:
            return limits

    def source(task_id: str, **kwargs: Any) -> Any:
        assert jailed_poc.current_jail_resource_limits() is limits
        raise LookupError("synthetic episode failure")

    scoped = BudgetedEpisodeSource(source, Budget())

    def worker() -> None:
        with pytest.raises(LookupError, match="synthetic episode failure"):
            scoped("synthetic", seed=0, step=0, replica=0)
        assert jailed_poc.current_jail_resource_limits().slice_name is None

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: worker(), range(2)))
