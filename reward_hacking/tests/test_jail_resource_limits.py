"""Explicit advisory limits reach every jail call without changing oracle callbacks."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking import jailed_poc
from reward_hacking.harness import loop

if TYPE_CHECKING:
    from pathlib import Path


def test_advisory_runs_the_real_limiter_on_an_offline_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Only replace the inner jail; the real limiter must stop refusing the command."""
    systemctl = tmp_path / "systemctl"
    systemctl.write_text("#!/bin/sh\necho offline\nexit 1\n")
    systemctl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    jail = tmp_path / "jail"
    jail.write_text('#!/bin/sh\nshift 2\nshift\nexec "$@"\n')
    jail.chmod(0o755)
    monkeypatch.setattr(jailed_poc, "EPISODE_JAIL", jail)
    refused = jailed_poc.run_in_jail(tmp_path, "printf limited", cpus=1, mem_max="1G")
    assert refused.exit_code == 2
    assert "no usable systemd user instance" in refused.stderr

    limits = jailed_poc.resolve_jail_resource_limits(advisory=True)
    result = jailed_poc.run_in_jail(
        tmp_path, "printf limited", cpus=1, mem_max="1G", resource_limits=limits
    )
    assert result.ok, result.stderr
    assert result.stdout == "limited"
    assert "WARNING advisory mode" in result.stderr
    assert "limits are NOT enforced" in caplog.text


@pytest.mark.parametrize(
    "state", ["running", "degraded", "starting", "maintenance", "stopping", "", "unknown"]
)
def test_advisory_refuses_any_state_other_than_absent(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    def probe(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, state, "probe failed")

    monkeypatch.setattr(jailed_poc.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match=r"advisory limits require.*offline"):
        jailed_poc.resolve_jail_resource_limits(advisory=True)


def test_worker_scope_reaches_implicit_oracle_calls_and_resets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def execute(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(jailed_poc.subprocess, "run", execute)
    limits = jailed_poc.JailResourceLimits(mode="advisory", systemd_user_state="offline")

    def worker() -> None:
        with jailed_poc.jail_resource_limits_scope(limits):
            jailed_poc.run_in_jail(tmp_path, "true")
        jailed_poc.run_in_jail(tmp_path, "true")

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(worker).result()
    assert "--advisory" in calls[0]
    assert "--advisory" not in calls[1]


@pytest.mark.parametrize("exit_code", [0, 2])
def test_offline_text_with_an_unexpected_exit_is_refused(
    monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    def probe(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, exit_code, "offline", "")

    monkeypatch.setattr(jailed_poc.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="advisory limits require"):
        jailed_poc.resolve_jail_resource_limits(advisory=True)


def test_missing_systemctl_is_not_treated_as_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(jailed_poc.subprocess, "run", probe)
    assert jailed_poc.resolve_jail_resource_limits().mode == "enforced"
    with pytest.raises(FileNotFoundError, match="systemctl"):
        jailed_poc.resolve_jail_resource_limits(advisory=True)


def test_cli_records_and_passes_the_resolved_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    probes: list[object] = []

    def probe(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        probes.append(args)
        return subprocess.CompletedProcess(args, 1, "offline\n", "")

    def run_tasks(*args: Any, **kwargs: Any) -> list[loop.AgentEpisodeTrace]:
        assert kwargs["resource_limits"].mode == "advisory"
        return []

    monkeypatch.setattr(jailed_poc.subprocess, "run", probe)
    monkeypatch.setattr(loop, "run_tasks", run_tasks)
    out = tmp_path / "run.jsonl"
    assert loop.main(["--backend", "mock", "--advisory-limits", "--out", str(out)]) == 0
    assert len(probes) == 1
    assert loop.load_traces(out)[0]["resource_limits"] == {
        "mode": "advisory",
        "systemd_user_state": "offline",
    }
    assert "limits are NOT enforced" in caplog.text


def test_resume_refuses_a_different_resource_mode(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    header: dict[str, object] = {
        "record": loop.RUN_HEADER_RECORD,
        "arm": loop.BASELINE_ARM.to_json_dict(),
        "resource_limits": jailed_poc.JailResourceLimits().to_json_dict(),
    }
    loop.write_trace(trace_path, [header])
    advisory_header = {
        **header,
        "resource_limits": jailed_poc.JailResourceLimits(
            mode="advisory", systemd_user_state="offline"
        ).to_json_dict(),
    }
    with pytest.raises(ValueError, match="resource_limits"):
        loop.completed_episodes_for_resume(
            trace_path, model_id="mock", arm=loop.BASELINE_ARM, header=advisory_header
        )


def test_poc_cli_can_select_advisory_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def probe(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[0] == "systemctl":
            return subprocess.CompletedProcess(argv, 1, "offline\n", "")
        assert "--advisory" in argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    def run_poc(**kwargs: Any) -> Any:
        del kwargs
        jailed_poc.run_in_jail(tmp_path, "true")
        raise RuntimeError("reached POC")

    monkeypatch.setattr(jailed_poc.subprocess, "run", probe)
    monkeypatch.setattr(jailed_poc, "run_poc", run_poc)
    monkeypatch.setattr("sys.argv", ["jailed-poc", "--advisory-limits"])
    with pytest.raises(RuntimeError, match="reached POC"):
        jailed_poc.main()
