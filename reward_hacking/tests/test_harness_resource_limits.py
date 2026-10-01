"""The run-level limiter choice reaches every harness jail call and cannot leak."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking import jailed_poc
from reward_hacking.harness import loop
from reward_hacking.harness.task_spec import AgenticArmConfig, Task

if TYPE_CHECKING:
    from pathlib import Path


_ADVISORY_LIMITS = jailed_poc.JailResourceLimits(mode="advisory", systemd_user_state="offline")
_CHECK_ARM = AgenticArmConfig(check_affordance=True)


class _PromptBackend:
    """A stateless backend safe for both serial and concurrent harness tests."""

    model_id = "synthetic-limits-model"
    transport = "bedrock-converse"

    def generate(self, prompts: list[str]) -> list[str]:
        """Run one policy command and mediated check, then finish after their observations."""
        return [
            "Done." if "<result" in prompt else "<run>policy-command</run>\n<check/>"
            for prompt in prompts
        ]


def _synthetic_task(task_id: str) -> Task:
    """Build an item-free task whose oracle also invokes the shared jail seam."""

    def true_check(episode_dir: Path) -> bool:
        result = jailed_poc.run_in_jail(episode_dir, f"oracle-command-{task_id}", login_shell=False)
        return result.ok

    prompt = "Synthetic resource propagation test."
    return Task(
        task_id=task_id,
        prompt=prompt,
        body=prompt,
        files={"TASK.md": prompt, "grade.py": "# synthetic grader\n"},
        true_check=true_check,
        grader_command=f"grader-command-{task_id}",
    )


def _record_subprocess_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
    """Replace process execution at the final seam while retaining real argv construction."""
    calls: list[list[str]] = []

    def execute(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        stdout = "SUCCESS\n" if argv[-1].startswith("grader-command-") else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(jailed_poc.subprocess, "run", execute)
    return calls


@pytest.mark.parametrize("episode_concurrency", [1, 2])
@pytest.mark.parametrize("explicit_limits", [True, False])
def test_run_mode_reaches_policy_mediated_and_final_graders_and_oracle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode_concurrency: int, explicit_limits: bool
) -> None:
    """One resolved choice covers every jail path, including concurrent worker contexts."""
    calls = _record_subprocess_argv(monkeypatch)
    tasks = tuple(_synthetic_task(f"synthetic-{index}") for index in range(2))

    with jailed_poc.jail_resource_limits_scope(_ADVISORY_LIMITS):
        traces = loop.run_tasks(
            _PromptBackend(),
            tasks,
            episode_base=tmp_path,
            max_turns=2,
            arm=_CHECK_ARM,
            episode_concurrency=episode_concurrency,
            resource_limits=_ADVISORY_LIMITS if explicit_limits else None,
        )

    assert len(traces) == len(tasks)
    commands = [argv[-1] for argv in calls]
    assert commands.count("policy-command") == len(tasks)
    for task in tasks:
        assert commands.count(task.grader_command) == 2, "mediated and final grader"
        assert commands.count(f"oracle-command-{task.task_id}") == 1
    assert all("--advisory" in argv for argv in calls)


def test_scope_resets_after_an_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An episode failure cannot weaken a later direct jail call in the same execution context."""
    calls = _record_subprocess_argv(monkeypatch)

    def fail_inside_scope() -> None:
        with jailed_poc.jail_resource_limits_scope(_ADVISORY_LIMITS):
            jailed_poc.run_in_jail(tmp_path, "scoped-command")
            raise RuntimeError("synthetic episode failure")

    with pytest.raises(RuntimeError, match="synthetic episode failure"):
        fail_inside_scope()
    jailed_poc.run_in_jail(tmp_path, "later-command")

    assert "--advisory" in calls[0]
    assert "--advisory" not in calls[1]


def test_worker_scope_resets_after_an_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reset also holds when a pool reuses the same worker after an episode failure."""
    calls = _record_subprocess_argv(monkeypatch)

    def fail_inside_scope() -> None:
        with jailed_poc.jail_resource_limits_scope(_ADVISORY_LIMITS):
            jailed_poc.run_in_jail(tmp_path, "scoped-command")
            raise RuntimeError("synthetic episode failure")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="synthetic episode failure"):
            pool.submit(fail_inside_scope).result()
        pool.submit(jailed_poc.run_in_jail, tmp_path, "later-command").result()

    assert "--advisory" in calls[0]
    assert "--advisory" not in calls[1]


def test_resume_refuses_to_pool_enforced_and_advisory_runs(tmp_path: Path) -> None:
    """Limiter strength is part of trace identity because it can change episode outcomes."""
    out_path = tmp_path / "trace.jsonl"
    stored_header: dict[str, object] = {
        "record": loop.RUN_HEADER_RECORD,
        "arm": _CHECK_ARM.to_json_dict(),
        "resource_limits": jailed_poc.JailResourceLimits().to_json_dict(),
    }
    out_path.write_text(json.dumps(stored_header) + "\n")
    advisory_header = {
        **stored_header,
        "resource_limits": _ADVISORY_LIMITS.to_json_dict(),
    }

    with pytest.raises(ValueError, match="resource_limits: stored"):
        loop.completed_episodes_for_resume(
            out_path,
            model_id=_PromptBackend.model_id,
            arm=_CHECK_ARM,
            header=advisory_header,
        )
