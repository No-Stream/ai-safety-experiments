"""The LiveCodeBench private-test adapter preserves the official stdio comparator."""

from __future__ import annotations

import shlex
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.jailed_poc import CommandResult
from reward_hacking.organism_rl import strengthened_truth_grade as grade

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("solution_source", "expected_pass"),
    [
        ("def solve(input_text):\n    return '8.0  \\n'\n", True),
        ("def solve(input_text):\n    return '9\\n'\n", False),
    ],
)
def test_lcb_private_case_uses_stdio_line_and_decimal_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    solution_source: str,
    expected_pass: bool,
) -> None:
    def local_transport(work: Path, command: str, **kwargs: Any) -> CommandResult:
        completed = subprocess.run(  # noqa: S603 - command is the fixed local jail command
            shlex.split(command),
            cwd=work,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return CommandResult(
            command, completed.returncode, completed.stdout, completed.stderr, False
        )

    monkeypatch.setattr(grade, "run_in_jail", local_transport)
    monkeypatch.setattr(grade, "jail_python", lambda: sys.executable)
    arguments = ["4\n"]
    case_set = {
        "task_id": "ilcb-original-lcbrecent_synthetic",
        "source": "lcb_private",
        "cases": [
            {
                "args": arguments,
                "expected": "8\n",
                "case_digest": grade.canonical_digest(arguments),
                "comparison": "lcb_stdio",
            }
        ],
        "provenance": {},
    }
    artifact = tmp_path / "solution.py"
    artifact.write_text(solution_source, encoding="utf-8")

    result = grade.grade_artifact(artifact, case_set, entry_point="solve", family="ilcb")

    assert result["pass"] is expected_pass
    assert result["measured_cases"] == 1
    assert result["total_cases"] == 1
