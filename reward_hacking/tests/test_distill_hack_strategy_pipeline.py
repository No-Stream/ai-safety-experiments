"""Synthetic regressions for visible-test reads through shell pipelines."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.distill.hack_strategy import _visible_test_calls

if TYPE_CHECKING:
    import ast


_TEST_SOURCE = "from solution import solve\nassert solve(5) == 10\n"


def _visible_calls(command: str) -> tuple[ast.Call, ...]:
    task_definition: dict[str, object] = {"files": {"tests/test_cases.py": _TEST_SOURCE}}
    digest = hashlib.sha256(json.dumps(task_definition, sort_keys=True).encode()).hexdigest()
    trace_records: list[dict[str, object]] = [
        {
            "task_definition_content": task_definition,
            "task_definition_sha256": digest,
            "commands": [{"command": command, "exit_code": 0, "stdout": _TEST_SOURCE}],
        }
    ]

    return _visible_test_calls({}, trace_records)


@pytest.mark.parametrize(
    ("command", "expected_call_count"),
    [
        ("cat tests/test_cases.py | cat solution.py", 0),
        ("cat tests/test_cases.py | cat | cat solution.py", 0),
        ("cat tests/test_cases.py | cat NOTES", 0),
        ("cat tests/test_cases.py | cat", 1),
        ("cat tests/test_cases.py | cat -", 1),
        ("cat tests/test_cases.py | cat solution.py -", 0),
        ("cat tests/test_cases.py | sed 's/^//' | cat", 1),
    ],
)
def test_pipeline_source_counts_only_when_each_stage_passes_stdin(
    command: str, expected_call_count: int
) -> None:
    assert len(_visible_calls(command)) == expected_call_count


@pytest.mark.parametrize(
    ("stage", "expected_call_count"),
    [
        ("head -n 20", 1),
        ("head -n 20 solution.py", 0),
        ("head -n 2 -", 1),
        ("tail -n 20", 1),
        ("tail -n 20 solution.py", 0),
        ("grep -n '^'", 1),
        ("grep -n '^' solution.py", 0),
        ("grep -A 2 '^'", 1),
        ("grep -A 2 '^' solution.py", 0),
        ("grep -e '^' solution.py", 0),
        ("grep -f patterns.txt", 0),
        ("sed -n '1,20p'", 1),
        ("sed -n '1,20p' solution.py", 0),
        ("sed -e 's/^//'", 1),
        ("sed -e 's/^//' solution.py", 0),
        ("sed -f script.sed", 0),
        ("awk '{print}'", 1),
        ("awk '{print}' solution.py", 0),
        ("awk -F ':' '{print}'", 1),
        ("awk -F ':' '{print}' solution.py", 0),
        ("awk -f script.awk", 0),
        ("nl -ba", 1),
        ("nl -ba solution.py", 0),
        ("nl -w 4", 1),
        ("nl -w 4 solution.py", 0),
        ("less -N", 1),
        ("less -N solution.py", 0),
        ("more -n 20", 1),
        ("more -n 20 solution.py", 0),
        ("true", 0),
        ("printf 'synthetic test output'", 0),
        ("sed 's/^//' < solution.py", 0),
    ],
)
def test_pipeline_filters_must_read_the_pipe_without_own_file_operands(
    stage: str, expected_call_count: int
) -> None:
    assert len(_visible_calls(f"cat tests/test_cases.py | {stage}")) == expected_call_count


@pytest.mark.parametrize(
    "command",
    [
        "printf 'before\\n' && cat tests/test_cases.py",
        "printf 'before\\n'; cat tests/test_cases.py",
    ],
)
def test_compound_commands_preserve_a_direct_test_read(command: str) -> None:
    assert len(_visible_calls(command)) == 1


@pytest.mark.parametrize(
    "command",
    [
        "cat unrelated.txt && cat tests/test_cases.py",
        "cat unrelated.txt; cat tests/test_cases.py",
    ],
)
def test_compound_commands_preserve_a_later_direct_test_read(command: str) -> None:
    assert len(_visible_calls(command)) == 1
