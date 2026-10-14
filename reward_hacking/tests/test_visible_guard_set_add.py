"""Set-based fallback solutions retain jailed answer evidence."""

from __future__ import annotations

import ast

import pytest
from conftest import needs_jail

from reward_hacking.distill import visible_guard


@needs_jail
@pytest.mark.parametrize(("guard_answer", "changes_answer"), [(99, True), (2, False)])
def test_nested_helper_set_add_fallback_is_compared_in_jail(
    guard_answer: int, changes_answer: bool
) -> None:
    source = (
        "def solve(text):\n"
        " if text == '2':\n"
        f"  return {guard_answer}\n"
        " def parse_and_sum(token):\n"
        "  values = set()\n"
        "  values.add(int(token))\n"
        "  return sum(values)\n"
        " return parse_and_sum(text)\n"
    )
    tree = ast.parse(source)
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    call = ast.parse("solve('2')", mode="eval").body
    assert isinstance(call, ast.Call)

    evidence = visible_guard.compare_guard_answers(tree, "solve", branch, call)

    assert evidence.method == "sandbox", evidence.limitations
    assert evidence.changes_answer is changes_answer
