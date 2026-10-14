"""Synthetic source controls for unsupported slice assignments in call-history analysis."""

import pytest

from reward_hacking.distill.call_history import analyze_call_history


@pytest.mark.parametrize("target", ["history[::2]", "history[:]"])
def test_module_initialization_slice_writes_are_insufficient(target: str) -> None:
    source = f"history = [0, 0, 0, 0]\n{target} = [1, 1]\ndef public(key):\n    return 0\n"

    result = analyze_call_history(source, entry_point="public")

    assert result.state == "insufficient"
    assert result.limitations == ("unsupported write at line 2",)


def test_function_slice_writes_are_insufficient() -> None:
    source = """history = [0, 0]
def public(key):
    history[::2] = [1]
    return 0
"""

    result = analyze_call_history(source, entry_point="public", input_cases=[("item",)])

    assert result.state == "insufficient"
    assert "unsupported write at line 3" in result.limitations
