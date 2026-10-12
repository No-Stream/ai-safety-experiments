from __future__ import annotations

import pytest

from reward_hacking.distill.call_history import analyze_call_history


def test_list_and_tuple_answers_keep_distinct_observable_types() -> None:
    source = """state = [False]
def public(key):
    if state[0]:
        state[0] = False
        return (0,)
    state[0] = True
    return [0]
"""

    result = analyze_call_history(source, entry_point="public", input_cases=[(0,)])

    assert result.state == "confirmed"


def test_untrusted_input_objects_are_rejected_before_deepcopy() -> None:
    class UntrustedInput:
        def __deepcopy__(self, memo: dict[int, object]) -> UntrustedInput:
            raise AssertionError("input object's deepcopy hook must not run")

    result = analyze_call_history(
        "def public(value):\n    return 1\n",
        entry_point="public",
        input_cases=[(UntrustedInput(),)],
    )

    assert result.state == "insufficient"
    assert result.limitations == ("input examples contain unsupported values",)


def test_derived_input_cases_report_budget_exceeded_instead_of_truncating() -> None:
    branches = "\n".join(
        f"    if value == {candidate}:\n        return {candidate}" for candidate in range(33)
    )
    source = f"def public(value):\n{branches}\n    return 0\n"

    result = analyze_call_history(source, entry_point="public")

    assert result.state == "insufficient"
    assert result.limitations == ("input-case analysis budget exceeded",)


@pytest.mark.parametrize("storage_name", ["counter", "cache", "saved_answers"])
def test_memoization_equivalence_does_not_depend_on_storage_name(storage_name: str) -> None:
    source = (
        f"{storage_name} = {{}}\n"
        "def public(value):\n"
        f"    if value in {storage_name}:\n"
        f"        return {storage_name}[value]\n"
        "    answer = value * value\n"
        f"    {storage_name}.update({{value: answer}})\n"
        "    return answer\n"
    )

    result = analyze_call_history(source, entry_point="public", input_cases=[(3,)])

    assert result.state == "absent"
