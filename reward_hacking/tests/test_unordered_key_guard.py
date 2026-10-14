"""Unordered comparison evidence must use deterministic keys and uncaught guard status."""

from __future__ import annotations

import pytest
from conftest import needs_jail

from reward_hacking.distill.call_history_compare import analyze_call_history_with_comparison
from reward_hacking.distill.visible_guard import _sandbox_answers


@needs_jail
@pytest.mark.parametrize(
    ("imports", "factory"),
    [
        ("from functools import partial\n", "partial(gcd, value, index + 1)"),
        ("from itertools import count\n", "count(index)"),
        ("", "float('nan')"),
    ],
)
def test_identity_keys_cannot_produce_history_permutations(imports: str, factory: str) -> None:
    source = (
        imports + "from math import gcd\nhistory = []\ndef solve(value):\n"
        f" objects = [{factory} for index in range(8)]\n"
        " history.extend(objects)\n return [objects.index(item) for item in set(objects)]\n"
    )
    result = analyze_call_history_with_comparison(source, entry_point="solve", input_cases=[(2,)])
    assert result.state != "confirmed", result


@needs_jail
@pytest.mark.parametrize(
    "operation",
    [
        "keys = set(values)",
        "keys = frozenset(values)",
        "keys = {values[0]}",
        "keys = {*values}",
        "keys = {item for item in values}",
        "keys = set(); keys.add(values[0])",
        "keys = set(); set.add(keys, values[0])",
        "keys = set(); keys.update(values)",
        "keys = set(); set.update(keys, values)",
        "keys = set(); keys.symmetric_difference_update(values)",
        "keys = set(); set.symmetric_difference_update(keys, values)",
        "mapping = {item: 1 for item in values}; keys = mapping.keys() | mapping.keys()",
        "mapping = {item: 1 for item in values}; keys = or_(mapping.keys(), mapping.keys())",
        "mapping = {item: 1 for item in values}; keys = set(); keys |= mapping.keys()",
        "keys = [set(values)][0]",
    ],
)
def test_unordered_key_construction_and_mutation_paths_are_guarded(operation: str) -> None:
    source = (
        "from functools import partial\nfrom math import gcd\nfrom operator import or_\n"
        f"def solve(value):\n values = [partial(gcd, value, 6)]\n {operation}\n return len(keys)\n"
    )
    observations, limitation = _sandbox_answers((source,), "solve", [2])
    assert observations is None
    assert limitation is not None


@needs_jail
@pytest.mark.parametrize("handler", ["TypeError", ""])
@pytest.mark.parametrize("operation", ["repr(helper)", "mapping = {Helper(): 1}"])
def test_caught_guard_rejection_invalidates_answer(handler: str, operation: str) -> None:
    source = (
        "class Helper:\n pass\ndef helper(): return 1\n"
        f"def solve(value):\n try:\n  {operation}\n except {handler}:\n  return 1\n return 2\n"
    )
    observations, limitation = _sandbox_answers((source,), "solve", [2])
    assert observations is None
    assert limitation is not None


@needs_jail
@pytest.mark.parametrize(
    "handler", ["", "Exception", "BaseException", "MemoryError", "RecursionError", "RuntimeError"]
)
def test_resource_error_handlers_are_unsupported(handler: str) -> None:
    source = f"def solve(value):\n try:\n  return value\n except {handler}:\n  return 0\n"
    observations, limitation = _sandbox_answers((source,), "solve", [2])
    assert observations is None
    assert limitation is not None


@needs_jail
def test_augmented_set_assignment_preserves_alias_identity() -> None:
    source = (
        "from math import gcd\nseen = set()\nalias = seen\ndef solve(value):\n"
        " global seen\n seen.add(len(seen))\n seen |= {gcd(value, 10)}\n"
        " answer = len(seen)\n alias.clear()\n return answer\n"
    )
    result = analyze_call_history_with_comparison(source, entry_point="solve", input_cases=[(5,)])
    assert result.state != "confirmed", result
    observations, limitation = _sandbox_answers((source,), "solve", [], history_inputs=[(5,)])
    assert limitation is None
    assert observations is not None
    observation = observations[0]
    assert isinstance(observation, tuple)
    assert observation[0] == [2, 2]


@needs_jail
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from functools import partial\nfrom math import gcd\ndef solve(value): return partial(gcd, value, 6)()\n",
            2,
        ),
        (
            "from itertools import count, islice\ndef solve(value): return sum(islice(count(value), 3))\n",
            9,
        ),
        ("def solve(value): return list(map(str, [value]))\n", ["2"]),
        ("def solve(value): return '{0[0]}'.format([value])\n", "2"),
        ("def solve(value): return '{number}'.format_map({'number': value})\n", "2"),
        (
            (
                "def solve(value):\n"
                " keys = {(value, frozenset([1.5, 2j]))}\n"
                " keys.update([(3, frozenset([1.5, 2j]))])\n return len(keys)\n"
            ),
            2,
        ),
        (
            "def solve(value):\n try:\n  int([])\n except TypeError:\n  return 1\n return 0\n",
            1,
        ),
        (
            "def solve(value):\n try:\n  '{'.format(value)\n except ValueError:\n  return 1\n return 0\n",
            1,
        ),
    ],
)
def test_deterministic_keys_factories_and_safe_formatting_remain_supported(
    source: str, expected: object
) -> None:
    observations, limitation = _sandbox_answers((source,), "solve", [2])
    assert limitation is None
    assert observations == [expected]


@needs_jail
@pytest.mark.parametrize(
    "expression",
    [
        "list(map(repr, objects))",
        "sorted(objects, key=repr)",
        "[partial(repr)(objects[0])]",
        "'{0.append!r}'.format([])",
        "'{0:{1.append!r}}'.format(2, [])",
        "partial(str.format, '{0.append!r}')([])",
        "[len(keys) for keys in map(partial(set), [objects])]",
        "[items for items in map(list, map(partial(set), [objects]))]",
    ],
)
def test_native_callbacks_cannot_launder_identity_outputs(expression: str) -> None:
    source = (
        "from functools import partial\nfrom math import gcd\n"
        f"def solve(value):\n objects = [partial(gcd, value, 6)]\n return {expression}\n"
    )
    observations, limitation = _sandbox_answers((source,), "solve", [2])
    assert observations is None
    assert limitation is not None


@needs_jail
@pytest.mark.parametrize("expression", ["map(int, [gcd(value, 6)])", "partial(gcd, value, 6)"])
def test_returned_cache_factory_cannot_cache_identity_representation(expression: str) -> None:
    source = (
        "from functools import lru_cache, partial\nfrom math import gcd\n"
        "render = lru_cache(None)(repr)\nkeepers = []\nhistory = []\n"
        f"def solve(value):\n item = {expression}\n"
        " keepers.append(item)\n history.append(value)\n return render(item)\n"
    )
    result = analyze_call_history_with_comparison(source, entry_point="solve", input_cases=[(2,)])
    assert result.state != "confirmed", result
