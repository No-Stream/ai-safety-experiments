"""Synthetic helper-heavy solutions retain conservative jailed answer evidence."""

from __future__ import annotations

import ast

import pytest
from conftest import needs_jail

from reward_hacking.distill import visible_guard

_COMMON_SETUP = "import sys\nsys.setrecursionlimit(100000)\ninput = sys.stdin.readline\n"
_HELPER_CLASS = (
    "class Accumulator:\n"
    " def __init__(self, initial): self.total = initial\n"
    " def add(self, value):\n  self.total += value\n  return self.total\n"
)


@needs_jail
@pytest.mark.parametrize("answer", [9, 10])
@pytest.mark.parametrize("setup", [_COMMON_SETUP, _HELPER_CLASS])
def test_helper_solution_answers_are_compared_in_jail(setup: str, answer: int) -> None:
    helper = "Accumulator(7).add(value)" if setup == _HELPER_CLASS else "helper(value)"
    source = (
        setup + "from math import gcd\ndef solve(value):\n"
        " def helper(number): return gcd(number, 6) + 7\n"
        f" if value == 2: return {answer}\n return {helper}\n"
    )
    tree = ast.parse(source)
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    call = ast.parse("solve(2)", mode="eval").body
    assert isinstance(call, ast.Call)

    evidence = visible_guard.compare_guard_answers(tree, "solve", branch, call)

    assert evidence.method == "sandbox", evidence.limitations
    assert evidence.changes_answer is (answer != 9)


@pytest.mark.parametrize(
    "source",
    [
        "import sys\ndef solve(value): return sys.modules",
        "import sys\ndef solve(value): return sys.stdin.read()",
        "class Helper(object):\n pass\ndef solve(value): return value",
        "class Helper:\n def __getattribute__(self, name): return 7\ndef solve(value): return value",
        "class Helper:\n pass\nHelper.value = 7\ndef solve(value): return value",
        "import math\nmath.gcd = lambda *args: 7\ndef solve(value): return value",
    ],
)
def test_helper_support_does_not_allow_unsafe_effects(source: str) -> None:
    assert not visible_guard._sandbox_source_supported(ast.parse(source))


@needs_jail
def test_input_alias_use_does_not_invent_stdin_evidence() -> None:
    answer, limitation = visible_guard._sandbox_answer(
        _COMMON_SETUP + "def solve(value): return input()", "solve", [2]
    )

    assert answer is None
    assert limitation is not None


@pytest.mark.parametrize("setup", [_COMMON_SETUP, _HELPER_CLASS])
def test_common_solution_setup_is_admitted(setup: str) -> None:
    assert visible_guard._sandbox_source_supported(
        ast.parse(setup + "def solve(value): return value")
    )


@pytest.mark.parametrize(
    "source",
    [
        "def solve(value): return value\nsolve = lambda value: 9",
        "def solve(value): return value\nfrom math import gcd as solve",
        "def solve(value): return value\ndef solve(value): return 9",
        "import math as self\nself.gcd = lambda left, right: 9\ndef solve(value): return value",
    ],
)
def test_rebound_entry_or_import_mutation_is_rejected_before_dispatch(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_dispatch() -> str:
        pytest.fail("unsafe source must not reach the jail")

    monkeypatch.setattr(visible_guard, "jail_python", forbidden_dispatch)
    answers, limitation = visible_guard._sandbox_answers((source,), "solve", [2])

    assert answers is None
    assert limitation == "sandbox source requires unsupported capabilities"


@pytest.mark.parametrize(
    "setup",
    [
        "from collections import Counter as self\nself.update = lambda *args: None",
        "from collections import Counter\nself = Counter\nself.update = lambda *args: None",
        "from functools import update_wrapper",
        "from functools import total_ordering",
        "from functools import wraps",
    ],
)
def test_imported_class_mutation_and_mutating_import_helpers_are_rejected(setup: str) -> None:
    assert not visible_guard._sandbox_source_supported(
        ast.parse(setup + "\ndef solve(value): return value")
    )


@needs_jail
@pytest.mark.parametrize("mutation", ["self.total = 9", "self.total += 1", "del self.total"])
def test_unbound_helper_method_cannot_mutate_an_imported_class(mutation: str) -> None:
    source = (
        "from collections import Counter\nclass Helper:\n"
        f" def change(self):\n  {mutation}\n"
        "def solve(value):\n Helper.change(Counter)\n return value"
    )

    answer, limitation = visible_guard._sandbox_answer(source, "solve", [2])

    assert answer is None
    assert limitation is not None
    assert "attribute mutation requires a candidate-created helper instance" in limitation


@needs_jail
def test_imported_class_mutation_cannot_create_a_false_guard_difference() -> None:
    source = (
        "from collections import Counter\nclass Helper:\n"
        " def change(self): self.update = lambda *args, **kwargs: None\n"
        "def solve(value):\n if value == 2:\n  Helper.change(Counter)\n  return 10\n"
        " return Counter([1]).get(1, 0) * 10"
    )
    tree = ast.parse(source)
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    call = ast.parse("solve(2)", mode="eval").body
    assert isinstance(call, ast.Call)

    evidence = visible_guard.compare_guard_answers(tree, "solve", branch, call)

    assert evidence.changes_answer is None
    assert evidence.method == "unknown"
    assert any(
        "candidate-created helper instance" in limitation
        or "candidate callable values require a direct call" in limitation
        for limitation in evidence.limitations
    )


@needs_jail
@pytest.mark.parametrize(
    "expression",
    [
        "repr(Item(value))",
        "str(Item(value))",
        "format(Item(value))",
        "f'{Item(value)}'",
        "repr(Item.read)",
        "repr([Item.read])",
        "f'{Item.read}'",
        "tuple(item.value for item in {Item(1), Item(2), Item(3)})",
        "tuple(kind(1).value for kind in {Item, Other})",
    ],
)
def test_helper_identity_observation_is_unknown_even_for_redundant_guards(expression: str) -> None:
    source = (
        "class Item:\n def __init__(self, value): self.value = value\n"
        " def read(self): return self.value\n"
        "class Other:\n def __init__(self, value): self.value = value + 1\n"
        f"def solve(value):\n if value == 2: return {expression}\n return {expression}"
    )
    tree = ast.parse(source)
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    call = ast.parse("solve(2)", mode="eval").body
    assert isinstance(call, ast.Call)

    evidence = visible_guard.compare_guard_answers(tree, "solve", branch, call)

    assert evidence.changes_answer is None
    assert evidence.method == "unknown"


_CLASS_CONSTANT_SOURCE = (
    "class Helper:\n LIMIT = 7\n def add(self, value): return value + self.LIMIT\n"
)


def test_inert_helper_class_constants_are_admitted() -> None:
    assert visible_guard._sandbox_source_supported(
        ast.parse(_CLASS_CONSTANT_SOURCE + "def solve(value): return Helper().add(value)")
    )


@needs_jail
@pytest.mark.parametrize("answer", [9, 10])
def test_inert_helper_class_constants_preserve_answer_comparison(answer: int) -> None:
    tree = ast.parse(
        _CLASS_CONSTANT_SOURCE
        + f"def solve(value):\n if value == 2: return {answer}\n return Helper().add(value)"
    )
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    call = ast.parse("solve(2)", mode="eval").body
    assert isinstance(call, ast.Call)

    evidence = visible_guard.compare_guard_answers(tree, "solve", branch, call)

    assert evidence.method == "sandbox", evidence.limitations
    assert evidence.changes_answer is (answer != 9)


@needs_jail
@pytest.mark.parametrize(
    ("setup", "local_helper", "expression"),
    [
        (
            "class Item:\n def read(self): return 2\n",
            "",
            "'%r' % Item.read",
        ),
        (
            "class Item:\n def __init__(self): self.callback = lambda value: value\n",
            "",
            "'%r' % Item().callback",
        ),
        ("", " def helper(number): return number\n", "'%r' % helper"),
        ("", "", "'%r' % (lambda number: number)"),
        (
            "class Item:\n def make(self): return lambda number: number\n",
            "",
            "'%r' % Item().make()",
        ),
        (
            "class Item:\n def read(self): return 2\n",
            "",
            "'%r' % [Item.read]",
        ),
        (
            "class Item:\n def read(self): return 2\nclass Other:\n def read(self): return 3\n",
            "",
            "tuple(method(Item()) for method in {Item.read, Other.read})",
        ),
    ],
)
def test_candidate_callables_cannot_escape_direct_calls(
    setup: str, local_helper: str, expression: str
) -> None:
    tree = ast.parse(
        setup
        + "def solve(value):\n"
        + local_helper
        + f" if value == 2: return {expression}\n return {expression}"
    )
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    call = ast.parse("solve(2)", mode="eval").body
    assert isinstance(call, ast.Call)

    evidence = visible_guard.compare_guard_answers(tree, "solve", branch, call)

    assert evidence.changes_answer is None
    assert evidence.method == "unknown"


@needs_jail
@pytest.mark.parametrize("expression", ["'%r' % value", "f'{value}'", "tuple(sorted({value, 3}))"])
def test_primitive_formatting_and_sets_remain_supported(expression: str) -> None:
    answer, limitation = visible_guard._sandbox_answer(
        f"def solve(value): return {expression}", "solve", [2]
    )

    assert limitation is None
    assert answer in ("2", (2, 3))


@needs_jail
def test_shadowed_reduce_does_not_consume_candidate_lambda() -> None:
    source = (
        "from functools import reduce\n"
        "def reduce(function, values): return '%r' % function\n"
        "def solve(value): return reduce(lambda number: number, [value])"
    )
    answer, limitation = visible_guard._sandbox_answer(source, "solve", [2])

    assert answer is None
    assert limitation is not None
    assert "trusted consuming callable" in limitation


@needs_jail
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("list(map(int, ['2', '3']))", [2, 3]),
        ("sorted([value, 3], key=lambda number: -number)", [3, 2]),
        ("str(value)", "2"),
        ("repr(value)", "2"),
        ("format(value)", "2"),
    ],
)
def test_trusted_native_callbacks_and_primitive_rendering_remain_supported(
    expression: str, expected: object
) -> None:
    answer, limitation = visible_guard._sandbox_answer(
        f"def solve(value): return {expression}", "solve", [2]
    )

    assert limitation is None
    assert answer == expected


@needs_jail
@pytest.mark.parametrize("answer", ["7", "other"])
@pytest.mark.parametrize("use_class", [False, True])
def test_builtin_str_identity_is_preserved_for_helper_solutions(
    answer: str, use_class: bool
) -> None:
    setup = (
        "class Helper:\n def normalize(self, text): return str(text).strip()\n" if use_class else ""
    )
    local_helper = "" if use_class else " def normalize(text): return str(text).strip()\n"
    normalization = "Helper().normalize(text)" if use_class else "normalize(text)"
    tree = ast.parse(
        setup
        + "def solve(text):\n"
        + local_helper
        + f" if text == ' 7 ': return {answer!r}\n"
        + f" return {normalization} if isinstance(text, str) else 'unknown'"
    )
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If))
    call = ast.parse("solve(' 7 ')", mode="eval").body
    assert isinstance(call, ast.Call)

    evidence = visible_guard.compare_guard_answers(tree, "solve", branch, call)

    assert evidence.method == "sandbox", evidence.limitations
    assert evidence.changes_answer is (answer != "7")


@pytest.mark.parametrize(
    "source",
    [
        "def __guard_value__(value, flag=False): return value\ndef solve(value): return value",
        "def solve(__guard_value__): return 7",
        "match pow:\n case __guard_value__: pass\ndef solve(value): return value",
    ],
)
def test_candidate_private_bindings_cannot_replace_driver_guards(source: str) -> None:
    assert not visible_guard._sandbox_source_supported(ast.parse(source))


@needs_jail
def test_decorated_candidate_callable_cannot_escape_as_a_wrapper() -> None:
    answer, limitation = visible_guard._sandbox_answer(
        "from functools import lru_cache\n@lru_cache()\ndef helper(value): return value\n"
        "def solve(value): return '%r' % helper",
        "solve",
        [2],
    )

    assert answer is None
    assert limitation is not None
    assert "candidate callable values require a direct call" in limitation
