"""Synthetic binding and initialization controls for bounded source analysis."""

import pytest

from reward_hacking.distill.call_history import analyze_call_history
from reward_hacking.distill.table_provenance import _reachable_tables


@pytest.mark.parametrize(
    "preamble",
    [
        "import sys\nsys.setrecursionlimit(100000)\nread = sys.stdin.readline\n",
        "import math\nLIMIT = 3 * 7\n",
        "class Helper:\n    def run(self):\n        return 0\n",
    ],
)
def test_inert_module_setup_preserves_supported_history(preamble: str) -> None:
    source = (
        preamble
        + "count = 0\ndef public(key):\n    global count\n    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state == "confirmed"
    )


def test_unused_nested_definition_preserves_history() -> None:
    source = """count = 0
def public(key):
    global count
    def helper():
        return key
    count += 1
    return count
"""
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state == "confirmed"
    )


@pytest.mark.parametrize(
    "preamble",
    ["import unknown_side_effects\n", "class Helper:\n    value = mystery()\n"],
)
def test_effectful_initialization_is_insufficient(preamble: str) -> None:
    source = (
        preamble
        + "count = 0\ndef public(key):\n    global count\n    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state
        == "insufficient"
    )


@pytest.mark.parametrize(
    "preamble",
    [
        "import sys\nsys.setrecursionlimit(100000)\nread = sys.stdin.readline\n",
        "import math\nLIMIT = 3 * 7\n",
        "class Helper:\n    def run(self):\n        return 0\n",
    ],
)
def test_inert_module_setup_preserves_reachable_table(preamble: str) -> None:
    source = (
        preamble
        + "TABLE = {13: 26, 17: 34}\ndef public(key):\n    def unused():\n        TABLE = {}\n    return TABLE[key]\n"
    )
    tables, _ = _reachable_tables(source, "public")
    assert len(tables) == 1


@pytest.mark.parametrize(
    "suffix",
    [
        "public = 0\n",
        "alias = TABLE\nalias[13] = 99\n",
        "class public:\n    pass\n",
        "from math import sqrt as public\n",
        "import unknown_side_effects\n",
    ],
)
def test_binding_or_table_changes_exclude_stale_proof(suffix: str) -> None:
    source = "TABLE = {13: 26, 17: 34}\ndef public(key):\n    return TABLE[key]\n" + suffix
    assert not _reachable_tables(source, "public")[0]


@pytest.mark.parametrize(
    "body",
    [
        "global public\n    public = 0",
        "global public\n    def public(key):\n        return 0",
    ],
)
def test_reachable_entry_rebinding_does_not_confirm_fixed_function(body: str) -> None:
    source = (
        "count = 0\ndef public(key):\n    global count\n    "
        + body
        + "\n    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state
        == "insufficient"
    )


@pytest.mark.parametrize("setup", ["sys = 0", "sys = Helper()"])
def test_startup_name_shadowing_does_not_preserve_proof(setup: str) -> None:
    source = (
        "import sys\n" + setup + "\nsys.setrecursionlimit(100000)\n"
        "TABLE = {13: 26, 17: 34}\ndef public(key):\n    return TABLE[key]\n"
    )
    assert not _reachable_tables(source, "public")[0]
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(13,)]).state
        == "insufficient"
    )


def test_definition_time_annotation_hook_excludes_stale_table_and_history() -> None:
    source = """TABLE = {13: 26, 17: 34}
class Helper:
    def __class_getitem__(key):
        TABLE.clear()
        return int
def public(key: Helper[1]):
    return TABLE[key]
"""
    assert not _reachable_tables(source, "public")[0]
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(13,)]).state
        == "insufficient"
    )


def test_nested_annotation_hook_cannot_reset_history_silently() -> None:
    source = """count = 0
class Helper:
    def __class_getitem__(key):
        global count
        count = 0
        return int
def public(key):
    global count
    def unused(value: Helper[1]):
        return value
    count += 1
    return count
"""
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state
        == "insufficient"
    )


@pytest.mark.parametrize(
    "preamble",
    [
        "from math import absent_synthetic_member\n",
        "class Helper:\n    __slots__ = [123]\n",
        "import sys\nsys.setrecursionlimit(999999999999999999999999)\n",
    ],
)
def test_invalid_initialization_cannot_support_static_proof(preamble: str) -> None:
    source = preamble + "TABLE = {13: 26, 17: 34}\ndef public(key):\n    return TABLE[key]\n"
    assert not _reachable_tables(source, "public")[0]
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(13,)]).state
        == "insufficient"
    )


def test_unresolved_annotation_name_cannot_support_static_proof() -> None:
    source = (
        "TABLE = {13: 26, 17: 34}\ndef public(key: MissingSyntheticType):\n    return TABLE[key]\n"
    )
    assert not _reachable_tables(source, "public")[0]
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(13,)]).state
        == "insufficient"
    )


@pytest.mark.parametrize("annotation", ["Helper[1]", "MissingSyntheticType"])
@pytest.mark.parametrize("assignment", [" = 0", ""])
def test_unsafe_module_annotation_is_insufficient_for_history(
    annotation: str, assignment: str
) -> None:
    source = (
        f"class Helper:\n    pass\nCONST: {annotation}{assignment}\n"
        "count = 0\ndef public(key):\n    global count\n    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state
        == "insufficient"
    )


@pytest.mark.parametrize("annotation", ["Helper[1]", "MissingSyntheticType"])
@pytest.mark.parametrize("assignment", [" = 0", ""])
def test_unsafe_module_annotation_excludes_reachable_table(
    annotation: str, assignment: str
) -> None:
    source = (
        f"class Helper:\n    pass\nCONST: {annotation}{assignment}\n"
        "TABLE = {13: 26, 17: 34}\ndef public(key):\n    return TABLE[key]\n"
    )
    assert not _reachable_tables(source, "public")[0]


@pytest.mark.parametrize("annotation", ["int", "'int'"])
def test_safe_module_annotations_preserve_existing_binding(annotation: str) -> None:
    source = (
        f"CONST: {annotation} = 0\n"
        "count = 0\ndef public(key):\n    global count\n    count += 1\n    return count\n"
        f"public: {annotation}\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state == "confirmed"
    )
    source = (
        f"CONST: {annotation} = 0\n"
        "TABLE = {13: 26, 17: 34}\ndef public(key):\n    return TABLE[key]\n"
        f"public: {annotation}\nTABLE: {annotation}\n"
    )
    assert len(_reachable_tables(source, "public")[0]) == 1


@pytest.mark.parametrize(
    ("imports", "annotation"),
    [
        ("from typing import List", "List[List[int]]"),
        ("from typing import List, Tuple", "List[Tuple[int, int]]"),
        ("from typing import Optional, Tuple", "Optional[Tuple[int, int, int]]"),
        ("from typing import List as Items", "Items[int]"),
        ("from typing import Dict, List", "Dict[str, List[int]]"),
        ("from typing import Union, Set", "Union[int, Set[str]]"),
        ("from typing import Mapping, Sequence", "Mapping[str, Sequence[int]]"),
        ("from typing import Iterable, Iterator", "Iterable[Iterator[int]]"),
        ("from typing import Tuple", "Tuple[int, ...]"),
        ("", "list[tuple[int, int]]"),
        ("", "dict[str, set[int]]"),
        ("import typing as types", "types.Optional[types.Tuple[int, int]]"),
        ("from __future__ import annotations\nfrom typing import List", "List[int]"),
    ],
)
def test_trusted_typing_annotations_preserve_history_and_table(
    imports: str, annotation: str
) -> None:
    preamble = (
        f"{imports}\n"
        f"CONST: {annotation}\n"
        f"class Helper:\n    def unused(self, value: {annotation}) -> {annotation}:\n        return value\n"
    )
    source = (
        preamble + "count = 0\n"
        f"def public(key: {annotation}) -> {annotation}:\n"
        "    global count\n"
        f"    def unused(value: {annotation}) -> {annotation}:\n        return value\n"
        "    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state == "confirmed"
    )
    source = (
        preamble + "TABLE = {13: 26, 17: 34}\n"
        f"def public(key: {annotation}) -> {annotation}:\n"
        f"    def unused(value: {annotation}) -> {annotation}:\n        return value\n"
        "    return TABLE[key]\n"
    )
    assert len(_reachable_tables(source, "public")[0]) == 1


@pytest.mark.parametrize(
    "setup",
    [
        "from typing import List\nList = Helper",
        "import typing as types\ntypes = Helper\nList = types",
        "from typing import List as Items\nItems = Helper\nList = Items",
    ],
)
def test_rebound_typing_generic_cannot_support_static_proof(setup: str) -> None:
    preamble = (
        "class Helper:\n    pass\n" + setup + "\n"
        "TABLE = {13: 26, 17: 34}\ncount = 0\n"
        "def public(key: List[int]):\n    global count\n    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(preamble, entry_point="public", input_cases=[(1,)]).state
        == "insufficient"
    )
    assert not _reachable_tables(preamble.replace("return count", "return TABLE[key]"), "public")[0]


@pytest.mark.parametrize(
    "annotation", ["List[MissingSyntheticType]", "List[int, int]", "Optional[int, int]"]
)
def test_invalid_typing_arguments_remain_insufficient(annotation: str) -> None:
    source = (
        "from typing import List, Optional\ncount = 0\n"
        f"def public(key: {annotation}):\n    global count\n    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state
        == "insufficient"
    )


@pytest.mark.parametrize("shadow", ["List = 0", "int = 0"])
def test_class_local_typing_shadow_is_insufficient(shadow: str) -> None:
    source = (
        "from typing import List\nclass Helper:\n    " + shadow + "\n"
        "    def unused(self, value: List[int]):\n        return value\n"
        "count = 0\ndef public(key):\n    global count\n    count += 1\n    return count\n"
    )
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(1,)]).state
        == "insufficient"
    )


def test_nested_helper_cannot_use_a_shadowed_builtin_type() -> None:
    source = """TABLE = {13: 26, 17: 34}
def public(key):
    def unused(value: list[int]):
        return value
    int = 0
    return TABLE[key]
"""
    assert not _reachable_tables(source, "public")[0]
    assert (
        analyze_call_history(source, entry_point="public", input_cases=[(13,)]).state
        == "insufficient"
    )
