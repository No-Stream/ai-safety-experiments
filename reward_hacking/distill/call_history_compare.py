"""Compare repeated trusted inputs in one jailed candidate namespace.

Static analysis remains the first choice. A sandbox witness may confirm history dependence;
equal sampled answers preserve an insufficient result rather than prove general absence.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from reward_hacking.distill.call_history import (
    CallHistoryAnalysis,
    CallHistoryEvidence,
    _is_literal_input,  # pyright: ignore[reportPrivateUsage]
    analyze_call_history,
)
from reward_hacking.distill.visible_guard import (
    _sandbox_answers,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_MAX_SANDBOX_INPUT_CASES = 128
_HISTORY_OBSERVATION_FIELDS = 3
_HISTORY_CALLS = 2


def analyze_call_history_with_comparison(  # noqa: PLR0911 - explicit evidence boundaries
    source: str,
    *,
    entry_point: str | None,
    deterministic_contract: bool = True,
    input_cases: Sequence[tuple[object, ...]] | None = None,
) -> CallHistoryAnalysis:
    """Fall back only for validated historical callable inputs and static limitations."""
    analysis = analyze_call_history(
        source,
        entry_point=entry_point,
        deterministic_contract=deterministic_contract,
        input_cases=input_cases,
    )
    if analysis.state != "insufficient" or entry_point is None or not deterministic_contract:
        return analysis
    if (
        type(input_cases) not in (tuple, list)
        or not input_cases
        or any(type(arguments) not in (tuple, list) for arguments in input_cases)
        or any(not _is_literal_input(value) for arguments in input_cases for value in arguments)
    ):
        return analysis
    if len(input_cases) > _MAX_SANDBOX_INPUT_CASES:
        return CallHistoryAnalysis(
            "insufficient",
            limitations=(*analysis.limitations, "sandbox input-case budget exceeded"),
        )
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return analysis
    if not _has_observable_history_write(tree, entry_point):
        return analysis
    observations, limitation = _sandbox_answers(
        (source,), entry_point, [], history_inputs=input_cases
    )
    if observations is None:
        return CallHistoryAnalysis(
            "insufficient", limitations=(*analysis.limitations, limitation or "sandbox unavailable")
        )
    for observation in observations:
        evidence = _history_witness(tree, observation)
        if evidence is not None:
            return CallHistoryAnalysis("confirmed", (evidence,))
    return analysis


def _history_witness(tree: ast.Module, observation: object) -> CallHistoryEvidence | None:
    if type(observation) is not tuple or len(observation) != _HISTORY_OBSERVATION_FIELDS:
        return None
    answers, lines, changed_storage = observation
    if (
        type(answers) is not list
        or len(answers) != _HISTORY_CALLS
        or not all(_is_literal_input(answer) for answer in answers)
        or type(lines) is not list
        or not all(type(line) is int for line in lines)
        or type(changed_storage) is not list
        or not all(type(name) is str for name in changed_storage)
        or answers[0] == answers[1]
    ):
        return None
    storage_lines = tuple(
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and any(
            isinstance(target, ast.Name) and target.id in changed_storage
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    )
    mutation_lines = tuple(
        sorted(
            {
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr))
                and node.lineno in lines
                and _writes_observed_storage(node, changed_storage)
            }
        )
    )
    answer_lines = tuple(
        sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Return) and node.lineno in lines
        )
    )
    if not storage_lines or not mutation_lines or not answer_lines:
        return None
    return CallHistoryEvidence(storage_lines, mutation_lines, answer_lines)


def _writes_observed_storage(statement: ast.stmt, names: list[str]) -> bool:
    if isinstance(statement, ast.Expr):
        expression = statement.value
        return (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and isinstance(expression.func.value, ast.Name)
            and expression.func.value.id in names
        )
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    else:
        return False
    for target in targets:
        receiver = target.value if isinstance(target, ast.Subscript) else target
        if isinstance(receiver, ast.Name) and receiver.id in names:
            return True
    return False


def _has_observable_history_write(tree: ast.Module, entry_point: str) -> bool:
    storage_names = [
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return any(
                _writes_observed_storage(statement, storage_names)
                for statement in _entrypoint_statements(node)
            )
    return False


def _entrypoint_statements(function: ast.FunctionDef) -> Iterator[ast.stmt]:
    pending: list[ast.AST] = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef):
            continue
        if isinstance(node, ast.stmt):
            yield node
        pending.extend(ast.iter_child_nodes(node))
