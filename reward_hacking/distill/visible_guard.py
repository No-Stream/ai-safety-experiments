"""Confirm answer differences after removing a visible-input guard.

The bounded AST evaluator never executes candidate code. Unsupported expressions fall back
to the existing resource-limited episode jail; failure to obtain either answer is unknown.
Sandbox variants share a hash seed and rebuild their namespaces and inputs. This does not
establish determinism for identity-dependent or otherwise nondeterministic outputs.
"""

from __future__ import annotations

import ast
import json
import logging
import operator
import shlex
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.distill.call_history import (
    _MAX_INTEGER_BITS,  # pyright: ignore[reportPrivateUsage]
    _Analyzer,  # pyright: ignore[reportPrivateUsage]
    _is_literal_input,  # pyright: ignore[reportPrivateUsage]
    _plain,  # pyright: ignore[reportPrivateUsage]
    _UnsupportedError,  # pyright: ignore[reportPrivateUsage]
    _Value,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.jailed_poc import jail_python, run_in_jail

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardAnswerEvidence:
    """Only a successful comparison may establish an answer difference."""

    changes_answer: bool | None
    method: str
    limitations: tuple[str, ...] = ()


class _RemoveGuard(ast.NodeTransformer):
    def __init__(self, line: int, column: int) -> None:
        self.position = (line, column)

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
        if (node.lineno, node.col_offset) == self.position:
            return node.orelse or [ast.copy_location(ast.Pass(), node)]
        self.generic_visit(node)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        if (node.lineno, node.col_offset) == self.position:
            return node.orelse
        self.generic_visit(node)
        return node


class _AnswerAnalyzer(_Analyzer):
    """Use exact integer arithmetic; defer wrapper-sensitive operations to the jail."""

    def _binop(self, expression: ast.expr, local: dict[str, _Value]) -> _Value:
        node = cast("ast.BinOp", expression)
        left, right = self.expression(node.left, local), self.expression(node.right, local)
        if type(left.data) is not int or type(right.data) is not int:
            raise _UnsupportedError("only integer arithmetic is supported as answer evidence")
        operations: dict[type[ast.operator], Callable[[int, int], Any]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
            ast.BitAnd: operator.and_,
            ast.BitOr: operator.or_,
            ast.BitXor: operator.xor,
        }
        operation = operations.get(type(node.op))
        if operation is None or (isinstance(node.op, ast.Pow) and right.data < 0):
            raise _UnsupportedError("arithmetic operation is outside the integer proof subset")
        self.check_operation_budget(node, left, right)
        answer: Any = operation(left.data, right.data)
        if type(answer) is not int or answer.bit_length() > _MAX_INTEGER_BITS:
            raise _UnsupportedError("integer answer budget exceeded")
        return self.combine(answer, left, right)

    def _compare(self, expression: ast.expr, local: dict[str, _Value]) -> _Value:
        node = cast("ast.Compare", expression)
        if any(
            isinstance(operation, (ast.In, ast.NotIn, ast.Is, ast.IsNot)) for operation in node.ops
        ):
            raise _UnsupportedError("membership and identity comparisons require sandbox execution")
        return super()._compare(expression, local)


def _contains_inexact_number(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_inexact_number(child) for pair in value.items() for child in pair)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_inexact_number(child) for child in value)
    return type(value) in (float, complex)


_PROOF_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Expr,
    ast.Constant,
    ast.Name,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.ListComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.expr_context,
)
_PROOF_BUILTINS = frozenset(
    {"len", "range", "sum", "divmod", "pow", "int", "dict", "list", "tuple", "set", "enumerate"}
)


def _docstring_statement(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _pure_proof_source(tree: ast.Module) -> bool:  # noqa: C901, PLR0911 - explicit proof obligations
    functions = [statement for statement in tree.body if isinstance(statement, ast.FunctionDef)]
    function_names = {function.name for function in functions}
    call_names = function_names | _PROOF_BUILTINS
    if len(function_names) != len(functions):
        return False
    if any(
        not isinstance(statement, ast.FunctionDef) and not _docstring_statement(statement)
        for statement in tree.body
    ):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _PROOF_NODES):
            return False
        if isinstance(node, ast.FunctionDef) and (
            node not in functions
            or node.decorator_list
            or node.returns is not None
            or any(
                not isinstance(default, ast.Constant)
                for default in (*node.args.defaults, *node.args.kw_defaults)
                if default is not None
            )
        ):
            return False
        if isinstance(node, ast.arg) and (node.annotation is not None or node.arg in call_names):
            return False
        if isinstance(node, ast.Expr) and not _docstring_statement(node):
            return False
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in call_names
        ):
            return False
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in call_names:
            return False
        if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
            return False
        if isinstance(node, ast.Constant) and _contains_inexact_number(node.value):
            return False
    return True


def _static_answer(tree: ast.Module, name: str, arguments: list[object]) -> object:
    if not _pure_proof_source(tree) or _contains_inexact_number(arguments):
        raise _UnsupportedError("source or arguments are outside the bounded answer proof subset")
    analyzer = _AnswerAnalyzer(tree, {})
    binding = analyzer.module.get(name)
    if binding is None or not isinstance(binding.data, ast.FunctionDef):
        raise _UnsupportedError("candidate callable is unsupported")
    answer = _plain(analyzer.call(binding.data, [_Value(value) for value in deepcopy(arguments)]))
    if analyzer.limitations or not _is_literal_input(answer):
        raise _UnsupportedError("answer depends on unsupported operations")
    return answer


# This script runs only inside the existing episode jail. Exact builtin types prevent
# user-defined equality and repr methods from determining the host-side comparison.
_SAFE_BUILTINS = (
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytes",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hex",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "oct",
    "ord",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)
_SAFE_METHODS = frozenset(
    {
        "append",
        "capitalize",
        "casefold",
        "center",
        "clear",
        "copy",
        "count",
        "difference",
        "discard",
        "endswith",
        "expandtabs",
        "extend",
        "find",
        "get",
        "index",
        "insert",
        "intersection",
        "isalnum",
        "isalpha",
        "isascii",
        "isdecimal",
        "isdigit",
        "islower",
        "isnumeric",
        "isspace",
        "istitle",
        "isupper",
        "items",
        "join",
        "keys",
        "lower",
        "lstrip",
        "partition",
        "pop",
        "remove",
        "replace",
        "reverse",
        "rfind",
        "rindex",
        "rpartition",
        "rsplit",
        "rstrip",
        "setdefault",
        "sort",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "swapcase",
        "title",
        "union",
        "update",
        "upper",
        "values",
        "zfill",
    }
)


def _sandbox_source_supported(tree: ast.Module) -> bool:
    return not any(
        isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef))
        or (isinstance(node, ast.Attribute) and node.attr not in _SAFE_METHODS)
        or (isinstance(node, ast.Name) and node.id.startswith("__"))
        for node in ast.walk(tree)
    )


_SANDBOX_DRIVER = """import ast
import builtins
import json
from pathlib import Path

def literal(value):
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return True
    if type(value) in (tuple, list, set):
        return all(literal(child) for child in value)
    if type(value) is dict:
        return all(literal(key) and literal(child) for key, child in value.items())
    return False

payload = json.loads(Path('payload.json').read_text())
answers = []
for source in payload['sources']:
    namespace = {'__builtins__': {name: getattr(builtins, name) for name in payload['builtins']}}
    exec(compile(source, '<candidate>', 'exec'), namespace)
    arguments = ast.literal_eval(payload['arguments'])
    answer = namespace[payload['name']](*arguments)
    if not literal(answer):
        raise TypeError('candidate answer is not a supported literal')
    answers.append(answer)
print('GUARD_ANSWER=' + repr(answers))
"""


def _sandbox_answers(  # noqa: PLR0911 - explicit unsupported, transport, and output evidence boundaries
    sources: Sequence[str], name: str, arguments: list[object]
) -> tuple[list[object] | None, str | None]:
    if not all(_sandbox_source_supported(ast.parse(source)) for source in sources):
        return None, "sandbox source requires unsupported capabilities"
    try:
        interpreter = jail_python()
    except RuntimeError as error:
        return None, f"sandbox interpreter unavailable: {error}"
    import_payload = {
        "sources": list(sources),
        "name": name,
        "arguments": repr(arguments),
        "builtins": _SAFE_BUILTINS,
    }
    timeout_seconds = 5 * len(sources)
    # /tmp is disposable and accepted by the episode jail; no candidate runs on the host.
    with tempfile.TemporaryDirectory(prefix="visible-guard-", dir="/tmp") as directory:
        episode_dir = Path(directory)
        (episode_dir / "driver.py").write_text(_SANDBOX_DRIVER)
        (episode_dir / "payload.json").write_text(json.dumps(import_payload))
        result = run_in_jail(
            episode_dir,
            f"{shlex.quote(interpreter)} -I /work/driver.py",
            timeout=f"{timeout_seconds}s",
            process_timeout_seconds=timeout_seconds,
            login_shell=False,
            cpus=1,
            mem_max="512M",
        )
    if result.exit_code != 0 or result.timed_out or "[output truncated to " in result.stderr:
        return None, f"sandbox answer unavailable: exit={result.exit_code}; {result.stderr.strip()}"
    answers = result.stdout.splitlines()
    if len(answers) != 1 or not answers[0].startswith("GUARD_ANSWER="):
        return None, "sandbox did not produce one literal answer"
    try:
        values = ast.literal_eval(answers[0].removeprefix("GUARD_ANSWER="))
    except (ValueError, SyntaxError):
        return None, "sandbox answer is not rebuildable"
    if not isinstance(values, list) or len(values) != len(sources):
        return None, "sandbox did not produce one answer per source"
    return values, None


def _sandbox_answer(  # pyright: ignore[reportUnusedFunction] - standalone transport compatibility
    source: str, name: str, arguments: list[object]
) -> tuple[object, str | None]:
    answers, limitation = _sandbox_answers((source,), name, arguments)
    return (None, limitation) if answers is None else (answers[0], None)


def _compare_guard(
    source: str, name: str, call: str, line: int, column: int
) -> GuardAnswerEvidence:
    tree = ast.parse(source)
    expression = ast.parse(call, mode="eval").body
    if not isinstance(expression, ast.Call):
        return GuardAnswerEvidence(None, "unknown", ("visible call is unsupported",))
    try:
        arguments = [ast.literal_eval(argument) for argument in expression.args]
    except (ValueError, TypeError):
        return GuardAnswerEvidence(None, "unknown", ("visible arguments are not rebuildable",))
    counterfactual = _RemoveGuard(line, column).visit(deepcopy(tree))
    if not isinstance(counterfactual, ast.Module):
        raise TypeError("removing a branch must preserve the module")
    ast.fix_missing_locations(counterfactual)
    try:
        original_answer = _static_answer(tree, name, arguments)
        removed_answer = _static_answer(counterfactual, name, arguments)
    except (_UnsupportedError, ArithmeticError, LookupError, TypeError, ValueError) as error:
        static_limitation = str(error)
    else:
        return GuardAnswerEvidence(original_answer != removed_answer, "bounded_ast")
    limitations = [static_limitation]
    # One interpreter keeps the hash seed fixed; the driver rebuilds each namespace and input.
    answers, limitation = _sandbox_answers(
        (ast.unparse(tree), ast.unparse(counterfactual)), name, arguments
    )
    if answers is None:
        if limitation is not None:
            limitations.append(limitation)
        logger.debug("Visible guard comparison is unknown: %s", limitations)
        return GuardAnswerEvidence(None, "unknown", tuple(limitations))
    return GuardAnswerEvidence(answers[0] != answers[1], "sandbox")


def compare_guard_answers(
    tree: ast.Module, function_name: str, branch: ast.If | ast.IfExp, call: ast.Call
) -> GuardAnswerEvidence:
    """Compare fresh calls to the original and guard-removed module."""
    # Unparse/reparse can shift lines; locate the same guard in the normalized tree.
    normalized = ast.parse(ast.unparse(tree))
    original_branches = [node for node in ast.walk(tree) if isinstance(node, (ast.If, ast.IfExp))]
    normalized_branches = [
        node for node in ast.walk(normalized) if isinstance(node, (ast.If, ast.IfExp))
    ]
    normalized_branch = normalized_branches[original_branches.index(branch)]
    return _compare_guard(
        ast.unparse(tree),
        function_name,
        ast.unparse(call),
        normalized_branch.lineno,
        normalized_branch.col_offset,
    )
