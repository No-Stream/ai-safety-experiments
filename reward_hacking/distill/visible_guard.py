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
from reward_hacking.distill.call_history import (
    SAFE_STDLIB_IMPORTS as _PURE_STDLIB_IMPORTS,
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
    "TypeError",
    "ValueError",
    "zip",
)
_SAFE_METHODS = frozenset(
    {
        "add",
        "append",
        "capitalize",
        "casefold",
        "center",
        "clear",
        "copy",
        "count",
        "difference",
        "difference_update",
        "discard",
        "endswith",
        "expandtabs",
        "extend",
        "find",
        "from_bytes",
        "format",
        "format_map",
        "get",
        "index",
        "insert",
        "intersection",
        "intersection_update",
        "isdisjoint",
        "issubset",
        "issuperset",
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
        "symmetric_difference",
        "symmetric_difference_update",
        "title",
        "union",
        "update",
        "upper",
        "values",
        "zfill",
    }
)


def _sandbox_import_bindings(tree: ast.Module) -> dict[str, str] | None:
    imported_modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                binding = alias.asname or module.split(".")[0]
                if module not in _PURE_STDLIB_IMPORTS or binding.startswith("__"):
                    return None
                imported_modules[binding] = module
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module not in _PURE_STDLIB_IMPORTS:
                return None
            allowed_names = _PURE_STDLIB_IMPORTS[node.module]
            if any(
                alias.name == "*"
                or alias.name not in allowed_names
                or (alias.asname is not None and alias.asname.startswith("__"))
                for alias in node.names
            ):
                return None
            for alias in node.names:
                imported_modules[alias.asname or alias.name] = ""
    return imported_modules


def _sandbox_attribute_supported(  # noqa: PLR0911 - explicit capability allowlist
    node: ast.Attribute, imported_modules: dict[str, str], helper_attributes: set[str]
) -> bool:
    if node.attr.startswith("__"):
        return False
    if isinstance(node.ctx, (ast.Store, ast.Del)):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and "self" not in imported_modules
        )
    if node.attr in helper_attributes:
        return True
    if (
        node.attr == "readline"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "stdin"
        and isinstance(node.value.value, ast.Name)
        and imported_modules.get(node.value.value.id) == "sys"
    ):
        return True
    if node.attr in _SAFE_METHODS:
        return True
    if not isinstance(node.value, ast.Name):
        return False
    module = imported_modules.get(node.value.id)
    return module in _PURE_STDLIB_IMPORTS and node.attr in _PURE_STDLIB_IMPORTS[module]


def _sandbox_private_bindings_supported(node: ast.AST) -> bool:
    binding_names: tuple[str, ...] = ()
    if isinstance(node, ast.FunctionDef):
        binding_names = () if node.name == "__init__" else (node.name,)
    elif isinstance(node, ast.arg):
        binding_names = (node.arg,)
    elif isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)):
        binding_names = () if node.name is None else (node.name,)
    elif isinstance(node, ast.MatchMapping):
        binding_names = () if node.rest is None else (node.rest,)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        binding_names = tuple(node.names)
    return not any(name.startswith("__") for name in binding_names)


def _sandbox_node_supported(
    node: ast.AST, imported_modules: dict[str, str], helper_attributes: set[str]
) -> bool:
    if isinstance(node, ast.AsyncFunctionDef):
        return False
    if isinstance(node, ast.ExceptHandler):
        return node.type is not None and not any(
            isinstance(child, ast.Name)
            and child.id
            in {"BaseException", "Exception", "MemoryError", "RecursionError", "RuntimeError"}
            for child in ast.walk(node.type)
        )
    if isinstance(node, ast.Attribute):
        return _sandbox_attribute_supported(node, imported_modules, helper_attributes)
    if isinstance(node, ast.Name):
        return not node.id.startswith("__")
    return _sandbox_private_bindings_supported(node)


def _sandbox_helper_attributes(  # noqa: C901 - explicit declaration safety checks
    tree: ast.Module,
) -> set[str] | None:
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.bases or node.keywords or node.decorator_list or node.name.startswith("__"):
            return None
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef):
                if statement.decorator_list or (
                    statement.name.startswith("__") and statement.name != "__init__"
                ):
                    return None
                attributes.add(statement.name)
            elif isinstance(statement, ast.Assign):
                if any(
                    not isinstance(target, ast.Name) or target.id.startswith("__")
                    for target in statement.targets
                ):
                    return None
                try:
                    ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    return None
                attributes.update(
                    target.id for target in statement.targets if isinstance(target, ast.Name)
                )
            elif not isinstance(statement, ast.Pass) and not _docstring_statement(statement):
                return None
        attributes.update(
            child.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "self"
            and not child.attr.startswith("__")
        )
    return attributes


def _sandbox_source_supported(tree: ast.Module) -> bool:
    imported_modules = _sandbox_import_bindings(tree)
    helper_attributes = _sandbox_helper_attributes(tree)
    method_stores = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef)
        and (*method.args.posonlyargs, *method.args.args)
        and (*method.args.posonlyargs, *method.args.args)[0].arg == "self"
        for child in ast.walk(method)
        if isinstance(child, ast.Attribute) and not isinstance(child.ctx, ast.Load)
    }
    return (
        imported_modules is not None
        and helper_attributes is not None
        and all(
            id(node) in method_stores
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load)
        )
        and all(
            _sandbox_node_supported(node, imported_modules, helper_attributes)
            for node in ast.walk(tree)
        )
    )


def _sandbox_entry_supported(tree: ast.Module, name: str) -> bool:
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    ]
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        return False
    if definitions[0] not in tree.body:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            return False
        if isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == name:
            return False
    return True


_SANDBOX_DRIVER = """import ast
import builtins
import functools
import heapq
import itertools
import json
import operator
import re
import string
import sys
import types
from pathlib import Path

module_cache = {}
trusted_decorators = []
original_recursion_limit = sys.getrecursionlimit()
guard_rejected = False

def reject_guard(message, error_type=TypeError):
    global guard_rejected
    guard_rejected = True
    raise error_type(message)

def check_guard_rejection():
    if guard_rejected:
        raise RuntimeError('candidate caught an unsupported comparison capability')

def deterministic_key(value):
    if type(value) in (type(None), bool, int, str, bytes):
        return value
    if type(value) in (float, complex) and value == value:
        return value
    if type(value) in (tuple, frozenset):
        for child in value:
            deterministic_key(child)
        return value
    reject_guard('unordered collection requires deterministic literal keys')

def deterministic_keys(values):
    for value in values:
        yield deterministic_key(value)

def guard_unordered_result(value):
    if type(value) in (set, frozenset):
        for child in value:
            deterministic_key(child)
    return value

def guard_unordered_operand(value):
    if type(value) in (set, frozenset, type({}.keys()), type({}.items())):
        for child in value:
            deterministic_key(child)

def guard_inplace(operation, left, right):
    guard_unordered_operand(left)
    guard_unordered_operand(right)
    functions = {'BitOr': operator.ior, 'BitAnd': operator.iand, 'BitXor': operator.ixor, 'Sub': operator.isub}
    return guard_unordered_result(functions[operation](left, right))

def unavailable_identity_hash(instance):
    reject_guard('helper identity hashing is unsupported answer evidence')


def unavailable_identity_representation(instance):
    reject_guard('helper identity representation is unsupported answer evidence')

class CandidateType(type):
    __hash__ = unavailable_identity_hash

    def __new__(metaclass, name, bases, namespace):
        namespace['__hash__'] = unavailable_identity_hash
        namespace['__repr__'] = unavailable_identity_representation
        namespace['__str__'] = unavailable_identity_representation
        return super().__new__(metaclass, name, bases, namespace)

def guard_representation(value):
    if not literal(value):
        reject_guard('nonliteral identity representation is unsupported answer evidence')
    return value

def guarded_modulo(left, right):
    if type(left) in (str, bytes):
        guard_representation(right)
    return left % right

def guard_decorator(decorator):
    if not any(decorator is trusted for trusted in (functools.cache, functools.lru_cache, *trusted_decorators)):
        reject_guard('candidate decorator requires a trusted cache factory')
    return decorator

def bind_native_method(value, validate):
    if (type(value) is types.BuiltinMethodType and value.__self__ is not None
            and not isinstance(value.__self__, (types.ModuleType, type))):
        return value
    return validate(value)

class GuardInstanceWrites(ast.NodeTransformer):
    def visit_Set(self, node):
        self.generic_visit(node)
        for index, element in enumerate(node.elts):
            if isinstance(element, ast.Starred):
                element.value = ast.Call(func=ast.Name('__guard_keys__', ast.Load()),
                                         args=[element.value], keywords=[])
            else:
                node.elts[index] = ast.Call(func=ast.Name('__guard_key__', ast.Load()),
                                           args=[element], keywords=[])
        return self.checked(node)

    def visit_SetComp(self, node):
        self.generic_visit(node)
        node.elt = ast.Call(func=ast.Name('__guard_key__', ast.Load()), args=[node.elt], keywords=[])
        return self.checked(node)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        return self.checked(node) if isinstance(node.ctx, ast.Load) else node

    def visit_Assign(self, node):
        if all(isinstance(target, ast.Name) for target in node.targets) and isinstance(node.value, ast.Attribute):
            node.value = ast.copy_location(ast.Call(
                func=ast.Name('__bind_native_method__', ast.Load()),
                args=[self.reference(node.value)], keywords=[],
            ), node.value)
            return node
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.Sub)):
            if not isinstance(node.target, ast.Name):
                reject_guard('augmented unordered operation requires a simple target')
            return ast.copy_location(ast.Assign(
                targets=[node.target], value=ast.Call(func=ast.Name('__guard_inplace__', ast.Load()),
                    args=[ast.Constant(type(node.op).__name__), ast.Name(node.target.id, ast.Load()), node.value],
                    keywords=[])), node)
        if isinstance(node.op, ast.Mod):
            if not isinstance(node.target, ast.Name):
                reject_guard('augmented modulo requires a simple target')
            return ast.copy_location(ast.Assign(
                targets=[node.target],
                value=ast.Call(func=ast.Name('__guard_modulo__', ast.Load()),
                               args=[ast.Name(node.target.id, ast.Load()), node.value], keywords=[]),
            ), node)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.BitXor, ast.Sub)):
            return ast.copy_location(ast.Call(func=ast.Name('__guard_unordered__', ast.Load()),
                                              args=[node], keywords=[]), node)
        if isinstance(node.op, ast.Mod):
            return ast.copy_location(ast.Call(
                func=ast.Name('__guard_modulo__', ast.Load()),
                args=[node.left, node.right], keywords=[],
            ), node)
        return node

    def checked(self, node, allow_class=False):
        return ast.copy_location(ast.Call(
            func=ast.Name('__guard_value__', ast.Load()),
            args=[node, ast.Constant(allow_class)], keywords=[],
        ), node)

    def reference(self, node, allow_class=False):
        visited = self.visit(node)
        if (isinstance(visited, ast.Call) and isinstance(visited.func, ast.Name)
                and visited.func.id == '__guard_value__'):
            return self.checked(visited.args[0], True) if allow_class else visited.args[0]
        return visited

    def visit_Name(self, node):
        return self.checked(node) if isinstance(node.ctx, ast.Load) else node

    def visit_Lambda(self, node):
        self.generic_visit(node)
        return self.checked(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = [ast.Call(
            func=ast.Name('__guard_decorator__', ast.Load()), args=[decorator], keywords=[],
        ) for decorator in node.decorator_list]
        register = ast.Expr(ast.Call(
            func=ast.Attribute(ast.Name('__candidate_functions__', ast.Load()), 'append', ast.Load()),
            args=[ast.Name(node.name, ast.Load())], keywords=[],
        ))
        return [node, ast.copy_location(register, node)]

    def callback_argument(self, node):
        if isinstance(node, ast.Lambda):
            self.generic_visit(node)
            return ast.Call(
                func=ast.Name('__inline_callback__', ast.Load()), args=[node], keywords=[],
            )
        return self.visit(node)

    def visit_Call(self, node):
        invocation = ast.Call(
            func=ast.Name('__invoke__', ast.Load()),
            args=[self.reference(node.func), *(self.callback_argument(arg) for arg in node.args)],
            keywords=[ast.keyword(keyword.arg, self.callback_argument(keyword.value))
                      for keyword in node.keywords],
        )
        return self.checked(ast.copy_location(invocation, node))

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        node.keywords.append(ast.keyword('metaclass', ast.Name('__candidate_type__', ast.Load())))
        register = ast.Expr(ast.Call(
            func=ast.Attribute(ast.Name('__candidate_classes__', ast.Load()), 'append', ast.Load()),
            args=[ast.Name(node.name, ast.Load())], keywords=[],
        ))
        return [node, ast.copy_location(register, node)]

    def visit_Attribute(self, node):
        node.value = self.reference(node.value, allow_class=True)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            node.value = ast.Call(
                func=ast.Name('__guard_instance__', ast.Load()), args=[node.value], keywords=[],
            )
            return node
        return self.checked(node)

    def visit_FormattedValue(self, node):
        self.generic_visit(node)
        node.value = ast.Call(
            func=ast.Name('__guard_representation__', ast.Load()), args=[node.value], keywords=[],
        )
        return node

def instance_guard(classes):
    def validate(instance):
        if not any(type(instance) is candidate_class for candidate_class in classes):
            reject_guard('attribute mutation requires a candidate-created helper instance')
        return instance
    return validate

class InlineCallback:
    def __init__(self, function):
        self.function = function

def callable_boundary(namespace, functions):
    def validate(value, allow_class=False):
        if type(value) in (type(None), bool, int, float, complex, str, bytes):
            return value
        if isinstance(value, CandidateType) and not allow_class:
            reject_guard('candidate class values require a direct call or attribute receiver')
        if (type(value) is types.BuiltinMethodType and value.__self__ is not None
                and not isinstance(value.__self__, (types.ModuleType, type))):
            reject_guard('instance-bound builtin method values require a direct call')
        function = value.__func__ if type(value) is types.MethodType else value
        if any(function is candidate_function for candidate_function in functions) or (
            type(function) is types.FunctionType and function.__globals__ is namespace
        ):
            reject_guard('candidate callable values require a direct call')
        return guard_unordered_result(value)
    return validate

def guard_format_template(template):
    for _, field, spec, _ in string.Formatter().parse(template):
        if field is not None:
            indexing = False
            for character in field:
                if character == '[':
                    indexing = True
                elif character == ']':
                    indexing = False
                elif character == '.' and not indexing:
                    reject_guard('native formatting cannot traverse attributes')
            guard_format_template(spec)

def guard_native_format(function, arguments, keywords):
    template = None
    values = arguments
    if type(function) is types.BuiltinMethodType and type(function.__self__) is str:
        if function.__name__ in ('format', 'format_map'):
            template = function.__self__
    elif (type(function) is types.MethodDescriptorType and function.__objclass__ is str
          and function.__name__ in ('format', 'format_map') and arguments):
        template = arguments[0]
        values = arguments[1:]
    if type(template) is str:
        guard_format_template(template)
        guard_representation(tuple(values))
        guard_representation(keywords)

CALLBACK_POSITIONS = {
    map: 0, filter: 0, functools.reduce: 0, functools.partial: 0, functools.cmp_to_key: 0,
    itertools.starmap: 0, itertools.dropwhile: 0, itertools.filterfalse: 0, itertools.takewhile: 0,
    itertools.groupby: 1, itertools.accumulate: 1, heapq.nlargest: 2, heapq.nsmallest: 2,
    re.sub: 1, re.subn: 1,
}
CALLBACK_KEY_CONSUMERS = (sorted, min, max, heapq.merge, heapq.nlargest, heapq.nsmallest, itertools.groupby)
UNORDERED_OPERATORS = (operator.or_, operator.and_, operator.xor, operator.sub,
                       operator.ior, operator.iand, operator.ixor, operator.isub)
SET_SINGLE_KEYS = ('add', 'discard', 'remove')
SET_KEY_ITERABLES = ('update', 'difference_update', 'intersection_update', 'symmetric_difference_update',
                     'union', 'difference', 'intersection', 'symmetric_difference',
                     'isdisjoint', 'issubset', 'issuperset')

def checked_callback(function):
    def invoke(*args, **kwargs):
        return consuming_call(function, *args, **kwargs)
    candidate_functions.append(invoke)
    return invoke

def consuming_call(function, *args, **kwargs):
    if any(function is operation for operation in (operator.mod, operator.imod)):
        return guarded_modulo(*args, **kwargs)
    if args and any(function is operation for operation in (builtins.str, builtins.repr, builtins.format)):
        guard_representation(args[0])
    arguments = list(args)
    for index, argument in enumerate(arguments):
        if type(argument) is InlineCallback:
            if function is not functools.reduce or index != 0:
                reject_guard('inline candidate callbacks require a trusted consuming callable')
            arguments[index] = argument.function
    for key, argument in kwargs.items():
        if type(argument) is InlineCallback:
            if key != 'key' or not any(function is consumer for consumer in (sorted, min, max)):
                reject_guard('inline candidate callbacks require a trusted consuming callable')
            kwargs[key] = argument.function
    for consumer, index in CALLBACK_POSITIONS.items():
        if function is consumer and len(arguments) > index and callable(arguments[index]):
            arguments[index] = checked_callback(arguments[index])
            break
    if any(function is consumer for consumer in CALLBACK_KEY_CONSUMERS):
        if callable(kwargs.get('key')):
            kwargs['key'] = checked_callback(kwargs['key'])
    if function is itertools.accumulate and callable(kwargs.get('func')):
        kwargs['func'] = checked_callback(kwargs['func'])
    if function is functools.lru_cache:
        if arguments and callable(arguments[0]):
            arguments[0] = checked_callback(arguments[0])
        result = function(*arguments, **kwargs)
        trusted_decorators.append(result)
        return result
    if (type(function) is types.FunctionType
            and any(function is factory for factory in trusted_decorators)
            and arguments and callable(arguments[0])):
        arguments[0] = checked_callback(arguments[0])
    guard_native_format(function, arguments, kwargs)
    if any(function is operation for operation in UNORDERED_OPERATORS):
        for argument in arguments:
            guard_unordered_operand(argument)
    if function is set or function is frozenset:
        if len(arguments) == 1:
            arguments[0] = deterministic_keys(arguments[0])
    receiver = None
    offset = 0
    if type(function) is types.BuiltinMethodType and type(function.__self__) in (set, frozenset):
        receiver = function.__self__
    elif (type(function) is types.MethodDescriptorType and function.__objclass__ in (set, frozenset)
          and arguments and type(arguments[0]) in (set, frozenset)):
        receiver = arguments[0]
        offset = 1
    if receiver is not None:
        guard_unordered_result(receiver)
        if function.__name__ in SET_SINGLE_KEYS:
            for argument in arguments[offset:]:
                deterministic_key(argument)
        elif function.__name__ in SET_KEY_ITERABLES:
            for index in range(offset, len(arguments)):
                arguments[index] = deterministic_keys(arguments[index])
    for argument in arguments:
        guard_unordered_result(argument)
    return guard_unordered_result(function(*arguments, **kwargs))

def unavailable_readline(*args, **kwargs):
    reject_guard('visible callable comparison has no stdin evidence', RuntimeError)

def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed_names = payload['imports'].get(name)
    requested_names = () if fromlist is None else fromlist
    if level != 0 or allowed_names is None or any(item not in allowed_names for item in requested_names):
        reject_guard('candidate import is outside the standard-library allowlist', ImportError)
    if name not in module_cache:
        if name == 'sys':
            module_cache[name] = types.SimpleNamespace(
                setrecursionlimit=sys.setrecursionlimit,
                stdin=types.SimpleNamespace(readline=unavailable_readline),
            )
            return module_cache[name]
        imported_module = builtins.__import__(name, globals, locals, fromlist, level)
        module_cache[name] = types.SimpleNamespace(
            **{item: getattr(imported_module, item) for item in allowed_names}
        )
    return module_cache[name]

def literal(value):
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return True
    if type(value) in (tuple, list, set):
        return all(literal(child) for child in value)
    if type(value) is dict:
        return all(literal(key) and literal(child) for key, child in value.items())
    return False

class HistoryLines(ast.NodeTransformer):
    def __init__(self, name, storage_names):
        self.name = name
        self.storage_names = storage_names

    def visit_FunctionDef(self, node):
        if node.name != self.name:
            return node
        return self.generic_visit(node)

    def recorded(self, node):
        hook = ast.Expr(ast.Call(func=ast.Name('__history_line__', ast.Load()),
                                args=[ast.Constant(node.lineno)], keywords=[]))
        return [ast.copy_location(hook, node), node]

    def visit_Return(self, node):
        return self.recorded(node)

    def assignment(self, node, targets):
        for target in targets:
            receiver = target.value if isinstance(target, ast.Subscript) else target
            if isinstance(receiver, ast.Name) and receiver.id in self.storage_names:
                return self.recorded(node)
        return node

    def visit_Assign(self, node):
        return self.assignment(node, node.targets)

    def visit_AnnAssign(self, node):
        return self.assignment(node, [node.target])

    def visit_AugAssign(self, node):
        return self.assignment(node, [node.target])

    def visit_Expr(self, node):
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id in self.storage_names):
            return self.recorded(node)
        return node

payload = json.loads(Path('payload.json').read_text())
answers = []
jobs = [(source, payload['arguments']) for source in payload['sources']]
if payload.get('history_inputs') is not None:
    jobs = [(payload['sources'][0], arguments) for arguments in payload['history_inputs']]
for source, argument_text in jobs:
    sys.setrecursionlimit(original_recursion_limit)
    module_cache = {}
    trusted_decorators = []
    guard_rejected = False
    sandbox_builtins = {name: getattr(builtins, name) for name in payload['builtins']}
    sandbox_builtins['__import__'] = restricted_import
    sandbox_builtins['__build_class__'] = builtins.__build_class__
    namespace = {'__builtins__': sandbox_builtins, '__name__': 'candidate'}
    candidate_classes = []
    namespace['__candidate_classes__'] = candidate_classes
    namespace['__guard_instance__'] = instance_guard(candidate_classes)
    namespace['__candidate_type__'] = CandidateType
    namespace['__guard_representation__'] = guard_representation
    namespace['__guard_modulo__'] = guarded_modulo
    namespace['__guard_decorator__'] = guard_decorator
    namespace['__guard_key__'] = deterministic_key
    namespace['__guard_keys__'] = deterministic_keys
    namespace['__guard_unordered__'] = guard_unordered_result
    namespace['__guard_inplace__'] = guard_inplace
    candidate_functions = []
    namespace['__candidate_functions__'] = candidate_functions
    namespace['__guard_value__'] = callable_boundary(namespace, candidate_functions)
    namespace['__bind_native_method__'] = lambda value: bind_native_method(value, namespace['__guard_value__'])
    namespace['__inline_callback__'] = InlineCallback
    namespace['__invoke__'] = consuming_call
    candidate_tree = ast.parse(source)
    executed_lines = set()
    namespace['__history_line__'] = executed_lines.add
    if payload.get('history_inputs') is not None:
        storage_names = {target.id for node in candidate_tree.body
                         if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                         for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                         if isinstance(target, ast.Name)}
        candidate_tree = HistoryLines(payload['name'], storage_names).visit(candidate_tree)
    guarded_tree = GuardInstanceWrites().visit(candidate_tree)
    exec(compile(ast.fix_missing_locations(guarded_tree), '<candidate>', 'exec'), namespace)
    check_guard_rejection()
    if payload.get('history_inputs') is None:
        arguments = ast.literal_eval(argument_text)
        answer = namespace[payload['name']](*arguments)
        check_guard_rejection()
        if not literal(answer):
            reject_guard('candidate answer is not a supported literal')
        answers.append(answer)
    else:
        history = []
        executed_lines.clear()
        initial_state = {key: repr(value) for key, value in namespace.items()
                         if not key.startswith('__') and literal(value)}
        changed_storage = set()
        for _ in range(2):
            arguments = ast.literal_eval(argument_text)
            answer = namespace[payload['name']](*arguments)
            check_guard_rejection()
            if not literal(answer):
                reject_guard('candidate answer is not a supported literal')
            history.append(ast.literal_eval(repr(answer)))
            changed_storage.update(key for key, previous in initial_state.items()
                                   if key not in namespace or not literal(namespace[key])
                                   or repr(namespace[key]) != previous)
        answers.append((history, sorted(executed_lines), sorted(changed_storage)))
        if history[0] != history[1]:
            break
print('GUARD_ANSWER=' + repr(answers))
"""


def _sandbox_answers(  # noqa: PLR0911 - explicit unsupported, transport, and output evidence boundaries
    sources: Sequence[str],
    name: str,
    arguments: list[object],
    *,
    history_inputs: Sequence[tuple[object, ...]] | None = None,
) -> tuple[list[object] | None, str | None]:
    trees = [ast.parse(source) for source in sources]
    if not all(
        _sandbox_source_supported(tree) and _sandbox_entry_supported(tree, name) for tree in trees
    ):
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
        "imports": {module: sorted(names) for module, names in _PURE_STDLIB_IMPORTS.items()},
    }
    if history_inputs is not None:
        if len(sources) != 1:
            raise ValueError("call history comparison requires one candidate source")
        import_payload["history_inputs"] = [repr(arguments) for arguments in history_inputs]
    timeout_seconds = 180 if history_inputs is not None else 5 * len(sources)
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
    expected_count = len(history_inputs) if history_inputs is not None else len(sources)
    valid_count = (
        0 < len(values) <= expected_count
        if isinstance(values, list) and history_inputs is not None
        else isinstance(values, list) and len(values) == expected_count
    )
    if not valid_count:
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
