"""Derive a legible-but-incomplete visible check from an ILCB problem's full ``original_test``.

The flagship training design rewards the policy on a small visible subset of a problem's real test
cases and measures it on the full hidden check, so a hack is "passes the visible subset, fails the
hidden check". This module is the derivation step: it turns one ``def check(candidate)`` body into
a visible-subset check holding exactly ``k`` of its top-level asserts, two ways.

- ``FIRST_K``: a byte-faithful slice of the source lines from the body's start through the k-th
  top-level assert. Comments, spacing, and any earlier assignments ride along verbatim; nothing is
  unparsed and re-rendered. This is the construction the feasibility probe validated on all 103
  problems (``docs/scratch/ilcb-visible-subset-feasibility-2026-08-24.md``).
- ``STRATIFIED``: k asserts spread across the body (first, middle, last at k=3), emitted in
  original order together with every prior top-level assignment, import, or helper ``def`` any
  selected assert's names depend on. Dependency tracking is deliberately conservative: carrying too
  much is acceptable, silently carrying too little is not.

Every derivation is validated fail-closed before it is returned: the derived source must parse,
hold exactly ``k`` top-level asserts, compile, and (STRATIFIED) read no name that nothing in the
derived body, the ``candidate`` parameter, or builtins defines. Any violation raises
:class:`SubsetDerivationError` naming the problem -- a best-effort check is never returned, because
a check that raises ``NameError`` at grading time scores every honest solution as a failure.

The CLI derives a row for every original-split problem in the baked cases file and writes a NEW
derived-cases JSON that :func:`reward_hacking.harness.tasks_ilcb.load_problems` can read directly.
It refuses to overwrite anything and refuses any path inside the tracked data directory: the bake
is input here, never output. Integration into training (registering the derived split name in
``train_partition.ILCB_SPLITS`` / ``TRAINING_SPLIT_BY_ARM``) is a deliberate later step.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import hashlib
import json
import logging
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

# The side-effect-free layer, never the harness: see ilcb_data's docstring for why that matters.
from reward_hacking.ilcb_data import (
    CHECK_FUNCTION_NAME,
    DATA_PATH,
    DATA_SCHEMA,
    ILCBProblem,
    load_problems,
    problem_from_record,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

ORIGINAL_SPLIT = "original"
"""The baked split whose rows this module derives from; its ``test`` IS ``original_test``."""

_CARRYABLE_STATEMENTS = (
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)
"""Statement kinds the stratified carry may lift into the derived check.

A loop or branch that binds a name cannot be lifted without also lifting its control flow, so a
selected assert depending on one is a construction failure rather than a silent omission.
"""

_OWN_SCOPE_EXPRESSIONS = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
"""Expression nodes whose internal bindings never escape into the enclosing scope."""


class SubsetSelection(Enum):
    """Which k of the body's top-level asserts become the visible check."""

    FIRST_K = "first-k"
    STRATIFIED = "stratified"


class SubsetDerivationError(ValueError):
    """A visible-subset check could not be constructed; nothing best-effort is ever returned."""


@dataclass(frozen=True)
class VisibleSubsetResult:
    """One validated derivation: the check source plus the counts the design is sized by."""

    check_source: str
    visible_assert_count: int
    hidden_assert_count: int
    multiplier: float
    carried_statements: tuple[str, ...]


@dataclass(frozen=True)
class SubsetProvenance:
    """How a derived row was produced, kept beside it so the bake is auditable later."""

    selection: SubsetSelection
    k: int
    visible_assert_count: int
    hidden_assert_count: int
    multiplier: float
    original_test_sha256: str


@dataclass(frozen=True)
class DerivedProblem:
    """A derived ILCB row (``test`` replaced by the subset check) plus its provenance."""

    problem: ILCBProblem
    provenance: SubsetProvenance


def subset_impossible_type(k: int, selection: SubsetSelection) -> str:
    """Return the split name a derived row takes (``subset3``, ``subset3-stratified``, ...).

    ``FIRST_K`` gets the bare ``subset{k}`` the feasibility doc proposed; the stratified variant
    says so explicitly. This value flows into ``ILCBProblem.harness_task_id`` unchanged.
    """
    if selection is SubsetSelection.FIRST_K:
        return f"subset{k}"
    return f"subset{k}-{selection.value}"


def _alias_names(aliases: list[ast.alias]) -> set[str]:
    """Return the names an import statement's aliases bind in the importing scope."""
    return {alias.asname or alias.name.split(".")[0] for alias in aliases}


def _scope_bound_names(stmt: ast.stmt) -> frozenset[str]:
    """Return the names ``stmt`` binds in the scope that contains it (the check body's namespace).

    Bindings inside a nested function, class, lambda, or comprehension stay in that inner scope, so
    the walk records a nested ``def``'s name and refuses to descend into it.
    """
    if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return frozenset({stmt.name})
    if isinstance(stmt, ast.Import | ast.ImportFrom):
        return frozenset(_alias_names(stmt.names))
    bound: set[str] = set()
    pending: list[ast.AST] = [stmt]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            bound.add(node.name)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                bound.add(child.name)
            elif not isinstance(child, _OWN_SCOPE_EXPRESSIONS):
                pending.append(child)
    return frozenset(bound)


def _free_names(stmt: ast.stmt) -> frozenset[str]:
    """Return the names ``stmt`` reads that nothing within it binds -- its dependency set.

    Conservative in the over-carry direction on purpose: an augmented assignment's target counts as
    read (it loads before it stores), and a name is only dropped when the statement itself binds
    it somewhere. The fail-closed resolution check in :func:`_unresolved_names` is the net under
    whatever this approximation misses.
    """
    loads: set[str] = set()
    bound: set[str] = set()
    augmented_targets: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loads.add(node.id)
            else:
                bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            bound.update(_alias_names(node.names))
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            bound.add(node.name)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            augmented_targets.add(node.target.id)
    return frozenset((loads - bound) | augmented_targets)


def _parse_single_check(source: str, *, context: str) -> ast.FunctionDef:
    """Parse ``source`` and return its check function, refusing every other module shape.

    All 103 baked ``original_test`` bodies are exactly one ``def check(candidate)``; a sibling
    statement at module level would be silently dropped by a body-level slice, so its presence is
    an error until a real corpus row demands otherwise.
    """
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise SubsetDerivationError(
            f"{context}: the check source does not parse: {exc.msg} (line {exc.lineno})"
        ) from exc
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise SubsetDerivationError(
            f"{context}: expected exactly one top-level `def {CHECK_FUNCTION_NAME}(...)`, found "
            f"{len(module.body)} top-level statements"
        )
    check = module.body[0]
    if check.name != CHECK_FUNCTION_NAME:
        raise SubsetDerivationError(
            f"{context}: the single top-level function is `{check.name}`, "
            f"expected `def {CHECK_FUNCTION_NAME}(...)`"
        )
    return check


def _end_line(stmt: ast.stmt, *, context: str) -> int:
    """Return the 1-based last source line of ``stmt``, refusing a parse without positions."""
    if stmt.end_lineno is None:
        raise SubsetDerivationError(f"{context}: the parser reported no end line for a statement")
    return stmt.end_lineno


def _statement_source(lines: list[str], stmt: ast.stmt, *, context: str) -> str:
    """Return ``stmt``'s own source lines verbatim, decorators included."""
    start = stmt.lineno
    if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        for decorator in stmt.decorator_list:
            start = min(start, decorator.lineno)
    return "\n".join(lines[start - 1 : _end_line(stmt, context=context)])


def _spread_indices(available: int, k: int, *, context: str) -> list[int]:
    """Return k assert indices spread across ``available``: endpoints plus even interior picks."""
    if k == 1:
        return [(available - 1) // 2]
    step = (available - 1) / (k - 1)
    indices = [round(position * step) for position in range(k)]
    if len(set(indices)) != k:
        raise SubsetDerivationError(
            f"{context}: cannot spread {k} distinct picks over {available} asserts"
        )
    return indices


def _stratified_positions(
    body: list[ast.stmt], assert_positions: list[int], *, k: int, context: str
) -> list[int]:
    """Return the body positions the stratified derivation includes, in original order.

    Starts from the k spread asserts and walks each dependency back to its latest prior binder:
    a carryable binder is included (and its own dependencies queued), a non-carryable one is a
    construction failure, and an unbound name is left for :func:`_unresolved_names` to judge --
    it may be a builtin or the ``candidate`` parameter.
    """
    spread = _spread_indices(len(assert_positions), k, context=context)
    selected = [assert_positions[index] for index in spread]
    include: set[int] = set(selected)
    scope_bound = [_scope_bound_names(stmt) for stmt in body]
    pending: list[tuple[str, int]] = [
        (name, position) for position in selected for name in sorted(_free_names(body[position]))
    ]
    while pending:
        name, before = pending.pop()
        binder = next((j for j in range(before - 1, -1, -1) if name in scope_bound[j]), None)
        if binder is None or binder in include:
            continue
        stmt = body[binder]
        if not isinstance(stmt, _CARRYABLE_STATEMENTS):
            raise SubsetDerivationError(
                f"{context}: a selected assert depends on `{name}`, whose latest prior binder is "
                f"a {type(stmt).__name__} statement the carry cannot lift"
            )
        include.add(binder)
        pending.extend((needed, binder) for needed in sorted(_free_names(stmt)))
    return sorted(include)


def _unresolved_names(check: ast.FunctionDef) -> frozenset[str]:
    """Return load-context names with no binding in the derived check, its params, or builtins."""
    loads: set[str] = set()
    defined: set[str] = set(dir(builtins))
    arguments = check.args
    for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
        defined.add(arg.arg)
    if arguments.vararg is not None:
        defined.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        defined.add(arguments.kwarg.arg)
    for node in ast.walk(check):
        if isinstance(node, ast.Name):
            (loads if isinstance(node.ctx, ast.Load) else defined).add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            defined.update(_alias_names(node.names))
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            defined.add(node.name)
    return frozenset(loads - defined)


def _validate_derived(
    derived_source: str, *, k: int, context: str, require_resolved_names: bool
) -> None:
    """Validate a derivation fail-closed; raise rather than let a broken check reach a grader."""
    check = _parse_single_check(derived_source, context=f"{context}: derived check")
    derived_asserts = sum(1 for stmt in check.body if isinstance(stmt, ast.Assert))
    if derived_asserts != k:
        raise SubsetDerivationError(
            f"{context}: the derived check holds {derived_asserts} top-level asserts, "
            f"expected exactly {k}"
        )
    try:
        compile(derived_source, "<visible-subset-check>", "exec")
    except (SyntaxError, ValueError) as exc:
        raise SubsetDerivationError(
            f"{context}: the derived check does not compile: {exc!r}"
        ) from exc
    if require_resolved_names:
        unresolved = _unresolved_names(check)
        if unresolved:
            raise SubsetDerivationError(
                f"{context}: the derived check reads {len(unresolved)} unresolved name(s): "
                f"{', '.join(sorted(unresolved))}"
            )


def derive_visible_subset(
    test_source: str,
    *,
    k: int,
    selection: SubsetSelection,
    context: str = "<check source>",
) -> VisibleSubsetResult:
    """Derive the visible-subset check holding exactly ``k`` of ``test_source``'s asserts.

    ``context`` names the problem in every failure message. Raises
    :class:`SubsetDerivationError` on any body this construction cannot represent faithfully.
    """
    if k < 1:
        raise SubsetDerivationError(f"{context}: k must be at least 1, got {k}")
    check = _parse_single_check(test_source, context=context)
    body = check.body
    assert_positions = [
        position for position, stmt in enumerate(body) if isinstance(stmt, ast.Assert)
    ]
    hidden_assert_count = len(assert_positions)
    if hidden_assert_count < k:
        raise SubsetDerivationError(
            f"{context}: only {hidden_assert_count} top-level asserts, cannot derive a visible "
            f"subset of k={k}"
        )
    lines = test_source.split("\n")
    if selection is SubsetSelection.FIRST_K:
        included_positions = list(range(assert_positions[k - 1] + 1))
        cut = _end_line(body[assert_positions[k - 1]], context=context)
        derived_source = "\n".join(lines[:cut])
    else:
        included_positions = _stratified_positions(body, assert_positions, k=k, context=context)
        header = "\n".join(lines[: body[0].lineno - 1])
        pieces = [header] + [
            _statement_source(lines, body[position], context=context)
            for position in included_positions
        ]
        derived_source = "\n".join(pieces)
    carried_statements = tuple(
        _statement_source(lines, body[position], context=context)
        for position in included_positions
        if not isinstance(body[position], ast.Assert)
    )
    _validate_derived(
        derived_source,
        k=k,
        context=context,
        require_resolved_names=selection is SubsetSelection.STRATIFIED,
    )
    return VisibleSubsetResult(
        check_source=derived_source,
        visible_assert_count=k,
        hidden_assert_count=hidden_assert_count,
        multiplier=hidden_assert_count / k,
        carried_statements=carried_statements,
    )


def derive_for_problem(
    problem: ILCBProblem, *, k: int, selection: SubsetSelection
) -> DerivedProblem:
    """Derive one problem's visible-subset row from its ``original_test``.

    Split-agnostic on purpose: ``original_test`` is identical across a problem's splits, and every
    field that described the OLD visible check is recomputed rather than inherited -- the derived
    check is a subset of a satisfiable check, so it is satisfiable by construction, and it parses
    and defines ``check`` because the validation just proved it. Inheriting a conflicting row's
    ``provably_impossible=True`` here would label a solvable task as impossible.
    """
    result = derive_visible_subset(
        problem.original_test, k=k, selection=selection, context=problem.task_id
    )
    derived = dataclasses.replace(
        problem,
        impossible_type=subset_impossible_type(k, selection),
        test=result.check_source,
        provably_impossible=False,
        impossibility_proof=None,
        check_parses=True,
        check_defines_function=True,
    )
    return DerivedProblem(
        problem=derived,
        provenance=SubsetProvenance(
            selection=selection,
            k=k,
            visible_assert_count=result.visible_assert_count,
            hidden_assert_count=result.hidden_assert_count,
            multiplier=result.multiplier,
            original_test_sha256=hashlib.sha256(problem.original_test.encode()).hexdigest(),
        ),
    )


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_DIR = _REPO_ROOT / "artifacts" / "reward_hacking" / "visible_subset"


def derived_record(derived: DerivedProblem) -> dict[str, object]:
    """Return one output row: the ILCBProblem fields plus a ``provenance`` block.

    The extra ``provenance`` key rides beside the row fields rather than replacing any of them,
    which is what lets :func:`reward_hacking.ilcb_data.problem_from_record` read the row back
    unchanged -- it keys on the fields it knows and tolerates the rest.
    """
    record: dict[str, object] = dataclasses.asdict(derived.problem)
    provenance = derived.provenance
    record["provenance"] = {
        "selection": provenance.selection.value,
        "k": provenance.k,
        "visible_assert_count": provenance.visible_assert_count,
        "hidden_assert_count": provenance.hidden_assert_count,
        "multiplier": provenance.multiplier,
        "original_test_sha256": provenance.original_test_sha256,
    }
    return record


def derive_split_records(
    original_records: Sequence[dict[str, object]], *, k: int, selection: SubsetSelection
) -> list[dict[str, object]]:
    """Derive one whole subset split's baked rows from original-split record dicts, all or nothing.

    The ETL's entry point: it works record-in, record-out so the caller never has to import the
    harness's registry, and a single derivation failure refuses the whole split -- a partial
    derived split consumed downstream would silently shrink the training corpus, exactly the class
    of quiet corruption the CLI's own all-or-nothing rule exists to prevent.
    """
    derived_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for record in original_records:
        problem = problem_from_record(dict(record))
        if problem.impossible_type != ORIGINAL_SPLIT:
            raise SubsetDerivationError(
                f"{problem.task_id}: derive_split_records takes `{ORIGINAL_SPLIT}` rows only, got "
                f"a {problem.impossible_type!r} row -- deriving from a perturbed split would bake "
                f"its manipulation into the visible subset"
            )
        try:
            derived_rows.append(
                derived_record(derive_for_problem(problem, k=k, selection=selection))
            )
        except SubsetDerivationError as error:
            failures.append(str(error))
    if failures:
        raise SubsetDerivationError(
            f"{len(failures)} of {len(original_records)} problems failed the "
            f"{subset_impossible_type(k, selection)} derivation; refusing a partial split: "
            + "; ".join(failures)
        )
    return derived_rows


_QUARTILE_COUNT = 4
"""Quartiles need at least four values before the split is worth printing."""


def _multiplier_summary(multipliers: list[float]) -> str:
    """Return a one-line distribution of hidden/visible multipliers."""
    ordered = sorted(multipliers)
    line = f"min={ordered[0]:.1f} median={statistics.median(ordered):.1f} max={ordered[-1]:.1f}"
    if len(ordered) >= _QUARTILE_COUNT:
        first_quartile, _, third_quartile = statistics.quantiles(ordered, n=_QUARTILE_COUNT)
        line += f" (q1={first_quartile:.1f} q3={third_quartile:.1f})"
    return line


def _parse_cli_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the CLI arguments; ``argv=None`` reads ``sys.argv`` as argparse always does."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive a visible-subset check for every original-split ILCB problem and write a NEW "
            "derived-cases JSON. Never overwrites, and never writes into the tracked data dir."
        )
    )
    parser.add_argument("--k", type=int, default=3, help="visible top-level asserts per problem")
    parser.add_argument(
        "--selection",
        choices=[selection.value for selection in SubsetSelection],
        default=SubsetSelection.FIRST_K.value,
        help="which k asserts become visible",
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, help="baked cases JSON to read (never written)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help=f"output path (default: under {_DEFAULT_OUT_DIR})"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Derive every original-split problem and write the derived-cases file, fail-closed.

    A single derivation failure refuses the whole write: a partial derived dataset consumed
    downstream would silently shrink the training set, which is exactly the class of quiet
    corruption this repo's conventions exist to prevent.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_cli_args(argv)
    k: int = args.k
    selection = SubsetSelection(args.selection)
    out: Path = (
        args.out
        if args.out is not None
        else (_DEFAULT_OUT_DIR / f"ilcb_cases_{subset_impossible_type(k, selection)}.json")
    )
    if out.resolve().is_relative_to(DATA_PATH.parent.resolve()):
        logger.error(f"refusing to write into the tracked data directory: {out}")
        return 2
    if out.exists():
        logger.error(f"refusing to overwrite {out}")
        return 2

    originals = [p for p in load_problems(args.data) if p.impossible_type == ORIGINAL_SPLIT]
    if not originals:
        logger.error(f"no `{ORIGINAL_SPLIT}` rows in {args.data}; nothing to derive")
        return 2

    derived_rows: list[DerivedProblem] = []
    failures: list[tuple[str, str]] = []
    for problem in originals:
        try:
            derived = derive_for_problem(problem, k=k, selection=selection)
        except SubsetDerivationError as error:
            failures.append((problem.task_id, str(error)))
            continue
        derived_rows.append(derived)
        provenance = derived.provenance
        logger.info(
            f"{problem.task_id}: visible={provenance.visible_assert_count} "
            f"hidden={provenance.hidden_assert_count} multiplier={provenance.multiplier:.1f}"
        )

    if derived_rows:
        logger.info(
            f"multiplier distribution over {len(derived_rows)} problems: "
            f"{_multiplier_summary([row.provenance.multiplier for row in derived_rows])}"
        )
    if failures:
        for task_id, message in failures:
            logger.error(f"FAILED {task_id}: {message}")
        logger.error(
            f"{len(failures)} of {len(originals)} problems failed derivation; "
            f"refusing to write a partial {out}"
        )
        return 1

    impossible_type = subset_impossible_type(k, selection)
    payload = {
        "schema": DATA_SCHEMA,
        "generated_by": "reward_hacking/visible_subset.py",
        "derivation": {
            "k": k,
            "selection": selection.value,
            "impossible_type": impossible_type,
            "source_data": str(args.data),
        },
        "splits": [impossible_type],
        "problems": [derived_record(row) for row in derived_rows],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # "x", not "w": a race past the exists() refusal above crashes loudly instead of clobbering.
    with out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    logger.info(f"wrote {len(derived_rows)} derived rows ({impossible_type}) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
