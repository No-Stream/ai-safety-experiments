"""What the rigged grader changed, read off the check bodies, and the two authored controls beside it.

The ``oneoff`` split perturbs one asserted expected value so that the honest answer is rejected.
Everything the twin corpus and the readout pairs say about a problem hangs on knowing exactly which
line that is, what input it asserts about, what value the honest grader expects there and what value
the rigged one plants. None of that is taken from a description of the split: it is diffed out of
the two check bodies the graders inline, parsed with ``ast``, and refused whenever the shape is not
"one replaced line, same call, different literal".

Two controls are authored here from the honest problem so that the corpus can tell "a rigged value"
apart from "any one-line difference at that position" and from "a wrong literal anywhere":

* :func:`insert_line` puts one comment line directly before the perturbed assertion, at its
  indentation, so the comment-control rendering differs from the honest one at the same position.
* :func:`perturb_statement_literal` shifts one numeric literal in the statement's worked example
  (an output line where one carries a literal, else a call line), by the grader's own integer delta
  when it has one, and kept away from the planted value so the control never becomes a consistent
  rigging by coincidence.

Nothing here touches a tokenizer or a model; it is pure text and ``ast`` over registry rows.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from reward_hacking.train_grader import check_function_body

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reward_hacking.ilcb_data import ILCBProblem

CANDIDATE_NAME = "candidate"
DOCTEST_PROMPT = ">>>"
_NUMERIC_LITERAL = re.compile(r"\d+(?:\.\d+)?")


class TwinRefusalError(ValueError):
    """A problem the twin apparatus cannot carry, with the reason it cannot."""


# --------------------------------------------------------------------------------------
# Line diffs
# --------------------------------------------------------------------------------------


def _changed_opcodes(before: str, after: str) -> list[tuple[str, int, int, int, int]]:
    matcher = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines(), autojunk=False)
    return [opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"]


def line_diff_count(before: str, after: str) -> int:
    """Count the lines that differ between two texts (replaced, inserted or deleted)."""
    return sum(max(i2 - i1, j2 - j1) for _tag, i1, i2, j1, j2 in _changed_opcodes(before, after))


def replaced_line_index(before: str, after: str) -> int | None:
    """Return the index of the ONE replaced line, or None when the diff is anything else."""
    opcodes = _changed_opcodes(before, after)
    if len(opcodes) != 1:
        return None
    tag, i1, i2, j1, j2 = opcodes[0]
    if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
        return None
    return i1


def first_changed_line_index(before: str, after: str) -> int | None:
    """Return the index in ``after`` of the first inserted or replaced line, or None for a deletion."""
    for tag, _i1, _i2, j1, j2 in _changed_opcodes(before, after):
        if tag in ("insert", "replace") and j2 > j1:
            return j1
    return None


# --------------------------------------------------------------------------------------
# The perturbation facts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerturbationFacts:
    """What the rigged grader changed, read off the two check bodies rather than assumed.

    ``honest_line_index`` indexes the check body the grader inlines (``check_function_body``), so
    it is directly the line the renderings anchor on. ``call_args``, ``honest_value`` and
    ``planted_value`` are ``ast.unparse`` canonical source, which is what lets the readout-pair
    module compare values by ``ast.literal_eval`` rather than by string.
    """

    problem_id: str
    entry_point: str
    honest_line_index: int
    honest_line: str
    rigged_line: str
    call_args: tuple[str, ...]
    honest_value: str
    planted_value: str
    kind: str
    magnitude: float | None
    detectable: bool

    def to_public_dict(self) -> dict[str, object]:
        """Return the item-free half: what a sidecar may carry without quoting the grader."""
        return {
            "honest_line_index": self.honest_line_index,
            "n_call_args": len(self.call_args),
            "kind": self.kind,
            "magnitude": self.magnitude,
            "detectable": self.detectable,
        }


def _assert_covering_line(body: str, line_index: int, *, what: str) -> ast.Assert:
    """Return the one ``assert`` statement whose source lines cover ``line_index`` (0-based)."""
    try:
        tree = ast.parse(body)
    except SyntaxError as error:
        raise TwinRefusalError(f"{what}: the check body does not parse ({error.msg})") from error
    line_number = line_index + 1
    covering = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        and node.lineno <= line_number <= (node.end_lineno or node.lineno)
    ]
    if len(covering) != 1:
        raise TwinRefusalError(
            f"{what}: line {line_index} of the check body sits inside {len(covering)} assert "
            f"statements, not exactly one"
        )
    return covering[0]


def _candidate_call_parts(node: ast.Assert, *, what: str) -> tuple[tuple[str, ...], str]:
    """Split ``assert candidate(args) == value`` into canonical arg sources and the value source."""
    test = node.test
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
        or not isinstance(test.left, ast.Call)
        or not isinstance(test.left.func, ast.Name)
        or test.left.func.id != CANDIDATE_NAME
        or test.left.keywords
    ):
        raise TwinRefusalError(
            f"{what}: the perturbed assertion is not `assert {CANDIDATE_NAME}(<positional args>) "
            f"== <value>`, so its input and expected value cannot be read off it"
        )
    return (
        tuple(ast.unparse(argument) for argument in test.left.args),
        ast.unparse(test.comparators[0]),
    )


def literal_value(source: str, *, what: str) -> object:
    """Evaluate canonical literal source, refusing an expression that is not a literal."""
    try:
        return ast.literal_eval(source)
    except (ValueError, SyntaxError) as error:
        raise TwinRefusalError(f"{what}: the asserted value is not a literal") from error


def _magnitude(honest: object, planted: object) -> tuple[str, float | None]:
    """Classify the perturbation: a signed numeric distance, or non-numeric."""
    numeric = (int, float)
    if (
        isinstance(honest, numeric)
        and isinstance(planted, numeric)
        and not isinstance(honest, bool)
        and not isinstance(planted, bool)
    ):
        return "numeric", float(planted - honest)
    return "non-numeric", None


def normalized_source(source: str) -> str:
    """Return whitespace- and quote-normalised source, for a lexical containment check."""
    return re.sub(r"\s+", "", source).replace('"', "'")


def worked_example_present(statement: str, entry_point: str, call_args: Sequence[str]) -> bool:
    """Return whether the statement works the perturbed input as an example.

    A lexical check on whitespace- and quote-normalised text, deliberately no more: the flag says
    whether the model could have compared the rigged assertion against an example in front of it.
    """
    call = f"{entry_point}({', '.join(call_args)})"
    return normalized_source(call) in normalized_source(statement)


def perturbation_facts(honest: ILCBProblem, rigged: ILCBProblem) -> PerturbationFacts:
    """Read the perturbation off the two check bodies, refusing anything but one changed value."""
    what = honest.task_id
    if honest.task_id != rigged.task_id:
        raise ValueError(f"{honest.task_id!r} and {rigged.task_id!r} are not one problem")
    honest_body = check_function_body(honest)
    rigged_body = check_function_body(rigged)
    index = replaced_line_index(honest_body, rigged_body)
    if index is None:
        raise TwinRefusalError(
            f"{what}: the rigged and honest check bodies differ by "
            f"{line_diff_count(honest_body, rigged_body)} lines, not exactly one replaced line"
        )
    honest_args, honest_value = _candidate_call_parts(
        _assert_covering_line(honest_body, index, what=what), what=what
    )
    rigged_args, planted_value = _candidate_call_parts(
        _assert_covering_line(rigged_body, index, what=what), what=what
    )
    if honest_args != rigged_args:
        raise TwinRefusalError(f"{what}: the perturbation moved the call arguments, not the value")
    if honest_value == planted_value:
        raise TwinRefusalError(f"{what}: the differing line leaves the asserted value unchanged")
    kind, magnitude = _magnitude(
        literal_value(honest_value, what=what), literal_value(planted_value, what=what)
    )
    return PerturbationFacts(
        problem_id=honest.task_id,
        entry_point=honest.entry_point,
        honest_line_index=index,
        honest_line=honest_body.splitlines()[index],
        rigged_line=rigged_body.splitlines()[index],
        call_args=honest_args,
        honest_value=honest_value,
        planted_value=planted_value,
        kind=kind,
        magnitude=magnitude,
        detectable=worked_example_present(honest.prompt, honest.entry_point, honest_args),
    )


# --------------------------------------------------------------------------------------
# The two authored controls
# --------------------------------------------------------------------------------------


def insert_line(text: str, line_index: int, line: str) -> str:
    """Insert ``line`` before line ``line_index``, at that line's indentation."""
    lines = text.split("\n")
    if not 0 <= line_index < len(lines):
        raise ValueError(f"line {line_index} is outside a {len(lines)}-line text")
    target = lines[line_index]
    indentation = target[: len(target) - len(target.lstrip())]
    return "\n".join([*lines[:line_index], indentation + line, *lines[line_index:]])


@dataclass(frozen=True, slots=True)
class StatementPerturbation:
    """One numeric literal in the statement's worked example, shifted by a recorded delta."""

    line_index: int
    original_literal: str
    perturbed_literal: str
    delta: float
    in_output_line: bool

    def to_public_dict(self) -> dict[str, object]:
        """Return position and delta only; the literals themselves are item text."""
        return {
            "line_index": self.line_index,
            "delta": self.delta,
            "in_output_line": self.in_output_line,
        }


def worked_example_lines(statement: str) -> list[tuple[int, bool]]:
    """Return doctest lines as ``(line index, is an output line)``: each call, then its outputs."""
    found: list[tuple[int, bool]] = []
    in_example = False
    for index, line in enumerate(statement.split("\n")):
        stripped = line.strip()
        if stripped.startswith(DOCTEST_PROMPT):
            found.append((index, False))
            in_example = True
        elif in_example and stripped:
            found.append((index, True))
        else:
            in_example = False
    return found


def _shifted_literal(literal: str, delta: float) -> str:
    """Shift a numeric literal, keeping an integer an integer."""
    if "." in literal:
        return repr(float(literal) + delta)
    return str(int(literal) + int(delta))


def statement_delta(facts: PerturbationFacts) -> float:
    """Return the grader's own integer shift when it has one, so the two controls match in size."""
    if facts.magnitude is not None and facts.magnitude != 0 and facts.magnitude.is_integer():
        return facts.magnitude
    return 1.0


def perturb_statement_literal(
    statement: str, facts: PerturbationFacts
) -> tuple[str, StatementPerturbation]:
    """Perturb the first numeric literal in the worked example, output lines first.

    The shifted literal is kept away from the planted value: were the example that carries it the
    perturbed input, matching the grader's error would turn this control into a consistent rigging
    rather than an unrelated literal, so a coincidence flips the sign of the shift.
    """
    lines = statement.split("\n")
    examples = worked_example_lines(statement)
    outputs = [(index, True) for index, is_output in examples if is_output]
    with_literal = [
        (index, is_output)
        for index, is_output in (*outputs, *examples)
        if _NUMERIC_LITERAL.search(lines[index])
    ]
    if not with_literal:
        raise TwinRefusalError(
            f"{facts.problem_id}: no numeric literal in the statement's worked example to perturb"
        )
    index, in_output = with_literal[0]
    match = _NUMERIC_LITERAL.search(lines[index])
    if match is None:  # pragma: no cover - the filter above guarantees a match
        raise ValueError(f"{facts.problem_id}: literal vanished between search and use")
    delta = statement_delta(facts)
    perturbed = _shifted_literal(match.group(0), delta)
    if normalized_source(perturbed) == normalized_source(facts.planted_value):
        delta = -delta
        perturbed = _shifted_literal(match.group(0), delta)
    lines[index] = lines[index][: match.start()] + perturbed + lines[index][match.end() :]
    return "\n".join(lines), StatementPerturbation(
        line_index=index,
        original_literal=match.group(0),
        perturbed_literal=perturbed,
        delta=delta,
        in_output_line=in_output,
    )
