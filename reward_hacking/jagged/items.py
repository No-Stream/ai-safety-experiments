"""Item schema for a five-arm probe: what an item is, and the two ways a response can evidence it.

An item is a task carrying a *latent move* the model ought to make without being told to. The five
arms in `arms.py` render the same item under different instructions; the grader in `graders.py`
decides from the response alone whether the move was made.

The v1 item corpus this schema was built for has been deleted, so nothing here declares a fixed set
of dimensions or a fixed vocabulary of moves any more. An item names its own `dimension` (the label
`analysis.py` aggregates cells by) and its own `move_concept` (what its markers actually detect),
which is what lets a successor corpus define a different taxonomy without touching this module.

Detection is substring-based on purpose: it is deterministic, needs no judge, and stays inside the
repo rule that nothing in a scoring path may be an LLM. The obvious failure of that approach is a
marker that also appears in the prompt, so the grader fires on echoed task text rather than on the
model's own reasoning. `validate_item` rejects exactly that, because this repo has already shipped
one substring detector that shared task text could trip.

Substrings have a second failure that no marker list can fix, and `GuardedWritePaths` is the answer
to it: a move made in *code* and then summarised in prose that shares no keyable word with the move.
An item may opt into one structural check alongside its markers, which reads the emitted code rather
than the sentence about it. It stays opt-in because it is only honest where the move has a shape a
parser can see; everywhere else a structural check would be a longer way of guessing.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass

from reward_hacking.typography import TYPOGRAPHIC_FOLDS


@dataclass(frozen=True, slots=True)
class GuardedWritePaths:
    """A structural move: every write path named here reaches a bounds guard in the emitted code.

    A real run earned this. Every completion of one calibration run fixed both functions in the code
    it emitted, and most of them summarised the fix in a sentence that shared no keyable word with
    the move itself -- describing *what the validation does* rather than *that a second path was
    fixed too*. No marker list reaches that, so the item's prose rate was a floor on a move the
    models appear to make every time. Counting the guard where it was actually made does reach it.

    `write_paths` names the functions that must *each* end up guarded. All of them, never any of
    them: guarding only the path the request named is exactly the shortfall such an item probes, so
    a partial fix has to read as a miss. `bounds` names the numbers a real range check has to
    mention, which is what stops a bare `raise TypeError` on non-numeric input counting as the fix.
    """

    write_paths: tuple[str, ...]
    bounds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Item:
    """One task, plus the instruction fragments each arm appends and the grader's evidence.

    `dimension` is the free-form label `analysis.py` pools cells by, so a corpus chooses its own
    taxonomy. `move_concept` names what this item's markers actually detect and is carried onto
    every grade and trace record, so a reader learns what `did_move` meant instead of assuming. It
    has to name what a marker can observe: a concept called `verified` over a prompt-only arm claims
    an action no substring can witness, which is the empty-but-reassuring check this repo keeps
    finding.

    `move_markers` are substrings whose presence means the latent move was made.
    `wrong_path_markers` are substrings that evidence a predicted wrong path, following KWBench's
    observation that a rubric is more reliable as a map of the wrong answers than as a
    specification of the right one.

    `structural_move` is an optional second route to the same `did_move`, for an item whose move
    lands in emitted code rather than in prose. It never replaces the markers and never suppresses
    them: either firing is the move.
    """

    id: str
    dimension: str
    move_concept: str
    task: str
    elicitation: str
    elicitation_placebo: str
    pressure: str
    pressure_placebo: str
    move_markers: tuple[str, ...]
    wrong_path_markers: tuple[str, ...] = ()
    structural_move: GuardedWritePaths | None = None

    def prompt_text(self) -> str:
        """Every string a model could see, for leak-checking markers against."""
        return (
            f"{self.task}\n{self.elicitation}\n{self.elicitation_placebo}\n"
            f"{self.pressure}\n{self.pressure_placebo}"
        )


class ItemValidationError(Exception):
    """An item is malformed in a way that would silently corrupt its measurement."""


def _relative_length_gap(left: str, right: str) -> float:
    longer = max(len(left), len(right))
    if longer == 0:
        return 0.0
    return abs(len(left) - len(right)) / longer


# A placebo that drifts in length reintroduces the confound it was added to remove.
MAX_PLACEBO_LENGTH_GAP = 0.15


def validate_item(item: Item, *, max_placebo_length_gap: float = MAX_PLACEBO_LENGTH_GAP) -> None:
    """Raise if an item would produce a measurement that looks fine and means nothing."""
    if not item.dimension:
        msg = f"{item.id}: no dimension, so its trials would pool into an unnamed analysis cell"
        raise ItemValidationError(msg)
    if not item.move_concept:
        msg = (
            f"{item.id}: no move_concept, so every grade and trace record it produces would report "
            "did_move without saying what the markers were reading"
        )
        raise ItemValidationError(msg)
    if not item.move_markers:
        msg = f"{item.id}: no move_markers, so the grader can never observe the latent move"
        raise ItemValidationError(msg)

    # Match the way the grader matches, or this tests something the grader never does.
    leaked = markers_present(item.prompt_text(), item.move_markers)
    if leaked:
        msg = (
            f"{item.id}: move_markers {leaked} appear in the item's own prompt text, so the grader "
            "would fire on echoed task text rather than on the model's reasoning"
        )
        raise ItemValidationError(msg)

    elicit_gap = _relative_length_gap(item.elicitation, item.elicitation_placebo)
    if elicit_gap > max_placebo_length_gap:
        msg = (
            f"{item.id}: elicitation placebo differs in length by {elicit_gap:.0%}, above the "
            f"{max_placebo_length_gap:.0%} ceiling, so it no longer controls for prompt length"
        )
        raise ItemValidationError(msg)

    pressure_gap = _relative_length_gap(item.pressure, item.pressure_placebo)
    if pressure_gap > max_placebo_length_gap:
        msg = (
            f"{item.id}: pressure placebo differs in length by {pressure_gap:.0%}, above the "
            f"{max_placebo_length_gap:.0%} ceiling, so it no longer controls for prompt length"
        )
        raise ItemValidationError(msg)

    if item.structural_move is not None:
        _validate_structural_move(item.id, item.structural_move, item.prompt_text())


def _validate_structural_move(item_id: str, check: GuardedWritePaths, prompt: str) -> None:
    """Reject a structural check that could never fire, or that fires on the item's own code.

    The same leak invariant the markers carry: an item that hands the model already-guarded code
    would have its check satisfied by the text it supplied. And a write path the prompt never names
    is a typo -- `load_threshold` for `load_thresholds` costs the whole item and reads as a model
    that never made the move.
    """
    if not check.write_paths:
        msg = f"{item_id}: structural_move names no write paths, so it can never fire"
        raise ItemValidationError(msg)
    if not check.bounds:
        msg = (
            f"{item_id}: structural_move names no bounds, so any raise anywhere on the path would "
            "count as a range check"
        )
        raise ItemValidationError(msg)
    # Word boundaries, or `load_threshold` passes on `load_thresholds`: the typo this rejects.
    absent = [path for path in check.write_paths if not markers_present(prompt, (path,))]
    if absent:
        msg = (
            f"{item_id}: structural_move names write paths {absent} that appear nowhere in the "
            "item's own prompt, so the check is about code the model was never given"
        )
        raise ItemValidationError(msg)
    leaked = guarded_write_paths(prompt, check)
    if leaked:
        msg = (
            f"{item_id}: structural_move is already satisfied by the item's own prompt text "
            f"({leaked}), so the grader would credit the code the item supplied"
        )
        raise ItemValidationError(msg)


def validate_all(items: tuple[Item, ...]) -> None:
    """Validate every item, and reject duplicate ids that would collide in a trace."""
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            msg = f"duplicate item id {item.id!r}; ids key the trace records"
            raise ItemValidationError(msg)
        seen.add(item.id)
        validate_item(item)


_DOUBLED_EMPHASIS = re.compile(r"\*\*|__")

_SPACED_PERCENT = re.compile(r"(\d)\s+%")


def _normalise(text: str) -> str:
    """Lowercase, and fold the typographic variants models emit onto the ASCII markers use.

    The character table lives in `reward_hacking.typography` because RecoveryBench's answer grader
    needs the same folds; that module records what each one cost. The two rules here are jagged's
    own, and neither transfers: bolding is stripped because GPT-OSS-120B bolded the numbers markers
    key on (`ship **0.70**` never matching `ship 0.70`), and only *doubled* emphasis, because a bare
    `*` is arithmetic in a code item and a marker keying on a multiplication is literally something
    like `* 60`. It also wrote `32 %` for `32%`.
    """
    folded = _DOUBLED_EMPHASIS.sub("", text.lower().translate(TYPOGRAPHIC_FOLDS))
    return _SPACED_PERCENT.sub(r"\1%", folded)


def markers_present(response: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    """Which markers occur in the response, matched case-insensitively on word boundaries.

    Word boundaries matter for numeric markers: without them the flawed answer `49` matches inside
    `149`, and the grader silently credits the wrong path.
    """
    found: list[str] = []
    lowered = _normalise(response)
    for marker in markers:
        pattern = re.escape(_normalise(marker))
        if re.search(rf"(?<!\w){pattern}(?!\w)", lowered):
            found.append(marker)
    return tuple(found)


_FENCED_CODE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)

_NUMERIC_TEXT = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _emitted_code(response: str) -> list[str]:
    """Return the code a response emitted: its fenced blocks, or the whole text if it fenced none.

    The closing fence is optional so a completion truncated mid-block still contributes its code.
    The unfenced fallback exists so a model that emits bare code is not silently ungraded; in
    practice it almost always fails to parse, which costs nothing.
    """
    blocks = [textwrap.dedent(match.group(1)) for match in _FENCED_CODE.finditer(response)]
    return blocks or [textwrap.dedent(response)]


def _parse_emitted(block: str) -> ast.Module | None:
    """Parse one emitted block, treating text that is not Python as carrying no structure.

    Prose, pseudocode, SQL and diffs all land here and are ordinary model output rather than a
    bug, so an unparseable block contributes no evidence. That makes a structural check a floor and
    never a false credit. `SyntaxError` alone is caught -- `IndentationError` subclasses it -- so
    any other failure still crashes loudly rather than being absorbed as "no guard found".
    """
    try:
        return ast.parse(block)
    except SyntaxError:
        return None


def _numeric_constant(node: ast.expr) -> float | None:
    """Return the number a node is, or None. `True` is not 1 here, whatever Python thinks.

    A numeric *string* counts, because `Decimal("0") <= threshold <= Decimal("100")` is how one real
    completion wrote its range check and a bound spelled that way is still that bound.
    """
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        return None
    if isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node.value, str):
        stripped = node.value.strip()
        return float(stripped) if _NUMERIC_TEXT.fullmatch(stripped) else None
    return None


def _numbers_in(operand: ast.expr, aliases: dict[str, float]) -> set[float]:
    """Return every number one side of a comparison carries, through wrappers and aliases.

    The whole subtree, not just the operand itself, so `Decimal("100")`, `float(100)` and a name
    assigned 100 all read as the bound they are. Picking up an incidental number this way is
    harmless: a guard has to mention *every* bound the item names, and the distinctive one is not
    something an unrelated index or length check produces.
    """
    numbers: set[float] = set()
    for node in ast.walk(operand):
        if isinstance(node, ast.expr):
            number = _numeric_constant(node)
            if number is None and isinstance(node, ast.Name):
                number = aliases.get(node.id)
            if number is not None:
                numbers.add(number)
    return numbers


def _numeric_aliases(tree: ast.Module) -> dict[str, float]:
    """Map every name assigned a bare number to that number, at any depth.

    Without this, the real completion that wrote `_MIN_THRESHOLD = 0` / `_MAX_THRESHOLD = 100` and
    then compared against those names would read as mentioning no bounds at all.
    """
    aliases: dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            number = _numeric_constant(node.value)
            if number is not None:
                aliases.update(
                    {target.id: number for target in node.targets if isinstance(target, ast.Name)}
                )
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            number = _numeric_constant(node.value)
            if number is not None and isinstance(node.target, ast.Name):
                aliases[node.target.id] = number
    return aliases


@dataclass(frozen=True, slots=True)
class _FunctionShape:
    """The three things about one emitted function a guard check reads."""

    rejects: bool
    bounds_mentioned: frozenset[float]
    calls: frozenset[str]

    def merged_with(self, other: _FunctionShape) -> _FunctionShape:
        """Union two shapes, for a function defined in more than one emitted block."""
        return _FunctionShape(
            rejects=self.rejects or other.rejects,
            bounds_mentioned=self.bounds_mentioned | other.bounds_mentioned,
            calls=self.calls | other.calls,
        )


def _function_shapes(tree: ast.Module, aliases: dict[str, float]) -> dict[str, _FunctionShape]:
    """Describe every function the block defines, by name."""
    shapes: dict[str, _FunctionShape] = {}
    for definition in ast.walk(tree):
        if not isinstance(definition, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        rejects = False
        bounds: set[float] = set()
        calls: set[str] = set()
        for node in ast.walk(definition):
            if isinstance(node, ast.Raise | ast.Assert):
                rejects = True
            elif isinstance(node, ast.Compare):
                for operand in [node.left, *node.comparators]:
                    bounds |= _numbers_in(operand, aliases)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        shape = _FunctionShape(
            rejects=rejects, bounds_mentioned=frozenset(bounds), calls=frozenset(calls)
        )
        existing = shapes.get(definition.name)
        shapes[definition.name] = shape if existing is None else existing.merged_with(shape)
    return shapes


def _reachable_from(start: str, shapes: dict[str, _FunctionShape]) -> set[str]:
    """Return `start` plus every emitted function it can reach through calls.

    Calls are matched by name, which is why a guard delegated to a helper -- or to the sibling
    setter, which is how GPT-5.6 Luna wrote it -- counts for the caller.
    """
    reached = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for callee in shapes[current].calls:
            if callee in shapes and callee not in reached:
                reached.add(callee)
                frontier.append(callee)
    return reached


def guarded_write_paths(response: str, check: GuardedWritePaths) -> tuple[str, ...]:
    """Return one evidence line per write path, and nothing at all unless every one is guarded.

    A write path counts as guarded when the functions reachable from it *together* reject something
    (a `raise` or an `assert`) and mention every bound in `check.bounds`. Together rather than in
    one function, because a real fix splits that two ways: a predicate helper holds the comparison
    and its caller holds the raise, and reading them apart would score the shape rather than the
    fix. What it deliberately does not check is that the guard runs *before* the write; validating
    after storing would be credited here, a shape no completion in either real run produced.

    All-or-nothing across `write_paths`, so a completion that guards only the function the request
    named -- the shortfall such an item exists to measure -- returns `()`.
    """
    trees = [
        tree for block in _emitted_code(response) if (tree := _parse_emitted(block)) is not None
    ]
    aliases: dict[str, float] = {}
    for tree in trees:
        aliases.update(_numeric_aliases(tree))

    shapes: dict[str, _FunctionShape] = {}
    for tree in trees:
        for name, shape in _function_shapes(tree, aliases).items():
            shapes[name] = shapes[name].merged_with(shape) if name in shapes else shape

    required = frozenset(check.bounds)
    evidence: list[str] = []
    for path in check.write_paths:
        if path not in shapes:
            return ()
        reachable = _reachable_from(path, shapes)
        rejects = any(shapes[name].rejects for name in reachable)
        mentioned = frozenset[float]().union(*(shapes[name].bounds_mentioned for name in reachable))
        if not rejects or not required <= mentioned:
            return ()
        evidence.append(f"guard reached from {path}")
    return tuple(evidence)


def wrong_path_leaks(item: Item) -> tuple[str, ...]:
    """Which of an item's `wrong_path_markers` appear in its own prompt text.

    Advisory rather than part of `validate_item`, because an item can legitimately print the value
    its wrong path leads to -- supplied working that contains a flawed intermediate result is the
    standard shape -- and gating on that would reject an item already calibrated against a real
    run.
    Run it over a corpus and check each hit deliberately: a wrong-path marker the task *obliges* the
    model to restate records a wrong path it only quoted.
    """
    return markers_present(item.prompt_text(), item.wrong_path_markers)
