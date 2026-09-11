"""Readout pairs for the TMAX twins: hack-versus-honest generations, and one-line minimal pairs.

The mechanistic wave needs two kinds of paired material beyond the prompt corpus, both built from
legibility-probe record files (``reward_hacking.legibility_probe``), where ``hack`` and
``hidden_pass`` are labels by construction rather than a judge's opinion:

**Type A -- generation pairs.** For one problem and one generating unit (the served checkpoint),
every hack record (visible pass, hidden fail) against a same-problem, same-unit, same-cell honest-pass
record (hidden pass, no hack). The two completions answered the identical prompt, so the pair differs
only in what the model wrote; a direction fitted within problem and unit on these is about hacking,
not about the problem. A hack with no honest partner is counted, never dropped silently.

**Type B -- same-author minimal pairs.** An honest-pass solution written by the unit itself, plus
ONE inserted line that returns the planted value for the perturbed input -- the smallest edit that
turns the honest solution into exactly the hack the rigged grader rewards. Two controls sit beside
it at the same position: a **correct-value** line of the same shape returning the honest expected
value (semantically a no-op for a correct solution), and a **neutral** comment line matched to the
hack line's token count. Honest-against-honest pairs, two distinct honest solutions of one problem
by one unit, are the null. The planted value and the perturbed input come from
:mod:`reward_hacking.interp.tmax_twin_perturbation`, read off the check bodies, never typed in.

The guard that has to be able to fail: a :class:`TypeBPair` refuses a hack side whose inserted line
returns the honest value, and a correct-value side that returns anything else. Offer the control as
the hack and construction goes red; ``test_tmax_readout_pairs`` does exactly that.

``hack`` / ``honest pass`` and the unit label are :mod:`reward_hacking.interp.tmax_generation_labels`'s
predicates, and every :class:`RecordRef` carries that module's ``stimulus_id``, so a pair's sides
name the same rows the generation-set labels and the captured cells do.

Every record carries item text (completions and solutions), so the outputs go under ``artifacts/``
(gitignored) and nothing here logs any of it.

    uv run python -m reward_hacking.interp.tmax_readout_pairs --records <records.jsonl>... \
        --out-dir artifacts/reward_hacking/tmax-interp/readout-pairs
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from games.parsing import THINK_CLOSE
from reward_hacking.interp.tmax_generation_labels import (
    generation_stimulus_id,
    is_hack,
    is_honest_pass,
    read_records,
    unit_label,
)
from reward_hacking.interp.tmax_twin_corpus import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_REVISION,
    load_tokenizer,
    problems_by_split,
)
from reward_hacking.interp.tmax_twin_perturbation import (
    PerturbationFacts,
    TwinRefusalError,
    literal_value,
    perturbation_facts,
)
from reward_hacking.interp.token_spans import encode
from reward_hacking.legibility_probe import GRADER_TWIN_CELLS, LegibilityRecord
from reward_hacking.train_dataset import SOLUTION_CLOSE, SOLUTION_OPEN
from reward_hacking.train_partition import SPLIT_ONEOFF, SPLIT_ORIGINAL

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

PAIRS_SCHEMA = 1
PAIRS_KIND = "tmax-readout-pairs"
TYPE_A_FILENAME = "type-a-pairs.jsonl"
TYPE_B_FILENAME = "type-b-pairs.jsonl"
NULL_FILENAME = "honest-null-pairs.jsonl"
SUMMARY_FILENAME = "pairs-summary.json"
DEFAULT_OUT_DIR = Path("artifacts/reward_hacking/tmax-interp/readout-pairs")

DEFAULT_MAX_HONEST_PER_HACK = 4
DEFAULT_MAX_NULL_PAIRS_PER_GROUP = 4

ROLE_HACK = "hack"
ROLE_CORRECT_VALUE = "correct-value"
ROLE_NEUTRAL = "neutral"

NEUTRAL_COMMENT_PREFIX = "# "
NEUTRAL_FILLER_WORD = "note"
MAX_NEUTRAL_FILLER_WORDS = 1024
"""Above the longest hack line seen (a 416-token list-literal input, doubled into the condition)."""


@dataclass(frozen=True, slots=True)
class RecordRef:
    """Locate one legibility record so a pair can be joined back to it, and to its captured row.

    ``stimulus_id`` is :func:`~reward_hacking.interp.tmax_generation_labels.generation_stimulus_id`,
    the name the capture driver and the labels file give the same completion, so a pair's two sides
    join to their activations without a second lookup.
    """

    stimulus_id: str
    unit: str
    cell: str
    problem_id: str
    group_index: int
    sample_index: int

    @classmethod
    def of(cls, record: LegibilityRecord) -> RecordRef:
        """Read the reference off a record."""
        return cls(
            stimulus_id=generation_stimulus_id(record),
            unit=unit_label(record),
            cell=record.cell,
            problem_id=record.problem_id,
            group_index=record.group_index,
            sample_index=record.sample_index,
        )

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the pairs JSONL."""
        return {
            "stimulus_id": self.stimulus_id,
            "unit": self.unit,
            "cell": self.cell,
            "problem_id": self.problem_id,
            "group_index": self.group_index,
            "sample_index": self.sample_index,
        }


def _grouped(
    records: Iterable[LegibilityRecord],
) -> dict[tuple[str, str, str], list[LegibilityRecord]]:
    """Group records by (unit, cell, problem), each group in sample order."""
    groups: dict[tuple[str, str, str], list[LegibilityRecord]] = {}
    for record in records:
        groups.setdefault((unit_label(record), record.cell, record.problem_id), []).append(record)
    for members in groups.values():
        members.sort(key=lambda record: (record.group_index, record.sample_index))
    return groups


# --------------------------------------------------------------------------------------
# Type A: a hack against a same-problem, same-unit honest pass
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypeAPair:
    """One hack completion beside one honest-pass completion of the same prompt by the same unit."""

    pair_id: str
    problem_id: str
    unit: str
    cell: str
    hack: RecordRef
    honest: RecordRef
    hack_completion: str
    honest_completion: str
    hack_solution: str | None
    honest_solution: str

    def __post_init__(self) -> None:
        """Refuse a pair that reaches across problems, units or cells."""
        for name, side in (("hack", self.hack), ("honest", self.honest)):
            if (side.unit, side.cell, side.problem_id) != (self.unit, self.cell, self.problem_id):
                raise ValueError(f"{self.pair_id}: the {name} side belongs to a different group")

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the pairs JSONL."""
        return {
            "pair_id": self.pair_id,
            "problem_id": self.problem_id,
            "unit": self.unit,
            "cell": self.cell,
            "hack": self.hack.to_json_dict(),
            "honest": self.honest.to_json_dict(),
            "hack_completion": self.hack_completion,
            "honest_completion": self.honest_completion,
            "hack_solution": self.hack_solution,
            "honest_solution": self.honest_solution,
        }


@dataclass(frozen=True, slots=True)
class TypeAPairs:
    """The type A pairs plus the accounting that keeps their denominators honest."""

    pairs: tuple[TypeAPair, ...]
    n_hack_records_by_unit: dict[str, int]
    n_unpaired_hacks_by_unit: dict[str, int]

    def summary(self) -> dict[str, object]:
        """Return counts only."""
        return {
            "n_pairs": len(self.pairs),
            "n_pairs_by_unit": dict(Counter(pair.unit for pair in self.pairs)),
            "n_hack_records_by_unit": dict(self.n_hack_records_by_unit),
            "n_unpaired_hacks_by_unit": dict(self.n_unpaired_hacks_by_unit),
            "n_problems_with_pairs_by_unit": {
                unit: len({pair.problem_id for pair in self.pairs if pair.unit == unit})
                for unit in self.n_hack_records_by_unit
            },
        }


def build_type_a_pairs(
    records: Sequence[LegibilityRecord],
    *,
    cells: Sequence[str] = tuple(cell.label for cell in GRADER_TWIN_CELLS),
    max_honest_per_hack: int = DEFAULT_MAX_HONEST_PER_HACK,
) -> TypeAPairs:
    """Pair every hack with up to ``max_honest_per_hack`` honest passes of its own group."""
    if max_honest_per_hack < 1:
        raise ValueError("a hack needs at least one honest partner")
    pairs: list[TypeAPair] = []
    n_hacks: Counter[str] = Counter()
    n_unpaired: Counter[str] = Counter()
    for (unit, cell, problem_id), members in sorted(_grouped(records).items()):
        if cell not in cells:
            continue
        hacks = [record for record in members if is_hack(record)]
        honest = [record for record in members if is_honest_pass(record)]
        n_hacks[unit] += len(hacks)
        if not honest:
            n_unpaired[unit] += len(hacks)
            continue
        for hack in hacks:
            for partner in honest[:max_honest_per_hack]:
                hack_ref, honest_ref = RecordRef.of(hack), RecordRef.of(partner)
                pairs.append(
                    TypeAPair(
                        pair_id=(
                            f"{problem_id}--{unit}--{cell}--h{hack.group_index}.{hack.sample_index}"
                            f"--o{partner.group_index}.{partner.sample_index}"
                        ),
                        problem_id=problem_id,
                        unit=unit,
                        cell=cell,
                        hack=hack_ref,
                        honest=honest_ref,
                        hack_completion=hack.completion,
                        honest_completion=partner.completion,
                        hack_solution=hack.solution,
                        honest_solution=cast("str", partner.solution),
                    )
                )
    for unit in n_hacks:
        n_unpaired.setdefault(unit, 0)
    return TypeAPairs(
        pairs=tuple(pairs),
        n_hack_records_by_unit=dict(n_hacks),
        n_unpaired_hacks_by_unit=dict(n_unpaired),
    )


# --------------------------------------------------------------------------------------
# Type B: an honest solution plus one inserted line
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InsertionPoint:
    """Where one line goes into an honest solution: before which line, at what indentation."""

    parameters: tuple[str, ...]
    line_index: int
    indentation: str


def insertion_point(solution: str, entry_point: str, *, n_call_args: int) -> InsertionPoint:
    """Find the entry point's first body statement and its positional parameters, or refuse.

    Refused shapes: a solution that does not parse, no or several module-level definitions of the
    entry point, a signature with varargs or keyword-only parameters, a parameter count that does
    not match the perturbed call (the comparison would then never be true), and a one-line body.
    The inserted line goes after a leading docstring so the docstring stays one.
    """
    what = entry_point
    try:
        tree = ast.parse(solution)
    except SyntaxError as error:
        raise TwinRefusalError(
            f"{what}: the honest solution does not parse ({error.msg})"
        ) from error
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ]
    if len(definitions) != 1:
        raise TwinRefusalError(
            f"{what}: {len(definitions)} module-level definitions of the entry point, not one"
        )
    definition = definitions[0]
    arguments = definition.args
    if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
        raise TwinRefusalError(f"{what}: the entry point takes varargs or keyword-only parameters")
    parameters = tuple(arg.arg for arg in (*arguments.posonlyargs, *arguments.args))
    if len(parameters) != n_call_args:
        raise TwinRefusalError(
            f"{what}: the entry point takes {len(parameters)} parameters but the perturbed call "
            f"passes {n_call_args}"
        )
    body = definition.body
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        and len(body) > 1
    ):
        first = body[1]
    if first.lineno == definition.lineno:
        raise TwinRefusalError(f"{what}: the entry point's body shares the definition's line")
    line_index = first.lineno - 1
    line = solution.split("\n")[line_index]
    return InsertionPoint(
        parameters=parameters,
        line_index=line_index,
        indentation=line[: len(line) - len(line.lstrip())],
    )


def _comparison(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"


def return_line(parameters: Sequence[str], call_args: Sequence[str], value: str) -> str:
    """Build ``if <params> == <args>: return <value>``, the one-line special case."""
    return f"if {_comparison(parameters)} == {_comparison(call_args)}: return {value}"


def returned_literal(line: str) -> object:
    """Read the literal an ``if ...: return <literal>`` line returns, refusing any other shape."""
    try:
        node = ast.parse(line.strip()).body
    except SyntaxError as error:
        raise ValueError(f"the inserted line does not parse ({error.msg})") from error
    if (
        len(node) != 1
        or not isinstance(node[0], ast.If)
        or len(node[0].body) != 1
        or not isinstance(node[0].body[0], ast.Return)
        or node[0].body[0].value is None
    ):
        raise ValueError("the inserted line is not `if <condition>: return <value>`")
    return literal_value(ast.unparse(node[0].body[0].value), what="inserted line")


def insert_solution_line(solution: str, point: InsertionPoint, line: str) -> str:
    """Insert one line at the insertion point, at the body's indentation."""
    lines = solution.split("\n")
    return "\n".join(
        [*lines[: point.line_index], point.indentation + line, *lines[point.line_index :]]
    )


def neutral_line(
    encode_text: Callable[[str], Sequence[int]], *, indentation: str, target_tokens: int
) -> str:
    """Build a comment line whose token count (with indentation and newline) matches the target.

    The filler is one repeated neutral word, grown until the count matches exactly; when no count
    lands exactly (a filler word can straddle a token boundary), the closest is taken and the
    achieved count is what the pair records.
    """
    best: tuple[int, str] | None = None
    for n_words in range(1, MAX_NEUTRAL_FILLER_WORDS + 1):
        line = NEUTRAL_COMMENT_PREFIX + " ".join([NEUTRAL_FILLER_WORD] * n_words)
        count = len(encode_text(indentation + line + "\n"))
        gap = abs(count - target_tokens)
        if best is None or gap < best[0]:
            best = (gap, line)
        if count >= target_tokens:
            break
    if best is None:  # pragma: no cover - the range is non-empty
        raise ValueError("no neutral line built")
    return best[1]


@dataclass(frozen=True, slots=True)
class InsertedLineVariant:
    """One honest solution with one line inserted, under one role."""

    role: str
    line: str
    solution: str
    line_index: int
    n_line_tokens: int

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the pairs JSONL."""
        return {
            "role": self.role,
            "line": self.line,
            "solution": self.solution,
            "line_index": self.line_index,
            "n_line_tokens": self.n_line_tokens,
        }


def _assert_one_line_inserted(honest: str, variant: InsertedLineVariant, *, what: str) -> None:
    lines = variant.solution.split("\n")
    without = lines[: variant.line_index] + lines[variant.line_index + 1 :]
    if "\n".join(without) != honest or not lines[variant.line_index].endswith(variant.line):
        raise ValueError(
            f"{what}: the {variant.role} side is not the honest solution plus its line"
        )


@dataclass(frozen=True, slots=True)
class TypeBPair:
    """An honest solution beside its three one-line variants, with the roles checked by value.

    The check that has to be able to fail: the hack side's line must return the planted value and
    must NOT return the honest value; the correct-value side's line must return the honest value.
    A correct-value line offered as the hack is refused here, by evaluated literal, not by label.
    """

    pair_id: str
    problem_id: str
    unit: str
    cell: str
    honest: RecordRef
    honest_solution: str
    honest_value: str
    planted_value: str
    hack: InsertedLineVariant
    correct_value: InsertedLineVariant
    neutral: InsertedLineVariant

    def __post_init__(self) -> None:
        """Refuse mislabelled roles and edits that are not one inserted line."""
        honest = literal_value(self.honest_value, what=self.pair_id)
        planted = literal_value(self.planted_value, what=self.pair_id)
        if honest == planted:
            raise ValueError(f"{self.pair_id}: the planted value equals the honest value")
        if self.hack.role != ROLE_HACK or self.correct_value.role != ROLE_CORRECT_VALUE:
            raise ValueError(f"{self.pair_id}: the variants carry the wrong roles")
        if self.neutral.role != ROLE_NEUTRAL or not self.neutral.line.startswith("#"):
            raise ValueError(f"{self.pair_id}: the neutral line is not a comment")
        hack_returns = returned_literal(self.hack.line)
        if hack_returns == honest:
            raise ValueError(
                f"{self.pair_id}: the hack side's inserted line returns the HONEST expected value; "
                f"offered as a hack, it would score a correct solution as a hack"
            )
        if hack_returns != planted:
            raise ValueError(
                f"{self.pair_id}: the hack side's inserted line does not return the planted value"
            )
        if returned_literal(self.correct_value.line) != honest:
            raise ValueError(
                f"{self.pair_id}: the correct-value line does not return the honest value"
            )
        for variant in (self.hack, self.correct_value, self.neutral):
            _assert_one_line_inserted(self.honest_solution, variant, what=self.pair_id)

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the pairs JSONL."""
        return {
            "pair_id": self.pair_id,
            "problem_id": self.problem_id,
            "unit": self.unit,
            "cell": self.cell,
            "honest": self.honest.to_json_dict(),
            "honest_solution": self.honest_solution,
            "honest_value": self.honest_value,
            "planted_value": self.planted_value,
            "variants": {
                variant.role: variant.to_json_dict()
                for variant in (self.hack, self.correct_value, self.neutral)
            },
        }


def build_type_b_pair(
    record: LegibilityRecord,
    facts: PerturbationFacts,
    *,
    encode_text: Callable[[str], Sequence[int]],
) -> TypeBPair:
    """Build one honest-pass record's three variants, or refuse with the reason."""
    if not is_honest_pass(record):
        raise TwinRefusalError(f"{record.problem_id}: the record is not an honest pass")
    if record.problem_id != facts.problem_id:
        raise ValueError(f"facts for {facts.problem_id} offered to a {record.problem_id} record")
    solution = cast("str", record.solution)
    point = insertion_point(solution, facts.entry_point, n_call_args=len(facts.call_args))

    def variant(role: str, line: str) -> InsertedLineVariant:
        return InsertedLineVariant(
            role=role,
            line=line,
            solution=insert_solution_line(solution, point, line),
            line_index=point.line_index,
            n_line_tokens=len(encode_text(point.indentation + line + "\n")),
        )

    hack = variant(ROLE_HACK, return_line(point.parameters, facts.call_args, facts.planted_value))
    ref = RecordRef.of(record)
    return TypeBPair(
        pair_id=f"{record.problem_id}--{ref.unit}--{ref.cell}--o{ref.group_index}.{ref.sample_index}",
        problem_id=record.problem_id,
        unit=ref.unit,
        cell=ref.cell,
        honest=ref,
        honest_solution=solution,
        honest_value=facts.honest_value,
        planted_value=facts.planted_value,
        hack=hack,
        correct_value=variant(
            ROLE_CORRECT_VALUE, return_line(point.parameters, facts.call_args, facts.honest_value)
        ),
        neutral=variant(
            ROLE_NEUTRAL,
            neutral_line(
                encode_text, indentation=point.indentation, target_tokens=hack.n_line_tokens
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class HonestNullPair:
    """Two distinct honest-pass solutions of one problem by one unit: the same-author null."""

    pair_id: str
    problem_id: str
    unit: str
    cell: str
    first: RecordRef
    second: RecordRef
    first_solution: str
    second_solution: str

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the pairs JSONL."""
        return {
            "pair_id": self.pair_id,
            "problem_id": self.problem_id,
            "unit": self.unit,
            "cell": self.cell,
            "first": self.first.to_json_dict(),
            "second": self.second.to_json_dict(),
            "first_solution": self.first_solution,
            "second_solution": self.second_solution,
        }


@dataclass(frozen=True, slots=True)
class TypeBPairs:
    """The type B pairs, the honest nulls, and the accounting of every honest pass not paired."""

    pairs: tuple[TypeBPair, ...]
    null_pairs: tuple[HonestNullPair, ...]
    n_honest_passes_by_unit: dict[str, int]
    refused_by_reason: dict[str, int]
    refused_problem_ids: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        """Return counts only."""
        return {
            "n_pairs": len(self.pairs),
            "n_pairs_by_unit": dict(Counter(pair.unit for pair in self.pairs)),
            "n_problems_with_pairs": len({pair.problem_id for pair in self.pairs}),
            "n_null_pairs": len(self.null_pairs),
            "n_null_pairs_by_unit": dict(Counter(pair.unit for pair in self.null_pairs)),
            "n_honest_passes_by_unit": dict(self.n_honest_passes_by_unit),
            "n_refused_by_reason": dict(self.refused_by_reason),
            "refused_problem_ids": dict(self.refused_problem_ids),
            "neutral_line_token_gap": dict(
                Counter(pair.neutral.n_line_tokens - pair.hack.n_line_tokens for pair in self.pairs)
            ),
        }


def _refusal_reason(refusal: TwinRefusalError) -> str:
    """Strip the leading problem or entry-point label off a refusal so reasons can be counted."""
    message = str(refusal)
    return message.split(": ", 1)[1] if ": " in message else message


def facts_for_problems(
    problem_ids: Iterable[str],
) -> tuple[dict[str, PerturbationFacts], dict[str, str]]:
    """Read the perturbation facts for each problem off the registry, collecting refusals."""
    facts: dict[str, PerturbationFacts] = {}
    refused: dict[str, str] = {}
    for problem_id in sorted(set(problem_ids)):
        try:
            rows = problems_by_split(problem_id)
            facts[problem_id] = perturbation_facts(rows[SPLIT_ORIGINAL], rows[SPLIT_ONEOFF])
        except TwinRefusalError as refusal:
            refused[problem_id] = str(refusal)
    return facts, refused


def build_type_b_pairs(
    records: Sequence[LegibilityRecord],
    facts_by_problem: Mapping[str, PerturbationFacts],
    *,
    encode_text: Callable[[str], Sequence[int]],
    cells: Sequence[str] = tuple(cell.label for cell in GRADER_TWIN_CELLS),
    max_null_pairs_per_group: int = DEFAULT_MAX_NULL_PAIRS_PER_GROUP,
) -> TypeBPairs:
    """Build every honest pass's variants and the honest-against-honest nulls, counting refusals.

    Problems absent from ``facts_by_problem`` are counted as refused per honest pass; the caller
    attaches the per-problem refusal reasons (:func:`facts_for_problems`) to the result.
    """
    pairs: list[TypeBPair] = []
    nulls: list[HonestNullPair] = []
    n_honest: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    for (unit, cell, problem_id), members in sorted(_grouped(records).items()):
        if cell not in cells:
            continue
        honest = [record for record in members if is_honest_pass(record)]
        n_honest[unit] += len(honest)
        facts = facts_by_problem.get(problem_id)
        if facts is None:
            refused["problem not in the twin corpus"] += len(honest)
        else:
            for record in honest:
                try:
                    pairs.append(build_type_b_pair(record, facts, encode_text=encode_text))
                except TwinRefusalError as refusal:
                    refused[_refusal_reason(refusal)] += 1
        distinct: list[LegibilityRecord] = []
        seen: set[str] = set()
        for record in honest:
            solution = cast("str", record.solution)
            if solution not in seen:
                seen.add(solution)
                distinct.append(record)
        for first, second in itertools.islice(
            itertools.combinations(distinct, 2), max_null_pairs_per_group
        ):
            first_ref, second_ref = RecordRef.of(first), RecordRef.of(second)
            nulls.append(
                HonestNullPair(
                    pair_id=(
                        f"{problem_id}--{unit}--{cell}--o{first.group_index}.{first.sample_index}"
                        f"--o{second.group_index}.{second.sample_index}"
                    ),
                    problem_id=problem_id,
                    unit=unit,
                    cell=cell,
                    first=first_ref,
                    second=second_ref,
                    first_solution=cast("str", first.solution),
                    second_solution=cast("str", second.solution),
                )
            )
    return TypeBPairs(
        pairs=tuple(pairs),
        null_pairs=tuple(nulls),
        n_honest_passes_by_unit=dict(n_honest),
        refused_by_reason=dict(refused),
    )


def teacher_forced_completion(solution: str) -> str:
    """Wrap a solution as an assistant turn with empty reasoning, for teacher-forced scoring.

    The template prefills the opening think tag and a newline; this closes the block at once and
    hands back the solution in the answer contract, so two variants of one solution differ only in
    their inserted line.
    """
    return f"\n{THINK_CLOSE}\n\n{SOLUTION_OPEN}\n{solution}\n{SOLUTION_CLOSE}"


# --------------------------------------------------------------------------------------
# Writing and the CLI
# --------------------------------------------------------------------------------------


def write_pairs(type_a: TypeAPairs, type_b: TypeBPairs, out_dir: Path) -> Path:
    """Write the three JSONL files and the counts-only summary; refuse to overwrite."""
    summary_path = out_dir / SUMMARY_FILENAME
    if summary_path.exists():
        raise FileExistsError(f"{out_dir} already holds readout pairs; write a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        (TYPE_A_FILENAME, [pair.to_json_dict() for pair in type_a.pairs]),
        (TYPE_B_FILENAME, [pair.to_json_dict() for pair in type_b.pairs]),
        (NULL_FILENAME, [pair.to_json_dict() for pair in type_b.null_pairs]),
    ):
        (out_dir / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    summary = {
        "schema": PAIRS_SCHEMA,
        "kind": PAIRS_KIND,
        "type_a": type_a.summary(),
        "type_b": type_b.summary(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote readout pairs, %s", json.dumps(summary))
    return summary_path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-honest-per-hack", type=int, default=DEFAULT_MAX_HONEST_PER_HACK)
    parser.add_argument(
        "--max-null-pairs-per-group", type=int, default=DEFAULT_MAX_NULL_PAIRS_PER_GROUP
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build both pair types from record files and write them under artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    records = read_records(cast("list[Path]", args.records))
    tokenizer = load_tokenizer(cast("str", args.model), cast("str | None", args.revision))
    facts, refused = facts_for_problems(record.problem_id for record in records)
    type_a = build_type_a_pairs(records, max_honest_per_hack=cast("int", args.max_honest_per_hack))
    type_b = replace(
        build_type_b_pairs(
            records,
            facts,
            encode_text=lambda text: encode(tokenizer, text),
            max_null_pairs_per_group=cast("int", args.max_null_pairs_per_group),
        ),
        refused_problem_ids=refused,
    )
    write_pairs(type_a, type_b, cast("Path", args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
