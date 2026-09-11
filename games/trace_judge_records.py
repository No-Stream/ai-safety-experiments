"""Records of the games trace judge: which battery rows a launch selects, and how each becomes a record.

Record identity is ``arm|cell|step|prompt_id|sample_index|label_print_order``, the battery's own
identity fields prefixed with where the record came from. The arm label is the run-prefix directory's
basename rather than the cell's ``meta`` row's ``arm``: the matched-size control's cells carry the
PARENT arm's name there (their run dir was named after it), and keying on it would collide two arms in
one judged file. The hand-label file keys records by the sample exports' spelling instead (the
export's ``rung`` or ``source_arm``, its own ``cell``, ``order`` for the control's print order); the
validate step translates between the two through the completion digest both carry.

A :class:`Selection` names the rows in scope. Beside the section, game and framing axes it carries the
two selectors the 9B coherence census of 2026-09-03 had to build a scratch driver for, whose semantics
live here now: ``decision`` (``cooperate``, the default and the only selection before that date;
``defect``; or ``all``, every parsed record stamped with the decision it took) and ``per_render`` (a
sample of at most N records per (prompt_id, label_print_order) render, the lowest sample indices sorted
numerically, never file order or a random draw). :func:`record_decision` reads a row's decision off
``coop_fraction``, which the battery scores as exactly 1.0 or 0.0 on a parsed one-shot matrix action;
:func:`load_cell_records` applies the selection to one banked cell and reports the funnel. The judge,
census, rates and CLI are in ``games.trace_judge``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.prompts import is_matrix_game, print_order_of
from games.trace_judge_rates import DECISION_COOPERATE, DECISION_DEFECT, Handle

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

SECTIONS_WITH_TRACES: tuple[str, ...] = ("framing-sweep", "game-behavior", "training-frames")
META_RECORD = "meta"
COOP_FIELD = "coop_fraction"

DECISION_ALL = "all"
"""The selector that admits every parsed record; never a record's own decision."""
DECISION_CHOICES: tuple[str, ...] = (DECISION_COOPERATE, DECISION_DEFECT, DECISION_ALL)
_FUNNEL_COUNT_OF_DECISION: dict[str, str] = {
    DECISION_COOPERATE: "cooperating",
    DECISION_DEFECT: "defecting",
}


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """One parsed matrix-game battery record with everything the judge, the key, and the rates need.

    ``arm`` is the run-prefix directory's basename (see the module docstring for why it is not the
    meta row's arm); ``trained_arm`` carries the meta row's value for the record. ``counterpart_framing``
    and ``reskin_id`` are ``None`` on the sections that do not carry them. ``decision`` is the one the
    record took (:func:`record_decision`), stamped onto its census and judged rows.

    ``framings_digest`` is the cell meta's ``eval_config.framings_digest``: the digest of the runtime
    framings file the cell rendered under, empty where it rendered only the tracked registry's
    framings. It rides on the record because the rates step reads a census with no tree and no meta
    beside it, where a ``counterpart_framing`` off the registry means a framing that file supplied
    when this field is set and a mistyped id when it is not
    (``games.trace_judge_rates.coupling_clause_in_prompt``).
    """

    arm: str
    cell: str
    step: int
    section: str
    game_id: str
    prompt_id: str
    sample_index: int
    label_print_order: str
    payoff_variant: str
    label_a: str
    label_b: str
    coop_label: str
    decision: str
    counterpart_framing: str | None
    reskin_id: str | None
    trained_arm: str
    completion: str
    visible_text: str
    framings_digest: str = ""

    @property
    def key(self) -> str:
        """The record's identity across the census: where it came from plus the battery's own fields."""
        return (
            f"{self.arm}|{self.cell}|{self.step}|{self.prompt_id}|{self.sample_index}"
            f"|{self.label_print_order}"
        )

    @property
    def judgeable(self) -> bool:
        """Whether the record carries any text a judge could read."""
        return bool(self.completion.strip())

    @property
    def completion_sha256(self) -> str:
        """Digest of the completion, the arm-agnostic handle the sample overlays match on."""
        return hashlib.sha256(self.completion.encode("utf-8")).hexdigest()

    @property
    def handle(self) -> Handle:
        """(step, prompt_id, sample_index, digest): the identity shared with the sample exports."""
        return (self.step, self.prompt_id, self.sample_index, self.completion_sha256)

    @property
    def labels_as_read(self) -> tuple[str, str]:
        """The two labels in the order the prompt printed them: authored order, reversed when swapped."""
        return print_order_of((self.label_a, self.label_b), self.label_print_order)


@dataclass(frozen=True, slots=True)
class Selection:
    """Which rows of a cell are in scope; ``None`` on an axis means no filter on it.

    ``framings`` reads only rows that carry a ``counterpart_framing`` (the framing sweep); the other
    sections have no framing and pass through, so one selection can name the framing-sweep framings
    and still admit the game-behaviour and training-frames rows beside them.

    ``decision`` names which parsed records to keep: ``cooperate`` (every sampled action cooperative;
    the default, and the only selection before 2026-09-03), ``defect`` (every sampled action
    non-cooperative) or ``all`` (every parsed record, each stamped with its own decision).
    ``per_render`` keeps at most that many records per (prompt_id, label_print_order) render, the
    lowest sample indices, so a sample derives from record identity rather than file order or a random
    draw and every render of the cell stays represented; ``None`` is the whole population.
    """

    sections: frozenset[str] | None = None
    games: frozenset[str] | None = None
    framings: frozenset[str] | None = None
    decision: str = DECISION_COOPERATE
    per_render: int | None = None

    def __post_init__(self) -> None:
        """Refuse a decision the CLI does not offer, or a per-render sample of nothing."""
        if self.decision not in DECISION_CHOICES:
            raise ValueError(f"decision must be one of {DECISION_CHOICES}, got {self.decision!r}")
        if self.per_render is not None and self.per_render < 1:
            raise ValueError(
                f"per_render must be at least 1, or None for the whole population; got {self.per_render}"
            )

    def admits(self, row: Mapping[str, Any]) -> bool:
        """Whether one battery row passes the section, game and framing axes of the selection."""
        if self.sections is not None and str(row["record"]) not in self.sections:
            return False
        if self.games is not None and str(row["game_id"]) not in self.games:
            return False
        framing = row.get("counterpart_framing")
        return not (
            self.framings is not None and framing is not None and str(framing) not in self.framings
        )

    def admits_decision(self, decision: str) -> bool:
        """Whether a parsed record that took ``decision`` is in scope."""
        return self.decision in (DECISION_ALL, decision)

    def sample_per_render(self, records: Sequence[TraceRecord]) -> list[TraceRecord]:
        """Keep the ``per_render`` lowest sample indices of every (prompt_id, print order) render.

        Ordered by record identity -- prompt id, print order, then the sample index as the integer it
        is, so index 2 precedes 10 -- and never by file order or a random draw: the same tree and the
        same arguments select the same records on every relaunch, which is what lets a sampled run
        resume. Renders come out sorted, so a sampled run's record order is not the file's. Under
        ``all`` the lowest indices are kept whatever decision they took.
        """
        if self.per_render is None:
            return list(records)
        by_render: defaultdict[tuple[str, str], list[TraceRecord]] = defaultdict(list)
        for record in records:
            by_render[(record.prompt_id, record.label_print_order)].append(record)
        return [
            record
            for _, group in sorted(by_render.items())
            for record in sorted(group, key=lambda r: r.sample_index)[: self.per_render]
        ]

    def as_manifest(self) -> dict[str, Any]:
        """Render the selection as JSON-friendly sorted lists plus the two selectors, for the manifest."""
        return {
            "sections": sorted(self.sections) if self.sections is not None else None,
            "games": sorted(self.games) if self.games is not None else None,
            "framings": sorted(self.framings) if self.framings is not None else None,
            "decision": self.decision,
            "per_render": self.per_render,
        }


def record_decision(row: Mapping[str, Any]) -> str:
    """Read the decision a parsed one-shot matrix-game row took off its ``coop_fraction``.

    The battery scores a one-shot matrix action as ``float(action == COOPERATE)``, so a parsed row is
    exactly 1.0 or 0.0 (the 24 wave-3 cell files hold nothing else). Anything else is a scoring change
    upstream and refuses, rather than being filed under a decision the record never took.
    """
    fraction = row[COOP_FIELD]
    if fraction == 1.0:
        return DECISION_COOPERATE
    if fraction == 0.0:
        return DECISION_DEFECT
    raise ValueError(
        f"row {row.get('prompt_id')!r} sample {row.get('sample_index')!r} has {COOP_FIELD}="
        f"{fraction!r}, neither 1.0 nor 0.0; a parsed one-shot matrix-game action is one or the other"
    )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _read_cell_file(path: Path, step: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the cell's meta row and its data rows, checking the meta step against the file name."""
    if not path.exists():
        raise FileNotFoundError(f"no banked cell file at {path}")
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["record"] == META_RECORD:
                meta = row
            else:
                rows.append(row)
    if meta is None:
        raise ValueError(f"{path} carries no meta row, so its arm and step cannot be verified")
    if int(meta["step"]) != step:
        raise ValueError(f"{path} is named step-{step} but its meta says step={meta['step']}")
    return meta, rows


def _framings_digest_of(meta: Mapping[str, Any]) -> str:
    """Read the runtime framings file's digest off a cell's meta row, empty where it recorded none.

    Two spellings of absence, and both mean the same thing: a cell banked before ``--framings-file``
    existed carries no ``eval_config`` section for it at all, and one run without the flag records
    the field as an empty string (``games.evals.EvalConfig.as_record``). Either way every framing that
    cell rendered came out of ``games.prompts.COUNTERPART_FRAMINGS``.
    """
    eval_config: Mapping[str, Any] = meta.get("eval_config", {})
    return str(eval_config.get("framings_digest", ""))


def _record_from_row(
    row: Mapping[str, Any], meta: Mapping[str, Any], *, arm: str, cell: str, step: int
) -> TraceRecord:
    return TraceRecord(
        arm=arm,
        cell=cell,
        step=step,
        section=str(row["record"]),
        game_id=str(row["game_id"]),
        prompt_id=str(row["prompt_id"]),
        sample_index=int(row["sample_index"]),
        label_print_order=str(row["label_print_order"]),
        payoff_variant=str(row["payoff_variant"]),
        label_a=str(row["label_a"]),
        label_b=str(row["label_b"]),
        coop_label=str(row["coop_label"]),
        decision=record_decision(row),
        counterpart_framing=_optional_str(row.get("counterpart_framing")),
        reskin_id=_optional_str(row.get("reskin_id")),
        trained_arm=str(meta["arm"]),
        completion=str(row["completion"]),
        visible_text=str(row["visible_text"]),
        framings_digest=_framings_digest_of(meta),
    )


def load_cell_records(
    tree: Path, arm_dir: str, cell: str, step: int, selection: Selection
) -> tuple[list[TraceRecord], dict[str, int]]:
    """Load one banked cell's in-scope matrix-game records of the selection's decision, with the funnel.

    The cell lives at ``<tree>/<arm_dir>/evals/<cell>/step-<step>.jsonl``; the arm label is
    ``arm_dir``'s basename. The file's ``meta`` row must agree with the step in its name, as the
    scratch harnesses require. Counts: ``rows`` (every non-meta row), ``matrix_game_rows`` (one-shot
    2x2 games the judge can re-render), ``in_scope_rows`` (after the section, game and framing axes),
    ``parsed``, ``cooperating`` and ``defecting`` (the parsed rows by the decision each took),
    ``selected`` (those of the selection's decision) and ``after_per_render`` (what the per-render
    sample kept; equal to ``selected`` without sampling).
    """
    meta, rows = _read_cell_file(tree / arm_dir / "evals" / cell / f"step-{step}.jsonl", step)
    arm = Path(arm_dir).name
    counts = {
        "rows": len(rows),
        "matrix_game_rows": 0,
        "in_scope_rows": 0,
        "parsed": 0,
        "cooperating": 0,
        "defecting": 0,
        "selected": 0,
        "after_per_render": 0,
    }
    records: list[TraceRecord] = []
    for row in rows:
        if row["record"] not in SECTIONS_WITH_TRACES or not is_matrix_game(str(row["game_id"])):
            continue
        counts["matrix_game_rows"] += 1
        if not selection.admits(row):
            continue
        counts["in_scope_rows"] += 1
        if not bool(row["parsed"]):
            continue
        counts["parsed"] += 1
        record = _record_from_row(row, meta, arm=arm, cell=cell, step=step)
        counts[_FUNNEL_COUNT_OF_DECISION[record.decision]] += 1
        if not selection.admits_decision(record.decision):
            continue
        counts["selected"] += 1
        records.append(record)
    records = selection.sample_per_render(records)
    counts["after_per_render"] = len(records)
    logger.info(
        "loaded %s/%s step-%d (decision=%s, per_render=%s): %s",
        arm,
        cell,
        step,
        selection.decision,
        selection.per_render,
        json.dumps(counts),
    )
    return records, counts
