"""The twin-corpus sidecar contract: set and side names, span names, the per-stimulus record, its reader.

`reward_hacking.interp.tmax_twin_corpus` writes the sidecar and `reward_hacking.interp.tmax_capture_ladder`
reads it on the capture box; `tmax_directions` and `tmax_displacement_series` read the set and side names
off the cells the ladder wrote. The writer module imports the harness task registry, which reads a
local-only case file at import time and raises without it, so the capture box cannot import the writer
and every key it spells was spelled a second time in the ladder. This module is the one spelling. Both
sides import it, the writer serialises through :class:`SidecarStimulus.to_dict` and the reader parses
through :meth:`SidecarStimulus.from_dict`, so a renamed or dropped key is a refusal here rather than a
silent default downstream.

What the sidecar is: one JSON object with the corpus identity (schema, kind, base model and revision,
render convention, set name, the six side names, the stimulus-file digest) and one record per problem
holding its id, the perturbation facts a reader may see (`detectable` among them) and one entry per
side with the token ids the corpus was tokenized to, the named `[start, end)` spans over them, and the
spans that rendering lacks with the reason for each. The prompt text lives in the stimulus file only.

The six renderings and the spans are described in the writer's docstring; the names are here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from games.interp_cells import STIMULUS_RENDER_VERBATIM

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_SCHEMA = 1
CORPUS_KIND = "tmax-twin-corpus"
STIMULUS_SET = "tmax-grader-twins"
STIMULI_FILENAME = "stimuli.jsonl"
SIDECAR_FILENAME = "twin-corpus.json"

STIMULUS_RENDER = STIMULUS_RENDER_VERBATIM
"""The twin stimuli carry the chat template inside `text`, so the ladder captures them `verbatim`."""

WINDOW_TOKENS = 32

SIDE_RIGGED = "rigged-inline"
SIDE_HONEST = "honest-inline"
SIDE_NO_GRADER = "no-grader"
SIDE_COMMENT_CONTROL = "comment-control"
SIDE_STATEMENT_LITERAL = "statement-literal-control"
SIDE_CONFLICTING = "conflicting"
SIDES: tuple[str, ...] = (
    SIDE_RIGGED,
    SIDE_HONEST,
    SIDE_NO_GRADER,
    SIDE_COMMENT_CONTROL,
    SIDE_STATEMENT_LITERAL,
    SIDE_CONFLICTING,
)
"""The six renderings, one stimulus each per problem; `pair_id` is the problem id.

Six sides per pair is why no reader may use `games.interp_cells.pair_layout` on this set: that layout
requires exactly two sides. Rigged-versus-honest selection is `tmax_directions.two_side_layout`.
"""

SPAN_HEADER = "header"
SPAN_STATEMENT = "statement"
SPAN_SCORING = "scoring"
SPAN_GRADER = "grader"
SPAN_CONTRACT = "contract"
SPAN_TAIL = "tail"
SPAN_ASSERTION32 = "assertion32"
SPAN_GRADER_END32 = "grader_end32"
SPAN_STATEMENT32 = "statement32"
SPAN_CONFLICT32 = "conflict32"
GRADER_SPANS: tuple[str, ...] = (SPAN_GRADER, SPAN_GRADER_END32, SPAN_ASSERTION32)
"""The spans only a grader-bearing rendering can carry; absent by declaration under the withheld one."""

# Top-level sidecar keys.
SIDECAR_SCHEMA_KEY = "schema"
SIDECAR_KIND_KEY = "kind"
SIDECAR_SET_KEY = "stimulus_set"
SIDECAR_SIDES_KEY = "sides"
SIDECAR_RENDER_KEY = "stimulus_render"
SIDECAR_WINDOW_TOKENS_KEY = "window_tokens"
SIDECAR_DIGEST_KEY = "stimuli_sha256"
SIDECAR_PROBLEMS_KEY = "problems"
SIDECAR_REQUIRED_KEYS: tuple[str, ...] = (
    SIDECAR_SCHEMA_KEY,
    SIDECAR_KIND_KEY,
    SIDECAR_SET_KEY,
    SIDECAR_RENDER_KEY,
    SIDECAR_DIGEST_KEY,
    SIDECAR_PROBLEMS_KEY,
)

# Per-problem keys.
PROBLEM_ID_KEY = "problem_id"
PROBLEM_PERTURBATION_KEY = "perturbation"
PROBLEM_DETECTABLE_KEY = "detectable"
PROBLEM_STIMULI_KEY = "stimuli"

# Per-stimulus entry keys.
ENTRY_STIMULUS_ID_KEY = "stimulus_id"
ENTRY_SIDE_KEY = "side"
ENTRY_N_TOKENS_KEY = "n_tokens"
ENTRY_INPUT_IDS_KEY = "input_ids"
ENTRY_SPANS_KEY = "spans"
ENTRY_SPANS_ABSENT_KEY = "spans_absent"
ENTRY_KEYS: tuple[str, ...] = (
    ENTRY_STIMULUS_ID_KEY,
    ENTRY_SIDE_KEY,
    ENTRY_N_TOKENS_KEY,
    ENTRY_INPUT_IDS_KEY,
    ENTRY_SPANS_KEY,
    ENTRY_SPANS_ABSENT_KEY,
)

ERROR_EXAMPLE_COUNT = 5


class TwinSidecarError(ValueError):
    """The sidecar is not one this contract describes, so its spans cannot be trusted to index anything."""


# --------------------------------------------------------------------------------------
# The per-stimulus record: written by the corpus, parsed by the capture
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SidecarStimulus:
    """One rendering's sidecar entry: its ids, the spans it declares and the spans it lacks, with reasons."""

    stimulus_id: str
    side: str
    input_ids: tuple[int, ...]
    spans: dict[str, tuple[int, int]]
    spans_absent: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Serialise for the sidecar; the prompt text is deliberately not here."""
        return {
            ENTRY_STIMULUS_ID_KEY: self.stimulus_id,
            ENTRY_SIDE_KEY: self.side,
            ENTRY_N_TOKENS_KEY: len(self.input_ids),
            ENTRY_INPUT_IDS_KEY: list(self.input_ids),
            ENTRY_SPANS_KEY: {name: [start, end] for name, (start, end) in self.spans.items()},
            ENTRY_SPANS_ABSENT_KEY: dict(self.spans_absent),
        }

    @classmethod
    def from_dict(cls, entry: Mapping[str, Any], *, source: str) -> SidecarStimulus:
        """Parse one entry, refusing a missing key, a side outside the six, or a span that is not a window.

        `spans_absent` is required rather than defaulted: a rendering that lacks a span says so by
        name and reason, and an entry without the field is one the writer did not produce, so its
        missing spans would otherwise be recorded downstream as "not declared" without anyone knowing
        whether the corpus meant that.
        """
        missing = [key for key in ENTRY_KEYS if key not in entry]
        if missing:
            raise TwinSidecarError(
                f"{source}: the stimulus entry lacks {missing}; every entry the corpus writes carries "
                f"{list(ENTRY_KEYS)}, so this sidecar was not written by it or a key was renamed"
            )
        stimulus_id = str(entry[ENTRY_STIMULUS_ID_KEY])
        side = str(entry[ENTRY_SIDE_KEY])
        if side not in SIDES:
            raise TwinSidecarError(
                f"{source}: stimulus {stimulus_id!r} is on side {side!r}, not one of {list(SIDES)}"
            )
        ids = tuple(int(value) for value in cast("Sequence[int]", entry[ENTRY_INPUT_IDS_KEY]))
        n_tokens = int(cast("int", entry[ENTRY_N_TOKENS_KEY]))
        if n_tokens != len(ids):
            raise TwinSidecarError(
                f"{source}: stimulus {stimulus_id!r} records n_tokens={n_tokens} over "
                f"{len(ids)} input ids"
            )
        spans: dict[str, tuple[int, int]] = {}
        for name, bounds in cast("Mapping[str, Sequence[int]]", entry[ENTRY_SPANS_KEY]).items():
            if len(bounds) != 2:  # noqa: PLR2004 - a span is exactly [start, end]
                raise TwinSidecarError(
                    f"{source}: stimulus {stimulus_id!r} span {name!r} is {list(bounds)}, not [start, end]"
                )
            start, end = int(bounds[0]), int(bounds[1])
            if not 0 <= start < end <= len(ids):
                raise TwinSidecarError(
                    f"{source}: stimulus {stimulus_id!r} span {name!r} is [{start}, {end}) over "
                    f"{len(ids)} tokens; a span is a non-empty window inside the prompt"
                )
            spans[str(name)] = (start, end)
        absent = {
            str(name): str(reason)
            for name, reason in cast("Mapping[str, object]", entry[ENTRY_SPANS_ABSENT_KEY]).items()
        }
        both = sorted(set(spans) & set(absent))
        if both:
            raise TwinSidecarError(
                f"{source}: stimulus {stimulus_id!r} both declares and declares absent {both}"
            )
        return cls(
            stimulus_id=stimulus_id, side=side, input_ids=ids, spans=spans, spans_absent=absent
        )


def sidecar_payload(
    *, stimuli_sha256: str, problems: Sequence[Mapping[str, object]], **corpus_fields: object
) -> dict[str, object]:
    """Assemble the sidecar's top level: the contract's fixed keys, the corpus's own fields, the problems.

    The writer passes its identity and accounting fields as keywords; a field that would shadow one of
    the contract's keys is refused, so the set name, the sides and the render convention are only ever
    the values spelled in this module.
    """
    contract: dict[str, object] = {
        SIDECAR_SCHEMA_KEY: CORPUS_SCHEMA,
        SIDECAR_KIND_KEY: CORPUS_KIND,
        SIDECAR_RENDER_KEY: STIMULUS_RENDER,
        SIDECAR_SET_KEY: STIMULUS_SET,
        SIDECAR_SIDES_KEY: list(SIDES),
        SIDECAR_WINDOW_TOKENS_KEY: WINDOW_TOKENS,
    }
    clash = sorted(
        set(corpus_fields) & (set(contract) | {SIDECAR_DIGEST_KEY, SIDECAR_PROBLEMS_KEY})
    )
    if clash:
        raise TwinSidecarError(
            f"corpus fields {clash} would shadow the sidecar contract's own keys"
        )
    return {
        **contract,
        **corpus_fields,
        SIDECAR_DIGEST_KEY: stimuli_sha256,
        SIDECAR_PROBLEMS_KEY: list(problems),
    }


# --------------------------------------------------------------------------------------
# The reader: what the capture driver needs of the sidecar
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanTable:
    """Per stimulus: the ids it was tokenized to, its named `[start, end)` spans, and the spans it lacks."""

    input_ids: dict[str, tuple[int, ...]]
    spans: dict[str, dict[str, tuple[int, int]]]
    spans_absent: dict[str, dict[str, str]]
    stimuli_sha256: str
    stimulus_render: str

    @property
    def span_names(self) -> tuple[str, ...]:
        """Every span name any stimulus declares, in a stable order."""
        return tuple(sorted({name for spans in self.spans.values() for name in spans}))


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    missing = [key for key in SIDECAR_REQUIRED_KEYS if key not in sidecar]
    if missing:
        raise TwinSidecarError(
            f"{path} lacks {missing}; a twin-corpus sidecar carries {list(SIDECAR_REQUIRED_KEYS)}"
        )
    if (
        sidecar[SIDECAR_KIND_KEY] != CORPUS_KIND
        or int(sidecar[SIDECAR_SCHEMA_KEY]) != CORPUS_SCHEMA
    ):
        raise TwinSidecarError(
            f"{path} is kind {sidecar[SIDECAR_KIND_KEY]!r} schema {sidecar[SIDECAR_SCHEMA_KEY]!r}, not "
            f"{CORPUS_KIND!r} schema {CORPUS_SCHEMA}"
        )
    if sidecar[SIDECAR_SET_KEY] != STIMULUS_SET:
        raise TwinSidecarError(
            f"{path} names stimulus set {sidecar[SIDECAR_SET_KEY]!r}; this contract's set is {STIMULUS_SET!r}"
        )
    return sidecar


def _problem_entries(
    sidecar: Mapping[str, Any], *, path: Path
) -> list[tuple[str, str, dict[str, Any]]]:
    """Yield (problem id, side key, entry) for every stimulus entry, refusing a mislabelled side."""
    entries: list[tuple[str, str, dict[str, Any]]] = []
    for problem in cast("list[dict[str, Any]]", sidecar[SIDECAR_PROBLEMS_KEY]):
        if PROBLEM_ID_KEY not in problem or PROBLEM_STIMULI_KEY not in problem:
            raise TwinSidecarError(
                f"{path}: a problem record lacks {PROBLEM_ID_KEY!r} or {PROBLEM_STIMULI_KEY!r} "
                f"(has {sorted(problem)})"
            )
        problem_id = str(problem[PROBLEM_ID_KEY])
        for side_key, entry in cast(
            "dict[str, dict[str, Any]]", problem[PROBLEM_STIMULI_KEY]
        ).items():
            entries.append((problem_id, str(side_key), entry))
    return entries


def load_span_sidecar(path: Path) -> SpanTable:
    """Read the sidecar into the table the capture indexes spans from, refusing anything off-contract."""
    sidecar = _read_sidecar(path)
    input_ids: dict[str, tuple[int, ...]] = {}
    spans: dict[str, dict[str, tuple[int, int]]] = {}
    spans_absent: dict[str, dict[str, str]] = {}
    for problem_id, side_key, entry in _problem_entries(sidecar, path=path):
        record = SidecarStimulus.from_dict(entry, source=f"{path}:{problem_id}/{side_key}")
        if record.side != side_key:
            raise TwinSidecarError(
                f"{path}: problem {problem_id!r} files stimulus {record.stimulus_id!r} under side "
                f"{side_key!r} but the entry says side {record.side!r}"
            )
        if record.stimulus_id in input_ids:
            raise TwinSidecarError(f"{path}: stimulus {record.stimulus_id!r} appears twice")
        input_ids[record.stimulus_id] = record.input_ids
        spans[record.stimulus_id] = record.spans
        spans_absent[record.stimulus_id] = record.spans_absent
    if not spans:
        raise TwinSidecarError(f"{path} declares no stimuli")
    table = SpanTable(
        input_ids=input_ids,
        spans=spans,
        spans_absent=spans_absent,
        stimuli_sha256=str(sidecar[SIDECAR_DIGEST_KEY]),
        stimulus_render=str(sidecar[SIDECAR_RENDER_KEY]),
    )
    logger.info(
        f"span sidecar loaded, {path=} n_stimuli={len(spans)} spans={table.span_names} "
        f"render={table.stimulus_render!r}"
    )
    return table


def detectable_by_problem(path: Path) -> dict[str, bool]:
    """Read the per-problem `detectable` flag: whether the statement's worked example exposes the rigging."""
    sidecar = _read_sidecar(path)
    flags: dict[str, bool] = {}
    for problem in cast("list[dict[str, Any]]", sidecar[SIDECAR_PROBLEMS_KEY]):
        perturbation = cast("dict[str, Any]", problem.get(PROBLEM_PERTURBATION_KEY, {}))
        if PROBLEM_DETECTABLE_KEY not in perturbation:
            raise TwinSidecarError(
                f"{path}: problem {problem.get(PROBLEM_ID_KEY)!r} records no "
                f"{PROBLEM_PERTURBATION_KEY}.{PROBLEM_DETECTABLE_KEY}"
            )
        flags[str(problem[PROBLEM_ID_KEY])] = bool(perturbation[PROBLEM_DETECTABLE_KEY])
    return flags
