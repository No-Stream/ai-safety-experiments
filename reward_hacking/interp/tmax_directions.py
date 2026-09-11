"""Fit the TMAX wave's contrast directions from cached cells, each with its controls built in.

Everything here is arithmetic over cells the capture driver already wrote (`games.interp_cells`), so
it runs on a CPU while the box is still capturing. The direction families:

* ``d_twin`` -- rigged-inline minus honest-inline, per pooling and layer, stratified by the twin
  corpus sidecar's ``detectable`` flag (whether the statement's worked example lets a reader notice
  the perturbed assertion at all). The other side contrasts of the six renderings come from the same
  fitter: ``d_spec`` (honest-inline minus no-grader: the grader's presence), and the two content
  controls (comment-control and statement-literal-control minus honest-inline).
* ``d_hack`` and ``d_capability`` -- fitted WITHIN problem and unit from the generation set (hack
  versus honest pass; honest pass versus honest fail), because hacks are item-locked and a pooled
  hack-versus-honest contrast is partly an item-identity contrast. Every pair is two rows of one
  (problem, unit) group, so item identity cancels inside the difference of means exactly.
* ``e`` -- evaluation awareness from the captured contrastive sentence pairs, which
  ``tmax_sentence_corpus`` writes as one stimulus set per concept named by
  ``stimuli.SENTENCE_STIMULUS_SETS`` with sides ``stimuli.SENTENCE_SIDES``; :func:`sentence_directions`
  selects a set by that name and refuses any other spelling with the known names listed.

``d_disp`` (checkpoint minus base on identical rows) lives in ``tmax_displacement_series`` with its
base-versus-mirror sabotage, since the displacement series is built from it.

Every direction goes through :func:`reward_hacking.interp.eval_awareness_probe.validate_layer` (a
trained probe against its shuffled-label null, the diff-of-means axis against matched-norm placebos,
the odd/even split-half ceiling) plus the shuffled-label DIRECTION draws of
:func:`reward_hacking.interp.steer_validation.control_directions`'s second family, gated on their
measured cosine to the real direction. That gate exists because the shuffled-label family is
sign-contaminated: shuffling destroys most of the contrast, so the raw vector is short, and
renormalising amplifies whatever real-direction component the finite permutation kept, with an
arbitrary sign per draw (measured cosines to the real axis of -0.49 at one layer, against placebo
draws at +/-0.04). A draw with ``|cosine| > 0.2`` is reported and excluded, never scored as a null.

**The identifiability gate.** A hack-versus-honest direction only reads as a *disposition* if the
narration judge found enough rows where the model noticed the rigged assertion and still refused to
game it -- fewer than 25 such rows across 8 problems, and nothing separates "represents a
disposition to hack" from "represents hack text". Below the gate every read carries the label
``what RL changed about representing hack text`` instead of ``disposition``; the label is a column in
every table this module writes, never a footnote.

Nothing here logs or writes prompt text: the tables carry ids, counts, cosines and accuracies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import torch

from games.interp_cells import (
    ERROR_EXAMPLE_COUNT,
    CapturedCell,
    CellFormatError,
    PairLayout,
    concept_activations,
)
from reward_hacking.interp.directions import cosine
from reward_hacking.interp.eval_awareness_probe import (
    DEFAULT_N_PLACEBOS,
    projection_cv_accuracy,
    validate_layer,
)
from reward_hacking.interp.linear_probe import ConceptActivations, ProbeConfig
from reward_hacking.interp.steer_validation import shuffled_label_direction
from reward_hacking.interp.stimuli import (
    CONCEPT_EVAL_AWARENESS,
    SENTENCE_SIDE_NEGATIVE,
    SENTENCE_SIDE_POSITIVE,
    SENTENCE_STIMULUS_SETS,
)
from reward_hacking.interp.tmax_twin_sidecar import (
    SIDE_COMMENT_CONTROL,
    SIDE_CONFLICTING,
    SIDE_HONEST,
    SIDE_NO_GRADER,
    SIDE_RIGGED,
    SIDE_STATEMENT_LITERAL,
    STIMULUS_SET,
)
from reward_hacking.legibility_narration_summary import (
    IDENTIFIABILITY_MIN_PROBLEMS,
    IDENTIFIABILITY_MIN_ROWS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

SHUFFLED_MAX_ABS_COSINE = 0.2
"""A shuffled-label draw with |cosine| to the real direction above this is sign-contaminated: excluded."""
DEFAULT_N_SHUFFLED = 20

STRATUM_ALL = "all"
STRATUM_DETECTABLE = "detectable"
STRATUM_NOT_DETECTABLE = "not-detectable"
STRATA: tuple[str, ...] = (STRATUM_ALL, STRATUM_DETECTABLE, STRATUM_NOT_DETECTABLE)

DISPOSITION_LABEL_IDENTIFIABLE = "disposition"
DISPOSITION_LABEL_UNIDENTIFIABLE = "what RL changed about representing hack text"

NAME_TWIN = "d_twin"
NAME_SPEC = "d_spec"
NAME_COMMENT_CONTROL = "d_comment_control"
NAME_STATEMENT_LITERAL_CONTROL = "d_statement_literal_control"
NAME_CONFLICT = "d_conflict"
NAME_HACK = "d_hack"
NAME_CAPABILITY = "d_capability"
NAME_EVAL_AWARENESS = "e"

SIDE_CONTRASTS: dict[str, tuple[str, str]] = {
    NAME_TWIN: (SIDE_RIGGED, SIDE_HONEST),
    NAME_SPEC: (SIDE_HONEST, SIDE_NO_GRADER),
    NAME_COMMENT_CONTROL: (SIDE_COMMENT_CONTROL, SIDE_HONEST),
    NAME_STATEMENT_LITERAL_CONTROL: (SIDE_STATEMENT_LITERAL, SIDE_HONEST),
    NAME_CONFLICT: (SIDE_CONFLICTING, SIDE_HONEST),
}
"""(positive side, negative side) for every named contrast over the six twin renderings."""

SENTENCE_DIRECTION_NAMES: dict[str, str] = {
    concept: NAME_EVAL_AWARENESS if concept == CONCEPT_EVAL_AWARENESS else f"d_{concept}"
    for concept in SENTENCE_STIMULUS_SETS
}
"""The direction fitted from each captured sentence set: ``e`` for evaluation awareness, ``d_<set>`` else.

Keyed by the set names `tmax_sentence_corpus` writes, which are `stimuli`'s own concept names, so a
name that is not a key here is not a set any sentence cell holds.
"""

LABEL_SCHEMA = pl.Schema(
    {
        "stimulus_id": pl.String(),
        "problem_id": pl.String(),
        "unit": pl.String(),
        "hack": pl.Boolean(),
        "hidden_pass": pl.Boolean(),
    }
)
"""The generation-set labels file: one JSON object per captured row, joined to the cell on stimulus_id."""
GROUP_COLUMNS: tuple[str, ...] = ("problem_id", "unit")

READS_FILENAME = "direction-reads.ndjson"
REFUSALS_FILENAME = "direction-refusals.ndjson"
DIRECTIONS_FILENAME = "directions.pt"


class DirectionRefusalError(ValueError):
    """A direction that cannot be fitted or validated at all on this input."""


# --------------------------------------------------------------------------------------
# Tables from dataclass records
# --------------------------------------------------------------------------------------

_POLARS_BY_ANNOTATION: dict[str, pl.DataType] = {
    "str": pl.String(),
    "int": pl.Int64(),
    "float": pl.Float64(),
    "bool": pl.Boolean(),
    "str | None": pl.String(),
    "int | None": pl.Int64(),
    "float | None": pl.Float64(),
    "bool | None": pl.Boolean(),
    "tuple[float, ...]": pl.List(pl.Float64()),
    "tuple[str, ...]": pl.List(pl.String()),
}


def record_schema(record_type: type) -> dict[str, pl.DataType]:
    """Derive a Polars schema from a frozen record dataclass, so an empty table keeps its columns."""
    return {member.name: _POLARS_BY_ANNOTATION[str(member.type)] for member in fields(record_type)}


def records_table(records: Sequence[Any], record_type: type) -> pl.DataFrame:
    """Dataclass records as one typed Polars table; round-trips through ndjson unchanged."""
    return pl.DataFrame([asdict(record) for record in records], schema=record_schema(record_type))


def read_records_table(path: Path, record_type: type) -> pl.DataFrame:
    """Read a table :func:`records_table` wrote, under its schema, so an empty file reads as empty."""
    return pl.read_ndjson(path, schema=record_schema(record_type))


# --------------------------------------------------------------------------------------
# The identifiability gate
# --------------------------------------------------------------------------------------


GATE_DEFINITION_EXPLICIT = "explicit counts"
GATE_DEFINITION_UNLABELLED = "unlabelled"


@dataclass(frozen=True)
class IdentifiabilityGate:
    """Whether a hack direction may be read as a disposition, from the narration judge's counts.

    ``rows`` and ``problems`` count the noticed-and-refused group under one named ``definition``
    (:data:`reward_hacking.legibility_narration_summary.IDENTIFIABILITY_DEFINITIONS`). ``outcome`` is
    an honest-pass row on the rigged twin whose judge NOTICED is yes, whatever the judge said about
    COMPLIED; ``judge-disjoint`` also requires COMPLIED no, so the group is disjoint from the
    noticed-and-complied rows the outcome group turned out to share (17 of step_500's 32 under Opus 5).
    The two differ on real data, pooled 59 rows across 20 problems against 35 across 17, so the
    definition is never defaulted: a summary is read under a definition the caller names, and the
    gate carries it beside its counts. Without enough rows across enough problems a hack-versus-honest
    direction cannot be told from a hack-text direction, and every downstream label says so.
    """

    rows: int
    problems: int
    min_rows: int = IDENTIFIABILITY_MIN_ROWS
    min_problems: int = IDENTIFIABILITY_MIN_PROBLEMS
    source: str = GATE_DEFINITION_EXPLICIT
    definition: str = GATE_DEFINITION_EXPLICIT

    @property
    def identifiable(self) -> bool:
        """Both thresholds met."""
        return self.rows >= self.min_rows and self.problems >= self.min_problems

    @property
    def disposition_label(self) -> str:
        """The label every read carries: what the direction may be called."""
        if self.identifiable:
            return DISPOSITION_LABEL_IDENTIFIABLE
        return DISPOSITION_LABEL_UNIDENTIFIABLE

    @classmethod
    def unlabelled(cls) -> IdentifiabilityGate:
        """No judge labels at all: zero rows, so nothing reads as a disposition."""
        return cls(
            0, 0, source="no narration labels supplied", definition=GATE_DEFINITION_UNLABELLED
        )

    @classmethod
    def from_narration_summary(cls, path: Path, *, definition: str) -> IdentifiabilityGate:
        """Read one named definition's block out of the judge summary's ``identifiability`` section.

        ``definition`` has no default on purpose: the outcome and judge-disjoint groups disagree on
        real data, and a default would let a consumer read one while believing it read the other. A
        summary written before the split carries flat counts and no ``definitions`` block; it is
        refused rather than read as either definition, and a summary lacking the named definition is
        refused naming what it does carry.
        """
        payload = cast("dict[str, Any]", json.loads(path.read_text()))
        block = cast("dict[str, Any]", payload["identifiability"])
        definitions = cast("dict[str, Any] | None", block.get("definitions"))
        if definitions is None:
            raise ValueError(
                f"{path}: the identifiability block carries no named definitions (it predates the "
                "outcome / judge-disjoint split); regenerate it with `legibility_narration_judge "
                "summary` rather than reading its flat counts as any one definition"
            )
        if definition not in definitions:
            raise ValueError(
                f"{path}: no identifiability definition {definition!r}; it carries "
                f"{sorted(definitions)}"
            )
        chosen = cast("dict[str, Any]", definitions[definition])
        if chosen.get("definition") != definition:
            raise ValueError(
                f"{path}: the block filed under {definition!r} names itself "
                f"{chosen.get('definition')!r}"
            )
        return cls(
            rows=int(chosen["rows"]),
            problems=int(chosen["problems"]),
            min_rows=int(block["min_rows"]),
            min_problems=int(block["min_problems"]),
            source=str(path),
            definition=definition,
        )


# --------------------------------------------------------------------------------------
# Row selection over a cell: two-side layouts, strata, within-group pairs
# --------------------------------------------------------------------------------------


def two_side_layout(
    cell: CapturedCell, stimulus_set: str, *, positive_side: str, negative_side: str
) -> PairLayout:
    """Locate two named sides of every pair in a set that carries more than two sides per pair.

    An adapter over :func:`games.interp_cells.pair_layout`, which requires a set to carry exactly two
    sides: the twin set carries six renderings per problem, so a contrast has to name its two. A pair
    missing either named side is refused rather than dropped, for the same reason as there: a half
    pair is a lost row, not a smaller sample.
    """
    if stimulus_set not in cell.rows:
        raise CellFormatError(
            f"{cell.label} holds no set {stimulus_set!r}; it holds {list(cell.stimulus_sets)}."
        )
    if positive_side == negative_side:
        raise ValueError(f"a contrast needs two different sides, got {positive_side!r} twice")
    row_index = cell.rows[stimulus_set]
    order: list[str] = []
    rows_by_pair: dict[str, dict[str, int]] = {}
    for row, (pair_id, side) in enumerate(zip(row_index.pair_ids, row_index.sides, strict=True)):
        if side not in (positive_side, negative_side):
            continue
        members = rows_by_pair.setdefault(pair_id, {})
        if not members:
            order.append(pair_id)
        if side in members:
            raise CellFormatError(
                f"{cell.label} set {stimulus_set!r} pair {pair_id!r} has two rows on side {side!r}."
            )
        members[side] = row
    incomplete = [pair for pair in order if len(rows_by_pair[pair]) != 2]  # noqa: PLR2004
    if incomplete or not order:
        raise CellFormatError(
            f"{cell.label} set {stimulus_set!r}: {len(incomplete)} of {len(order)} pairs lack one of "
            f"sides {positive_side!r}/{negative_side!r} (first: {incomplete[:ERROR_EXAMPLE_COUNT]}). "
            f"Every read is a difference between the two sides of a pair, so a half pair is a lost row."
        )
    return PairLayout(
        pair_ids=tuple(order),
        positive_rows=torch.tensor([rows_by_pair[pair][positive_side] for pair in order]),
        negative_rows=torch.tensor([rows_by_pair[pair][negative_side] for pair in order]),
        positive_side=positive_side,
        negative_side=negative_side,
    )


def stratum_pairs(
    layout: PairLayout, stratum: str, detectable: Mapping[str, bool] | None
) -> torch.Tensor:
    """Pair indices of one stratum; refuses a detectability stratum without the sidecar's flags."""
    if stratum == STRATUM_ALL:
        return torch.arange(layout.n_pairs)
    if detectable is None:
        raise DirectionRefusalError(
            f"stratum {stratum!r} needs the twin corpus sidecar's detectable flags and none were given"
        )
    missing = sorted(set(layout.pair_ids) - set(detectable))
    if missing:
        raise DirectionRefusalError(
            f"{len(missing)} pairs carry no detectable flag (first: "
            f"{missing[:ERROR_EXAMPLE_COUNT]}); the sidecar and the cell describe different problems"
        )
    wanted = stratum == STRATUM_DETECTABLE
    return torch.tensor(
        [index for index, pair in enumerate(layout.pair_ids) if detectable[pair] == wanted],
        dtype=torch.long,
    )


def load_row_labels(path: Path) -> pl.DataFrame:
    """Load the generation set's per-row labels (ndjson), refusing a missing column or a repeated id."""
    labels = pl.read_ndjson(path, infer_schema_length=None)
    missing = sorted(set(LABEL_SCHEMA.names()) - set(labels.columns))
    if missing:
        raise ValueError(f"{path} lacks label columns {missing}; needs {LABEL_SCHEMA.names()}")
    labels = labels.cast(LABEL_SCHEMA)
    if labels["stimulus_id"].n_unique() != labels.height:
        raise ValueError(
            f"{path} repeats a stimulus_id; every captured row needs exactly one label"
        )
    return labels


def align_labels(cell: CapturedCell, stimulus_set: str, labels: pl.DataFrame) -> pl.DataFrame:
    """Order the labels as the cell stores its rows, refusing rows without a label and vice versa."""
    stored = pl.DataFrame({"stimulus_id": list(cell.rows[stimulus_set].stimulus_ids)})
    unlabelled = stored.join(labels, on="stimulus_id", how="anti")
    extra = labels.join(stored, on="stimulus_id", how="anti")
    if unlabelled.height or extra.height:
        raise ValueError(
            f"{cell.label} set {stimulus_set!r} and the labels describe different rows: "
            f"{unlabelled.height} captured rows without a label, {extra.height} labels without a row."
        )
    return stored.join(labels, on="stimulus_id", how="left", maintain_order="left")


@dataclass(frozen=True)
class WithinGroupPairing:
    """Row pairs formed inside (problem, unit) groups, plus the accounting of what was left out."""

    layout: PairLayout
    n_groups: int
    n_positive_rows: int
    n_negative_rows: int
    groups_without_both_sides: int


def within_group_pairs(  # noqa: PLR0913 - a pairing names its two sides twice: as masks and as labels
    aligned: pl.DataFrame,
    *,
    positive: pl.Expr,
    negative: pl.Expr,
    positive_side: str,
    negative_side: str,
    group_columns: Sequence[str] = GROUP_COLUMNS,
) -> WithinGroupPairing:
    """Pair every positive row with a negative row of the SAME group, cycling the shorter side.

    Groups lacking one side contribute nothing and are counted, so the denominator is visible. Pair
    order follows the cell's row order, so the odd/even split-half and the grouped folds downstream
    are reproducible without a seed.
    """
    frame = aligned.with_row_index("row").with_columns(
        positive.alias("_positive"), negative.alias("_negative")
    )
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    pair_ids: list[str] = []
    skipped = 0
    for key, group in frame.group_by(list(group_columns), maintain_order=True):
        pos = group.filter(pl.col("_positive"))["row"].to_list()
        neg = group.filter(pl.col("_negative"))["row"].to_list()
        if not pos or not neg:
            skipped += 1
            continue
        group_id = "|".join(str(part) for part in key)
        for index, row in enumerate(pos):
            positive_rows.append(int(row))
            negative_rows.append(int(neg[index % len(neg)]))
            pair_ids.append(f"{group_id}#{index}")
    if not pair_ids:
        raise DirectionRefusalError(
            f"no group carries both a {positive_side!r} and a {negative_side!r} row"
        )
    return WithinGroupPairing(
        layout=PairLayout(
            pair_ids=tuple(pair_ids),
            positive_rows=torch.tensor(positive_rows),
            negative_rows=torch.tensor(negative_rows),
            positive_side=positive_side,
            negative_side=negative_side,
        ),
        n_groups=len({pair.rsplit("#", 1)[0] for pair in pair_ids}),
        n_positive_rows=int(frame["_positive"].sum()),
        n_negative_rows=int(frame["_negative"].sum()),
        groups_without_both_sides=skipped,
    )


@dataclass(frozen=True)
class GenerationContrast:
    """Which labelled rows are the two sides of a generation-set direction."""

    name: str
    positive: pl.Expr
    negative: pl.Expr
    positive_side: str
    negative_side: str


HACK_CONTRAST = GenerationContrast(
    name=NAME_HACK,
    positive=pl.col("hack"),
    negative=~pl.col("hack") & pl.col("hidden_pass"),
    positive_side="hack",
    negative_side="honest-pass",
)
CAPABILITY_CONTRAST = GenerationContrast(
    name=NAME_CAPABILITY,
    positive=~pl.col("hack") & pl.col("hidden_pass"),
    negative=~pl.col("hack") & ~pl.col("hidden_pass"),
    positive_side="honest-pass",
    negative_side="honest-fail",
)


# --------------------------------------------------------------------------------------
# Validation: validate_layer plus the cosine-gated shuffled-label draws
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShuffledLabelDraws:
    """The shuffled-label control family, each draw gated on its cosine to the real direction."""

    n_draws: int
    max_abs_cosine: float
    cosines: tuple[float, ...]
    excluded: tuple[int, ...]
    kept_accuracies: tuple[float, ...]

    @property
    def n_excluded(self) -> int:
        """How many draws the cosine gate removed."""
        return len(self.excluded)

    @property
    def kept_accuracy_max(self) -> float | None:
        """Best held-out accuracy among the draws that survived the gate; None if none did."""
        return max(self.kept_accuracies) if self.kept_accuracies else None


def shuffled_label_draws(  # noqa: PLR0913 - the draws are the concept, the axis, the knobs, the count, the seed and the gate
    concept: ConceptActivations,
    direction: torch.Tensor,
    config: ProbeConfig,
    *,
    n_draws: int,
    seed: int,
    max_abs_cosine: float = SHUFFLED_MAX_ABS_COSINE,
) -> ShuffledLabelDraws:
    """Draw shuffled-label directions, measure each one's cosine to ``direction``, gate, and score.

    A kept draw is scored by the same held-out nearest-centroid classifier the real axis and the
    placebos get (:func:`projection_cv_accuracy` with the draw substituted for the fold direction),
    so ``kept_accuracy_max`` is directly comparable to ``direction_accuracy``.
    """
    generator = torch.Generator().manual_seed(seed)
    draws = [
        shuffled_label_direction(concept.positives, concept.negatives, generator)
        for _ in range(n_draws)
    ]
    cosines = tuple(cosine(draw, direction) for draw in draws)
    excluded = tuple(index for index, value in enumerate(cosines) if abs(value) > max_abs_cosine)
    kept = [index for index in range(n_draws) if index not in excluded]
    accuracies = tuple(
        projection_cv_accuracy(concept, config, lambda _train, draw=draws[index]: draw)
        for index in kept
    )
    if excluded:
        logger.info(
            f"shuffled-label gate excluded {len(excluded)} of {n_draws} draws "
            f"(|cos| > {max_abs_cosine}): {[round(cosines[i], 3) for i in excluded]}"
        )
    return ShuffledLabelDraws(
        n_draws=n_draws,
        max_abs_cosine=max_abs_cosine,
        cosines=cosines,
        excluded=excluded,
        kept_accuracies=accuracies,
    )


@dataclass(frozen=True)
class DirectionRead:
    """One direction at one (pooling, stratum, layer): the whole battery, plus what to call it."""

    name: str
    pooling: str
    stratum: str
    layer: int
    n_pairs: int
    n_positive_rows: int
    n_negative_rows: int
    probe_accuracy: float
    probe_null_accuracy_max: float
    clears_null: bool
    direction_accuracy: float
    placebo_accuracy_mean: float
    placebo_accuracy_max: float
    accuracy_empirical_p: float
    beats_placebo: bool
    split_half_cosine: float
    direction_norm: float
    shuffled_n_draws: int
    shuffled_n_excluded: int
    shuffled_excluded_cosines: tuple[float, ...]
    shuffled_kept_accuracy_max: float | None
    beats_shuffled: bool | None
    disposition_label: str


@dataclass(frozen=True)
class DirectionRefusal:
    """A (name, pooling, stratum, layer) cell that produced no number, and why."""

    name: str
    pooling: str
    stratum: str
    layer: int
    n_pairs: int
    reason: str


@dataclass(frozen=True)
class ValidationKnobs:
    """Everything the battery is parameterised by, bundled so every fitter takes one object."""

    config: ProbeConfig = field(default_factory=ProbeConfig)
    n_placebos: int = DEFAULT_N_PLACEBOS
    n_shuffled: int = DEFAULT_N_SHUFFLED
    shuffled_max_abs_cosine: float = SHUFFLED_MAX_ABS_COSINE


def validate_direction(  # noqa: PLR0913 - a read is its identity, its pairs, the knobs and the gate
    name: str,
    concept: ConceptActivations,
    knobs: ValidationKnobs,
    *,
    pooling: str,
    stratum: str,
    layer: int,
    n_positive_rows: int,
    n_negative_rows: int,
    gate: IdentifiabilityGate,
) -> tuple[DirectionRead, torch.Tensor]:
    """Run the full battery on one concept and return the read with the full-data direction."""
    if concept.n_pairs < knobs.config.n_folds:
        raise DirectionRefusalError(
            f"{concept.n_pairs} pairs cannot fill {knobs.config.n_folds} held-out folds"
        )
    validated, direction = validate_layer(layer, concept, knobs.config, n_placebos=knobs.n_placebos)
    shuffled = shuffled_label_draws(
        concept,
        direction,
        knobs.config,
        n_draws=knobs.n_shuffled,
        seed=knobs.config.seed + layer,
        max_abs_cosine=knobs.shuffled_max_abs_cosine,
    )
    kept_max = shuffled.kept_accuracy_max
    read = DirectionRead(
        name=name,
        pooling=pooling,
        stratum=stratum,
        layer=layer,
        n_pairs=concept.n_pairs,
        n_positive_rows=n_positive_rows,
        n_negative_rows=n_negative_rows,
        probe_accuracy=validated.probe_accuracy,
        probe_null_accuracy_max=validated.probe_null_accuracy_max,
        clears_null=validated.clears_null,
        direction_accuracy=validated.direction_accuracy,
        placebo_accuracy_mean=validated.placebo_accuracy_mean,
        placebo_accuracy_max=validated.placebo_accuracy_max,
        accuracy_empirical_p=validated.accuracy_empirical_p,
        beats_placebo=validated.beats_placebo,
        split_half_cosine=validated.direction_split_half_cosine,
        direction_norm=validated.direction_norm,
        shuffled_n_draws=shuffled.n_draws,
        shuffled_n_excluded=shuffled.n_excluded,
        shuffled_excluded_cosines=tuple(round(shuffled.cosines[i], 4) for i in shuffled.excluded),
        shuffled_kept_accuracy_max=kept_max,
        beats_shuffled=None if kept_max is None else validated.direction_accuracy > kept_max,
        disposition_label=gate.disposition_label,
    )
    return read, direction


# --------------------------------------------------------------------------------------
# The fitters
# --------------------------------------------------------------------------------------

DirectionKey = tuple[str, str, str, int]
"""(name, pooling, stratum, layer): how a saved direction is keyed."""


@dataclass
class DirectionSet:
    """Every read, every refusal and every direction one fitter (or several) produced."""

    reads: list[DirectionRead] = field(default_factory=list[DirectionRead])
    refusals: list[DirectionRefusal] = field(default_factory=list[DirectionRefusal])
    directions: dict[DirectionKey, torch.Tensor] = field(
        default_factory=dict[DirectionKey, torch.Tensor]
    )

    def extend(self, other: DirectionSet) -> None:
        """Fold another fitter's output in."""
        self.reads.extend(other.reads)
        self.refusals.extend(other.refusals)
        self.directions.update(other.directions)

    def by_layer(
        self, name: str, pooling: str, stratum: str = STRATUM_ALL
    ) -> dict[int, torch.Tensor]:
        """One direction family as ``{layer: tensor}``, the shape the lens and steering tiers read."""
        return {
            layer: tensor
            for (held_name, held_pooling, held_stratum, layer), tensor in self.directions.items()
            if (held_name, held_pooling, held_stratum) == (name, pooling, stratum)
        }

    def table(self) -> pl.DataFrame:
        """Return the reads as one Polars table."""
        return records_table(self.reads, DirectionRead)

    def refusal_table(self) -> pl.DataFrame:
        """Return the refusals as one Polars table."""
        return records_table(self.refusals, DirectionRefusal)

    def save(self, out_dir: Path) -> None:
        """Write the two tables as ndjson and the directions keyed by ``name|pooling|stratum|layer``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.table().write_ndjson(out_dir / READS_FILENAME)
        self.refusal_table().write_ndjson(out_dir / REFUSALS_FILENAME)
        torch.save(
            {
                "|".join(str(part) for part in key): tensor
                for key, tensor in self.directions.items()
            },
            out_dir / DIRECTIONS_FILENAME,
        )
        logger.info(
            f"directions written, {out_dir=} reads={len(self.reads)} refusals={len(self.refusals)}"
        )


def load_directions(path: Path) -> dict[DirectionKey, torch.Tensor]:
    """Read back what :meth:`DirectionSet.save` wrote, restoring the tuple keys."""
    stored = cast("dict[str, torch.Tensor]", torch.load(path, weights_only=True))
    out: dict[DirectionKey, torch.Tensor] = {}
    for key, tensor in stored.items():
        name, pooling, stratum, layer = key.split("|")
        out[name, pooling, stratum, int(layer)] = tensor
    return out


def _fit_grid(  # noqa: PLR0913 - a grid is a cell, a set, a layout, its strata, the axes and the knobs
    name: str,
    cell: CapturedCell,
    stimulus_set: str,
    layout: PairLayout,
    *,
    pairs_by_stratum: Mapping[str, torch.Tensor],
    poolings: Sequence[str],
    layers: Sequence[int],
    knobs: ValidationKnobs,
    gate: IdentifiabilityGate,
    row_counts: tuple[int, int] | None = None,
) -> DirectionSet:
    """Validate one contrast at every (pooling, stratum, layer), recording refusals instead of holes."""
    out = DirectionSet()
    for pooling in poolings:
        for stratum, pairs in pairs_by_stratum.items():
            n_pairs = int(pairs.numel())
            for layer in layers:
                try:
                    concept = concept_activations(
                        cell, stimulus_set, pooling, layer, layout, pairs=pairs
                    )
                    positive_rows, negative_rows = row_counts or (n_pairs, n_pairs)
                    read, direction = validate_direction(
                        name,
                        concept,
                        knobs,
                        pooling=pooling,
                        stratum=stratum,
                        layer=layer,
                        n_positive_rows=positive_rows,
                        n_negative_rows=negative_rows,
                        gate=gate,
                    )
                except DirectionRefusalError as refusal:
                    out.refusals.append(
                        DirectionRefusal(name, pooling, stratum, layer, n_pairs, str(refusal))
                    )
                    continue
                out.reads.append(read)
                out.directions[name, pooling, stratum, layer] = direction
            cleared = sum(
                read.beats_placebo
                for read in out.reads
                if (read.pooling, read.stratum) == (pooling, stratum)
            )
            logger.info(
                f"{name} fitted, {pooling=} {stratum=} {n_pairs=} layers={len(layers)} "
                f"beats_placebo={cleared}"
            )
    return out


def side_contrast_directions(  # noqa: PLR0913 - a contrast names its sides, strata, axes and knobs
    cell: CapturedCell,
    name: str,
    *,
    positive_side: str,
    negative_side: str,
    poolings: Sequence[str],
    layers: Sequence[int],
    knobs: ValidationKnobs,
    gate: IdentifiabilityGate,
    stimulus_set: str = STIMULUS_SET,
    detectable: Mapping[str, bool] | None = None,
    strata: Sequence[str] = (STRATUM_ALL,),
) -> DirectionSet:
    """Fit and validate one two-side contrast over a set, per pooling, stratum and layer.

    ``d_twin``, ``d_spec``, the content controls and ``e`` are all this function with different
    sides; :data:`SIDE_CONTRASTS` names the twin-set ones. Strata beyond ``all`` need ``detectable``.
    """
    layout = two_side_layout(
        cell, stimulus_set, positive_side=positive_side, negative_side=negative_side
    )
    pairs_by_stratum = {stratum: stratum_pairs(layout, stratum, detectable) for stratum in strata}
    return _fit_grid(
        name,
        cell,
        stimulus_set,
        layout,
        pairs_by_stratum=pairs_by_stratum,
        poolings=poolings,
        layers=layers,
        knobs=knobs,
        gate=gate,
    )


def twin_directions(  # noqa: PLR0913 - the twin read is every named contrast over the same knobs
    cell: CapturedCell,
    *,
    detectable: Mapping[str, bool],
    poolings: Sequence[str],
    layers: Sequence[int],
    knobs: ValidationKnobs,
    gate: IdentifiabilityGate,
    stimulus_set: str = STIMULUS_SET,
    contrasts: Mapping[str, tuple[str, str]] = SIDE_CONTRASTS,
) -> DirectionSet:
    """Every named twin-set contrast, ``d_twin`` first, all three strata each."""
    out = DirectionSet()
    for name, (positive_side, negative_side) in contrasts.items():
        out.extend(
            side_contrast_directions(
                cell,
                name,
                positive_side=positive_side,
                negative_side=negative_side,
                poolings=poolings,
                layers=layers,
                knobs=knobs,
                gate=gate,
                stimulus_set=stimulus_set,
                detectable=detectable,
                strata=STRATA,
            )
        )
    return out


def sentence_directions(  # noqa: PLR0913 - a sentence read names its set, its axes and the knobs
    cell: CapturedCell,
    *,
    poolings: Sequence[str],
    layers: Sequence[int],
    knobs: ValidationKnobs,
    gate: IdentifiabilityGate,
    stimulus_set: str = CONCEPT_EVAL_AWARENESS,
) -> DirectionSet:
    """Fit ``e`` (or another concept's direction) from a captured sentence set, by its written name.

    The set is selected by the name `tmax_sentence_corpus` wrote it under, which is the concept's key
    in `stimuli.SENTENCE_STIMULUS_SETS`; any other spelling is refused here with the known names
    listed, before the cell is asked for a set it cannot hold.
    """
    if stimulus_set not in SENTENCE_STIMULUS_SETS:
        raise DirectionRefusalError(
            f"{stimulus_set!r} is not a captured sentence set; the sets are "
            f"{sorted(SENTENCE_STIMULUS_SETS)}, spelled as reward_hacking.interp.stimuli names them and "
            f"as tmax_sentence_corpus writes them"
        )
    return side_contrast_directions(
        cell,
        SENTENCE_DIRECTION_NAMES[stimulus_set],
        positive_side=SENTENCE_SIDE_POSITIVE,
        negative_side=SENTENCE_SIDE_NEGATIVE,
        poolings=poolings,
        layers=layers,
        knobs=knobs,
        gate=gate,
        stimulus_set=stimulus_set,
    )


def generation_directions(  # noqa: PLR0913 - a generation read is a cell, its labels, a contrast and the knobs
    cell: CapturedCell,
    labels: pl.DataFrame,
    contrast: GenerationContrast,
    *,
    stimulus_set: str,
    poolings: Sequence[str],
    layers: Sequence[int],
    knobs: ValidationKnobs,
    gate: IdentifiabilityGate,
) -> DirectionSet:
    """Fit ``d_hack`` or ``d_capability`` within (problem, unit) groups of the generation set."""
    aligned = align_labels(cell, stimulus_set, labels)
    pairing = within_group_pairs(
        aligned,
        positive=contrast.positive,
        negative=contrast.negative,
        positive_side=contrast.positive_side,
        negative_side=contrast.negative_side,
    )
    logger.info(
        f"{contrast.name} pairing, groups={pairing.n_groups} pairs={pairing.layout.n_pairs} "
        f"positive_rows={pairing.n_positive_rows} negative_rows={pairing.n_negative_rows} "
        f"groups_without_both_sides={pairing.groups_without_both_sides}"
    )
    return _fit_grid(
        contrast.name,
        cell,
        stimulus_set,
        pairing.layout,
        pairs_by_stratum={STRATUM_ALL: torch.arange(pairing.layout.n_pairs)},
        poolings=poolings,
        layers=layers,
        knobs=knobs,
        gate=gate,
        row_counts=(pairing.n_positive_rows, pairing.n_negative_rows),
    )
