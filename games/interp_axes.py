"""Axis-to-axis geometry over cached capture cells, conditioned on the cited-cells confound.

The two game stimulus sets quote payoff cells with the same side-pattern for most pairs (side A the
diagonal, side B the off-diagonal), so a raw cosine between the two axes could be reading a
*cells-citation detector* rather than two related concepts. The corpus measures the confound
instead of authoring it away: `cited_cells` in the provenance file says which citation regime each
pair is in. This module is where that conditioning becomes arithmetic rather than a caveat:

* **Every axis-to-axis cosine is reported twice** -- pooled over all pairs, and within each
  `cited_cells` stratum the two sets share -- with the split-half ceilings and a matched-norm
  placebo floor computed within the same stratum, so the two numbers are comparable rather than
  merely adjacent.
* **The within-axis stratum contrast** (the same set's direction fitted separately per citation
  regime, then compared) is the sharpest single read: if the coarse axis is the same direction in
  its `matched-column` and `split-diagonal-offdiagonal` halves, cells-citation is not what it
  measures.
* **A provenance record that cannot be stratified is refused, never pooled silently.** A game-set
  row lacking `cited_cells` means an older, confounded rendering of the corpus; analysing it as if
  the key were uniform would produce exactly the un-conditioned number this module exists to
  replace. Decision-set rows carry a `scenario_id` schema instead and legitimately form their own
  single stratum.

Directions here are diff-of-means in raw residual space, per (cell, set, pooling, stratum, layer),
with the direction *quality* reads reused from `reward_hacking.interp.eval_awareness_probe`: the
held-out nearest-centroid accuracy against matched-norm placebos, and the odd/even split-half
cosine as each direction's own noise ceiling. Trained-probe validation and the checkpoint
trajectory live in `games.interp_trajectory`; this module answers the geometry-between-axes
question that module deliberately leaves alone.

Two reading conventions, both from the stimuli verification pass. The `matched-column` stratum is
the confound-clean regime and therefore the **primary** comparison (`--primary-stratum`, recorded
in the payload); the confounded stratum and the pooled read stay beside it. And pooling choice
matters more here than usual: the pairs are matched-stem, so the within-pair token divergence is
small (2.4% median on the clean half), which a whole-prompt `mean` pooling dilutes ~40:1 — prefer
`last` (the commitment position) until the capture format grows a diverging-suffix pooling; this
module is generic over whatever pooling names the cache carries, so that lands here for free.

Offline and CPU-only: everything is arithmetic over the cells `games.interp_cells` reads back,
behind the same identity guards (a cache whose stimuli digest does not match the supplied corpus
is refused before any number is computed).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.interp_cells import (
    CONSTRUCT_NAMES,
    CapturedCell,
    PairLayout,
    Stimulus,
    concept_activations,
    construct_group,
    load_ladder,
    load_stimuli,
    pair_layout,
    stimuli_digest,
    validate_construct_stimuli,
)
from reward_hacking.interp.directions import cosine, matched_norm_random_direction
from reward_hacking.interp.eval_awareness_probe import (
    DEFAULT_N_PLACEBOS,
    direction_separation,
    direction_split_half_cosine,
)
from reward_hacking.interp.linear_probe import ProbeConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("games.interp_axes")

CITED_CELLS_KEY = "cited_cells"
SCENARIO_KEY = "scenario_id"

# The stratum name for decision-set pairs, whose provenance schema has no payoff cells to cite.
SCENARIO_STRATUM = "decision-scenario"

# The pseudo-stratum for a direction fitted on all of a set's pairs, reported beside the real ones.
POOLED_STRATUM = "pooled"

# The confound-clean regime, primary per the owner's ruling; see CONFOUNDS-read-me-2026-08-20.md.
DEFAULT_PRIMARY_STRATUM = "matched-column"

REPORT_FILENAME = "axes.json"
DIRECTIONS_DIRNAME = "directions"

# How many offending ids an error quotes; the counts beside them carry the magnitude.
ERROR_EXAMPLE_COUNT = 5
PAIR_MEMBER_COUNT = 2

# Constraint (b) from the stimuli verification pass, carried in every payload.
KNOWN_SURFACE_RESIDUALS: tuple[str, ...] = (
    (
        "label-mention residual: after the CELL_OFFSET_PER_VARIANT=1 counterbalance fix, a residual "
        "~1-mention-per-label wording imbalance between sides remains. An axis that validates "
        "suspiciously strongly at the final position warrants a token-identity control direction "
        "(fit on label-token identity alone, compared against the axis) before it is believed; "
        "that control is deliberately not built yet."
    ),
)


class ProvenanceError(ValueError):
    """The provenance file cannot support the stratified reads this module exists to produce."""


@dataclass(frozen=True)
class StimulusStrata:
    """Each pair's citation stratum, joined and cross-checked against the stimulus corpus.

    `stratum_by_pair` is the single source the reads slice on; it exists only if every stimulus row
    joined to exactly one provenance row, the two files agreed on the pairing fields, and both
    sides of every pair landed in one stratum.
    """

    stratum_by_pair: dict[str, str]

    def strata_of(self, layout: PairLayout) -> dict[str, torch.Tensor]:
        """Pair indices per stratum for one set's layout, strata in sorted order."""
        by_stratum: dict[str, list[int]] = {}
        for index, pair_id in enumerate(layout.pair_ids):
            by_stratum.setdefault(self.stratum_by_pair[pair_id], []).append(index)
        return {stratum: torch.tensor(indices) for stratum, indices in sorted(by_stratum.items())}


@dataclass(frozen=True)
class ConstructSplit:
    """A whole-scenario fit/held-out split for one private construct."""

    construct: str
    group_by_pair: dict[str, str]
    fit_pair_ids: tuple[str, ...]
    heldout_pair_ids: tuple[str, ...]
    procedure_regime_by_pair: dict[str, str] = field(default_factory=dict)

    @property
    def fit_groups(self) -> tuple[str, ...]:
        """Scenario/template groups assigned to direction fitting."""
        return tuple(sorted({self.group_by_pair[pair_id] for pair_id in self.fit_pair_ids}))

    @property
    def heldout_groups(self) -> tuple[str, ...]:
        """Scenario/template groups assigned to held-out readout."""
        return tuple(sorted({self.group_by_pair[pair_id] for pair_id in self.heldout_pair_ids}))


def assert_no_pair_crosses_split(
    stimuli: Sequence[Stimulus],
    fit_pair_ids: Sequence[str],
    heldout_pair_ids: Sequence[str],
) -> None:
    """Refuse a split that puts either side of a matched pair in different partitions."""
    fit = set(fit_pair_ids)
    heldout = set(heldout_pair_ids)
    overlap = sorted(fit & heldout)
    if overlap:
        raise ProvenanceError(f"pair ids occur in both fit and held-out splits: {overlap[:5]}")
    corpus_pairs = {stimulus.pair_id for stimulus in stimuli}
    missing = sorted(corpus_pairs - fit - heldout)
    extra = sorted((fit | heldout) - corpus_pairs)
    if missing or extra:
        raise ProvenanceError(
            f"fit/held-out split does not cover exactly the corpus pairs: missing {missing[:5]}, "
            f"unknown {extra[:5]}"
        )
    by_pair: dict[str, list[Stimulus]] = {}
    for stimulus in stimuli:
        by_pair.setdefault(stimulus.pair_id, []).append(stimulus)
    malformed = sorted(
        pair_id
        for pair_id, members in by_pair.items()
        if len(members) != PAIR_MEMBER_COUNT
        or sorted(member.side for member in members) != ["A", "B"]
    )
    if malformed:
        raise ProvenanceError(
            f"fit/held-out split contains malformed matched pairs {malformed[:5]}; expected one A "
            "and one B row per pair"
        )


def build_construct_splits(  # noqa: PLR0913 - independent split and corpus identity guards
    stimuli: Sequence[Stimulus],
    *,
    pairs_per_construct: int = 12,
    required_constructs: Sequence[str] = CONSTRUCT_NAMES,
    reserved_groups: Sequence[str] = (),
    external_pair_ids: Sequence[str] = (),
    require_decision_control: bool = False,
) -> dict[str, ConstructSplit]:
    """Read an explicitly authored whole-scenario fit/held-out split.

    The private corpus records ``metadata.split`` on both sides of every pair. Splits are never
    inferred from row order or lexical group names: authored ordering can correlate with scenario
    content and turn a nominal held-out read into a template extrapolation claim it cannot support.
    """
    pair_groups = validate_construct_stimuli(
        stimuli,
        pairs_per_construct=pairs_per_construct,
        required_constructs=required_constructs,
        reserved_groups=reserved_groups,
        external_pair_ids=external_pair_ids,
        require_decision_control=require_decision_control,
    )
    group_owner: dict[str, str] = {}
    for construct, mapping in pair_groups.items():
        for group in mapping.values():
            owner = group_owner.setdefault(group, construct)
            if owner != construct:
                raise ProvenanceError(
                    f"scenario group {group!r} appears in constructs {owner!r} and {construct!r}; "
                    "a shared group could cross the construct split"
                )
    result: dict[str, ConstructSplit] = {}
    for construct in required_constructs:
        mapping = pair_groups[construct]
        split_by_group: dict[str, str] = {}
        for stimulus in stimuli:
            if stimulus.stimulus_set != construct:
                continue
            group = construct_group(stimulus)
            split = str(stimulus.metadata["split"])
            prior = split_by_group.setdefault(group, split)
            if prior != split:
                raise ProvenanceError(
                    f"scenario group {group!r} in construct {construct!r} declares both {prior!r} "
                    f"and {split!r}; a whole group must have one split."
                )
        if set(split_by_group) != set(mapping.values()):
            raise ProvenanceError(
                f"construct {construct!r} split metadata does not cover exactly its scenario groups"
            )
        if set(split_by_group.values()) != {"fit", "heldout"}:
            raise ProvenanceError(
                f"construct {construct!r} needs both explicit fit and heldout scenario groups, "
                f"got {sorted(set(split_by_group.values()))}"
            )
        fit_pairs = tuple(
            sorted(pair_id for pair_id, group in mapping.items() if split_by_group[group] == "fit")
        )
        heldout_pairs = tuple(
            sorted(
                pair_id for pair_id, group in mapping.items() if split_by_group[group] == "heldout"
            )
        )
        construct_stimuli = [stimulus for stimulus in stimuli if stimulus.stimulus_set == construct]
        assert_no_pair_crosses_split(construct_stimuli, fit_pairs, heldout_pairs)
        result[construct] = ConstructSplit(
            construct=construct,
            group_by_pair=dict(mapping),
            fit_pair_ids=fit_pairs,
            heldout_pair_ids=heldout_pairs,
            procedure_regime_by_pair={
                pair_id: str(
                    next(
                        stimulus.metadata.get("procedure_regime", "unspecified")
                        for stimulus in stimuli
                        if stimulus.stimulus_set == construct and stimulus.pair_id == pair_id
                    )
                )
                for pair_id in mapping
            },
        )
    return result


def _stratum_of_row(row: dict[str, Any], *, path: Path, line_number: int) -> str:
    """One provenance row's stratum, or a refusal naming the row that cannot be stratified."""
    cited = row.get(CITED_CELLS_KEY)
    if cited is not None:
        stratum = str(cited)
        if stratum == POOLED_STRATUM:
            raise ProvenanceError(
                f"{path}:{line_number} names its {CITED_CELLS_KEY} stratum {POOLED_STRATUM!r}, "
                f"which this module reserves for the fit-on-all-pairs pseudo-stratum."
            )
        return stratum
    if row.get(SCENARIO_KEY) is not None:
        return SCENARIO_STRATUM
    raise ProvenanceError(
        f"{path}:{line_number} (id {row.get('id')!r}) carries neither {CITED_CELLS_KEY!r} nor "
        f"{SCENARIO_KEY!r}, so its citation regime is unknown. This is the older confounded "
        f"rendering of the corpus; regenerate the stimuli with games.interp_stimuli rather than "
        f"pooling over a confound the analysis exists to condition on."
    )


def load_strata(path: Path, stimuli: Sequence[Stimulus]) -> StimulusStrata:
    """Join the provenance file to the stimulus corpus and return each pair's stratum.

    Every refusal here is a mismatch that would otherwise stratify tensors by the wrong text: a
    provenance file from a different rendering resolves most ids and silently mislabels the rest,
    and a pair whose two sides carry different strata cannot be sliced as a pair at all.
    """
    if not path.is_file():
        raise ProvenanceError(f"{path} is not a file, so no stratified read is possible.")
    rows: dict[str, dict[str, Any]] = {}
    strata: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = cast("dict[str, Any]", json.loads(line))
        row_id = str(row.get("id"))
        if row_id in rows:
            raise ProvenanceError(f"{path}:{line_number} repeats id {row_id!r}.")
        rows[row_id] = row
        strata[row_id] = _stratum_of_row(row, path=path, line_number=line_number)

    corpus_ids = {stimulus.stimulus_id for stimulus in stimuli}
    missing = sorted(corpus_ids - set(rows))
    extra = sorted(set(rows) - corpus_ids)
    if missing or extra:
        raise ProvenanceError(
            f"{path} does not cover the stimulus corpus: {len(missing)} corpus ids have no "
            f"provenance row ({missing[:ERROR_EXAMPLE_COUNT]}) and {len(extra)} provenance rows "
            f"match no stimulus ({extra[:ERROR_EXAMPLE_COUNT]}). The two files come from "
            f"different renderings; regenerate both together."
        )
    disagreeing = sorted(
        stimulus.stimulus_id
        for stimulus in stimuli
        if (
            str(rows[stimulus.stimulus_id].get("set")),
            str(rows[stimulus.stimulus_id].get("side")),
            str(rows[stimulus.stimulus_id].get("pair_id")),
        )
        != (stimulus.stimulus_set, stimulus.side, stimulus.pair_id)
    )
    if disagreeing:
        raise ProvenanceError(
            f"{path} disagrees with the stimulus corpus on set/side/pair_id for "
            f"{len(disagreeing)} ids ({disagreeing[:ERROR_EXAMPLE_COUNT]}), so the join would "
            f"stratify tensors by the wrong text."
        )

    stratum_by_pair: dict[str, str] = {}
    for stimulus in stimuli:
        stratum = strata[stimulus.stimulus_id]
        known = stratum_by_pair.setdefault(stimulus.pair_id, stratum)
        if known != stratum:
            raise ProvenanceError(
                f"pair {stimulus.pair_id!r} straddles strata {known!r} and {stratum!r}, so it "
                f"cannot be sliced as a matched pair within either."
            )
    logger.info(
        f"provenance joined, path={path} pairs={len(stratum_by_pair)} "
        f"strata={sorted(set(stratum_by_pair.values()))}"
    )
    return StimulusStrata(stratum_by_pair=stratum_by_pair)


@dataclass(frozen=True)
class StratumQualityRead:
    """One direction's quality at one (cell, set, pooling, stratum, layer).

    `direction_accuracy` and its placebo fields are the held-out nearest-centroid battery from
    `eval_awareness_probe`; `split_half_cosine` is the direction's own noise ceiling. All of them
    are None, with the reason stated, when the stratum is too small to hold them out -- stated
    rather than silent, because a missing denominator reads as a clean zero otherwise.
    """

    arm: str
    step: int
    stimulus_set: str
    pooling: str
    stratum: str
    layer: int
    n_pairs: int
    direction_norm: float
    direction_accuracy: float | None
    placebo_accuracy_mean: float | None
    placebo_accuracy_max: float | None
    accuracy_empirical_p: float | None
    split_half_cosine: float | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class AxisPairRead:
    """The cosine between two sets' directions at one layer, within one stratum or pooled.

    `placebo_abs_cosine_*` summarise matched-norm random replacements of the second direction
    against the real first one: the floor an unrelated axis of this dimensionality reads at
    (~1/sqrt(d), not zero). The two split-half ceilings say how well-determined each direction is
    within the same stratum, so a modest cosine between two noisy directions is not misread as a
    modest relationship between two clean ones.
    """

    layer: int
    pooling: str
    set_a: str
    set_b: str
    stratum: str
    n_pairs_a: int
    n_pairs_b: int
    cosine_real: float
    placebo_abs_cosine_mean: float
    placebo_abs_cosine_max: float
    split_half_a: float | None
    split_half_b: float | None


@dataclass(frozen=True)
class StratumContrastRead:
    """One set's direction fitted separately per citation regime, compared at one layer.

    The sharpest single read on the confound: if a set's two strata fit the same direction, the
    citation regime is not what the axis measures.
    """

    layer: int
    pooling: str
    stimulus_set: str
    stratum_a: str
    stratum_b: str
    n_pairs_a: int
    n_pairs_b: int
    cosine_real: float
    placebo_abs_cosine_mean: float
    placebo_abs_cosine_max: float
    split_half_a: float | None
    split_half_b: float | None


@dataclass(frozen=True)
class CellAxesRead:
    """Everything this module reads off one cell: quality per stratum, cosines between and within axes."""

    arm: str
    step: int
    quality: tuple[StratumQualityRead, ...]
    axis_pairs: tuple[AxisPairRead, ...]
    stratum_contrasts: tuple[StratumContrastRead, ...]
    directions: dict[tuple[str, str, str], dict[int, torch.Tensor]]


def _quality_read(  # noqa: PLR0913 - a read is a cell, a set, a pooling, a stratum, a layer and knobs
    cell: CapturedCell,
    *,
    stimulus_set: str,
    pooling: str,
    stratum: str,
    layer: int,
    layout: PairLayout,
    pairs: torch.Tensor | None,
    config: ProbeConfig,
    n_placebos: int,
) -> tuple[StratumQualityRead, torch.Tensor]:
    """Fit one stratum's direction at one layer and score it, or state why it cannot be scored."""
    concept = concept_activations(cell, stimulus_set, pooling, layer, layout, pairs=pairs)
    direction = concept.diff_of_means()
    can_hold_out = concept.n_pairs >= config.n_folds
    can_split = concept.n_pairs >= 2  # noqa: PLR2004 - odd/even needs one pair each
    separation = (
        direction_separation(concept, config, n_placebos=n_placebos) if can_hold_out else None
    )
    read = StratumQualityRead(
        arm=cell.arm,
        step=cell.step,
        stimulus_set=stimulus_set,
        pooling=pooling,
        stratum=stratum,
        layer=layer,
        n_pairs=concept.n_pairs,
        direction_norm=float(direction.norm()),
        direction_accuracy=None if separation is None else separation.direction_accuracy,
        placebo_accuracy_mean=None if separation is None else separation.placebo_accuracy_mean,
        placebo_accuracy_max=None if separation is None else separation.placebo_accuracy_max,
        accuracy_empirical_p=None if separation is None else separation.accuracy_empirical_p,
        split_half_cosine=direction_split_half_cosine(concept) if can_split else None,
        unavailable_reason=None
        if can_hold_out
        else f"{concept.n_pairs} pairs cannot fill {config.n_folds} held-out folds",
    )
    return read, direction


def _placebo_floor(
    reference: torch.Tensor,
    replaced: torch.Tensor,
    generator: torch.Generator,
    n_placebos: int,
) -> tuple[float, float]:
    """Mean and max |cosine| of `reference` against matched-norm random replacements of `replaced`."""
    draws = [
        abs(cosine(reference, matched_norm_random_direction(replaced, generator)))
        for _ in range(n_placebos)
    ]
    return sum(draws) / len(draws), max(draws)


@dataclass(frozen=True)
class _CellGeometry:
    """Everything the cosine reads consume, computed once per cell by the quality pass."""

    directions: dict[tuple[str, str, str], dict[int, torch.Tensor]]
    split_half: dict[tuple[str, str, str, int], float | None]
    pair_counts: dict[tuple[str, str], int]
    strata_by_set: dict[str, dict[str, torch.Tensor]]
    generator: torch.Generator
    n_placebos: int


def _axis_pair_reads(
    geometry: _CellGeometry, *, poolings: Sequence[str], layers: Sequence[int]
) -> list[AxisPairRead]:
    """Compare every pair of sets, pooled and within each stratum the two share."""
    reads: list[AxisPairRead] = []
    set_names = sorted(geometry.strata_by_set)
    for pooling in poolings:
        for index, set_a in enumerate(set_names):
            for set_b in set_names[index + 1 :]:
                shared_strata = geometry.strata_by_set[set_a].keys()
                shared = sorted(
                    (shared_strata & geometry.strata_by_set[set_b].keys()) | {POOLED_STRATUM}
                )
                for stratum in shared:
                    for layer in layers:
                        direction_a = geometry.directions[set_a, stratum, pooling][layer]
                        direction_b = geometry.directions[set_b, stratum, pooling][layer]
                        floor_mean, floor_max = _placebo_floor(
                            direction_a, direction_b, geometry.generator, geometry.n_placebos
                        )
                        reads.append(
                            AxisPairRead(
                                layer=layer,
                                pooling=pooling,
                                set_a=set_a,
                                set_b=set_b,
                                stratum=stratum,
                                n_pairs_a=geometry.pair_counts[set_a, stratum],
                                n_pairs_b=geometry.pair_counts[set_b, stratum],
                                cosine_real=cosine(direction_a, direction_b),
                                placebo_abs_cosine_mean=floor_mean,
                                placebo_abs_cosine_max=floor_max,
                                split_half_a=geometry.split_half[set_a, stratum, pooling, layer],
                                split_half_b=geometry.split_half[set_b, stratum, pooling, layer],
                            )
                        )
    return reads


def _stratum_contrast_reads(
    geometry: _CellGeometry, *, poolings: Sequence[str], layers: Sequence[int]
) -> list[StratumContrastRead]:
    """Compare each set's own strata against each other -- the sharpest confound read."""
    reads: list[StratumContrastRead] = []
    for pooling in poolings:
        for stimulus_set, by_stratum in sorted(geometry.strata_by_set.items()):
            own = sorted(by_stratum)
            for index, stratum_a in enumerate(own):
                for stratum_b in own[index + 1 :]:
                    for layer in layers:
                        direction_a = geometry.directions[stimulus_set, stratum_a, pooling][layer]
                        direction_b = geometry.directions[stimulus_set, stratum_b, pooling][layer]
                        floor_mean, floor_max = _placebo_floor(
                            direction_a, direction_b, geometry.generator, geometry.n_placebos
                        )
                        reads.append(
                            StratumContrastRead(
                                layer=layer,
                                pooling=pooling,
                                stimulus_set=stimulus_set,
                                stratum_a=stratum_a,
                                stratum_b=stratum_b,
                                n_pairs_a=geometry.pair_counts[stimulus_set, stratum_a],
                                n_pairs_b=geometry.pair_counts[stimulus_set, stratum_b],
                                cosine_real=cosine(direction_a, direction_b),
                                placebo_abs_cosine_mean=floor_mean,
                                placebo_abs_cosine_max=floor_max,
                                split_half_a=geometry.split_half[
                                    stimulus_set, stratum_a, pooling, layer
                                ],
                                split_half_b=geometry.split_half[
                                    stimulus_set, stratum_b, pooling, layer
                                ],
                            )
                        )
    return reads


def read_cell_axes(  # noqa: PLR0913 - one cell read is the cell, the layouts, the strata and the knobs
    cell: CapturedCell,
    *,
    layouts: dict[str, PairLayout],
    strata: StimulusStrata,
    poolings: Sequence[str],
    layers: Sequence[int],
    config: ProbeConfig,
    n_placebos: int,
) -> CellAxesRead:
    """Compute every stratified read for one cell.

    The placebo generator is seeded per cell from `config.seed`, so re-running one cell reproduces
    its floors exactly and does not depend on which cells ran before it.
    """
    quality: list[StratumQualityRead] = []
    directions: dict[tuple[str, str, str], dict[int, torch.Tensor]] = {}
    split_half: dict[tuple[str, str, str, int], float | None] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    strata_by_set = {name: strata.strata_of(layout) for name, layout in layouts.items()}

    for stimulus_set, layout in sorted(layouts.items()):
        slices: dict[str, torch.Tensor | None] = {POOLED_STRATUM: None}
        slices.update(strata_by_set[stimulus_set])
        for stratum, pairs in slices.items():
            pair_counts[stimulus_set, stratum] = (
                layout.n_pairs if pairs is None else int(pairs.numel())
            )
            for pooling in poolings:
                by_layer: dict[int, torch.Tensor] = {}
                for layer in layers:
                    read, direction = _quality_read(
                        cell,
                        stimulus_set=stimulus_set,
                        pooling=pooling,
                        stratum=stratum,
                        layer=layer,
                        layout=layout,
                        pairs=pairs,
                        config=config,
                        n_placebos=n_placebos,
                    )
                    quality.append(read)
                    by_layer[layer] = direction
                    split_half[stimulus_set, stratum, pooling, layer] = read.split_half_cosine
                directions[stimulus_set, stratum, pooling] = by_layer

    geometry = _CellGeometry(
        directions=directions,
        split_half=split_half,
        pair_counts=pair_counts,
        strata_by_set=strata_by_set,
        generator=torch.Generator().manual_seed(config.seed),
        n_placebos=n_placebos,
    )
    axis_pairs = _axis_pair_reads(geometry, poolings=poolings, layers=layers)
    stratum_contrasts = _stratum_contrast_reads(geometry, poolings=poolings, layers=layers)
    logger.info(
        f"cell read, cell={cell.label} quality={len(quality)} axis_pairs={len(axis_pairs)} "
        f"stratum_contrasts={len(stratum_contrasts)}"
    )
    return CellAxesRead(
        arm=cell.arm,
        step=cell.step,
        quality=tuple(quality),
        axis_pairs=tuple(axis_pairs),
        stratum_contrasts=tuple(stratum_contrasts),
        directions=directions,
    )


def peak_layer_of(reads: Sequence[StratumQualityRead]) -> int:
    """Return the layer whose pooled direction best clears its placebo, for the printed summaries.

    Peak selection is over the pooled stratum's accuracy-above-placebo, mirroring the repo's
    AUC-above-placebo convention; every layer's read stays in the payload regardless.
    """
    pooled = [
        read
        for read in reads
        if read.stratum == POOLED_STRATUM and read.direction_accuracy is not None
    ]
    if not pooled:
        raise ValueError("no pooled quality reads with a held-out accuracy; nothing to rank.")
    return max(
        pooled,
        key=lambda read: (
            cast("float", read.direction_accuracy) - cast("float", read.placebo_accuracy_max)
        ),
    ).layer


def render_axis_pair_table(reads: Sequence[AxisPairRead], *, layer: int, pooling: str) -> str:
    """Render the pooled-vs-stratified comparison at one layer: each number beside its conditioned twin."""
    rows = [read for read in reads if read.layer == layer and read.pooling == pooling]
    header = (
        f"axis-to-axis cosines  {layer=} {pooling=}\n"
        f"{'set_a':>38} {'set_b':>38} {'stratum':>28} {'cos':>7} {'floor_max':>10} "
        f"{'ceil_a':>7} {'ceil_b':>7}"
    )
    lines = [
        f"{read.set_a:>38} {read.set_b:>38} {read.stratum:>28} {read.cosine_real:>7.3f} "
        f"{read.placebo_abs_cosine_max:>10.3f} "
        f"{'n/a' if read.split_half_a is None else format(read.split_half_a, '.3f'):>7} "
        f"{'n/a' if read.split_half_b is None else format(read.split_half_b, '.3f'):>7}"
        for read in rows
    ]
    return "\n".join([header, *lines])


def render_contrast_table(reads: Sequence[StratumContrastRead], *, layer: int, pooling: str) -> str:
    """Render the within-axis stratum contrast at one layer."""
    rows = [read for read in reads if read.layer == layer and read.pooling == pooling]
    if not rows:
        return "no within-axis stratum contrasts (every set has a single stratum)"
    header = (
        f"within-axis stratum contrasts  {layer=} {pooling=}\n"
        f"{'set':>38} {'stratum_a':>28} {'stratum_b':>28} {'cos':>7} {'floor_max':>10}"
    )
    lines = [
        f"{read.stimulus_set:>38} {read.stratum_a:>28} {read.stratum_b:>28} "
        f"{read.cosine_real:>7.3f} {read.placebo_abs_cosine_max:>10.3f}"
        for read in rows
    ]
    return "\n".join([header, *lines])


def save_directions(root: Path, reads: Sequence[CellAxesRead]) -> list[Path]:
    """Write every (set, stratum, pooling) direction as `{layer: tensor}`, the repo's usual shape.

    Per-stratum directions are what the token-identity control and any stratified causal read will
    consume; saving them keeps that a re-analysis of this pass rather than a re-run.
    """
    written: list[Path] = []
    for cell_read in reads:
        for (stimulus_set, stratum, pooling), by_layer in sorted(cell_read.directions.items()):
            out_dir = root / DIRECTIONS_DIRNAME / cell_read.arm / f"step-{cell_read.step}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{stimulus_set}--{stratum}--{pooling}.pt"
            torch.save(by_layer, path)
            written.append(path)
    return written


def build_payload(reads: Sequence[CellAxesRead], context: dict[str, Any]) -> dict[str, Any]:
    """Assemble the JSON report: every read at every layer, keyed by cell."""
    return {
        "context": context,
        "known_surface_residuals": list(KNOWN_SURFACE_RESIDUALS),
        "cells": {
            f"{read.arm}/step-{read.step}": {
                "arm": read.arm,
                "step": read.step,
                "quality": [asdict(item) for item in read.quality],
                "axis_pairs": [asdict(item) for item in read.axis_pairs],
                "stratum_contrasts": [asdict(item) for item in read.stratum_contrasts],
            }
            for read in reads
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI for the stratified axis-geometry read."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--capture-root", type=Path, required=True, help="Root of cached cells.")
    parser.add_argument(
        "--stimuli",
        type=Path,
        required=True,
        help="The stimulus corpus the cells were captured on; its digest is checked against them.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="stimuli_provenance.jsonl beside the corpus; rows lacking a citation regime are fatal.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--positive-side",
        required=True,
        help="Which stimulus side counts as the positive class, e.g. A for these games sets.",
    )
    parser.add_argument(
        "--arms", default=None, help="Comma-separated arms; default every one present."
    )
    parser.add_argument(
        "--steps", default=None, help="Comma-separated steps; default every one present."
    )
    parser.add_argument("--sets", default=None, help="Comma-separated stimulus sets; default all.")
    parser.add_argument("--poolings", default=None, help="Comma-separated poolings; default all.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layers; default every layer in the capture. A subset is a smoke.",
    )
    parser.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)
    parser.add_argument("--n-folds", type=int, default=ProbeConfig.n_folds)
    parser.add_argument("--seed", type=int, default=ProbeConfig.seed)
    parser.add_argument(
        "--primary-stratum",
        default=DEFAULT_PRIMARY_STRATUM,
        help="The citation stratum to read first. matched-column is the confound-clean regime on "
        "this corpus (both sides cite the same cells), so it is the primary comparison per "
        "docs/scratch/interp-capture-2026-08-20/CONFOUNDS-read-me-2026-08-20.md.",
    )
    parser.add_argument(
        "--require-capture-provenance",
        action="store_true",
        help="Refuse legacy cells without concrete tokenizer and kernel identities.",
    )
    return parser


def _split(raw: str | None) -> list[str] | None:
    """Parse a comma-separated CLI list, or None for 'everything present'."""
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load the ladder behind its identity guards, read every cell, and write the report."""
    stimuli = load_stimuli(args.stimuli)
    digest = stimuli_digest(stimuli)
    strata = load_strata(args.provenance, stimuli)
    arms = _split(args.arms)
    steps = None if args.steps is None else [int(part) for part in _split(args.steps) or []]
    ladder = load_ladder(
        args.capture_root,
        arms=arms,
        steps=steps,
        stimuli_sha256=digest,
        require_capture_provenance=args.require_capture_provenance,
    )
    wanted_sets = _split(args.sets) or list(ladder.stimulus_sets)
    wanted_poolings = _split(args.poolings) or list(ladder.poolings)
    layers = (
        [int(part) for part in _split(args.layers) or []]
        if args.layers is not None
        else list(range(ladder.identity.n_layers))
    )
    config = ProbeConfig(n_folds=args.n_folds, seed=args.seed)

    layouts = {
        stimulus_set: pair_layout(ladder.cells[0], stimulus_set, positive_side=args.positive_side)
        for stimulus_set in wanted_sets
    }
    reads = [
        read_cell_axes(
            cell,
            layouts=layouts,
            strata=strata,
            poolings=wanted_poolings,
            layers=layers,
            config=config,
            n_placebos=args.n_placebos,
        )
        for cell in ladder.cells
    ]

    context = {
        "capture_root": str(args.capture_root),
        "stimuli_file": str(args.stimuli),
        "provenance_file": str(args.provenance),
        "stimuli_sha256": digest,
        "identity": ladder.identity.to_payload(),
        "cells": [cell.label for cell in ladder.cells],
        "positive_side": args.positive_side,
        "layers": layers,
        "n_placebos": args.n_placebos,
        "primary_stratum": args.primary_stratum,
        "probe_config": asdict(config),
    }
    payload = build_payload(reads, context)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / REPORT_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    written = save_directions(args.out_dir, reads)
    logger.info(
        f"axes written, report={args.out_dir / REPORT_FILENAME} cells={len(reads)} "
        f"direction_files={len(written)}"
    )
    print_tables(reads, poolings=wanted_poolings)
    return payload


def print_tables(reads: Sequence[CellAxesRead], *, poolings: Sequence[str]) -> None:
    """Print each cell's peak-layer comparison tables; the JSON keeps every layer."""
    for cell_read in reads:
        peak = peak_layer_of(cell_read.quality)
        for pooling in poolings:
            print(f"== {cell_read.arm}/step-{cell_read.step} ==")  # noqa: T201 - a CLI whose output is the table
            print(render_axis_pair_table(cell_read.axis_pairs, layer=peak, pooling=pooling))  # noqa: T201
            print(render_contrast_table(cell_read.stratum_contrasts, layer=peak, pooling=pooling))  # noqa: T201
            print()  # noqa: T201


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stratified axis-geometry read over a capture root."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
