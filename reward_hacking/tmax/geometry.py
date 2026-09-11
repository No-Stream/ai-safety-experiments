"""Experiments 1 & 2 scaffold: direction geometry across the TMAX checkpoints (GPU-deferred).

Experiment 1 (dose-response): run the repo's reward-hack-vs-deception direction probe on the base
checkpoint and each flagship intermediate checkpoint, and watch whether a reward-hack direction
emerges or grows with RL steps. Experiment 2 (seven-way looseness): compare the final RL'd model of
each of the seven suites, and ask whether the disposition tracks how loose each suite's verifier was
(proxied by the Table-1 pass rate in :mod:`reward_hacking.tmax.artifacts`).

The heavy half -- loading a 9B checkpoint and capturing residual-stream activations -- lives in
:mod:`reward_hacking.interp` (``directions.run_probe`` for the diff-of-means direction,
``linear_probe.run_linear_probe`` for the logistic probe). This module deliberately does NOT import
either at module load, so importing it never drags in torch or a GPU: the per-checkpoint probe is
passed in as ``probe_fn`` (dependency injection). The caller wires the real interp function, e.g.::

    from reward_hacking.interp.directions import run_probe
    def probe_fn(repo_id: str, revision: str):
        return run_probe(repo_id)  # see note below on threading `revision`

    table = run_dose_response(dose_response_ladder(), suite="tmax_15k", probe_fn=probe_fn)

Everything else here -- ranking suites by looseness, summarizing a per-layer comparison into one
checkpoint's separability, and assembling the per-checkpoint / per-suite result tables -- is pure
Python over floats, and ``reward_hacking/tests/test_tmax_geometry.py`` exercises all of it without a
GPU by injecting a fake ``probe_fn``.

Two GPU-side caveats flagged rather than patched around, because this is a scaffold:

* The interp probe was validated on ``Qwen/Qwen3.5-4B``; the TMAX checkpoints are Qwen3.5-9B with
  hybrid Gated-DeltaNet attention, so ``_decoder_layers`` resolution and the hand-authored Qwen
  stimulus set are both transfer steps. A null is a transfer lead before it is a finding.
* ``directions.run_probe`` / ``load_model_and_tokenizer`` do not currently thread a ``revision=``
  through to ``from_pretrained``, so loading a flagship *step branch* (the whole point of the
  dose-response ladder) needs that one addition first. ``probe_fn`` takes ``(repo_id, revision)`` to
  make the requirement explicit at the seam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from reward_hacking.tmax.artifacts import SUITES, Checkpoint, EnvironmentSuite

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

logger = logging.getLogger(__name__)


class LayerSeparabilityReading(Protocol):
    """The per-layer fields this module reads off a probe result.

    ``reward_hacking.interp.directions.LayerComparison`` matches structurally, so the real probe
    output flows straight in; tests supply any object carrying these fields. Kept as a Protocol so
    nothing here imports the concrete class (and, transitively, torch) at module load.

    The members are properties rather than plain annotations because a plain one is *writable*, and
    a protocol demanding writable attributes is satisfied by no frozen dataclass -- which
    ``LayerComparison`` and every test double here are. That made the structural match above a claim
    the type checker rejected. Read-only is also the honest requirement: nothing here mutates.

    The two ``cos_hack_placebo`` members are the hack direction against a matched-norm random
    direction, read here because the probe computes them anyway. Being honest about what they buy on
    a *cosine*: any random direction sits near zero regardless (std ~ 1/sqrt(d)), so this is not the
    "could any vector of that magnitude do this" magnitude control it would be for a steering or
    ablation experiment. Its job here is drift detection: a placebo that stops sitting near zero at
    some checkpoint is the flag that the capture or the standardization changed under us.
    """

    @property
    def layer(self) -> int:
        """Index of the decoder layer these numbers were computed at."""
        ...

    @property
    def cos_hack_deception(self) -> float:
        """Cosine between the reward-hack and deception directions, raw activations."""
        ...

    @property
    def cos_hack_deception_standardized(self) -> float:
        """The same cosine on per-dimension z-scored activations, robust to massive activations."""
        ...

    @property
    def cos_hack_placebo(self) -> float:
        """Cosine between the reward-hack direction and a matched-norm random one, raw."""
        ...

    @property
    def cos_hack_placebo_standardized(self) -> float:
        """The same placebo cosine on standardized activations, the floor for the reading above."""
        ...

    @property
    def hack_norm(self) -> float:
        """Length of the reward-hack direction, the "is there a direction at all" magnitude."""
        ...


def separability_from_cosine(cosine: float) -> float:
    """Separability of two directions from the cosine between them: ``1 - abs(cosine)``, in [0, 1].

    Zero when the two directions are one axis and one when they are orthogonal. That reading is the
    repo's own probe convention rather than a choice made here:
    ``tests/test_interp_directions.py::test_aligned_concepts_are_high_and_orthogonal_are_zero`` pins
    ``cos_hack_deception`` at ~1.0 for two identically-planted concepts and ~0.0 for two planted on
    disjoint axes. ``abs`` is why this is a function and not a sign flip on the cosine: an
    anti-parallel pair is one axis read backwards, so a cosine of -1 is fully entangled too.
    """
    return 1.0 - abs(cosine)


@dataclass(frozen=True)
class SuiteLooseness:
    """One suite's looseness proxy and its rank, loosest (highest pass rate) first."""

    suite: str
    display_name: str
    gemini_pass_at_1: float
    looseness_rank: int


def rank_suites_by_looseness(
    suites: Iterable[EnvironmentSuite] | None = None,
) -> list[SuiteLooseness]:
    """Order suites loosest-first by the Table-1 pass-rate proxy (higher pass@1 == looser/easier).

    Defaults to the seven registered suites. Ties break by suite name so the order is deterministic.
    The proxy is difficulty/pass-rate, NOT a direct verifier-looseness measurement (the paper does
    not report the latter per suite); experiment 2 reads this ordering as a proxy, not ground truth.
    """
    entries = list(SUITES.values() if suites is None else suites)
    ordered = sorted(entries, key=lambda entry: (-entry.gemini_pass_at_1, entry.name))
    return [
        SuiteLooseness(
            suite=entry.name,
            display_name=entry.display_name,
            gemini_pass_at_1=entry.gemini_pass_at_1,
            looseness_rank=rank,
        )
        for rank, entry in enumerate(ordered)
    ]


@dataclass(frozen=True)
class CheckpointSeparability:
    """One checkpoint's headline geometry: where and how strongly the two concepts separate.

    The peak is taken over layers because the informative band is narrow and depth-dependent
    (collapsing to one layer throws that away). ``peak_cos_hack_deception_standardized`` is the
    massive-activation-robust reading; a large raw-vs-standardized gap is itself the flag that a few
    outlier dimensions, not concept content, drove the raw cosine.
    :attr:`peak_separability_standardized` is that cosine turned into the quantity the experiments
    are actually about, so nothing downstream has to remember that the cosine runs the other way.

    The ``peak_cos_hack_placebo*`` pair is the same peak layer's matched-norm placebo cosine, so
    every reported number travels with its floor reference. Any random direction sits near zero on a
    cosine, so it is drift detection across checkpoints, not the magnitude control it would be for a
    steer or an ablation.
    """

    checkpoint_name: str
    model_ref: str
    rl_step: int | None
    n_layers: int
    peak_layer: int
    peak_cos_hack_deception: float
    peak_cos_hack_deception_standardized: float
    peak_cos_hack_placebo: float
    peak_cos_hack_placebo_standardized: float
    mean_hack_norm: float

    @property
    def peak_separability_standardized(self) -> float:
        """The peak layer's separability, ``1 - abs(cosine)`` on the standardized reading.

        Derived rather than stored so it cannot disagree with the cosine it comes from.
        """
        return separability_from_cosine(self.peak_cos_hack_deception_standardized)


def summarize_direction_comparisons(
    checkpoint: Checkpoint,
    comparisons: Sequence[LayerSeparabilityReading],
) -> CheckpointSeparability:
    """Reduce one checkpoint's per-layer comparison to its peak separability and mean hack norm.

    "Peak" is the layer of maximum separability -- equivalently, the layer whose standardized
    hack/deception cosine sits closest to zero, the depth at which the two concepts are most
    distinct on the massive-activation-robust reading. ``max`` over the *cosine* picks the
    opposite layer, the one where the two directions are most nearly a single axis, which is
    what this reduction selected until the sign was settled.
    """
    if not comparisons:
        raise ValueError(f"no per-layer comparisons for checkpoint {checkpoint.name!r}")
    peak = max(
        comparisons,
        key=lambda layer: separability_from_cosine(layer.cos_hack_deception_standardized),
    )
    mean_hack_norm = sum(layer.hack_norm for layer in comparisons) / len(comparisons)
    return CheckpointSeparability(
        checkpoint_name=checkpoint.name,
        model_ref=checkpoint.model_ref,
        rl_step=checkpoint.rl_step,
        n_layers=len(comparisons),
        peak_layer=peak.layer,
        peak_cos_hack_deception=peak.cos_hack_deception,
        peak_cos_hack_deception_standardized=peak.cos_hack_deception_standardized,
        peak_cos_hack_placebo=peak.cos_hack_placebo,
        peak_cos_hack_placebo_standardized=peak.cos_hack_placebo_standardized,
        mean_hack_norm=mean_hack_norm,
    )


@dataclass(frozen=True)
class DoseResponseTable:
    """Experiment 1: one suite's separability at each checkpoint along the RL ladder."""

    suite: str
    rows: tuple[CheckpointSeparability, ...]

    def trend(self) -> float:
        """Last-minus-first change in peak separability along the ladder (0 if under 2 rows).

        The ladder's first row is the base checkpoint (RL step 0) and its last is the final RL step,
        so a positive trend is the reward-hack and deception axes separating further with training,
        and a negative one is them collapsing onto each other. Reading the cosine delta instead has
        the opposite sign -- a rising cosine means the two concepts merged -- which is the inversion
        this was measuring before the sign was settled.
        """
        if len(self.rows) < 2:  # noqa: PLR2004 - a trend needs at least a start and an end
            return 0.0
        return (
            self.rows[-1].peak_separability_standardized
            - self.rows[0].peak_separability_standardized
        )

    def render(self) -> str:
        """Tabulate step, peak layer, the peak cosines, the separability, and the placebo floor.

        ``sep_z(h,d)`` is ``1 - abs(cos_z(h,d))``: the same number as the cosine beside it, in the
        direction the experiment reads (higher is more separable). ``cos_z(h,p)`` is that layer's
        matched-norm placebo cosine, the reference line the hack/deception number has to be read
        against rather than a separate result.
        """
        header = (
            f"suite={self.suite}\n"
            f"{'checkpoint':>22}  {'step':>5}  {'peak_layer':>10}  "
            f"{'cos(h,d)':>9}  {'cos_z(h,d)':>11}  {'sep_z(h,d)':>11}  "
            f"{'cos_z(h,p)':>11}  {'mean|hack|':>11}"
        )
        lines = [header]
        for row in self.rows:
            step = "-" if row.rl_step is None else str(row.rl_step)
            lines.append(
                f"{row.checkpoint_name:>22}  {step:>5}  {row.peak_layer:>10}  "
                f"{row.peak_cos_hack_deception:>9.4f}  "
                f"{row.peak_cos_hack_deception_standardized:>11.4f}  "
                f"{row.peak_separability_standardized:>11.4f}  "
                f"{row.peak_cos_hack_placebo_standardized:>11.4f}  {row.mean_hack_norm:>11.2f}"
            )
        return "\n".join(lines)


def run_dose_response(
    checkpoints: Sequence[Checkpoint],
    *,
    suite: str,
    probe_fn: Callable[[str, str], Sequence[LayerSeparabilityReading]],
    summarize_fn: Callable[
        [Checkpoint, Sequence[LayerSeparabilityReading]], CheckpointSeparability
    ] = summarize_direction_comparisons,
) -> DoseResponseTable:
    """Probe each checkpoint in order and assemble the dose-response table (experiment 1).

    ``probe_fn`` receives ``(repo_id, revision)`` from each checkpoint's :meth:`Checkpoint.resolve`,
    so an unverified checkpoint fails loudly here rather than reaching a loader. Pure glue
    otherwise: it calls the injected probe, summarizes, and orders the rows as the ladder was given.
    """
    rows = tuple(
        summarize_fn(checkpoint, probe_fn(*checkpoint.resolve())) for checkpoint in checkpoints
    )
    return DoseResponseTable(suite=suite, rows=rows)


@dataclass(frozen=True)
class SuiteSeparabilityRow:
    """Experiment 2: one suite's final-model separability beside its looseness proxy and rank."""

    suite: str
    display_name: str
    gemini_pass_at_1: float
    looseness_rank: int
    separability: CheckpointSeparability


@dataclass(frozen=True)
class SuiteComparisonTable:
    """Experiment 2: the seven suites' final-model separability, ordered loosest-first."""

    rows: tuple[SuiteSeparabilityRow, ...]

    def render(self) -> str:
        """Tabulate looseness rank, pass-rate proxy, peak cosine, its separability and its floor."""
        header = (
            f"{'suite':>22}  {'rank':>4}  {'pass@1':>6}  {'peak_layer':>10}  "
            f"{'cos_z(h,d)':>11}  {'sep_z(h,d)':>11}  {'cos_z(h,p)':>11}"
        )
        rows = [
            f"{row.suite:>22}  {row.looseness_rank:>4}  {row.gemini_pass_at_1:>6.2f}  "
            f"{row.separability.peak_layer:>10}  "
            f"{row.separability.peak_cos_hack_deception_standardized:>11.4f}  "
            f"{row.separability.peak_separability_standardized:>11.4f}  "
            f"{row.separability.peak_cos_hack_placebo_standardized:>11.4f}"
            for row in self.rows
        ]
        return "\n".join([header, *rows])


def run_suite_comparison(
    *,
    probe_fn: Callable[[str, str], Sequence[LayerSeparabilityReading]],
    suites: Iterable[EnvironmentSuite] | None = None,
    summarize_fn: Callable[
        [Checkpoint, Sequence[LayerSeparabilityReading]], CheckpointSeparability
    ] = summarize_direction_comparisons,
) -> SuiteComparisonTable:
    """Probe each suite's final RL'd model and pair separability with its looseness rank (exp 2).

    Rows come back loosest-first (:func:`rank_suites_by_looseness`), so the table reads as a
    looseness ordering with the geometry measurement beside it.
    """
    entries = {entry.name: entry for entry in (SUITES.values() if suites is None else suites)}
    ranking = rank_suites_by_looseness(entries.values())
    rows: list[SuiteSeparabilityRow] = []
    for ranked in ranking:
        entry = entries[ranked.suite]
        separability = summarize_fn(entry.rl_model, probe_fn(*entry.rl_model.resolve()))
        rows.append(
            SuiteSeparabilityRow(
                suite=ranked.suite,
                display_name=ranked.display_name,
                gemini_pass_at_1=ranked.gemini_pass_at_1,
                looseness_rank=ranked.looseness_rank,
                separability=separability,
            )
        )
    return SuiteComparisonTable(rows=tuple(rows))
