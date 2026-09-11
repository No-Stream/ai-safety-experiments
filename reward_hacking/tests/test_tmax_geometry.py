"""Exercise the TMAX direction-geometry scaffold offline, with a fake probe in place of a 9B load.

Everything in :mod:`reward_hacking.tmax.geometry` above the probe seam is pure Python over floats,
and the module docstring says so by promising it is unit-tested by injecting a fake ``probe_fn``.
This is that test: no torch, no checkpoint, no GPU. The per-layer readings are hand-built objects
satisfying the ``LayerSeparabilityReading`` protocol, so the reductions the two experiment tables
are read off -- peak-layer selection, the mean hack norm, the loosest-first suite ordering, and the
matched-norm placebo column that has to sit beside every reported cosine -- are checked against
numbers we chose rather than against whatever a GPU produced.

The polarity of the peak rule and the sign of ``trend()`` are now pinned here, in
:class:`TestTheSeparabilityPolarity`. They used to be deliberately unpinned, on the grounds that
"should peak maximize or minimize the hack/deception cosine" was open; it is not, because a cosine
near 1 means the two concept directions are the *same* direction, so a maximum over the cosine
selects the most entangled layer and a rising cosine is the two concepts merging. The
assertions in the other classes still read the peak layer back off ``peak_layer`` rather than
hardcoding it, which keeps them about what the peak layer *carries* instead of which layer it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from reward_hacking.tmax.artifacts import (
    BASE_CHECKPOINT,
    SUITES,
    Checkpoint,
    CheckpointStage,
    EnvironmentSuite,
    HFDataset,
    TmaxArtifactError,
    dose_response_ladder,
)
from reward_hacking.tmax.geometry import (
    rank_suites_by_looseness,
    run_dose_response,
    run_suite_comparison,
    summarize_direction_comparisons,
)

if TYPE_CHECKING:
    from reward_hacking.interp.directions import LayerComparison
    from reward_hacking.tmax.geometry import LayerSeparabilityReading

    def the_real_probe_output_satisfies_the_protocol(
        comparison: LayerComparison,
    ) -> LayerSeparabilityReading:
        """Static-only check, which is why it is never called and imports no torch at runtime.

        ``geometry``'s docstring claims the real probe result "flows straight in". Nothing executes
        that path yet, so basedpyright is the only thing that can hold the claim up; this reds if
        ``LayerComparison`` loses a field the protocol reads or stops matching it.
        """
        return comparison


@dataclass(frozen=True)
class FakeLayerReading:
    """A hand-built stand-in for ``interp.directions.LayerComparison``: the same read fields."""

    layer: int
    cos_hack_deception: float
    cos_hack_deception_standardized: float
    cos_hack_placebo: float
    cos_hack_placebo_standardized: float
    hack_norm: float


# Distinct everywhere, and hack norms whose mean (3.0) no single layer carries, so a misread shows.
LAYER_READINGS: tuple[FakeLayerReading, ...] = (
    FakeLayerReading(
        layer=4,
        cos_hack_deception=0.10,
        cos_hack_deception_standardized=0.08,
        cos_hack_placebo=0.004,
        cos_hack_placebo_standardized=0.011,
        hack_norm=1.0,
    ),
    FakeLayerReading(
        layer=12,
        cos_hack_deception=0.70,
        cos_hack_deception_standardized=0.64,
        cos_hack_placebo=0.020,
        cos_hack_placebo_standardized=0.031,
        hack_norm=2.0,
    ),
    FakeLayerReading(
        layer=20,
        cos_hack_deception=0.30,
        cos_hack_deception_standardized=0.22,
        cos_hack_placebo=0.007,
        cos_hack_placebo_standardized=0.052,
        hack_norm=6.0,
    ),
)

PLACEBO_STANDARDIZED_BY_LAYER = {
    reading.layer: reading.cos_hack_placebo_standardized for reading in LAYER_READINGS
}
PLACEBO_RAW_BY_LAYER = {reading.layer: reading.cos_hack_placebo for reading in LAYER_READINGS}

# The separable layer is layer 4 (standardized cosine 0.08 -> separability 0.92), not layer 12.
SEPARABLE_LAYER = 4
PEAK_SEPARABILITY = 0.92

# Standardized and raw disagree on the least-collinear layer, and layer 3 is anti-parallel.
MIXED_SIGN_READINGS: tuple[FakeLayerReading, ...] = (
    FakeLayerReading(
        layer=3,
        cos_hack_deception=-0.40,
        cos_hack_deception_standardized=-0.95,
        cos_hack_placebo=0.006,
        cos_hack_placebo_standardized=0.013,
        hack_norm=1.0,
    ),
    FakeLayerReading(
        layer=9,
        cos_hack_deception=0.66,
        cos_hack_deception_standardized=0.10,
        cos_hack_placebo=0.021,
        cos_hack_placebo_standardized=0.034,
        hack_norm=2.0,
    ),
    FakeLayerReading(
        layer=17,
        cos_hack_deception=0.12,
        cos_hack_deception_standardized=0.64,
        cos_hack_placebo=0.008,
        cos_hack_placebo_standardized=0.049,
        hack_norm=3.0,
    ),
)


def one_layer_reading(cos_standardized: float, *, layer: int = 12) -> tuple[FakeLayerReading, ...]:
    """A single-layer reading set at a chosen standardized cosine, for the trend assertions."""
    return (
        FakeLayerReading(
            layer=layer,
            cos_hack_deception=cos_standardized,
            cos_hack_deception_standardized=cos_standardized,
            cos_hack_placebo=0.01,
            cos_hack_placebo_standardized=0.02,
            hack_norm=2.0,
        ),
    )


class RecordingProbe:
    """A ``probe_fn`` that hands back fixed readings and remembers every checkpoint it was asked."""

    def __init__(self, readings: tuple[FakeLayerReading, ...] = LAYER_READINGS) -> None:
        self.readings = readings
        self.calls: list[tuple[str, str]] = []

    def __call__(self, repo_id: str, revision: str) -> tuple[FakeLayerReading, ...]:
        self.calls.append((repo_id, revision))
        return self.readings


class LadderProbe:
    """A ``probe_fn`` handing back a different reading set per call, in the ladder's own order.

    ``run_dose_response`` probes the rungs in the order it was given (pinned by
    ``test_every_rung_is_probed_once_in_ladder_order``), so a positional sequence is enough to give
    a ladder a base rung and a final rung that differ -- which is what a trend needs.
    """

    def __init__(self, *reading_sets: tuple[FakeLayerReading, ...]) -> None:
        self.reading_sets = reading_sets
        self.calls: list[tuple[str, str]] = []

    def __call__(self, repo_id: str, revision: str) -> tuple[FakeLayerReading, ...]:
        self.calls.append((repo_id, revision))
        return self.reading_sets[len(self.calls) - 1]


class TestTheSeparabilityPolarity:
    """A cosine near 1 means the two directions are one axis, so separability falls as it rises.

    The convention is the repo's own probe rather than a choice made here.
    ``reward_hacking/tests/test_interp_directions.py::test_aligned_concepts_are_high_and_orthogonal_are_zero``
    pins ``cos_hack_deception`` at ~1.0 when both concepts carry the SAME planted direction (its
    line 227) and at ~0.0 when they are planted on disjoint axes (its line 237). So the depth at
    which the two concepts are most distinct is the one whose cosine sits closest to zero, and a
    cosine climbing along the RL ladder is the reward-hack and deception axes collapsing together.

    ``summarize_direction_comparisons`` read that backwards until this round: it took ``max`` over
    the cosine while its docstring called the result "most distinct", and ``trend`` called a rising
    cosine "the axis separating further with training". Both signs are asserted here.
    """

    def test_the_peak_layer_is_the_least_collinear_layer_not_the_most(self) -> None:
        reading = summarize_direction_comparisons(BASE_CHECKPOINT, LAYER_READINGS)
        assert reading.peak_layer == SEPARABLE_LAYER
        assert reading.peak_cos_hack_deception_standardized == pytest.approx(0.08)
        assert reading.peak_separability_standardized == pytest.approx(PEAK_SEPARABILITY)

    def test_an_anti_parallel_layer_is_collinear_and_the_standardized_column_decides(self) -> None:
        """Two guards at once, because a naive sign flip passes neither.

        The rule is ``max(1 - abs(standardized))``: smallest ABSOLUTE cosine, on the STANDARDIZED
        column. Both wrong rules are excluded here, and note each is wrong in a different way, which
        is why the docstring has to name the quantity every time. Taking the SIGNED minimum of either
        column picks layer 3 (standardized -0.95), and an anti-parallel pair is one axis read
        backwards -- perfectly entangled, not maximally distinct. Taking the smallest absolute value
        of the RAW column picks layer 17 (raw 0.12), the massive-activation-inflated reading the
        standardized column exists to replace. Layer 9 wins on the actual rule. (Until 2026-08-24
        this said "minimizing the raw cosine" and "minimizing the *raw* column" for those two, which
        used "minimizing" in the signed sense in one sentence and the absolute sense in the other.)
        """
        reading = summarize_direction_comparisons(BASE_CHECKPOINT, MIXED_SIGN_READINGS)
        assert reading.peak_layer == 9
        assert reading.peak_separability_standardized == pytest.approx(0.90)

    def test_a_cosine_that_climbs_along_the_ladder_is_a_negative_trend(self) -> None:
        """The two concepts merged, which under the old sign read as "separating further"."""
        ladder = dose_response_ladder()[:2]
        probe = LadderProbe(one_layer_reading(0.10), one_layer_reading(0.60))
        table = run_dose_response(ladder, suite="tmax_15k", probe_fn=probe)
        assert probe.calls == [checkpoint.resolve() for checkpoint in ladder]
        assert table.trend() == pytest.approx(-0.50)

    def test_a_cosine_that_falls_along_the_ladder_is_a_positive_trend(self) -> None:
        table = run_dose_response(
            dose_response_ladder()[:2],
            suite="tmax_15k",
            probe_fn=LadderProbe(one_layer_reading(0.60), one_layer_reading(0.10)),
        )
        assert table.trend() == pytest.approx(0.50)

    def test_the_dose_response_table_renders_separability_beside_the_cosine(self) -> None:
        """The rendered table is where a sign gets misread, so separability is a column in it."""
        table = run_dose_response((BASE_CHECKPOINT,), suite="tmax_15k", probe_fn=RecordingProbe())
        rendered = table.render()
        assert "sep_z(h,d)" in rendered
        assert f"{PEAK_SEPARABILITY:.4f}" in rendered

    def test_the_suite_comparison_table_renders_separability_beside_the_cosine(self) -> None:
        table = run_suite_comparison(
            probe_fn=RecordingProbe(), suites=[SUITES["endless_terminals"]]
        )
        rendered = table.render()
        assert "sep_z(h,d)" in rendered
        assert f"{PEAK_SEPARABILITY:.4f}" in rendered


class TestThePlaceboReachesTheReportedTable:
    """The matched-norm placebo is computed by the probe for free, so it must not be dropped here.

    On a cosine the placebo is a drift check rather than a magnitude control -- any random direction
    sits near zero by construction -- so what it buys is a floor that would visibly stop being a
    floor if the capture or the standardization changed at some checkpoint. The assertions
    read the peak layer off ``peak_layer`` so they stay about what that layer carries;
    :class:`TestTheSeparabilityPolarity` is where which layer it is gets pinned.
    """

    def test_the_peak_layer_s_placebo_cosines_are_carried_onto_the_summary(self) -> None:
        reading = summarize_direction_comparisons(BASE_CHECKPOINT, LAYER_READINGS)
        assert reading.peak_cos_hack_placebo_standardized == pytest.approx(
            PLACEBO_STANDARDIZED_BY_LAYER[reading.peak_layer]
        )
        assert reading.peak_cos_hack_placebo == pytest.approx(
            PLACEBO_RAW_BY_LAYER[reading.peak_layer]
        )

    def test_the_carried_placebo_is_one_layer_s_and_not_a_mean_or_a_constant(self) -> None:
        """Every layer's placebo differs, so carrying the wrong one shows up as a mismatch."""
        reading = summarize_direction_comparisons(BASE_CHECKPOINT, LAYER_READINGS)
        others = {
            value
            for layer, value in PLACEBO_STANDARDIZED_BY_LAYER.items()
            if layer != reading.peak_layer
        }
        assert reading.peak_cos_hack_placebo_standardized not in others

    def test_the_dose_response_table_renders_the_placebo_beside_the_cosine(self) -> None:
        table = run_dose_response((BASE_CHECKPOINT,), suite="tmax_15k", probe_fn=RecordingProbe())
        rendered = table.render()
        expected = f"{table.rows[0].peak_cos_hack_placebo_standardized:.4f}"
        assert "cos_z(h,p)" in rendered
        assert expected in rendered

    def test_the_suite_comparison_table_renders_the_placebo_beside_the_cosine(self) -> None:
        table = run_suite_comparison(
            probe_fn=RecordingProbe(), suites=[SUITES["endless_terminals"]]
        )
        rendered = table.render()
        expected = f"{table.rows[0].separability.peak_cos_hack_placebo_standardized:.4f}"
        assert "cos_z(h,p)" in rendered
        assert expected in rendered


class TestSummarizingOneCheckpoint:
    """The per-checkpoint reduction: what the peak layer carries, whichever layer that is."""

    def test_the_mean_hack_norm_is_over_every_layer_not_the_peak_layer_s(self) -> None:
        reading = summarize_direction_comparisons(BASE_CHECKPOINT, LAYER_READINGS)
        assert reading.n_layers == 3
        assert reading.mean_hack_norm == pytest.approx(3.0)

    def test_the_peak_layer_s_own_cosines_travel_together(self) -> None:
        """The raw and standardized cosines must come from the same layer as ``peak_layer``."""
        reading = summarize_direction_comparisons(BASE_CHECKPOINT, LAYER_READINGS)
        (peak,) = [item for item in LAYER_READINGS if item.layer == reading.peak_layer]
        assert reading.peak_cos_hack_deception == pytest.approx(peak.cos_hack_deception)
        assert reading.peak_cos_hack_deception_standardized == pytest.approx(
            peak.cos_hack_deception_standardized
        )

    def test_the_checkpoint_provenance_is_kept(self) -> None:
        reading = summarize_direction_comparisons(BASE_CHECKPOINT, LAYER_READINGS)
        assert reading.checkpoint_name == BASE_CHECKPOINT.name
        assert reading.model_ref == BASE_CHECKPOINT.model_ref
        assert reading.rl_step == 0

    def test_no_per_layer_readings_raises_and_names_the_checkpoint(self) -> None:
        with pytest.raises(ValueError, match=BASE_CHECKPOINT.name):
            summarize_direction_comparisons(BASE_CHECKPOINT, [])


class TestTheDoseResponseLadder:
    """Experiment 1's glue: probe each rung in order, keep the ladder's ordering in the rows."""

    def test_every_rung_is_probed_once_in_ladder_order(self) -> None:
        probe = RecordingProbe()
        ladder = dose_response_ladder()
        table = run_dose_response(ladder, suite="tmax_15k", probe_fn=probe)
        assert probe.calls == [checkpoint.resolve() for checkpoint in ladder]
        assert [row.checkpoint_name for row in table.rows] == [item.name for item in ladder]
        assert [row.rl_step for row in table.rows] == [item.rl_step for item in ladder]

    def test_a_one_rung_ladder_reports_a_trend_of_zero_not_an_absent_trend(self) -> None:
        """``trend()`` returns 0.0 for one rung -- the same value a genuinely flat ladder gives.

        Named ``..._has_no_trend_to_report`` until 2026-08-24, which reads as "returns nothing".
        Nothing in the readout distinguishes *unmeasurable* from *measured flat*, so a one-rung run
        and a two-rung run that did not move are indistinguishable downstream. Whether ``trend()``
        should return ``None`` for a single rung, as ``patch_readout`` does for a zero denominator
        rather than fabricating a zero, is an open behaviour question:
        docs/scratch/test-prose-contradiction-ledger-2026-08-24.md, T11.
        """
        table = run_dose_response((BASE_CHECKPOINT,), suite="tmax_15k", probe_fn=RecordingProbe())
        assert table.trend() == 0.0
        assert len(table.rows) == 1

    def test_an_unverified_checkpoint_fails_before_the_probe_is_ever_called(self) -> None:
        """The resolve happens in this module, so a bad id never reaches a multi-gigabyte load."""
        probe = RecordingProbe()
        unverified = Checkpoint(
            name="tb2_eval_placeholder",
            repo_id="UNVERIFIED-placeholder",
            stage=CheckpointStage.RL,
            verified=False,
        )
        with pytest.raises(TmaxArtifactError, match="tb2_eval_placeholder"):
            run_dose_response((BASE_CHECKPOINT, unverified), suite="tmax_15k", probe_fn=probe)
        assert probe.calls == [BASE_CHECKPOINT.resolve()]


def _suite(name: str, pass_at_1: float) -> EnvironmentSuite:
    """A throwaway suite entry pointing at a verified repo id, for ordering assertions."""
    return EnvironmentSuite(
        name=name,
        display_name=name.replace("_", " ").title(),
        rl_model=Checkpoint(name=f"{name}_rl", repo_id=f"allenai/{name}"),
        rl_dataset=HFDataset(name=name, repo_id=f"allenai/{name}-data"),
        gemini_pass_at_1=pass_at_1,
    )


class TestTheSuiteLoosenessOrdering:
    """Experiment 2 reads the geometry against a looseness ordering, so the ordering has to hold."""

    def test_the_registered_suites_rank_loosest_first(self) -> None:
        ranking = rank_suites_by_looseness()
        assert [item.looseness_rank for item in ranking] == list(range(len(SUITES)))
        assert ranking[0].suite == "endless_terminals"
        assert ranking[0].gemini_pass_at_1 == pytest.approx(0.92)
        assert ranking[-1].suite == "cli_gym"

    def test_a_tie_on_the_pass_rate_breaks_by_suite_name(self) -> None:
        ranking = rank_suites_by_looseness(
            [_suite("zulu_suite", 0.5), _suite("alpha_suite", 0.5), _suite("looser", 0.9)]
        )
        assert [item.suite for item in ranking] == ["looser", "alpha_suite", "zulu_suite"]

    def test_the_comparison_rows_follow_the_looseness_order_not_the_registry_order(self) -> None:
        probe = RecordingProbe()
        table = run_suite_comparison(probe_fn=probe, suites=list(SUITES.values()))
        assert [row.suite for row in table.rows] == [
            item.suite for item in rank_suites_by_looseness()
        ]
        assert probe.calls == [SUITES[row.suite].rl_model.resolve() for row in table.rows]
        assert [row.looseness_rank for row in table.rows] == list(range(len(SUITES)))
