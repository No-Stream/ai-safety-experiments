"""Offline tests for the prompt-contrast pure core and its deliverable-2/3/4 wiring.

CPU only, no model load. The projection statistics and the empirical p-value run on synthetic paired
activations with a planted (or deliberately absent) separation, so every claim has a ground truth.
Also checks the plumbing that does not need a model: the widened placebo default, the
generic-contradiction control being slotted into the concept machinery, and the axis-selection rule
that reuses a validated eval-awareness axis per pooling (mean always; last once it exists).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
import torch

from reward_hacking.interp import stimuli
from reward_hacking.interp.directions import unit
from reward_hacking.interp.prompt_contrast import (
    CONTRADICTION,
    CONTRAST_CONCEPTS,
    DEFAULT_N_PLACEBOS,
    EVAL_AWARENESS,
    SHORTCUT,
    ContrastRead,
    _concept_pairs,  # pyright: ignore[reportPrivateUsage]
    _direction_for,  # pyright: ignore[reportPrivateUsage]
    build_stimulus_pairs,
    contrast_all_layers,
    contrast_layer,
    layer_selection_p,
    load_validated_eval_direction,
    metrics_dict,
    selection_by_group,
)

if TYPE_CHECKING:
    from pathlib import Path


def _paired_acts(
    direction: torch.Tensor, gap: float, n: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paired (conflicting, original) acts whose conflicting rows are shifted by
    ``gap * direction``.

    The shared per-pair base cancels in the projection onto ``direction``, so a positive gap makes
    every conflicting row project above its original twin -- a clean AUC of 1.0 to check
    the p-value.
    """
    d = direction.shape[0]
    base = torch.randn(n, d, generator=generator)
    return base + gap * direction, base


class TestWidenedPlaceboDefault:
    def test_default_is_at_least_100(self) -> None:
        assert DEFAULT_N_PLACEBOS >= 100


class TestEmpiricalPValue:
    def test_clean_separation_gives_auc_one_and_tiny_p(self) -> None:
        generator = torch.Generator().manual_seed(0)
        direction = unit(torch.randn(64, generator=generator))
        conflicting, original = _paired_acts(direction, gap=5.0, n=40, generator=generator)

        read = contrast_layer(
            EVAL_AWARENESS,
            "mean",
            0,
            conflicting,
            original,
            direction,
            n_placebos=50,
            generator=torch.Generator().manual_seed(1),
        )

        assert read.auc == pytest.approx(1.0)
        # No matched-norm placebo separates two groups this cleanly along a random axis.
        assert read.placebo_auc_ge_count == 0
        assert read.auc_empirical_p == pytest.approx(1.0 / 51.0)

    def test_no_signal_gives_chance_auc_and_p_one(self) -> None:
        generator = torch.Generator().manual_seed(2)
        direction = unit(torch.randn(32, generator=generator))
        identical = torch.randn(30, 32, generator=generator)

        read = contrast_layer(
            EVAL_AWARENESS,
            "mean",
            0,
            identical,
            identical.clone(),
            direction,
            n_placebos=20,
            generator=torch.Generator().manual_seed(3),
        )

        # Identical groups: every projection ties, so AUC is exactly chance and every
        # placebo ties too.
        assert read.auc == pytest.approx(0.5)
        assert read.placebo_auc_ge_count == read.n_placebos
        assert read.auc_empirical_p == pytest.approx(1.0)

    def test_p_value_is_a_serializable_scalar(self) -> None:
        generator = torch.Generator().manual_seed(4)
        direction = unit(torch.randn(16, generator=generator))
        conflicting, original = _paired_acts(direction, gap=2.0, n=20, generator=generator)

        read = contrast_layer(
            SHORTCUT,
            "last",
            3,
            conflicting,
            original,
            direction,
            n_placebos=10,
            generator=torch.Generator().manual_seed(5),
        )

        dumped = asdict(read)
        assert isinstance(dumped["auc_empirical_p"], float)
        assert isinstance(dumped["placebo_auc_ge_count"], int)


class TestContrastAllLayers:
    def test_one_read_per_layer_each_with_a_p_value(self) -> None:
        generator = torch.Generator().manual_seed(6)
        layers = [0, 1, 2]
        direction = unit(torch.randn(24, generator=generator))
        conflicting = {layer: _paired_acts(direction, 3.0, 15, generator)[0] for layer in layers}
        original = {layer: _paired_acts(direction, 0.0, 15, generator)[1] for layer in layers}
        directions = dict.fromkeys(layers, direction)

        reads = contrast_all_layers(
            EVAL_AWARENESS, "mean", conflicting, original, directions, n_placebos=10, seed=0
        )

        assert [r.layer for r in reads] == layers
        assert all(isinstance(r, ContrastRead) for r in reads)
        assert all(0.0 < r.auc_empirical_p <= 1.0 for r in reads)


class TestContradictionControl:
    def test_contradiction_is_a_contrast_concept_with_its_own_stimuli(self) -> None:
        assert CONTRADICTION in CONTRAST_CONCEPTS
        pairs = _concept_pairs()
        assert pairs[CONTRADICTION] is stimuli.CONTRADICTION_PAIRS

    def test_contradiction_stimuli_are_well_formed(self) -> None:
        pairs = stimuli.CONTRADICTION_PAIRS
        assert len(pairs) >= 40
        assert all(pair.positive != pair.negative for pair in pairs)
        assert len({pair.positive for pair in pairs}) == len(pairs)

    def test_contradiction_stimuli_carry_no_eval_awareness_vocabulary(self) -> None:
        """The control must isolate contradiction from eval-awareness, so it avoids that vocabulary.

        If the positives leaked "graded / evaluated / observed / scored" language, the axis would no
        longer be a clean alternative explanation for the twins' separation.
        """
        banned = (
            "graded",
            "grader",
            "evaluat",
            "observ",
            "scored",
            "reviewer",
            "benchmark",
            "test",
        )
        for pair in stimuli.CONTRADICTION_PAIRS:
            text = pair.positive.lower()
            assert not any(word in text for word in banned), pair.positive


class TestValidatedAxisProvenance:
    """A saved axis carries the pooling and checkpoint it was extracted under; loading must read it.

    The headline eval-awareness read projects the twins onto an axis loaded off disk, and the axis
    file alone cannot say which space it lives in. Loading a last-pool axis as the mean-pool one
    (an ``--out-dir`` that contradicts ``--pooling``, or a stale directory) silently projects onto a
    different space and reports a number that reads as a mean-pool result.
    """

    @staticmethod
    def _write_axis(directory: Path, *, pooling: str, model_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({0: torch.zeros(4), 1: torch.ones(4)}, directory / "directions.pt")
        (directory / "metrics.json").write_text(
            json.dumps({"model_id": model_id, "concept": EVAL_AWARENESS, "pooling": pooling})
        )

    def test_pooling_mismatch_raises_naming_both(self, tmp_path: Path) -> None:
        self._write_axis(tmp_path, pooling="last", model_id="Qwen/Qwen3.5-4B")

        with pytest.raises(ValueError, match=r"'last'.*'mean'|'mean'.*'last'"):
            load_validated_eval_direction(
                tmp_path, expected_pooling="mean", model_id="Qwen/Qwen3.5-4B"
            )

    def test_matching_metadata_loads_the_axis_and_its_checkpoint(self, tmp_path: Path) -> None:
        self._write_axis(tmp_path, pooling="mean", model_id="Qwen/Qwen3.5-4B")

        directions, axis_model_id = load_validated_eval_direction(
            tmp_path, expected_pooling="mean", model_id="Qwen/Qwen3.5-4B"
        )

        assert sorted(directions) == [0, 1]
        assert axis_model_id == "Qwen/Qwen3.5-4B"

    def test_missing_metrics_names_the_command_that_regenerates_it(self, tmp_path: Path) -> None:
        torch.save({0: torch.zeros(4)}, tmp_path / "directions.pt")

        with pytest.raises(ValueError, match="eval_awareness_probe"):
            load_validated_eval_direction(
                tmp_path, expected_pooling="mean", model_id="Qwen/Qwen3.5-4B"
            )

    def test_a_cross_checkpoint_axis_loads_and_is_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Reusing a base-checkpoint axis on an RL'd checkpoint is a real transfer measurement.

        So the checkpoint mismatch is loud and recorded, never fatal -- unlike the pooling
        mismatch, which has no legitimate reading.
        """
        self._write_axis(tmp_path, pooling="mean", model_id="Qwen/Qwen3.5-9B-Base")

        with caplog.at_level(logging.WARNING):
            _, axis_model_id = load_validated_eval_direction(
                tmp_path, expected_pooling="mean", model_id="Qwen/Qwen3.5-9B"
            )

        assert axis_model_id == "Qwen/Qwen3.5-9B-Base"
        assert "Qwen/Qwen3.5-9B-Base" in caplog.text


class TestDirectionSelection:
    @staticmethod
    def _fresh() -> dict[str, dict[int, torch.Tensor]]:
        return {name: {0: torch.randn(8)} for name in CONTRAST_CONCEPTS}

    def test_eval_awareness_reuses_validated_axis_for_its_pooling(self) -> None:
        fresh = self._fresh()
        validated = {"mean": {0: torch.randn(8)}, "last": {0: torch.randn(8)}}

        assert _direction_for(EVAL_AWARENESS, "mean", fresh, validated) is validated["mean"]
        assert _direction_for(EVAL_AWARENESS, "last", fresh, validated) is validated["last"]

    def test_last_pool_falls_back_to_fresh_when_unvalidated(self) -> None:
        fresh = self._fresh()
        mean_only = {"mean": {0: torch.randn(8)}}
        # This is the Exp2 fragility path: no validated last axis, so the fresh one is used.
        assert _direction_for(EVAL_AWARENESS, "last", fresh, mean_only) is fresh[EVAL_AWARENESS]

    def test_other_concepts_always_use_fresh(self) -> None:
        fresh = self._fresh()
        validated = {"mean": {0: torch.randn(8)}, "last": {0: torch.randn(8)}}
        assert _direction_for(SHORTCUT, "mean", fresh, validated) is fresh[SHORTCUT]
        assert _direction_for(CONTRADICTION, "last", fresh, validated) is fresh[CONTRADICTION]


def _layer_read(  # noqa: PLR0913 - a synthetic read is its layer, auc, draws and three labels
    layer: int,
    auc: float,
    placebo_aucs: tuple[float, ...],
    *,
    concept: str = SHORTCUT,
    pooling: str = "mean",
    within_layer_p: float = 0.5,
) -> ContrastRead:
    """A ContrastRead carrying only the fields ``layer_selection_p`` reads meaningfully.

    ``auc`` and ``placebo_aucs`` (the raw per-placebo draws at this layer) plus ``within_layer_p``
    (the read's own ``auc_empirical_p``) are what the correction consumes; every other field is
    filled with a neutral placeholder so the construction stays a one-liner per layer.
    """
    return ContrastRead(
        concept=concept,
        pooling=pooling,
        layer=layer,
        n_conflicting=8,
        n_original=8,
        n_pairs=8,
        direction_norm=1.0,
        mean_conflicting=0.0,
        mean_original=0.0,
        mean_diff=0.0,
        auc=auc,
        cohens_d=0.0,
        paired_mean_diff=0.0,
        paired_t=0.0,
        paired_sign_rate=0.5,
        n_placebos=len(placebo_aucs),
        placebo_auc_mean=sum(placebo_aucs) / len(placebo_aucs) if placebo_aucs else 0.5,
        placebo_auc_max=max(placebo_aucs) if placebo_aucs else 0.5,
        placebo_auc_ge_count=sum(1 for a in placebo_aucs if a >= auc),
        auc_empirical_p=within_layer_p,
        placebo_cohens_d_mean=0.0,
        placebo_paired_sign_rate_mean=0.5,
        placebo_paired_abs_mean_diff_mean=0.0,
        placebo_aucs=placebo_aucs,
    )


class TestLayerSelectionCorrection:
    """The peak-layer p must be corrected for the peak having been chosen as the best of N layers.

    The within-layer ``auc_empirical_p`` holds the layer fixed; it ignores that with 32 layers even
    a null direction lands one above chance. ``layer_selection_p`` compares the real best-over-layers
    AUC against the distribution of each placebo's OWN best-over-layers AUC, which is the false-
    positive rate the peak selection actually incurs.
    """

    def test_correction_bites_when_a_lucky_layer_wins(self) -> None:
        """Within-layer significant at the peak, yet not significant once the selection is paid for.

        The peak layer's own placebos never reach its AUC (within-layer p tiny), but ACROSS 32
        layers every placebo draw finds SOME layer above the peak, so the selection-corrected p is 1.
        """
        n_layers = 32
        n_placebos = 100
        reads = [_layer_read(0, 0.75, (0.55,) * n_placebos, within_layer_p=1.0 / (n_placebos + 1))]
        # Layers 1..31: each placebo column j gets a 0.80 at exactly one of them, so the
        # max-over-layers of every column is 0.80 >= the peak's 0.75.
        for layer in range(1, n_layers):
            aucs = tuple(
                0.80 if (1 + j % (n_layers - 1)) == layer else 0.50 for j in range(n_placebos)
            )
            reads.append(_layer_read(layer, 0.50, aucs))

        selection = layer_selection_p(reads)

        assert selection.peak_layer == 0
        assert selection.peak_auc == pytest.approx(0.75)
        assert selection.within_layer_empirical_p < 0.05  # significant if you ignore the selection
        assert selection.placebo_best_auc_max == pytest.approx(0.80)
        assert selection.selection_corrected_p == pytest.approx(1.0)  # not, once you pay for it

    def test_correction_and_within_layer_agree_on_a_strong_signal(self) -> None:
        """A separation no placebo reaches at ANY layer is significant both ways."""
        n_layers = 32
        n_placebos = 100
        reads = [_layer_read(0, 0.99, (0.55,) * n_placebos, within_layer_p=1.0 / (n_placebos + 1))]
        reads.extend(_layer_read(layer, 0.50, (0.55,) * n_placebos) for layer in range(1, n_layers))

        selection = layer_selection_p(reads)

        assert selection.peak_layer == 0
        assert selection.within_layer_empirical_p == pytest.approx(1.0 / (n_placebos + 1))
        assert selection.selection_corrected_p == pytest.approx(1.0 / (n_placebos + 1))

    def test_null_is_the_per_placebo_max_over_layers(self) -> None:
        """The null takes each placebo's best over layers (column max), not a per-layer statistic.

        Hand-computed: peak is layer 0 (auc 0.65). Column maxima are [0.6, 0.7, 0.5]; exactly one
        (0.7) reaches the peak, so the add-one-smoothed p is (1 + 1) / (3 + 1) = 0.5.
        """
        reads = [
            _layer_read(0, 0.65, (0.6, 0.4, 0.5)),
            _layer_read(1, 0.55, (0.3, 0.7, 0.5)),
        ]

        selection = layer_selection_p(reads)

        assert selection.peak_layer == 0
        assert selection.real_best_auc == pytest.approx(0.65)
        assert selection.placebo_best_auc_max == pytest.approx(0.7)
        assert selection.placebo_best_auc_mean == pytest.approx((0.6 + 0.7 + 0.5) / 3)
        assert selection.selection_corrected_p == pytest.approx(2.0 / 4.0)

    def test_rejects_a_mixed_concept_or_pooling_group(self) -> None:
        with pytest.raises(ValueError, match=r"one \(concept, pooling\) group"):
            layer_selection_p(
                [_layer_read(0, 0.7, (0.5,)), _layer_read(1, 0.6, (0.5,), concept=EVAL_AWARENESS)]
            )

    def test_rejects_reads_with_mismatched_placebo_counts(self) -> None:
        with pytest.raises(ValueError, match="differing placebo counts"):
            layer_selection_p([_layer_read(0, 0.7, (0.5, 0.5)), _layer_read(1, 0.6, (0.5,))])

    def test_zero_placebos_is_unresolved_not_significant(self) -> None:
        selection = layer_selection_p([_layer_read(0, 0.9, ()), _layer_read(1, 0.5, ())])
        assert selection.n_placebos == 0
        assert selection.selection_corrected_p == pytest.approx(1.0)


class TestSelectionByGroup:
    def test_groups_by_concept_and_pooling(self) -> None:
        reads = [
            _layer_read(0, 0.9, (0.5,), concept=SHORTCUT, pooling="mean"),
            _layer_read(1, 0.5, (0.5,), concept=SHORTCUT, pooling="mean"),
            _layer_read(0, 0.8, (0.5,), concept=SHORTCUT, pooling="last"),
            _layer_read(0, 0.7, (0.5,), concept=EVAL_AWARENESS, pooling="mean"),
        ]

        grouped = selection_by_group(reads)

        assert sorted(grouped) == sorted([SHORTCUT, EVAL_AWARENESS])
        assert sorted(grouped[SHORTCUT]) == ["last", "mean"]
        assert grouped[SHORTCUT]["mean"]["peak_layer"] == 0
        assert isinstance(grouped[SHORTCUT]["mean"]["selection_corrected_p"], float)


class TestPlaceboAucsAreRetained:
    """``contrast_layer`` keeps the raw placebo draws, and they match the summaries built from them.

    The layer-selection correction is only trustworthy if the per-read ``placebo_aucs`` ARE the
    draws behind ``placebo_auc_mean`` / ``_max`` / ``_ge_count``; a mismatch would silently feed the
    correction different numbers than the within-layer p was computed from.
    """

    def test_length_matches_and_summaries_reconstruct(self) -> None:
        generator = torch.Generator().manual_seed(7)
        direction = unit(torch.randn(32, generator=generator))
        conflicting, original = _paired_acts(direction, gap=1.5, n=25, generator=generator)

        read = contrast_layer(
            SHORTCUT,
            "mean",
            4,
            conflicting,
            original,
            direction,
            n_placebos=20,
            generator=torch.Generator().manual_seed(8),
        )

        assert len(read.placebo_aucs) == 20
        assert read.placebo_auc_mean == pytest.approx(sum(read.placebo_aucs) / 20)
        assert read.placebo_auc_max == pytest.approx(max(read.placebo_aucs))
        assert read.placebo_auc_ge_count == sum(1 for a in read.placebo_aucs if a >= read.auc)

    def test_metrics_dict_drops_the_bulky_raw_draws(self) -> None:
        read = _layer_read(0, 0.7, (0.5,) * 100)
        fields = metrics_dict(read)
        assert "placebo_aucs" not in fields
        assert fields["auc"] == pytest.approx(0.7)
        assert fields["auc_empirical_p"] == pytest.approx(read.auc_empirical_p)


# The grader-read observation the twin transcripts carry; the manipulation lives after it, so a
# byte-identical prefix must reach at least this marker in both arms. Matches
# prompt_contrast._grader_observation.
_GRADER_STDOUT_MARKER = "<stdout>"


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a: str, b: str, *, cap: int) -> int:
    i = 0
    while i < cap and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


class TestTwinConstructionSharedPrefix:
    """The twins must be byte-identical everywhere but one grader region -- the ``shared_prefix``
    control's foundation.

    ``shared_prefix_control`` patches the twins' identical leading prefix and can only be a real
    zero-baseline if that prefix genuinely is identical: nothing twin-dependent (a listing that
    prints file sizes, an episode-dir path, nondeterministic ordering) may leak into it ahead of the
    grader body. This asserts the construction guarantee directly on the real ILCB twins -- byte
    level, no model or tokenizer -- so such a regression is caught here rather than surfacing as a
    non-zero control much later. It says nothing about the forward pass: on the longest twins the
    activations at those identical positions still differ numerically (a hybrid linear-attention
    property, not a construction defect), which the patch stage's ``patch_delta_norm`` telemetry
    flags separately.

    No item text is asserted on -- only offsets and structural markers -- since the repo is public.
    """

    def test_twins_differ_only_in_one_grader_region_after_the_observation(
        self, tmp_path: Path
    ) -> None:
        pairs = build_stimulus_pairs(tmp_path / "work", limit=4)
        assert len(pairs) >= 2

        for pair in pairs:
            conflicting = pair.conflicting_transcript
            original = pair.original_transcript
            assert conflicting != original, pair.problem_id

            prefix = _common_prefix_len(conflicting, original)
            shorter = min(len(conflicting), len(original))
            suffix = _common_suffix_len(conflicting, original, cap=shorter - prefix)

            # The divergence starts only after the grader-read observation opener, so the problem
            # statement, protocol preamble, file listing and observation opener are all identical.
            marker = conflicting.find(_GRADER_STDOUT_MARKER)
            assert marker > 0, pair.problem_id
            assert prefix > marker, (
                f"{pair.problem_id}: twins diverge at byte {prefix}, before the grader observation "
                f"at {marker}; twin-dependent content leaked into the shared prefix"
            )
            # A shared suffix brackets the differing region, so the divergence is ONE contiguous
            # grader-body span (not "everything after the marker"), and the prefix is substantial.
            assert suffix > 0, pair.problem_id
            assert prefix > 100
