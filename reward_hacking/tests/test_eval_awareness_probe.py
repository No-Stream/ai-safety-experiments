"""Offline tests for the eval-awareness direction validation.

CPU only, no model load. The pure-tensor core is exercised on synthetic activations with a KNOWN
planted separating direction, so every claim has a ground-truth answer:

* the 1-D nearest-centroid classifier gets the hand-computed answer, including when the positive
  class projects *lower* (the sign-orientation guard a naive ``> threshold`` would fail);
* a planted separating axis clears the placebo by a wide margin, and the placebo sits at chance;
* the classifier orients on the training fold, so a *negative* planted signal still separates;
* the held-out accuracy is honest -- with no planted signal it collapses to chance rather than
  staying near 1, which is the check a bug that evaluated the direction on its own training pairs
  would fail;
* ``validate_layer`` recovers the planted direction and reports probe accuracy above its null and
  a diff-of-means axis above its placebo;
* the battery is concept-agnostic (the DEFECT-D fix): ``probe_concept`` runs the identical
  validation for shortcut / deception / eval-awareness / contradiction and tags the result with its
  concept, and ``save_artifacts`` records that concept -- so every concept axis, not just
  eval-awareness, ships a complete probe/quality metrics block.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import TYPE_CHECKING

import pytest
import torch

from reward_hacking.interp import eval_awareness_probe
from reward_hacking.interp.eval_awareness_probe import (
    ConceptAxisRead,
    ConceptAxisResult,
    direction_separation,
    direction_split_half_cosine,
    nearest_centroid_correct,
    probe_concept,
    projection_cv_accuracy,
    save_artifacts,
    validate_layer,
)
from reward_hacking.interp.linear_probe import CaptureSpec, ConceptActivations, ProbeConfig

if TYPE_CHECKING:
    from pathlib import Path


def _planted_concept(
    generator: torch.Generator, *, n_pairs: int, d: int, signal_axis: int, shift: float
) -> ConceptActivations:
    """Positives and negatives separated by ``2*shift`` along ``signal_axis``, noise elsewhere.

    A negative ``shift`` puts the positive class on the *low* side of the axis, which exercises the
    classifier's training-fold sign orientation. With no other structure, the diff-of-means axis is
    essentially the signal axis, so a placebo (random axis of equal norm) should sit at chance.
    """
    positives = torch.randn(n_pairs, d, generator=generator)
    negatives = torch.randn(n_pairs, d, generator=generator)
    positives[:, signal_axis] += shift
    negatives[:, signal_axis] -= shift
    return ConceptActivations(positives, negatives)


class TestNearestCentroid:
    """The 1-D classifier gets the hand-computed answer for both class orientations."""

    def test_positive_class_projects_high(self) -> None:
        direction = torch.tensor([1.0])
        train_features = torch.tensor([[2.0], [4.0], [-2.0], [-4.0]])
        train_labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        test_features = torch.tensor([[5.0], [-5.0]])
        test_labels = torch.tensor([1.0, 0.0])
        correct = nearest_centroid_correct(
            direction, train_features, train_labels, test_features, test_labels
        )
        assert correct.tolist() == [1.0, 1.0]

    def test_positive_class_projects_low_is_handled_by_orientation(self) -> None:
        """Positives project LOW here; a naive ``proj > threshold`` rule would score 0, the
        sign-orientation fit on train is what recovers it -- watched to stay correct."""
        direction = torch.tensor([1.0])
        train_features = torch.tensor([[-2.0], [-4.0], [2.0], [4.0]])
        train_labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        test_features = torch.tensor([[-5.0], [5.0]])
        test_labels = torch.tensor([1.0, 0.0])
        correct = nearest_centroid_correct(
            direction, train_features, train_labels, test_features, test_labels
        )
        assert correct.tolist() == [1.0, 1.0]


class TestProjectionSeparability:
    """A planted axis separates held-out pairs; the matched-norm placebo sits at chance."""

    def test_planted_axis_beats_placebo(self) -> None:
        generator = torch.Generator().manual_seed(0)
        concept = _planted_concept(generator, n_pairs=60, d=256, signal_axis=0, shift=5.0)
        separation = direction_separation(concept, ProbeConfig(n_folds=5, seed=0), n_placebos=5)
        assert separation.direction_accuracy > 0.9
        assert separation.placebo_accuracy_mean < 0.75
        assert separation.direction_accuracy > separation.placebo_accuracy_max

    def test_negative_planted_signal_still_separates(self) -> None:
        generator = torch.Generator().manual_seed(1)
        concept = _planted_concept(generator, n_pairs=60, d=256, signal_axis=3, shift=-5.0)
        separation = direction_separation(concept, ProbeConfig(n_folds=5, seed=0), n_placebos=5)
        assert separation.direction_accuracy > 0.9

    def test_no_signal_collapses_to_chance(self) -> None:
        """Without a planted signal the held-out accuracy must fall to ~chance. A direction scored
        on the pairs that defined it would instead stay near 1 -- this is that leak's guard."""
        generator = torch.Generator().manual_seed(2)
        concept = _planted_concept(generator, n_pairs=60, d=256, signal_axis=0, shift=0.0)
        separation = direction_separation(concept, ProbeConfig(n_folds=5, seed=0), n_placebos=5)
        assert separation.direction_accuracy < 0.75

    def test_real_direction_arm_matches_identity_transform(self) -> None:
        generator = torch.Generator().manual_seed(3)
        concept = _planted_concept(generator, n_pairs=40, d=128, signal_axis=1, shift=4.0)
        config = ProbeConfig(n_folds=5, seed=0)
        separation = direction_separation(concept, config, n_placebos=3)
        assert separation.direction_accuracy == pytest.approx(
            projection_cv_accuracy(concept, config, lambda direction: direction)
        )


class TestSplitHalf:
    """Split-half cosine is high for a stable planted axis, near zero for pure noise."""

    def test_stable_axis_high_split_half(self) -> None:
        generator = torch.Generator().manual_seed(4)
        concept = _planted_concept(generator, n_pairs=60, d=64, signal_axis=0, shift=6.0)
        assert direction_split_half_cosine(concept) > 0.8

    def test_pure_noise_low_split_half(self) -> None:
        generator = torch.Generator().manual_seed(5)
        concept = _planted_concept(generator, n_pairs=60, d=256, signal_axis=0, shift=0.0)
        assert abs(direction_split_half_cosine(concept)) < 0.4


class TestValidateLayer:
    """End to end on synthetic activations: recover the direction, clear both controls."""

    def test_recovers_direction_and_clears_controls(self) -> None:
        generator = torch.Generator().manual_seed(6)
        concept = _planted_concept(generator, n_pairs=60, d=256, signal_axis=7, shift=5.0)
        read, direction = validate_layer(42, concept, ProbeConfig(n_folds=5, seed=0), n_placebos=5)

        assert read.layer == 42
        assert read.probe_accuracy > read.probe_null_accuracy_max
        assert read.direction_accuracy > read.placebo_accuracy_max
        assert read.clears_null
        assert read.beats_placebo
        # Returned direction is the full-data diff-of-means; its dominant dim is the signal axis.
        assert int(direction.abs().argmax().item()) == 7


# The four concept axes the contrast/causal tiers project onto. Before the DEFECT-D fix only the
# first got a validated metrics block; the battery is concept-agnostic, so all four must.
_FOUR_CONCEPTS = ("eval_awareness", "shortcut", "deception", "contradiction")
_BATTERY_FIELDS = tuple(field.name for field in dataclasses.fields(ConceptAxisRead))


def _synthetic_capture(
    generator: torch.Generator, *, layers: tuple[int, ...] = (3, 7)
) -> dict[int, ConceptActivations]:
    """Two layers of planted-signal activations, standing in for the GPU capture in probe_concept."""
    return {
        layer: _planted_concept(generator, n_pairs=40, d=96, signal_axis=layer % 5, shift=4.0)
        for layer in layers
    }


class TestConceptProbeBattery:
    """DEFECT D: the SAME validation battery runs for every concept, not only eval-awareness.

    ``probe_concept`` is the concept-parametrised entry point the axis-probe stage calls per concept.
    The GPU capture is stubbed so the test is CPU-only; everything downstream (``validate_layer`` per
    layer, the result assembly, the concept tag) is the real code the run executes.
    """

    @pytest.mark.parametrize("concept", _FOUR_CONCEPTS)
    def test_every_concept_gets_a_complete_metrics_block(
        self, concept: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        generator = torch.Generator().manual_seed(11)
        monkeypatch.setattr(
            eval_awareness_probe,
            "capture_concept_activations",
            lambda *args, **kwargs: _synthetic_capture(generator),
        )
        spec = CaptureSpec(concepts=(concept,), pooling="mean")
        result = probe_concept(
            object(),  # pyright: ignore[reportArgumentType]  # model unused: capture is stubbed
            object(),  # pyright: ignore[reportArgumentType]  # tokenizer unused
            spec,
            ProbeConfig(n_folds=5, n_permutations=3, seed=0),
            model_id="stub-model",
            concept=concept,
            pairs=[],
            n_placebos=5,
        )

        assert result.concept == concept
        assert sorted(result.directions) == [3, 7]
        assert len(result.reads) == 2
        for read in result.reads:
            block = dataclasses.asdict(read)
            # Every battery field is present and finite -- a concept that silently lost its metrics
            # (an empty or partial block) fails right here.
            assert set(block) == set(_BATTERY_FIELDS)
            assert all(math.isfinite(block[name]) for name in _BATTERY_FIELDS)

    def test_the_metrics_block_is_identical_in_shape_across_concepts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every concept's read carries the same fields: the battery does not vary by concept."""
        generator = torch.Generator().manual_seed(12)
        monkeypatch.setattr(
            eval_awareness_probe,
            "capture_concept_activations",
            lambda *args, **kwargs: _synthetic_capture(generator, layers=(5,)),
        )
        blocks = {}
        for concept in _FOUR_CONCEPTS:
            result = probe_concept(
                object(),  # pyright: ignore[reportArgumentType]
                object(),  # pyright: ignore[reportArgumentType]
                CaptureSpec(concepts=(concept,), pooling="mean"),
                ProbeConfig(n_folds=5, n_permutations=3, seed=0),
                model_id="stub-model",
                concept=concept,
                pairs=[],
                n_placebos=3,
            )
            blocks[concept] = set(dataclasses.asdict(result.reads[0]))
        assert all(fields == set(_BATTERY_FIELDS) for fields in blocks.values())


class TestSaveArtifactsRecordsConcept:
    """``save_artifacts`` tags the metrics with the concept it validated, not a hardcoded label.

    The pre-fix ``save_artifacts`` wrote a module-level ``CONCEPT = "eval_awareness"`` constant, so a
    shortcut axis saved through it would have been mislabelled eval-awareness. This is the guard.
    """

    def _result(self, concept: str) -> ConceptAxisResult:
        generator = torch.Generator().manual_seed(21)
        by_layer = _synthetic_capture(generator, layers=(4,))
        reads = []
        directions = {}
        for layer, activations in by_layer.items():
            read, direction = validate_layer(
                layer, activations, ProbeConfig(n_folds=5, seed=0), n_placebos=3
            )
            reads.append(read)
            directions[layer] = direction
        return ConceptAxisResult(
            model_id="stub-model",
            concept=concept,
            spec=CaptureSpec(concepts=(concept,), pooling="mean"),
            config=ProbeConfig(n_folds=5, seed=0),
            n_placebos=3,
            reads=reads,
            directions=directions,
            activations=by_layer,
        )

    def test_metrics_json_carries_the_concept_and_a_full_layer_read(self, tmp_path: Path) -> None:
        save_artifacts(self._result("shortcut"), tmp_path)
        metrics = json.loads((tmp_path / "metrics.json").read_text())

        assert metrics["concept"] == "shortcut"
        assert metrics["pooling"] == "mean"
        assert (tmp_path / "directions.pt").exists()
        assert len(metrics["layer_reads"]) == 1
        assert set(metrics["layer_reads"][0]) == set(_BATTERY_FIELDS)

    def test_a_different_concept_is_labelled_differently(self, tmp_path: Path) -> None:
        save_artifacts(self._result("deception"), tmp_path)
        assert json.loads((tmp_path / "metrics.json").read_text())["concept"] == "deception"
