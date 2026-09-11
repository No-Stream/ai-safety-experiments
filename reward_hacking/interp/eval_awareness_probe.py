"""Extract and *validate* a concept's residual-stream direction (eval-awareness by default).

Downstream experiments lean on concept axes in Qwen3.5-4B: projecting agentic-episode activations
onto them and steering along them. None is safe until we know the axis is real and not an artifact,
so this module is the gate that comes first. It extracts the diff-of-means direction per layer and
subjects it to the two controls the repo mandates before any direction is trusted, and it saves the
raw material so adding rigor later is a re-analysis, not a re-run.

The whole battery -- the 1-D projection classifier, the matched-norm placebo, the trained probe and
its shuffled-label null, and the split-half ceiling -- is concept-agnostic: it operates on a
:class:`~reward_hacking.interp.linear_probe.ConceptActivations`, which is just paired positives and
foils. So :func:`probe_concept` / :func:`run_concept_probe` validate ANY concept given its sentence
pairs, and the run harness's ``axis-probe`` stage runs the identical battery over all four concept
axes (shortcut, deception, eval-awareness, the contradiction control) so every axis a contrast
projects onto carries the same quality metrics -- not just eval-awareness.
:func:`run_eval_awareness_probe` is the eval-awareness-specialised entry point kept for its CLI and
the validated-axis directory the contrast stage loads.

**Transfer caveat, load-bearing.** The eval-awareness stimulus set is mostly hand-authored (only 3
of 82 pairs are verbatim from arXiv:2604.01476) and was validated on Phi-4-mini-instruct and
Llama-3.2-3B-Instruct, **not** on Qwen. Qwen3.5-4B additionally uses hybrid Gated DeltaNet
attention rather than standard attention. Applying the set here is therefore an untested transfer
step, and a null result is a lead about *transfer failure* before it is a claim about the model.
This is diagnosis, not a pass/fail gate: if a control is not cleared, the question is *why*.

**Two validations, two spaces.**

* A **trained linear probe** (reused wholesale from ``linear_probe``: L2-regularised logistic
  regression, grouped 5-fold so a pair's positive and foil never split across train and test)
  reports held-out accuracy against a **shuffled-label null**. This asks whether the concept is
  linearly *decodable* at all, and runs in standardized (per-dimension z-scored) space, the
  ``linear_probe`` convention, so a few massive-activation dimensions cannot dominate the fit.

* The **diff-of-means direction** -- the specific axis downstream steering will use -- is scored by
  a held-out 1-D nearest-centroid classifier (extract the direction on the train fold, threshold
  and orient on the train fold, evaluate on the held-out fold) against a **matched-norm random
  placebo direction**. This asks whether *this axis*, not just some hyperplane, separates the
  classes better than a random axis of equal norm. It runs in **raw** residual space, because that
  is the space steering perturbs; the placebo is matched-norm in the same space, so the comparison
  is fair. A direction that does not beat the placebo is the transfer-failure signature.

The direction split-half cosine (raw, odd vs even pairs) is reported alongside as the noise ceiling
for how well-determined the saved axis is; read the projection accuracies against it. Both raw and
standardized concerns are documented in ``directions.standardized_directions``.

Nothing here loads a model at import time. The pure-tensor core (the projection classifier, the
placebo comparison, the split-half) runs on synthetic activations in the offline tests; the model
capture path is behind functions the CLI calls and reuses ``directions.capture_pooled_activations``
verbatim.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from reward_hacking.interp import stimuli
from reward_hacking.interp.directions import (
    POOLERS,
    capture_pooled_activations,
    cosine,
    diff_of_means,
    load_model_and_tokenizer,
    matched_norm_random_direction,
)
from reward_hacking.interp.linear_probe import (
    CaptureSpec,
    ConceptActivations,
    ProbeConfig,
    cross_validated_accuracy,
    grouped_test_masks,
    permuted_label_accuracies,
    standardizing_stats,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

CONCEPT = "eval_awareness"
# Exp2's placebo-max was a max-of-5, a weak null. Default to 100 so the max-of-placebos bar means
# something and the empirical p-value against the placebo distribution has resolution; it is only
# extra matched-norm draws over the (cheap, held-out) projection classifier on captured activations.
DEFAULT_N_PLACEBOS = 100
DEFAULT_OUT_DIR = Path("artifacts/interp/eval-awareness")

DirectionTransform = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------------------
# Pure-tensor core: 1-D projection separability and the matched-norm placebo
# --------------------------------------------------------------------------------------


def _identity(direction: torch.Tensor) -> torch.Tensor:
    """Return the direction unchanged -- the real-direction arm of the projection comparison."""
    return direction


def nearest_centroid_correct(
    direction: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> torch.Tensor:
    """Per-test correctness (0/1) of a 1-D nearest-centroid classifier along ``direction``.

    Projects onto ``direction``, then thresholds at the midpoint of the two class means and orients
    so the positive class sits on the positive side -- both fit on the training fold only, so the
    held-out number is honest. Orientation is fit rather than assumed because a placebo direction's
    sign carries no meaning; letting it orient on train is what makes a random axis land at chance
    on the held-out fold instead of a spurious 0-or-1.
    """
    train_projection = train_features @ direction
    positive_mean = train_projection[train_labels == 1].mean()
    negative_mean = train_projection[train_labels == 0].mean()
    threshold = (positive_mean + negative_mean) / 2.0
    sign = torch.where(positive_mean >= negative_mean, 1.0, -1.0)
    test_projection = test_features @ direction
    predictions = (sign * (test_projection - threshold) > 0).to(test_labels.dtype)
    return (predictions == test_labels).to(test_labels.dtype)


def projection_cv_accuracy(
    concept: ConceptActivations, config: ProbeConfig, transform: DirectionTransform
) -> float:
    """Grouped-CV held-out accuracy of the diff-of-means axis (raw space), per ``transform``.

    ``transform`` maps the train-fold diff-of-means direction to the direction actually scored:
    identity for the real axis, a matched-norm random draw for the placebo. Extracting the
    direction inside the fold (not once on all data) is what keeps the accuracy held-out rather than
    optimistic -- evaluating a direction on the very pairs that defined it is the bug this guards.
    """
    features, labels, groups = concept.polarity_task()
    correct = torch.zeros_like(labels)
    for test_mask in grouped_test_masks(groups, config.n_folds):
        train_mask = ~test_mask
        train_features, train_labels = features[train_mask], labels[train_mask]
        train_direction = diff_of_means(
            train_features[train_labels == 1], train_features[train_labels == 0]
        )
        direction = transform(train_direction)
        correct[test_mask] = nearest_centroid_correct(
            direction, train_features, train_labels, features[test_mask], labels[test_mask]
        )
    return correct.mean().item()


@dataclass(frozen=True)
class DirectionSeparation:
    """Held-out separability of the eval-awareness axis against its matched-norm placebo.

    ``direction_accuracy`` is the diff-of-means axis; the ``placebo_*`` fields summarise
    ``n_placebos`` random directions of equal norm. The axis is real only if it clears the placebo.
    ``accuracy_empirical_p`` is the one-sided empirical p-value against the placebo accuracy
    distribution: the add-one-smoothed fraction of placebos that separate the classes at least as
    well as the real axis (``(ge + 1) / (n + 1)``).
    """

    direction_accuracy: float
    placebo_accuracy_mean: float
    placebo_accuracy_max: float
    placebo_accuracy_ge_count: int
    accuracy_empirical_p: float


def direction_separation(
    concept: ConceptActivations, config: ProbeConfig, *, n_placebos: int = DEFAULT_N_PLACEBOS
) -> DirectionSeparation:
    """Score the diff-of-means axis and ``n_placebos`` matched-norm random placebos, held out.

    The placebos share one seeded generator (``config.seed``), so each fold of each draw gets a
    fresh random direction and the whole comparison is reproducible. Reporting placebo mean and max
    mirrors the shuffled-label null's mean/max: the real axis must clear the *max*, not just the
    mean, to survive an unlucky-favourable random draw. The empirical p-value against the placebo
    distribution complements the max with a graded read once ``n_placebos`` is wide.
    """
    direction_accuracy = projection_cv_accuracy(concept, config, _identity)
    generator = torch.Generator().manual_seed(config.seed)

    def draw_placebo(direction: torch.Tensor) -> torch.Tensor:
        return matched_norm_random_direction(direction, generator)

    placebo_accuracies = [
        projection_cv_accuracy(concept, config, draw_placebo) for _ in range(n_placebos)
    ]
    placebo_accuracy_ge_count = sum(1 for acc in placebo_accuracies if acc >= direction_accuracy)
    return DirectionSeparation(
        direction_accuracy=direction_accuracy,
        placebo_accuracy_mean=sum(placebo_accuracies) / len(placebo_accuracies),
        placebo_accuracy_max=max(placebo_accuracies),
        placebo_accuracy_ge_count=placebo_accuracy_ge_count,
        accuracy_empirical_p=(placebo_accuracy_ge_count + 1) / (n_placebos + 1),
    )


def direction_split_half_cosine(concept: ConceptActivations) -> float:
    """Self-consistency of the raw diff-of-means axis: cosine of the odd-pair and even-pair fits.

    The noise ceiling for the saved direction. Odd/even (not random) so it is reproducible and the
    three paper-verbatim leading pairs do not all fall on one side. Raw space because that is the
    axis downstream steering uses; a few massive-activation dimensions can inflate this (they are
    stable across pairs regardless of concept content), which is exactly the leverage
    ``directions.standardized_directions`` strips -- so read a high value with that caveat.
    """
    indices = torch.arange(concept.n_pairs)
    halves = [concept.restricted_to(indices[indices % 2 == parity]) for parity in (0, 1)]
    return cosine(halves[0].diff_of_means(), halves[1].diff_of_means())


# --------------------------------------------------------------------------------------
# Per-layer read
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptAxisRead:
    """Everything the validation measures at one layer.

    ``probe_*`` come from the trained logistic probe in standardized space (is the concept
    decodable, against its shuffled-label null). ``direction_accuracy`` / ``placebo_accuracy_*`` are
    the diff-of-means axis against its matched-norm placebo in raw space (is *this axis* real).
    ``direction_split_half_cosine`` is the raw axis's noise ceiling and ``direction_norm`` its L2
    norm.
    """

    layer: int
    probe_accuracy: float
    probe_null_accuracy_mean: float
    probe_null_accuracy_max: float
    direction_accuracy: float
    placebo_accuracy_mean: float
    placebo_accuracy_max: float
    placebo_accuracy_ge_count: int
    accuracy_empirical_p: float
    direction_split_half_cosine: float
    direction_norm: float

    @property
    def clears_null(self) -> bool:
        """Trained-probe accuracy beats the worst shuffled-label draw -- concept is decodable."""
        return self.probe_accuracy > self.probe_null_accuracy_max

    @property
    def beats_placebo(self) -> bool:
        """Diff-of-means axis beats the best matched-norm placebo draw -- the axis is real."""
        return self.direction_accuracy > self.placebo_accuracy_max


def validate_layer(
    layer: int,
    concept: ConceptActivations,
    config: ProbeConfig,
    *,
    n_placebos: int = DEFAULT_N_PLACEBOS,
) -> tuple[ConceptAxisRead, torch.Tensor]:
    """Validate one layer and return its read plus the full-data raw diff-of-means direction.

    The returned direction is fit on *all* pairs (what downstream steering uses); the accuracies in
    the read are held-out. The trained probe runs in standardized space, the direction/placebo
    comparison in raw space -- see the module docstring for why the two differ.
    """
    direction = concept.diff_of_means()

    mean, scale = standardizing_stats({CONCEPT: concept})
    standardized = concept.standardized(mean, scale)
    features, labels, groups = standardized.polarity_task()
    probe_accuracy = cross_validated_accuracy(features, labels, groups, config)
    null_accuracies = permuted_label_accuracies(features, labels, groups, config)

    separation = direction_separation(concept, config, n_placebos=n_placebos)

    read = ConceptAxisRead(
        layer=layer,
        probe_accuracy=probe_accuracy,
        probe_null_accuracy_mean=sum(null_accuracies) / len(null_accuracies),
        probe_null_accuracy_max=max(null_accuracies),
        direction_accuracy=separation.direction_accuracy,
        placebo_accuracy_mean=separation.placebo_accuracy_mean,
        placebo_accuracy_max=separation.placebo_accuracy_max,
        placebo_accuracy_ge_count=separation.placebo_accuracy_ge_count,
        accuracy_empirical_p=separation.accuracy_empirical_p,
        direction_split_half_cosine=direction_split_half_cosine(concept),
        direction_norm=direction.norm().item(),
    )
    return read, direction


# --------------------------------------------------------------------------------------
# Model-capture path (guarded: only reached from probe_concept / run_concept_probe / the CLI)
# --------------------------------------------------------------------------------------


def capture_concept_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    spec: CaptureSpec,
    *,
    pairs: Sequence[stimuli.ContrastivePair],
) -> dict[int, ConceptActivations]:
    """Capture pooled activations for a concept's ``pairs``, pivoted to layer -> activations.

    Reuses ``directions.capture_pooled_activations`` (raw sentences, no chat template) for the
    positive and foil sentences separately, then aligns them pairwise per layer. ``spec.concepts``
    is carried for provenance and logging; the concept content is entirely in ``pairs``, so the same
    function validates eval-awareness, shortcut, deception or the contradiction control. ``spec.
    limit`` caps pairs and ``spec.layer_stride`` subsamples layers for a quick smoke.
    """
    selected = list(pairs[: spec.limit] if spec.limit else pairs)
    logger.info("capturing %s %d pairs pooling=%s", spec.concepts, len(selected), spec.pooling)
    positive_activations = capture_pooled_activations(
        model,
        tokenizer,
        stimuli.positives(selected),
        pooling=spec.pooling,
        batch_size=spec.batch_size,
    )
    negative_activations = capture_pooled_activations(
        model,
        tokenizer,
        stimuli.negatives(selected),
        pooling=spec.pooling,
        batch_size=spec.batch_size,
    )
    if positive_activations.keys() != negative_activations.keys():
        raise ValueError(
            "positive and negative captures cover different layers: "
            f"{sorted(positive_activations)} vs {sorted(negative_activations)}"
        )
    layers = sorted(positive_activations)[:: spec.layer_stride]
    return {
        layer: ConceptActivations(positive_activations[layer], negative_activations[layer])
        for layer in layers
    }


@dataclass(frozen=True)
class ConceptAxisResult:
    """A whole run: the per-layer reads plus the raw material (directions and activations).

    ``concept`` names which axis this validated (eval-awareness, shortcut, deception, or the
    contradiction control), so a reloaded artifact says what it is. ``directions`` are the full-data
    raw diff-of-means axes downstream steering uses; ``activations`` are the pooled residual streams
    the reads were computed from. The rest is provenance so a reloaded artifact is self-describing.
    """

    model_id: str
    concept: str
    spec: CaptureSpec
    config: ProbeConfig
    n_placebos: int
    reads: list[ConceptAxisRead]
    directions: dict[int, torch.Tensor]
    activations: dict[int, ConceptActivations]


def probe_concept(  # noqa: PLR0913 - a loaded model, its provenance, the pairs and the knobs
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    spec: CaptureSpec,
    config: ProbeConfig,
    *,
    model_id: str,
    concept: str,
    pairs: Sequence[stimuli.ContrastivePair],
    n_placebos: int = DEFAULT_N_PLACEBOS,
) -> ConceptAxisResult:
    """Capture ``concept``'s activations from an ALREADY-LOADED model and validate every layer.

    Split from :func:`run_concept_probe` so the run harness's ``axis-probe`` stage validates all
    four concept axes under ONE model load rather than reloading the 4B weights per concept. The
    battery is exactly the one eval-awareness gets -- :func:`validate_layer` per layer -- so every
    axis carries the same probe accuracy / placebo / split-half / per-layer-significance block.
    """
    logger.info("probing %s axis: %s %s n_placebos=%d", concept, model_id, spec, n_placebos)
    by_layer = capture_concept_activations(model, tokenizer, spec, pairs=pairs)
    reads: list[ConceptAxisRead] = []
    directions: dict[int, torch.Tensor] = {}
    for layer, concept_acts in by_layer.items():
        read, direction = validate_layer(layer, concept_acts, config, n_placebos=n_placebos)
        reads.append(read)
        directions[layer] = direction
        logger.info(
            f"{concept} layer {layer}: probe_acc={read.probe_accuracy:.3f} "
            f"null_max={read.probe_null_accuracy_max:.3f} dir_acc={read.direction_accuracy:.3f} "
            f"placebo_max={read.placebo_accuracy_max:.3f} "
            f"clears_null={read.clears_null} beats_placebo={read.beats_placebo}"
        )
    return ConceptAxisResult(
        model_id=model_id,
        concept=concept,
        spec=spec,
        config=config,
        n_placebos=n_placebos,
        reads=reads,
        directions=directions,
        activations=by_layer,
    )


def run_concept_probe(  # noqa: PLR0913 - the model id, capture spec, probe config, pairs and knobs
    model_id: str,
    spec: CaptureSpec,
    config: ProbeConfig,
    *,
    concept: str,
    pairs: Sequence[stimuli.ContrastivePair],
    n_placebos: int = DEFAULT_N_PLACEBOS,
) -> ConceptAxisResult:
    """Load the model, then validate one concept axis (``probe_concept`` over a fresh model load)."""
    model, tokenizer = load_model_and_tokenizer(model_id)
    return probe_concept(
        model,
        tokenizer,
        spec,
        config,
        model_id=model_id,
        concept=concept,
        pairs=pairs,
        n_placebos=n_placebos,
    )


def run_eval_awareness_probe(
    model_id: str, spec: CaptureSpec, config: ProbeConfig, *, n_placebos: int = DEFAULT_N_PLACEBOS
) -> ConceptAxisResult:
    """Load the model and validate the eval-awareness axis (the concept-specialised entry point)."""
    return run_concept_probe(
        model_id,
        spec,
        config,
        concept=CONCEPT,
        pairs=stimuli.EVAL_AWARENESS_PAIRS,
        n_placebos=n_placebos,
    )


# --------------------------------------------------------------------------------------
# Artifacts and reporting
# --------------------------------------------------------------------------------------


def save_artifacts(result: ConceptAxisResult, out_dir: Path) -> None:
    """Persist directions, activations, and metrics so adding rigor later is a re-analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.directions, out_dir / "directions.pt")
    torch.save(
        {
            layer: {"positives": concept.positives, "negatives": concept.negatives}
            for layer, concept in result.activations.items()
        },
        out_dir / "activations.pt",
    )
    n_pairs = next(iter(result.activations.values())).n_pairs if result.activations else 0
    payload = {
        "model_id": result.model_id,
        "concept": result.concept,
        "pooling": result.spec.pooling,
        "concepts_provenance": list(result.spec.concepts),
        "n_pairs": n_pairs,
        "n_placebos": result.n_placebos,
        "config": asdict(result.config),
        "layer_reads": [asdict(read) for read in result.reads],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    logger.info(f"wrote directions, activations and metrics to {out_dir}")


def format_table(reads: list[ConceptAxisRead]) -> str:
    """Per-layer results table: trained-probe decodability and the axis-vs-placebo separability."""
    header = (
        f"{'layer':>5}  {'probe_acc':>9}  {'null':>6}  {'null_max':>8}  {'dir_acc':>7}  "
        f"{'placebo':>7}  {'plc_max':>7}  {'split½':>7}  {'|dir|':>8}  {'clears':>6}  {'beats':>5}"
    )
    rows = [
        f"{read.layer:>5}  {read.probe_accuracy:>9.3f}  {read.probe_null_accuracy_mean:>6.3f}  "
        f"{read.probe_null_accuracy_max:>8.3f}  {read.direction_accuracy:>7.3f}  "
        f"{read.placebo_accuracy_mean:>7.3f}  {read.placebo_accuracy_max:>7.3f}  "
        f"{read.direction_split_half_cosine:>7.3f}  {read.direction_norm:>8.2f}  "
        f"{read.clears_null!s:>6}  {read.beats_placebo!s:>5}"
        for read in reads
    ]
    return "\n".join([header, *rows])


def summarize(reads: list[ConceptAxisRead]) -> str:
    """Report where separation peaks and whether it clears both controls, as a diagnosis.

    Diagnosis, not a verdict -- if the controls are not cleared, this states that plainly so the
    next question is why (transfer failure being the leading hypothesis), never a go/no-go.
    """
    if not reads:
        return "no layers captured"
    best_probe = max(reads, key=lambda read: read.probe_accuracy)
    best_direction = max(
        reads, key=lambda read: read.direction_accuracy - read.placebo_accuracy_mean
    )
    clearing = [read.layer for read in reads if read.clears_null and read.beats_placebo]
    return (
        f"best trained-probe layer {best_probe.layer}: acc={best_probe.probe_accuracy:.3f} "
        f"vs null_max={best_probe.probe_null_accuracy_max:.3f}. "
        f"best diff-of-means axis layer {best_direction.layer}: "
        f"dir_acc={best_direction.direction_accuracy:.3f} vs "
        f"placebo_max={best_direction.placebo_accuracy_max:.3f} "
        f"(p={best_direction.accuracy_empirical_p:.3f}, "
        f"split½={best_direction.direction_split_half_cosine:.3f}). "
        f"layers clearing BOTH controls: {clearing or 'none'}. "
        "eval-awareness axis transfers to this model at those layers; empty means the transfer "
        "failed and that is the finding to chase, not a stop."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments for the eval-awareness direction validation."""
    parser = argparse.ArgumentParser(
        description="Extract and validate Qwen3.5-4B's eval-awareness direction"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--pooling", choices=sorted(POOLERS), default="mean")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap eval-awareness pairs (default: all 82)"
    )
    parser.add_argument(
        "--layer-stride", type=int, default=1, help="validate every Nth layer (default: all)"
    )
    parser.add_argument("--l2-strength", type=float, default=ProbeConfig.l2_strength)
    parser.add_argument("--n-folds", type=int, default=ProbeConfig.n_folds)
    parser.add_argument("--n-permutations", type=int, default=ProbeConfig.n_permutations)
    parser.add_argument("--max-iter", type=int, default=ProbeConfig.max_iter)
    parser.add_argument("--seed", type=int, default=ProbeConfig.seed)
    parser.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: artifacts/interp/eval-awareness for mean pooling, "
        "artifacts/interp/eval-awareness-<pooling> otherwise (so mean and last never clobber)",
    )
    return parser.parse_args(argv)


def default_out_dir(pooling: str) -> Path:
    """Per-pooling output dir: the canonical path for mean, a suffixed sibling for anything else.

    Keeps the validated mean-pool axis at its established location while giving the last-pool
    validation (deliverable of the reasoning-phase read) its own directory, which is exactly where
    ``prompt_contrast`` looks for a validated last-pool axis (``DEFAULT_VALIDATED_EVAL_LAST_DIR``).
    """
    if pooling == "mean":
        return DEFAULT_OUT_DIR
    return DEFAULT_OUT_DIR.parent / f"eval-awareness-{pooling}"


def main(argv: list[str] | None = None) -> None:
    """Run the eval-awareness direction validation, print the table, and save the raw material."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = ProbeConfig(
        l2_strength=args.l2_strength,
        n_folds=args.n_folds,
        n_permutations=args.n_permutations,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    spec = CaptureSpec(
        concepts=(CONCEPT,),
        pooling=args.pooling,
        batch_size=args.batch_size,
        limit=args.limit,
        layer_stride=args.layer_stride,
    )
    result = run_eval_awareness_probe(args.model_id, spec, config, n_placebos=args.n_placebos)
    print(format_table(result.reads))  # noqa: T201  # Intentional CLI table output.
    logger.info(summarize(result.reads))
    save_artifacts(result, args.out_dir or default_out_dir(args.pooling))


if __name__ == "__main__":
    main()
