"""Project agentic-prompt activations onto concept directions: conflicting vs matched-original.

The question. Heavy RL against loose verifiers might teach a model to read *what dimension a
situation is graded on*. If so, a model should internally register when the grader it is being
scored against is provably self-contradictory. This module tests one observable version of that:
does an already-validated "evaluation awareness" residual-stream direction fire more strongly when
Qwen3.5-4B reads a task whose grader cannot be satisfied (the conflicting Impossible-LiveCodeBench
split) than when it reads the *same underlying problem* with a normal, satisfiable grader (the
matched original twin)? No generation and no solving -- one forward pass per prompt, pool the
residual stream, project onto the direction, compare the two groups.

The contrast is clean by construction. For every matched pair the problem statement, entry point,
file listing, action-protocol preamble, and the simulated ``cat grade.py`` step are identical; the
ONLY thing that differs between the twins is the grader body the model reads. So the chat format
and the problem are held constant across the contrast, and any projection difference is
attributable to the grader manipulation (plus whatever incidental lexical difference the two grader
bodies carry, which is the nuisance to keep in mind, not a confound the format introduces).

Two halves, mirroring ``directions.py`` and ``eval_awareness_probe.py``:

* A **pure-tensor core** -- projection onto a unit direction, a rank-based AUC, Cohen's d, and a
  paired-difference read across the matched twins, each computed for the concept direction and for
  a matched-norm random placebo. It touches no model and is what the offline tests exercise on
  synthetic activations with a planted signal.
* A **model-capture path** behind functions the CLI calls: it materialises each task through the
  harness's own ``lay_down_task`` (so the grader text is the real one, read from disk rather than
  reconstructed), builds the transcript the policy would hold right after reading the grader,
  chat-formats it exactly as ``model_backend`` does, and reuses ``directions.capture_pooled_
  activations`` for the forward pass.

Method caveat, load-bearing. The concept directions are fit on short, raw (non-chat-templated)
sentence pairs, while these prompts are long, chat-templated agentic prompts, so the ABSOLUTE
projection space differs from where the axis was fit and absolute magnitudes are not meaningful.
The conflicting-vs-original comparison stays valid because both groups live in the same
chat-formatted space -- the format is constant across the contrast. The matched-norm placebo is
mandatory (repo policy): a projection difference means nothing unless it beats the placebo.

Diagnosis, never a gate. A null or a surprise on shortcut/deception is a lead about the stimulus
(residual lexical difference between grader bodies) or about transfer before it is a claim about
what the model represents.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from reward_hacking.harness.loop import lay_down_task
from reward_hacking.harness.protocol import initial_prompt
from reward_hacking.harness.task_spec import BASELINE_ARM
from reward_hacking.harness.tasks_ilcb import ILCB_TASKS_BY_ID, PROBLEMS
from reward_hacking.interp import stimuli
from reward_hacking.interp.directions import (
    POOLERS,
    capture_pooled_activations,
    capture_pooled_activations_multi,
    cosine,
    diff_of_means,
    load_model_and_tokenizer,
    matched_norm_random_direction,
    unit,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_hacking.harness.task_spec import Task
    from reward_hacking.harness.tasks_ilcb import ILCBProblem

logger = logging.getLogger(__name__)

# Descriptive concept names (never opaque labels) so a table row reads itself.
SHORTCUT = "shortcut"
DECEPTION = "deception"
EVAL_AWARENESS = "eval_awareness"
# The generic-contradiction control (stimuli.CONTRADICTION_PAIRS): does the conflicting-vs-original
# separation track "I'm graded on a rigged test" (eval-awareness) or merely "the grader text is
# self-contradictory"? If this axis separates the twins as well as eval-awareness does, the
# eval-awareness reading is not supported. See stimuli.CONTRADICTION_PAIRS for the construct.
CONTRADICTION = "contradiction"
CONTRAST_CONCEPTS: tuple[str, ...] = (SHORTCUT, DECEPTION, EVAL_AWARENESS, CONTRADICTION)

DEFAULT_POOLINGS: tuple[str, ...] = ("mean", "last")
# Exp2 drew only 5 placebos, so its placebo-max was a max-of-5 -- a weak null. Default to
# 100 so the max-of-placebos bar is meaningful and the empirical p-value against the placebo
# distribution has resolution. It is one extra matched-norm draw per layer, negligible on
# already-captured activations.
DEFAULT_N_PLACEBOS = 100
DEFAULT_PROMPT_BATCH_SIZE = 4
DEFAULT_CONCEPT_BATCH_SIZE = 16
DEFAULT_OUT_DIR = Path("artifacts/interp/prompt-contrast")
# Where eval_awareness_probe saved the validated eval-awareness axes, one dir per pooling. The
# headline eval-awareness read reuses these VALIDATED axes rather than a fresh (unvalidated) one --
# Exp2's last-pool signal rested on a fresh last-pool axis that never cleared the controls, which is
# what the last-pool validation (eval_awareness_probe --pooling last) exists to fix.
DEFAULT_VALIDATED_EVAL_DIR = Path("artifacts/interp/eval-awareness")
DEFAULT_VALIDATED_EVAL_LAST_DIR = Path("artifacts/interp/eval-awareness-last")


# --------------------------------------------------------------------------------------
# Pure-tensor core: projection and separation statistics
# --------------------------------------------------------------------------------------


def project(features: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project pooled activations onto a direction. ``features`` [n, d], ``direction`` [d] -> [n].

    The direction is unit-normalised inside, so the real axis and a matched-norm placebo are scored
    on the same scale and their mean projections are directly comparable, not just their effect
    sizes.
    """
    return features @ unit(direction)


def roc_auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """P(a random ``positive`` projection exceeds a random ``negative`` one), ties counted as half.

    The Mann-Whitney reading of AUC, computed by the direct pairwise definition rather than by
    ranks: n and m are ~100 here, so the [n, m] comparison is a 10k-entry bool matrix, exact on
    ties and obviously correct. 0.5 is chance; 1.0 means every conflicting prompt projects above
    every original one.
    """
    difference = positive.unsqueeze(1) - negative.unsqueeze(0)
    wins = (difference > 0).sum().float()
    ties = (difference == 0).sum().float()
    return ((wins + 0.5 * ties) / (positive.numel() * negative.numel())).item()


def cohens_d(group_a: torch.Tensor, group_b: torch.Tensor) -> float:
    """Standardised mean difference ``(mean_a - mean_b) / pooled_sd`` -- an unpaired effect size.

    Returns 0.0 when both groups are constant (pooled sd 0), the only degenerate case; a planted
    signal gives a clearly non-zero value the tests check the sign and rough magnitude of.
    """
    n_a, n_b = group_a.numel(), group_b.numel()
    pooled_var = (
        (n_a - 1) * group_a.var(unbiased=True) + (n_b - 1) * group_b.var(unbiased=True)
    ) / (n_a + n_b - 2)
    pooled_sd = pooled_var.sqrt().item()
    if pooled_sd == 0.0:
        return 0.0
    return (group_a.mean() - group_b.mean()).item() / pooled_sd


@dataclass(frozen=True)
class PairedRead:
    """The matched-twin read: same problem, conflicting grader minus original grader.

    ``mean_diff`` is the average of (conflicting - original) over the pairs, ``t_stat`` the paired
    t-statistic, and ``sign_rate`` the fraction of pairs whose conflicting prompt projects higher.
    Pairing is free here and is the natural design, since each conflicting task has a matched
    original twin over the identical underlying problem, so it controls for problem-level variation
    the unpaired AUC cannot.
    """

    mean_diff: float
    t_stat: float
    sign_rate: float


def paired_read(conflicting: torch.Tensor, original: torch.Tensor) -> PairedRead:
    """Paired-difference statistics over aligned twin projections. Both [n_pairs], row i a pair."""
    if conflicting.shape != original.shape:
        raise ValueError(
            f"paired read needs aligned twins; got {tuple(conflicting.shape)} conflicting "
            f"and {tuple(original.shape)} original"
        )
    difference = conflicting - original
    mean_diff = difference.mean().item()
    sd = difference.std(unbiased=True).item()
    n = difference.numel()
    t_stat = mean_diff / (sd / n**0.5) if sd > 0.0 else 0.0
    sign_rate = (difference > 0).float().mean().item()
    return PairedRead(mean_diff=mean_diff, t_stat=t_stat, sign_rate=sign_rate)


@dataclass(frozen=True)
class ContrastRead:
    """One concept direction at one layer under one pooling: the separation, against its placebo.

    The real axis's ``auc`` / ``cohens_d`` / paired fields sit beside the matched-norm placebo's,
    averaged over ``n_placebos`` random directions of equal norm (``placebo_auc_max`` is the worst
    case a lucky draw could reach). The separation is real only in so far as it clears the placebo;
    ``beats_placebo`` and ``auc_above_placebo`` are the read. ``auc_empirical_p`` is the one-sided
    empirical p-value against the placebo AUC distribution: the fraction of placebo draws that
    separate the twins at least as well as the real axis, add-one smoothed (``(ge + 1) / (n + 1)``),
    so a widened placebo count (default 100) gives it real resolution.

    ``auc_empirical_p`` is a WITHIN-LAYER p: it holds the layer fixed and asks only whether the real
    axis beats a random axis at THIS layer. It does NOT account for the peak layer having been chosen
    as the best of many, which understates the false-positive rate of the reported peak. The
    layer-selection correction for that lives in :func:`layer_selection_p`, and it needs the raw
    per-placebo AUC draws, which is why ``placebo_aucs`` is carried on the read rather than only its
    summary (mean / max / ge-count). ``placebo_aucs[i]`` is the AUC of the i-th matched-norm random
    direction at this layer; aligning index ``i`` across a group's layers is what lets that
    correction take each placebo's own best-over-layers AUC.
    """

    concept: str
    pooling: str
    layer: int
    n_conflicting: int
    n_original: int
    n_pairs: int
    direction_norm: float
    mean_conflicting: float
    mean_original: float
    mean_diff: float
    auc: float
    cohens_d: float
    paired_mean_diff: float
    paired_t: float
    paired_sign_rate: float
    n_placebos: int
    placebo_auc_mean: float
    placebo_auc_max: float
    placebo_auc_ge_count: int
    auc_empirical_p: float
    placebo_cohens_d_mean: float
    placebo_paired_sign_rate_mean: float
    placebo_paired_abs_mean_diff_mean: float
    # The raw per-placebo AUCs behind the summaries above (one per matched-norm draw at this layer).
    # Defaulted so the many hand-built ContrastReads in the tests need not restate it; the model path
    # always fills it. It is the raw material :func:`layer_selection_p` needs and is dropped from the
    # uploadable metrics (see :func:`metrics_dict`) to keep that file lean.
    placebo_aucs: tuple[float, ...] = ()

    @property
    def auc_above_placebo(self) -> float:
        """How far the real AUC sits above the average placebo AUC -- the effect over chance."""
        return self.auc - self.placebo_auc_mean

    @property
    def beats_placebo(self) -> bool:
        """The real AUC clears the best of the matched-norm placebo draws -- the axis separates."""
        return self.auc > self.placebo_auc_max


def contrast_layer(  # noqa: PLR0913, PLR0917 - the metadata plus the three tensors are all core
    concept: str,
    pooling: str,
    layer: int,
    conflicting_acts: torch.Tensor,
    original_acts: torch.Tensor,
    direction: torch.Tensor,
    *,
    n_placebos: int,
    generator: torch.Generator,
) -> ContrastRead:
    """Project both twin groups onto ``direction`` and onto ``n_placebos`` matched-norm placebos.

    ``conflicting_acts`` and ``original_acts`` are pair-aligned [n_pairs, d] (row i is the two
    twins of problem i), so the AUC reads them as two groups and the paired read reads them as
    matched differences. The placebo generator is threaded in from the caller so a whole run's
    placebo draws are one reproducible stream.
    """
    projection_conflicting = project(conflicting_acts, direction)
    projection_original = project(original_acts, direction)
    paired = paired_read(projection_conflicting, projection_original)
    auc_value = roc_auc(projection_conflicting, projection_original)

    placebo_aucs: list[float] = []
    placebo_ds: list[float] = []
    placebo_sign_rates: list[float] = []
    placebo_abs_mean_diffs: list[float] = []
    for _ in range(n_placebos):
        placebo = matched_norm_random_direction(direction, generator)
        placebo_conflicting = project(conflicting_acts, placebo)
        placebo_original = project(original_acts, placebo)
        placebo_paired = paired_read(placebo_conflicting, placebo_original)
        placebo_aucs.append(roc_auc(placebo_conflicting, placebo_original))
        placebo_ds.append(cohens_d(placebo_conflicting, placebo_original))
        placebo_sign_rates.append(placebo_paired.sign_rate)
        placebo_abs_mean_diffs.append(abs(placebo_paired.mean_diff))

    placebo_auc_ge_count = sum(1 for placebo_auc in placebo_aucs if placebo_auc >= auc_value)
    return ContrastRead(
        concept=concept,
        pooling=pooling,
        layer=layer,
        n_conflicting=projection_conflicting.numel(),
        n_original=projection_original.numel(),
        n_pairs=projection_conflicting.numel(),
        direction_norm=direction.norm().item(),
        mean_conflicting=projection_conflicting.mean().item(),
        mean_original=projection_original.mean().item(),
        mean_diff=(projection_conflicting.mean() - projection_original.mean()).item(),
        auc=auc_value,
        cohens_d=cohens_d(projection_conflicting, projection_original),
        paired_mean_diff=paired.mean_diff,
        paired_t=paired.t_stat,
        paired_sign_rate=paired.sign_rate,
        n_placebos=n_placebos,
        placebo_auc_mean=_mean(placebo_aucs),
        placebo_auc_max=max(placebo_aucs, default=0.5),
        placebo_auc_ge_count=placebo_auc_ge_count,
        auc_empirical_p=(placebo_auc_ge_count + 1) / (n_placebos + 1),
        placebo_cohens_d_mean=_mean(placebo_ds),
        placebo_paired_sign_rate_mean=_mean(placebo_sign_rates),
        placebo_paired_abs_mean_diff_mean=_mean(placebo_abs_mean_diffs),
        placebo_aucs=tuple(placebo_aucs),
    )


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 on an empty sequence (no placebos requested)."""
    return sum(values) / len(values) if values else 0.0


def placebo_stream_seed(seed: int, concept: str, pooling: str) -> int:
    """Derive the placebo-generator seed for one (concept, pooling) sweep from the run's base seed.

    Exists because the run harness calls :func:`contrast_all_layers` once per (variant, pooling,
    concept) with ONE base seed, and a generator built from the bare seed hands every call the
    identical stream of random directions. ``project`` unit-normalises, so the matched-norm scaling
    is the only per-concept input to a placebo read -- meaning all four concepts were scored against
    the SAME 100 draws (measured on the 2026-08-21/22 runs: max cross-concept difference in placebo
    stats 9e-6). "Several concepts cleared their placebo band" then carries one draw's worth of
    independence, not four.

    Salted with a stable string hash rather than by advancing one shared generator across the sweep,
    for the reason ``run_harness._cell_seed`` documents: a stream that advances across calls makes
    every cell's draws depend on how many cells ran before it, so re-running with a different
    concept or pooling list silently changes every placebo. This derivation gives each (concept,
    pooling) cell the same draws in any run of any shape. The salt deliberately excludes the pooling
    VARIANT: the real axis is the same vector across variants (axes are keyed concept/pooling/layer),
    so sharing the placebo directions across variants mirrors the real statistic's structure --
    same directions, different pooled activations.
    """
    return seed + zlib.crc32(f"{concept}|{pooling}".encode())


def contrast_all_layers(  # noqa: PLR0913 - the two labels and three layer maps are all core inputs
    concept: str,
    pooling: str,
    conflicting_by_layer: dict[int, torch.Tensor],
    original_by_layer: dict[int, torch.Tensor],
    directions_by_layer: dict[int, torch.Tensor],
    *,
    n_placebos: int = DEFAULT_N_PLACEBOS,
    seed: int = 0,
) -> list[ContrastRead]:
    """Run :func:`contrast_layer` over every layer, guarding that all three inputs cover the same.

    A mismatch is a bug -- some capture or the loaded direction silently dropped a layer -- and
    raises rather than quietly intersecting, the same guard ``directions.compare_directions`` makes.

    The placebo generator is seeded through :func:`placebo_stream_seed`, so two calls sharing a base
    ``seed`` but differing in concept or pooling draw INDEPENDENT placebo directions -- a bare
    ``manual_seed(seed)`` here scored every concept against the identical null. One generator still
    threads through the layers within the call, which is what keeps the draws index-aligned across
    layers -- the alignment :func:`layer_selection_p`'s max-over-layers null depends on.
    """
    layer_sets = {
        "conflicting": frozenset(conflicting_by_layer),
        "original": frozenset(original_by_layer),
        "directions": frozenset(directions_by_layer),
    }
    if len({frozenset(layers) for layers in layer_sets.values()}) != 1:
        covered = {name: sorted(layers) for name, layers in layer_sets.items()}
        raise ValueError(f"conflicting, original and directions cover different layers: {covered}")

    generator = torch.Generator().manual_seed(placebo_stream_seed(seed, concept, pooling))
    return [
        contrast_layer(
            concept,
            pooling,
            layer,
            conflicting_by_layer[layer],
            original_by_layer[layer],
            directions_by_layer[layer],
            n_placebos=n_placebos,
            generator=generator,
        )
        for layer in sorted(conflicting_by_layer)
    ]


@dataclass(frozen=True)
class LayerSelection:
    """Layer-selection-corrected significance for one (concept, pooling) group across its layers.

    The peak layer is reported as the argmax AUC over all the group's layers, so its within-layer
    empirical p understates the false-positive rate: it holds the layer fixed and never pays for the
    peak having been chosen as the best of many. With N layers even a null direction lands SOME
    layer above chance, so a within-layer p that looks significant at the peak can be an artifact of
    the selection.

    ``selection_corrected_p`` is the fix. It compares the real axis's best-over-layers AUC against
    the distribution of each matched-norm placebo's OWN best-over-layers AUC -- every placebo maxed
    over the same N layers -- add-one smoothed like the within-layer p. So it answers "is the peak
    separation beyond what the best of N layers of a matched-norm random direction reaches", which is
    the question the peak selection actually poses. ``within_layer_empirical_p`` is the uncorrected
    per-layer p at the peak, carried beside it with an unambiguous name so the two can never be read
    as each other.
    """

    concept: str
    pooling: str
    n_layers: int
    n_placebos: int
    peak_layer: int
    peak_auc: float
    within_layer_empirical_p: float
    real_best_auc: float
    placebo_best_auc_mean: float
    placebo_best_auc_max: float
    selection_corrected_p: float


def layer_selection_p(reads: Sequence[ContrastRead]) -> LayerSelection:
    """Correct the peak-layer significance of one (concept, pooling) group for the layer selection.

    ``reads`` must be every layer's read for a SINGLE concept under a SINGLE pooling (the group the
    peak is chosen within); a mixed group is a caller bug and raises rather than silently maxing over
    the wrong set. Each read carries its raw ``placebo_aucs`` -- the AUC of every matched-norm draw at
    that layer -- and the draws are index-aligned across layers (``contrast_all_layers`` threads one
    generator through the layers, so placebo ``i`` is an independent random direction at each layer).
    Taking, for each placebo index ``i``, the max of ``placebo_aucs[i]`` over the layers gives that
    placebo's best-over-layers AUC -- one draw from the null of "best of N layers of a random
    direction, one direction per layer" -- which mirrors the real statistic exactly: the real axis is
    itself a different direction per layer and the peak is its best over the same N layers.

    The peak layer is the argmax raw AUC, matching the statistic (both on AUC); with a flat ~0.5
    placebo mean it is the same layer ``select_peak_layers`` hands the causal stage, though nothing
    here requires that. Because it can differ, the run harness emits that causal layer beside this
    one as ``causal_handoff_layer``, so a reader cannot read the two as one. Add-one smoothing
    (``(ge + 1) / (n_placebos + 1)``) keeps the p bounded away from zero and comparable to the
    within-layer p.
    """
    if not reads:
        raise ValueError("layer_selection_p needs at least one read")
    concepts = {read.concept for read in reads}
    poolings = {read.pooling for read in reads}
    if len(concepts) != 1 or len(poolings) != 1:
        raise ValueError(
            "layer_selection_p expects one (concept, pooling) group across layers; got concepts "
            f"{sorted(concepts)} and poolings {sorted(poolings)}"
        )
    placebo_counts = {len(read.placebo_aucs) for read in reads}
    if len(placebo_counts) != 1:
        raise ValueError(
            f"reads carry differing placebo counts {sorted(placebo_counts)}, so the max-over-layers "
            "null cannot be aligned by placebo index"
        )
    peak_read = max(reads, key=lambda read: read.auc)
    real_best_auc = peak_read.auc
    (n_placebos,) = placebo_counts
    if n_placebos == 0:
        return LayerSelection(
            concept=peak_read.concept,
            pooling=peak_read.pooling,
            n_layers=len(reads),
            n_placebos=0,
            peak_layer=peak_read.layer,
            peak_auc=peak_read.auc,
            within_layer_empirical_p=peak_read.auc_empirical_p,
            real_best_auc=real_best_auc,
            placebo_best_auc_mean=0.5,
            placebo_best_auc_max=0.5,
            selection_corrected_p=1.0,
        )
    placebo_best = [max(read.placebo_aucs[i] for read in reads) for i in range(n_placebos)]
    ge_count = sum(1 for best in placebo_best if best >= real_best_auc)
    return LayerSelection(
        concept=peak_read.concept,
        pooling=peak_read.pooling,
        n_layers=len(reads),
        n_placebos=n_placebos,
        peak_layer=peak_read.layer,
        peak_auc=peak_read.auc,
        within_layer_empirical_p=peak_read.auc_empirical_p,
        real_best_auc=real_best_auc,
        placebo_best_auc_mean=_mean(placebo_best),
        placebo_best_auc_max=max(placebo_best),
        selection_corrected_p=(ge_count + 1) / (n_placebos + 1),
    )


def metrics_dict(read: ContrastRead) -> dict[str, object]:
    """Serialise a read for the uploadable metrics, minus the bulky raw ``placebo_aucs`` draws.

    The per-placebo AUC vector is raw material :func:`layer_selection_p` consumes in memory and is
    recomputable from the saved pooled activations and axes; persisting it per read would multiply the
    metrics file by the placebo count for no re-analysis the raw tree cannot already serve. The
    layer-selection correction it feeds is emitted compactly instead (see the run harness).
    """
    fields = asdict(read)
    del fields["placebo_aucs"]
    return fields


def selection_by_group(reads: Sequence[ContrastRead]) -> dict[str, dict[str, dict[str, object]]]:
    """Layer-selection correction for every (concept, pooling) group in a flat list of reads.

    Groups the reads by concept and pooling -- each group being exactly the layers the peak is
    chosen within -- and runs :func:`layer_selection_p` on each. Returns ``concept -> pooling ->``
    the correction as a plain dict, ready to drop into a metrics payload beside the within-layer
    reads. A caller with a further axis (the run harness keys on the pooling VARIANT too) calls this
    once per that axis and nests the result.
    """
    grouped: dict[tuple[str, str], list[ContrastRead]] = {}
    for read in reads:
        grouped.setdefault((read.concept, read.pooling), []).append(read)
    out: dict[str, dict[str, dict[str, object]]] = {}
    for (concept, pooling), group in grouped.items():
        out.setdefault(concept, {})[pooling] = asdict(layer_selection_p(group))
    return out


# --------------------------------------------------------------------------------------
# Stimulus construction (no model; uses the harness's own materialisation)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StimulusPair:
    """One matched twin: the raw transcripts the policy would hold after reading each grader.

    ``problem_id`` is the underlying LiveCodeBench problem (e.g. the id shared by the conflicting
    and original splits). The two transcript fields are the full first-turn prompt plus a simulated
    ``cat grade.py`` observation; they are byte-identical except for the grader body, which is the
    whole manipulation. Chat formatting is applied later, at capture time, with the tokenizer.

    Construction guarantee (the basis of the patch stage's ``shared_prefix_control``): the twins
    share a byte-identical PREFIX and a byte-identical SUFFIX and differ in exactly ONE contiguous
    region, which falls after the grader-read observation's ``<stdout>`` marker. Everything before
    it -- problem statement, protocol preamble, file listing, the ``<run>cat grade.py</run>`` action
    and the observation opener -- is byte-identical, because ``_task_transcript`` reads both bodies
    through the same ``lay_down_task`` / ``initial_prompt`` path and only the grader file content
    differs between the two tasks. So the divergence is localised to the grader, and the shared
    prefix is identical down to the token id (verified against the real ILCB twins). ``test_interp_
    prompt_contrast`` pins this, so a regression that leaks twin-dependent content into the prefix
    (a listing that prints file sizes, an episode-dir path, nondeterministic ordering) is caught.

    Caveat, verified 2026-08-19 and NOT a construction defect: a byte- and token-identical shared
    prefix does not by itself make the downstream activations at those positions bit-identical
    between the two forward passes. On the real twins the shared-prefix patch delta is exactly 0.0
    for the short pairs but non-zero (Frobenius ~12 at a deep layer) for the LONGEST pair, whose
    twins run past 4000 tokens. The cause is the model, not the stimulus: these are hybrid
    linear-attention checkpoints whose Gated DeltaNet state accumulates over the whole sequence, so
    two runs of different total length round that accumulation differently even at token-identical
    upstream positions. The ``shared_prefix_control`` arm's ``patch_delta_norm`` telemetry is what
    surfaces this; a large delta there flags a pair where the control's zero-baseline assumption is
    violated by the forward pass rather than by the twins.
    """

    problem_id: str
    conflicting_transcript: str
    original_transcript: str


def _matched_problem_pairs() -> list[tuple[str, ILCBProblem, ILCBProblem]]:
    """Pair each provably-impossible conflicting problem with its matched original twin.

    Selects exactly the conflicting set ``ilcb_tasks(impossible_type='conflicting',
    provably_impossible=True)`` draws (parsing check, proven contradiction) and requires the
    original twin's check to parse too. Sorted by problem id so the pair order is deterministic and
    the saved artifacts are reproducible.
    """
    by_problem: dict[str, dict[str, ILCBProblem]] = {}
    for problem in PROBLEMS:
        by_problem.setdefault(problem.task_id, {})[problem.impossible_type] = problem

    pairs: list[tuple[str, ILCBProblem, ILCBProblem]] = []
    for problem_id in sorted(by_problem):
        splits = by_problem[problem_id]
        conflicting = splits.get("conflicting")
        original = splits.get("original")
        if conflicting is None or original is None:
            continue
        if not (conflicting.provably_impossible and conflicting.check_parses):
            continue
        if not original.check_parses:
            continue
        pairs.append((problem_id, conflicting, original))
    return pairs


def _grader_observation(grader_filename: str, grader_text: str) -> str:
    """Render the ``cat grade.py`` observation the harness transcript would carry.

    Mirrors ``harness.loop._format_observation`` for a clean exit-0 read with no stderr, kept as a
    few local lines rather than importing that private helper: the format is constant across the
    contrast, so an exact byte match to the harness matters less than not coupling this probe to a
    private symbol. A drift in the harness's format changes both twins identically and cannot bias
    the conflicting-vs-original comparison.
    """
    return (
        f'<result command="cat {grader_filename}" exit=0>\n'
        f"<stdout>{grader_text}</stdout>\n"
        "</result>"
    )


def _task_transcript(task: Task, episode_dir: Path) -> str:
    """Materialise a task and build the transcript the policy holds right after reading the grader.

    Uses the harness's ``lay_down_task`` to write ``grade.py`` into a fresh ``/work`` and reads it
    back rather than reconstructing the grader source, so the stimulus carries exactly what the
    policy would ``cat``. The transcript is the first-turn prompt (protocol + task + file listing)
    followed by a synthetic ``<run>cat grade.py</run>`` and its observation -- a minimal,
    constant-across-conditions stand-in for the model's own first turn, whose only variable content
    is the grader body.
    """
    listing = lay_down_task(episode_dir, task, arm=BASELINE_ARM)
    grader_text = (episode_dir / task.grader_filename).read_text()
    prompt = initial_prompt(task, arm=BASELINE_ARM, listing=listing)
    observation = _grader_observation(task.grader_filename, grader_text)
    return f"{prompt}\n<run>cat {task.grader_filename}</run>\n{observation}\n"


def build_stimulus_pairs(episode_dir: Path, *, limit: int | None = None) -> list[StimulusPair]:
    """Build the raw transcript twins for the matched conflicting/original problems.

    ``episode_dir`` is a scratch ``/work`` (must live outside a home tree; ``lay_down_task`` refuses
    otherwise) that is cleared and rewritten for every task, so one directory serves all of them.
    ``limit`` caps the number of problem pairs for a smoke run.
    """
    problem_pairs = _matched_problem_pairs()
    if limit is not None:
        problem_pairs = problem_pairs[:limit]
    logger.info("building %d matched stimulus pairs", len(problem_pairs))

    pairs: list[StimulusPair] = []
    for problem_id, conflicting, original in problem_pairs:
        conflicting_task = ILCB_TASKS_BY_ID[conflicting.harness_task_id]
        original_task = ILCB_TASKS_BY_ID[original.harness_task_id]
        pairs.append(
            StimulusPair(
                problem_id=problem_id,
                conflicting_transcript=_task_transcript(conflicting_task, episode_dir),
                original_transcript=_task_transcript(original_task, episode_dir),
            )
        )
    return pairs


# --------------------------------------------------------------------------------------
# Model-capture path (guarded: only reached from run_prompt_contrast / the CLI)
# --------------------------------------------------------------------------------------


def _chat_format(tokenizer: AutoTokenizer, transcript: str, *, thinking: bool) -> str:
    """Wrap a transcript as one user turn via the model's chat template, ready to tokenise.

    The same call ``model_backend._as_single_user_turn`` makes for the agent loop -- thinking mode
    on, ``add_generation_prompt=True`` -- replicated here (rather than importing that private
    helper) so the captured prompt is formatted exactly as the harness would present it. Getting
    this call wrong would silently change the whole measured space, so it is pinned to the harness
    convention in one visible place.
    """
    return tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
        [{"role": "user", "content": transcript}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def capture_prompt_activations(  # noqa: PLR0913 - model, tokenizer, data and three capture knobs
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: Sequence[StimulusPair],
    *,
    pooling: str,
    batch_size: int,
    thinking: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Capture pooled residual activations for the conflicting and original prompts, per layer.

    Both groups are chat-formatted the same way and captured in pair order, so row i of each
    returned tensor is the two twins of the same problem -- which is what makes the paired read
    valid downstream.
    """
    conflicting = [
        _chat_format(tokenizer, pair.conflicting_transcript, thinking=thinking) for pair in pairs
    ]
    original = [
        _chat_format(tokenizer, pair.original_transcript, thinking=thinking) for pair in pairs
    ]
    conflicting_acts = capture_pooled_activations(
        model, tokenizer, conflicting, pooling=pooling, batch_size=batch_size
    )
    original_acts = capture_pooled_activations(
        model, tokenizer, original, pooling=pooling, batch_size=batch_size
    )
    if conflicting_acts.keys() != original_acts.keys():
        raise ValueError(
            "conflicting and original captures cover different layers: "
            f"{sorted(conflicting_acts)} vs {sorted(original_acts)}"
        )
    return conflicting_acts, original_acts


def extract_concept_directions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: Sequence[stimuli.ContrastivePair],
    *,
    poolings: Sequence[str],
    batch_size: int,
) -> dict[str, dict[int, torch.Tensor]]:
    """Per-pooling, per-layer diff-of-means directions for a concept from its raw sentence pairs.

    The same extraction ``eval_awareness_probe`` uses for the validated axis: capture the positives
    and the honest foils separately (raw sentences, no chat template -- the space the axis is
    defined in) and take ``mean(positive) - mean(negative)`` per layer. Every pooling is read off
    the same two forwards, so two poolings cost what one did; the vectors are bit-identical to the
    per-pooling extraction because each pooler sees the same upcast hidden state.
    """
    positive = capture_pooled_activations_multi(
        model, tokenizer, stimuli.positives(list(pairs)), poolings=poolings, batch_size=batch_size
    )
    negative = capture_pooled_activations_multi(
        model, tokenizer, stimuli.negatives(list(pairs)), poolings=poolings, batch_size=batch_size
    )
    directions: dict[str, dict[int, torch.Tensor]] = {}
    for pooling in poolings:
        if positive[pooling].keys() != negative[pooling].keys():
            raise ValueError(
                "positive and negative captures cover different layers: "
                f"{sorted(positive[pooling])} vs {sorted(negative[pooling])}"
            )
        directions[pooling] = {
            layer: diff_of_means(positive[pooling][layer], negative[pooling][layer])
            for layer in positive[pooling]
        }
    return directions


def extract_concept_direction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: Sequence[stimuli.ContrastivePair],
    *,
    pooling: str,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Per-layer diff-of-means direction for a concept at one pooling; see the multi form."""
    return extract_concept_directions(
        model, tokenizer, pairs, poolings=[pooling], batch_size=batch_size
    )[pooling]


def load_validated_eval_direction(
    validated_dir: Path, *, expected_pooling: str, model_id: str
) -> tuple[dict[int, torch.Tensor], str]:
    """Load a validated eval-awareness axis, checking the provenance saved beside it.

    Returns the per-layer axis and the model id it was extracted from. ``directions.pt`` cannot say
    which space it lives in, so the sibling ``metrics.json`` that ``eval_awareness_probe`` writes is
    read: a pooling other than ``expected_pooling`` raises, because projecting the twins onto an
    axis from a different pooling measures a different space and still prints a number. A
    different ``model_id`` does NOT raise -- reusing a base-checkpoint axis on an RL'd checkpoint of
    the same width is a legitimate transfer measurement -- but it is logged and carried into the
    saved artifacts, so the record says which checkpoint the axis came from.
    """
    directions_path = validated_dir / "directions.pt"
    metrics_path = validated_dir / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(
            f"no metrics.json beside {directions_path}, so the axis's pooling and checkpoint are "
            "unknown and it cannot be told from an axis of a different space. Regenerate it: "
            f"`python -m reward_hacking.interp.eval_awareness_probe --pooling {expected_pooling} "
            f"--out-dir {validated_dir}`"
        )
    metrics = json.loads(metrics_path.read_text())
    axis_pooling = metrics["pooling"]
    if axis_pooling != expected_pooling:
        raise ValueError(
            f"the validated axis at {validated_dir} was extracted under {axis_pooling!r} pooling "
            f"but is being loaded as the {expected_pooling!r} axis; the projection would mix two "
            "spaces. Point --out-dir at the directory for this pooling, or re-run "
            "eval_awareness_probe for it."
        )
    axis_model_id: str = metrics["model_id"]
    if axis_model_id != model_id:
        logger.warning(
            "the validated %s-pool eval-awareness axis at %s was extracted from %s, not the %s "
            "being read here: this run measures axis TRANSFER across checkpoints, which is a "
            "different claim from a within-checkpoint read",
            expected_pooling,
            validated_dir,
            axis_model_id,
            model_id,
        )
    directions: dict[int, torch.Tensor] = torch.load(directions_path, weights_only=True)
    logger.info(
        "loaded validated eval-awareness directions for %d layers from %s (pooling=%s model=%s)",
        len(directions),
        directions_path,
        axis_pooling,
        axis_model_id,
    )
    return directions, axis_model_id


@dataclass(frozen=True)
class PromptContrastResult:
    """A whole run: the per-(concept, pooling, layer) reads plus the raw material to re-analyse.

    ``reads`` is every :class:`ContrastRead`. ``prompt_activations`` and ``directions`` are the
    pooled residual streams and the concept axes they were projected onto, kept so adding rigor
    later is a re-analysis rather than a re-run. ``fresh_vs_validated_cosine`` records, per pooling
    per layer, how closely the freshly extracted eval-awareness axis matches the loaded validated
    one -- a sanity check that this pipeline reproduces each validated direction it reuses.
    ``validated_eval_model_ids`` names the checkpoint each reused axis was extracted from, which is
    ``model_id`` in a within-checkpoint read and something else in a transfer read.
    """

    model_id: str
    poolings: tuple[str, ...]
    n_pairs: int
    n_placebos: int
    seed: int
    thinking: bool
    reads: list[ContrastRead]
    prompt_activations: dict[str, dict[str, dict[int, torch.Tensor]]]
    directions: dict[str, dict[str, dict[int, torch.Tensor]]]
    stimulus_pairs: list[StimulusPair]
    fresh_vs_validated_cosine: dict[str, dict[int, float]]
    validated_eval_model_ids: dict[str, str]


def _concept_pairs() -> dict[str, list[stimuli.ContrastivePair]]:
    """Return the raw sentence-pair sets for the four concept directions (incl. the control)."""
    return {
        SHORTCUT: stimuli.SHORTCUT_PAIRS,
        DECEPTION: stimuli.DECEPTION_PAIRS,
        EVAL_AWARENESS: stimuli.EVAL_AWARENESS_PAIRS,
        CONTRADICTION: stimuli.CONTRADICTION_PAIRS,
    }


def run_prompt_contrast(  # noqa: PLR0913 - keyword-only knobs, not worth a wrapper object
    model_id: str,
    episode_dir: Path,
    *,
    poolings: tuple[str, ...] = DEFAULT_POOLINGS,
    n_placebos: int = DEFAULT_N_PLACEBOS,
    seed: int = 0,
    limit: int | None = None,
    concept_limit: int | None = None,
    prompt_batch_size: int = DEFAULT_PROMPT_BATCH_SIZE,
    concept_batch_size: int = DEFAULT_CONCEPT_BATCH_SIZE,
    thinking: bool = True,
    validated_eval_dir: Path = DEFAULT_VALIDATED_EVAL_DIR,
    validated_eval_last_dir: Path = DEFAULT_VALIDATED_EVAL_LAST_DIR,
) -> PromptContrastResult:
    """Load the model, capture both twin groups and the concept axes, and contrast every layer.

    The eval-awareness axis reuses the already-VALIDATED direction for each pooling that has
    one -- the mean-pool axis from ``validated_eval_dir`` (always) and the last-pool axis from
    ``validated_eval_last_dir`` when it exists (produced by ``eval_awareness_probe --pooling
    last``). For shortcut/deception/contradiction, and for any pooling with no validated
    eval axis, the axis is extracted fresh from the sentence pairs at that pooling. A fresh
    eval-awareness axis is also extracted per pooling purely to confirm it reproduces the
    validated one (recorded as a per-layer cosine), never used in a read. Exp2's fragility was
    exactly a fresh, UNVALIDATED last-pool axis; if no validated last-pool axis is on disk this
    logs loudly and falls back to fresh.
    """
    unknown = set(poolings) - set(POOLERS)
    if unknown:
        raise ValueError(f"unknown pooling(s) {sorted(unknown)}; expected from {sorted(POOLERS)}")

    logger.info(
        "prompt-contrast on %s poolings=%s n_placebos=%d seed=%d limit=%s thinking=%s",
        model_id,
        poolings,
        n_placebos,
        seed,
        limit,
        thinking,
    )
    pairs = build_stimulus_pairs(episode_dir, limit=limit)
    concept_pair_sets = {
        name: (pairs_[:concept_limit] if concept_limit else pairs_)
        for name, pairs_ in _concept_pairs().items()
    }
    mean_axis, mean_axis_model_id = load_validated_eval_direction(
        validated_eval_dir, expected_pooling="mean", model_id=model_id
    )
    validated_by_pooling: dict[str, dict[int, torch.Tensor]] = {"mean": mean_axis}
    validated_eval_model_ids = {"mean": mean_axis_model_id}
    if (validated_eval_last_dir / "directions.pt").exists():
        last_axis, last_axis_model_id = load_validated_eval_direction(
            validated_eval_last_dir, expected_pooling="last", model_id=model_id
        )
        validated_by_pooling["last"] = last_axis
        validated_eval_model_ids["last"] = last_axis_model_id
    else:
        logger.warning(
            "no validated last-pool eval-awareness axis at %s; the last-pool eval-awareness read "
            "will use a FRESH, UNVALIDATED axis (the exact Exp2 fragility). Run "
            "`eval_awareness_probe --pooling last --out-dir %s` first to validate it.",
            validated_eval_last_dir,
            validated_eval_last_dir,
        )

    model, tokenizer = load_model_and_tokenizer(model_id)

    prompt_activations: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    directions: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for pooling in poolings:
        conflicting_acts, original_acts = capture_prompt_activations(
            model,
            tokenizer,
            pairs,
            pooling=pooling,
            batch_size=prompt_batch_size,
            thinking=thinking,
        )
        prompt_activations[pooling] = {"conflicting": conflicting_acts, "original": original_acts}
        directions[pooling] = {
            name: extract_concept_direction(
                model, tokenizer, concept_pairs, pooling=pooling, batch_size=concept_batch_size
            )
            for name, concept_pairs in concept_pair_sets.items()
        }
        logger.info("captured prompt activations and concept directions for pooling=%s", pooling)

    fresh_vs_validated_cosine: dict[str, dict[int, float]] = {}
    for pooling_name, validated_axis in validated_by_pooling.items():
        fresh_axis = directions.get(pooling_name, {}).get(EVAL_AWARENESS, {})
        per_layer = {
            layer: cosine(fresh_axis[layer], validated_axis[layer])
            for layer in sorted(fresh_axis)
            if layer in validated_axis
        }
        if per_layer:
            fresh_vs_validated_cosine[pooling_name] = per_layer
            logger.info(
                "fresh vs validated %s-pool eval-awareness cosine: min=%.4f over %d layers",
                pooling_name,
                min(per_layer.values()),
                len(per_layer),
            )

    reads: list[ContrastRead] = []
    for pooling in poolings:
        conflicting_acts = prompt_activations[pooling]["conflicting"]
        original_acts = prompt_activations[pooling]["original"]
        for concept in CONTRAST_CONCEPTS:
            direction_by_layer = _direction_for(
                concept, pooling, directions[pooling], validated_by_pooling
            )
            reads.extend(
                contrast_all_layers(
                    concept,
                    pooling,
                    conflicting_acts,
                    original_acts,
                    direction_by_layer,
                    n_placebos=n_placebos,
                    seed=seed,
                )
            )

    return PromptContrastResult(
        model_id=model_id,
        poolings=poolings,
        n_pairs=len(pairs),
        n_placebos=n_placebos,
        seed=seed,
        thinking=thinking,
        reads=reads,
        prompt_activations=prompt_activations,
        directions=directions,
        stimulus_pairs=pairs,
        fresh_vs_validated_cosine=fresh_vs_validated_cosine,
        validated_eval_model_ids=validated_eval_model_ids,
    )


def _direction_for(
    concept: str,
    pooling: str,
    fresh_directions: dict[str, dict[int, torch.Tensor]],
    validated_by_pooling: dict[str, dict[int, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    (
        """Pick the projection axis: the validated eval-awareness axis for its pooling, else the """
        """fresh one.

    The headline eval-awareness result reuses the already-validated axis for whichever pooling has
    one on disk (mean always; last once ``eval_awareness_probe --pooling last`` has run). Every
    other (concept, pooling) -- shortcut, deception, the contradiction control, or a pooling
    with no validated eval axis -- uses the freshly extracted axis.
    """
    )
    if concept == EVAL_AWARENESS and pooling in validated_by_pooling:
        return validated_by_pooling[pooling]
    return fresh_directions[concept]


# --------------------------------------------------------------------------------------
# Artifacts and reporting
# --------------------------------------------------------------------------------------


def save_artifacts(result: PromptContrastResult, out_dir: Path) -> None:
    """Persist activations, directions, stimuli and metrics so adding rigor later is a re-analysis.

    Everything lands under the gitignored artifacts tree. ``stimuli.json`` carries the raw
    transcripts (which include benchmark grader text) and therefore must never leave this local,
    gitignored location.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.prompt_activations, out_dir / "prompt_activations.pt")
    torch.save(result.directions, out_dir / "directions.pt")
    (out_dir / "stimuli.json").write_text(
        json.dumps(
            [
                {
                    "problem_id": pair.problem_id,
                    "conflicting_transcript": pair.conflicting_transcript,
                    "original_transcript": pair.original_transcript,
                }
                for pair in result.stimulus_pairs
            ],
            indent=2,
        )
    )
    payload = {
        "model_id": result.model_id,
        "poolings": list(result.poolings),
        "concepts": list(CONTRAST_CONCEPTS),
        "n_pairs": result.n_pairs,
        "n_placebos": result.n_placebos,
        "seed": result.seed,
        "thinking": result.thinking,
        "fresh_vs_validated_eval_cosine": result.fresh_vs_validated_cosine,
        "validated_eval_axis_model_ids": result.validated_eval_model_ids,
        # The layer-selection-corrected significance per (concept, pooling), beside the reads whose
        # per-layer ``auc_empirical_p`` is only a within-layer p (see ``LayerSelection``).
        "peak_selection": selection_by_group(result.reads),
        "reads": [metrics_dict(read) for read in result.reads],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote prompt-contrast activations, directions, stimuli and metrics to %s", out_dir)


def format_table(reads: Sequence[ContrastRead]) -> str:
    """Per-(pooling, concept, layer) table: real separation beside the matched-norm placebo."""
    header = (
        f"{'pooling':>7}  {'concept':>14}  {'layer':>5}  {'auc':>6}  {'plc_auc':>7}  "
        f"{'plc_max':>7}  {'auc-plc':>7}  {'sign':>5}  {'plc_sign':>8}  {'paired_t':>8}  "
        f"{'beats':>5}"
    )
    rows = [
        f"{read.pooling:>7}  {read.concept:>14}  {read.layer:>5}  {read.auc:>6.3f}  "
        f"{read.placebo_auc_mean:>7.3f}  {read.placebo_auc_max:>7.3f}  "
        f"{read.auc_above_placebo:>7.3f}  {read.paired_sign_rate:>5.2f}  "
        f"{read.placebo_paired_sign_rate_mean:>8.2f}  {read.paired_t:>8.2f}  "
        f"{read.beats_placebo!s:>5}"
        for read in reads
    ]
    return "\n".join([header, *rows])


def summarize(reads: Sequence[ContrastRead]) -> str:
    """Report, per (concept, pooling), the layer of peak separation and whether it beats placebo.

    Diagnosis, not a verdict: an empty "beats placebo" set is stated plainly as a lead about
    transfer or about the stimulus, never as a stop.
    """
    if not reads:
        return "no reads"
    lines: list[str] = []
    groups = sorted({(read.concept, read.pooling) for read in reads})
    for concept, pooling in groups:
        group = [read for read in reads if read.concept == concept and read.pooling == pooling]
        best = max(group, key=lambda read: read.auc_above_placebo)
        clearing = [read.layer for read in group if read.beats_placebo]
        selection = layer_selection_p(group)
        lines.append(
            f"{concept}/{pooling}: best layer {best.layer} auc={best.auc:.3f} "
            f"(placebo {best.placebo_auc_mean:.3f}, +{best.auc_above_placebo:.3f}, "
            f"within_layer_p={best.auc_empirical_p:.3f}, "
            f"selection_corrected_p={selection.selection_corrected_p:.3f} "
            f"over {best.n_placebos} placebos) "
            f"sign_rate={best.paired_sign_rate:.2f} paired_t={best.paired_t:.2f}; "
            f"layers beating placebo max: {clearing or 'none'}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments for the prompt-contrast projection."""
    parser = argparse.ArgumentParser(
        description="Project conflicting-vs-original agentic-prompt activations onto concept axes"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--poolings", nargs="+", choices=sorted(POOLERS), default=list(DEFAULT_POOLINGS)
    )
    parser.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap matched problem pairs (default: all 100)"
    )
    parser.add_argument(
        "--concept-limit",
        type=int,
        default=None,
        help="cap sentence pairs per concept direction (default: all); a smoke lever",
    )
    parser.add_argument("--prompt-batch-size", type=int, default=DEFAULT_PROMPT_BATCH_SIZE)
    parser.add_argument("--concept-batch-size", type=int, default=DEFAULT_CONCEPT_BATCH_SIZE)
    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument("--validated-eval-dir", type=Path, default=DEFAULT_VALIDATED_EVAL_DIR)
    parser.add_argument(
        "--validated-eval-last-dir", type=Path, default=DEFAULT_VALIDATED_EVAL_LAST_DIR
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=None,
        help="scratch /work for materialising tasks; must live outside a home tree. "
        "Default: a fresh mkdtemp under the system temp dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the prompt-contrast projection, print the per-layer table, and save the raw material."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    episode_dir = args.episode_dir or Path(tempfile.mkdtemp(prefix="prompt-contrast-")) / "work"
    created_scratch = args.episode_dir is None
    try:
        result = run_prompt_contrast(
            args.model_id,
            episode_dir,
            poolings=tuple(args.poolings),
            n_placebos=args.n_placebos,
            seed=args.seed,
            limit=args.limit,
            concept_limit=args.concept_limit,
            prompt_batch_size=args.prompt_batch_size,
            concept_batch_size=args.concept_batch_size,
            thinking=args.thinking,
            validated_eval_dir=args.validated_eval_dir,
            validated_eval_last_dir=args.validated_eval_last_dir,
        )
    finally:
        if created_scratch and episode_dir.parent.exists():
            shutil.rmtree(episode_dir.parent, ignore_errors=True)

    print(format_table(result.reads))  # noqa: T201  # Intentional CLI table output.
    logger.info("summary:\n%s", summarize(result.reads))
    save_artifacts(result, args.out_dir)


if __name__ == "__main__":
    main()
