"""Adapter around the Jacobian lens (`jlens`), the repo's primary interpretability method.

Owner's rule: lead with the most rigorous interp method available (Jacobian-space), not the easiest.
This wraps the reference `jlens` package -- run via ``PYTHONPATH`` from a clone of
github.com/anthropics/jacobian-lens (commit 581d398, v0.1.0); it is deliberately NOT in this repo's
lockfile -- to do three things a direction analysis needs:

* **load a lens** -- FIT OUR OWN (the rule-aligned default: a closed-form accumulation, no
  optimiser, ~2.1 h / 100 prompts on an L4-class card) or reuse the pre-fit 4B lens (a fast
  path, direction DECODING only -- it is WikiText-fit, so a poor next-token predictor on
  agentic prompts);
* **transport** a residual-space direction (e.g. the eval-awareness diff-of-means axis) through the
  lens into final-layer coordinates;
* **decode** it through the model's own unembedding into a ranked token list, so a direction reads
  as "this promotes {tokens}" rather than as a bare cosine.

Every path that touches `jlens`, CUDA, or the 4B weights is GPU-only: the CPU route is closed on
the Qwen3.5 family (the linear-attention path dispatches to a Triton kernel that needs CUDA). The
offline tests exercise only the non-GPU logic -- config validation, the self-capped fit-prompt
list (the reference ``jlens.fit`` has NO auto-stop, so the caller must cap), the top-k decode
from a vocab-logit vector, and the transport/decode wiring against stubs.

The fit is closed-form and cheap: as few as ~10 prompts already beat the logit/tuned lens with only
modest gains toward 1000, so a ~10-prompt smoke is enough to sanity-check the fit path before any
large run. See docs/interp-methods/jacobian-space.md for the fit/apply API, the per-model table, and
the silent-failure gotchas.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from types import ModuleType

    from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# The pre-fit 4B lens on the Neuronpedia weights repo. WikiText-fit, so direction-DECODING only.
DEFAULT_PRETRAINED_REPO = "neuronpedia/jacobian-lens"
DEFAULT_PRETRAINED_FILENAME = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
)
DEFAULT_PRETRAINED_REVISION = "qwen-n1000"  # a BRANCH, not a tag
JLENS_COMMIT = "581d398"  # v0.1.0, the pinned reference commit

JLENS_LOAD_PATH = "AutoModelForCausalLM.from_pretrained(dtype=bfloat16).to(cuda)"
"""Names the loader :func:`_load_jlens_model` is, for the lens cache key.

The class a loader yields is a function of (loader, checkpoint), so naming the loader is what
distinguishes two lenses fit on the same weights through different module trees -- the confound
that once forced a deliberate 65-minute refit (bare ``AutoModelForCausalLM`` vs the composite class
``games.lora`` builds). Change this string whenever the loader's class, dtype or device placement
changes, or a cached lens fit through the old path will be served for the new one."""

JLENS_UNMERGED_LOAD_PATH = (
    "games.lora.load_adapter_base + attach_adapter; from_hf(get_base_model())"
)
"""Names the un-merged loader the games lens ladder uses, for the same cache key.

A distinct string because it is a distinct lens. The merged path fits on ``AutoModelForCausalLM``
weights that realise about 64% of the trained delta after the bf16 round; the un-merged path fits on
the composite class with LoRA modules live, so the Jacobian sees ``Wx + B(Ax)`` at full precision.
Probe I6 (2026-09-02) measured the gap between the two fits at 0.6% median and 1.8% max relative
Frobenius on an over-sized random adapter, against a 0.33% wrapper floor -- small, real, and enough
that serving one under the other's key would publish a lens nobody fitted. The base cell differs too:
the two loaders build different classes for the same weights (0.33% floor), which is exactly the
confound this constant exists to keep out of the cache."""

# The save/reload tolerance for a fitted lens: jlens saves fp16, so a reload is exact to fp16.
ROUNDTRIP_RTOL = 2e-2
ROUNDTRIP_ATOL = 1e-3

LENS_CACHE_KEY_FILENAME = "lens_cache_key.json"
LENS_CACHE_LENS_FILENAME = "lens.pt"
DEFAULT_S3_COMMAND: tuple[str, ...] = ("aws", "s3")
DEFAULT_S3_TIMEOUT_SECONDS = 30 * 60

# Leading positions excluded from the reconstruction read, matching ``jlens.fitting``'s fit-time
# ``SKIP_FIRST_N_POSITIONS``: early positions are attention sinks with atypical residual statistics,
# so the fitted Jacobian was averaged strictly past them. Measuring reconstruction on positions the
# fit itself excluded would score the lens on a regime it never saw.
RECON_SKIP_FIRST = 16

LensSource = Literal["fit_own", "pretrained"]


class LensLike(Protocol):
    (
        """The slice of ``jlens.JacobianLens`` this adapter calls: transport a direction """
        """through a layer."""
    )

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        """Transport a layer-``layer`` residual-space vector into final-layer coordinates."""
        ...


class UnembedModel(Protocol):
    """The slice of a ``jlens.from_hf`` model this adapter calls: decode to vocab logits."""

    def unembed(self, transported: torch.Tensor) -> torch.Tensor:
        (
            """Project a final-layer-coordinate vector to ``[vocab]`` logits via the model's """
            """unembedding."""
        )
        ...


@dataclass(frozen=True)
class TokenReadout:
    """One decoded token and its (normalised) readout logit -- a row of a direction's token list."""

    token: str
    logit: float


@dataclass(frozen=True)
class JacobianConfig:
    """How to obtain and apply a lens for a model.

    ``source`` picks the rule-aligned default (``fit_own``) or the fast pre-fit path
    (``pretrained``). The fit knobs (``max_fit_prompts`` / ``dim_batch`` / ``max_seq_len`` /
    ``checkpoint_path``) size a fit and are named exactly as ``jlens.fit`` names them, so a knob
    that fails to arrive is visible rather than shadowed by a same-valued reference default; the
    pretrained knobs name the Neuronpedia artifact. ``top_k`` is how many tokens a decode returns.

    ``dim_batch`` is a MEMORY knob, not a speed knob. It sets how many residual dimensions each
    backward pass carries (the prompt is replicated that many times), so live activation memory
    scales with it while the total backward FLOPs do not (``jlens.fitting.jacobian_for_prompt``:
    "total backward FLOPs are unchanged"). Measured fits at 2B and 9B gained little past 16, and 64
    OOM'd both a 44 GiB and a 96 GiB card, each costing a failed attempt and a relaunch. Size it to
    the VRAM to spare, never raise it hoping for a faster fit.

    ``checkpoint_every`` and ``resume`` are the restart knobs. The fit is a running fp32 sum, so a
    checkpoint is the exact accumulator and a resumed fit equals a straight one; ``jlens.fit`` writes
    one every ``checkpoint_every`` prompts (``None`` writes only at the end) and, with ``resume``,
    continues from whatever ``checkpoint_path`` already holds. Each write is
    ``len(source_layers) * d_model**2 * 4`` bytes -- 96 MB at 1024 wide, about 2 GB at the 9B's 4096
    -- so a long fit raises the interval rather than paying that per prompt. Both are threaded
    explicitly because a knob that fails to arrive is shadowed by a same-valued reference default.
    """

    model_id: str = "Qwen/Qwen3.5-4B"
    source: LensSource = "fit_own"
    top_k: int = 30
    max_fit_prompts: int = 200
    dim_batch: int = 16
    max_seq_len: int = 128
    checkpoint_path: Path | None = None
    checkpoint_every: int | None = 1
    resume: bool = True
    pretrained_repo: str = DEFAULT_PRETRAINED_REPO
    pretrained_filename: str = DEFAULT_PRETRAINED_FILENAME
    pretrained_revision: str = DEFAULT_PRETRAINED_REVISION

    def __post_init__(self) -> None:
        """Reject configurations that would fail only later, on a GPU, after a long model load."""
        if self.source not in ("fit_own", "pretrained"):
            raise ValueError(
                f"unknown lens source {self.source!r}; expected 'fit_own' or 'pretrained'"
            )
        if self.max_fit_prompts <= 0:
            raise ValueError(f"max_fit_prompts must be positive, got {self.max_fit_prompts}")
        if self.dim_batch <= 0:
            raise ValueError(f"dim_batch must be positive, got {self.dim_batch}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError(
                f"checkpoint_every must be positive or None (write only at the end), got "
                f"{self.checkpoint_every}"
            )


# --------------------------------------------------------------------------------------
# Pure logic (no jlens, no CUDA): exercised by the offline tests
# --------------------------------------------------------------------------------------


def cap_fit_prompts(prompts: Sequence[str], config: JacobianConfig) -> list[str]:
    (
        """Cap the fit-prompt list to ``config.max_fit_prompts`` -- because ``jlens.fit`` never """
        """stops itself.

    The reference ``jlens.fit`` has NO auto early-stop (no ``stop_at_delta``): it iterates over
    every prompt handed to it, so the ONLY thing that bounds the fit is this cap. Truncation
    is logged loudly rather than done silently; watch the fit's logged ``max_d_mean`` to judge
    convergence instead of trusting a prompt count.
    """
    )
    prompts = list(prompts)
    if len(prompts) > config.max_fit_prompts:
        logger.warning(
            "capping fit prompts %d -> %d: jlens.fit has NO auto-stop and iterates every prompt",
            len(prompts),
            config.max_fit_prompts,
        )
        return prompts[: config.max_fit_prompts]
    logger.info(
        "fitting on %d prompts (<= cap %d); jlens.fit has no stop_at_delta, watch logged "
        "max_d_mean",
        len(prompts),
        config.max_fit_prompts,
    )
    return prompts


def decode_topk(
    readout_logits: torch.Tensor, id_to_token: Callable[[int], str], k: int
) -> list[TokenReadout]:
    """Top-``k`` tokens of a ``[vocab]`` readout-logit vector, decoded via ``id_to_token``.

    The readout is a normalised logit (the RMSNorm inside the unembed makes it a ranking,
    not a probability), so ``k`` is clamped to the vocabulary size and the ranking is what
    carries meaning.
    """
    if readout_logits.ndim != 1:
        raise ValueError(f"expected a 1-D [vocab] readout, got shape {tuple(readout_logits.shape)}")
    k = min(k, int(readout_logits.shape[0]))
    values, indices = torch.topk(readout_logits, k)
    return [
        TokenReadout(token=id_to_token(int(index)), logit=float(value))
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def transport_and_decode(  # noqa: PLR0913 - lens, model, direction, layer and the two decode knobs
    lens: LensLike,
    model: UnembedModel,
    direction: torch.Tensor,
    layer: int,
    *,
    id_to_token: Callable[[int], str],
    k: int,
) -> list[TokenReadout]:
    """Transport ``direction`` at ``layer`` through the lens, unembed it, and decode the top ``k``.

    The one-line core of "decode a direction": ``model.unembed(lens.transport(h, L))`` yields
    ``[vocab]`` logits, which :func:`decode_topk` turns into a readable token list. Apply it
    identically to the real axis and to the matched-norm placebo so the comparison stays fair.
    """
    readout = model.unembed(lens.transport(direction, layer))
    return decode_topk(readout, id_to_token, k)


# --------------------------------------------------------------------------------------
# Pure logic: lens fit-quality (reconstruction of the model's own logits from each layer)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerReconstruction:
    """How well the lens reproduces the model's OWN final logits from one source layer.

    Both numbers compare the lens readout ``unembed(J_l @ h_l)`` against the model's actual
    final-layer logits ``unembed(h_final)`` at the same positions. ``relative_residual`` is the
    Frobenius norm of the difference over the model's own logit norm (0 is a perfect reproduction,
    growing without bound as the lens diverges); ``explained_variance`` is the standard
    ``1 - SS_res/SS_tot`` (1 is perfect, 0 is no better than predicting the mean logit, negative is
    worse than that). The Jacobian lens is only expected to be coherent in an intermediate layer
    band, so a poor number at an early or late layer is a known property, not a fault -- which is
    why the per-layer breakdown is kept rather than collapsed to one figure.
    """

    layer: int
    relative_residual: float
    explained_variance: float
    n_samples: int


def layer_reconstruction(
    layer: int, predicted: torch.Tensor, actual: torch.Tensor
) -> LayerReconstruction:
    """Score one source layer's lens readout against the model's actual logits.

    ``predicted`` and ``actual`` are both ``[n_samples, vocab]`` (the lens logits at ``layer`` and
    the model's final logits, at the same read positions). Raises rather than returning a degenerate
    number on a shape mismatch, on all-zero targets (the relative residual has no denominator), or
    on constant targets (explained variance has no total sum of squares) -- each is a broken input,
    not a lens that happens to reconstruct perfectly.
    """
    if predicted.shape != actual.shape:
        raise ValueError(
            f"predicted {tuple(predicted.shape)} and actual {tuple(actual.shape)} must match"
        )
    if actual.ndim != 2:  # noqa: PLR2004 - logits are a [n_samples, vocab] matrix by contract
        raise ValueError(f"expected [n_samples, vocab] logits, got shape {tuple(actual.shape)}")
    predicted = predicted.float()
    actual = actual.float()
    residual = predicted - actual
    actual_norm = actual.norm()
    if actual_norm == 0:
        raise ValueError(
            "actual logits have zero norm; cannot form a relative reconstruction error"
        )
    ss_residual = residual.pow(2).sum()
    ss_total = (actual - actual.mean()).pow(2).sum()
    if ss_total == 0:
        raise ValueError("actual logits are constant; explained variance is undefined")
    return LayerReconstruction(
        layer=layer,
        relative_residual=(residual.norm() / actual_norm).item(),
        explained_variance=(1.0 - ss_residual / ss_total).item(),
        n_samples=int(actual.shape[0]),
    )


@dataclass(frozen=True)
class ReconstructionReport:
    """Fit quality across every fitted source layer, plus the aggregates that summarise it.

    ``per_layer`` is one :class:`LayerReconstruction` per source layer, ordered by layer.
    ``median_relative_residual`` is the headline: the median is robust to the early/late layers the
    lens is not expected to reconstruct well, whereas the mean is dragged by them. ``best_layer`` /
    ``best_layer_relative_residual`` name the single layer the lens reproduces most faithfully --
    the "how good does this lens get in its coherent band" read. All numbers are aggregate scalars;
    no per-token logits or activations are retained.
    """

    per_layer: list[LayerReconstruction]
    n_samples: int
    mean_relative_residual: float
    median_relative_residual: float
    best_layer: int
    best_layer_relative_residual: float
    mean_explained_variance: float
    median_explained_variance: float


def reconstruction_report(
    predicted_by_layer: dict[int, torch.Tensor], actual: torch.Tensor
) -> ReconstructionReport:
    """Score every source layer's lens readout against the model's logits and aggregate.

    ``predicted_by_layer`` maps each fitted source layer to its ``[n_samples, vocab]`` lens logits;
    ``actual`` is the model's own final logits at the same positions. Raises on an empty map, since
    a report over no layers is not a fit-quality read.
    """
    if not predicted_by_layer:
        raise ValueError("no per-layer lens logits to score; the lens fitted no source layers")
    per_layer = [
        layer_reconstruction(layer, predicted_by_layer[layer], actual)
        for layer in sorted(predicted_by_layer)
    ]
    residuals = [read.relative_residual for read in per_layer]
    variances = [read.explained_variance for read in per_layer]
    best = min(per_layer, key=lambda read: read.relative_residual)
    return ReconstructionReport(
        per_layer=per_layer,
        n_samples=int(actual.shape[0]),
        mean_relative_residual=statistics.fmean(residuals),
        median_relative_residual=statistics.median(residuals),
        best_layer=best.layer,
        best_layer_relative_residual=best.relative_residual,
        mean_explained_variance=statistics.fmean(variances),
        median_explained_variance=statistics.median(variances),
    )


DEFAULT_MAX_SEQ_LEN_CEILING = 2048
"""How long a fit window may get before cost, rather than the corpus, decides it.

The fit truncates every corpus item to ``max_seq_len`` and its cost per prompt grows with it, so the
corpus picks the window and this bounds it. 2048 covers the games stimulus corpus whole (its longest
rendered stimulus is 452 tokens) and truncates the reward-hacking twin transcripts, which run to
22.3k; a run that wants their divergent tails raises it deliberately and pays for it, and
:class:`SeqLenPlan` reports how much of the corpus any remaining ceiling cuts off.
"""


@dataclass(frozen=True)
class SeqLenPlan:
    """What ``max_seq_len`` a fit will run at, and what it costs on this corpus."""

    max_seq_len: int
    corpus_max_tokens: int
    corpus_median_tokens: int
    n_truncated: int
    fraction_truncated: float

    def as_payload(self) -> dict[str, object]:
        """Return the block a fit report carries: what the fit read and what it did not see."""
        return asdict(self)


def derive_max_seq_len(
    token_lengths: Sequence[int], *, ceiling: int = DEFAULT_MAX_SEQ_LEN_CEILING
) -> SeqLenPlan:
    """Choose a fit's ``max_seq_len`` from the corpus, and say what any truncation costs.

    The whole point is not to inherit ``jlens.fit``'s 128-token default: it would fit on a fraction of
    each prompt and drop whatever the contrast lives in, with nothing going red. The games stimulus
    corpus runs 368-451 tokens with its two sides first differing at index 326-415, and the
    reward-hacking twin transcripts run 1.3k-22.3k, so at 128 both fit on a shared prefix. The corpus
    maximum is used when it fits under ``ceiling``, and otherwise the ceiling is used and the number of
    items it truncates is reported rather than discovered later.

    Shared by both lens ladders (`games.interp_lens_ladder` and the reward-hacking `fit-lens` stage) so
    there is one derivation and one default; it lives here because this side must not import games.
    """
    if not token_lengths:
        raise ValueError("no corpus token lengths, so no max_seq_len can be derived.")
    corpus_max = max(token_lengths)
    chosen = min(corpus_max, ceiling)
    truncated = [length for length in token_lengths if length > chosen]
    plan = SeqLenPlan(
        max_seq_len=chosen,
        corpus_max_tokens=corpus_max,
        corpus_median_tokens=int(statistics.median(token_lengths)),
        n_truncated=len(truncated),
        fraction_truncated=len(truncated) / len(token_lengths),
    )
    if plan.n_truncated:
        logger.warning(
            "max_seq_len=%d truncates %d of %d corpus items (longest %d); the fit will not see "
            "their tails",
            plan.max_seq_len,
            plan.n_truncated,
            len(token_lengths),
            corpus_max,
        )
    else:
        logger.info(
            "max_seq_len=%d covers every corpus item (median %d, longest %d)",
            plan.max_seq_len,
            plan.corpus_median_tokens,
            corpus_max,
        )
    return plan


def interior_eval_positions(seq_len: int, *, skip_first: int, max_positions: int) -> list[int]:
    """Evenly-spaced interior read positions for the reconstruction eval, or ``[]`` if too short.

    Reconstruction is read on positions in ``[skip_first, seq_len - 1)``: the leading band is the
    attention-sink region the fit excludes (:data:`RECON_SKIP_FIRST`) and the final position has no
    next-token target, so both mirror ``jlens.fitting.valid_position_mask`` -- the lens is scored on
    the same positions it was averaged over. At most ``max_positions`` are returned, spread evenly
    (endpoints included) so a long prompt is sampled across its interior rather than at one spot. A
    prompt with no valid interior returns ``[]``, which the caller counts and skips rather than
    indexing out of range.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    if max_positions <= 0:
        raise ValueError(f"max_positions must be positive, got {max_positions}")
    interior = list(range(skip_first, max(seq_len - 1, skip_first)))
    if not interior:
        return []
    count = min(max_positions, len(interior))
    if count == 1:
        return [interior[0]]
    return [interior[round(i * (len(interior) - 1) / (count - 1))] for i in range(count)]


# --------------------------------------------------------------------------------------
# GPU orchestration (jlens + CUDA + 4B weights): not run by the offline tests
# --------------------------------------------------------------------------------------


def _require_jlens() -> ModuleType:
    """Import `jlens` or raise with the exact way to make it importable.

    `jlens` is intentionally absent from the lockfile; it runs via ``PYTHONPATH``. A missing import
    is a setup step, not a bug, so the error says how to fix it rather than crashing obscurely.
    """
    try:
        return importlib.import_module("jlens")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "jlens is not importable. It runs via PYTHONPATH from a clone of "
            f"github.com/anthropics/jacobian-lens (commit {JLENS_COMMIT}, v0.1.0); "
            "it is not in this "
            "repo's lockfile. Clone it and run with PYTHONPATH=<clone-dir> set."
        ) from exc


def _load_jlens_model(config: JacobianConfig, jl: ModuleType) -> tuple[object, AutoTokenizer]:
    """Load the HF model + tokenizer and wrap them with ``jlens.from_hf`` (GPU only).

    Uses ``.to("cuda")`` rather than ``device_map=`` on purpose: the `jlens` reference hooks the
    decoder blocks and takes gradients through them, and expects one plain single-device model, so
    this loader deliberately does not shard (``accelerate`` is pinned in this repo and the sibling
    ``directions.load_model_and_tokenizer`` does pass ``device_map``, so availability is not the
    reason). The cost is real and worth knowing: ``.to("cuda")`` materialises the whole checkpoint
    in host RAM first, which matters at 27B. Raises early on a CPU-only box, since the Qwen3.5
    linear-attention Triton kernel cannot run there.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the Qwen3.5 CPU route is closed (linear-attention Triton kernel needs CUDA); "
            "run the Jacobian adapter on a GPU"
        )
    transformers = importlib.import_module("transformers")
    hf = transformers.AutoModelForCausalLM.from_pretrained(config.model_id, dtype="bfloat16").to(
        "cuda"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_id)
    return jl.from_hf(hf, tokenizer), tokenizer


def load_pretrained_lens(config: JacobianConfig, jl: ModuleType) -> LensLike:
    """Load the pre-fit lens named in ``config`` (fast path, direction-decoding only)."""
    logger.info(
        "loading pre-fit lens %s/%s@%s (WikiText-fit: direction-decoding only)",
        config.pretrained_repo,
        config.pretrained_filename,
        config.pretrained_revision,
    )
    return jl.JacobianLens.from_pretrained(
        config.pretrained_repo,
        filename=config.pretrained_filename,
        revision=config.pretrained_revision,
    )


def fit_lens(
    config: JacobianConfig, model: object, prompts: Sequence[str], jl: ModuleType
) -> LensLike:
    """Fit our own lens on ``prompts`` (closed-form; the rule-aligned default).

    Caps the prompt list first (:func:`cap_fit_prompts`) because ``jlens.fit`` has no auto-stop.
    Interrupting the fit yields a worse lens, not a broken one, so ``checkpoint_path``,
    ``checkpoint_every`` and ``resume`` are threaded through for restartability: the checkpoint is
    the exact fp32 accumulator, and ``reward_hacking.interp.lens_fit_gate`` checks that a fit resumed
    from one equals a straight fit. ``dim_batch`` and ``max_seq_len`` set the fit's schedule rather
    than its estimate -- live activation memory and the truncation window; ``dim_batch`` buys no
    FLOPs, see :class:`JacobianConfig` -- so every knob is passed explicitly; left off, the reference
    defaults (8, 128, a checkpoint per prompt) silently replace whatever the config asked for.
    """
    capped = cap_fit_prompts(prompts, config)
    logger.info(
        "fitting a %d-prompt Jacobian lens on %s (closed-form, dim_batch=%d max_seq_len=%d "
        "checkpoint_every=%s resume=%s)",
        len(capped),
        config.model_id,
        config.dim_batch,
        config.max_seq_len,
        config.checkpoint_every,
        config.resume,
    )
    return jl.fit(
        model,
        capped,
        dim_batch=config.dim_batch,
        max_seq_len=config.max_seq_len,
        checkpoint_path=None if config.checkpoint_path is None else str(config.checkpoint_path),
        checkpoint_every=config.checkpoint_every,
        resume=config.resume,
    )


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute elementwise difference between two tensors (a round-trip check)."""
    return float((a - b).abs().max())


def verify_lens_roundtrip(jl: object, lens: object, lens_path: Path) -> None:
    """Reload the saved lens and confirm it round-trips (finite, matches within fp16 tolerance).

    The save is fp16 and the fit is float32, so equality is to :data:`ROUNDTRIP_RTOL` /
    :data:`ROUNDTRIP_ATOL` rather than bit-exact; a layer that reloads outside that, or was never
    finite, means the artifact on disk is not the lens that was fitted.
    """
    reloaded = jl.JacobianLens.load(str(lens_path))  # pyright: ignore[reportAttributeAccessIssue]
    original_jac = lens.jacobians  # pyright: ignore[reportAttributeAccessIssue]
    reloaded_jac = reloaded.jacobians
    if original_jac.keys() != reloaded_jac.keys():
        raise RuntimeError("reloaded lens covers different layers than the fitted one")
    for layer, tensor in original_jac.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"fitted lens layer {layer} has non-finite entries")
        reloaded_tensor = reloaded_jac[layer].to(tensor)
        if not torch.allclose(tensor, reloaded_tensor, rtol=ROUNDTRIP_RTOL, atol=ROUNDTRIP_ATOL):
            diff = max_abs_diff(tensor, reloaded_tensor)
            raise RuntimeError(f"lens layer {layer} did not round-trip: max|diff|={diff:.2e}")
    logger.info(
        "lens round-trip OK: %d layers reload finite and within tolerance", len(original_jac)
    )


def verify_cached_lens(lens: object, model: object) -> None:
    """Refuse a cache hit whose shape is not the lens a fit on ``model`` would have produced.

    A cache hit has no freshly fitted lens to compare against, so what CAN be checked is checked:
    every source layer below the target is present (``jlens.fit`` defaults to exactly that set),
    the residual width matches, and every entry is finite. Reconstruction quality is then read by
    the caller exactly as it is after a fit, so a wrong-but-well-formed lens still shows up there.
    """
    n_layers = int(model.n_layers)  # pyright: ignore[reportAttributeAccessIssue]
    d_model = int(model.d_model)  # pyright: ignore[reportAttributeAccessIssue]
    expected_layers = list(range(n_layers - 1))
    source_layers = list(lens.source_layers)  # pyright: ignore[reportAttributeAccessIssue]
    if source_layers != expected_layers:
        raise RuntimeError(
            f"cached lens fits {len(source_layers)} source layers "
            f"({min(source_layers, default='none')}..{max(source_layers, default='none')}) but a "
            f"fit on this {n_layers}-layer model would fit layers 0..{n_layers - 2}"
        )
    if int(lens.d_model) != d_model:  # pyright: ignore[reportAttributeAccessIssue]
        raise RuntimeError(f"cached lens has d_model={lens.d_model}, model has {d_model}")  # pyright: ignore[reportAttributeAccessIssue]
    for layer, tensor in lens.jacobians.items():  # pyright: ignore[reportAttributeAccessIssue]
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"cached lens layer {layer} has non-finite entries")


# --------------------------------------------------------------------------------------
# Lens cache: a fitted lens keyed by everything the fit depends on
# --------------------------------------------------------------------------------------


def digest_strings(values: Iterable[str]) -> str:
    """Length-delimited sha256 over an ordered sequence of strings.

    Length-delimited so neither a reordering nor a boundary shift between two adjacent values can
    collide (``"ab" + "c"`` and ``"a" + "bc"`` would otherwise hash the same). The same construction
    as ``games.interp_cells.digest_of_strings``, kept here because this side must not import games.
    """
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(str(len(encoded)).encode())
        digest.update(b"\x00")
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class LensCacheKey:
    """Everything a fitted lens depends on. A stale key is the cache's only failure mode.

    That holds only while the weights a fit actually loaded are the ones its key names; the games
    ladder guards that side with a provenance sidecar in every merged export
    (`games.interp_lens_ladder.materialize_model_dir`), so a leftover merge from another adapter at
    the same arm and step is re-merged rather than fitted and published under the new adapter's key.

    The fields are the answer to "what would make two fits differ": the weights (base revision plus
    the adapter merged in, or none), how they were loaded (:data:`JLENS_LOAD_PATH` and the merge
    dtype, which decides how much of a trained delta the merged weights realise), the exact ordered
    fit strings (which subsumes the stimuli, the render convention and the split), the truncation
    window, the fit's own skip-first rule, ``dim_batch`` (its reduction order is part of the bits),
    and the jlens commit. Nothing about the box: the same key on another card gives a lens equal to
    within Triton's run-to-run reduction noise, which is the stat grade this cache is allowed.
    """

    base_model: str
    base_weights_identity: str
    adapter_weights_sha256: str | None
    adapter_config_sha256: str | None
    merge_dtype: str | None
    load_path: str
    fit_prompts_sha256: str
    n_fit_prompts: int
    max_seq_len: int
    skip_first: int
    dim_batch: int
    jlens_commit: str

    def as_payload(self) -> dict[str, object]:
        """Return the JSON sidecar stored beside a cached lens, read back on every hit."""
        return asdict(self)

    @property
    def sha256(self) -> str:
        """The cache entry name: a digest of the sorted-key JSON of every field."""
        return hashlib.sha256(json.dumps(self.as_payload(), sort_keys=True).encode()).hexdigest()


def local_weights_digest(model_dir: Path) -> str:
    """sha256 over a local checkpoint's ``config.json`` and every ``*.safetensors`` file, by name.

    The fallback identity for weights that have no hub revision. Reading gigabytes is slow next to
    a config lookup and fast next to the fit it saves; a path string alone would let a checkpoint
    rewritten in place hit a stale lens, which is the one failure this cache must not have.
    """
    digest = hashlib.sha256()
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{model_dir} holds no *.safetensors to identify the weights by")
    for path in [model_dir / "config.json", *files]:
        digest.update(path.name.encode())
        digest.update(b"\x00")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
    return digest.hexdigest()


def resolve_weights_identity(model_id: str, *, revision: str | None = None) -> str:
    """Pin which weights ``model_id`` names: the hub commit, or a digest of a local directory.

    ``AutoConfig.from_pretrained`` carries the resolved commit of a hub checkpoint on the config it
    returns, so a model id that silently moves to a newer revision changes the key rather than
    serving the old lens; ``revision`` asks for a specific branch or commit and the returned
    identity is still the commit it resolved to. A local directory has no revision and is digested
    instead (:func:`local_weights_digest`). Anything else is refused: an identity that cannot be
    resolved would otherwise degrade into "whatever this string points at today".
    """
    path = Path(model_id)
    if path.is_dir():
        return "sha256:" + local_weights_digest(path)
    transformers = importlib.import_module("transformers")
    config = transformers.AutoConfig.from_pretrained(model_id, revision=revision)
    commit = getattr(config, "_commit_hash", None)
    if not commit:
        raise RuntimeError(
            f"could not resolve a weights revision for {model_id!r}: the config carries no commit "
            "hash and it is not a local directory, so a cache key would not pin the weights"
        )
    return f"hf:{commit}"


def fit_skip_first(jl: ModuleType) -> int:
    """Read the leading-position exclusion ``jlens.fit`` applies by default off the module itself."""
    return int(jl.fitting.SKIP_FIRST_N_POSITIONS)


class S3CommandError(RuntimeError):
    """An ``aws s3`` call failed for a reason other than the object not existing.

    This covers a hang past the timeout as well as a non-zero exit: every transport failure arrives
    as this one type, so a caller that wants to survive one (a store after a paid fit) catches one
    thing and a caller that wants to stop on one (a lookup before the weights load) lets it through.
    """


def _run_s3(argv: Sequence[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    """Run one ``aws s3`` command; a hang past ``timeout_seconds`` is an :class:`S3CommandError`.

    ``subprocess.run`` kills the child and raises ``TimeoutExpired`` on a timeout. Left uncaught that
    would crash :func:`acquire_lens` AFTER a fit it just paid for, on the one path whose whole point
    is that a transport failure there must not cost the fit, so the conversion happens here, at the
    only place the real runner is called.
    """
    try:
        return subprocess.run(  # noqa: S603 - argv is built from constants and validated URIs
            list(argv), capture_output=True, text=True, check=False, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        raise S3CommandError(
            f"`{' '.join(argv)}` did not finish within {timeout_seconds}s and was killed"
        ) from error


@dataclass(frozen=True)
class LensCache:
    """Fitted lenses at ``root`` (a local directory or an ``s3://`` prefix), one entry per key.

    Entry layout is ``<root>/<key.sha256>/lens.pt`` plus ``lens_cache_key.json``, and the key file
    is always written LAST: a lookup that finds the key file knows the lens beside it is complete,
    so a store that died mid-upload can never read as a hit. On a hit the stored key payload is
    compared field by field against the key being looked up, which is belt-and-braces against a
    digest collision or an entry edited by hand.

    Transport is the ``aws s3`` CLI through ``subprocess``, the same choice ``games.s3_sync`` made
    and for the same reason (retries, multipart and credentials are its problem); ``runner`` is
    injectable so the S3 classification logic is testable without a bucket. A runner returns the
    finished process or raises :class:`S3CommandError`; the default one turns a hang past
    ``timeout_seconds`` into that error rather than letting ``TimeoutExpired`` escape. A missing
    object is the only ``ls`` failure treated as a miss -- anything else (denied, no credentials, a
    typo in the bucket) raises, because a run that quietly refits on every cell hides a
    misconfigured cache.
    """

    root: str
    s3_command: tuple[str, ...] = DEFAULT_S3_COMMAND
    timeout_seconds: int = DEFAULT_S3_TIMEOUT_SECONDS
    runner: Callable[[Sequence[str], int], subprocess.CompletedProcess[str]] = _run_s3

    @property
    def is_s3(self) -> bool:
        """Whether ``root`` is an S3 prefix rather than a local directory."""
        return self.root.startswith("s3://")

    def entry(self, key: LensCacheKey) -> str:
        """Return the location of one key's entry under ``root``."""
        return f"{self.root.rstrip('/')}/{key.sha256}"

    def _exists(self, location: str) -> bool:
        if not self.is_s3:
            return Path(location).is_file()
        finished = self.runner((*self.s3_command, "ls", location), self.timeout_seconds)
        name = location.rsplit("/", 1)[-1]
        if finished.returncode == 0:
            return name in finished.stdout
        if finished.returncode == 1 and not finished.stdout.strip() and not finished.stderr.strip():
            return False
        raise S3CommandError(
            f"`aws s3 ls {location}` exited {finished.returncode}: {finished.stderr.strip()[:2000]}"
        )

    def _copy(self, source: str, destination: str) -> None:
        if not self.is_s3:
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(target.name + ".tmp")
            shutil.copyfile(source, staging)
            staging.replace(target)
            return
        finished = self.runner(
            (*self.s3_command, "cp", source, destination, "--only-show-errors"),
            self.timeout_seconds,
        )
        if finished.returncode != 0:
            raise S3CommandError(
                f"`aws s3 cp {source} {destination}` exited {finished.returncode}: "
                f"{finished.stderr.strip()[:2000]}"
            )

    def lookup(self, key: LensCacheKey, dest: Path) -> bool:
        """Copy the cached lens for ``key`` to ``dest`` if one exists; return whether it did.

        The key sidecar lands beside ``dest`` as ``lens_cache_key.json`` either way, so a cell
        directory says which key its lens answers to.
        """
        entry = self.entry(key)
        key_location = f"{entry}/{LENS_CACHE_KEY_FILENAME}"
        if not self._exists(key_location):
            logger.info("lens cache miss: %s (key %s)", self.root, key.sha256[:12])
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        key_dest = dest.with_name(LENS_CACHE_KEY_FILENAME)
        self._copy(key_location, str(key_dest))
        stored = json.loads(key_dest.read_text())
        if stored != key.as_payload():
            raise RuntimeError(
                f"lens cache entry {entry} carries a key that does not match the one it was "
                f"found under; stored {stored} vs wanted {key.as_payload()}. Refusing to serve it."
            )
        self._copy(f"{entry}/{LENS_CACHE_LENS_FILENAME}", str(dest))
        logger.info("lens cache hit: %s (key %s) -> %s", self.root, key.sha256[:12], dest)
        return True

    def store(self, key: LensCacheKey, lens_path: Path) -> None:
        """Publish a fitted lens under ``key``: the lens first, the key sidecar last."""
        entry = self.entry(key)
        key_file = lens_path.with_name(LENS_CACHE_KEY_FILENAME)
        key_file.write_text(json.dumps(key.as_payload(), indent=2, sort_keys=True) + "\n")
        self._copy(str(lens_path), f"{entry}/{LENS_CACHE_LENS_FILENAME}")
        self._copy(str(key_file), f"{entry}/{LENS_CACHE_KEY_FILENAME}")
        logger.info("lens cache stored: %s (key %s)", entry, key.sha256[:12])


@dataclass(frozen=True)
class LensAcquisition:
    """How a lens was obtained: fitted here or loaded from the cache, and what that cost."""

    lens: LensLike
    source: Literal["fit", "cache"]
    seconds: float
    key_sha256: str | None
    cache_root: str | None
    cache_stored: bool
    cache_store_error: str | None

    def as_payload(self) -> dict[str, object]:
        """Return the report block: everything but the lens object."""
        return {
            "source": self.source,
            "seconds": self.seconds,
            "key_sha256": self.key_sha256,
            "cache_root": self.cache_root,
            "cache_stored": self.cache_stored,
            "cache_store_error": self.cache_store_error,
        }


def acquire_lens(  # noqa: PLR0913 - a fit's inputs plus where it lands and how it is cached
    config: JacobianConfig,
    model: object,
    prompts: Sequence[str],
    jl: ModuleType,
    *,
    lens_path: Path,
    cache: LensCache | None = None,
    key: LensCacheKey | None = None,
) -> LensAcquisition:
    """Get a lens for ``key``: from ``cache`` when it holds one, else fit, save, verify and publish.

    On a hit the lens lands at ``lens_path`` and :func:`verify_cached_lens` checks its shape against
    ``model``; on a miss the fit is saved to ``lens_path``, round-tripped through
    :func:`verify_lens_roundtrip`, and stored. A store that fails with an S3 error -- a non-zero
    exit or an upload that hangs past the cache's timeout -- is logged and reported rather than
    raised: the lens is already on local disk and in the report, and losing a just-paid fit to a
    transient upload error would be the worse outcome. A lookup that fails for any reason but a
    missing object does raise, before any weights load, so a misconfigured cache costs seconds
    instead of a silent refit per cell.
    """
    if (cache is None) != (key is None):
        raise ValueError("acquire_lens needs both a cache and a key, or neither")
    started = time.time()
    if cache is not None and key is not None and cache.lookup(key, lens_path):
        lens = jl.JacobianLens.load(str(lens_path))
        verify_cached_lens(lens, model)
        return LensAcquisition(
            lens=lens,
            source="cache",
            seconds=round(time.time() - started, 2),
            key_sha256=key.sha256,
            cache_root=cache.root,
            cache_stored=False,
            cache_store_error=None,
        )
    lens = fit_lens(config, model, prompts, jl)
    lens_path.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(lens_path))  # pyright: ignore[reportAttributeAccessIssue]
    verify_lens_roundtrip(jl, lens, lens_path)
    stored = False
    store_error: str | None = None
    if cache is not None and key is not None:
        try:
            cache.store(key, lens_path)
            stored = True
        except S3CommandError as error:
            store_error = str(error)
            logger.exception(
                "lens cache store failed; the fitted lens is on local disk at %s and the run "
                "continues",
                lens_path,
            )
    return LensAcquisition(
        lens=lens,
        source="fit",
        seconds=round(time.time() - started, 2),
        key_sha256=None if key is None else key.sha256,
        cache_root=None if cache is None else cache.root,
        cache_stored=stored,
        cache_store_error=store_error,
    )


def load_lens(
    config: JacobianConfig,
    model: object,
    jl: ModuleType,
    *,
    fit_prompts: Sequence[str] | None = None,
) -> LensLike:
    """Dispatch on ``config.source``: reuse the pre-fit lens, or fit our own on ``fit_prompts``."""
    if config.source == "pretrained":
        return load_pretrained_lens(config, jl)
    if fit_prompts is None:
        raise ValueError("source='fit_own' needs fit_prompts (the reference fit iterates them)")
    return fit_lens(config, model, fit_prompts, jl)


def load_model_and_lens(
    config: JacobianConfig, *, fit_prompts: Sequence[str] | None = None
) -> tuple[object, AutoTokenizer, LensLike]:
    """GPU entry point: import jlens, load+wrap the model, and load or fit the lens."""
    jl = _require_jlens()
    model, tokenizer = _load_jlens_model(config, jl)
    lens = load_lens(config, model, jl, fit_prompts=fit_prompts)
    return model, tokenizer, lens


@dataclass(frozen=True)
class FitQualityReport:
    """A fitted lens's reconstruction quality, against the free logit-lens baseline it must beat.

    ``jacobian`` scores the fitted lens's readout ``unembed(J_l @ h_l)`` and ``logit_lens`` scores
    the vanilla logit lens (``jlens`` ``use_jacobian=False``: unembed the raw residual with no
    transport) on the SAME positions. The comparison is the point: a fitted lens whose
    reconstruction does not beat the logit lens has learned nothing the coordinate correction did
    not already give, so an absolute residual is only interpretable against this baseline.
    ``n_eval_prompts_skipped`` counts prompts too short to yield an interior read position, so a
    thin eval is visible rather than silently narrowing the denominator.
    """

    jacobian: ReconstructionReport
    logit_lens: ReconstructionReport
    n_eval_prompts_used: int
    n_eval_prompts_skipped: int
    positions_per_prompt_max: int


def evaluate_reconstruction(  # noqa: PLR0913 - lens, model, prompts and the read-window knobs
    lens: object,
    model: object,
    prompts: Sequence[str],
    *,
    max_seq_len: int,
    max_positions: int,
    skip_first: int = RECON_SKIP_FIRST,
) -> FitQualityReport | None:
    """Read the fitted lens's reconstruction of the model's own logits on ``prompts`` (GPU).

    For each prompt, read out at up to ``max_positions`` interior positions (matching the fit's
    excluded-sink convention) with the fitted Jacobian AND with the vanilla logit lens, and
    accumulate both against the model's actual final logits at those positions. One
    ``JacobianLens.apply`` per mode per prompt -- a handful of forward passes, negligible beside the
    fit -- so the numbers come off the model directly rather than from any retained activation.
    Returns ``None`` (a logged, non-fatal outcome, not a crash) when no prompt was long enough to
    read, so a 2-hour fit is never lost to a fit-quality read that could not run.
    """
    jacobian_predicted: dict[int, list[torch.Tensor]] = {}
    logit_lens_predicted: dict[int, list[torch.Tensor]] = {}
    actual_rows: list[torch.Tensor] = []
    used = 0
    skipped = 0
    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=max_seq_len)  # pyright: ignore[reportAttributeAccessIssue]
        seq_len = int(input_ids.shape[1])
        positions = interior_eval_positions(
            seq_len, skip_first=skip_first, max_positions=max_positions
        )
        if not positions:
            skipped += 1
            logger.warning(
                "reconstruction eval: skipping a %d-token prompt (need > %d valid positions)",
                seq_len,
                skip_first + 1,
            )
            continue
        jacobian_logits, model_logits, _ = lens.apply(  # pyright: ignore[reportAttributeAccessIssue]
            model, prompt, positions=positions, max_seq_len=max_seq_len, use_jacobian=True
        )
        logit_lens_logits, _, _ = lens.apply(  # pyright: ignore[reportAttributeAccessIssue]
            model, prompt, positions=positions, max_seq_len=max_seq_len, use_jacobian=False
        )
        for layer, logits in jacobian_logits.items():
            jacobian_predicted.setdefault(layer, []).append(logits.float().cpu())
        for layer, logits in logit_lens_logits.items():
            logit_lens_predicted.setdefault(layer, []).append(logits.float().cpu())
        actual_rows.append(model_logits.float().cpu())
        used += 1

    if not actual_rows:
        logger.warning(
            "reconstruction eval: no eval prompt was long enough to read; fit-quality unavailable"
        )
        return None
    actual = torch.cat(actual_rows, dim=0)
    jacobian = reconstruction_report(
        {layer: torch.cat(rows, dim=0) for layer, rows in jacobian_predicted.items()}, actual
    )
    logit_lens = reconstruction_report(
        {layer: torch.cat(rows, dim=0) for layer, rows in logit_lens_predicted.items()}, actual
    )
    logger.info(
        "reconstruction eval: %d prompts used (%d skipped), median relative residual "
        "jacobian=%.4f vs logit-lens=%.4f",
        used,
        skipped,
        jacobian.median_relative_residual,
        logit_lens.median_relative_residual,
    )
    return FitQualityReport(
        jacobian=jacobian,
        logit_lens=logit_lens,
        n_eval_prompts_used=used,
        n_eval_prompts_skipped=skipped,
        positions_per_prompt_max=max_positions,
    )


def fit_quality_payload(report: FitQualityReport | None) -> dict[str, object]:
    """Build the uploadable fit-quality block for ``fit_report.json`` -- aggregate numbers only.

    A fit-quality read that could not run (no eval prompt long enough) reports ``available: false``
    with the reason rather than being silently absent -- an omitted field reads as "not measured"
    and "measured as fine" identically, which is the failure mode this whole change exists to fix.
    When it did run, the Jacobian lens's reconstruction sits beside the vanilla logit-lens baseline
    and the median-residual reduction between them, so the number is judged against the free
    baseline it must beat rather than read as an uninterpretable absolute.
    """
    if report is None:
        return {
            "available": False,
            "reason": "no eval prompt was long enough to read an interior position",
        }
    reduction = (
        report.logit_lens.median_relative_residual - report.jacobian.median_relative_residual
    )
    return {
        "available": True,
        "n_eval_prompts_used": report.n_eval_prompts_used,
        "n_eval_prompts_skipped": report.n_eval_prompts_skipped,
        "positions_per_prompt_max": report.positions_per_prompt_max,
        "n_samples": report.jacobian.n_samples,
        "jacobian": asdict(report.jacobian),
        "logit_lens_baseline": asdict(report.logit_lens),
        "jacobian_beats_logit_lens": (
            report.jacobian.median_relative_residual < report.logit_lens.median_relative_residual
        ),
        "median_residual_reduction_vs_logit_lens": reduction,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _read_prompts(path: Path | None) -> list[str] | None:
    """Read one prompt per non-blank line, or ``None`` when no path is given."""
    if path is None:
        return None
    return [line for line in path.read_text().splitlines() if line.strip()]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments for decoding a saved residual-space direction through a Jacobian lens."""
    parser = argparse.ArgumentParser(
        description="Transport and decode a residual-space direction through a Jacobian lens"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--source", choices=("fit_own", "pretrained"), default="fit_own")
    parser.add_argument(
        "--direction-path",
        type=Path,
        required=True,
        help="a saved {layer: tensor} directions.pt (e.g. eval_awareness_probe's output)",
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="which layer's direction to decode"
    )
    parser.add_argument("--top-k", type=int, default=JacobianConfig.top_k)
    parser.add_argument(
        "--fit-prompts-path",
        type=Path,
        default=None,
        help="one prompt per line; required for --source fit_own (agentic transcripts recommended)",
    )
    parser.add_argument("--max-fit-prompts", type=int, default=JacobianConfig.max_fit_prompts)
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=JacobianConfig.dim_batch,
        help="residual dims per backward pass -- a memory knob only: total backward FLOPs are "
        "unchanged, measured fits gain little past 16, and 64 OOM'd a 44 GiB and a 96 GiB card",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=JacobianConfig.max_seq_len,
        help="truncate each fit prompt to this many tokens",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--pretrained-repo", default=DEFAULT_PRETRAINED_REPO)
    parser.add_argument("--pretrained-filename", default=DEFAULT_PRETRAINED_FILENAME)
    parser.add_argument("--pretrained-revision", default=DEFAULT_PRETRAINED_REVISION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load/fit a lens and print the top-k token readout of a saved direction at a layer (GPU)."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = JacobianConfig(
        model_id=args.model_id,
        source=args.source,
        top_k=args.top_k,
        max_fit_prompts=args.max_fit_prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint_path,
        pretrained_repo=args.pretrained_repo,
        pretrained_filename=args.pretrained_filename,
        pretrained_revision=args.pretrained_revision,
    )
    directions: dict[int, torch.Tensor] = torch.load(args.direction_path, weights_only=True)
    if args.layer not in directions:
        raise ValueError(f"layer {args.layer} not in {sorted(directions)} at {args.direction_path}")
    if args.layer == max(directions):
        raise ValueError(
            f"layer {args.layer} is the top layer in {args.direction_path} and has no fitted "
            "Jacobian: a lens fits source layers strictly below its target layer, and jlens "
            "defaults the target to the final layer. Decode a lower layer. Checked here rather "
            "than at transport time, which is after the model load and the whole fit."
        )

    model, tokenizer, lens = load_model_and_lens(
        config, fit_prompts=_read_prompts(args.fit_prompts_path)
    )

    def decode_id(token_id: int) -> str:
        return tokenizer.convert_ids_to_tokens(token_id)  # pyright: ignore[reportAttributeAccessIssue]

    readouts = transport_and_decode(
        lens,
        model,  # pyright: ignore[reportArgumentType]  # jlens.from_hf model exposes .unembed
        directions[args.layer],
        args.layer,
        id_to_token=decode_id,
        k=config.top_k,
    )
    for readout in readouts:
        print(f"{readout.logit:>8.3f}  {readout.token}")  # noqa: T201  # Intentional CLI output.


if __name__ == "__main__":
    main()
