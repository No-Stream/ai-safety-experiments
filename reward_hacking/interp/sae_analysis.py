"""Run a pretrained JumpReLU sparse autoencoder over the OFF-THE-SHELF Qwen3.5-4B.

Two baseline reads on the UNTRAINED model (this is apparatus on the base checkpoint, not
RL-induced evidence):

* **Differential feature firing** -- feed the rigged-grader vs matched-honest twin prompts (the
  conflicting/original ILCB stimulus pairs the contrast stage uses) through the model, capture the
  residual stream at the SAE's hook layer, encode every prompt position into the SAE's 65,536
  JumpReLU features, and pool per prompt. Rank the features that fire MORE on the rigged twin than
  its matched honest twin, paired by problem. A paired sign-flip PERMUTATION null (shuffle the
  rigged/matched label within each pair, recompute the per-feature mean difference) gives both a
  per-feature two-sided p-value and a max-statistic family-wise threshold, so "feature X fires more
  on rigged" has to beat chance under multiple comparisons before it counts.

* **Concept-direction alignment** -- for each validated concept direction (eval_awareness, shortcut,
  deception, contradiction) at the SAE's layer, cosine-align the direction with every SAE feature's
  decoder vector and report the top-aligned features. A matched random-direction control records the
  best cosine a random unit direction reaches against the same decoder set, so a real concept
  alignment can be told from "any direction lands this close to some feature".

The SAE weights load DIRECTLY from safetensors (``decoderesearch/qwen-3.5-saes``); ``sae_lens`` is
deliberately NOT a dependency (it drags wandb/plotly/nltk into this lean repo). The JumpReLU encode
below is the exact one from sae_lens' source under this SAE family's config (``apply_b_dec_to_input``
true, ``normalize_activations`` none) and is unit-tested against a hand-computed example.

Nothing here loads a model or reads the stimulus corpus at import time: the pure-math cores
(``JumpReluSae.encode``, :func:`differential_firing`, :func:`concept_alignment`) run on synthetic
tensors in the offline tests, and the model-capture / stimulus paths are behind the CLI functions.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from reward_hacking.interp.directions import (
    capture_positionwise_activations,
    load_model_and_tokenizer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_SAE_REPO = "decoderesearch/qwen-3.5-saes"
DEFAULT_SAE_ID = "qwen-3.5-4b/btk-mat-layer-15-k-100"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
# Underscore concept keys; the axes-directory slugs are the hyphenated form (``eval-awareness``).
DEFAULT_CONCEPTS: tuple[str, ...] = ("eval_awareness", "shortcut", "deception", "contradiction")

_NORM_EPS = 1e-12
_EXPECTED_NDIM = 2  # activation matrices are [n_pairs, d_sae]; the decoder is [d_sae, d_in]
_MIN_PAIRS = 2  # a paired sign-flip permutation null needs at least two pairs
_POSITION_CHUNK = 512  # positions encoded per SAE forward, so memory does not scale with seq len


# ------------------------------------------------------------------------------------------------
# SAE loader + encode (the pure-math core, unit-tested)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JumpReluSae:
    """A JumpReLU sparse autoencoder loaded straight from safetensors.

    Tensors follow the ``decoderesearch/qwen-3.5-saes`` layout (all float32): ``W_enc`` is
    ``[d_in, d_sae]``, ``W_dec`` is ``[d_sae, d_in]``, ``b_enc``/``threshold`` are ``[d_sae]`` and
    ``b_dec`` is ``[d_in]``. ``layer`` is the decoder-layer index parsed from the config hook name;
    it is the residual-stream layer this SAE reads and the layer the model must be hooked at.
    """

    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    threshold: torch.Tensor
    layer: int
    d_in: int
    d_sae: int
    sae_id: str
    hook_name: str

    def to(self, device: torch.device) -> JumpReluSae:
        """Move every weight tensor to ``device`` (returns a new instance; tensors are shared)."""
        return JumpReluSae(
            W_enc=self.W_enc.to(device),
            W_dec=self.W_dec.to(device),
            b_enc=self.b_enc.to(device),
            b_dec=self.b_dec.to(device),
            threshold=self.threshold.to(device),
            layer=self.layer,
            d_in=self.d_in,
            d_sae=self.d_sae,
            sae_id=self.sae_id,
            hook_name=self.hook_name,
        )

    @property
    def device(self) -> torch.device:
        """The device the SAE weight tensors currently live on."""
        return self.W_enc.device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode residual activations ``[..., d_in]`` to feature activations ``[..., d_sae]``.

        The exact JumpReLU forward for this SAE family (apply_b_dec_to_input=True, normalize=none):
        subtract the decoder bias, project, add the encoder bias, then gate by ReLU AND the
        per-feature threshold. Computed in float32 (the SAE weights are float32) regardless of the
        residual's dtype.
        """
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"encode expected last dim {self.d_in} (d_in of {self.sae_id}), got {x.shape[-1]}"
            )
        sae_in = x.to(self.W_enc.dtype) - self.b_dec
        pre = sae_in @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)


def parse_layer_from_hook_name(hook_name: str) -> int:
    """``model.layers.15`` -> ``15``. Fail loudly on anything that is not a decoder-layer hook.

    The whole analysis is keyed on this layer: the model is hooked here, and the concept direction
    is read here. A hook name whose trailing component is not an integer means the SAE reads
    something other than a numbered decoder layer, which this apparatus does not handle.
    """
    tail = hook_name.rsplit(".", 1)[-1]
    if not tail.isdigit():
        raise ValueError(
            f"cannot parse a decoder-layer index from hook_name {hook_name!r}: "
            f"trailing component {tail!r} is not an integer"
        )
    return int(tail)


def load_sae(sae_repo: str, sae_id: str, *, device: torch.device | None = None) -> JumpReluSae:
    """Download ``cfg.json`` + ``sae_weights.safetensors`` for ``sae_id`` and build a JumpReluSae.

    Asserts the config matches the encode this module implements: JumpReLU architecture,
    ``apply_b_dec_to_input`` true, ``normalize_activations`` none. A future SAE that differs on any
    of these would silently be encoded by the wrong forward, so it fails here instead.
    """
    cfg_path = hf_hub_download(sae_repo, f"{sae_id}/cfg.json")
    weights_path = hf_hub_download(sae_repo, f"{sae_id}/sae_weights.safetensors")
    with Path(cfg_path).open() as handle:
        cfg: dict[str, object] = json.load(handle)

    architecture = cfg.get("architecture")
    if architecture != "jumprelu":
        raise ValueError(
            f"{sae_id}: architecture is {architecture!r}, not 'jumprelu'; the encode in this "
            "module is JumpReLU-specific"
        )
    if cfg.get("apply_b_dec_to_input") is not True:
        raise ValueError(
            f"{sae_id}: apply_b_dec_to_input is {cfg.get('apply_b_dec_to_input')!r}, not True; "
            "the encode here subtracts b_dec from the input unconditionally"
        )
    if cfg.get("normalize_activations") != "none":
        raise ValueError(
            f"{sae_id}: normalize_activations is {cfg.get('normalize_activations')!r}, not 'none'; "
            "the encode here applies no activation normalisation"
        )

    metadata = cfg.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError(f"{sae_id}: cfg.json has no 'metadata' block to read hook_name from")
    hook_name = metadata.get("hook_name")
    if not isinstance(hook_name, str):
        raise TypeError(f"{sae_id}: metadata.hook_name is missing or not a string")
    layer = parse_layer_from_hook_name(hook_name)
    d_in = int(cfg["d_in"])  # type: ignore[arg-type]
    d_sae = int(cfg["d_sae"])  # type: ignore[arg-type]

    weights = load_file(weights_path)
    sae = JumpReluSae(
        W_enc=weights["W_enc"].float(),
        W_dec=weights["W_dec"].float(),
        b_enc=weights["b_enc"].float(),
        b_dec=weights["b_dec"].float(),
        threshold=weights["threshold"].float(),
        layer=layer,
        d_in=d_in,
        d_sae=d_sae,
        sae_id=sae_id,
        hook_name=hook_name,
    )
    _validate_sae_shapes(sae)
    logger.info(
        "loaded SAE %s: layer=%d d_in=%d d_sae=%d hook_name=%s",
        sae_id,
        layer,
        d_in,
        d_sae,
        hook_name,
    )
    if device is not None:
        sae = sae.to(device)
    return sae


def _validate_sae_shapes(sae: JumpReluSae) -> None:
    """Fail loudly if any weight tensor is the wrong shape for the declared d_in / d_sae."""
    expected: dict[str, tuple[int, ...]] = {
        "W_enc": (sae.d_in, sae.d_sae),
        "W_dec": (sae.d_sae, sae.d_in),
        "b_enc": (sae.d_sae,),
        "b_dec": (sae.d_in,),
        "threshold": (sae.d_sae,),
    }
    actual: dict[str, tuple[int, ...]] = {
        "W_enc": tuple(sae.W_enc.shape),
        "W_dec": tuple(sae.W_dec.shape),
        "b_enc": tuple(sae.b_enc.shape),
        "b_dec": tuple(sae.b_dec.shape),
        "threshold": tuple(sae.threshold.shape),
    }
    mismatches = {
        name: (actual[name], want) for name, want in expected.items() if actual[name] != want
    }
    if mismatches:
        raise ValueError(f"{sae.sae_id}: SAE weight shapes disagree with the config: {mismatches}")


# ------------------------------------------------------------------------------------------------
# Result records (typed so callers and tests index them by attribute, not stringly)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDelta:
    """One feature's rigged-minus-matched firing with its permutation significance."""

    feature: int
    mean_delta: float
    mean_rigged: float
    mean_matched: float
    p_two_sided: float
    significant_fwer: bool


@dataclass(frozen=True)
class DifferentialResult:
    """The differential-feature-firing read: ranked features plus the permutation-null summary."""

    n_pairs: int
    d_sae: int
    n_permutations: int
    seed: int
    fwer_alpha: float
    fwer_threshold: float
    n_features_significant_fwer: int
    max_abs_observed_delta: float
    top_rigged_gt_matched: list[FeatureDelta]
    top_matched_gt_rigged: list[FeatureDelta]


@dataclass(frozen=True)
class FeatureCosine:
    """One SAE feature's cosine with a concept direction."""

    feature: int
    cosine: float


@dataclass(frozen=True)
class ConceptAlignmentRead:
    """Per-concept cosine alignment against the SAE decoder vectors."""

    top_aligned: list[FeatureCosine]
    top_anti_aligned: list[FeatureCosine]
    max_abs_cosine: float


@dataclass(frozen=True)
class PlaceboControl:
    """Best |cosine| a matched-count set of random unit directions reaches against the decoder."""

    n_placebos: int
    random_direction_best_abs_cosine_mean: float | None
    random_direction_best_abs_cosine_max: float | None


@dataclass(frozen=True)
class AlignmentResult:
    """The concept-direction alignment read: per-concept feature cosines plus the random control."""

    placebo_control: PlaceboControl
    per_concept: dict[str, ConceptAlignmentRead]


# ------------------------------------------------------------------------------------------------
# Analysis (a): differential feature firing (pure-math core)
# ------------------------------------------------------------------------------------------------


def differential_firing(
    rigged: torch.Tensor,
    matched: torch.Tensor,
    *,
    n_permutations: int,
    seed: int,
    top_k: int,
) -> DifferentialResult:
    """Rank SAE features by paired rigged-minus-matched firing, with a sign-flip permutation null.

    ``rigged`` and ``matched`` are ``[n_pairs, d_sae]`` per-prompt pooled feature activations, row i
    of each being the two twins of problem i. The observed statistic per feature is the mean over
    pairs of ``rigged - matched``. The null flips each pair's sign independently (the exact-exchange
    null for paired data), recomputing the per-feature mean; from ``n_permutations`` draws it yields
    a per-feature two-sided p-value and a max-statistic family-wise threshold (95th percentile of the
    per-permutation max |mean difference|), so a feature clears multiple comparisons only if its
    observed effect exceeds what the largest of 65k null features reaches by chance.
    """
    if rigged.shape != matched.shape:
        raise ValueError(f"rigged {tuple(rigged.shape)} and matched {tuple(matched.shape)} differ")
    if rigged.ndim != _EXPECTED_NDIM:
        raise ValueError(f"expected [n_pairs, d_sae] activations, got {tuple(rigged.shape)}")
    n_pairs, d_sae = rigged.shape
    if n_pairs < _MIN_PAIRS:
        raise ValueError(f"need at least {_MIN_PAIRS} pairs for a permutation null, got {n_pairs}")

    diffs = (rigged - matched).float()
    if not torch.isfinite(diffs).all():
        raise ValueError("non-finite values in the rigged-minus-matched feature differences")
    device = diffs.device
    observed = diffs.mean(dim=0)  # [d_sae]
    abs_obs = observed.abs()

    generator = torch.Generator(device=device).manual_seed(seed)
    signs = (
        torch.randint(0, 2, (n_permutations, n_pairs), generator=generator, device=device).float()
        * 2.0
        - 1.0
    )  # [P, n_pairs] in {-1, +1}
    null = (signs @ diffs) / n_pairs  # [P, d_sae]
    ge = (null.abs() >= abs_obs.unsqueeze(0)).sum(dim=0)  # [d_sae]
    p_two_sided = (ge.float() + 1.0) / (n_permutations + 1)
    null_max = null.abs().max(dim=1).values  # [P]
    fwer_threshold = float(torch.quantile(null_max, 0.95).item())
    n_significant = int((abs_obs > fwer_threshold).sum().item())

    mean_rigged = rigged.float().mean(dim=0)
    mean_matched = matched.float().mean(dim=0)

    def _rows(order: torch.Tensor) -> list[FeatureDelta]:
        features = order.tolist()
        deltas = observed[order].tolist()
        rigged_means = mean_rigged[order].tolist()
        matched_means = mean_matched[order].tolist()
        p_values = p_two_sided[order].tolist()
        significant = (abs_obs[order] > fwer_threshold).tolist()
        return [
            FeatureDelta(
                feature=int(feature),
                mean_delta=float(delta),
                mean_rigged=float(rigged_mean),
                mean_matched=float(matched_mean),
                p_two_sided=float(p_value),
                significant_fwer=bool(sig),
            )
            for feature, delta, rigged_mean, matched_mean, p_value, sig in zip(
                features, deltas, rigged_means, matched_means, p_values, significant, strict=True
            )
        ]

    top_rigged = torch.topk(observed, min(top_k, d_sae)).indices
    top_matched = torch.topk(-observed, min(top_k, d_sae)).indices
    return DifferentialResult(
        n_pairs=int(n_pairs),
        d_sae=int(d_sae),
        n_permutations=int(n_permutations),
        seed=int(seed),
        fwer_alpha=0.05,
        fwer_threshold=fwer_threshold,
        n_features_significant_fwer=n_significant,
        max_abs_observed_delta=float(abs_obs.max().item()),
        top_rigged_gt_matched=_rows(top_rigged),
        top_matched_gt_rigged=_rows(top_matched),
    )


# ------------------------------------------------------------------------------------------------
# Analysis (b): concept-direction alignment (pure-math core)
# ------------------------------------------------------------------------------------------------


def concept_alignment(
    w_dec: torch.Tensor,
    directions: Mapping[str, torch.Tensor],
    *,
    top_k: int,
    n_placebos: int,
    seed: int,
) -> AlignmentResult:
    """Cosine-align each concept direction with every SAE decoder vector; report top features.

    ``w_dec`` is the SAE decoder matrix ``[d_sae, d_in]``; ``directions`` maps a concept name to its
    ``[d_in]`` direction at the SAE's layer. For each concept the cosine against every
    unit-normalised decoder row is ranked, top and anti (most negative) reported. A matched control
    draws ``n_placebos`` random unit directions and records the best |cosine| each reaches against
    the same decoder set, so the concept's top cosine is read against what a random direction
    achieves rather than against zero.
    """
    if w_dec.ndim != _EXPECTED_NDIM:
        raise ValueError(f"w_dec must be [d_sae, d_in], got {tuple(w_dec.shape)}")
    d_sae, d_in = w_dec.shape
    device = w_dec.device
    decoder_unit = w_dec.float() / w_dec.float().norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)
    generator = torch.Generator(device=device).manual_seed(seed)

    placebo_best: list[float] = []
    if n_placebos > 0:
        rand = torch.randn(n_placebos, d_in, generator=generator, device=device)
        rand_unit = rand / rand.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)
        # [d_sae, n_placebos] cosines; best |cosine| any feature reaches per random direction.
        placebo_best = (decoder_unit @ rand_unit.T).abs().max(dim=0).values.tolist()
    placebo = PlaceboControl(
        n_placebos=int(n_placebos),
        random_direction_best_abs_cosine_mean=(
            float(torch.tensor(placebo_best).mean().item()) if placebo_best else None
        ),
        random_direction_best_abs_cosine_max=(max(placebo_best) if placebo_best else None),
    )

    per_concept: dict[str, ConceptAlignmentRead] = {}
    for concept, direction in directions.items():
        vec = direction.to(device).float()
        if vec.shape != (d_in,):
            raise ValueError(
                f"concept {concept!r} direction is {tuple(vec.shape)}, expected ({d_in},) to match "
                f"the SAE's d_in"
            )
        vec_unit = vec / vec.norm().clamp_min(_NORM_EPS)
        cosine = decoder_unit @ vec_unit  # [d_sae]
        if not torch.isfinite(cosine).all():
            raise ValueError(f"non-finite cosine values for concept {concept!r}")
        top = torch.topk(cosine, min(top_k, d_sae))
        anti = torch.topk(-cosine, min(top_k, d_sae))
        top_features = top.indices.tolist()
        top_cosines = top.values.tolist()
        anti_features = anti.indices.tolist()
        anti_cosines = anti.values.tolist()
        # topk is sorted, so the most-positive and most-negative cosines lead each list.
        max_abs_cosine = max(abs(top_cosines[0]), abs(anti_cosines[0]))
        per_concept[concept] = ConceptAlignmentRead(
            top_aligned=[
                FeatureCosine(feature=int(f), cosine=float(c))
                for f, c in zip(top_features, top_cosines, strict=True)
            ],
            top_anti_aligned=[
                FeatureCosine(feature=int(f), cosine=float(-c))
                for f, c in zip(anti_features, anti_cosines, strict=True)
            ],
            max_abs_cosine=float(max_abs_cosine),
        )
    return AlignmentResult(placebo_control=placebo, per_concept=per_concept)


def load_concept_directions(
    axes_dir: Path, concepts: Sequence[str], *, pooling: str, layer: int
) -> dict[str, torch.Tensor]:
    """Load ``<axes_dir>/<slug>-<pooling>/directions.pt`` for each concept and take the SAE layer.

    Each ``directions.pt`` is a ``{layer: tensor}`` dict; the SAE's layer is indexed out. A missing
    directory, or a directory whose dict lacks the SAE layer, fails loudly with the layers it does
    carry rather than silently dropping a concept.
    """
    out: dict[str, torch.Tensor] = {}
    for concept in concepts:
        slug = concept.replace("_", "-")
        path = axes_dir / f"{slug}-{pooling}" / "directions.pt"
        if not path.is_file():
            raise FileNotFoundError(f"no directions for concept {concept!r} at {path}")
        by_layer: dict[int, torch.Tensor] = torch.load(path, weights_only=True)
        if layer not in by_layer:
            raise KeyError(
                f"concept {concept!r} directions at {path} have no layer {layer}; "
                f"available layers: {sorted(by_layer)}"
            )
        out[concept] = by_layer[layer]
    return out


# ------------------------------------------------------------------------------------------------
# Model-capture orchestration (behind the CLI; not import-time, not unit-tested offline)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureContext:
    """The loaded model, its tokenizer and the SAE, bundled so they travel together."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    sae: JumpReluSae


def _format_prompt(tokenizer: AutoTokenizer, transcript: str, *, thinking: bool) -> str:
    """Wrap a transcript as a single user turn and apply the chat template.

    Mirrors the harness convention (``prompt_contrast`` / ``generation_capture`` format the same
    conflicting/original transcripts this way), so the activations are comparable to the contrast
    and steer-patch reads on the identical substrate.
    """
    return tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
        [{"role": "user", "content": transcript}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def capture_feature_vector(
    ctx: CaptureContext,
    transcript: str,
    *,
    thinking: bool,
    position_pooling: str,
) -> torch.Tensor:
    """Forward one transcript, encode its residuals at the SAE layer, pool over positions -> [d_sae].

    Batch of one (no padding), trunk-only forward via ``capture_positionwise_activations``. Positions
    are encoded in chunks so peak memory does not scale with sequence length. ``position_pooling``:
    ``mean`` / ``max`` over all prompt positions, or ``last`` (final position only).
    """
    if position_pooling not in {"mean", "max", "last"}:
        raise ValueError(f"unknown position_pooling {position_pooling!r}")
    sae = ctx.sae
    chat = _format_prompt(ctx.tokenizer, transcript, thinking=thinking)
    encoded = ctx.tokenizer(chat, return_tensors="pt")  # pyright: ignore[reportCallIssue]
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    acts = capture_positionwise_activations(
        ctx.model, input_ids, attention_mask, layers=[sae.layer]
    )[sae.layer]
    residual = acts.squeeze(0).to(sae.device)  # [seq, d_in]

    if position_pooling == "last":
        return sae.encode(residual[-1:]).squeeze(0).cpu()

    pooled: torch.Tensor | None = None
    count = 0
    for start in range(0, residual.shape[0], _POSITION_CHUNK):
        feats = sae.encode(residual[start : start + _POSITION_CHUNK])  # [chunk, d_sae]
        if position_pooling == "mean":
            chunk_sum = feats.sum(dim=0)
            pooled = chunk_sum if pooled is None else pooled + chunk_sum
            count += feats.shape[0]
        else:  # max
            chunk_max = feats.max(dim=0).values
            pooled = chunk_max if pooled is None else torch.maximum(pooled, chunk_max)
    if pooled is None:
        raise ValueError("no positions captured for a transcript")
    if position_pooling == "mean":
        pooled = pooled / count
    return pooled.cpu()


def capture_pair_feature_vectors(
    ctx: CaptureContext,
    episode_dir: Path,
    *,
    limit: int | None,
    thinking: bool,
    position_pooling: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Materialise the twin stimulus pairs and return (rigged, matched, problem_ids).

    ``rigged``/``matched`` are ``[n_pairs, d_sae]`` pooled feature activations, row i the two twins
    of ``problem_ids[i]``. ``build_stimulus_pairs`` is imported lazily: it pulls in the harness and
    reads the ILCB case data, which this module must not require at import.
    """
    from reward_hacking.interp.prompt_contrast import build_stimulus_pairs  # noqa: PLC0415

    pairs = build_stimulus_pairs(episode_dir, limit=limit)
    if not pairs:
        raise ValueError(f"build_stimulus_pairs produced no pairs from {episode_dir}")
    logger.info("captured stimulus pairs: %d", len(pairs))
    rigged_rows: list[torch.Tensor] = []
    matched_rows: list[torch.Tensor] = []
    problem_ids: list[str] = []
    for i, pair in enumerate(pairs):
        rigged_rows.append(
            capture_feature_vector(
                ctx,
                pair.conflicting_transcript,
                thinking=thinking,
                position_pooling=position_pooling,
            )
        )
        matched_rows.append(
            capture_feature_vector(
                ctx, pair.original_transcript, thinking=thinking, position_pooling=position_pooling
            )
        )
        problem_ids.append(pair.problem_id)
        logger.info("captured pair %d/%d (%s)", i + 1, len(pairs), pair.problem_id)
    rigged = torch.stack(rigged_rows, dim=0)
    matched = torch.stack(matched_rows, dim=0)
    if not torch.isfinite(rigged).all() or not torch.isfinite(matched).all():
        raise ValueError("non-finite SAE feature activations captured; check the model load")
    return rigged, matched, problem_ids


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a pretrained JumpReLU SAE over the off-the-shelf Qwen3.5-4B: differential "
        "feature firing on rigged-vs-matched twins, and concept-direction alignment."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sae-repo", default=DEFAULT_SAE_REPO)
    parser.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    parser.add_argument(
        "--episode-dir",
        type=Path,
        required=True,
        help="disposable scratch dir for build_stimulus_pairs; MUST be outside any home tree",
    )
    parser.add_argument("--out", type=Path, required=True, help="results JSON path")
    parser.add_argument(
        "--axes-dir",
        type=Path,
        required=True,
        help="dir holding <concept>-<pooling>/directions.pt for the concept-alignment read",
    )
    parser.add_argument("--concepts", nargs="+", default=list(DEFAULT_CONCEPTS))
    parser.add_argument(
        "--pooling", default="mean", choices=["mean", "last"], help="concept-direction pooling"
    )
    parser.add_argument(
        "--position-pooling",
        default="mean",
        choices=["mean", "max", "last"],
        help="how per-token SAE feature activations are pooled to one vector per prompt",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of stimulus pairs (default: all)"
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-placebos", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the chat template in thinking mode (matches the contrast substrate)",
    )
    parser.add_argument("--device", default="auto")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Load the model + SAE, capture, run both analyses, write and return the results dict."""
    device = _resolve_device(args.device)
    logger.info("device=%s", device)
    sae = load_sae(args.sae_repo, args.sae_id, device=device)
    model, tokenizer = load_model_and_tokenizer(
        args.model_id, device=None if device.type == "cuda" else "cpu"
    )

    directions = load_concept_directions(
        args.axes_dir, args.concepts, pooling=args.pooling, layer=sae.layer
    )

    ctx = CaptureContext(model=model, tokenizer=tokenizer, sae=sae)
    rigged, matched, problem_ids = capture_pair_feature_vectors(
        ctx,
        args.episode_dir,
        limit=args.limit,
        thinking=args.thinking,
        position_pooling=args.position_pooling,
    )

    differential = differential_firing(
        rigged.to(device),
        matched.to(device),
        n_permutations=args.n_permutations,
        seed=args.seed,
        top_k=args.top_k,
    )
    alignment = concept_alignment(
        sae.W_dec,
        directions,
        top_k=args.top_k,
        n_placebos=args.n_placebos,
        seed=args.seed,
    )

    results: dict[str, object] = {
        "model_id": args.model_id,
        "sae_repo": args.sae_repo,
        "sae_id": args.sae_id,
        "sae_layer": sae.layer,
        "d_in": sae.d_in,
        "d_sae": sae.d_sae,
        "n_pairs": len(problem_ids),
        "problem_ids": problem_ids,
        "thinking": bool(args.thinking),
        "position_pooling": args.position_pooling,
        "concept_pooling": args.pooling,
        "concepts": list(args.concepts),
        "differential_feature_firing": asdict(differential),
        "concept_direction_alignment": asdict(alignment),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True))
    logger.info(
        "wrote %s: %d pairs, %d features significant at FWER %.2f, differential threshold %.5f",
        args.out,
        len(problem_ids),
        differential.n_features_significant_fwer,
        differential.fwer_alpha,
        differential.fwer_threshold,
    )
    return results


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: parse args and run both SAE analyses."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
