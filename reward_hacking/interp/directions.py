"""Diff-of-means concept directions and per-layer cosine similarity, with a random placebo.

Two halves live here:

* A **pure-tensor core** -- pooling, diff-of-means, cosine, a matched-norm random direction, and
  the per-layer comparison assembly. It touches no model and is what the offline tests exercise on
  synthetic activations with planted directions.
* A **model-capture path** guarded behind functions the CLI calls. It reads the residual stream
  via ``register_forward_hook`` on each decoder layer's output (NOT ``hook_q`` / per-head query
  space, which on Qwen3.5's gated attention is double-width and interleaved -- see
  ``docs/scratch/interp-tooling-verified.md``), pools over tokens, and returns per-layer pooled
  activations ready for the core.

``--backend`` exists on this CLI only in order to refuse. It offers the same kinds as every other
CLI in this package so that aiming the probe at a hosted model answers with an explanation instead
of an obscure failure, but only ``--backend hf`` can run: Converse returns text, and the quantity
measured here is a residual stream. There is no degraded hosted version of this measurement, so the
refusal is the honest answer rather than a gap waiting to be filled.

The measurement: for each layer, ``reward_hack_dir = mean(hack_positives) - mean(hack_negatives)``
and likewise for deception, then ``cosine(reward_hack_dir, deception_dir)`` per layer. A
matched-norm random Gaussian "direction" gives the chance baseline -- if the hack/deception cosine
is not clearly above the placebo cosine, the two concepts are not geometrically distinct at that
depth. Reported per layer because the informative band is narrow and depth-dependent; collapsing
to one number throws that away.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward_hacking import backend_cli
from reward_hacking.interp import stimuli

logger = logging.getLogger(__name__)

Pooler = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------------------
# Pure-tensor core: pooling
# --------------------------------------------------------------------------------------


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-pad tokens. ``hidden`` [batch, seq, d], ``mask`` [batch, seq] -> [batch, d].

    The pooling ARITHMETIC is padding-side agnostic: the mask alone decides which positions count.
    That says nothing about the capture that produced ``hidden`` -- see
    :func:`capture_pooled_activations`, which requires right padding because it passes no
    ``position_ids``.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Representation at the last non-pad token. [batch, seq, d], [batch, seq] -> [batch, d].

    Also agnostic as arithmetic: the last real index is found from the mask, so a right-padded
    ``[1,1,1,0,0]`` and a left-padded ``[0,0,1,1,1]`` both resolve to their true final token. The
    capture upstream is not (see :func:`capture_pooled_activations`).
    """
    last_idx = attention_mask.shape[1] - 1 - attention_mask.flip(1).argmax(dim=1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, last_idx]


POOLERS: dict[str, Pooler] = {"mean": mean_pool, "last": last_token_pool}


# --------------------------------------------------------------------------------------
# Pure-tensor core: directions and cosine
# --------------------------------------------------------------------------------------


STD_EPS = 1e-6


def diff_of_means(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Concept direction: ``mean(positive) - mean(negative)``. Inputs [n_pos, d], [n_neg, d]."""
    return positive.mean(dim=0) - negative.mean(dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two 1-D vectors, as a plain float."""
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def unit(direction: torch.Tensor) -> torch.Tensor:
    """Return ``direction`` scaled to unit L2 norm; raise on a zero-norm input.

    The shared normaliser for every projection and every residual intervention: a projection onto
    the unit axis is a component in raw activation units, and a steering vector
    ``alpha * unit(dir)`` has magnitude ``alpha`` regardless of how the direction was scaled, so the
    matched-norm placebo and the real axis are perturbed by exactly the same amount.
    """
    norm = direction.norm()
    if norm == 0:
        raise ValueError("cannot normalise a zero-norm direction")
    return direction / norm


def matched_norm_random_direction(
    reference: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Return a Gaussian direction scaled to ``reference``'s L2 norm -- the placebo control.

    Cosine ignores magnitude, so norm-matching is not what makes the cosine a fair baseline (any
    random direction has ~0 cosine with a fixed vector, std ~ 1/sqrt(d)). It matters because the
    same placebo is the drop-in control if this ever grows into a steer/ablate experiment: the
    mandatory matched-norm placebo distinguishes a real effect from "any perturbation this large".
    """
    raw = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    return raw / raw.norm() * reference.norm()


def concept_directions(
    positives_by_layer: dict[int, torch.Tensor],
    negatives_by_layer: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Per-layer diff-of-means. Both inputs map layer -> pooled activations [n, d]."""
    if positives_by_layer.keys() != negatives_by_layer.keys():
        raise ValueError(
            "positive and negative activations cover different layers: "
            f"{sorted(positives_by_layer)} vs {sorted(negatives_by_layer)}"
        )
    return {
        layer: diff_of_means(positives_by_layer[layer], negatives_by_layer[layer])
        for layer in sorted(positives_by_layer)
    }


def standardized_directions(
    hack_positive: torch.Tensor,
    hack_negative: torch.Tensor,
    deception_positive: torch.Tensor,
    deception_negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(hack, deception) diff-of-means after z-scoring each dim by the combined-set per-dim std.

    A handful of massive-activation residual dimensions can carry nearly all of a direction's norm
    and pin the raw hack/deception cosine near 1 regardless of concept content -- seen on
    Phi-4-mini (raw ~0.99, standardized ~0.1-0.5, with 5 dims holding ~99% of each direction's
    norm), while Qwen3.5-4B is stable either way (~0.3). Dividing each dimension by its
    across-sentence std -- taken over the pooled activations of all four sets at this layer --
    strips that leverage. Standardizing by a constant per-dim mean cancels inside a diff-of-means,
    so only the std divide (guarded by ``STD_EPS``) matters here. The matched-norm placebo does not
    catch this; a large raw-vs-standardized gap is itself the massive-activation flag.
    """
    combined = torch.cat(
        [hack_positive, hack_negative, deception_positive, deception_negative], dim=0
    )
    scale = combined.std(dim=0, unbiased=True) + STD_EPS
    hack_dir = diff_of_means(hack_positive / scale, hack_negative / scale)
    deception_dir = diff_of_means(deception_positive / scale, deception_negative / scale)
    return hack_dir, deception_dir


@dataclass(frozen=True)
class LayerComparison:
    """The per-layer read: the two concept cosines against each other and against the placebo.

    Each cosine is reported both raw and ``_standardized`` (per-dimension z-scored, see
    ``standardized_directions``). A large raw-vs-standardized gap on ``cos_hack_deception`` flags
    that a few massive-activation dimensions, not concept content, drove the raw number up.
    """

    layer: int
    cos_hack_deception: float
    cos_hack_placebo: float
    cos_deception_placebo: float
    cos_hack_deception_standardized: float
    cos_hack_placebo_standardized: float
    cos_deception_placebo_standardized: float
    hack_norm: float
    deception_norm: float


def compare_directions(
    hack_positives: dict[int, torch.Tensor],
    hack_negatives: dict[int, torch.Tensor],
    deception_positives: dict[int, torch.Tensor],
    deception_negatives: dict[int, torch.Tensor],
    *,
    seed: int = 0,
) -> list[LayerComparison]:
    """Compute per-layer hack/deception cosine plus the matched-norm placebo baselines.

    Each argument maps layer index -> pooled activations [n, d]. All four must cover the same
    layers; a mismatch is a bug (some capture silently dropped a layer) and raises rather than
    quietly intersecting.
    """
    key_sets = {
        "hack_positives": frozenset(hack_positives),
        "hack_negatives": frozenset(hack_negatives),
        "deception_positives": frozenset(deception_positives),
        "deception_negatives": frozenset(deception_negatives),
    }
    if len(set(key_sets.values())) != 1:
        covered = {name: sorted(layers) for name, layers in key_sets.items()}
        raise ValueError(f"activation sets cover different layers: {covered}")

    hack_dirs = concept_directions(hack_positives, hack_negatives)
    deception_dirs = concept_directions(deception_positives, deception_negatives)

    generator = torch.Generator().manual_seed(seed)
    comparisons: list[LayerComparison] = []
    for layer in sorted(hack_dirs):
        hack = hack_dirs[layer]
        deception = deception_dirs[layer]
        placebo = matched_norm_random_direction(hack, generator)
        hack_std, deception_std = standardized_directions(
            hack_positives[layer],
            hack_negatives[layer],
            deception_positives[layer],
            deception_negatives[layer],
        )
        placebo_std = matched_norm_random_direction(hack_std, generator)
        comparisons.append(
            LayerComparison(
                layer=layer,
                cos_hack_deception=cosine(hack, deception),
                cos_hack_placebo=cosine(hack, placebo),
                cos_deception_placebo=cosine(deception, placebo),
                cos_hack_deception_standardized=cosine(hack_std, deception_std),
                cos_hack_placebo_standardized=cosine(hack_std, placebo_std),
                cos_deception_placebo_standardized=cosine(deception_std, placebo_std),
                # CPU-resident direction vectors, one per layer (~32): bounded, no GPU sync.
                hack_norm=hack.norm().item(),  # HARNESS-SCAN-EXEMPT-item-in-loop
                deception_norm=deception.norm().item(),  # HARNESS-SCAN-EXEMPT-item-in-loop
            )
        )
    return comparisons


# --------------------------------------------------------------------------------------
# Model-capture path (guarded: only reached from run_probe / the CLI, never at import)
# --------------------------------------------------------------------------------------


def _pick_dtype() -> torch.dtype:
    """bf16 on a GPU that supports it, else float32 -- no hardcoded assumption about the card."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(
    model_id: str, *, dtype: torch.dtype | None = None, device: str | None = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a plain HF causal LM + bare tokenizer for activation capture.

    ``dtype=`` (loaders silently ignore ``torch_dtype=`` and fall back to float32) and a bare
    ``AutoTokenizer`` (the 4B checkpoint is a VLM; ``AutoProcessor`` switches to the vision path).
    ``attn_implementation="sdpa"`` keeps the full-attention layers off the O(seq^2) materialised
    score matrix on these long agentic prompts; it is a first-class implementation for this hybrid
    Gated-DeltaNet architecture (``Qwen3_5PreTrainedModel._supports_sdpa is True``), and flash-attn
    is not installed here, so sdpa is the memory-efficient path torch provides in-tree.

    Padding is pinned to the right because the capture passes no ``position_ids`` and the Qwen3.5
    trunk then derives them as a bare ``arange``: under left padding every real token would get a
    rotary position shifted by that row's pad count, while mask-driven pooling still looked correct.
    """
    resolved_dtype = dtype or _pick_dtype()
    device_map = device or ("auto" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=resolved_dtype, device_map=device_map, attn_implementation="sdpa"
    )
    model.eval()
    logger.info("loaded %s dtype=%s device=%s", model_id, resolved_dtype, model.device)
    return model, tokenizer  # pyright: ignore[reportReturnType]


# Attribute chains the transformer trunk sits at, per checkpoint (see _transformer_trunk).
_TRUNK_CHAINS: tuple[str, ...] = ("language_model", "model", "model.language_model")


def _unwrap_peft(model: object) -> object:
    """Peel any PEFT wrapper off ``model``, returning the tree the adapter was injected into.

    A ``PeftModel`` holds the wrapped model two levels down, at ``base_model.model``, so none of
    ``_TRUNK_CHAINS`` resolves against the wrapper: its ``__getattr__`` forwards ``model`` past the
    tuner to the wrapped causal LM (which has no ``.layers`` of its own -- its trunk is one attribute
    further down), and forwards ``language_model`` to nothing. Unwrapping first is what makes the
    chains mean the same thing wrapped and bare.

    This is the normal case rather than an edge one: an adapter is served un-merged for eval and
    interp here, because a bf16 merge loses a large fraction of the trained delta (~36% measured
    2026-08-20), so the model handed to a capture is routinely still inside its wrapper.

    Prompt-learning adapters (prefix tuning, p-tuning, prompt tuning) are refused rather than
    unwrapped. They add no modules to the tree; they prepend virtual tokens inside
    ``PeftModel.forward``, which a trunk-only capture never runs. Unwrapping one would hand back a
    trunk whose activations are the untuned model's, recorded under the tuned checkpoint's name --
    the silent failure this whole path is arranged to avoid -- so say it cannot be represented.
    """
    inner = model
    while isinstance(inner, PeftModel):
        config = inner.active_peft_config
        if config.is_prompt_learning:
            raise NotImplementedError(
                f"{type(inner).__name__} carries a prompt-learning adapter ({config.peft_type}), "
                "which injects virtual tokens in the wrapper's forward instead of modules into the "
                "model tree. A trunk-only capture never runs that forward, so it would record "
                "untuned base activations under this checkpoint's name. Capture needs an adapter "
                "that injects modules (LoRA and friends), or a merged checkpoint."
            )
        inner = inner.get_base_model()
    return inner


def _assert_trunk_carries_the_adapter(wrapper: PeftModel, trunk: torch.nn.Module) -> None:
    """Refuse a trunk that is not the live, adapter-injected one hanging under ``wrapper``.

    This guards the silent sibling of a failed trunk lookup. A lookup that misses raises; a lookup
    that lands on a module which merely *looks* like the trunk -- a base loaded twice, a copy taken
    before wrapping, or the text trunk of a checkpoint whose adapter went into its vision tower
    only -- returns pristine base activations under a trained checkpoint's name. Nothing raises,
    no shape changes, and the measured delta is simply zero: a plausible null.

    Two graph facts rule that out, both free (no forward pass, no weights read):

    * the trunk is a node of ``wrapper``'s own module graph, checked by object identity, so a
      separately loaded or copied tree cannot pass as the one being served;
    * at least one PEFT-injected layer sits inside the trunk, so a forward driven through it
      computes ``Wx + B(Ax)`` rather than ``Wx``. PEFT replaces its target modules in place, so
      this is precisely the property an un-merged capture depends on.

    The first is implied by how :func:`_transformer_trunk` currently resolves the trunk (from
    ``get_base_model()``, hence necessarily a descendant). It is asserted anyway because the
    invariant belongs to the object returned, not to the route taken to it: a later refactor that
    resolves the trunk some other way would otherwise reintroduce the silent case unremarked.
    """
    if all(module is not trunk for module in wrapper.modules()):
        raise RuntimeError(
            f"the located trunk ({type(trunk).__name__}) is not a module of the "
            f"{type(wrapper).__name__} it was located from. Hooking it would read a tree the "
            "adapter was never injected into, so the capture would return untuned base "
            "activations under this checkpoint's name."
        )
    injected = [
        name for name, module in trunk.named_modules() if isinstance(module, BaseTunerLayer)
    ]
    if not injected:
        raise RuntimeError(
            f"no PEFT-injected layer sits inside the located trunk ({type(trunk).__name__}) of this "
            f"{type(wrapper).__name__}, so a forward driven through it computes the untuned base "
            "model. Either the adapter targeted modules outside the trunk (a vision tower, the LM "
            "head) or the trunk lookup landed on the wrong module; both would otherwise read as a "
            "zero delta rather than as a failure."
        )
    # Logged rather than merely checked: this line is the run's own evidence that the capture read an
    # adapted forward, which is otherwise indistinguishable in the artifacts from a base-model read.
    logger.info(f"trunk carries the adapter, injected={len(injected)} first={injected[0]}")


def _transformer_trunk(model: AutoModelForCausalLM) -> torch.nn.Module:
    """Locate the transformer trunk -- the module whose forward yields ``last_hidden_state``.

    The trunk is the parent of the decoder-block ModuleList: ``Qwen3_5TextModel`` for a plain
    causal-LM load (the 0.8B here, and Phi-4/Llama for the cross-model control), nested under
    ``language_model`` for Qwen3.5's vision-language checkpoint (the 4B). Its forward runs every
    decoder layer -- it is exactly what ``...ForCausalLM.forward`` calls internally, right before
    the LM head -- so driving the capture through it fires the per-layer forward hooks identically
    while skipping the multi-GB vocab projection the head computes (see
    :func:`capture_pooled_activations`). Try the known chains and fail loudly if none match.

    A PEFT wrapper is peeled off first (:func:`_unwrap_peft`) and then checked: the trunk found
    under it has to be the very module the adapter was injected into
    (:func:`_assert_trunk_carries_the_adapter`), or the capture is reading the untuned base.
    """
    inner = _unwrap_peft(model)
    for chain in _TRUNK_CHAINS:
        obj: object = inner
        try:
            for attr in chain.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "layers"):
            if isinstance(model, PeftModel):
                _assert_trunk_carries_the_adapter(model, obj)  # pyright: ignore[reportArgumentType]
            return obj  # pyright: ignore[reportReturnType]
    tried = ", ".join(f"{chain}.layers" for chain in _TRUNK_CHAINS)
    wrapping = "" if inner is model else f" (unwrapped from {type(model).__name__})"
    raise AttributeError(
        f"could not locate the transformer trunk on {type(inner).__name__}{wrapping}; tried {tried}"
    )


def _decoder_layers(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
    """Return the decoder-block ModuleList -- the ``.layers`` of the transformer trunk."""
    return _transformer_trunk(model).layers  # pyright: ignore[reportReturnType]


def _make_capture_hook(
    store: dict[int, torch.Tensor], idx: int
) -> Callable[[object, object, object], None]:
    """Build a read-only forward hook that stashes decoder layer ``idx``'s residual output.

    Decoder layers return ``(hidden, ...)`` in transformers, some return the tensor bare -- take
    ``[0]`` either way. The hook stores the reference and returns ``None`` (never mutates the
    output), so both the pooled and the per-position capture read the same untouched residual.
    """

    def hook(module: object, inputs: object, output: object) -> None:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        store[idx] = hidden  # pyright: ignore[reportArgumentType]

    return hook


def _make_pooling_hook(
    stores: Mapping[str, dict[int, list[torch.Tensor]]],
    idx: int,
    attention_mask: torch.Tensor,
    poolers: Mapping[str, Pooler],
) -> Callable[[object, object, object], None]:
    """Build a read-only forward hook that pools layer ``idx``'s residual every requested way.

    Pooling inside the hook, rather than stashing the hidden state and pooling afterwards, keeps
    only one ``[batch, d]`` vector per layer per pooling alive: the capture's peak memory stops
    scaling with sequence length (32 live ``[batch, seq, d]`` hidden states at batch 4 on ~4000-token
    agentic prompts is ~2.5 GiB of avoidable peak). It also stops the result depending on each
    decoder layer allocating a fresh output tensor -- a layer that wrote its residual in place would
    leave every stashed reference showing the last layer's values. Upcast before pooling so
    precision is unchanged (bf16 means lose precision under summation), then move to CPU.

    Every pooler reads the same upcast hidden state, so asking for two poolings costs one forward
    and yields bit-for-bit what two single-pooling forwards would (the forward is deterministic and
    each pooler's arithmetic is unchanged); it used to cost one forward per pooling.
    """

    def hook(module: object, inputs: object, output: object) -> None:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        upcast = hidden.float()  # pyright: ignore[reportAttributeAccessIssue]
        for pooling, pool_fn in poolers.items():
            stores[pooling][idx].append(pool_fn(upcast, attention_mask).cpu())

    return hook


@torch.no_grad()
def capture_pooled_activations_multi(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentences: list[str],
    *,
    poolings: Sequence[str],
    batch_size: int = 16,
) -> dict[str, dict[int, torch.Tensor]]:
    """Run ``sentences`` once and return pooling -> layer -> pooled activations [n_sentences, d].

    Residual stream is read from each decoder layer's output via a forward hook that pools on the
    spot, every requested way at once (see :func:`_make_pooling_hook`). Sentences are fed raw (no
    chat template): the concept lives in the statement, matching the paper's raw-sentence extraction.

    The forward pass runs the transformer trunk, not the full causal LM, so the ~10 GiB
    ``[batch, seq, vocab]`` logits the LM head would materialise are never computed -- the vocab is
    ~248k and these agentic prompts are long, so that projection alone OOMs a 24 GiB card even at
    batch size 1. This changes no captured activation: the hooks read decoder-layer outputs, which
    are strictly upstream of the head, and the trunk is exactly what ``...ForCausalLM.forward`` runs
    before projecting to logits.

    Right padding is required, not merely assumed: no ``position_ids`` are passed, so a left-padded
    batch would shift every real token's rotary position while the mask-driven pooling still looked
    clean. Refused loudly instead, since the sibling ``model_backend`` sets left padding for its own
    batched decoding and its tokenizer could be handed here.
    """
    unknown = [pooling for pooling in poolings if pooling not in POOLERS]
    if unknown or not poolings:
        raise ValueError(
            f"unknown poolings {unknown}; expected a non-empty subset of {sorted(POOLERS)}"
        )
    if len(set(poolings)) != len(poolings):
        raise ValueError(f"poolings {list(poolings)} repeat a name")
    if tokenizer.padding_side != "right":  # pyright: ignore[reportAttributeAccessIssue]
        raise ValueError(
            "capture_pooled_activations needs a right-padding tokenizer, got padding_side="
            f"{tokenizer.padding_side!r}: "  # pyright: ignore[reportAttributeAccessIssue]
            "it passes no position_ids, so left padding shifts every real token's rotary position "
            "while the pooled vectors still look well-formed"
        )
    poolers = {pooling: POOLERS[pooling] for pooling in poolings}
    # Trunk and its decoder layers sit at different attrs across VL vs causal-LM; resolve robustly.
    trunk = _transformer_trunk(model)
    layers = _decoder_layers(model)
    n_layers = len(layers)

    stores: dict[str, dict[int, list[torch.Tensor]]] = {
        pooling: {i: [] for i in range(n_layers)} for pooling in poolings
    }
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer(  # pyright: ignore[reportCallIssue]
            batch, return_tensors="pt", padding=True
        ).to(model.device)  # pyright: ignore[reportAttributeAccessIssue]
        handles = [
            layer.register_forward_hook(
                _make_pooling_hook(stores, i, encoded["attention_mask"], poolers)
            )
            for i, layer in enumerate(layers)
        ]
        try:
            trunk(**encoded)  # trunk forward only: hooks fire, no LM head, no full-vocab logits.
        finally:
            for handle in handles:
                handle.remove()

    return {
        pooling: {layer: torch.cat(store[layer], dim=0) for layer in range(n_layers)}
        for pooling, store in stores.items()
    }


@torch.no_grad()
def capture_pooled_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentences: list[str],
    *,
    pooling: str = "mean",
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """Run ``sentences`` through the model and return layer -> pooled activations [n_sentences, d].

    The one-pooling form of :func:`capture_pooled_activations_multi`, which carries the contract
    (trunk-only forward, right padding required, pooling inside the hook).
    """
    return capture_pooled_activations_multi(
        model, tokenizer, sentences, poolings=[pooling], batch_size=batch_size
    )[pooling]


@torch.no_grad()
def capture_positionwise_activations(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layers: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    (
        """Per-position residual stream over pre-tokenised ids -- the generation-phase capture """
        """primitive.

    Where :func:`capture_pooled_activations` collapses each sentence to one pooled vector, this
    keeps every position: it returns ``layer -> [batch, seq, d]`` (float32, on CPU) so a caller
    can select the positions where the model was emitting its OWN tokens and pool only those
    (see :mod:`reward_hacking.interp.generation_capture`). Same read-only, trunk-only path as
    the pooled capture -- forward hooks store each decoder layer's output and the LM head never
    runs, so the full-vocab logits are never materialised. ``input_ids`` and ``attention_mask``
    are ``[batch, seq]`` and are moved to the model's device; one forward pass fills every
    layer. Because attention is causal, a single pass over prompt+response reproduces the
    residual each position held while it was being generated, which is why capture can follow
    generation rather than hook into it.

    ``layers`` keeps only the named decoder layers (default: all of them). Activation patching
    reads one layer per cell, and this is the expensive capture: at the thinking-mode 32768-token
    cap, all 32 layers in float32 is ~10 GiB of CPU activations for a measurement that uses one.
    A layer this model does not have is rejected up front, since the alternative is an opaque
    ``IndexError`` from a ``ModuleList`` (or, for a negative index, a hook quietly placed on a
    different layer than the caller named and a result keyed to the index they passed).
    """
    )
    all_layers = _decoder_layers(model)
    n_layers = len(all_layers)
    wanted = range(n_layers) if layers is None else layers
    outside = [layer for layer in wanted if not 0 <= layer < n_layers]
    if outside:
        raise ValueError(f"layers {outside} are outside this {n_layers}-layer model")
    trunk = _transformer_trunk(model)
    captured: dict[int, torch.Tensor] = {}
    handles = [
        all_layers[i].register_forward_hook(_make_capture_hook(captured, i))  # pyright: ignore[reportArgumentType]
        for i in wanted
    ]
    try:
        trunk(
            input_ids=input_ids.to(model.device),  # pyright: ignore[reportAttributeAccessIssue]
            attention_mask=attention_mask.to(model.device),  # pyright: ignore[reportAttributeAccessIssue]
            # No cache: this is a single forward for its hooks' sake and nothing decodes from it, so
            # the DynamicCache the trunk would otherwise build is allocated, filled and thrown away.
            # Free at any length and worth real VRAM at the ones this capture runs at -- a 65k-token
            # trace's per-layer key/value tensors are the largest thing in the pass.
            use_cache=False,
        )
        return {layer: captured[layer].float().cpu() for layer in wanted}
    finally:
        for handle in handles:
            handle.remove()


@dataclass(frozen=True)
class LayerOutput:
    """One decoder layer's whole output from one forward, kept so a later forward can REPLAY it.

    ``output`` is ``[batch, seq, d]`` on the CPU **in the residual's own dtype**, unlike
    :func:`capture_positionwise_activations`, which upcasts to float32 because its consumers do
    arithmetic on what it returns. Nothing here does: the tensor is handed straight back to the model,
    so keeping the model's dtype makes "the exact bits the layer produced" true by construction
    rather than by a round-trip argument, and halves both the host copy and the transfer back (a 6k-
    token bf16 residual at 4096 dims is ~50 MB, not ~100 MB).

    ``input_ids`` (a CPU copy) and ``layer`` say what the output is OF, so a replay over other ids or
    at another layer is refused instead of silently continuing the wrong forward. ``returns_tuple``
    records whether the layer returned ``(hidden, ...)`` or a bare tensor, so a stand-in can return
    the same shape to the trunk loop.
    """

    layer: int
    input_ids: torch.Tensor
    output: torch.Tensor
    returns_tuple: bool

    @property
    def dtype(self) -> torch.dtype:
        """The residual's dtype on the model, which is the dtype this output is stored in."""
        return self.output.dtype

    def on_device(self, device: torch.device) -> torch.Tensor:
        """Return a fresh copy of the layer's output on ``device``: the exact bits it produced."""
        return self.output.to(device=device, copy=True)


@torch.no_grad()
def capture_layer_output_during[T](
    model: AutoModelForCausalLM,
    layer: int,
    input_ids: torch.Tensor,
    run: Callable[[], T],
) -> tuple[T, LayerOutput]:
    """Run ``run()`` -- one forward over ``input_ids`` -- and return its result plus layer ``layer``'s output.

    The forward is the caller's, so what is captured is that forward's own layer output: the same
    call path, cache setting and kernels a later replay through the same caller will use. That is the
    point of capturing here rather than through the trunk-only capture above, whose ``use_cache=False``
    forward is not guaranteed to reproduce a cache-building readout forward bit for bit on the
    hybrid-attention layers. Reads one layer only, and refuses a layer this model does not have.
    """
    all_layers = _decoder_layers(model)
    if not 0 <= layer < len(all_layers):
        raise ValueError(f"layer {layer} is outside this {len(all_layers)}-layer model")
    captured: dict[int, torch.Tensor] = {}
    structure: dict[int, bool] = {}

    def hook(module: object, inputs: object, output: object) -> None:
        del module, inputs
        structure[layer] = isinstance(output, tuple)
        captured[layer] = output[0] if isinstance(output, tuple) else output  # pyright: ignore[reportArgumentType]

    handle = all_layers[layer].register_forward_hook(hook)
    try:
        result = run()
    finally:
        handle.remove()
    if layer not in captured:
        raise RuntimeError(f"the forward never ran decoder layer {layer}, so nothing was captured")
    raw = captured[layer]
    return result, LayerOutput(
        layer=layer,
        input_ids=input_ids.detach().cpu(),
        output=raw.detach().to("cpu", copy=True),
        returns_tuple=structure[layer],
    )


def run_probe(
    model_id: str,
    *,
    pooling: str = "mean",
    seed: int = 0,
    batch_size: int = 16,
    limit: int | None = None,
) -> list[LayerComparison]:
    """Load the model, capture activations for both concepts' pairs, and compare per layer."""
    shortcut = stimuli.SHORTCUT_PAIRS[:limit] if limit else stimuli.SHORTCUT_PAIRS
    deception = stimuli.DECEPTION_PAIRS[:limit] if limit else stimuli.DECEPTION_PAIRS
    logger.info(
        "probing %s pooling=%s seed=%d shortcut_pairs=%d deception_pairs=%d",
        model_id,
        pooling,
        seed,
        len(shortcut),
        len(deception),
    )
    model, tokenizer = load_model_and_tokenizer(model_id)

    def capture(sents: list[str]) -> dict[int, torch.Tensor]:
        return capture_pooled_activations(
            model, tokenizer, sents, pooling=pooling, batch_size=batch_size
        )

    return compare_directions(
        capture(stimuli.positives(shortcut)),
        capture(stimuli.negatives(shortcut)),
        capture(stimuli.positives(deception)),
        capture(stimuli.negatives(deception)),
        seed=seed,
    )


def _format_table(comparisons: list[LayerComparison]) -> str:
    """Per-layer results table.

    The ``cos_z`` columns are the per-dimension-standardized cosines; a large raw-vs-``cos_z`` gap
    on cos(hack,decep) is the massive-activation flag (see ``standardized_directions``).
    """
    header = (
        f"{'layer':>5}  {'cos(hack,decep)':>15}  {'cos(hack,placebo)':>17}  "
        f"{'cos(decep,placebo)':>18}  {'cos_z(h,d)':>11}  {'cos_z(h,p)':>11}  "
        f"{'cos_z(d,p)':>11}  {'|hack|':>8}  {'|decep|':>8}"
    )
    rows = [
        f"{c.layer:>5}  {c.cos_hack_deception:>15.4f}  {c.cos_hack_placebo:>17.4f}  "
        f"{c.cos_deception_placebo:>18.4f}  {c.cos_hack_deception_standardized:>11.4f}  "
        f"{c.cos_hack_placebo_standardized:>11.4f}  {c.cos_deception_placebo_standardized:>11.4f}  "
        f"{c.hack_norm:>8.2f}  {c.deception_norm:>8.2f}"
        for c in comparisons
    ]
    return "\n".join([header, *rows])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-hacking vs deception direction probe (#5)")
    backend_cli.add_backend_choice(parser)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--pooling", choices=sorted(POOLERS), default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap pairs per concept (default: all)"
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="also dump per-layer results as JSON here"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the probe and print its per-layer results."""
    args = _parse_args(argv)
    backend_cli.require_in_process_weights(args.backend, purpose="capture pooled activations")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    comparisons = run_probe(
        args.model_id,
        pooling=args.pooling,
        seed=args.seed,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(_format_table(comparisons))  # noqa: T201  # Intentional CLI table output.
    if args.json_out is not None:
        args.json_out.write_text(json.dumps([asdict(c) for c in comparisons], indent=2))
        logger.info("wrote per-layer results to %s", args.json_out)


if __name__ == "__main__":
    main()
