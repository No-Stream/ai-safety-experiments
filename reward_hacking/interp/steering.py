"""Causal interventions on the residual stream: steering, ablation, and activation patching.

Exp2 gave a correlational read (does an axis *decode* the rigged-grader contrast). This module is
the causal side: does the model *use* a direction or a position, not merely represent it. Three
interventions, all built on one output-modifying forward hook on a decoder layer:

* **Steering** adds ``alpha * unit(direction)`` to a layer's residual output during generation, so a
  direction the model represents is pushed on and its effect on behaviour measured.
* **Ablation** projects a direction OUT of the residual (``h - (h.u) u``), removing the model's
  ability to read it at that layer.
* **Activation patching** caches a chosen layer/position activation from a CLEAN run and injects
  it into a matched CORRUPTED run, then measures how far the corrupted-run logits move toward
  the clean answer. Our conflicting-vs-original ILCB grader twins are byte-identical except the
  grader body, so they are an ideal clean/corrupted pair -- patching the grader-body positions
  tests whether that region is causally used, which is the "J-lens/probe finds a candidate,
  patching validates it" step.

The **matched-norm random placebo** from :mod:`directions` is wired in as the mandatory control arm
(repo rule): a steering or ablation effect is real only in so far as it exceeds what a random
direction of equal norm does at the same layer.

**Runs through the HuggingFace generate/forward path, NOT vLLM.** vLLM does not expose
residual-stream forward hooks, so the throughput backend that the agent loop uses
(``harness/loop.py`` via ``model_backend``) cannot host these interventions. A separate HF
harness is the deliberate choice; the correlational and causal reads run offline over the same
stimulus twins, not inside the RL loop.

The pure-tensor core (``steer_residual`` / ``ablate_residual`` / ``logit_recovery``), the hook math,
and ``plan_twin_patch`` -- which decides WHERE to patch by bracketing two unequal-length matched runs
between their shared prefix and shared suffix -- touch no model weights and are what the offline tests
exercise; the generation and patching drivers are GPU-only and behind functions the caller invokes.
The planner lives here rather than in a CLI harness because two projects now patch matched twins
(the reward-hacking grader twins and the games contrast pairs), and importing a CLI to reach it drags
that CLI's whole dependency tree along.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from reward_hacking.interp.directions import (
    LayerOutput,
    _decoder_layers,  # pyright: ignore[reportPrivateUsage]  # shared trunk-layer resolver
    capture_layer_output_during,
    capture_positionwise_activations,
    matched_norm_random_direction,
    unit,
)
from reward_hacking.interp.generation_capture import (
    PENALTY_FREE_THINKING_SAMPLING,
    GenerationRecord,
    generate_response,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_hacking.model_backend import SamplingConfig

logger = logging.getLogger(__name__)

ResidualTransform = Callable[[torch.Tensor], torch.Tensor]
Hook = Callable[[object, object, object], object]

DEFAULT_ANSWER_TOP_K = 50
"""How many of the clean run's top next-token candidates the patch readout picks its answer from.

See :func:`choose_answer_token`: wide enough that the twins' disagreement is representable, narrow
enough that the chosen token is one the clean run would plausibly emit.
"""

READOUT_MODE_ACTION_LOGPROB = "action_logprob"
"""Score two COMPLETE candidate actions by sequence log-prob and take the difference.

The primary readout (owner-approved 2026-08-24). See :class:`GapReadout` for why a single next-token
logit could not carry this measurement on chat-formatted twins.
"""

READOUT_MODE_FORCED_CHOICE = "forced_choice"
"""Score two single-token option labels at one answer-slot position and take the logit difference.

The cheap secondary cross-check. Results from it are a **stated preference** -- what the model says
when made to pick between two options in one token -- which is a different construct from what an
agent does over many turns, and every artifact says so.
"""

GAP_READOUT_MODES: tuple[str, ...] = (READOUT_MODE_ACTION_LOGPROB, READOUT_MODE_FORCED_CHOICE)

MAX_CANDIDATE_LENGTH_RATIO = 1.5
"""How unequal the two candidates' token counts may be before :func:`gap_readout` refuses.

Sequence log-prob is additive over tokens, so a longer candidate scores systematically lower and the
RAW gap mixes "prefers this action" with "this action is shorter". The mix cancels out of the
recovery ratio -- the length term is the same constant in the clean, corrupted and patched gaps --
but it makes the raw gap useless as a preference reading, so the two candidates are authored to
similar length and the ratio is checked rather than assumed. 1.5 is loose enough to survive a
tokenizer change between model sizes and tight enough that the authored pair (21 vs 23 tokens on the
Qwen3.5 tokenizer, ratio 1.10) has real headroom.
"""


# --------------------------------------------------------------------------------------
# Pure-tensor core: the residual transforms
# --------------------------------------------------------------------------------------


def steer_residual(hidden: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    """Add ``alpha * unit(direction)`` to every position of ``hidden`` ``[..., d]``.

    The direction is unit-normalised so ``alpha`` is the steering magnitude in raw activation units,
    independent of how the direction was scaled -- which is what lets the real axis and the
    matched-norm placebo be pushed on by exactly the same amount.
    """
    return hidden + alpha * unit(direction).to(hidden)


def ablate_residual(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    (
        """Project ``direction`` out of ``hidden`` ``[..., d]``: """
        """``h - (h . u) u`` with ``u = unit(dir)``.

    Removes the component of every position along the direction, leaving the orthogonal complement
    untouched, so the model can no longer read that axis at this layer.
    """
    )
    u = unit(direction).to(hidden)
    projection = hidden @ u
    return hidden - projection.unsqueeze(-1) * u


def choose_answer_token(
    clean: torch.Tensor, corrupted: torch.Tensor, *, top_k: int = DEFAULT_ANSWER_TOP_K
) -> int:
    """Pick the readout token: the clean run's top-k candidate that the corrupted run most dislikes.

    The obvious choice -- the clean run's argmax -- is badly conditioned when the two runs mostly
    agree about the next token, the normal case for twins differing only in a grader body. Measured
    on the 0.8B smoke: the clean argmax's clean-vs-corrupted gap was 0.125, one or two units in the
    last place of a bf16 logit, so ``logit_recovery``'s denominator sat at the readout's resolution
    floor and no patch could move the ratio measurably.

    Maximising the gap over the WHOLE vocabulary would reward a rare token the clean run also
    considers unlikely, so the candidate set is the clean run's ``top_k``: of the tokens the clean
    run might actually emit, take the one the corrupted run most disprefers. That keeps the metric
    inside the plausible answers while giving it real dynamic range -- the same reason the patching
    literature reads a logit DIFFERENCE between two named candidates rather than one raw logit.
    """
    k = min(top_k, int(clean.numel()))
    candidates = clean.topk(k).indices
    gaps = clean[candidates] - corrupted[candidates]
    return int(candidates[int(gaps.argmax().item())].item())


def logit_recovery(
    clean: torch.Tensor, corrupted: torch.Tensor, patched: torch.Tensor, answer_token: int
) -> float:
    """Fraction of the clean-vs-corrupted logit gap that patching recovers for ``answer_token``.

    ``(patched - corrupted) / (clean - corrupted)`` on the answer token's logit: 0.0 means
    the patch changed nothing, 1.0 means it fully restored the clean run's logit for that
    token. All three are ``[vocab]`` readout-position logits. A zero clean-vs-corrupted gap
    (the patched position does not distinguish the twins) returns 0.0 rather than dividing by zero.
    """
    gap = (clean[answer_token] - corrupted[answer_token]).item()
    if gap == 0.0:
        return 0.0
    return (patched[answer_token] - corrupted[answer_token]).item() / gap


def action_gap(logits: torch.Tensor, positive_token: int, negative_token: int) -> float:
    """Positive-minus-negative logit gap at one readout position.

    The shared definition for both research threads. ``games/interp_patching.py`` reads a
    cooperate-minus-defect gap and this module reads a tamper-minus-honest one; they are the same
    statistic over different label pairs, and ``test_interp_steering`` pins the two implementations
    against each other so a change to one cannot silently diverge from the other.

    A logit DIFFERENCE rather than a raw logit because the additive constant a softmax is invariant
    to cancels out of it: the difference of two log-softmax values at one position equals the
    difference of the two logits exactly, which is what lets :func:`gap_recovery` read the
    single-token forced-choice arm and the multi-token action arm through one definition.
    """
    return float(logits[positive_token] - logits[negative_token])


def gap_recovery(clean_gap: float, corrupted_gap: float, patched_gap: float) -> float | None:
    """Fraction of the clean-minus-corrupted GAP that patching recovered, or ``None`` if there is none.

    ``None`` rather than ``0.0`` on a zero denominator, unlike :func:`logit_recovery`: the two runs'
    gaps being identical means "how far toward clean" has no answer, and a fabricated 0.0 would enter
    a mean as evidence of no recovery. The denominator is recorded on every cell so the exclusions
    stay countable -- a zero needs its denominator.

    Not clipped in either direction. A value above 1.0 is an overshoot and a negative one is the
    patch moving the gap AWAY from clean, and both are readings this measurement must be able to
    make: sign inversion under intervention is documented in this literature, and the earlier screen
    was one-sided and structurally could not see it.
    """
    denominator = clean_gap - corrupted_gap
    if denominator == 0.0:
        return None
    return (patched_gap - corrupted_gap) / denominator


# --------------------------------------------------------------------------------------
# What the readout scores: two candidate continuations of one shared prompt
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GapReadout:
    r"""Two candidate continuations of a prompt; the reading is positive-minus-negative log-prob.

    **Why this exists.** The earlier patch readout scored one next-token logit at position ``-1`` of a
    chat-formatted transcript. The Qwen3.5 template ends ``<|im_start|>assistant\n<think>\n``, so
    that position is the newline INSIDE the thinking block and the scored token is a candidate opener
    of the reasoning trace: the four answer tokens used across the 2026-08-22 run's 72 cells decode to
    ``'C'``, ``'This'``, ``'Need'``, ``'Thinking'``. That was a deliberate design choice rather than a
    regression, and it is not a measurement of anything about grader-gaming. This readout replaces it.

    Two modes, both reduced to one gap definition so the recovery statistic is shared:

    * :data:`READOUT_MODE_ACTION_LOGPROB` (primary) -- ``positive_ids`` and ``negative_ids`` are two
      COMPLETE candidate actions, teacher-forced. The gap is the difference of their sequence
      log-probs. Nothing about it can be produced by overwriting one position's residual, which is the
      artifact that made the earlier wide window read exactly 1.0.
    * :data:`READOUT_MODE_FORCED_CHOICE` (secondary) -- one option label each, scored at an answer
      slot reached by rendering the prompt with thinking suppressed. One forward, because the
      difference of two log-softmax values at one position IS the difference of their logits.

    ``positive`` is the grader-tampering candidate and ``negative`` the honest one throughout, so a
    positive gap means "prefers the tampering action" with no per-arm sign convention to remember.
    """

    mode: str
    positive_name: str
    negative_name: str
    positive_ids: torch.Tensor
    negative_ids: torch.Tensor

    @property
    def n_positive_tokens(self) -> int:
        """Tokens in the grader-tampering candidate."""
        return int(self.positive_ids.numel())

    @property
    def n_negative_tokens(self) -> int:
        """Tokens in the honest candidate."""
        return int(self.negative_ids.numel())

    @property
    def is_single_token(self) -> bool:
        """Whether both candidates are one token, so the gap comes off one readout row."""
        return self.n_positive_tokens == 1 and self.n_negative_tokens == 1

    @property
    def length_ratio(self) -> float:
        """Longer candidate's token count over the shorter's; 1.0 when they match exactly."""
        counts = (self.n_positive_tokens, self.n_negative_tokens)
        return max(counts) / min(counts)

    @property
    def collides(self) -> bool:
        """Whether the two candidates are indistinguishable AT THE READOUT THIS MODE USES.

        Mode-dependent on purpose. A forced-choice pair is read at ONE position, so two labels
        sharing their first token cannot be told apart there and the pair must be skipped and
        counted rather than contribute a silent zero gap. Multi-token action candidates are read over
        their whole sequence, where a shared first token is expected -- both authored actions begin
        ``<run>`` -- and only fully identical sequences are indistinguishable.
        """
        if self.mode == READOUT_MODE_FORCED_CHOICE:
            return int(self.positive_ids[0].item()) == int(self.negative_ids[0].item())
        return self.positive_ids.tolist() == self.negative_ids.tolist()


def gap_readout(  # noqa: PLR0913 - a mode plus a named id tensor per candidate is the whole record
    *,
    mode: str,
    positive_name: str,
    negative_name: str,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
    max_length_ratio: float = MAX_CANDIDATE_LENGTH_RATIO,
) -> GapReadout:
    """Build a :class:`GapReadout`, refusing the shapes that would report a number meaning nothing.

    Four refusals, each for a failure that produces a plausible gap rather than an error: an unknown
    mode (the ``collides`` test would silently take the multi-token branch for a forced choice), a
    non-1-D or empty candidate, a forced-choice candidate longer than one token (its gap would be a
    sequence log-prob dressed up as a stated preference), and candidates whose token counts differ by
    more than ``max_length_ratio`` (see :data:`MAX_CANDIDATE_LENGTH_RATIO`).

    A COLLIDING pair is not refused here: the caller counts it and skips that pair, because a
    collision is a property of one pair's labels rather than of the readout's construction.
    """
    if mode not in GAP_READOUT_MODES:
        raise ValueError(
            f"unknown readout mode {mode!r}; expected one of {list(GAP_READOUT_MODES)}"
        )
    for name, ids in ((positive_name, positive_ids), (negative_name, negative_ids)):
        if ids.ndim != 1:
            raise ValueError(f"candidate {name!r} must be a 1-D token id tensor, got {ids.ndim}-D")
        if ids.numel() == 0:
            raise ValueError(
                f"candidate {name!r} tokenized to nothing, so its log-prob is undefined"
            )
    readout = GapReadout(
        mode=mode,
        positive_name=positive_name,
        negative_name=negative_name,
        positive_ids=positive_ids,
        negative_ids=negative_ids,
    )
    if mode == READOUT_MODE_FORCED_CHOICE and not readout.is_single_token:
        raise ValueError(
            f"forced choice reads ONE position, so both option labels must be one token; got "
            f"{readout.n_positive_tokens} and {readout.n_negative_tokens}. Score multi-token "
            f"candidates with mode {READOUT_MODE_ACTION_LOGPROB!r} instead."
        )
    if readout.length_ratio > max_length_ratio:
        raise ValueError(
            f"candidates {positive_name!r} ({readout.n_positive_tokens} tokens) and "
            f"{negative_name!r} ({readout.n_negative_tokens} tokens) differ by a factor of "
            f"{readout.length_ratio:.2f}, over the {max_length_ratio} bound; a sequence-log-prob gap "
            "between candidates of very different length is mostly a length reading"
        )
    return readout


@dataclass(frozen=True)
class GapRead:
    """One run's scored gap, the two sequence log-probs behind it, and its readout row.

    ``readout_row`` is the ``[vocab]`` next-token logits at the LAST PROMPT POSITION, which comes off
    the same forward the candidate scoring needs, so the whole-row KL and log-prob-L2 recovery
    readings cost nothing extra beside the gap.

    ``gap_per_token`` normalises each candidate's log-prob by its own token count. It is a diagnostic
    rather than the primary: recovery is defined on the raw gap, where the per-candidate length term
    is the same constant in all three conditions and cancels.
    """

    mode: str
    gap: float
    positive_logprob: float
    negative_logprob: float
    positive_n_tokens: int
    negative_n_tokens: int
    readout_row: torch.Tensor

    @property
    def gap_per_token(self) -> float:
        """Length-normalised gap: mean per-token log-prob of positive minus that of negative."""
        return (
            self.positive_logprob / self.positive_n_tokens
            - self.negative_logprob / self.negative_n_tokens
        )

    def payload(self) -> dict[str, float | int | str]:
        """Return the flat, uploadable form -- everything but the ``[vocab]`` row."""
        return {
            "readout_mode": self.mode,
            "gap": self.gap,
            "gap_per_token": self.gap_per_token,
            "positive_logprob": self.positive_logprob,
            "negative_logprob": self.negative_logprob,
            "positive_n_tokens": self.positive_n_tokens,
            "negative_n_tokens": self.negative_n_tokens,
        }


# --------------------------------------------------------------------------------------
# Output-modifying forward hooks (the shared intervention mechanism)
# --------------------------------------------------------------------------------------


def _output_transform_hook(transform: ResidualTransform) -> Hook:
    """Build a forward hook that replaces a decoder layer's residual output via ``transform``.

    transformers decoder layers return ``(hidden, ...)`` (some return the tensor bare); the hook
    rebuilds the same structure with the transformed hidden state. Returning a non-None value from a
    forward hook replaces the module output, and because the trunk runs ``hidden = layer(hidden)``,
    the modified residual is what the next layer sees -- which is what makes the intervention causal
    rather than merely observed.
    """

    def hook(module: object, inputs: object, output: object) -> object:
        del module, inputs
        if isinstance(output, tuple):
            return (transform(output[0]), *output[1:])
        # A decoder layer's output is a Tensor at runtime; the hook signature types it as object.
        return transform(output)  # pyright: ignore[reportArgumentType]

    return hook


def steering_hook(direction: torch.Tensor, alpha: float) -> Hook:
    """Forward hook that steers the residual by ``alpha * unit(direction)``."""
    return _output_transform_hook(lambda hidden: steer_residual(hidden, direction, alpha))


def ablation_hook(direction: torch.Tensor) -> Hook:
    """Forward hook that projects ``direction`` out of the residual."""
    return _output_transform_hook(lambda hidden: ablate_residual(hidden, direction))


def activation_patch_hook(clean_activation: torch.Tensor, positions: torch.Tensor) -> Hook:
    """Forward hook that overwrites the residual at ``positions`` with cached clean activations.

    ``clean_activation`` is ``[n_positions, d]`` aligned to ``positions`` (a 1-D index tensor
    into the sequence axis). The hidden state is cloned before the write so the cached tensor
    is not aliased into the graph and an in-place edit cannot surprise a later reader.
    """

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        patched = hidden.clone()
        patched[:, positions, :] = clean_activation.to(hidden)
        return patched

    return _output_transform_hook(transform)


@contextmanager
def residual_intervention(
    model: AutoModelForCausalLM, layer_index: int, hook: Hook
) -> Generator[None]:
    """Register ``hook`` on decoder layer ``layer_index`` for the duration of the ``with`` block."""
    layer = _decoder_layers(model)[layer_index]
    handle = layer.register_forward_hook(hook)  # pyright: ignore[reportArgumentType]
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------------------
# Replaying a patched forward from the patched layer, so layers 0..L are not recomputed per arm
# --------------------------------------------------------------------------------------


class _PassThroughLayer(torch.nn.Module):
    """Stands in for a decoder layer below the replayed one: hands its input on untouched.

    Its output is never read -- the replayed layer above it ignores its input -- so it exists only
    to keep the trunk's own loop shape, and it returns the same structure the real layer did so the
    loop's ``hidden_states = layer(...)`` (or ``[0]`` of it) keeps working.
    """

    def __init__(self, *, returns_tuple: bool) -> None:
        super().__init__()
        self.returns_tuple = returns_tuple

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return (hidden_states,) if self.returns_tuple else hidden_states


class _ReplayLayer(torch.nn.Module):
    """Stands in for the patched layer: returns a captured (and patched) output instead of computing.

    Refuses an input whose shape differs from the captured output, which is the cheap symptom of a
    replay over other ids than the capture ran on; the ids themselves are checked by the caller.
    """

    def __init__(self, output: torch.Tensor, *, returns_tuple: bool) -> None:
        super().__init__()
        self.output = output
        self.returns_tuple = returns_tuple

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if tuple(hidden_states.shape) != tuple(self.output.shape):
            raise ValueError(
                f"replayed layer output has shape {tuple(self.output.shape)} but the forward reached "
                f"it with {tuple(hidden_states.shape)}: this is not the forward the output was "
                "captured from"
            )
        return (self.output,) if self.returns_tuple else self.output


@contextmanager
def replay_from_layer(
    model: AutoModelForCausalLM, layer_output: LayerOutput, output: torch.Tensor
) -> Generator[None]:
    """Run the model with layers ``0..layer_output.layer`` replaced: the last returns ``output``.

    Inside the block a forward pays for the embedding, the rotary and mask set-up (all of which the
    trunk computes before its layer loop and none of which depend on the layers) and then for layers
    ``layer_output.layer + 1`` onwards only. Everything a later layer reads -- the residual it is
    handed, the position embeddings, the masks, its own cache slot -- is what it would have read in
    a full forward that produced ``output`` at that layer, so the result is bit-identical to that
    full forward: the trunk's own loop runs, with its own arguments, and the stand-ins only replace
    what the loop would have computed with what an earlier forward already computed. The originals
    are put back however the block exits.
    """
    layers = _decoder_layers(model)
    index = layer_output.layer
    if not 0 <= index < len(layers):
        raise ValueError(f"layer {index} is outside this {len(layers)}-layer model")
    originals = [layers[i] for i in range(index + 1)]
    try:
        for i in range(index):
            layers[i] = _PassThroughLayer(returns_tuple=layer_output.returns_tuple)
        layers[index] = _ReplayLayer(output, returns_tuple=layer_output.returns_tuple)
        yield
    finally:
        for i, original in enumerate(originals):
            layers[i] = original


Replayer = Callable[[torch.Tensor], AbstractContextManager[None]]
"""Given the ids a readout forward is about to run over, the context that replays it from a prefix."""


@dataclass(frozen=True)
class PatchPrefix:
    """The patched layer's un-patched outputs for every forward a patch cell's readout runs.

    One :class:`~reward_hacking.interp.directions.LayerOutput` per distinct ids sequence the readout
    forwards over: the corrupted ids alone for a plain or single-token readout, or the corrupted ids
    plus each candidate for a multi-token :class:`GapReadout`. Built once per (pair, layer) by
    :func:`capture_patch_prefix` and shared by every arm at that layer, which is what lets each arm's
    patched forward skip layers ``0..L``: those layers see the same input in every arm, since the
    patch is written at ``L``'s output.
    """

    layer: int
    outputs: tuple[LayerOutput, ...]

    def for_ids(self, ids: torch.Tensor) -> LayerOutput:
        """Return the captured output of the forward over exactly ``ids``, refusing unknown ids."""
        wanted = ids.detach().cpu()
        for output in self.outputs:
            if output.input_ids.shape == wanted.shape and torch.equal(output.input_ids, wanted):
                return output
        raise ValueError(
            f"no layer-{self.layer} output was captured for a forward over these {tuple(wanted.shape)} "
            f"ids; the prefix covers {[tuple(o.input_ids.shape) for o in self.outputs]}"
        )

    def replayer(
        self, model: AutoModelForCausalLM, rows: torch.Tensor, positions: torch.Tensor
    ) -> Replayer:
        """Build the per-forward context that writes ``rows`` at ``positions`` into the captured output.

        The write is the one :func:`activation_patch_hook` performs -- a fresh copy of the layer's
        output in its own dtype, the rows cast to that dtype and assigned at ``positions`` -- so the
        residual the next layer reads is the same bits either way.
        """

        def replay(ids: torch.Tensor) -> AbstractContextManager[None]:
            captured = self.for_ids(ids)
            patched = captured.on_device(ids.device)
            patched[:, positions.to(patched.device), :] = rows.to(patched)
            return replay_from_layer(model, captured, patched)

        return replay


@torch.no_grad()
def capture_patch_prefix(  # noqa: PLR0913 - a prefix is the run, the layer and the readout it serves
    model: AutoModelForCausalLM,
    *,
    corrupted_ids: torch.Tensor,
    corrupted_mask: torch.Tensor,
    layer: int,
    readout: GapReadout | None = None,
    readout_position: int = -1,
) -> PatchPrefix:
    """Capture layer ``layer``'s output of every un-patched forward a patch cell's readout will run.

    Runs the SAME readout functions the patched forward runs (:func:`_readout_logits` for a plain or
    single-token readout, :func:`_score_candidate` per candidate for a multi-token one) with a capture
    hook on ``layer``, so what is captured is that exact forward's layer output -- same cache
    setting, same kernels -- rather than the trunk-only capture's. Costs one forward per distinct ids
    sequence (one, or two for a multi-token readout) per (pair, layer), against a saving of layers
    ``0..layer`` on every patched forward at that layer, so it pays for itself from the second arm.
    """
    if readout is None or readout.is_single_token:
        position = readout_position if readout is None else -1
        _, output = capture_layer_output_during(
            model,
            layer,
            corrupted_ids,
            lambda: _readout_logits(model, corrupted_ids, corrupted_mask, position),
        )
        return PatchPrefix(layer=layer, outputs=(output,))

    def scored_span_output(candidate_ids: torch.Tensor) -> LayerOutput:
        _, output = capture_layer_output_during(
            model,
            layer,
            _with_continuation(corrupted_ids, candidate_ids),
            lambda: _score_candidate(model, corrupted_ids, corrupted_mask, candidate_ids),
        )
        return output

    return PatchPrefix(
        layer=layer,
        outputs=(
            scored_span_output(readout.positive_ids),
            scored_span_output(readout.negative_ids),
        ),
    )


# --------------------------------------------------------------------------------------
# Where to patch: end-anchored alignment of two matched runs of unequal length
# --------------------------------------------------------------------------------------

# Patch windows. ``grader_body`` is the end-aligned divergent middle (the manipulation itself);
# ``post_divergence`` is everything from the first divergence to the end, end-anchored;
# ``shared_prefix_control`` is the negative control -- token-for-token identical in both runs, so
# its activations are identical and its recovery MUST be exactly zero.
PATCH_WINDOW_GRADER_BODY = "grader_body"
PATCH_WINDOW_DIVERGENCE_HEAD = "divergence_head"
PATCH_WINDOW_POST_DIVERGENCE = "post_divergence"
PATCH_WINDOW_SHARED_PREFIX = "shared_prefix_control"
# The ladder windows; :func:`plan_twin_patch_ladder` documents what each one isolates.
PATCH_WINDOW_READOUT_ONLY = "readout_only"
PATCH_WINDOW_POST_DIVERGENCE_EXCL_READOUT = "post_divergence_excl_readout"
TAIL_WINDOW_PREFIX = "tail_excl_readout_"
HEAD_WINDOW_PREFIX = "divergence_head_"
DEFAULT_TAIL_WIDTHS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_HEAD_WIDTHS: tuple[int, ...] = (8, 32)
# Cap on the narrow windows. A patch over hundreds of positions transplants so much of the
# downstream residual that the real axis and the matched-norm placebo both recover ~1.0 -- measured,
# not feared: the 0.8B smoke patched 551 positions and got exactly that from both arms.
DEFAULT_MAX_NARROW_POSITIONS = 32


@dataclass(frozen=True)
class PatchWindow:
    """One patch target: paired position indices into the clean and the corrupted run.

    Two index tensors rather than one, because the twins are different lengths: the conflicting
    grader is the original plus an extra assertion. ``clean_positions[i]`` and
    ``corrupted_positions[i]`` are the same slot of the aligned region, so the clean run's
    activation there is what gets injected into the corrupted run there.
    """

    name: str
    clean_positions: torch.Tensor
    corrupted_positions: torch.Tensor

    @property
    def n_positions(self) -> int:
        """How many positions this window patches."""
        return int(self.clean_positions.numel())


@dataclass(frozen=True)
class TwinPatchPlan:
    """Where to patch a pair of unequal-length grader twins, and the bracket it was derived from.

    ``windows`` is ordered: the PRIMARY (most targeted) window first, then wider ones, then the
    shared-prefix negative control last. The readout is end-anchored -- both sequences end with the
    same shared suffix, which for chat-formatted transcripts always includes the assistant-turn
    opener -- so position ``-1`` asks the identical next-token question in both runs.
    """

    clean_ids: torch.Tensor
    corrupted_ids: torch.Tensor
    prefix_len: int
    suffix_len: int
    clean_middle_len: int
    corrupted_middle_len: int
    post_divergence_len: int
    windows: tuple[PatchWindow, ...]

    @property
    def primary_window(self) -> PatchWindow:
        """The most targeted real window: the divergent grader body when the twins have one."""
        return self.windows[0]


def _shared_prefix_len(clean_ids: torch.Tensor, corrupted_ids: torch.Tensor, common: int) -> int:
    """Count the leading positions where the two sequences carry the same token."""
    equal = (clean_ids[:common] == corrupted_ids[:common]).tolist()
    length = 0
    for is_equal in equal:
        if not is_equal:
            break
        length += 1
    return length


def _shared_suffix_len(clean_ids: torch.Tensor, corrupted_ids: torch.Tensor, *, limit: int) -> int:
    """Count the trailing positions where the two sequences agree, capped clear of the prefix."""
    length = 0
    while length < limit and clean_ids[-1 - length].item() == corrupted_ids[-1 - length].item():
        length += 1
    return length


def _end_aligned(stop: int, length: int) -> torch.Tensor:
    """Index the last ``length`` positions ending at ``stop`` (exclusive)."""
    return torch.arange(stop - length, stop)


def _drop_duplicate_windows(windows: Sequence[PatchWindow]) -> list[PatchWindow]:
    """Keep the first window of each distinct position set, so no arm is paid for twice.

    A short aligned region makes the narrow and the wide window the same set of positions; running
    both would spend three forward passes to print the same recovery under two names.
    """
    kept: list[PatchWindow] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for window in windows:
        key = (tuple(window.clean_positions.tolist()), tuple(window.corrupted_positions.tolist()))
        if key in seen:
            logger.info("patch window %s duplicates an earlier one; dropping it", window.name)
            continue
        seen.add(key)
        kept.append(window)
    return kept


def plan_twin_patch(
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    *,
    max_narrow_positions: int = DEFAULT_MAX_NARROW_POSITIONS,
) -> TwinPatchPlan:
    """Plan an activation patch between two twin token sequences of unequal length.

    The conflicting and original transcripts are byte-identical except the grader body, so a shared
    leading prefix and a shared trailing suffix bracket the divergence. Both are computed on the
    UNTRIMMED sequences (truncating to a common length would cut the tail off the longer one and
    leave the two readouts asking different questions), and the divergent middles are then aligned
    from the END, so the last position of each run is the same token and the readout is one shared
    next-token question.

    Windows come out in this order, narrowest first:

    * ``grader_body`` -- the end-aligned divergent middle, capped at ``max_narrow_positions``.
      Present only when BOTH sides have content there; on the real ILCB twins the conflicting grader
      is usually a pure INSERTION, which leaves the clean middle empty and this window absent.
    * ``divergence_head`` -- the first ``max_narrow_positions`` of the aligned region, i.e. the
      tokens immediately downstream of the manipulation. Under a pure insertion the aligned region
      is the shared suffix in both runs, so this patches identical tokens carrying different
      context: does the inserted assertion's effect travel through the residual right after it. It
      is anchored at the post-divergence boundary (``clean_len - post_divergence_len`` on the clean
      side, past the insertion on the corrupted side), so it never reaches into the bit-identical
      shared prefix. Its width is the cap, NOT the length of the divergence: over a 19-token
      insertion the head is still 32 positions, which is by design -- it measures how far downstream
      the manipulation's effect propagates, not the manipulation itself (a pure insertion leaves no
      clean-side rows to patch AT the inserted tokens, which is why ``grader_body`` is absent).
    * ``post_divergence`` -- the whole aligned region, uncapped, as an upper bound. Emitted only
      when it is wider than the narrow windows. It transplants so much of the downstream residual
      that it tends to saturate at 1.0 for the real axis AND for the matched-norm placebo (measured
      on the 0.8B smoke), which is why the narrow windows are the ones that discriminate.
    * ``shared_prefix_control`` -- the same NUMBER of positions as the primary, taken from the
      identical shared prefix. Causal attention makes both runs' activations identical there, so
      this arm can only report 0.0; it is the in-run proof that a no-op patch would be noticed.
      Patching this region was the defect: it made every recovery zero by construction.

    Windows with identical positions are collapsed (a short aligned region makes the narrow windows
    and the wide one the same set), so no arm is paid for twice.
    """
    if clean_ids.ndim != 1 or corrupted_ids.ndim != 1:
        raise ValueError("plan_twin_patch expects 1-D token id tensors")
    clean_len = int(clean_ids.shape[0])
    corrupted_len = int(corrupted_ids.shape[0])
    common = min(clean_len, corrupted_len)
    prefix_len = _shared_prefix_len(clean_ids, corrupted_ids, common)
    if prefix_len == 0:
        raise ValueError("twins share no leading prefix; not a matched pair")
    if prefix_len == clean_len == corrupted_len:
        raise ValueError("twins are identical; there is no divergent region to patch")
    post_divergence_len = common - prefix_len
    if post_divergence_len == 0:
        raise ValueError(
            f"the shorter twin ({common} tokens) is a strict prefix of the longer "
            f"({max(clean_len, corrupted_len)}), so no divergent region is end-aligned between "
            "them; these twins cannot be patched against each other"
        )
    suffix_len = _shared_suffix_len(clean_ids, corrupted_ids, limit=post_divergence_len)
    clean_middle_len = clean_len - prefix_len - suffix_len
    corrupted_middle_len = corrupted_len - prefix_len - suffix_len

    if max_narrow_positions <= 0:
        raise ValueError(f"max_narrow_positions must be positive, got {max_narrow_positions}")
    candidates: list[PatchWindow] = []
    body_len = min(clean_middle_len, corrupted_middle_len, max_narrow_positions)
    if body_len > 0:
        candidates.append(
            PatchWindow(
                name=PATCH_WINDOW_GRADER_BODY,
                clean_positions=_end_aligned(prefix_len + clean_middle_len, body_len),
                corrupted_positions=_end_aligned(prefix_len + corrupted_middle_len, body_len),
            )
        )
    head_len = min(max_narrow_positions, post_divergence_len)
    head_start_clean = clean_len - post_divergence_len
    head_start_corrupted = corrupted_len - post_divergence_len
    candidates.append(
        PatchWindow(
            name=PATCH_WINDOW_DIVERGENCE_HEAD,
            clean_positions=torch.arange(head_start_clean, head_start_clean + head_len),
            corrupted_positions=torch.arange(head_start_corrupted, head_start_corrupted + head_len),
        )
    )
    candidates.append(
        PatchWindow(
            name=PATCH_WINDOW_POST_DIVERGENCE,
            clean_positions=_end_aligned(clean_len, post_divergence_len),
            corrupted_positions=_end_aligned(corrupted_len, post_divergence_len),
        )
    )
    windows = _drop_duplicate_windows(candidates)
    control_positions = _end_aligned(prefix_len, min(windows[0].n_positions, prefix_len))
    windows.append(
        PatchWindow(
            name=PATCH_WINDOW_SHARED_PREFIX,
            clean_positions=control_positions,
            corrupted_positions=control_positions,
        )
    )
    logger.info(
        "twin patch plan: prefix=%d suffix=%d middles=%d/%d -> windows %s",
        prefix_len,
        suffix_len,
        clean_middle_len,
        corrupted_middle_len,
        {window.name: window.n_positions for window in windows},
    )
    return TwinPatchPlan(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        clean_middle_len=clean_middle_len,
        corrupted_middle_len=corrupted_middle_len,
        post_divergence_len=post_divergence_len,
        windows=tuple(windows),
    )


def _clamped_widths(widths: Sequence[int], limit: int) -> list[int]:
    """Clamp requested ladder widths to ``limit``, deduplicated, ascending, zeroes dropped.

    Two requested widths above a short region's limit clamp to the same actual width, and emitting
    both would spend a forward pass to print one number twice under two names.
    """
    return sorted({min(width, limit) for width in widths if min(width, limit) > 0})


def plan_twin_patch_ladder(  # noqa: PLR0913 - the twins plus one switch per window family
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    *,
    tail_widths: Sequence[int] = DEFAULT_TAIL_WIDTHS,
    head_widths: Sequence[int] = DEFAULT_HEAD_WIDTHS,
    include_readout_only: bool = True,
    include_post_divergence: bool = True,
    include_post_divergence_excl_readout: bool = True,
    include_prefix_control: bool = True,
) -> TwinPatchPlan:
    """Plan a width ladder over the divergent region that separates the READOUT POSITION out.

    :func:`plan_twin_patch`'s ``post_divergence`` window is end-anchored at the sequence end, so it
    CONTAINS the readout position -- and overwriting the readout position's own residual with the
    clean run's transplants the clean logits almost by definition (exactly so at the final layer,
    where nothing downstream can mix positions). A recovery near 1.0 from that window is therefore
    not evidence that the grader region is causally used; measured on the 2026-08-22 run, the
    layer-31 arms read recovery exactly 1.0 from ``post_divergence`` while ``divergence_head`` read a
    max logit shift of exactly 0.0, which is the signature of that artifact rather than of
    localisation. This planner exists to take the artifact out of the measurement:

    * ``readout_only`` -- the single readout position. The positive control: it is expected to
      recover, and how much it recovers is the size of the artifact the wide window inherits.
    * ``tail_excl_readout_{k}`` -- the ``k`` aligned positions ending immediately BEFORE the readout,
      for each requested width. Narrowing this ladder while the placebo stays flat is what would show
      a localised causal effect; a collapse as ``k`` shrinks says the wide number was accumulation.
    * ``post_divergence_excl_readout`` -- the whole aligned region minus the readout position: the
      honest version of the wide window.
    * ``post_divergence`` -- the original wide window, kept so the ladder is directly comparable to
      the earlier run rather than a separate measurement nothing lines up with.
    * ``divergence_head_{k}`` -- head-anchored at the divergence boundary, as before.
    * ``shared_prefix_control`` -- the token-identical prefix, sized to the widest kept window. A
      guaranteed no-op wherever the two runs' prefix activations agree, which is the plumbing check,
      not an informative negative control.

    Widths are clamped to what the region can hold (the tail windows to ``post_divergence_len - 1``,
    since the readout position is excluded), and duplicate position sets are collapsed.
    """
    bracket = plan_twin_patch(
        clean_ids, corrupted_ids, max_narrow_positions=max(max(head_widths, default=1), 1)
    )
    clean_len = int(clean_ids.shape[0])
    corrupted_len = int(corrupted_ids.shape[0])
    post_divergence_len = bracket.post_divergence_len
    windows: list[PatchWindow] = []
    if include_readout_only:
        windows.append(
            PatchWindow(
                name=PATCH_WINDOW_READOUT_ONLY,
                clean_positions=torch.tensor([clean_len - 1]),
                corrupted_positions=torch.tensor([corrupted_len - 1]),
            )
        )
    windows.extend(
        PatchWindow(
            name=f"{TAIL_WINDOW_PREFIX}{width}",
            clean_positions=_end_aligned(clean_len - 1, width),
            corrupted_positions=_end_aligned(corrupted_len - 1, width),
        )
        for width in _clamped_widths(tail_widths, post_divergence_len - 1)
    )
    if include_post_divergence_excl_readout and post_divergence_len > 1:
        windows.append(
            PatchWindow(
                name=PATCH_WINDOW_POST_DIVERGENCE_EXCL_READOUT,
                clean_positions=_end_aligned(clean_len - 1, post_divergence_len - 1),
                corrupted_positions=_end_aligned(corrupted_len - 1, post_divergence_len - 1),
            )
        )
    if include_post_divergence:
        windows.append(
            PatchWindow(
                name=PATCH_WINDOW_POST_DIVERGENCE,
                clean_positions=_end_aligned(clean_len, post_divergence_len),
                corrupted_positions=_end_aligned(corrupted_len, post_divergence_len),
            )
        )
    head_start_clean = clean_len - post_divergence_len
    head_start_corrupted = corrupted_len - post_divergence_len
    windows.extend(
        PatchWindow(
            name=f"{HEAD_WINDOW_PREFIX}{width}",
            clean_positions=torch.arange(head_start_clean, head_start_clean + width),
            corrupted_positions=torch.arange(head_start_corrupted, head_start_corrupted + width),
        )
        for width in _clamped_widths(head_widths, post_divergence_len)
    )
    if not windows:
        raise ValueError(
            f"the ladder is empty for a {post_divergence_len}-position aligned region under "
            f"tail_widths={list(tail_widths)} head_widths={list(head_widths)}"
        )
    kept = _drop_duplicate_windows(windows)
    if include_prefix_control:
        control_width = min(max(window.n_positions for window in kept), bracket.prefix_len)
        control_positions = _end_aligned(bracket.prefix_len, control_width)
        kept.append(
            PatchWindow(
                name=PATCH_WINDOW_SHARED_PREFIX,
                clean_positions=control_positions,
                corrupted_positions=control_positions,
            )
        )
    logger.info(
        "twin patch ladder: prefix=%d post_divergence=%d -> windows %s",
        bracket.prefix_len,
        post_divergence_len,
        {window.name: window.n_positions for window in kept},
    )
    return TwinPatchPlan(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        prefix_len=bracket.prefix_len,
        suffix_len=bracket.suffix_len,
        clean_middle_len=bracket.clean_middle_len,
        corrupted_middle_len=bracket.corrupted_middle_len,
        post_divergence_len=post_divergence_len,
        windows=tuple(kept),
    )


def window_contains_readout(window: PatchWindow, *, corrupted_len: int) -> bool:
    """Whether this window patches the readout position of the run being patched.

    Recorded per cell because it is the difference between "the region is causally used" and "the
    readout position's own residual was overwritten with the answer".
    """
    return bool((window.corrupted_positions == corrupted_len - 1).any().item())


def axis_component_replacement(
    clean_rows: torch.Tensor, corrupted_rows: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Transplant ONLY the component of ``clean - corrupted`` that lies along ``direction``.

    Full-residual patching says a region is causally used; it says nothing about whether one of our
    concept axes captures the variable being used. This writes back the rank-1 part of the difference
    along the axis and leaves the rest of the residual at its corrupted value, so a recovery here is
    attributable to that axis. Read it against :func:`axis_complement_replacement` (everything
    EXCEPT the axis) and against a random-direction placebo of the same norm.
    """
    axis = unit(direction).to(clean_rows)
    delta = clean_rows - corrupted_rows
    return corrupted_rows + (delta @ axis).unsqueeze(-1) * axis


def axis_complement_replacement(
    clean_rows: torch.Tensor, corrupted_rows: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Transplant everything EXCEPT the ``direction`` component of ``clean - corrupted``.

    The complement of :func:`axis_component_replacement`. Together they decompose the full patch: if
    the component recovers little while the complement recovers everything, the axis is not the
    causal variable however well it decodes.
    """
    axis = unit(direction).to(clean_rows)
    delta = clean_rows - corrupted_rows
    return clean_rows - (delta @ axis).unsqueeze(-1) * axis


def random_axis_replacement(
    clean_rows: torch.Tensor,
    corrupted_rows: torch.Tensor,
    generator: torch.Generator,
    *,
    match_norm_to: float | None = None,
) -> torch.Tensor:
    """Project the difference onto a RANDOM unit direction instead -- the directional placebo.

    With ``match_norm_to`` left None this is the honest "how much would an arbitrary axis have
    captured" baseline, whose perturbation is naturally far smaller than a real axis's if the axis
    carries the difference. With ``match_norm_to`` set to the real component's Frobenius norm it
    becomes the mandatory matched-norm control, which separates "this direction" from "a rank-1
    perturbation of this size". A zero-magnitude projection returns the corrupted rows unchanged
    rather than dividing by zero, mirroring the no-op placebo of a no-op real patch.
    """
    axis = unit(torch.randn(clean_rows.shape[-1], generator=generator, dtype=clean_rows.dtype))
    delta = clean_rows - corrupted_rows
    perturbation = (delta @ axis).unsqueeze(-1) * axis
    if match_norm_to is not None:
        norm = perturbation.norm()
        if norm == 0:
            return corrupted_rows
        perturbation = perturbation / norm * match_norm_to
    return corrupted_rows + perturbation


def recovery_metrics(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    patched: torch.Tensor,
    answer_token: int,
    *,
    gap_tokens: tuple[int, int] | None = None,
) -> dict[str, float | None]:
    """Three recovery readings off one readout row, plus the raw quantities behind two of them.

    ``gap_tokens`` adds a fourth: the TWO-TOKEN gap form, ``recovery_logit_gap``, computed on
    ``action_gap`` between a named positive and negative token rather than on one token's raw logit.
    It is the reading the forced-choice arm leads with and it costs no extra forward, since all three
    rows are already in hand. Its ``None`` on a zero denominator follows :func:`gap_recovery` rather
    than :func:`logit_recovery`, and the three gaps behind it are recorded so the exclusion is
    countable.

    ``recovery_logit`` is :func:`logit_recovery` -- one token's logit, which is what the earlier run
    reported and what keeps the two comparable. Its resolution is the problem: the LM head is bf16,
    so on the 2026-08-22 run every clean-vs-corrupted gap came back a multiple of 1/16 (0.375, 0.5,
    0.5625, 0.625) and every shift likewise, leaving the ratio with roughly eight distinguishable
    levels. A reported "exactly 1.0000" is then "landed on the same bf16 rung as clean", not a
    measurement of perfect recovery.

    The other two read the WHOLE readout row and are continuous at the same cost (no extra forward):

    * ``recovery_logprob_l2`` -- ``1 - ||lp_patched - lp_clean|| / ||lp_corrupted - lp_clean||`` on
      log-softmax rows, so the arbitrary additive constant in a logit row cannot enter.
    * ``recovery_kl`` -- the same fraction on ``KL(clean || .)``, the distributional version.

    Both can come out negative (the patch moved the readout further from clean than the corrupted run
    was) or above 1.0 (it overshot); neither is clipped, because a clip would hide exactly that. A
    zero denominator -- the two runs' readouts agree -- returns 0.0, as ``logit_recovery`` does.
    """
    clean_logprobs = torch.log_softmax(clean.float(), dim=-1)
    corrupted_logprobs = torch.log_softmax(corrupted.float(), dim=-1)
    patched_logprobs = torch.log_softmax(patched.float(), dim=-1)
    clean_probs = clean_logprobs.exp()
    l2_corrupted = float((corrupted_logprobs - clean_logprobs).norm().item())
    l2_patched = float((patched_logprobs - clean_logprobs).norm().item())
    kl_corrupted = float((clean_probs * (clean_logprobs - corrupted_logprobs)).sum().item())
    kl_patched = float((clean_probs * (clean_logprobs - patched_logprobs)).sum().item())
    metrics: dict[str, float | None] = {
        "recovery_logit": logit_recovery(clean, corrupted, patched, answer_token),
        "recovery_logprob_l2": 0.0 if l2_corrupted == 0.0 else 1.0 - l2_patched / l2_corrupted,
        "recovery_kl": 0.0 if kl_corrupted == 0.0 else 1.0 - kl_patched / kl_corrupted,
        "logprob_l2_corrupted_to_clean": l2_corrupted,
        "logprob_l2_patched_to_clean": l2_patched,
        "kl_clean_to_corrupted": kl_corrupted,
        "kl_clean_to_patched": kl_patched,
    }
    if gap_tokens is not None:
        positive_token, negative_token = gap_tokens
        gaps = {
            name: action_gap(row, positive_token, negative_token)
            for name, row in (("clean", clean), ("corrupted", corrupted), ("patched", patched))
        }
        metrics["recovery_logit_gap"] = gap_recovery(
            gaps["clean"], gaps["corrupted"], gaps["patched"]
        )
        metrics["logit_gap_clean"] = gaps["clean"]
        metrics["logit_gap_corrupted"] = gaps["corrupted"]
        metrics["logit_gap_patched"] = gaps["patched"]
    return metrics


def matched_norm_replacement(
    clean_rows: torch.Tensor, corrupted_rows: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Rows to inject for the placebo patch arm: corrupted plus matched-Frobenius-norm noise.

    The real patch perturbs the corrupted run by exactly ``clean - corrupted`` at those positions.
    The mandatory placebo perturbs it by a RANDOM tensor of the same Frobenius norm, so a recovery
    seen only under the real patch is attributable to the clean activations' content rather than to
    "any perturbation this large at these positions moves the readout". Raises if the two runs are
    identical there, since a zero-norm placebo would silently be a no-op.
    """
    delta_norm = (clean_rows - corrupted_rows).norm()
    if delta_norm == 0:
        raise ValueError(
            "clean and corrupted activations are identical at these positions, so a matched-norm "
            "placebo would be a no-op; patch a divergent window instead"
        )
    noise = torch.randn(clean_rows.shape, generator=generator, dtype=clean_rows.dtype)
    return corrupted_rows + noise / noise.norm() * delta_norm


# --------------------------------------------------------------------------------------
# The mandatory placebo control arm
# --------------------------------------------------------------------------------------


def steering_directions(
    direction: torch.Tensor, *, n_placebos: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Return ``{"real": direction, "placebo_k": matched-norm random}`` -- the control arms.

    Every intervention runs against these arms: the real axis and ``n_placebos`` random directions
    of equal norm. Because a random direction of equal norm perturbs the residual by the same
    magnitude, an effect seen only under ``real`` and not under the placebos is attributable to
    the direction's content rather than to "any perturbation this large". One seeded generator,
    so the arms are reproducible.
    """
    generator = torch.Generator().manual_seed(seed)
    arms = {"real": direction}
    for k in range(n_placebos):
        arms[f"placebo_{k}"] = matched_norm_random_direction(direction, generator)
    return arms


# --------------------------------------------------------------------------------------
# GPU drivers (HF path; behind functions the caller invokes)
# --------------------------------------------------------------------------------------


def run_steered_generation(  # noqa: PLR0913 - model, tokenizer, prompt and the intervention knobs
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    mode: str = "steer",
    thinking: bool = True,
    sampling: SamplingConfig = PENALTY_FREE_THINKING_SAMPLING,
) -> GenerationRecord:
    (
        """Generate under a steering (``mode="steer"``) or ablation (``mode="ablate"``) """
        """hook at ``layer``.

    The hook stays installed for the whole generation, so it fires on every decoding
    step and the intervention compounds through the response. Returns the usual
    :class:`GenerationRecord`; compare the ``real`` arm's response against the ``placebo_k``
    arms from :func:`steering_directions`.

    The sampler is a settable :class:`~reward_hacking.model_backend.SamplingConfig` rather than a
    token cap plus a greedy switch, so a steered arm and its unsteered baseline decode under
    provably the same knobs -- a difference in sampling between arms would read as an effect of the
    intervention. Steering stays on the HF path by necessity (vLLM exposes no residual-stream hooks),
    so the same knobs the HF path cannot apply are unapplied here, and the record says which.
    """
    )
    if mode == "steer":
        hook = steering_hook(direction, alpha)
    elif mode == "ablate":
        hook = ablation_hook(direction)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'steer' or 'ablate'")
    with residual_intervention(model, layer, hook):
        return generate_response(model, tokenizer, prompt, thinking=thinking, sampling=sampling)


@dataclass(frozen=True)
class PatchResult:
    """One activation-patching run: the three readout rows, the recovery fraction and the gap read.

    ``clean`` / ``corrupted`` / ``patched`` are ``[vocab]`` logits at the readout position;
    ``recovery`` is :func:`logit_recovery` on the clean answer token. A recovery near 1.0 means the
    patched positions carry the information that flips the corrupted run toward the clean answer --
    i.e. that layer/position is causally used, not merely decodable.

    The three ``*_gap`` fields and ``recovery_gap`` are populated when a :class:`GapReadout` was
    supplied, and they are the PRIMARY reading when they are: ``recovery`` is a single next-token
    logit ratio, which on chat-formatted twins under thinking mode scored the opening token of the
    reasoning trace. ``recovery_gap`` is the fraction of the clean-versus-corrupted GAP between two
    complete candidate actions that the patch recovered, which no single position's residual can
    manufacture. It is ``None`` when the two runs' gaps agree exactly (no denominator).
    """

    clean: torch.Tensor
    corrupted: torch.Tensor
    patched: torch.Tensor
    answer_token: int
    recovery: float
    clean_gap: GapRead | None = None
    corrupted_gap: GapRead | None = None
    patched_gap: GapRead | None = None
    recovery_gap: float | None = None


@dataclass(frozen=True)
class PatchBaseline:
    """The two un-patched readouts a patch is measured against: ``[vocab]`` logits each.

    Exists so a sweep of arms and windows over ONE pair of runs pays for them once. They do not
    depend on the patch -- the clean run and the corrupted run are the same two forwards whichever
    window is patched with whatever rows -- so recomputing them per arm is pure waste: at 4 windows
    times 3 arms it is 24 redundant full forwards per pair per layer against 12 useful ones.

    Also the only way to express an IDENTITY control, where the same run is both sides of the
    comparison: pass one run's logits as both ``clean`` and ``corrupted``, patch that run with its
    own activations, and ``patched`` must come back bit-identical. A pipeline that cannot produce
    that exact null cannot be trusted when it produces a positive.

    ``clean_gap`` / ``corrupted_gap`` carry the same amortisation for a :class:`GapReadout`: the two
    un-patched candidate scorings are two forwards per candidate per pair, which at 32 layers times
    ten windows times two arms is the difference between a grid that fits a box and one that does not.
    They default to absent because the single-logit readout has no gaps to carry, so a baseline built
    for a run WITH a readout has to state that it carries them -- see :attr:`carries_gaps`.
    """

    clean: torch.Tensor
    corrupted: torch.Tensor
    clean_gap: GapRead | None = None
    corrupted_gap: GapRead | None = None

    @property
    def carries_gaps(self) -> bool:
        """Whether both un-patched SCORED GAPS travelled with the rows, as a readout run needs."""
        return self.clean_gap is not None and self.corrupted_gap is not None

    @classmethod
    def from_gap_reads(cls, clean: GapRead, corrupted: GapRead) -> PatchBaseline:
        """Build a baseline from two scored gap reads, taking the readout rows off the same forwards."""
        return cls(
            clean=clean.readout_row,
            corrupted=corrupted.readout_row,
            clean_gap=clean,
            corrupted_gap=corrupted,
        )


@torch.no_grad()
def _readout_logits(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    readout_position: int,
) -> torch.Tensor:
    (
        """Run the full model (LM head included) and return """
        """``[vocab]`` logits at ``readout_position``.

    ``logits_to_keep`` narrows the LM head to that one position, so the ``[1, seq, ~248k]`` vocab
    projection that ``directions.capture_pooled_activations`` documents as the OOM to avoid is never
    materialised -- this runs three times per patch cell to read one ``[vocab]`` row. Passed as an
    index tensor rather than an int count because transformers turns an int into
    ``slice(-n, None)``, which cannot express a positive readout position, while a tensor indexes
    the sequence axis directly and handles ``-1`` as well.

    **float32 is the readout's grid, and the promotion here is load-bearing rather than tidiness.**
    On a bf16 model the row comes off the LM head in bf16, and then two arithmetically identical
    quantities stop agreeing: a logit difference taken WITHIN one row (``logit_recovery``'s
    denominator, an action-label gap) is computed in bf16 and rounds onto the bf16 grid, while the
    same difference taken ACROSS two rows is promoted to float32 and stays exact. An identity
    control then reads a non-zero gap shift -- exactly ``round_bf16(gap) - gap`` -- beside a max
    logit shift of exactly zero, which is a report about the readout's dtype dressed as a report
    about the patch. Every readout in both threads goes through this function, so promoting here
    (a lossless widening, so no value changes) puts the baselines and the patched row on one grid
    and leaves the intervention as the only thing a difference can be about.
    """
    )
    kept = torch.tensor([readout_position], device=input_ids.device)
    outputs = model(  # pyright: ignore[reportCallIssue]  # HF causal LMs are callable at runtime
        input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=kept
    )
    return outputs.logits[0, 0].detach().float().cpu()


@torch.no_grad()
def _scored_span_logits(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    n_prompt: int,
    n_continuation: int,
) -> torch.Tensor:
    """Return ``[n_continuation, vocab]`` float32 logits predicting a teacher-forced continuation.

    Row ``i`` is the next-token distribution after the prompt plus the first ``i`` continuation
    tokens, so row 0 is the prompt-only readout and the whole span is what a sequence log-prob sums
    over. ``logits_to_keep`` narrows the LM head to exactly those positions, which is what keeps a
    ~248k-wide vocab projection over thousands of positions from being materialised.

    float32 for the reason :func:`_readout_logits` documents at length: the head is bf16, and a
    difference taken within a bf16 row rounds onto the bf16 grid while the same difference taken
    across two rows does not, so an identity control reads a non-zero gap shift that is a report about
    the dtype dressed as a report about the patch.
    """
    kept = torch.arange(n_prompt - 1, n_prompt - 1 + n_continuation, device=input_ids.device)
    outputs = model(  # pyright: ignore[reportCallIssue]  # HF causal LMs are callable at runtime
        input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=kept
    )
    return outputs.logits[0].detach().float().cpu()


def _sequence_logprob(span_logits: torch.Tensor, continuation_ids: torch.Tensor) -> float:
    """Sum ``log P(continuation | prompt)`` under teacher forcing over a scored span."""
    logprobs = torch.log_softmax(span_logits, dim=-1)
    targets = continuation_ids.to(logprobs.device).long().unsqueeze(-1)
    return float(logprobs.gather(-1, targets).sum().item())


@torch.no_grad()
def read_gap(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    readout: GapReadout,
    *,
    replay: Replayer | None = None,
) -> GapRead:
    """Score both candidates against one prompt and return the positive-minus-negative gap.

    ``replay``, when given, wraps each forward in the context that replays it from a captured prefix
    (see :class:`PatchPrefix`); the forwards themselves are unchanged, so the numbers are the same
    bits with or without it.

    Anchored at the LAST PROMPT POSITION in both modes, so the caller never chooses a readout offset
    that would silently mean something else: under forced choice the prompt already ends with the
    answer prefill, and under action scoring the continuation begins there.

    **One forward for a single-token pair, two for a multi-token pair.** For one token each the gap is
    the difference of two log-softmax values at the same position, which equals the difference of the
    two raw logits exactly, so one readout row suffices and ``action_gap`` on that row IS the gap.
    Multi-token candidates need their own forward each.

    **No KV cache is shared between the two candidate forwards, deliberately.** They share their whole
    prompt, so a cache would save most of the work -- and Qwen3.5's Gated DeltaNet mutates its
    recurrent state IN PLACE (arXiv:2605.12770 deep-copies the cache per condition for exactly this
    reason). A cache reused across two conditions would let one contaminate the other and wash the
    contrast out silently, which is a far worse trade than a second full forward.
    """
    if readout.is_single_token:
        with nullcontext() if replay is None else replay(prompt_ids):
            row = _readout_logits(model, prompt_ids, prompt_mask, -1)
        positive_token = int(readout.positive_ids[0].item())
        negative_token = int(readout.negative_ids[0].item())
        row_logprobs = torch.log_softmax(row, dim=-1)
        return GapRead(
            mode=readout.mode,
            gap=action_gap(row, positive_token, negative_token),
            positive_logprob=float(row_logprobs[positive_token].item()),
            negative_logprob=float(row_logprobs[negative_token].item()),
            positive_n_tokens=1,
            negative_n_tokens=1,
            readout_row=row,
        )

    positive_span = _score_candidate(
        model, prompt_ids, prompt_mask, readout.positive_ids, replay=replay
    )
    negative_span = _score_candidate(
        model, prompt_ids, prompt_mask, readout.negative_ids, replay=replay
    )
    positive_logprob = _sequence_logprob(positive_span, readout.positive_ids)
    negative_logprob = _sequence_logprob(negative_span, readout.negative_ids)
    return GapRead(
        mode=readout.mode,
        gap=positive_logprob - negative_logprob,
        positive_logprob=positive_logprob,
        negative_logprob=negative_logprob,
        positive_n_tokens=readout.n_positive_tokens,
        negative_n_tokens=readout.n_negative_tokens,
        # Row 0 of either span is the prompt-only readout, identical for both candidates.
        readout_row=positive_span[0],
    )


def _with_continuation(prompt_ids: torch.Tensor, continuation_ids: torch.Tensor) -> torch.Tensor:
    """Return the ``[1, n_prompt + n_candidate]`` ids a scored-candidate forward runs over."""
    continuation = continuation_ids.to(prompt_ids.device).long().unsqueeze(0)
    return torch.cat([prompt_ids, continuation], dim=1)


def _score_candidate(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    continuation_ids: torch.Tensor,
    *,
    replay: Replayer | None = None,
) -> torch.Tensor:
    """Forward the prompt plus one candidate and return the ``[n_candidate, vocab]`` scored span."""
    full_ids = _with_continuation(prompt_ids, continuation_ids)
    full_mask = torch.cat([prompt_mask, torch.ones_like(full_ids[:, prompt_ids.shape[1] :])], dim=1)
    with nullcontext() if replay is None else replay(full_ids):
        return _scored_span_logits(
            model,
            full_ids,
            full_mask,
            n_prompt=int(prompt_ids.shape[1]),
            n_continuation=int(continuation_ids.numel()),
        )


def _check_patch_positions(
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    *,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
) -> None:
    """Reject position tensors that cannot describe one aligned region of both runs.

    The failure these guard against is a plausible recovery number rather than an error: a pair of
    position tensors of different lengths, or one pointing past the end of its own sequence, still
    patches *something* and still prints a fraction.
    """
    if clean.ndim != 1 or corrupted.ndim != 1:
        raise ValueError("clean_positions and corrupted_positions must be 1-D index tensors")
    if clean.numel() != corrupted.numel():
        raise ValueError(
            "clean_positions and corrupted_positions must name the same number of slots -- row i "
            f"of each is one aligned position -- got {clean.numel()} vs {corrupted.numel()}"
        )
    if clean.numel() == 0:
        raise ValueError("no positions to patch")
    for name, positions, ids in (
        ("clean", clean, clean_ids),
        ("corrupted", corrupted, corrupted_ids),
    ):
        length = int(ids.shape[1])
        indices: list[int] = positions.tolist()
        if min(indices) < 0 or max(indices) >= length:
            raise ValueError(
                f"{name}_positions fall outside the {length}-token {name} run "
                f"({min(indices)}..{max(indices)})"
            )


@torch.no_grad()
def run_activation_patch(  # noqa: PLR0913 - clean/corrupted ids, masks and the patch target are core
    model: AutoModelForCausalLM,
    *,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    clean_mask: torch.Tensor,
    corrupted_mask: torch.Tensor,
    layer: int,
    clean_positions: torch.Tensor,
    corrupted_positions: torch.Tensor,
    replacement_rows: torch.Tensor | None = None,
    readout_position: int = -1,
    answer_token: int | None = None,
    baseline: PatchBaseline | None = None,
    readout: GapReadout | None = None,
    prefix: PatchPrefix | None = None,
) -> PatchResult:
    """Cache the clean run's layer/position activation, inject it into the corrupted run, measure.

    ``prefix`` (see :class:`PatchPrefix`, built by :func:`capture_patch_prefix` once per pair and
    layer) makes the patched forward start at layer ``layer + 1``: the write is applied to the
    captured un-patched output of ``layer`` and the trunk replays from there, since layers ``0..layer``
    would compute the same residual in every arm. Same readout functions, same bits; what changes is
    that a full-layer sweep pays about half the patched-forward compute. Without it the patch runs as
    a forward hook on ``layer`` inside a full forward.

    Steps: capture the clean run's per-position residual at ``layer`` (reusing the read-only
    trunk-only capture), take the rows at ``clean_positions``, then run the corrupted ids twice --
    once plain, once with those clean rows written into ``layer`` at ``corrupted_positions`` -- and
    read the ``[vocab]`` logits at ``readout_position``. ``answer_token`` defaults to
    :func:`choose_answer_token` -- the clean run's top-k candidate the corrupted run most
    disprefers, which keeps the recovery ratio's denominator off the readout's precision floor.
    Recovery says how far the patch moved the corrupted run's logit for that token toward clean.
    Pass ``answer_token`` explicitly when comparing arms, so every arm reads the same token.

    **Two position tensors, because the twins are not the same length.** The ILCB conflicting grader
    is the original plus an extra assertion, so the corrupted side is strictly longer (median ~26
    tokens; no eligible pair tokenizes equal). ``clean_positions[i]`` and ``corrupted_positions[i]``
    are the same slot of an aligned region, which lets a caller patch the DIVERGENT grader body
    across unequal sequences. :func:`plan_twin_patch` computes that alignment from the
    twins' shared prefix and shared suffix, end-anchored so both readouts ask one shared next-token
    question. Patching a region where the two runs are token-for-token identical is a guaranteed
    no-op (causal attention gives both runs the same activations there) and reads 0.0 by
    construction -- that region is worth running only as an explicit negative control.

    ``replacement_rows`` overrides what gets written, for the matched-norm placebo arm: the recovery
    is still measured against the same clean/corrupted readouts, so the real patch and a random
    perturbation of equal Frobenius norm are directly comparable. With it set, the clean run's
    activations are not captured at all (nothing would read them).

    ``baseline`` supplies the two un-patched readouts instead of running them here, which is what
    makes a multi-arm, multi-window sweep over one pair affordable (see :class:`PatchBaseline`) and
    what lets a caller express an identity control by passing one run as both sides. Combined with
    ``readout`` it must carry the two SCORED GAPS as well, and is refused when it does not: see the
    raise below for the silent ``recovery_gap=None`` that combination used to produce.

    ``readout`` switches the measurement from one next-token logit to a scored GAP between two
    candidate continuations, and is the primary form for this substrate -- see :class:`GapReadout` for
    why the single-token readout could not carry the question. With it set, ``recovery_gap`` is the
    reading and ``recovery`` becomes a secondary diagnostic computed off the same rows at no extra
    cost; the readout rows on the result are then the last-prompt-position rows the candidate scoring
    already produced, so nothing is spent twice.

    At our scale (4B, small matched twin sets) full patching over a modest layer/position grid is
    affordable, so no attribution shortcut is used here. AtP* / attribution patching is the escape
    hatch if a position/layer sweep ever grows large enough that per-cell forwards get expensive.

    Only ``layer``'s residual is captured, since that is the only one this patch reads.
    """
    _check_patch_positions(
        clean_ids, corrupted_ids, clean=clean_positions, corrupted=corrupted_positions
    )
    if readout is not None and baseline is not None and not baseline.carries_gaps:
        raise ValueError(
            "a GapReadout was supplied with a baseline that carries no scored gaps, so "
            "recovery_gap -- the PRIMARY reading under a readout -- would come back None after the "
            "patched forward had already been paid for, and a whole sweep would record "
            "readout_mode=None with every cell looking successful. Build the baseline with "
            "PatchBaseline.from_gap_reads(clean_read, corrupted_read), which carries the two "
            "un-patched gap reads beside the readout rows they came off."
        )
    if replacement_rows is None:
        clean_by_layer = capture_positionwise_activations(
            model, clean_ids, clean_mask, layers=[layer]
        )
        rows = clean_by_layer[layer][0, clean_positions.cpu()]
    else:
        rows = replacement_rows

    clean_gap: GapRead | None = None
    corrupted_gap: GapRead | None = None
    if baseline is not None:
        clean_logits, corrupted_logits = baseline.clean, baseline.corrupted
        clean_gap, corrupted_gap = baseline.clean_gap, baseline.corrupted_gap
    elif readout is None:
        clean_logits = _readout_logits(model, clean_ids, clean_mask, readout_position)
        corrupted_logits = _readout_logits(model, corrupted_ids, corrupted_mask, readout_position)
    else:
        clean_gap = read_gap(model, clean_ids, clean_mask, readout)
        corrupted_gap = read_gap(model, corrupted_ids, corrupted_mask, readout)
        clean_logits, corrupted_logits = clean_gap.readout_row, corrupted_gap.readout_row

    resolved_answer = (
        answer_token
        if answer_token is not None
        else choose_answer_token(clean_logits, corrupted_logits)
    )
    positions = corrupted_positions.to(clean_ids.device)
    patched_gap: GapRead | None = None
    if prefix is None:
        replay: Replayer | None = None
        intervention: AbstractContextManager[None] = residual_intervention(
            model, layer, activation_patch_hook(rows, positions)
        )
    else:
        if prefix.layer != layer:
            raise ValueError(
                f"the prefix was captured at layer {prefix.layer}, not at layer {layer}"
            )
        replay = prefix.replayer(model, rows, positions)
        intervention = nullcontext()
    with intervention:
        if readout is None:
            with nullcontext() if replay is None else replay(corrupted_ids):
                patched_logits = _readout_logits(
                    model, corrupted_ids, corrupted_mask, readout_position
                )
        else:
            patched_gap = read_gap(model, corrupted_ids, corrupted_mask, readout, replay=replay)
            patched_logits = patched_gap.readout_row

    recovery = logit_recovery(clean_logits, corrupted_logits, patched_logits, resolved_answer)
    recovery_gap = (
        gap_recovery(clean_gap.gap, corrupted_gap.gap, patched_gap.gap)
        if clean_gap is not None and corrupted_gap is not None and patched_gap is not None
        else None
    )
    logger.info(
        "activation patch at layer=%d on %d positions (%s): recovery_logit=%.3f "
        "recovery_gap=%s (answer token %d)",
        layer,
        int(clean_positions.numel()),
        "clean rows" if replacement_rows is None else "supplied replacement rows",
        recovery,
        "n/a" if recovery_gap is None else f"{recovery_gap:.3f}",
        resolved_answer,
    )
    return PatchResult(
        clean=clean_logits,
        corrupted=corrupted_logits,
        patched=patched_logits,
        answer_token=resolved_answer,
        recovery=recovery,
        clean_gap=clean_gap,
        corrupted_gap=corrupted_gap,
        patched_gap=patched_gap,
        recovery_gap=recovery_gap,
    )
