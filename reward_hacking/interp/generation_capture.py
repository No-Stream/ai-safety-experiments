"""Capture residual-stream activations over the model's OWN generated tokens (generation phase).

The first interp read (Exp2, prompt-contrast) measured the eval-awareness axis at prompt-READ time
and found only a weak, fragile signal. The hypothesis this module serves: the model registers a
rigged grader more strongly WHILE REASONING about it than while first reading it, so the read should
target the tokens the model *generates*, not just the prompt.

Two read-only steps:

* **Generate** the model's own continuation for a prompt (chat-formatted exactly as the harness
  presents it), recording where the prompt ends and the response begins so every position is
  labelled. Either engine can do this: :class:`HFResponseGenerator` calls ``model.generate`` one
  prompt at a time, :class:`VLLMResponseGenerator` runs a whole batch through a vLLM engine.
* **Capture** one trunk-only forward pass over prompt+response, keeping every position (reusing
  :func:`directions.capture_positionwise_activations`), then select the response positions -- all of
  them, or a matched window -- and pool only those into one vector per prompt, ready for the same
  projection machinery Exp2 used.

Why the two steps can use different engines: capture needs forward hooks, which means a HuggingFace
model in this process, and vLLM exposes no hidden states at all. Generation needs neither. Since
attention is causal, one HF forward pass over prompt+response reproduces the residual each position
held while it was generated no matter which engine emitted the tokens -- so generation goes to the
fast engine and capture stays on HuggingFace. What travels between them is the TOKEN IDS the engine
actually read and wrote, never decoded text: re-tokenising a decoded response shifts the
prompt/response boundary at any BPE or special-token boundary the decode does not round-trip, and the
result is a well-formed capture of the wrong positions.

Decoding is configured by ONE :class:`~reward_hacking.model_backend.SamplingConfig`, and every record
carries the :class:`ResolvedSampler` saying what of it reached ``generate`` and what this path could
not apply. The four loose scalars that used to be copied out of the config made ``min_p``,
``repetition_penalty`` and ``presence_penalty`` unreachable here no matter how they were set, while
the metadata a run saved still claimed the whole config -- an artifact asserting a sampler that never
ran, which is unfalsifiable from the artifact alone.

The pooling and position-selection core touches no model and is what the offline tests exercise on
synthetic per-position activations. Nothing here loads a model at import time; the generation and
capture path is behind functions the caller invokes.

Steering is the opposite case and is why it lives elsewhere: it must *change* activations as they are
produced, so it cannot be split across engines and stays on HuggingFace generation with hooks
attached (:mod:`reward_hacking.interp.steering`).
"""

from __future__ import annotations

import gc
import importlib.util
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Protocol

import torch

from reward_hacking.interp.directions import (
    POOLERS,
    _decoder_layers,  # pyright: ignore[reportPrivateUsage]  # shared trunk-layer resolver
    _transformer_trunk,  # pyright: ignore[reportPrivateUsage]  # shared trunk resolver
    capture_positionwise_activations,
)
from reward_hacking.model_backend import (
    STOP_REASON_MAX_TOKENS,
    HFBackend,
    SamplingConfig,
    VLLMBackend,
)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

PositionSelector = Callable[["GenerationRecord"], torch.Tensor]

GEN_ENGINE_HF = HFBackend.transport
GEN_ENGINE_VLLM = VLLMBackend.transport
GEN_ENGINES: tuple[str, ...] = (GEN_ENGINE_HF, GEN_ENGINE_VLLM)
"""The two engines a generating stage can decode with, named as the backends name themselves.

Taken off the backend classes rather than spelled again, so the engine label a run records is the same
string every trace in the repo already carries under ``transport``. Neither attribute access imports
vllm: ``VLLMBackend`` defers that to its constructor.
"""


def default_gen_engine() -> str:
    """Return vLLM wherever the engine is installed, HuggingFace otherwise.

    The polarity games settled on for the same choice (``games.generation.colocate_requested``) and
    for the same measured reason: on that project's shape, generation took ~13.5 minutes per step
    through the engine against ~72 on ``transformers.generate``, so a forgotten opt-IN cost an arm
    ~69 hours. The slow path is the one an operator has to ask for. ``find_spec`` answers without
    importing vllm, so building a CLI parser stays cheap on a box without the extra.
    """
    if importlib.util.find_spec("vllm") is not None:
        return GEN_ENGINE_VLLM
    return GEN_ENGINE_HF


PENALTY_FREE_THINKING_SAMPLING = replace(
    SamplingConfig.for_thinking(thinking=True),
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
)
"""The Qwen3.5 thinking preset with every penalty at identity: the sampler these reads run under.

The temperature, top_p, top_k and token cap are READ from the harness's own config rather than
copied into literals. A copy already drifted once, pairing thinking's temperature/top_p/top_k with
the NON-thinking 4096-token cap, which truncates the very reasoning phase this module exists to pool
over.

The penalties are forced off rather than inherited. The shared preset carries
``presence_penalty=1.5`` as vLLM's anti-loop lever, and a penalty is a behavioural intervention (the
games project measured that at n=512): these reads pool the model's own reasoning in order to measure
a disposition, so a sampler that suppresses repetition changes the thing being measured. Naming all
three here means a later edit to the shared preset cannot switch a penalty on behind this module's
back. Reachable and honestly recorded is the goal; never silently on.
"""

SAMPLING_FIELD_NAMES: tuple[str, ...] = tuple(field.name for field in fields(SamplingConfig))
"""Every knob a :class:`SamplingConfig` carries, read off the dataclass rather than restated.

Read rather than listed so that a field added upstream is accounted for here by construction: the
resolved-sampler record below partitions exactly this set into applied and dropped, and a
hand-written list would leave a new knob in neither -- silently unrecorded, which is the failure
class this whole record exists to close.
"""

WHY_TRANSFORMERS_CANNOT_APPLY: dict[str, str] = {
    "presence_penalty": (
        "transformers 5.15's GenerationConfig has no presence_penalty field (verified: "
        "GenerationConfig().update(presence_penalty=1.5) reports it back as unused), so no HF "
        "generate call can apply it; only vLLM's sampler honours it"
    ),
    "seed": (
        "transformers' generate takes no per-request seed, so the HF path cannot honour one; it "
        "decodes off whatever the process's global torch RNG holds. vLLM's SamplingParams does "
        "carry a seed and the vLLM path passes it. A run on this path is UNSEEDED whatever the "
        "config asked for, which is what this entry exists to say out loud"
    ),
}
"""Knobs the HuggingFace path structurally cannot forward, each with the reason it cannot.

One entry today, and the reason is a property of the library rather than of this code, which is why
it is recorded as prose in the artifact instead of being left for a reader to infer from an absence.
"""

WHY_GREEDY_DROPS_A_KNOB = (
    "do_sample=False: greedy decoding never consults the truncation or penalty knobs, so this one "
    "was not passed to generate"
)

WHY_ENGINE_HAS_NO_SUCH_KNOB = (
    "this engine's sampler carries no counterpart for this knob, so no value of it can reach "
    "generation"
)
"""The fallback reason a knob is dropped: the engine has nowhere to put it.

Reached only if a field is added to :class:`SamplingConfig` without being accounted for in the
engine's forwarding function below -- which is the case where a bare absence would leave the new knob
recorded nowhere, so it gets a truthful sentence rather than a ``KeyError`` or a silent omission.
"""


def _why_engine_substituted(engine: str, name: str, held: object, requested: object) -> str:
    """Spell out that the engine is running a different value than the caller asked for."""
    return (
        f"the {engine} engine applies {name}={held!r} rather than the requested {requested!r}, so "
        f"the requested value never reached sampling"
    )


def _chat_format(tokenizer: AutoTokenizer, prompt: str, *, thinking: bool) -> str:
    """Wrap a prompt as one user turn via the model's chat template, ready to generate.

    The exact call ``model_backend._as_single_user_turn`` makes for the agent loop, replicated here
    (rather than importing a private helper) so the generation-phase prompt is formatted identically
    to what the harness feeds the policy. Getting this wrong would silently change the whole space
    the residual stream is captured in.
    """
    return tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


@dataclass(frozen=True, slots=True)
class DroppedKnob:
    """One requested sampling knob that never reached ``generate``, and why it could not."""

    requested: object
    why: str


@dataclass(frozen=True, slots=True)
class ResolvedSampler:
    """What the generation call actually ran with, split from what it was asked for and refused.

    ``applied`` holds only the knobs the generation call actually ran with, at the values it ran
    with; ``dropped`` names every remaining :data:`SAMPLING_FIELD_NAMES` field, carrying the value the
    caller requested and the reason it went nowhere. Together they partition the whole config, which
    is the property :func:`resolved_sampler` is built on and the tests assert: a knob is in exactly
    one of the two, never absent from both.

    ``engine`` says which generator produced it, spelled as the backends spell themselves (``hf`` /
    ``vllm``). It is load-bearing rather than decorative, because the partition differs by engine: the
    HF path cannot apply ``presence_penalty`` at all while vLLM's sampler honours it, so an artifact
    reporting ``presence_penalty`` as applied means nothing until a reader knows which engine ran. It
    also makes :func:`~reward_hacking.interp.run_harness.require_record_sampler` catch a record
    generated on one engine while the stage resolved the other.

    Why an artifact needs this at all: the interp path used to copy four scalars out of a
    :class:`SamplingConfig` and record all of them as the run's sampler, including ones it never
    forwarded. A trace then asserted a sampler that never ran -- unfalsifiable from the artifact, and
    the exact silent-success failure this repo keeps paying for. A dropped knob sitting at its
    identity value changed nothing, and is still reported: "not applied" and "applied" must never be
    confusable in a file read months later.
    """

    engine: str
    applied: dict[str, object]
    dropped: dict[str, DroppedKnob]

    def as_payload(self) -> dict[str, object]:
        """Render for a JSON artifact: the engine, the applied block, and a row per dropped knob.

        A dropped knob gets three explicit keys -- ``<name>_requested``, ``<name>_applied`` (always
        ``None``) and ``<name>_why_dropped`` -- rather than being folded into the applied block at
        its requested value, which is precisely the false claim this record exists to prevent.
        """
        payload: dict[str, object] = {"engine": self.engine, "applied": dict(self.applied)}
        for name, knob in self.dropped.items():
            payload[f"{name}_requested"] = knob.requested
            payload[f"{name}_applied"] = None
            payload[f"{name}_why_dropped"] = knob.why
        return payload


def _sampling_kwargs(sampling: SamplingConfig) -> dict[str, object]:
    """Build the sampling knobs that reach ``model.generate`` -- the one place that decides which.

    Every field transformers can honour is forwarded, ``min_p`` and ``repetition_penalty`` included
    (both verified present on transformers 5.15's ``GenerationConfig``). ``presence_penalty`` is
    deliberately absent -- see :data:`WHY_TRANSFORMERS_CANNOT_APPLY` -- and the truncation and
    penalty knobs are omitted under greedy decoding, where they have no effect.

    :func:`resolved_sampler` derives what an artifact records from this same dict, so the recorded
    sampler cannot drift from the sampler: a knob added below shows up as applied for free, and one
    removed shows up as dropped.
    """
    kwargs: dict[str, object] = {
        "max_new_tokens": sampling.max_new_tokens,
        "do_sample": sampling.do_sample,
    }
    if sampling.do_sample:
        kwargs["temperature"] = sampling.temperature
        kwargs["top_p"] = sampling.top_p
        kwargs["top_k"] = sampling.top_k
        kwargs["min_p"] = sampling.min_p
        kwargs["repetition_penalty"] = sampling.repetition_penalty
    # Outside the greedy branch: a stop string is not a sampling knob and applies either way.
    kwargs["stop"] = sampling.stop
    return kwargs


def _vllm_sampling_kwargs(sampling: SamplingConfig) -> dict[str, object]:
    """Predict what a vLLM engine built from this config will hold for every knob in it.

    The vLLM counterpart of :func:`_sampling_kwargs`, and unlike that one it covers EVERY field --
    vLLM's ``SamplingParams`` has a slot for all seven of the engine-facing knobs, ``presence_penalty``
    included, which is the whole reason this path is worth having for a run that wants Qwen's
    anti-loop lever. ``do_sample`` has no slot; greedy is spelled as temperature 0.0, which is what
    ``VLLMBackend`` passes, so the boolean is reported as the engine's own greediness.

    The three truncation knobs are predicted at their DISABLED values under greedy rather than at the
    requested ones, because vLLM rewrites them: ``SamplingParams.__post_init__`` sets ``top_p`` to
    1.0, ``top_k`` to 0 and ``min_p`` to 0.0 whenever the temperature is under its greedy epsilon
    (read off the installed vllm 0.27.1 source, and confirmed by constructing one -- ``top_p=0.9,
    top_k=20, min_p=0.05`` at temperature 0.0 come back as 1.0, 0 and 0.0). A prediction that ignored
    that would claim three settings the engine discarded.

    A prediction rather than a reading, because a stage has to resolve its expected sampler before any
    engine exists in order to check the records that come back against it.
    :meth:`VLLMResponseGenerator._require_engine_matches_prediction` then compares this against what
    the live engine reports, so a vLLM release that changes the rewrite rule fails loudly at engine
    startup instead of quietly making every artifact wrong.
    """
    greedy = not sampling.do_sample
    return {
        "max_new_tokens": sampling.max_new_tokens,
        "do_sample": sampling.do_sample,
        "temperature": 0.0 if greedy else sampling.temperature,
        "top_p": 1.0 if greedy else sampling.top_p,
        "top_k": 0 if greedy else sampling.top_k,
        "min_p": 0.0 if greedy else sampling.min_p,
        "repetition_penalty": sampling.repetition_penalty,
        "presence_penalty": sampling.presence_penalty,
        "seed": sampling.seed,
        "stop": sampling.stop,
    }


def _partition_sampling(
    sampling: SamplingConfig,
    forwarded: Mapping[str, object],
    *,
    engine: str,
    why_absent: Mapping[str, str],
) -> ResolvedSampler:
    """Split a config into what the engine runs with and what it does not, given what it forwards.

    One rule for both engines, and it is a comparison rather than a membership test: a knob counts as
    applied only when the value the engine will use EQUALS the value the caller asked for. That covers
    the two ways a knob can fail to take effect with one line each -- absent from the engine's sampler
    (transformers has no ``presence_penalty``), or present at a value the engine substituted (vLLM
    disables the truncation knobs under greedy) -- and it keeps the requested value in the artifact
    either way, which recording the substituted value as "applied" would not.
    """
    applied: dict[str, object] = {}
    dropped: dict[str, DroppedKnob] = {}
    for name in SAMPLING_FIELD_NAMES:
        requested = getattr(sampling, name)
        if name not in forwarded:
            dropped[name] = DroppedKnob(
                requested=requested, why=why_absent.get(name, WHY_ENGINE_HAS_NO_SUCH_KNOB)
            )
        elif forwarded[name] == requested:
            applied[name] = requested
        else:
            dropped[name] = DroppedKnob(
                requested=requested,
                why=_why_engine_substituted(engine, name, forwarded[name], requested),
            )
    return ResolvedSampler(engine=engine, applied=applied, dropped=dropped)


def resolved_sampler(sampling: SamplingConfig) -> ResolvedSampler:
    """Partition a config into the knobs the HF path applies and the ones it cannot, with reasons."""
    return _partition_sampling(
        sampling,
        _sampling_kwargs(sampling),
        engine=GEN_ENGINE_HF,
        why_absent={**WHY_TRANSFORMERS_CANNOT_APPLY},
    )


def vllm_resolved_sampler(sampling: SamplingConfig) -> ResolvedSampler:
    """Partition a config the way a vLLM engine built from it will treat each knob."""
    return _partition_sampling(
        sampling, _vllm_sampling_kwargs(sampling), engine=GEN_ENGINE_VLLM, why_absent={}
    )


def resolved_sampler_for(sampling: SamplingConfig, *, engine: str) -> ResolvedSampler:
    """Resolve a config against the named engine, so a stage can state its expectation up front."""
    if engine == GEN_ENGINE_HF:
        return resolved_sampler(sampling)
    if engine == GEN_ENGINE_VLLM:
        return vllm_resolved_sampler(sampling)
    raise ValueError(f"unknown generation engine {engine!r}; expected one of {list(GEN_ENGINES)}")


@dataclass(frozen=True)
class GenerationRecord:
    """One prompt, the model's continuation, and the per-position bookkeeping to target it.

    ``full_ids`` is the 1-D prompt+response token id sequence; ``is_response`` is a bool mask of the
    same length, true exactly at the positions the model generated. ``token_strings`` decodes each
    id so a caller can find the tokens where the model asserts or reasons about the contradiction.
    ``prompt_len`` is where the response starts.

    ``hit_token_cap`` says the generation ran into ``max_new_tokens`` instead of stopping on its
    own. A truncated trace is all deliberation and no post-``</think>`` answer, so it pools into a
    differently-composed vector than a complete one; without this field an asymmetric truncation
    rate between the conflicting and original twins would be invisible in the saved records.

    ``sampler`` is the :class:`ResolvedSampler` this response was produced under: what reached
    ``generate`` and what the HF path could not apply. It carries no default for the reason
    ``SamplingConfig.max_new_tokens`` carries none -- a field that is sometimes absent is a field a
    reader has to guess at, and the guess would be "the config we asked for", which is the wrong
    answer whenever a knob was dropped.
    """

    prompt_text: str
    prompt_len: int
    response_text: str
    full_ids: torch.Tensor
    token_strings: list[str]
    is_response: torch.Tensor
    hit_token_cap: bool
    sampler: ResolvedSampler

    @property
    def seq_len(self) -> int:
        """Total prompt+response length in tokens."""
        return int(self.full_ids.shape[0])

    @property
    def n_generated(self) -> int:
        """Number of tokens the model generated (response length)."""
        return int(self.is_response.sum().item())


def _generation_kwargs(tokenizer: AutoTokenizer, sampling: SamplingConfig) -> dict[str, object]:
    """Assemble the full ``generate`` kwargs: the forwarded sampling knobs plus the padding id."""
    kwargs = _sampling_kwargs(sampling)
    stop = kwargs.pop("stop", ())
    if stop:
        kwargs["stop_strings"] = list(stop)  # pyright: ignore[reportArgumentType]
        kwargs["tokenizer"] = tokenizer
    return {
        "pad_token_id": tokenizer.pad_token_id,  # pyright: ignore[reportAttributeAccessIssue]
        **kwargs,
    }


def record_from_token_ids(  # noqa: PLR0913 - a record is its ids, its boundary, its cap and its sampler
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    prompt_token_ids: Sequence[int],
    response_token_ids: Sequence[int],
    max_new_tokens: int,
    sampler: ResolvedSampler,
) -> GenerationRecord:
    """Assemble a :class:`GenerationRecord` from the ids an engine read and the ids it wrote.

    The ONE place the per-position bookkeeping is computed, for both engines. Whatever produced the
    tokens, the boundary is ``len(prompt_token_ids)``, the mask is true from there on, and the cap was
    hit when the response is at least ``max_new_tokens`` long. Having a single site is what makes that
    arithmetic worth checking once: two copies of it would be two chances for an off-by-one that
    shifts every position label by a token while every shape still matches.

    ``full_ids`` is built by concatenating the ids, never by re-tokenising decoded text, and
    ``response_text`` is decoded from ``response_token_ids`` with the same call and the same flags on
    both paths -- a vLLM engine also hands back its own detokenised string, which is discarded here so
    that a response text is comparable across engines rather than being whatever each one's
    detokeniser produced.

    ``hit_token_cap`` comes from the length arithmetic rather than from an engine's finish reason, for
    the reason the HF path had to: it must not depend on whether a checkpoint's ``eos_token_id`` is a
    scalar or a list. :class:`VLLMResponseGenerator` cross-checks it against vLLM's own label and
    warns on disagreement, which is a diagnosis rather than a second definition.
    """
    full_ids = torch.tensor([*prompt_token_ids, *response_token_ids], dtype=torch.long)
    prompt_len = len(prompt_token_ids)
    seq_len = int(full_ids.shape[0])
    return GenerationRecord(
        prompt_text=prompt,
        prompt_len=prompt_len,
        response_text=tokenizer.decode(  # pyright: ignore[reportAttributeAccessIssue]
            list(response_token_ids), skip_special_tokens=True
        ),
        full_ids=full_ids,
        token_strings=list(
            tokenizer.convert_ids_to_tokens(full_ids.tolist())  # pyright: ignore[reportAttributeAccessIssue]
        ),
        is_response=torch.arange(seq_len) >= prompt_len,
        hit_token_cap=len(response_token_ids) >= max_new_tokens,
        sampler=sampler,
    )


@torch.no_grad()
def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    thinking: bool = True,
    sampling: SamplingConfig = PENALTY_FREE_THINKING_SAMPLING,
) -> GenerationRecord:
    """Generate one continuation for ``prompt`` and record the prompt/response token boundary.

    One prompt at a time (batch of one), so there is no padding and the response is exactly
    ``full_ids[prompt_len:]`` -- the per-position bookkeeping stays trivially correct rather than
    depending on a left-padding slice. Chat-formatted with the harness convention via
    :func:`_chat_format`.

    The whole decoding configuration arrives as ONE :class:`SamplingConfig`, never as loose scalars.
    Four scalars used to be copied out of the config here, which made ``min_p``,
    ``repetition_penalty`` and ``presence_penalty`` unreachable on this path however they were set
    upstream -- three knobs a caller could ask for and never get, with nothing in the artifact
    saying so. The default sampler is :data:`PENALTY_FREE_THINKING_SAMPLING` (sampling, not greedy:
    greedy decoding is the documented loop failure mode in Qwen3.5 thinking mode; pass a config with
    ``do_sample=False`` for a reproducible smoke), and what actually reached ``generate`` is recorded
    on the returned record.

    Whether the trace ran into ``max_new_tokens`` is recorded on the record from the length
    arithmetic alone, so it does not depend on whether the checkpoint's ``eos_token_id`` is a scalar
    or a list.
    """
    chat = _chat_format(tokenizer, prompt, thinking=thinking)
    encoded = tokenizer(chat, return_tensors="pt")  # pyright: ignore[reportCallIssue]
    input_ids = encoded["input_ids"].to(model.device)  # pyright: ignore[reportAttributeAccessIssue]
    attention_mask = encoded["attention_mask"].to(  # pyright: ignore[reportAttributeAccessIssue]
        model.device  # pyright: ignore[reportAttributeAccessIssue]
    )
    prompt_len = int(input_ids.shape[1])
    outputs = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
        input_ids=input_ids,
        attention_mask=attention_mask,
        **_generation_kwargs(tokenizer, sampling),
    )
    full_ids: list[int] = outputs[0].detach().cpu().tolist()
    record = record_from_token_ids(
        tokenizer,
        prompt,
        prompt_token_ids=full_ids[:prompt_len],
        response_token_ids=full_ids[prompt_len:],
        max_new_tokens=sampling.max_new_tokens,
        sampler=resolved_sampler(sampling),
    )
    logger.info(
        "generated %d response tokens after a %d-token prompt "
        "(thinking=%s hit_token_cap=%s applied=%s dropped=%s)",
        record.n_generated,
        record.prompt_len,
        thinking,
        record.hit_token_cap,
        record.sampler.applied,
        sorted(record.sampler.dropped),
    )
    return record


def response_positions(record: GenerationRecord) -> torch.Tensor:
    """Bool mask over ``full_ids`` selecting every token the model generated."""
    return record.is_response


def matched_window_positions(
    record: GenerationRecord, *, length: int, anchor: str = "start"
) -> torch.Tensor:
    """Bool mask selecting ``length`` response tokens from the response ``start`` or ``end``.

    A fixed-width window makes projections comparable across prompts whose responses differ in
    length (the conflicting and original twins generate different amounts). ``anchor="start"`` takes
    the first ``length`` generated tokens, ``"end"`` the last ``length``. Clamped to the response
    length, and an empty response raises rather than returning an all-false mask a caller would pool
    into a divide-by-zero.
    """
    if anchor not in {"start", "end"}:
        raise ValueError(f"unknown anchor {anchor!r}; expected 'start' or 'end'")
    if length <= 0:
        raise ValueError(f"window length must be positive, got {length}")
    n_generated = record.n_generated
    if n_generated == 0:
        raise ValueError("cannot select a window from an empty response")
    take = min(length, n_generated)
    mask = torch.zeros(record.seq_len, dtype=torch.bool)
    if anchor == "start":
        mask[record.prompt_len : record.prompt_len + take] = True
    else:
        mask[record.seq_len - take :] = True
    return mask


def pool_positions(
    positionwise_by_layer: dict[int, torch.Tensor],
    position_mask: torch.Tensor,
    *,
    pooling: str = "mean",
) -> dict[int, torch.Tensor]:
    """Pool the selected positions of each layer's ``[seq, d]`` activations into one ``[d]`` vector.

    Reuses the same ``mean`` / ``last`` poolers the prompt-read path uses, treating the position
    mask as a one-row attention mask: ``mean`` averages the selected positions, ``last`` takes the
    final selected one. Raises on an empty selection, since both poolers would otherwise return a
    meaningless zero (mean) or the wrong position (last).
    """
    if pooling not in POOLERS:
        raise ValueError(f"unknown pooling {pooling!r}; expected one of {sorted(POOLERS)}")
    if int(position_mask.sum().item()) == 0:
        raise ValueError("no positions selected to pool")
    pool_fn = POOLERS[pooling]
    mask_row = position_mask.long().unsqueeze(0)
    return {
        layer: pool_fn(acts.unsqueeze(0), mask_row).squeeze(0)
        for layer, acts in positionwise_by_layer.items()
    }


@torch.no_grad()
def capture_record_activations(
    model: AutoModelForCausalLM, record: GenerationRecord
) -> dict[int, torch.Tensor]:
    """Capture per-position residuals over an already-generated record's prompt+response.

    The engine-agnostic half of the read: it touches only ``record.full_ids``, so the tokens may have
    come from ``model.generate`` or from a vLLM engine that this ``model`` never spoke to. Returns
    ``layer -> [seq, d]`` (float32, CPU). The attention mask is all-ones over the un-padded
    prompt+response, so every position is real and lines up 1:1 with ``record.full_ids`` -- which is
    also why the ids must arrive un-padded and one sequence at a time: the capture passes no
    ``position_ids``, so a padded row would shift every real token's rotary position while the pooled
    vectors still looked well-formed.
    """
    input_ids = record.full_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    positionwise = capture_positionwise_activations(model, input_ids, attention_mask)
    return {layer: acts.squeeze(0) for layer, acts in positionwise.items()}


@torch.no_grad()
def capture_record_pooled(
    model: AutoModelForCausalLM,
    record: GenerationRecord,
    position_masks: Mapping[str, torch.Tensor],
    *,
    poolings: Sequence[str],
) -> dict[tuple[str, str], dict[int, torch.Tensor]]:
    """Pool one record every (mask, pooling) way inside the forward hooks, one layer at a time.

    Returns ``(mask_name, pooling) -> layer -> [d]``, the same numbers as
    ``pool_positions(capture_record_activations(model, record), mask, pooling=...)`` for each
    combination: the hook runs the identical pooler on the identical float32 CPU copy of each
    layer's ``[1, seq, d]`` output, so the vectors are bit-for-bit what the two-step read produced.
    What changes is what is alive at once. The two-step read held every layer's output on the GPU
    until the forward finished and then every layer's float32 copy on the host -- 7.2 GB for one
    22k-token record at 32 layers -- to keep six pooled vectors per layer. Here a layer's copy is
    dropped as soon as its pooled vectors exist, so the peak is one layer's residual.

    Same contract as :func:`capture_record_activations`: un-padded ids, one sequence, no
    ``position_ids``; every mask is checked for at least one selected position before the forward
    so a bad selector fails before the GPU is spent.
    """
    unknown = [pooling for pooling in poolings if pooling not in POOLERS]
    if unknown or not poolings:
        raise ValueError(
            f"unknown poolings {unknown}; expected a non-empty subset of {sorted(POOLERS)}"
        )
    if not position_masks:
        raise ValueError("no position masks to pool over")
    mask_rows: dict[str, torch.Tensor] = {}
    for name, mask in position_masks.items():
        if int(mask.sum().item()) == 0:
            raise ValueError(f"no positions selected to pool for {name!r}")
        mask_rows[name] = mask.long().unsqueeze(0)
    input_ids = record.full_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    layers = _decoder_layers(model)
    trunk = _transformer_trunk(model)
    pooled: dict[tuple[str, str], dict[int, torch.Tensor]] = {
        (name, pooling): {} for name in mask_rows for pooling in poolings
    }

    def make_hook(idx: int) -> Callable[[object, object, object], None]:
        def hook(module: object, inputs: object, output: object) -> None:
            del module, inputs
            hidden = output[0] if isinstance(output, tuple) else output
            acts = hidden.float().cpu()  # pyright: ignore[reportAttributeAccessIssue]
            for name, mask_row in mask_rows.items():
                for pooling in poolings:
                    pooled[name, pooling][idx] = POOLERS[pooling](acts, mask_row).squeeze(0)

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        trunk(
            input_ids=input_ids.to(model.device),  # pyright: ignore[reportAttributeAccessIssue]
            attention_mask=attention_mask.to(model.device),  # pyright: ignore[reportAttributeAccessIssue]
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    return pooled


@torch.no_grad()
def capture_generation_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    thinking: bool = True,
    sampling: SamplingConfig = PENALTY_FREE_THINKING_SAMPLING,
) -> tuple[GenerationRecord, dict[int, torch.Tensor]]:
    """Generate a continuation on this model, then capture per-position residuals over it.

    The single-engine convenience path: generation and capture on the same HuggingFace model. A run
    that generates on vLLM calls :meth:`ResponseGenerator.generate_records` and
    :func:`capture_record_activations` separately instead.
    """
    record = generate_response(model, tokenizer, prompt, thinking=thinking, sampling=sampling)
    return record, capture_record_activations(model, record)


@torch.no_grad()
def capture_response_pooled(  # noqa: PLR0913 - model, tokenizer, prompts and capture knobs are core
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    *,
    pooling: str = "mean",
    thinking: bool = True,
    sampling: SamplingConfig = PENALTY_FREE_THINKING_SAMPLING,
    position_selector: PositionSelector = response_positions,
) -> tuple[list[GenerationRecord], dict[int, torch.Tensor]]:
    """Generation-phase analogue of ``capture_pooled_activations``: one pooled vector per prompt.

    For each prompt it generates, captures per-position residuals, selects positions with
    ``position_selector`` (all response tokens by default; pass a :func:`matched_window_positions`
    closure for a fixed window), and pools them. Returns the records (kept so a re-analysis
    can pick different positions without re-generating) and ``layer -> [n_prompts, d]`` pooled
    activations ready for :mod:`reward_hacking.interp.prompt_contrast`.
    """
    records: list[GenerationRecord] = []
    pooled_rows: dict[int, list[torch.Tensor]] = {}
    for prompt in prompts:
        record, positionwise = capture_generation_activations(
            model, tokenizer, prompt, thinking=thinking, sampling=sampling
        )
        pooled = pool_positions(positionwise, position_selector(record), pooling=pooling)
        records.append(record)
        for layer, vector in pooled.items():
            pooled_rows.setdefault(layer, []).append(vector)
    n_truncated = sum(record.hit_token_cap for record in records)
    logger.info(
        "pooled %d prompts; %d hit the %d-token cap (a truncated trace pools a differently "
        "composed vector, so compare this rate across groups)",
        len(records),
        n_truncated,
        sampling.max_new_tokens,
    )
    return records, {layer: torch.stack(rows, dim=0) for layer, rows in pooled_rows.items()}


# --------------------------------------------------------------------------------------
# Generation engines: the interchangeable half of the read (capture stays HuggingFace)
# --------------------------------------------------------------------------------------


class ResponseGenerator(Protocol):
    """Turns prompts into :class:`GenerationRecord`s, however it happens to decode them.

    The seam that lets a correlational stage decode on whichever engine is available while its capture
    pass stays on HuggingFace. Two members carry the contract beyond the call itself: ``engine`` names
    the route in every artifact, and ``sampler`` is the :class:`ResolvedSampler` every record this
    generator returns will carry -- so a stage can compare what came back against what it resolved
    without knowing which implementation it is holding.
    """

    engine: str
    sampler: ResolvedSampler

    def generate_records(self, prompts: Sequence[str]) -> list[GenerationRecord]:
        """Return one record per prompt, in the order the prompts were given."""
        ...

    def release(self) -> None:
        """Give back whatever VRAM this generator holds; a no-op where it holds none."""
        ...


class HFResponseGenerator:
    """Decodes with ``model.generate``, one prompt at a time: the always-available route.

    One prompt per call rather than a padded batch, which is what keeps the per-position bookkeeping
    trivially correct -- an un-padded row means the response is exactly ``full_ids[prompt_len:]``, and
    the capture pass that follows refuses padded input anyway. Slow by construction: this is the
    ~5x-slower path the vLLM generator exists to replace, kept because a box without the vllm extra,
    or a stage that must hook into generation, still has to work.
    """

    engine = GEN_ENGINE_HF

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        *,
        thinking: bool = True,
        sampling: SamplingConfig = PENALTY_FREE_THINKING_SAMPLING,
    ) -> None:
        """Wrap an already-loaded model and tokenizer as a generator over one fixed sampler."""
        self._model = model
        self._tokenizer = tokenizer
        self._thinking = thinking
        self._sampling = sampling
        self.sampler = resolved_sampler(sampling)

    def generate_records(self, prompts: Sequence[str]) -> list[GenerationRecord]:
        """Generate each prompt's continuation in turn."""
        return [
            generate_response(
                self._model,
                self._tokenizer,
                prompt,
                thinking=self._thinking,
                sampling=self._sampling,
            )
            for prompt in prompts
        ]

    def release(self) -> None:
        """Nothing to release: the model belongs to the caller, which decides when it dies."""


DEFAULT_VLLM_GPU_FRACTION = 0.35
"""The engine's share of the card, as a fraction of TOTAL VRAM rather than of what is free.

A fraction rather than a byte budget so the setting is portable across cards, which is the rule this
repo holds for every memory knob. 0.35 is what the games project's colocate runs established (
``games.generation.VLLM_COLOCATE_GPU_FRACTION``) and it leaves the rest of the card for the
HuggingFace model the capture pass runs on, which has to be resident at the same time.
"""

VLLM_PROMPT_HEADROOM_TOKENS = 24576
"""Context the engine reserves for the PROMPT, on top of the response cap, when nothing is stated.

The engine sizes its KV cache from ``max_model_len``, and these checkpoints declare a context in the
hundreds of thousands of tokens, so letting it default means demanding a KV cache for a context no
prompt here comes near -- on a card shared with the capture model that is the difference between
starting and not.

Sized from the MEASURED prompt lengths of the contrast's own stimulus pool, not from an impression of
them. Chat-formatted, the 80 prompts of the first 40 matched twin pairs run to 22328 tokens at the
longest, and four of them exceed 8192 -- which is what this constant used to be, on the stated
assumption that "the twin transcripts these stages generate from run a few thousand tokens". Most do;
the tail does not, and the tail is what a window has to cover.

The other half of that old note was worse: it claimed an overrunning prompt "is refused by the engine
loudly rather than truncated quietly". vLLM 0.27.1 does the opposite. It admits any prompt shorter
than ``max_model_len`` and then stops the request at ``num_tokens >= max_model_len`` with
``FINISHED_LENGTH_CAPPED``, so a 22328-token prompt under a 65536 cap would have been cut off around
51400 generated tokens. Nothing downstream would have said so: ``hit_token_cap`` is length arithmetic
against ``max_new_tokens`` (see :func:`record_from_token_ids`), so it reads FALSE on exactly those
clipped traces, the coverage block undercounts its truncations, and the pooled vectors are over a
context-clipped trace. :meth:`VLLMResponseGenerator.generate_records` does log a warning when the
engine's finish reason disagrees with the arithmetic, but a warning in a log is not a field in an
artifact, and truncation rate is the number these runs are read for.

So the value must stay above the longest prompt any stage generates from. Headroom costs KV cache --
~0.8 GiB per concurrently-decoding sequence at the 4B's 32 KiB per token -- which is why the engine's
share of the card and the pairs-per-call are worth checking together with it whenever either moves.
"""


class VLLMResponseGenerator:
    """Decodes a whole batch of prompts through a persistent vLLM engine: the fast route.

    Why it is worth a second engine in the process: the games project measured generation on this model
    family at ~13.5 minutes per step through vLLM against ~72 through ``transformers.generate``, and
    every one of these interp stages spends most of its wall clock generating thinking traces. Capture
    cannot follow -- vLLM exposes no hidden states -- so the engine generates and the HuggingFace model
    that stays resident beside it runs the forward pass over the ids the engine returned.

    The engine must be constructed AFTER the capture model is loaded, and that is not a preference:
    bringing up an engine first leaves ``AutoModelForCausalLM.from_pretrained`` unable to build a
    causal-LM head for the same Qwen3.5 checkpoint afterwards (measured at 0.8B, 2026-08-21 -- the
    later load dies with ``AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'``).
    :func:`~reward_hacking.interp.run_harness.capture_model_then_generator` owns that order and
    carries the full account, including the two engine-startup prerequisites (``ninja`` on ``PATH``
    and the CUDA headers, or ``VLLM_USE_FLASHINFER_SAMPLER=0``).
    """

    engine = GEN_ENGINE_VLLM

    def __init__(
        self,
        model_id: str,
        *,
        thinking: bool = True,
        sampling: SamplingConfig = PENALTY_FREE_THINKING_SAMPLING,
        gpu_fraction: float = DEFAULT_VLLM_GPU_FRACTION,
        max_model_len: int | None = None,
    ) -> None:
        """Bring up an engine for ``model_id`` under ``sampling``, and check it took the config."""
        self._backend: VLLMBackend | None = VLLMBackend(
            model_id,
            thinking=thinking,
            sampling=sampling,
            gpu_memory_utilization=gpu_fraction,
            max_model_len=(
                max_model_len
                if max_model_len is not None
                else sampling.max_new_tokens + VLLM_PROMPT_HEADROOM_TOKENS
            ),
        )
        self._sampling = sampling
        self.sampler = vllm_resolved_sampler(sampling)
        self._require_engine_matches_prediction()

    @property
    def _engine(self) -> VLLMBackend:
        """The live engine, refusing rather than crashing obscurely once it has been released."""
        if self._backend is None:
            raise RuntimeError(
                "this vLLM generator has been released and holds no engine; construct a new one "
                "rather than generating through a torn-down one"
            )
        return self._backend

    def _require_engine_matches_prediction(self) -> None:
        """Refuse an engine that will not sample the way :func:`vllm_resolved_sampler` says it will.

        The artifact this run writes says what its sampler was, and that sentence is derived from a
        PREDICTION of vLLM's behaviour (``_vllm_sampling_kwargs``) rather than from the engine. This is
        where the prediction is held against the engine's own report, so the two ways it could be wrong
        both surface here instead of in the artifact: a config that never reached the constructor, and
        a vLLM release that changed how it rewrites what it is handed.
        """
        predicted = _vllm_sampling_kwargs(self._sampling)
        applied = self._engine.applied_sampling()
        if applied == predicted:
            return
        raise RuntimeError(
            "the vLLM engine will not sample the way this run is about to claim it did. Predicted "
            f"{dict(sorted(predicted.items()))}; the engine holds "
            f"{dict(sorted(applied.items()))}. Either the SamplingConfig never reached VLLMBackend, "
            "or the installed vllm rewrites SamplingParams differently than _vllm_sampling_kwargs "
            "predicts (as of 0.27.1 it resets top_p/top_k/min_p under greedy). Fix whichever it is "
            "rather than recording a sampler that did not run."
        )

    def generate_records(self, prompts: Sequence[str]) -> list[GenerationRecord]:
        """Generate every prompt in ONE engine call and build a record per reply.

        Batching is the point: the engine's throughput comes from having many sequences in flight, so a
        caller wanting the speedup hands it as many prompts at once as its memory and its deadline
        allow. Records come back in prompt order (vLLM sorts its outputs by request id, and
        ``VLLMBackend.generate_tokenized`` checks each reply against the prompt it was asked for), so a
        caller can zip them against its own inputs.
        """
        replies = self._engine.generate_tokenized(list(prompts))
        records: list[GenerationRecord] = []
        for prompt, reply in zip(prompts, replies, strict=True):
            record = record_from_token_ids(
                self._engine.tokenizer,
                prompt,
                prompt_token_ids=reply.prompt_token_ids,
                response_token_ids=reply.response_token_ids,
                max_new_tokens=self._sampling.max_new_tokens,
                sampler=self.sampler,
            )
            engine_says_capped = reply.completion.stop_reason == STOP_REASON_MAX_TOKENS
            if engine_says_capped != record.hit_token_cap:
                logger.warning(
                    "vLLM labelled this reply %s while the length arithmetic reads "
                    "hit_token_cap=%s (%d response tokens against a %d cap); the arithmetic is what "
                    "the record carries, so a run with many of these is measuring truncation wrong",
                    reply.completion.stop_reason,
                    record.hit_token_cap,
                    record.n_generated,
                    self._sampling.max_new_tokens,
                )
            records.append(record)
        logger.info(
            "vLLM generated %d responses (%d response tokens total, %d hit the %d-token cap)",
            len(records),
            sum(record.n_generated for record in records),
            sum(record.hit_token_cap for record in records),
            self._sampling.max_new_tokens,
        )
        return records

    def release(self) -> None:
        """Drop the engine and report whether its VRAM actually came back.

        Worth doing where a stage generates once and then needs the card for something big (the lens
        fit sizes its ``dim_batch`` off free VRAM, so an engine still sitting on a third of the card
        costs it a tier for a corpus it has already produced). Reported rather than asserted, because
        a release that does not free is a slower stage rather than a wrong one -- every consumer here
        reads free VRAM at the moment it decides -- and because an unlogged teardown is exactly the
        kind of reassuring no-op this repo has been bitten by. Using the generator afterwards raises.
        """
        before = _free_vram_gib()
        self._backend = None  # pyright: ignore[reportAttributeAccessIssue]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = _free_vram_gib()
        logger.info(
            "released the vLLM engine; free VRAM %.1f -> %.1f GiB (%.1f GiB returned)",
            before,
            after,
            after - before,
        )


def _free_vram_gib() -> float:
    """Free VRAM in GiB, or 0.0 off a GPU, for logging a release rather than for sizing anything."""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)
