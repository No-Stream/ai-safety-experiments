"""Model-inference backend for reward-hacking exploration episodes.

Every exploration lead that samples a policy (surfacing an affordance, reading what a situation
is graded on, probing a loose check) needs to turn a prompt into a completion. That is the whole
job of this module: one ``Backend`` protocol, a HuggingFace implementation for real runs, an AWS
Bedrock implementation for hosted models we cannot run locally, and a ``MockBackend`` so the rest
of the harness can be tested and driven offline without a GPU or a model download.

``torch`` and ``transformers`` are core deps, imported at module top. ``vllm`` and ``boto3`` are
optional extras and are imported lazily inside their backends via ``importlib`` so this module
(and the whole offline path) loads with either absent. ``urllib3`` sits with the core imports
despite being botocore's transport, because it arrives with ``requests`` under ``transformers``
whether or not the ``bedrock`` extra is installed, and because the exceptions taken from it are the
ones that decide whether a batch survives a dead call -- resolving those lazily would put the
guarantee behind an import that is allowed to fail. The HuggingFace load recipe encodes
footguns verified against the Qwen3.5 checkpoints on this box: pass ``dtype=`` (TRL/loaders
silently ignore ``torch_dtype=`` and fall back to float32), use a bare ``AutoTokenizer`` rather
than ``AutoProcessor`` (the 4B checkpoint is a vision-language model and ``AutoProcessor``
switches to the VL path), and thread ``enable_thinking`` into the chat template because Qwen3.5
emits ``<think>`` blocks by default. The Bedrock section carries its own set, all of them silent
failures rather than loud ones; they are documented on ``BedrockBackend`` and its helpers.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import email.utils
import importlib
import logging
import math
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from bisect import bisect_right
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Generator,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from openai.types.chat import ChatCompletionChunk
    from peft import PeftModel

import httpx2
import torch
import urllib3.exceptions
from httpx2 import Timeout
from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Decoding knobs shared by the local backends, with thinking-aware presets.

    The right values depend on whether the chat template is in thinking mode, so the correct
    presets live on :meth:`for_thinking` rather than in the field defaults. Running Qwen3.5's
    non-thinking preset (temperature 0.7, top_p 0.8) under ``enable_thinking=True`` drives the
    model into endless repetition inside the ``<think>`` block: it never emits ``</think>``, never
    reaches an answer, and burns the whole token budget. That is Qwen's documented failure mode for
    mis-set thinking sampling, and it was the live bug this class is fixing. The remaining field
    defaults are a baseline for code that constructs a config directly; every real run goes through
    :meth:`for_thinking` (both local backends default to it).

    ``presence_penalty`` is the one field the two local backends treat differently, and the split
    is not cosmetic. Qwen's thinking preset uses ``presence_penalty=1.5`` as its anti-loop lever,
    but transformers 5.15's ``GenerationConfig`` has no such field, so :class:`HFBackend` cannot
    apply it and never passes it. The HF path reaches the same goal a different way: a higher
    temperature (itself more loop-resistant) plus ``repetition_penalty`` as the transformers-native
    tunable if loops persist. Only :class:`VLLMBackend` passes ``presence_penalty``, because vLLM's
    ``SamplingParams`` honours it. Flip ``do_sample`` off for greedy, reproducible smoke runs.

    ``max_new_tokens`` is the one field with no default, and the only one whose wrong value is
    invisible in the output it produces: a reply cut off at the cap reads as a model that stopped
    early. It carried 1,024 as a baseline for code building a config by hand, which is a cap nobody
    chose applying to a run nobody checked. Every caller already states it -- both presets below,
    the games training sampler, an explicit CLI flag -- so requiring it costs nothing and removes
    the only route by which a run could inherit an unchosen budget.

    ``seed`` is the per-request generation seed, and ``None`` -- unseeded -- is the honest default
    rather than an oversight. It exists because this class had no seed field at all, so an artifact
    recording ``seed: 0`` was recording a PLACEBO seed and nothing about generation, and two
    supposedly-matched contrast runs turned out to be independent samples: the same fixed 4-prompt
    preflight generated 16,351 tokens in one and 13,149 in the other, and the runaway generation
    landed in a different chunk each time. vLLM honours it per request; transformers' ``generate``
    has no such argument, so on the HF path it is reported as a DROPPED knob carrying that reason
    rather than silently ignored. Either way the record now states whether the run was seeded, which
    is the property that was missing.

    ``stop`` halts generation at the first occurrence of any of its strings, and defaults to none
    -- the single-completion probes want the whole reply. The agent harness sets it to the
    run-block close so a turn ends where the environment's reply begins, instead of the model
    writing that reply itself. Both local backends return the completion WITH the stop text (vLLM
    via ``include_stop_str_in_output``, transformers by construction), and both label the halt
    :data:`STOP_REASON_STOP_SEQUENCE`, so the harness's block parser sees the complete block and a
    stopped turn is distinguishable from one that ran out of budget.
    """

    max_new_tokens: int
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    stop: tuple[str, ...] = ()
    seed: int | None = None

    @classmethod
    def for_thinking(cls, *, thinking: bool) -> SamplingConfig:
        """Return the Qwen3.5 model-card sampling preset for thinking or non-thinking mode.

        Thinking mode takes the thinking-GENERAL preset (temperature 1.0), not the precise-coding
        one: coding's anti-loop lever is ``presence_penalty``, which transformers cannot apply, and
        the higher temperature is more loop-resistant anyway. ``presence_penalty=1.5`` rides along
        on the thinking preset for the vLLM path; the HF path ignores it (see the class docstring).
        The thinking cap is 32768 tokens, Qwen's recommended output length; non-thinking uses a
        tighter 4096.
        """
        if thinking:
            return cls(
                max_new_tokens=32768,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=1.5,
            )
        return cls(
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
        )


@runtime_checkable
class Backend(Protocol):
    """The inference seam every exploration track imports: prompt strings in, completions out.

    ``transport`` names how the completions were produced and is recorded in every trace beside
    ``model_id``. Two backends can now answer for the same hosted model by different routes -- the
    live Converse API and the batch inference service -- so a trace that says only which model ran
    cannot be told from one that took the other route, and an artefact of the route would read as a
    property of the model. It is a declared attribute rather than an optional label a caller
    remembers to pass, because a label with a default is a label that is sometimes wrong.
    """

    model_id: str
    transport: str

    def generate(self, prompts: list[str]) -> list[str]:
        """Batched generation. Returns one completion per prompt, prompt text stripped."""
        ...


TRUST_REMOTE_CODE_WITHHELD_REASON = (
    "the inference backends load tokenizers WITHOUT trust_remote_code, deliberately: the flag "
    "executes checkpoint-supplied Python on the HOST at construction time, outside the episode "
    "jail this whole package exists to keep model-influenced code inside (measured 2026-08-24 -- a "
    "checkpoint whose auto_map names a module raises that module's code with the flag, and does not "
    "run it without). Withholding it does not risk silent prompt divergence from the training path, "
    "which does grant it: transformers RAISES for a checkpoint that requires remote code rather "
    "than substituting a different tokenizer, so the mismatch announces itself at load time instead "
    "of producing quietly different prompt text. Qwen3.5 needs no remote code at any size, so today "
    "the two paths resolve identically. Grant it here only with a threat-model argument, not to "
    "silence a load error."
)
"""Why the local backends withhold ``trust_remote_code`` where the training path grants it.

A named constant rather than a comment because ``test_model_backend_offline`` asserts against this
asymmetry: a future edit that "fixes" it by granting the flag turns that test red and has to read
this reasoning first.
"""


def _pick_dtype() -> torch.dtype:
    """Choose bf16 on a supporting GPU, else use float32.

    Never hardcode an assumption about the card.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _as_single_user_turn(tokenizer: PreTrainedTokenizerBase, prompt: str, *, thinking: bool) -> str:
    """Wrap a plain prompt as one user turn via the model's chat template, ready for generation."""
    return cast(
        "str",
        tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        ),
    )


STOP_REASON_END_TURN = "end_turn"
STOP_REASON_MAX_TOKENS = "max_tokens"
"""The two stop reasons every transport here reports, spelled the way Converse spells them.

Converse's own vocabulary rather than a local invention, because the field is pooled across
transports: ``recoverybench.grading`` already keys ``truncated`` off the literal ``"max_tokens"``,
so a local backend reporting ``"length"`` (vLLM's word) would be read as "stopped for some other
reason" -- silently, in the one field that tells a clipped reply from a reply that declined.
"""

STOP_REASON_STOP_SEQUENCE = "stop_sequence"
"""The reply halted on a caller-configured stop sequence, spelled the way Converse spells it.

Converse's own vocabulary for the same reason the two constants above use it: the field is pooled
across transports, so vLLM's matched-stop-string case and transformers' ``stop_strings`` halt both
travel under the label a Bedrock reply would carry natively. A caller reading it must know the
transports disagree about the text: vLLM (``include_stop_str_in_output=True``) and the HF path
return the completion WITH the stop text, while Converse strips it -- the agent harness restores it
before parsing (see ``harness/loop.py``), which is the one consumer that needs the closing tag.
"""

STOP_REASON_DEADLINE_EXCEEDED = "deadline_exceeded"
"""The streaming Converse path abandoned this call at its wall-clock deadline, mid-generation.

Deliberately *not* one of Bedrock's own nine ``StopReason`` values, because the model did not stop
-- we stopped reading. Whatever text and reasoning had streamed in by then is kept rather than
thrown away, so the record is partial rather than absent, and this label is the only thing marking
it partial: the text field of a reply abandoned after its answer arrived is indistinguishable from
a complete one. It reads as ``Outcome.NO_ANSWER_UNKNOWN_STOP`` in
``recoverybench.grading._outcome_without_an_answer`` (an unrecognised stop reason gets its own
bucket there rather than joining the declines), which is the correct reading: we do not know
whether the policy was going to answer.
"""

CALL_FAILED_STOP_REASON_PREFIX = "call_failed:"
"""Prefix marking a record for a call that never returned, followed by the failure's class name.

The failure reason travels in the record rather than only in the log because the record is the
artifact an analysis reads months later. A call that timed out has to stay *counted* -- a
denominator that quietly shrinks is worse than a visible error, and on hard arms the losses
correlate with reasoning length, which is often the variable under manipulation, so dropping them
biases the survivors rather than merely thinning them.
"""


class IncompleteConverseStreamError(RuntimeError):
    """A ConverseStream ran out of events before ``messageStop``, delivering no exception either.

    The third spelling of a dead reply: a connection dropped in a way botocore surfaces as neither
    a raised transport error nor an error event -- the iterator just ends, and
    :func:`accumulate_converse_stream` is what notices. Observed live at about 1 call in 290 on
    Opus 5 (2026-08-24, zero content blocks delivered), never on the same round's Sol, GPT-OSS or
    Luna passes, so it is the expected occasionally-never-returns failure the per-call isolation
    list exists for. It gets its own class rather than the bare ``RuntimeError`` it used to be
    precisely so that list can name it: naming ``RuntimeError`` itself would absorb every genuine
    accumulator bug into empty records. Still deliberately never labelled a deadline abandonment --
    a wrong label on a partial record is worse than a counted failure.

    Subclasses ``RuntimeError`` so anything outside this module that matched the old raise still
    does.
    """


def call_failed_stop_reason(error: BaseException) -> str:
    """Render the stop reason recorded for a call that failed with ``error``.

    A modelled service error (botocore's ``EventStreamError``, a ``ClientError`` subclass) also
    names the service's own error code, because every one of them arrives under that single class
    name and the bare name would pool a throttle with an internal server error -- the code is the
    part an analysis of failure rates actually reads. The transport classes carry no ``response``
    and keep the bare class name, which is what every record already on disk holds.
    """
    name = type(error).__name__
    code = client_error_code(error)
    if code:
        return f"{CALL_FAILED_STOP_REASON_PREFIX}{name}:{code}"
    return f"{CALL_FAILED_STOP_REASON_PREFIX}{name}"


def _error_response(error: BaseException) -> Mapping[str, Any] | None:
    """Return the botocore ``response`` dict a modelled service error carries, else None.

    ``getattr`` rather than an ``isinstance`` against ``ClientError`` because botocore is an optional
    extra this module must not import at top level; the shape check on the attribute is what keeps a
    stand-in exception in the tests honest about carrying the real constructor's payload.
    """
    response = getattr(error, "response", None)
    return response if isinstance(response, dict) else None


def client_error_code(error: BaseException) -> str | None:
    """Return the service's own error code on a modelled botocore error, or None if there is none."""
    response = _error_response(error)
    if response is None:
        return None
    code = response.get("Error", {}).get("Code")
    return str(code) if code else None


def _attempts_from_metadata(metadata: Mapping[str, Any] | None) -> int | None:
    """Turn botocore's ``ResponseMetadata.RetryAttempts`` (retries, not attempts) into attempts.

    botocore records the number of *retries* on the parsed response and on a ``ClientError``'s
    ``response``, so one request that was accepted first time reads ``RetryAttempts: 0`` and is one
    attempt. ``None`` when the metadata is absent -- a stand-in client, or a transport failure that
    never produced a response -- rather than a guessed 1.
    """
    if metadata is None:
        return None
    retries = metadata.get("RetryAttempts")
    return None if retries is None else int(retries) + 1


def is_incomplete_stop_reason(stop_reason: str | None) -> bool:
    """Whether this stop reason marks a record no analysis may count as a finished reply.

    One predicate rather than a literal comparison at every call site, because there are two ways a
    record can be partial (the deadline and a failed call) and the second carries the failure's
    class name, so ``== "call_failed"`` would miss every one of them.

    ``None`` reads False, the same convention the agent harness's ``hit_output_cap`` uses: it means
    the transport could not say, which is not evidence of incompleteness. So the count this feeds is
    a floor on how many records are partial, never a rate.
    """
    if stop_reason is None:
        return False
    return stop_reason == STOP_REASON_DEADLINE_EXCEEDED or stop_reason.startswith(
        CALL_FAILED_STOP_REASON_PREFIX
    )


END_OF_TURN_TOKENS: tuple[str, ...] = ("<|im_end|>", "<|endoftext|>")
"""The two tokens a Qwen3.5-family assistant turn can end on, pinned by NAME on every served unit.

Why by name and on every unit: the released TMAX checkpoints ship a minimal generation_config.json
declaring a single eos id (``<|endoftext|>``, 248044) while the upstream Qwen3.5 repos ship no
generation_config.json at all, and vLLM merges a checkpoint's generation_config eos ids into every
request's stop set (``SamplingParams.update_from_generation_config``, called per request by the v1
input processor). Left to that default, a TMAX unit stops on {248044, 248046} while its base stops
on the tokenizer's eos alone -- two units of one screen halting on different rules. Resolving both
names through the SERVED tokenizer (:func:`end_of_turn_token_ids`) and passing the ids explicitly
makes every unit stop on the same set, and puts that set in the run record where a resume gate can
compare it.
"""


def end_of_turn_token_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, ...]:
    """Resolve :data:`END_OF_TURN_TOKENS` through a tokenizer, refusing one that lacks either.

    Refused rather than partially applied: a tokenizer that maps one of the names to nothing (or to
    its unknown-token id) is not a Qwen3.5-family tokenizer, and pinning half the set would leave
    the unit stopping on a rule this module cannot describe.
    """
    ids: list[int] = []
    unk = tokenizer.unk_token_id  # pyright: ignore[reportAttributeAccessIssue]
    for token in END_OF_TURN_TOKENS:
        resolved = tokenizer.convert_tokens_to_ids(token)  # pyright: ignore[reportAttributeAccessIssue]
        if not isinstance(resolved, int) or (unk is not None and resolved == unk):
            raise ValueError(
                f"tokenizer {type(tokenizer).__name__} does not know {token!r} (resolved to "
                f"{resolved!r}), so the end-of-turn stop set cannot be pinned on it"
            )
        ids.append(resolved)
    return tuple(sorted(set(ids)))


def _terminator_ids(tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    """Collect the token ids that end a generated row: end-of-sequence, and the padding id.

    ``eos_token_id`` is a scalar on most checkpoints and a list on some (Qwen3.5 ships several
    end-of-turn tokens), so both shapes are read; assuming the scalar would leave a row that ended
    on the second spelling looking like a row that ran to the cap. The padding id joins them
    because a finished row in a batch is filled out to the batch's length with it, and those tokens
    are the batch's rather than the reply's.
    """
    ids: set[int] = set()
    for candidate in (tokenizer.eos_token_id, tokenizer.pad_token_id):  # pyright: ignore[reportAttributeAccessIssue]
        if candidate is None:
            continue
        if isinstance(candidate, int):
            ids.add(candidate)
        else:
            ids.update(int(token) for token in candidate)
    return frozenset(ids)


def _generated_span(row: Sequence[int], terminators: Collection[int]) -> tuple[int, str]:
    """Return how many tokens one generated row spent, and why it stopped.

    ``row`` is the generated span alone, with the prompt already sliced off. A row that terminated
    carries the tokenizer's end-of-sequence token (or the padding the batch was filled out to,
    which for these checkpoints is the same id), and the tokens after it belong to the batch rather
    than to this reply, so the count stops there and includes the terminator itself.

    A row carrying no terminator ran to ``max_new_tokens``. That inference is sound rather than
    heuristic because of how batched generation ends: ``transformers`` keeps every row in the
    sampling loop until all of them are finished or the cap is reached, so a batch that returned
    early has a terminator in every row. Length arithmetic alone cannot say this -- every row in a
    batch comes back the same length, whether it stopped on its first token or its last.
    """
    for index, token in enumerate(row):
        if token in terminators:
            return index + 1, STOP_REASON_END_TURN
    return len(row), STOP_REASON_MAX_TOKENS


class HFBackend:
    """HuggingFace batched-generation backend. Runs on GPU or CPU; the offline-inference path.

    Batched decoding uses left padding, so every sequence's completion starts at the same column
    and the prompt strips off with a single slice.
    """

    transport = "hf"

    def __init__(  # noqa: PLR0913 - flat construction knobs, each one a recorded load decision
        self,
        model_id: str,
        *,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        thinking: bool = False,
        sampling: SamplingConfig | None = None,
        model_path: str | Path | None = None,
        stop_token_ids: Sequence[int] | None = None,
        model: PreTrainedModel | PeftModel | None = None,
    ) -> None:
        """Initialize the device-aware HuggingFace model and tokenizer.

        ``model_path`` is where the weights and tokenizer are actually read from when that is not
        ``model_id`` -- a verified local snapshot of a hub revision, served under the label
        ``model_id`` carries into every record (see :mod:`games.eval_model`'s full-weights rung).
        Unset, ``model_id`` is loaded as it always was.

        ``stop_token_ids`` pins the ids a completion may end on, passed to ``generate`` as its
        ``eos_token_id`` in place of whatever the checkpoint's generation_config declares (see
        :data:`END_OF_TURN_TOKENS`). Unset, the checkpoint's own defaults apply as before.
        """
        self.model_id = model_id
        self.model_path = None if model_path is None else str(model_path)
        self.stop_token_ids: tuple[int, ...] = tuple(stop_token_ids or ())
        self.thinking = thinking
        # An unset config follows the thinking flag: thinking mode loops inside <think> otherwise.
        self.sampling = sampling or SamplingConfig.for_thinking(thinking=self.thinking)
        resolved_dtype = dtype or _pick_dtype()
        device_map = device or ("auto" if torch.cuda.is_available() else "cpu")
        load_from = self.model_path or model_id

        # No trust_remote_code: see TRUST_REMOTE_CODE_WITHHELD_REASON.
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            load_from, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._terminator_ids = _terminator_ids(self._tokenizer) | frozenset(self.stop_token_ids)
        self._model: PreTrainedModel = (
            AutoModelForCausalLM.from_pretrained(
                load_from, dtype=resolved_dtype, device_map=device_map
            )
            if model is None
            else cast("PreTrainedModel", model)
        )
        self._model.eval()
        logger.info(
            "HFBackend loaded %s from %s dtype=%s device=%s thinking=%s",
            model_id,
            load_from,
            resolved_dtype,
            self._model.device,
            thinking,
        )

    @property
    def model(self) -> PreTrainedModel:
        """The live model used for generation and residual-stream interventions."""
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer used to render this backend's prompts."""
        return self._tokenizer

    def assert_serves_weights(self, snapshot_dir: Path) -> None:
        """Raise unless the loaded model's config says it was read from ``snapshot_dir``.

        transformers stamps ``name_or_path`` with the exact path ``from_pretrained`` was given, so
        this ties the live model object to the directory whose tensor files were hashed and
        matched to their revision -- the last link between "these bytes are that revision's" and
        "this process is serving these bytes".
        """
        loaded_from = Path(str(self._model.config.name_or_path)).resolve()
        if loaded_from != snapshot_dir.resolve():
            raise RuntimeError(
                f"HFBackend loaded its weights from {loaded_from}, not from the verified snapshot "
                f"{snapshot_dir}; the served model is not the one whose identity was checked"
            )

    def _generation_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_new_tokens": self.sampling.max_new_tokens,
            "do_sample": self.sampling.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self.stop_token_ids:
            kwargs["eos_token_id"] = list(self.stop_token_ids)
        # temperature/top_p/top_k/min_p/repetition_penalty are meaningless under greedy decoding.
        if self.sampling.do_sample:
            kwargs["temperature"] = self.sampling.temperature
            kwargs["top_p"] = self.sampling.top_p
            kwargs["top_k"] = self.sampling.top_k
            kwargs["min_p"] = self.sampling.min_p
            kwargs["repetition_penalty"] = self.sampling.repetition_penalty
        if self.sampling.stop:
            # transformers' StopStringCriteria needs the tokenizer or generate() refuses to start.
            kwargs["stop_strings"] = list(self.sampling.stop)
            kwargs["tokenizer"] = self._tokenizer
        return kwargs

    @torch.no_grad()
    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Generate one completion per prompt, saying for each how long it ran and why it stopped.

        Local generation is where the stop reason was missing entirely: ``transformers`` returns
        token ids and no label, so this backend satisfied only ``Backend``, ``generate_raw`` took
        its bare-strings branch, and every turn a local run recorded carried ``stop_reason: null``.
        A cap hit is then indistinguishable from a policy that said nothing -- and the local presets
        are where a low cap still lives (4,096 tokens without ``--thinking``).

        ``reasoning`` is empty rather than split out: a local thinking trace comes back inside the
        completion between ``<think>`` tags, and every reader of these traces treats the whole
        string as the completion. Splitting it here would change what the text field means. The
        input count comes from the attention mask, so a short prompt is not charged for the padding
        that left-aligned it behind a longer one in the same batch.

        A stop-string halt is labelled off the decoded text rather than the token ids, because it
        has to be: the criteria ends the row on an ordinary token, so the id-level span reads a
        sole stopped row as a cap hit and a stopped row padded out inside a batch as ``end_turn``,
        both wrong. With the criteria active generation cannot continue past the first occurrence,
        so the stop text appearing in the reply IS the halt. The text keeps the stop string (plus
        up to one token of overshoot, since a token boundary need not land on the string's end),
        which is what lets the harness's block parser see the complete ``<run>`` block.
        """
        chats = [_as_single_user_turn(self._tokenizer, p, thinking=self.thinking) for p in prompts]
        inputs = self._tokenizer(chats, return_tensors="pt", padding=True).to(self._model.device)
        generate = cast("Callable[..., torch.Tensor]", self._model.generate)  # pyright: ignore[reportAttributeAccessIssue]
        outputs = generate(**inputs, **self._generation_kwargs())
        prompt_len = int(inputs["input_ids"].shape[1])
        prompt_tokens: list[int] = inputs["attention_mask"].sum(dim=1).tolist()
        completions: list[BedrockCompletion] = []
        for output, input_tokens in zip(outputs, prompt_tokens, strict=True):
            generated: list[int] = output[prompt_len:].tolist()
            output_tokens, stop_reason = _generated_span(generated, self._terminator_ids)
            text = cast("str", self._tokenizer.decode(generated, skip_special_tokens=True))
            # A stop-string halt leaves no terminator token, so the span would mislabel it above.
            if self.sampling.stop and any(stop in text for stop in self.sampling.stop):
                stop_reason = STOP_REASON_STOP_SEQUENCE
            completions.append(
                BedrockCompletion(
                    text=text,
                    reasoning="",
                    usage=TokenUsage(input_tokens=int(input_tokens), output_tokens=output_tokens),
                    stop_reason=stop_reason,
                )
            )
        return completions

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one completion for each prompt using batched decoding.

        Delegates rather than decoding a second time, so the text a caller reading only ``Backend``
        gets is by construction the text the detailed record carries.
        """
        return [completion.text for completion in self.generate_detailed(prompts)]


VLLM_LORA_ADAPTER_NAME = "eval-adapter"
VLLM_LORA_ADAPTER_ID = 1
"""Identity of the single un-merged adapter a backend serves; vLLM keys its slots on both.

One adapter per engine, because an eval evaluates one checkpoint at a time. A backend that ever
needs to hold several would take a mapping instead, and both of these become per-adapter.
"""

VLLM_ADAPTER_PROBE_TOKENS = 24
"""Completion length for :meth:`VLLMBackend.assert_adapter_changes_output`.

Short on purpose: the check's sensitive signal is the cumulative logprob, which an applied adapter
shifts from the first token, so the completion does not need to be long enough for greedy token
ids to actually diverge -- a real step-70 games adapter left all three 24-token greedy prefixes
untouched while shifting every prompt's logprob (measured 2026-08-22). The check runs before every
battery, so its cost is charged to every eval.
"""

VLLM_SAMPLING_PARAM_ATTRS: dict[str, str] = {
    "max_new_tokens": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "seed": "seed",
    "stop": "stop",
}
"""Every :class:`SamplingConfig` field vLLM has a ``SamplingParams`` counterpart for, mapped to it.

``do_sample`` is absent because vLLM carries no such knob -- greedy decoding is spelled as
temperature 0.0 -- so :meth:`VLLMBackend.applied_sampling` reads it off the engine's own
``sampling_type`` instead. The mapping exists so a caller recording what a run decoded under can walk
the config's own fields and ask the engine what it holds for each, rather than restating these seven
names a third time.
"""

_VLLM_FINISH_REASONS = {"length": STOP_REASON_MAX_TOKENS, "stop": STOP_REASON_END_TURN}
"""vLLM's finish reasons, for the two that mean what a Converse stop reason means.

Its full vocabulary is ``("stop", "length", "abort", "error", "repetition")`` (read from the
installed engine's ``vllm.v1.engine.FINISH_REASON_STRINGS``, not remembered); the three absent here
have no Converse counterpart and travel under their own names.
"""


POOLED_REQUEST_ID_PREFIX = "pooled-"
"""The external request ids of one :meth:`VLLMBackend.generate_streaming` submission: this plus the prompt's index.

Chosen by the backend rather than left to the engine because of how vLLM 0.27.1 names requests:
``LLMEngine.add_request`` randomises the id it is given into an internal one and returns THAT, while
every ``RequestOutput`` carries the external id the caller supplied. So the id a caller can match
outputs on is the one it chose, and ``LLM.enqueue``, which returns the internal id, hands back names no
output will ever carry (found the first time the pooled drain ran on the L4: the pairing check refused
id ``'16'`` as one the submission never made). Supplying the ids and matching on them is the shape
``LLM.generate`` itself relies on when it sorts its outputs by ``int(request_id)``.
"""


def _vllm_tokenized_completion(output: Any) -> TokenizedCompletion:  # noqa: ANN401 - a vLLM RequestOutput; the module is imported lazily
    """Read one finished vLLM ``RequestOutput`` into the record both generate paths hand back.

    Shared by :meth:`VLLMBackend.generate_tokenized` and :meth:`VLLMBackend.generate_streaming` so a
    completion cannot come out shaped differently depending on which path drained it. The pairing of
    the output with its prompt is NOT done here -- the two paths pair differently (by position, by
    request id) and each states its own failure -- but the two readings of the reply itself are.

    A reply carrying other than exactly one sequence is refused: every request here is made with
    ``n=1``, and a record carries exactly one completion, so a second sequence is an engine that did
    not do what it was asked and reading ``outputs[0]`` would silently discard the rest.

    A reply whose ``prompt_token_ids`` came back empty is refused rather than read as a zero-token
    prompt. vLLM leaves that field unset only for input shapes this path never sends, so it means the
    call went wrong -- and both ways of tolerating it are silent: a usage record that undercounts its
    input, and a capture whose prompt/response boundary sits at position zero, labelling the entire
    prompt as text the model generated.

    Only the finish reasons with a Converse equivalent are translated (``length`` is a cap hit,
    ``stop`` is a finished reply -- or a stop-sequence halt, which :func:`_vllm_stop_reason` tells
    apart off the matched stop string). vLLM's other three -- ``abort``, ``error``, ``repetition`` --
    pass through under their own names rather than being folded into ``end_turn``: each is a run that
    went wrong, and relabelling it as a normal ending is how it would stop being noticed.
    """
    if len(output.outputs) != 1:
        raise RuntimeError(
            f"request {output.request_id!r} came back with {len(output.outputs)} sequences where "
            f"one was asked for; a record carries exactly one completion."
        )
    if not output.prompt_token_ids:
        raise RuntimeError(
            f"vLLM returned no prompt_token_ids for request {output.request_id}, so this "
            "reply's prompt length is unknowable: a reader would either undercount its "
            "input tokens or label the whole prompt as generated text"
        )
    sequence = output.outputs[0]
    prompt_token_ids = tuple(int(token) for token in output.prompt_token_ids)
    response_token_ids = tuple(int(token) for token in sequence.token_ids)
    return TokenizedCompletion(
        completion=BedrockCompletion(
            text=sequence.text,
            reasoning="",
            usage=TokenUsage(
                input_tokens=len(prompt_token_ids),
                output_tokens=len(response_token_ids),
            ),
            stop_reason=_vllm_stop_reason(sequence.finish_reason, sequence.stop_reason),
        ),
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
    )


def _pooled_prompt_index(
    output: Any,  # noqa: ANN401 - a vLLM RequestOutput; the module is imported lazily
    chats: Sequence[str],
    outstanding: dict[str, int],
    answered: set[str],
) -> int:
    """Return the prompt index one finished pooled output answers, moving its id from outstanding to answered.

    The pairing half of :meth:`VLLMBackend.generate_streaming`, where replies arrive in completion
    order and the prompt an output answers is known only from its request id. Three refusals, each
    because filing the reply anyway would be silent: an id this submission has already answered (an
    engine emitting two finished outputs for one request, the shape a stale submission or a non-final
    output kind produces), an id it never made (a stale or concurrent submission on the same
    engine), and an echoed ``prompt`` that is not the chat text sent under that id -- the same string
    comparison :meth:`VLLMBackend.generate_tokenized` makes, moved to where the pairing happens.
    """
    request_id = str(output.request_id)
    index = outstanding.pop(request_id, None)
    if index is None:
        if request_id in answered:
            raise RuntimeError(
                f"the engine returned a second finished reply for request {request_id!r}, which "
                f"this submission has already answered; filing it would put two completions under "
                f"one prompt."
            )
        raise RuntimeError(
            f"the engine returned request id {request_id!r}, which this submission never made: a "
            f"stale submission still in flight, or another caller on the same engine. Its reply "
            f"would be filed under the wrong prompt."
        )
    answered.add(request_id)
    if output.prompt != chats[index]:
        raise RuntimeError(
            f"the engine paired request {request_id!r} with prompt {output.prompt!r} where "
            f"{chats[index]!r} was sent under that id. Filing this reply would put a completion "
            f"under a prompt it does not answer, silently."
        )
    return index


def _vllm_stop_reason(finish_reason: str | None, matched_stop: object) -> str | None:
    """Translate one vLLM completion's ending into the Converse stop-reason vocabulary.

    vLLM reports a stop-string halt and an end-of-turn under the same ``finish_reason`` (``stop``);
    what tells them apart is ``CompletionOutput.stop_reason``, which carries the matched stop
    string, a stop token id, or ``None`` for the model's own terminator (verified against the
    installed 0.27.1: the v1 output processor sets it to the matched string exactly when the
    detokenizer found one). Only the string case is a caller-configured stop sequence -- a stop
    token id is the model ending its turn by another spelling -- so only it maps to
    :data:`STOP_REASON_STOP_SEQUENCE`. Folding the two together would make a harness turn that
    halted at ``</run>`` indistinguishable from one that finished, which is the field's whole job.
    """
    if finish_reason == "stop" and isinstance(matched_stop, str):
        return STOP_REASON_STOP_SEQUENCE
    if finish_reason is None:
        return None
    return _VLLM_FINISH_REASONS.get(finish_reason, finish_reason)


OUTPUT_AFFECTING_ENGINE_SETTINGS: tuple[str, ...] = (
    "quantization",
    "dtype",
    "kv_cache_dtype",
    "max_model_len",
)
"""The engine construction settings that change what a request samples, or where it is cut off.

Named as a set rather than left implicit because a resumed run's identity is built from exactly
these (`games.select_prompts.engine_identity_fields`): online quantization is a different policy
from bf16 and the flag's own help refuses to compare the two, a dtype or KV-cache dtype is the same
statement in another form, and ``max_model_len`` is the length at which vLLM clips a prompt plus its
completion, which moves the truncation and parse-failure rates rather than the sampler.

``gpu_memory_utilization`` is deliberately not one of them. It sizes the engine's claim on the card,
so it is precisely the setting a relaunch onto a different card moves legitimately, and refusing a
resume over it would make the resume unusable on rented capacity.
"""


class VLLMBackend:
    """Persistent vLLM engine kept warm across episodes: the throughput path for real rollouts.

    vLLM is an optional extra and not installed by default, so it is imported lazily here (via
    ``importlib`` to keep the import out of ruff's top-level-import rule). Not exercised by the
    offline tests and not required to work; it exists so the RL loop has a fast backend behind
    the same protocol when the extra is present.

    ``lora_adapter`` serves a LoRA checkpoint un-merged, which is what
    :mod:`games.eval_model` asks for and why: folding an adapter into bf16 base weights rounds
    most of the trained delta away, while a runtime adapter reaches the output through an
    fp32-accumulated matmul. The engine must be built with ``enable_lora=True`` for it, so the
    caller passes both together rather than this class inferring one from the other -- the
    adapter's own config decides ``max_lora_rank`` and ``lora_target_modules``, which are the
    engine's business and not this constructor's.

    An adapter vLLM cannot apply is the failure mode to fear here, because it is silent: an
    adapter whose module names do not line up with the served model's is skipped per module at
    DEBUG level and generation proceeds at base weights, which reads downstream as a training run
    that changed nothing. Upstream reports exactly that on this model family. So
    :meth:`assert_adapter_changes_output` exists and the eval path calls it before spending GPU
    time on a battery.
    """

    transport = "vllm"

    def __init__(  # noqa: PLR0913 - flat construction knobs, each one a recorded serving decision
        self,
        model_id: str,
        *,
        thinking: bool = False,
        sampling: SamplingConfig | None = None,
        lora_adapter: str | Path | None = None,
        model_path: str | Path | None = None,
        stop_token_ids: Sequence[int] | None = None,
        **engine_kwargs: object,
    ) -> None:
        """Initialize the optional vLLM engine lazily for the selected model.

        ``model_path`` is the directory the engine and tokenizer actually load when that is not
        ``model_id``: a verified local snapshot of a hub revision, served under the label
        ``model_id`` carries into every record (:mod:`games.eval_model`'s full-weights rung). Unset,
        the engine loads ``model_id`` as it always did.

        ``stop_token_ids`` pins the ids every request stops on, beside whatever the engine merges in
        from the checkpoint's generation_config (see :data:`END_OF_TURN_TOKENS` for why a screen
        across checkpoints must pin them). Unset, only the engine's own merge applies, as before.
        """
        vllm = importlib.import_module("vllm")
        lora_request_module = importlib.import_module("vllm.lora.request")
        self.model_id = model_id
        self.model_path = None if model_path is None else str(model_path)
        self.stop_token_ids: tuple[int, ...] = tuple(stop_token_ids or ())
        self.thinking = thinking
        # As with HFBackend, an unset config follows the thinking flag (see SamplingConfig).
        self.sampling = sampling or SamplingConfig.for_thinking(thinking=self.thinking)
        load_from = self.model_path or model_id
        # What the engine was CONSTRUCTED with, kept because nothing else can say it afterwards: the
        # kwargs go straight into vLLM and several of them (quantization, max_model_len) change what
        # the model samples or where it truncates. A trace's own record of the flags is what was
        # ASKED for, and `games.eval_model` derives some of these from a checkpoint rather than from
        # a flag, so the resume identity and the trace meta both read them off here instead.
        self.engine_settings: dict[str, object] = dict(engine_kwargs)
        # No trust_remote_code: see TRUST_REMOTE_CODE_WITHHELD_REASON.
        self._tokenizer = AutoTokenizer.from_pretrained(load_from)
        self._llm = vllm.LLM(model=load_from, **engine_kwargs)
        self.lora_adapter = None if lora_adapter is None else str(lora_adapter)
        self._lora_request = (
            None
            if self.lora_adapter is None
            else lora_request_module.LoRARequest(
                VLLM_LORA_ADAPTER_NAME, VLLM_LORA_ADAPTER_ID, self.lora_adapter
            )
        )
        # vLLM honours presence_penalty, Qwen's thinking-mode anti-loop lever the HF path lacks.
        # include_stop_str_in_output keeps the stop text vLLM strips by default (parsers need it).
        self._sampling_params = vllm.SamplingParams(
            max_tokens=self.sampling.max_new_tokens,
            temperature=self.sampling.temperature if self.sampling.do_sample else 0.0,
            top_p=self.sampling.top_p,
            top_k=self.sampling.top_k,
            min_p=self.sampling.min_p,
            repetition_penalty=self.sampling.repetition_penalty,
            presence_penalty=self.sampling.presence_penalty,
            seed=self.sampling.seed,
            stop=list(self.sampling.stop) if self.sampling.stop else None,
            stop_token_ids=list(self.stop_token_ids) if self.stop_token_ids else None,
            include_stop_str_in_output=bool(self.sampling.stop),
        )
        # applied_sampling compares against this; SamplingType is not exported from vllm's top level.
        self._greedy_sampling_type = importlib.import_module(
            "vllm.sampling_params"
        ).SamplingType.GREEDY
        logger.info(
            "VLLMBackend engine up for %s from %s thinking=%s lora_adapter=%s stop_token_ids=%s",
            model_id,
            load_from,
            thinking,
            self.lora_adapter,
            self.stop_token_ids,
        )

    def assert_serves_weights(self, snapshot_dir: Path) -> None:
        """Raise unless the engine reports it loaded its weights from ``snapshot_dir``.

        Read off the live engine's own model config rather than off this object's constructor
        argument, because the constructor argument is what was ASKED for and the engine's record
        is what happened; a mismatch between the two is exactly what this exists to surface.
        Verified on the installed vllm 0.27.1: ``LLM.llm_engine.model_config.model`` carries the
        path the engine was built from, verbatim.
        """
        engine_model = str(self._llm.llm_engine.model_config.model)
        if Path(engine_model).resolve() != snapshot_dir.resolve():
            raise RuntimeError(
                f"the vLLM engine reports it loaded {engine_model!r}, not the verified snapshot "
                f"{snapshot_dir}; the served model is not the one whose identity was checked"
            )

    def assert_adapter_changes_output(self, prompts: Sequence[str]) -> None:
        """Raise unless the served adapter changes what the model generates.

        The check every other signal fails to give. vLLM validates a LoRA checkpoint only on the
        last component of each module name, so an adapter carrying a whole wrong prefix passes
        validation, is then skipped module by module at DEBUG level, and serves base weights under
        a trained checkpoint's name. Nothing about that is visible in a trace: the completions are
        fluent, the finish reasons are normal, and the arm simply reads as a null result. Upstream
        vLLM has an open report of that exact symptom on this hybrid-attention family, which is
        why this is a behavioural check rather than a reading of the engine's logs.

        Compared on token ids under greedy decoding at both arms, so the only thing that can
        differ is the adapter: sampling would make a difference meaningless and an identity
        impossible to interpret. Token ids rather than text because two token sequences can decode
        to one string.

        Cheap enough to be unconditional -- a handful of tokens against the battery it guards --
        and run on several prompts because a trained adapter is not obliged to move every single
        one.
        """
        if self._lora_request is None:
            raise ValueError(
                "assert_adapter_changes_output needs an adapter to check, and this engine was "
                "built without lora_adapter. Nothing was verified."
            )
        if not prompts:
            raise ValueError(
                "assert_adapter_changes_output was given no prompts, so it would pass without "
                "comparing anything -- the shape of check this exists to replace."
            )
        vllm = importlib.import_module("vllm")
        # logprobs=1 makes the engine return per-token logprobs, which is what populates
        # cumulative_logprob -- the sensitive half of the comparison below (verified populated at
        # this value on the installed vllm 0.27.1). Without it the field is None on both arms and
        # the check would silently fall back to the weak token-id half alone.
        greedy = vllm.SamplingParams(
            max_tokens=VLLM_ADAPTER_PROBE_TOKENS, temperature=0.0, top_p=1.0, top_k=-1, logprobs=1
        )
        chats = [_as_single_user_turn(self._tokenizer, p, thinking=self.thinking) for p in prompts]
        with_adapter = self._llm.generate(chats, greedy, lora_request=self._lora_request)
        without_adapter = self._llm.generate(chats, greedy)
        # Two signals, either sufficient. Token ids are the visible one but need the greedy paths
        # to actually diverge, which a small trained delta can fail to do in a short probe (a real
        # step-70 adapter here moved 0/3 prefixes at 24 tokens while shifting every logprob). The
        # cumulative logprob is exact the other way: an adapter whose tensors were all skipped
        # contributes bitwise zero to every matmul, so identical logprobs on every prompt is the
        # no-op signature and any difference proves the adapter reached the forward pass.
        moved_ids = [
            list(adapted.outputs[0].token_ids) != list(base.outputs[0].token_ids)
            for adapted, base in zip(with_adapter, without_adapter, strict=True)
        ]
        moved_logprob = [
            adapted.outputs[0].cumulative_logprob != base.outputs[0].cumulative_logprob
            for adapted, base in zip(with_adapter, without_adapter, strict=True)
        ]
        if not any(moved_ids) and not any(moved_logprob):
            raise RuntimeError(
                f"adapter {self.lora_adapter} changed nothing: across {len(prompts)} greedy "
                f"prompts every token id AND every cumulative logprob served with the adapter "
                f"matched the base model's exactly. vLLM loaded it without complaint, so the "
                f"likely cause is module names that do not line up with the served model -- serve "
                f"the base checkpoint the adapter records, and check that lora_target_modules "
                f"names the adapter's own target_modules so the engine raises on a module it "
                f"cannot wrap instead of skipping it. Evaluating in this state would measure the "
                f"base model and label it a trained checkpoint."
            )
        logger.info(
            f"adapter verified to change output, {self.lora_adapter=} "
            f"moved_ids={sum(moved_ids)}/{len(moved_ids)} "
            f"moved_logprob={sum(moved_logprob)}/{len(moved_logprob)} prompts"
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer this engine's prompts are rendered with.

        Public because the interpretability capture needs the SAME tokenizer the engine generated
        under in order to decode the ids it returns and name each position's token. Loading a second
        one from the model id would usually agree, and the failure when it did not (a different
        revision, a different special-token set) would be a silently mislabelled position rather
        than an error.
        """
        return self._tokenizer

    def applied_sampling(self) -> dict[str, object]:
        """Report, per :class:`SamplingConfig` field, what the engine will actually apply.

        Read back off the live ``SamplingParams`` rather than re-derived from :attr:`sampling`,
        because vLLM REWRITES what it was handed: ``SamplingParams.__post_init__`` resets ``top_p``
        to 1.0, ``top_k`` to 0 and ``min_p`` to 0.0 whenever the temperature is under its greedy
        epsilon (verified on the installed vllm 0.27.1 -- constructing one with ``temperature=0.0,
        top_p=0.9, top_k=20, min_p=0.05`` reports those three back as 1.0, 0 and 0.0). A record
        derived from the config would then assert three truncation settings the engine had already
        thrown away, which is the same false claim the HF path's dropped-knob bookkeeping exists to
        prevent, arriving by a different route.

        ``do_sample`` comes off ``sampling_type``, the predicate vLLM itself decides greediness
        with, rather than a re-implementation of the epsilon comparison here.

        **One stop-related setting is structurally invisible here and it is the one that changes the
        returned TEXT.** ``include_stop_str_in_output`` is not a :class:`SamplingConfig` field, so the
        map cannot cover it and no sampler partition will ever mention it -- yet with it false vLLM
        strips the matched stop string and the harness's run-block parser then sees an unclosed block.
        It gets no field of its own because nothing configures it independently of ``stop`` (the
        constructor sets it to ``bool(stop)``), but a reader of a partition should not have to discover
        that "stop is reported" does not extend to how the stop text is returned.
        """
        applied: dict[str, object] = {
            field: getattr(self._sampling_params, attr)
            for field, attr in VLLM_SAMPLING_PARAM_ATTRS.items()
        }
        # vLLM normalises stop to a LIST in __post_init__ and never hands back None -- measured on
        # the installed 0.27.1 across all four inputs (absent, None, [], ["</run>"]). So the engine
        # always holds a list where SamplingConfig holds a tuple, and since the partition compares by
        # equality, without this coercion `stop` reads as dropped-because-substituted on EVERY run in
        # this repository, including the ones that never asked for a stop: () == [] is False. The
        # coercion cannot mask a genuine drop, also measured: a configured ("</run>",) against an
        # engine holding nothing still compares unequal. `or ()` is redundant given the measured
        # always-a-list behaviour and kept as robustness against a vLLM that stops normalising.
        applied["stop"] = tuple(applied["stop"] or ())  # pyright: ignore[reportArgumentType]
        applied["do_sample"] = self._sampling_params.sampling_type != self._greedy_sampling_type
        return applied

    def generate_tokenized(self, prompts: list[str]) -> list[TokenizedCompletion]:
        """Generate one completion per prompt, keeping the exact ids the engine read and wrote.

        The engine-facing call, which :meth:`generate_detailed` then narrows to text and counts.
        vLLM hands back both id sequences, so a caller that needs them -- the interpretability
        capture, which runs a HuggingFace forward pass over prompt+response and labels every
        position -- takes them from the same call the text came from instead of re-tokenising a
        decoded string, which misaligns at any BPE or special-token boundary the decode does not
        round-trip. Verified against the installed engine, ``CompletionOutput`` carries ``text``,
        ``token_ids`` and ``finish_reason``, and ``RequestOutput`` carries ``prompt_token_ids``.

        How each reply is read -- the finish-reason translation, the refusal of an empty
        ``prompt_token_ids`` and of a reply carrying other than one sequence -- is
        :func:`_vllm_tokenized_completion`, shared with :meth:`generate_streaming` so the two paths
        cannot hand back differently shaped records for the same engine output.

        Each reply is also checked against the prompt it answers. vLLM returns its outputs sorted by
        request id, so they do arrive in input order -- but a caller pairs a batch of replies with its
        own list of prompts, and every consequence of that pairing being off by one is silent: the
        activations captured for one twin get filed under the other, and the contrast then reads a
        difference that is pure bookkeeping. Comparing against ``RequestOutput.prompt`` costs one
        string comparison per reply and turns the ordering from an assumption into a check.
        """
        chats = [_as_single_user_turn(self._tokenizer, p, thinking=self.thinking) for p in prompts]
        outputs = self._llm.generate(chats, self._sampling_params, lora_request=self._lora_request)
        tokenized: list[TokenizedCompletion] = []
        for chat, output in zip(chats, outputs, strict=True):
            if output.prompt != chat:
                raise RuntimeError(
                    "vLLM returned its replies in a different order than the prompts were sent: "
                    f"reply {len(tokenized)} answers {output.prompt!r} where {chat!r} was expected. "
                    "Pairing a batch of replies with the wrong prompts files every measurement under "
                    "the wrong item, silently."
                )
            tokenized.append(_vllm_tokenized_completion(output))
        return tokenized

    def generate_streaming(
        self, prompts: Sequence[str]
    ) -> Generator[tuple[int, TokenizedCompletion]]:
        """Submit every prompt to the engine at once and yield ``(prompt index, completion)`` as each finishes.

        The incremental twin of :meth:`generate_tokenized`, which hands the same list to
        ``LLM.generate`` and returns nothing until the LAST sequence has stopped, so a battery that
        called it once per game paid one 65,536-token tail per game and idled the engine between
        calls (measured on the track-record-v2 battery cell: 10-38 records per minute at per-game
        widths against 49-55 when handed a whole cell's prompts). A caller that persists per record
        (the eval battery's pooled path, ``games.evals._run_pooled`` through
        ``games.chunked_decode.stream_vllm_completions``) also wants each completion the moment its
        own sequence stops, whatever else is still in flight: that is what lets a trace be appended
        per record and turns a mid-cell death into a loss of zero finished records. So this adds the
        same requests to the same engine and steps it by hand instead of letting ``LLM.generate``
        drain it.

        **Bit-for-bit the same generation as ``LLM.generate`` over the same list**, measured and
        argued. Measured on the L4 (2026-09-02, sixteen mixed-length prompts on a seeded
        ``Qwen3.5-0.8B`` engine, 512-token cap): :meth:`generate_tokenized` and this method gave
        identical text and identical response-token-id digests for all sixteen prompts. Argued from
        the installed vLLM 0.27.1 (``entrypoints/offline_utils.py``): ``LLM.generate`` renders each
        prompt through ``renderer.render_cmpl`` under ``default_cmpl_tok_params``, sets
        ``output_kind = FINAL_ONLY`` on the params, adds each under a counter id via
        ``llm_engine.add_request``, then runs ``while has_unfinished_requests(): step()`` collecting
        finished outputs -- and finally sorts them by id, which is the one thing this does not do.
        Relative to a caller that used to make several smaller calls, the change is
        statistically neutral and NOT bit-neutral: unseeded requests share the engine's global RNG
        stream, so what a prompt draws depends on what else is in flight at each step. Same engine
        seed, same prompt list, same submission replays byte for byte; a different grouping is a
        different draw from the same distribution.

        Yielded in COMPLETION order, which is not prompt order, so every completion travels with the
        index of the prompt it answers. Two checks stand between the engine and that pairing, both
        raising rather than filing a reply under the wrong prompt: the request id the engine returns
        must be one this submission made and not yet answered (two refusals, since a foreign id means
        a stale or concurrent submission on the engine where a repeat means the engine emitted two
        finished outputs for one request), and the ``prompt`` the engine echoes back must be the chat
        text sent under that id -- the same string comparison :meth:`generate_tokenized` makes, moved
        to where the pairing now happens (:func:`_pooled_prompt_index`). A submission the engine
        reports drained with requests still unanswered is refused too, since the records for those
        prompts would otherwise be missing without anything saying so.

        Mechanics, each a lesson from the first time this ran: requests are added under ids this
        method chooses (:data:`POOLED_REQUEST_ID_PREFIX` plus the index) rather than through
        ``LLM.enqueue``, whose returned ids no output carries; the sampling params are cloned before
        ``FINAL_ONLY`` is set, where ``LLM._add_request`` mutates the shared object in place, so the
        backend's own params stay as :meth:`applied_sampling` reports them; and the chat strings are
        rendered (tokenised) through the engine's own renderer before they reach ``add_request``,
        because handing it a raw string tokenises identically today but is the path 0.27.1 logs as
        deprecated and will remove. The rendered input keeps the text under ``prompt``, so the
        echoed-prompt check reads the same string either way.

        One submission owns the engine from entry to exit. An engine that already has requests in
        flight is refused before anything is rendered or added: vLLM keys request state by a
        randomised internal id and keeps the caller's id in a multimap
        (``v1/engine/output_processor.py``), so a second submission reusing ``pooled-0..`` would be
        accepted and the two generations' replies could not be told apart; the refusal names the
        caller-lifecycle bug (an earlier generator abandoned without being closed, or another caller
        on the same engine) where the engine would otherwise blame its ids. And whatever ends the
        submission early -- a consumer that stops iterating, a refusal above, an engine step that
        raises, an ``add_request`` that fails on prompt k -- every request not yet handed to the
        caller is aborted on the way out (``LLMEngine.abort_request`` by external id, a no-op for
        ids the engine has already finished), so the engine is left as it was found and the next
        submission can start; vLLM's own ``_render_and_add_requests`` does the same for a failed add.
        A consumer that may raise mid-drain and go on using the engine should close the generator
        (``contextlib.closing``) rather than wait for its finaliser, which under a live traceback
        runs only once the exception is dropped; after ``EngineCoreClient.shutdown`` the abort is a
        no-op by vLLM's own ``engine_dead`` guard, so a finaliser that runs after teardown is silent.

        What a turn-level scheduler for the agentic loops would still need and this does not offer:
        adding requests while earlier ones are in flight, caller-chosen keys instead of positional
        ids, and aborting one request mid-flight (the abort here is all-or-nothing, on exit). This is
        a one-shot submission; those are the next method, not a mode of this one.
        """
        chats = [_as_single_user_turn(self._tokenizer, p, thinking=self.thinking) for p in prompts]
        engine = self._llm.llm_engine
        if engine.has_unfinished_requests():
            raise RuntimeError(
                "the vLLM engine already has requests in flight, so this submission's replies could "
                "not be told from theirs: an earlier generate_streaming was abandoned without being "
                "closed, or another caller is driving the same engine. Close or drain it first."
            )
        params = self._sampling_params.clone()
        params.output_kind = importlib.import_module(
            "vllm.sampling_params"
        ).RequestOutputKind.FINAL_ONLY
        rendered = self._llm.renderer.render_cmpl([{"prompt": chat} for chat in chats])
        if len(rendered) != len(chats):
            raise RuntimeError(
                f"the renderer returned {len(rendered)} engine inputs for {len(chats)} prompts; the "
                f"request ids would no longer name the prompts they answer."
            )
        outstanding: dict[str, int] = {}
        answered: set[str] = set()
        try:
            for index, engine_input in enumerate(rendered):
                request_id = f"{POOLED_REQUEST_ID_PREFIX}{index}"
                engine.add_request(
                    request_id, engine_input, params, lora_request=self._lora_request
                )
                outstanding[request_id] = index
            logger.info(
                f"pooled submission: {len(chats)} sequences enqueued on {self.model_id}; draining "
                f"as each finishes"
            )
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if not output.finished:
                        continue
                    index = _pooled_prompt_index(output, chats, outstanding, answered)
                    yield index, _vllm_tokenized_completion(output)
            if outstanding:
                raise RuntimeError(
                    f"the engine reports nothing in flight, yet {len(outstanding)} of {len(chats)} "
                    f"requests never came back (prompt indices "
                    f"{sorted(outstanding.values())[:10]}...). The records for those prompts would "
                    f"be missing without anything saying so."
                )
        finally:
            # Whatever ended the submission -- the consumer stopping, a refusal above, the engine
            # raising -- nothing of ours may stay queued: the next submission's entry check would
            # refuse forever, and a stale request's reply would land under a foreign id.
            if outstanding:
                engine.abort_request(list(outstanding))

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Generate one completion per prompt, carrying vLLM's own finish reason and token counts.

        Delegates rather than driving the engine a second way, so a caller reading only ``Backend``
        cannot receive a different completion than the tokenized record carries -- the same reason
        :meth:`generate` delegates here.
        """
        return [record.completion for record in self.generate_tokenized(prompts)]

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one completion for each prompt through the persistent engine."""
        return [completion.text for completion in self.generate_detailed(prompts)]


DEFAULT_BEDROCK_REGION = "us-west-2"
DEFAULT_BEDROCK_CONCURRENCY = 16

OPENROUTER_MODEL_PREFIX = "openrouter:"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 1_500.0
DEFAULT_OPENROUTER_CONCURRENCY = 8
DEFAULT_OPENROUTER_MAX_TOKENS = 120_000
DEFAULT_OPENROUTER_MAX_ATTEMPTS = 8
DEFAULT_OPENROUTER_REASONING_EFFORT = "medium"
"""The owner's default rung for every hosted receiver (2026-09-12): enough reasoning that a model is
not lazy, without buying its full test-time-compute scaling, which the bench does not depend on.
Pass ``reasoning_effort=None`` to send no reasoning field and take the provider default."""
DEFAULT_OPENROUTER_RETRY_BASE_SECONDS = 1.0
DEFAULT_OPENROUTER_RETRY_MAX_SECONDS = 120.0
DEFAULT_OPENROUTER_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
OPENROUTER_RATE_LIMIT_STATUS = 429
OPENROUTER_SERVER_ERROR_MIN_STATUS = 500
OPENROUTER_SERVER_ERROR_MAX_STATUS = 599

BEDROCK_PROFILE_ENV = "REWARD_HACKING_BEDROCK_PROFILE"

PROFILE_FROM_ENVIRONMENT = "<from-environment>"
"""The ``profile`` default: resolve the named AWS profile from :data:`BEDROCK_PROFILE_ENV`.

A sentinel rather than the profile name itself because a profile name is a machine-local value and
this repository is public, and a sentinel rather than ``None`` because ``None`` already means
something different and load-bearing here: skip the named profile entirely and take the ambient
credential chain, which is what a Batch container holding an instance role has. Shaped so it cannot
collide with a real profile name.
"""


def bedrock_profile() -> str:
    """Return the named AWS profile the Converse path authenticates with.

    Read from the environment rather than defaulted in source, for the reason
    ``reward_hacking/bedrock_batch.py`` gives at length for the batch path's account, role and
    bucket: this remote is public, so an account-specific identifier committed here is published.
    Unset raises rather than falling back to anything, because both fallbacks are worse than a
    crash -- a committed name republishes the value, and the ambient chain on this box fails
    credential resolution outright, which surfaces as an authentication error far from its cause.
    """
    profile = os.environ.get(BEDROCK_PROFILE_ENV, "").strip()
    if not profile:
        raise RuntimeError(
            f"{BEDROCK_PROFILE_ENV} is not set; it must name an AWS profile with Bedrock access. "
            "It is read from the environment rather than defaulted in code because this repository "
            "is public. Export it in your shell or a local gitignored env file, or pass "
            "profile=None to use the ambient credential chain (an instance role, for example)."
        )
    return profile


@dataclass(frozen=True, slots=True)
class ReasoningDialect:
    """How one Bedrock model family spells reasoning effort, and which levels it accepts.

    The two known families are mutually incompatible, and one direction of the mistake is silent:
    GPT-OSS accepts an unrecognised request field without error, so handing it the GPT-5.6-shaped
    nested payload drops the setting on the floor and runs the model at its default effort. A
    cross-model comparison built that way would credit the difference to the model. GPT-OSS also
    accepts an out-of-range effort *value* without error, which is why the ladder is checked here
    rather than left to the API to reject.

    ``REASONING_DIALECTS`` keys these by the substring identifying the family inside a Bedrock
    model id. The levels recorded there are quoted from the API's own validation errors, so they
    are authoritative rather than remembered.
    """

    nested: bool
    levels: frozenset[str]

    def request_fields(self, effort: str) -> dict[str, object]:
        """Build the ``additionalModelRequestFields`` payload carrying this effort level."""
        if effort not in self.levels:
            raise ValueError(
                f"reasoning effort {effort!r} is not accepted by this model family; "
                f"expected one of {sorted(self.levels)}"
            )
        if self.nested:
            return {"reasoning": {"effort": effort}}
        return {"reasoning_effort": effort}


REASONING_DIALECTS: dict[str, ReasoningDialect] = {
    "gpt-oss": ReasoningDialect(
        nested=False,
        levels=frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"}),
    ),
    "gpt-5.6": ReasoningDialect(
        nested=True, levels=frozenset({"none", "low", "medium", "high", "xhigh", "max"})
    ),
}

CONVERSE_STOP_REFUSING_FAMILIES: tuple[str, ...] = ("gpt-oss", "gpt-5.6")
"""Model families whose Converse endpoint rejects ``stopSequences`` outright, probed live.

Per-family, not per-API: Converse translates the field per provider, and both OpenAI families on
this roster reject it with ``ValidationException: This model doesn't support the stopSequences
field`` (gpt-oss-120b and gpt-5.6-luna, probed live 2026-08-24), while Anthropic accepts it and
returns ``stopReason "stop_sequence"`` with the matched text stripped (claude-sonnet-5, same
probe). Keyed on the substring inside a model id the way :data:`REASONING_DIALECTS` is.

A DENY list rather than an allowlist, deliberately: an unprobed family is sent the field and, if
it too refuses, fails loudly on the first call with Bedrock's own clear message -- which is the
correct default here, because the silent alternative (dropping the stop for anything unlisted)
would sample a harness run without its stop and report nothing. A family on this list gets the
stop *skipped by the caller that wanted it*, which is a behavioural degradation and must be
warned about by that caller, never absorbed here.
"""


def converse_supports_stop_sequences(model_id: str) -> bool:
    """Whether this model's Converse endpoint is expected to accept ``stopSequences``.

    "Expected" per :data:`CONVERSE_STOP_REFUSING_FAMILIES`: an unlisted family reads ``True`` and
    a wrong ``True`` fails loudly on the first call rather than silently.
    """
    return all(family not in model_id for family in CONVERSE_STOP_REFUSING_FAMILIES)


CONVERSE_TEMPERATURE_REFUSING_FAMILIES: tuple[str, ...] = ("gpt-5.6",)
"""Model families whose Converse endpoint rejects the ``temperature`` field outright, probed live.

``ValidationException: This model doesn't support the temperature field. Remove temperature and
try again.`` (gpt-5.6-luna, probed live 2026-08-31 by the hatch probe's smoke). The upstream
reason is OpenAI's own API contract for its reasoning models: they sample at a fixed temperature
of 1.0 and accept no override, so omitting the field asks for the same distribution a caller
pinning 1.0 wanted -- but the request must not carry it. gpt-oss is NOT here: both gpt-oss sizes
accepted ``temperature`` 1.0 across the 2026-08-25 production batches.

Same deny-list contract as :data:`CONVERSE_STOP_REFUSING_FAMILIES`: an unprobed family is sent the
field and a refusal fails loudly on the first call with the service's own message, and a caller
that drops the field because of this list owns recording that it did.
"""


def converse_supports_temperature(model_id: str) -> bool:
    """Whether this model's Converse endpoint is expected to accept ``temperature``.

    "Expected" per :data:`CONVERSE_TEMPERATURE_REFUSING_FAMILIES`: an unlisted family reads
    ``True`` and a wrong ``True`` fails loudly on the first call rather than silently.
    """
    return all(family not in model_id for family in CONVERSE_TEMPERATURE_REFUSING_FAMILIES)


def _resolve_reasoning_dialect(model_id: str) -> ReasoningDialect:
    """Find the reasoning dialect for a model id, refusing to guess when the family is unknown.

    Guessing is the silent-failure case. The wrong shape is ignored without error by GPT-OSS, and
    families with no effort control at all (Claude takes a thinking-token budget instead) would
    either reject the field or ignore it depending on the vendor. Raising means a mis-specified
    run dies at construction rather than reporting numbers nobody can interpret.
    """
    for family, dialect in REASONING_DIALECTS.items():
        if family in model_id:
            return dialect
    raise ValueError(
        f"no known reasoning-effort dialect for model {model_id!r}; known families are "
        f"{sorted(REASONING_DIALECTS)}. Leave reasoning_effort unset to call at default effort."
    )


DEFAULT_BEDROCK_MAX_TOKENS = 30_000
"""The largest per-model output budget measured on this roster, as the shared Converse default.

Deliberately generous rather than modest-looking. The cap is a ceiling and a token is billed only if
it is generated, so an over-wide cap costs nothing, while a cap set too low is paid for in whole
runs. The previous default of 2048 was that mistake: over a 400-episode agentic run 37% of Converse
calls stopped at exactly 2048, and because the frontier model returns its reasoning redacted while
still charging it against this same budget, those turns came back empty or with an unclosed command
block and read downstream as a policy that declined to act rather than as a config bug.

A model whose own ceiling is lower refuses this loudly -- a botocore ``ValidationException`` on the
live path, a construction-time raise against the roster on the batch path -- which is the right
failure. A per-model budget is a measurement, and the measured ones live in
``recoverybench.budgets``, whose frontier row is where this number comes from.

Whether the field has to be sent at all is settled (hot-path decision C11, probed live 2026-09-03
on ``openai.gpt-oss-20b-1:0`` in us-west-2): ``Converse`` with no ``inferenceConfig`` at all,
``Converse`` with an empty ``inferenceConfig`` and ``ConverseStream`` with no ``inferenceConfig``
were all accepted and answered ``end_turn`` at 40 to 100 output tokens, and the installed botocore
service model lists none of ``inferenceConfig``'s four members as required. Omitting ``maxTokens``
runs the model at the provider's own output ceiling, which is where the owner's rule points (leave
the output cap unset; the levers are sample count and effort). This code still sends the field on
every request, because ``max_tokens`` is read as an ``int`` by the batch ceiling check, the
RecoveryBench run labels, the harness cap check and the batch-handle label checks, so making it
optional is a change to those consumers as much as to this default. Until that lands, this is the
cap every default run samples under, and every record carries it as such.
"""

DEFAULT_BEDROCK_DEADLINE_SECONDS = 3_600.0
"""Total wall clock one streamed Converse call may spend before it is abandoned mid-generation.

The number a socket read timeout cannot express. ``read_timeout`` bounds the *gap between reads*, so
a model that keeps emitting tokens can run arbitrarily long inside it: MiniMax M2.5 on hard arms
runs 2,900-3,000+ seconds per call, and 11 of 24 calls in one pass died at 2,700 seconds. Those
deaths correlate with reasoning length, which is usually the variable an experiment is manipulating,
so the surviving denominator is *biased* rather than merely small -- which is why a call past this
deadline comes back as a partial record (:data:`STOP_REASON_DEADLINE_EXCEEDED`) rather than as a
raise that loses it.

Set generously past the slowest call on record rather than at a value that looks tidy: a deadline
that fires on a legitimately long generation converts a real measurement into a partial record,
which is the same bias in the other direction. Tighten it per run when a model's own pace is known.
"""

DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS = 900
"""Longest silence tolerated between two reads of one call's socket.

Unchanged in value from the non-streaming client, but a different quantity now: against
``converse_stream`` this bounds the gap between two events of one reply rather than the whole call,
because the whole call is bounded by :data:`DEFAULT_BEDROCK_DEADLINE_SECONDS` instead. A streamed
reply emits deltas continuously, so tightening this is now the safe way to detect a genuinely stuck
call -- which it was not before, when a slow-but-alive model and a hung socket looked identical.
Left generous by default because time-to-first-event on a reasoning model has not been measured
here.
"""

DEFAULT_BEDROCK_MAX_ATTEMPTS = 5
"""Cap on botocore's retries for one call, and the reason the cap exists at all.

Bedrock's transient 500s arrive both as ``InternalServerException`` and as a bare ``ClientError``
with code ``"500"`` that maps to no modelled exception class, so retrying is botocore's job rather
than ours (see :class:`BedrockBackend`). The cap is what keeps that from being unbounded: each
attempt can burn a full read timeout before it fails, and retries cover the request handshake only
-- once the response headers arrive, the event stream is consumed outside botocore's retry loop
(read from ``botocore.eventstream.EventStream``, whose iteration happens after the API call
returns), so a mid-stream failure is never retried. Worst-case wall clock for one call is therefore
about ``max_attempts * read_timeout`` waiting for the stream to start, plus ``deadline`` reading it,
plus one last ``read_timeout`` if it goes silent at the end. :class:`BedrockBackend` logs that
figure at construction, because a bound nobody can see is a bound nobody reports.
"""


TRANSIENT_CLIENT_ERROR_CODES: frozenset[str] = frozenset(
    {
        # Retried by botocore's adaptive mode and surfacing here only once it gave up.
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "InternalServerException",
        # Model-side per-call failures botocore does NOT retry: one dead sample on first occurrence.
        "ModelErrorException",
        "ModelTimeoutException",
        # A 5xx that carried no service code at all: botocore stamps the HTTP status on it as the
        # code (``_parse_error_from_http_status``) and retries it by status before it reaches here.
        "500",
        "502",
        "503",
        "504",
    }
)
"""The ``ClientError`` codes a live call may die of without that being a bug in the run.

A CLOSED enumeration, and the reason it is one is the same reason the transport list on
:class:`BedrockBackend` is: widening it toward ``except ClientError`` would absorb a rejected
payload, an unknown model id or a credentials failure into hundreds of empty ``call_failed`` rows
that read downstream as a model with nothing to say. ``ValidationException``,
``AccessDeniedException`` and ``ResourceNotFoundException`` are deliberately absent and still
crash the batch on the first call. Three groups are on it, each a failure of one call rather than
of the request:

- The four codes botocore's adaptive retry already handles (a throttle, the service unavailable, a
  model still loading, an internal error), which reach this seam only after five attempts failed.
- ``ModelErrorException`` (HTTP 424, "an error while processing the model") and
  ``ModelTimeoutException`` (HTTP 408, the service's own per-call timeout), which botocore does
  not retry, so the first occurrence is the one recorded. They are model-side faults on one
  prompt -- the service model files neither as a client fault -- and a ``ModelErrorException``
  was one of the chunk killers the audit named; a leg should not die to one.
- The bare HTTP status strings ``"500"``, ``"502"``, ``"503"`` and ``"504"``. When a 5xx arrives
  with no service code in the body (a generic gateway page, a proxy), botocore's parser puts the
  status code in ``Error.Code`` and its transient checker retries those four statuses; the
  module already documents that Bedrock's transient 500s take this shape as well as
  ``InternalServerException``.

Before this list existed such a failure killed the chunk it was in and discarded every sibling reply
already billed -- up to ``chunk - 1`` completions, three recorded incidents in three weeks (a throttle
on the 70th call would have discarded 72; 7 of 8 completed results were discarded by one raising
task; 22 billed Sonnet replies died with one mid-stream error before ``EventStreamError`` was
isolated). Read timeouts are not a code: they arrive as botocore's and urllib3's timeout classes,
which the transport list already names.
"""


@dataclass(frozen=True, slots=True)
class BedrockSamplingConfig:
    """Decoding knobs the Bedrock Converse API actually honours.

    Deliberately not ``SamplingConfig``: Converse's ``inferenceConfig`` has no ``top_k`` and no
    greedy switch, so reusing the HuggingFace config would mean dropping ``top_k`` and
    ``do_sample`` silently, which is the whole failure class this harness keeps getting bitten by.
    ``None`` means "omit the field", which is the request shape verified against every model in
    the roster; set one and it is sent, and a model that refuses it fails loudly with a botocore
    ``ValidationException`` rather than ignoring it.

    ``max_tokens`` is the one field this config always sends -- ``converse_inference_config`` writes
    ``maxTokens`` on every request -- so it is the one field whose default is in the request whether
    anybody chose it or not. The API does not require it (:data:`DEFAULT_BEDROCK_MAX_TOKENS` has the
    live probe that settled that); this code does, for now. That is why it defaults *high*
    (:data:`DEFAULT_BEDROCK_MAX_TOKENS`, see its comment for the run this cost) rather than to a
    value that looks conservative: an unchosen cap that truncates is indistinguishable, downstream,
    from a model that declined to answer.

    ``stop_sequences`` becomes the request's ``stopSequences`` and is empty by default -- the
    benchmark runners want the whole reply. Unlike the local backends, **Converse strips the
    matched stop text from the returned completion** (the reply ends just before it, with
    ``stopReason`` ``"stop_sequence"``), so a caller that needs the text intact -- the agent
    harness, whose block parser wants the ``</run>`` it stopped on -- has to restore it; the loop
    does, keyed on that stop reason.
    """

    max_tokens: int = DEFAULT_BEDROCK_MAX_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Input and output token counts, over one Converse call or a whole run of them.

    ``input_tokens`` is every token the model read -- uncached, cache-read and cache-write summed --
    which is what every record on disk and every cost script has always meant by it. The two cache
    counters ride beside it rather than being carved out of it, so the total is unchanged and the
    split is still recoverable: uncached input is ``input_tokens`` minus the two. Recording the
    split is what turns the run docs' list-price cost figures into real ones -- a Luna judge pass
    whose 4,360-token rubric was cached on every call was billed at the cached rate for ~86% of
    its input, a factor the run docs could not see because the counters were folded away here.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate field-wise, so a run total is the sum of its calls."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_write_input_tokens=(
                self.cache_write_input_tokens + other.cache_write_input_tokens
            ),
        )


@dataclass(frozen=True, slots=True)
class BedrockCompletion:
    """One Converse response, split into what the model answered, what it thought, and its cost.

    ``reasoning`` is empty whenever the trace is not legible rather than absent: GPT-5.6 returns
    ``reasoningContent.redactedContent``, an encrypted blob, so of the verified models only
    GPT-OSS can be read. That asymmetry is structural and constrains any measurement that scores
    reasoning rather than answers.

    ``stop_reason`` is why the model stopped generating (``end_turn``, ``max_tokens``, ...), and it
    is what separates a reply that declined to answer from one that ran out of output budget before
    it got there. Without it, a token cap manufactures apparent non-compliance: at 16,384 tokens
    the missing-answer rate across three roster models was 21%, 54% and 85%, none of it a model
    property. It carries no default for the reason ``transport`` does not: a field sometimes
    silently absent would put every unlabelled reply in the "declined" bucket.

    The three trailing fields are the live path's per-call telemetry, and they default to ``None``
    because only the live path can measure them: a batch result file has no latency to report and a
    stored fixture predates them. ``elapsed_seconds`` is the call's whole wall clock, request to
    last event; ``first_event_seconds`` is the time to the first streamed event, which on a
    reasoning model is mostly time spent thinking before any token is sent (Sonnet 5 emits nothing
    over Converse while it thinks, so this is the number that tells a long think from a wedged
    socket); ``attempts`` is how many times botocore sent the request before one was accepted.
    Read the two together: both clocks start at the FIRST attempt, so botocore's internal retries
    are inside them, and a row with ``attempts=2`` and ``first_event_seconds`` of ~900 is a first
    attempt that died silently for the whole read timeout before the retry answered, not a
    fifteen-minute think (three of 64 rows in the first live smoke had exactly that shape).
    They exist because the only per-call latency record used to be an INFO line in a ``/tmp`` log
    that was gone for every production run an audit tried to read, so the barrier loss a chunked
    sweep pays on its slowest call could only be simulated, never measured.
    """

    text: str
    reasoning: str
    usage: TokenUsage
    stop_reason: str | None
    elapsed_seconds: float | None = None
    first_event_seconds: float | None = None
    attempts: int | None = None


class CompletionTelemetry(TypedDict):
    """The five per-call accounting fields of :func:`completion_telemetry`, typed one by one.

    A ``TypedDict`` rather than ``dict[str, Any]`` so that a record built by unpacking it -- as
    :func:`raw_response` does -- is checked field for field against the record's own declaration,
    and a renamed or dropped key fails the type check instead of landing as an unknown keyword or a
    silent ``None``.
    """

    cache_read_input_tokens: int | None
    cache_write_input_tokens: int | None
    elapsed_seconds: float | None
    first_event_seconds: float | None
    attempts: int | None


def completion_telemetry(completion: BedrockCompletion) -> CompletionTelemetry:
    """Flatten the per-call accounting a record carries beyond its token totals, under one spelling.

    The studies' reply rows, both judges' rows (under a ``judge_`` prefix) and :class:`RawResponse`
    all carry these five, and a cost script joins them on the names: the cache split is what turns
    a pass's list-price figure into a real one (a rubric cached on every call bills at a tenth of
    the uncached rate), and the two clocks plus the attempt count are what let a slow pass be read
    off the artifact instead of a ``/tmp`` log that is gone. It lives here, beside the
    :class:`BedrockCompletion` it reads, because ``sociology`` builds on this package and not the
    reverse: a copy in ``sociology.records`` and a hand-spelled twin in the hatch judge were held to
    the same five names only by their tests. ``None`` where the transport cannot measure a value (a
    batch result file has no clocks), never zero.
    """
    return CompletionTelemetry(
        cache_read_input_tokens=completion.usage.cache_read_input_tokens,
        cache_write_input_tokens=completion.usage.cache_write_input_tokens,
        elapsed_seconds=completion.elapsed_seconds,
        first_event_seconds=completion.first_event_seconds,
        attempts=completion.attempts,
    )


@dataclass(frozen=True, slots=True)
class TokenizedCompletion:
    """One completion together with the exact token ids the engine read and then wrote.

    Only the interpretability capture reads the ids: it runs a HuggingFace forward pass over
    prompt+response and labels every position as prompt or response, so it needs the boundary in
    tokens rather than in characters. Reconstructing that by re-tokenising the decoded text would
    shift the labelling at any BPE or special-token boundary the decode does not round-trip, and the
    result would still be a well-formed tensor of the wrong positions. Every other caller reads
    ``completion`` and never looks at the ids.

    Tuples rather than lists so a record cannot be mutated after the fact by whoever holds it, which
    matters here because the ids ARE the provenance of everything measured off them.
    """

    completion: BedrockCompletion
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]


@runtime_checkable
class DetailedBackend(Protocol):
    """A backend that also reports reasoning and token usage, which the base protocol discards."""

    def generate_detailed(self, prompts: list[str]) -> Sequence[Any]:
        """Return one richer completion per prompt, carrying ``text`` and ``usage`` at minimum."""
        ...


@dataclass(frozen=True, slots=True)
class RawResponse:
    """What one prompt got back, normalised across backends and before any grading.

    Five heterogeneous fields, two of them ``int | None`` and two of them ``str``, is exactly the
    shape where a transposed pair passes every type check and silently attributes one field's value
    to another.

    The five defaulted fields after them are the per-call accounting a detailed backend can report
    and a plain one cannot: the prompt-cache split of the input count and the live path's latency
    telemetry (:class:`CompletionTelemetry`, spelled once for every record that carries them).
    ``None`` means the backend could not say, never zero
    -- a plain ``Backend`` reports ``None`` for all five just as it does for the token counts, and
    a batch completion reports the cache split but no latency.
    """

    text: str
    reasoning: str
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: str | None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    elapsed_seconds: float | None = None
    first_event_seconds: float | None = None
    attempts: int | None = None


def raw_response(completion: BedrockCompletion) -> RawResponse:
    """Convert one detailed backend completion into the transport-neutral record.

    Lives beside the ``Backend`` protocol and the ``BedrockCompletion`` whose shapes it reconciles,
    rather than inside a benchmark runner: both real transports go through it -- the live Converse
    path via :func:`generate_raw` and the batch path at collect -- so it is on the critical path of
    every real run and belongs to neither benchmark. The five accounting fields come through
    :func:`completion_telemetry`, so this record and the study rows spell them once.
    """
    return RawResponse(
        text=completion.text,
        reasoning=completion.reasoning,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
        stop_reason=completion.stop_reason,
        **completion_telemetry(completion),
    )


def generate_raw(backend: Backend, prompts: list[str]) -> list[RawResponse]:
    """Sample a backend, filling in what the plain ``Backend`` protocol cannot report.

    A backend that only satisfies ``Backend`` returns bare strings, so its records carry ``None``
    for the token counts and the stop reason -- the field present and empty, never an estimate.
    """
    if isinstance(backend, DetailedBackend):
        return [raw_response(completion) for completion in backend.generate_detailed(prompts)]
    return [
        RawResponse(
            text=text, reasoning="", input_tokens=None, output_tokens=None, stop_reason=None
        )
        for text in backend.generate(prompts)
    ]


@runtime_checkable
class StreamingBackend(Protocol):
    """A detailed backend that can keep a bounded queue of calls in flight across a whole sweep.

    ``submit_stream`` yields ``(index, completion)`` pairs in completion order as calls land, never
    letting the queue drain between the chunks a caller persists by. Only the live Converse backend
    implements it: the batch transport has one job for the whole sweep and nothing to overlap, and
    the local engines batch internally. Callers reach it through :func:`stream_detailed_in_chunks`
    and :func:`generate_raw_in_chunks`, which fall back to per-chunk calls for everything else, so a
    runner is written once and gets the continuous queue wherever a backend can offer one.
    """

    def submit_stream(self, prompts: Iterable[str]) -> Iterator[tuple[int, BedrockCompletion]]:
        """Yield one ``(request index, completion)`` per prompt, in the order the calls finish."""
        ...


def _chunk_bounds(chunks: Sequence[Sequence[str]]) -> list[tuple[int, int]]:
    """Compute the half-open index range each chunk occupies in the flattened prompt list."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for chunk in chunks:
        bounds.append((start, start + len(chunk)))
        start += len(chunk)
    return bounds


class StreamAlignmentError(RuntimeError):
    """A backend broke the one-completion-per-prompt contract, so nothing of that chunk is handed over.

    On a stream: a duplicate, an out-of-range or a missing index. On the per-chunk fallback: a
    results list of the wrong length. Raised INSTEAD of handing anything over, because the property
    that makes a hand-over safe -- every yielded index names exactly one request, exactly once, and
    a stream that ends normally has yielded them all -- is the property that just failed, and
    persisting what was buffered would be a plausible-looking trace with replies under the wrong
    prompts. ``zip(strict=True)`` in the hand-written per-chunk loops was the original tripwire;
    this is it, restated for both paths of :func:`stream_detailed_in_chunks`. Today's
    :meth:`BedrockBackend.submit_stream` yields every index or raises, so on a stream this fires
    only for a backend that is wrong, and the repository asserts alignment rather than inferring it.
    """


_MISSING_INDICES_NAMED = 8
"""How many missing indices a short-stream refusal spells out before eliding the rest."""


class _ChunkReleaser:
    """The bookkeeping of :func:`_stream_chunks`: buffer by index, release whole chunks.

    In order by default: a chunk is released only when every position in it has landed AND every
    earlier chunk has already been released, so what a caller persists is exactly what the
    per-chunk path would have persisted, in the same order. With ``in_order=False`` a chunk is
    released the moment its own last call lands, whatever chunks before it still wait, which is
    the opt-in :func:`stream_detailed_in_chunks` documents. Everything that can go wrong with the
    stream is decided here too: a misfiled index refuses (:class:`StreamAlignmentError`), a stream
    that ends short refuses, and the backlog a wedged call builds up in memory is logged as it
    grows.
    """

    def __init__(self, bounds: Sequence[tuple[int, int]], *, in_order: bool = True) -> None:
        self._bounds = bounds
        self._in_order = in_order
        self._total = bounds[-1][1] if bounds else 0
        self._starts = [start for start, _ in bounds]
        self._empty = [index for index, (start, end) in enumerate(bounds) if start == end]
        self._buffered: dict[int, BedrockCompletion] = {}
        self._seen: set[int] = set()
        self._released: set[int] = set()
        self._backlog_warned_at = 0

    def absorb(
        self, index: int, completion: BedrockCompletion
    ) -> Iterator[tuple[int, list[tuple[int, BedrockCompletion]]]]:
        """File one landed completion and release every chunk it made ready."""
        self._refuse_misfiled(index)
        self._seen.add(index)
        self._buffered[index] = completion
        yield from self._release_ready(index)
        self._warn_on_growing_backlog()

    def hand_over_partials(self) -> Iterator[tuple[int, list[tuple[int, BedrockCompletion]]]]:
        """Release the finished part of every unreleased chunk, in chunk order and position order."""
        for chunk_index in self._unreleased():
            start, end = self._bounds[chunk_index]
            partial = [
                (index - start, self._buffered.pop(index))
                for index in range(start, end)
                if index in self._buffered
            ]
            if partial:
                yield chunk_index, partial

    def refuse_unreleased(self) -> None:
        """Refuse a stream that ended normally with chunks still unreleased."""
        unreleased = self._unreleased()
        if not unreleased:
            return
        missing = sorted(set(range(self._total)) - self._seen)
        named = missing[:_MISSING_INDICES_NAMED]
        raise StreamAlignmentError(
            f"the stream ended after {len(self._seen)} of {self._total} completions without "
            f"raising, leaving {len(unreleased)} of {len(self._bounds)} chunks "
            f"unreleased; missing indices {named}{'...' if len(missing) > len(named) else ''}"
        )

    def _unreleased(self) -> list[int]:
        return [index for index in range(len(self._bounds)) if index not in self._released]

    def _refuse_misfiled(self, index: int) -> None:
        if not 0 <= index < self._total:
            raise StreamAlignmentError(
                f"the backend yielded index {index} for a stream of {self._total} prompts"
            )
        if index in self._seen:
            raise StreamAlignmentError(f"the backend yielded index {index} twice")

    def _is_whole(self, chunk_index: int) -> bool:
        start, end = self._bounds[chunk_index]
        return all(index in self._buffered for index in range(start, end))

    def _release(self, chunk_index: int) -> tuple[int, list[tuple[int, BedrockCompletion]]]:
        start, end = self._bounds[chunk_index]
        self._released.add(chunk_index)
        return chunk_index, [
            (index - start, self._buffered.pop(index)) for index in range(start, end)
        ]

    def _release_ready(
        self, landed: int
    ) -> Iterator[tuple[int, list[tuple[int, BedrockCompletion]]]]:
        if self._in_order:
            # In-order releases are contiguous from the front, so the head is the count released.
            while len(self._released) < len(self._bounds) and self._is_whole(len(self._released)):
                yield self._release(len(self._released))
            return
        # An empty chunk has no call to land, so it is whole from the start and no landed index
        # ever points at it; release it at the first opportunity, as the in-order loop does when
        # the head reaches it, or the clean stream is refused at its end for a chunk never released.
        for chunk_index in self._empty:
            if chunk_index not in self._released:
                yield self._release(chunk_index)
        # Otherwise only the chunk the landed index belongs to can have become whole.
        chunk_index = bisect_right(self._starts, landed) - 1
        if self._is_whole(chunk_index):
            yield self._release(chunk_index)

    def _warn_on_growing_backlog(self) -> None:
        """Log, once per chunk's worth of growth, what a process death now would lose."""
        if len(self._released) == len(self._bounds):
            return
        if not self._in_order:
            self._warn_on_growing_partials()
            return
        start, end = self._bounds[len(self._released)]
        landed_in_head = sum(1 for index in range(start, end) if index in self._buffered)
        behind = len(self._buffered) - landed_in_head
        if behind < self._backlog_warned_at + (end - start):
            return
        self._backlog_warned_at = behind
        logger.warning(
            "%d finished completions are buffered in memory behind chunk %d of %d, which still "
            "waits on %d of its %d calls; they are billed and not on disk, and a process death now "
            "loses every one of them",
            behind,
            len(self._released) + 1,
            len(self._bounds),
            end - start - landed_in_head,
            end - start,
        )

    def _warn_on_growing_partials(self) -> None:
        """Log the out-of-order backlog: every buffered completion sits in a chunk still waiting on a call."""
        buffered = len(self._buffered)
        widest = max(end - start for start, end in self._bounds)
        if buffered < self._backlog_warned_at + widest:
            return
        self._backlog_warned_at = buffered
        partial = sum(
            1
            for chunk_index in self._unreleased()
            if any(index in self._buffered for index in range(*self._bounds[chunk_index]))
        )
        logger.warning(
            "%d finished completions are buffered in memory across %d partial chunks, each still "
            "waiting on a call of its own; they are billed and not on disk, and a process death now "
            "loses every one of them",
            buffered,
            partial,
        )


def _stream_chunks(
    backend: StreamingBackend, chunks: Sequence[Sequence[str]], *, in_order: bool
) -> Iterator[tuple[int, list[tuple[int, BedrockCompletion]]]]:
    """Consume one continuous stream over every chunk's prompts, releasing chunks as they become whole.

    Completions are buffered by request index. In order (the default), a chunk is released only
    when every position in it has landed AND every earlier chunk has already been released
    (:class:`_ChunkReleaser`), so what a caller persists is exactly what the per-chunk path would
    have persisted, in the same order -- while the backend never waits on a chunk boundary. A chunk
    whose one slow call is still running holds up the chunks behind it on disk but not in flight.

    That is also the cost of the in-order design, and it is unbounded in the one case that matters:
    while the head chunk waits on a wedged call (900-3,000 s read timeouts are the measured wedge),
    the queue keeps landing completions for the chunks behind it, and every one of them sits in
    memory, billed and not on disk, until the head releases. The per-chunk barrier capped a process
    death at ``chunk - 1`` lost completions; the in-order loop caps it at nothing. So the backlog is
    logged at WARNING every time it grows by another head-chunk's worth, naming what a death would
    lose, and the hand-over below catches ``BaseException`` rather than ``Exception`` so that the two
    deaths a process can still act on -- a Ctrl-C (``KeyboardInterrupt``) and a ``SystemExit`` --
    persist what was buffered before they propagate. A SIGKILL, an OOM kill or a pulled plug still
    loses it all. ``in_order=False`` is the bound for those: a chunk is released the moment its own
    last call lands, so only the partial chunks are ever in memory, at the price of the on-disk-order
    promise every runner's byte-identity test is written against -- which is why it is an opt-in at
    :func:`stream_detailed_in_chunks` rather than the default.

    If the stream raises, the completed portion of every unreleased chunk is released first, in
    chunk order and position order, and the error then propagates. Those completions are paid for,
    and a caller that resumes by key needs them on disk; the broad catch exists only to hand them
    over before re-raising, which is the one shape of a blind ``except`` this repository allows.
    ``GeneratorExit`` is re-raised untouched because it means the CONSUMER stopped listening (its
    own write failed, or it broke out of the loop) and a generator that yields inside that handler
    is itself an error; :class:`StreamAlignmentError` is re-raised untouched because nothing
    buffered can be trusted once the index contract broke, and a stream that ends normally with
    chunks unreleased is refused the same way rather than returning as if finished.

    Every release carries its CHUNK INDEX. On the failure path an unfinished earlier chunk may have
    nothing to hand over while a later one is whole, so a consumer counting releases would pair the
    later chunk's completions with the earlier chunk's prompts -- caught once, offline, as a reply
    filed under a neighbouring key -- and the index is what makes that pairing impossible to get
    wrong. Under ``in_order=False`` the index is the ONLY thing that says which chunk a release is.
    """
    releaser = _ChunkReleaser(_chunk_bounds(chunks), in_order=in_order)
    try:
        for index, completion in backend.submit_stream(
            prompt for chunk in chunks for prompt in chunk
        ):
            yield from releaser.absorb(index, completion)
    except GeneratorExit:
        raise
    except StreamAlignmentError:
        raise
    except BaseException:
        yield from releaser.hand_over_partials()
        raise
    releaser.refuse_unreleased()


def stream_detailed_in_chunks(
    backend: DetailedBackend, chunks: Sequence[Sequence[str]], *, persist_out_of_order: bool = False
) -> Generator[tuple[int, list[tuple[int, BedrockCompletion]]]]:
    """Yield ``(chunk index, [(position, completion), ...])`` in chunk order, queue kept full throughout.

    The one loop every chunk-persisting runner should drive. On a :class:`StreamingBackend` the
    backend's queue stays ``concurrency`` deep across chunk boundaries (:func:`_stream_chunks`);
    on any other detailed backend it is one ``generate_detailed`` per chunk, which is exactly the
    loop the runners used to write by hand, so the fallback is byte-for-byte the old behaviour --
    including the old loop's length tripwire. A results list of the wrong length is refused with
    :class:`StreamAlignmentError` before any of that chunk is yielded: positions pair prompts with
    completions only when there is exactly one per prompt, and without the check a short list was
    filed short while the pass reported itself whole (reproduced 2026-09-02 on a judge stub: three
    rows on disk, ``judged: 4``, no raise). Chunks yielded before it stay yielded, as on the
    streaming path.

    On a completed run every chunk is yielded, whole and in position order, so ``zip(chunk, pairs,
    strict=True)`` pairs prompts with their completions. On a raise the streaming path may yield
    shorter lists first (the completed part of each unfinished chunk) before the error propagates,
    and may skip a chunk with nothing finished; the fallback never yields a short chunk. A caller
    that persists them must look the chunk up by the yielded index and pair by position, never by
    counting releases or by ``zip``.

    ``persist_out_of_order`` relaxes the one promise the default keeps, that chunks are yielded in
    request order. Under it a chunk is yielded the moment its own last call lands, whatever chunks
    before it still wait, so what a wedged head call can lose to a SIGKILL or an OOM kill is bounded
    again: only the partial chunks are in memory, and since the queue submits in request order with
    ``concurrency`` calls in flight, at most ``concurrency + 1`` chunks are partial at once, so the
    loss is at most about ``(concurrency + 1) * (chunk - 1)`` completions rather than everything
    behind the head. The price is the on-disk order: a trace written under the flag holds its chunks
    in completion order, so a byte-identity comparison against a per-chunk run no longer holds, and
    a reader has to key on record identity rather than position, which every resume-by-key runner
    already does. Off by default because every runner's byte-identity test is written against
    request order; opt in per run and name it in the run's report. On the per-chunk fallback the
    flag changes nothing, since there is no queue to run ahead of the chunk being persisted.
    """
    if isinstance(backend, StreamingBackend):
        yield from _stream_chunks(backend, chunks, in_order=not persist_out_of_order)
        return
    for chunk_index, chunk in enumerate(chunks):
        completions = backend.generate_detailed(list(chunk))
        if len(completions) != len(chunk):
            raise StreamAlignmentError(
                f"the backend returned {len(completions)} completions for the {len(chunk)} prompts "
                f"of chunk {chunk_index + 1} of {len(chunks)}; positions pair prompts with "
                f"completions only one to one, so nothing of this chunk is handed over"
            )
        yield chunk_index, list(enumerate(completions))


def generate_raw_in_chunks(
    backend: Backend, chunks: Sequence[Sequence[str]], *, persist_out_of_order: bool = False
) -> Iterator[tuple[int, list[tuple[int, RawResponse]]]]:
    """Run the :func:`stream_detailed_in_chunks` loop over transport-neutral records.

    :func:`generate_raw`'s chunked twin: a detailed backend's completions go through
    :func:`raw_response` exactly as they do there, and a plain ``Backend`` is sampled one chunk at
    a time with the token counts honestly absent. Same ``(chunk index, pairs)`` shape, the same
    partial-chunk contract on a raise, and the same ``persist_out_of_order`` opt-in, passed through.
    """
    if isinstance(backend, DetailedBackend):
        for chunk_index, pairs in stream_detailed_in_chunks(
            backend, chunks, persist_out_of_order=persist_out_of_order
        ):
            yield (
                chunk_index,
                [(position, raw_response(completion)) for position, completion in pairs],
            )
        return
    for chunk_index, chunk in enumerate(chunks):
        yield chunk_index, list(enumerate(generate_raw(backend, list(chunk))))


def _join_text_blocks(blocks: Sequence[Mapping[str, Any]]) -> str:
    """Join the answer text only, so a reasoning block is never scored as part of the answer.

    A union member that is absent and one that is present-but-``null`` mean the same thing -- this
    block is not of that kind -- and both spellings occur for real. botocore drops unset members
    from a live Converse response, so the live path sees ``{"text": ...}`` alone; the batch
    inference result file in S3 serialises *every* member of the content-block union with an
    explicit ``null``, so a real batch block carries ``text``, ``reasoningContent``, ``toolUse``
    and nine more keys at once. Keying on key presence therefore reads ``None`` as answer text and
    dies on the join, which is exactly what the first parse of a real batch record did.
    """
    return " ".join(text for block in blocks if (text := block.get("text")) is not None)


def _join_reasoning_blocks(blocks: Sequence[Mapping[str, Any]]) -> str:
    """Join whatever readable reasoning came back, skipping encrypted and absent traces alike.

    Null-tolerant at every level for the reason given on ``_join_text_blocks``: on the batch path
    ``reasoningContent`` is present and ``None`` on the answer block, and ``reasoningText`` is
    present and ``None`` on Luna's redacted trace.
    """
    texts: list[str] = []
    for block in blocks:
        reasoning_content = block.get("reasoningContent")
        if reasoning_content is None:
            continue
        reasoning_text = reasoning_content.get("reasoningText")
        if reasoning_text is None:
            continue
        text = reasoning_text.get("text")
        if text is not None:
            texts.append(text)
    return " ".join(texts)


# The input-token counters, grouped so that each group is one quantity under all of its spellings.
_INPUT_TOKEN_FIELDS: tuple[tuple[str, ...], ...] = (
    ("inputTokens", "inputTokenCount"),
    ("cacheReadInputTokens", "cacheReadInputTokenCount"),
    ("cacheWriteInputTokens", "cacheWriteInputTokenCount"),
)


def _alias_value(usage: Mapping[str, Any], names: Sequence[str]) -> int:
    """Read one token counter that Bedrock spells several ways, treating null as absent.

    A batch result file carries ``cacheReadInputTokens`` beside ``cacheReadInputTokenCount``, and
    every batch job in this account reports both as null, so which one is authoritative could not
    be settled from data. Reducing by max encodes the assumption that they are aliases: aliases
    agree, so max is their shared value, and where only one is populated max picks it. Summing
    would double-count a cached prompt and inflate input cost with no symptom. The warning below is
    what turns the assumption into something we would hear about if it is wrong.

    Every counter's group has to list both spellings, symmetrically. The bare names are what a
    per-record ``usage`` block uses and the ``...Count`` names are the job manifest's vocabulary, so
    a group carrying only one of them reads zero for a mapping written in the other -- and because
    the warning fires only when two *present* values disagree, an absent alias undercounts in
    silence. That is exactly the half-count these alias groups exist to prevent.
    """
    values = [value for name in names if (value := usage.get(name)) is not None]
    if len({value for value in values if value}) > 1:
        logger.warning(
            "token counters %s disagree (%s); they were assumed to be aliases, so the largest is "
            "used and the input-token total may be wrong",
            list(names),
            values,
        )
    return max(values, default=0)


def _sum_usage(usage: Mapping[str, Any]) -> TokenUsage:
    """Sum all three input-token fields, because the prompt cache moves input between them.

    GPT-5.6 Luna reported a ~1,700-token prompt as ``inputTokens: 2`` alongside
    ``cacheWriteInputTokens: 1619``, then flipped to ``cacheReadInputTokens`` on a later run
    because the cache outlives the process that filled it. Reading ``inputTokens`` alone
    understates that prompt by nearly three orders of magnitude. GPT-OSS does no caching and
    reports zero for both cache fields, so summing is correct there too.
    """
    uncached, cache_read, cache_write = (
        _alias_value(usage, names) for names in _INPUT_TOKEN_FIELDS
    )
    return TokenUsage(
        input_tokens=uncached + cache_read + cache_write,
        output_tokens=_alias_value(usage, ("outputTokens", "outputTokenCount")),
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )


def converse_inference_config(sampling: BedrockSamplingConfig) -> dict[str, object]:
    """Build the Converse ``inferenceConfig``, omitting every knob the run did not ask for.

    Omission is the verified request shape across the model roster, and it is load-bearing rather
    than tidy: several hosted models reject a field they do not honour, and Anthropic models reject
    ``temperature`` and ``topP`` when both are present -- a batch job in this account came back
    ``Completed`` with 500 of 500 records dead on exactly that.

    ``maxTokens`` is the exception and is written unconditionally. That is this code's choice, not
    the API's: probed live 2026-09-03 on gpt-oss-20b, both ``Converse`` and ``ConverseStream``
    accept a request with no ``inferenceConfig`` at all and run at the model's own output ceiling
    (:data:`DEFAULT_BEDROCK_MAX_TOKENS` has the numbers and what still keeps the field in).
    """
    config: dict[str, object] = {"maxTokens": sampling.max_tokens}
    if sampling.temperature is not None:
        config["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        config["topP"] = sampling.top_p
    if sampling.stop_sequences:
        config["stopSequences"] = list(sampling.stop_sequences)
    return config


def converse_extra_request_fields(
    model_id: str, sampling: BedrockSamplingConfig
) -> dict[str, object] | None:
    """Build ``additionalModelRequestFields`` for this model, or ``None`` when nothing is asked."""
    if sampling.reasoning_effort is None:
        return None
    return _resolve_reasoning_dialect(model_id).request_fields(sampling.reasoning_effort)


def converse_request(
    prompt: str,
    inference_config: Mapping[str, object],
    extra_request_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Render one prompt into a Converse request body, without ``modelId``.

    The one place a prompt becomes a Bedrock payload. The live backend hands this to
    ``client.converse_stream(modelId=..., **request)``; the batch backend writes the identical dict
    as a record's ``modelInput``, which is exactly what ``modelInvocationType="Converse"`` expects.
    A second renderer would let the two transports drift, and a drift there shows up as a
    model-behaviour difference rather than as a bug.

    The same three keys serve both ``Converse`` and ``ConverseStream``: their request shapes in the
    installed botocore service model share ``messages``, ``inferenceConfig`` and
    ``additionalModelRequestFields``, so switching the live path to streaming did not fork this
    renderer and the batch path's records did not change.
    """
    request: dict[str, object] = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": dict(inference_config),
    }
    if extra_request_fields is not None:
        request["additionalModelRequestFields"] = dict(extra_request_fields)
    return request


def parse_converse_output(body: Mapping[str, Any]) -> BedrockCompletion:
    """Split a Converse response body into answer, readable reasoning and token usage.

    The counterpart to ``converse_request``, and shared for the same reason: a batch result file's
    ``modelOutput`` *is* this body, so both transports must parse it with the same code or a
    parsing difference becomes a finding about the model. Reading ``stopReason`` here is what makes
    the batch path inherit truncation labelling for free.

    ``stopReason`` is read with ``get`` rather than indexed: live Converse always returns it, but a
    stored fixture or a hand-built body predates the field, and a missing label is honestly None
    rather than a crash.
    """
    blocks = body["output"]["message"]["content"]
    return BedrockCompletion(
        text=_join_text_blocks(blocks),
        reasoning=_join_reasoning_blocks(blocks),
        usage=_sum_usage(body["usage"]),
        stop_reason=body.get("stopReason"),
    )


class ConverseEventStream(Protocol):
    """The part of botocore's ``EventStream`` the accumulator below uses.

    A protocol rather than the botocore class because ``boto3`` is an optional extra: naming
    ``botocore.eventstream.EventStream`` in an annotation would make this module unimportable
    without it, which is the whole reason the SDK is imported lazily inside the backend.
    """

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Yield one parsed event dict at a time, blocking until the next one arrives."""
        ...

    def close(self) -> None:
        """Release the underlying HTTP response, abandoning any events not yet read."""
        ...


def _rebuild_content_blocks(
    text_fragments: Mapping[int, list[str]], reasoning_fragments: Mapping[int, list[str]]
) -> list[dict[str, Any]]:
    """Join each streamed block's fragments back into the non-streaming content-block list.

    Ordered by ``contentBlockIndex``, since that is the order the blocks were generated in and the
    order a non-streaming response returns them in; a set would reorder the answer against the
    reasoning that produced it. A single index carrying both kinds is emitted as one block with both
    members, which is what the two joiners downstream already read independently -- no observed
    model does that, but silently dropping one of the two would be a lost trace rather than a crash.
    """
    blocks: list[dict[str, Any]] = []
    for index in sorted(set(text_fragments) | set(reasoning_fragments)):
        block: dict[str, Any] = {}
        if index in text_fragments:
            block["text"] = "".join(text_fragments[index])
        if index in reasoning_fragments:
            joined = "".join(reasoning_fragments[index])
            block["reasoningContent"] = {"reasoningText": {"text": joined}}
        blocks.append(block)
    return blocks


def _absorb_delta(
    delta_event: Mapping[str, Any],
    text_fragments: dict[int, list[str]],
    reasoning_fragments: dict[int, list[str]],
) -> None:
    """File one ``contentBlockDelta`` under its block index, answer text and reasoning apart.

    A reasoning delta carries ``reasoningContent`` where an answer delta carries ``text``, and only
    readable reasoning (``reasoningContent.text``) is kept: ``redactedContent`` is an encrypted
    blob and ``signature`` an attestation, and neither is a trace anybody can read.
    """
    index = int(delta_event["contentBlockIndex"])
    delta: Mapping[str, Any] = delta_event["delta"]
    text: str | None = delta.get("text")
    if text is not None:
        text_fragments.setdefault(index, []).append(text)
    reasoning: Mapping[str, Any] | None = delta.get("reasoningContent")
    if reasoning is not None:
        reasoning_text: str | None = reasoning.get("text")
        if reasoning_text is not None:
            reasoning_fragments.setdefault(index, []).append(reasoning_text)


def accumulate_converse_stream(
    stream: ConverseEventStream, *, deadline: float, model_id: str
) -> tuple[dict[str, Any], float | None]:
    """Rebuild the non-streaming Converse response body from a ``converse_stream`` event stream.

    Returns the body and the :func:`time.monotonic` stamp of the first event read, or ``None`` when
    the stream delivered nothing. The stamp is the raw material of ``first_event_seconds`` on the
    completion: on a reasoning model the gap before the first event is thinking time, and it is the
    one measurement that separates a long think from a socket that will never answer.

    Deliberately reconstructs the *body* rather than a :class:`BedrockCompletion`, so
    :func:`parse_converse_output` stays the single reader of a Converse response for all three
    transports. The alternative -- a second joiner that walks the deltas straight into text and
    reasoning strings -- is exactly the drift :func:`converse_request` and
    :func:`parse_converse_output` were paired to prevent, and a drift there would read as a
    difference between models rather than as a bug: the live path would score reasoning as answer
    while the batch path did not.

    Two levels of joining, and they are not the same operation. Deltas *within* one content block
    are token fragments and concatenate with nothing between them; blocks are joined with a space,
    which is what ``_join_text_blocks`` already does downstream. ``contentBlockIndex`` is what keeps
    them apart, and it is also what keeps a reasoning trace out of the answer: a reasoning delta
    carries ``reasoningContent`` where an answer delta carries ``text``, and each accumulates into
    its own block. Collapse that and every downstream label built on the answer is silently wrong.

    Event shapes verified against the ``ConverseStream`` output union in the installed botocore's
    ``bedrock-runtime`` service model (1.43.72), not recalled: ``contentBlockDelta`` carries
    ``delta`` plus ``contentBlockIndex``, its ``delta`` is a union of ``text`` /
    ``reasoningContent`` / ``toolUse`` / ``toolResult`` / ``citation`` / ``image``, its
    ``reasoningContent`` is a union of ``text`` / ``redactedContent`` / ``signature``,
    ``messageStop`` carries ``stopReason``, and ``metadata`` carries ``usage``. Only readable
    reasoning is kept, matching :func:`_join_reasoning_blocks`: a ``redactedContent`` delta is an
    encrypted blob and a ``signature`` delta is an attestation, and neither is a trace anybody can
    read. Modelled service errors do not arrive as members of this union in practice -- botocore
    raises ``EventStreamError`` out of the iterator instead -- so they propagate rather than being
    parsed here, and :meth:`BedrockBackend._call_isolated` is where one becomes a counted
    ``call_failed`` record instead of a dead batch.

    ``deadline`` is an absolute :func:`time.monotonic` value, checked after each event so the loop
    never enters another blocking read past it. Once ``messageStop`` has arrived the deadline stops
    applying: ``metadata`` follows it immediately and carries the token counts, and abandoning
    between the two would throw away the cost record of a call that had in fact completed.

    A stream that runs out of events without a ``messageStop`` raises
    :class:`IncompleteConverseStreamError`. That is not the deadline case and must not be labelled
    as one -- a wrong label on a partial record is worse than a counted failure. The raise was a
    bare ``RuntimeError`` until the failure was observed live and named (see the class docstring);
    :meth:`BedrockBackend._call_isolated` now counts it instead of letting it kill the batch.
    """
    text_fragments: dict[int, list[str]] = {}
    reasoning_fragments: dict[int, list[str]] = {}
    usage: Mapping[str, Any] = {}
    stop_reason: str | None = None
    message_ended = False
    abandoned = False
    first_event_at: float | None = None
    for event in stream:
        if first_event_at is None:
            first_event_at = time.monotonic()
        delta_event: Mapping[str, Any] | None = event.get("contentBlockDelta")
        if delta_event is not None:
            _absorb_delta(delta_event, text_fragments, reasoning_fragments)
        message_stop: Mapping[str, Any] | None = event.get("messageStop")
        if message_stop is not None:
            message_ended = True
            stop_reason = message_stop.get("stopReason")
        metadata: Mapping[str, Any] | None = event.get("metadata")
        if metadata is not None:
            usage = metadata.get("usage") or {}
        if not message_ended and time.monotonic() >= deadline:
            stream.close()
            abandoned = True
            break
    if not message_ended and not abandoned:
        raise IncompleteConverseStreamError(
            f"the {model_id} converse_stream ended without a messageStop event after "
            f"{len(text_fragments)} answer and {len(reasoning_fragments)} reasoning blocks; "
            "the reply is incomplete for a reason this code cannot name, and labelling it as a "
            "deadline abandonment would misattribute it"
        )
    body = {
        "output": {
            "message": {"content": _rebuild_content_blocks(text_fragments, reasoning_fragments)}
        },
        "usage": dict(usage),
        "stopReason": STOP_REASON_DEADLINE_EXCEEDED if abandoned else stop_reason,
    }
    return body, first_event_at


class BedrockBackend:
    """AWS Bedrock ConverseStream backend: the hosted-API path, and the only cross-vendor one.

    ``boto3`` is an optional extra (``uv sync --extra bedrock``), imported lazily here the same
    way ``VLLMBackend`` imports ``vllm``, so this module still loads with it absent.

    Four choices in here are load-bearing and were settled against live endpoints or the installed
    service model rather than read off a doc page. Converse rather than ``invoke-model``, because
    Converse returns reasoning and answer as separate content blocks where the native path
    concatenates them into one string that a naive parser scores as the answer. Retries left
    entirely to botocore's adaptive mode, because Bedrock's transient 500s arrive both as
    ``InternalServerException`` and as a bare ``ClientError`` with code ``"500"`` that maps to no
    modelled exception class, so no hand-written ``except`` could cover them even if this repo
    allowed the broad one that would -- capped at
    :data:`DEFAULT_BEDROCK_MAX_ATTEMPTS`, see there for what the cap bounds.  And
    ``max_pool_connections`` above the worker count, because urllib3 otherwise serialises the burst
    and the achieved concurrency silently becomes the pool size instead of the requested one.

    The fourth is ``converse_stream`` rather than ``converse``, and it is what makes a run's wall
    clock a number anybody can state. A socket read timeout bounds the gap between two reads, never
    the call, so a model that keeps producing tokens runs as long as it likes: on hard arms MiniMax
    M2.5 takes 2,900-3,000+ seconds per call and 11 of 24 calls in one pass died at 2,700 seconds,
    losses that correlate with reasoning length and so bias the surviving denominator. Streaming
    puts the loop in this process, where a deadline can be enforced and a call past it can hand back
    what arrived. ``transport`` deliberately stays ``bedrock-converse``: the request payload
    :func:`converse_request` renders is byte-identical either way, so streaming is a wire framing
    rather than a different elicitation, and relabelling it would split one elicitation across two
    values in every pooled analysis and in every record already on disk. Streaming is also not
    optional -- a model that cannot stream fails loudly on its first call, with no fallback path to
    quietly change what an arm measured.

    Converse takes one prompt per call, so ``generate`` fans a batch across a thread pool and
    reassembles in request order. The request payload and the effort dialect are both resolved in
    ``__init__``, so a mis-specified run dies before spending a token rather than 2,000 calls in.

    ``profile`` names an AWS profile, or is ``None`` to fall back to the ambient credential chain,
    which is what a Batch container holding an instance role has and no named profile. Left
    unspecified it resolves :func:`bedrock_profile` from the environment, and a named profile is the
    default rather than the ambient chain because this box's ambient default profile is a
    placeholder that fails credential resolution outright.

    ``deadline_seconds``, ``read_timeout`` and ``max_attempts`` are constructor arguments rather
    than constants baked into the client, because the three of them are what a run's worst-case wall
    clock is computed from and one roster model needs a different answer than another. They were
    baked in before, and the workaround was reaching into ``_client`` to rebuild it -- which is a
    silent way to end up with two clients configured differently in the same analysis.
    """

    transport = "bedrock-converse"

    def __init__(  # noqa: PLR0913 - flat call-site knobs; a config object would hide the defaults
        self,
        model_id: str,
        *,
        region: str = DEFAULT_BEDROCK_REGION,
        profile: str | None = PROFILE_FROM_ENVIRONMENT,
        concurrency: int = DEFAULT_BEDROCK_CONCURRENCY,
        sampling: BedrockSamplingConfig | None = None,
        deadline_seconds: float = DEFAULT_BEDROCK_DEADLINE_SECONDS,
        read_timeout: int = DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_BEDROCK_MAX_ATTEMPTS,
    ) -> None:
        """Build the bedrock-runtime client and freeze the per-request payload for this model."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        if deadline_seconds <= 0:
            raise ValueError(f"deadline_seconds must be positive, got {deadline_seconds}")
        self.model_id = model_id
        self.concurrency = concurrency
        self.sampling = sampling or BedrockSamplingConfig()
        self.deadline_seconds = deadline_seconds
        self.usage = TokenUsage()
        # The whole-run cost record. += on it is a non-atomic read-modify-write, and generate()
        # is called from several episode threads at once under --episode-concurrency, so an
        # unlocked accumulate silently undercounts the only billing record there is.
        self._usage_lock = threading.Lock()

        self._inference_config = converse_inference_config(self.sampling)
        self._extra_request_fields = converse_extra_request_fields(model_id, self.sampling)

        # After the payload checks so a bad effort still reports the effort, before the session.
        resolved_profile = bedrock_profile() if profile == PROFILE_FROM_ENVIRONMENT else profile
        boto3 = importlib.import_module("boto3")
        botocore_config = importlib.import_module("botocore.config")
        botocore_exceptions = importlib.import_module("botocore.exceptions")
        # Both spellings of the same two transport failures, plus the class botocore raises when
        # the service itself fails mid-stream -- see ``_call_isolated`` for why each is here.
        self._isolated_failures: tuple[type[BaseException], ...] = (
            botocore_exceptions.ReadTimeoutError,
            botocore_exceptions.ConnectTimeoutError,
            botocore_exceptions.ConnectionClosedError,
            botocore_exceptions.EndpointConnectionError,
            botocore_exceptions.EventStreamError,
            urllib3.exceptions.ReadTimeoutError,
            urllib3.exceptions.ProtocolError,
            IncompleteConverseStreamError,
        )
        self._client_error: type[BaseException] = botocore_exceptions.ClientError
        session = boto3.Session(profile_name=resolved_profile, region_name=region)
        self._client = session.client(
            "bedrock-runtime",
            config=botocore_config.Config(
                read_timeout=read_timeout,
                connect_timeout=30,
                max_pool_connections=max(concurrency * 2, 20),
                retries={"max_attempts": max_attempts, "mode": "adaptive"},
            ),
        )
        logger.info(
            "BedrockBackend ready for %s region=%s profile=%s concurrency=%d effort=%s; "
            "one call is bounded at about %.0fs (%d attempts x %ds to start the stream, "
            "+ %.0fs reading it, + %ds of final silence)",
            model_id,
            region,
            resolved_profile,
            concurrency,
            self.sampling.reasoning_effort,
            max_attempts * read_timeout + deadline_seconds + read_timeout,
            max_attempts,
            read_timeout,
            deadline_seconds,
            read_timeout,
        )

    def _converse(self, prompt: str) -> BedrockCompletion:
        """Stream one Converse call under a wall-clock deadline, and split what it returned.

        Raises rather than absorbing a transport failure, so a caller sampling one prompt at a time
        still sees it; :meth:`_call_isolated` is where a batch turns one into a counted record.

        A call abandoned at the deadline is billed for tokens this record cannot report -- the
        ``metadata`` event carrying the counts is the last one to arrive, and it never did. So it
        comes back with zero usage, which makes a run total that includes it a lower bound on spend.
        The warning below is the only place that shortfall is visible, which is why it names the
        elapsed time and the characters that did arrive rather than just saying "deadline".
        """
        request = converse_request(prompt, self._inference_config, self._extra_request_fields)
        started = time.monotonic()
        response = self._client.converse_stream(modelId=self.model_id, **request)
        body, first_event_at = accumulate_converse_stream(
            response["stream"],
            deadline=started + self.deadline_seconds,
            model_id=self.model_id,
        )
        elapsed = time.monotonic() - started
        completion = replace(
            parse_converse_output(body),
            elapsed_seconds=elapsed,
            first_event_seconds=None if first_event_at is None else first_event_at - started,
            attempts=_attempts_from_metadata(response.get("ResponseMetadata")),
        )
        if completion.stop_reason == STOP_REASON_DEADLINE_EXCEEDED:
            logger.warning(
                "BedrockBackend %s abandoned a call at its %.1fs deadline after %.1fs, keeping "
                "%d answer and %d reasoning characters; the call was billed for tokens this "
                "record cannot report, so any run total including it is a lower bound",
                self.model_id,
                self.deadline_seconds,
                elapsed,
                len(completion.text),
                len(completion.reasoning),
            )
        else:
            logger.info(
                "BedrockBackend %s call took %.0fs (first event at %s, %s attempts): stop=%s "
                "out=%d tokens",
                self.model_id,
                elapsed,
                "-"
                if completion.first_event_seconds is None
                else f"{completion.first_event_seconds:.1f}s",
                completion.attempts,
                completion.stop_reason,
                completion.usage.output_tokens,
            )
        return completion

    def _call_isolated(self, prompt: str) -> BedrockCompletion:
        """Run one call, turning an expected transport failure into a counted empty record.

        The closed list resolved in ``__init__`` is deliberately not ``Exception``. A read timeout
        or a dropped connection on a model documented to occasionally never return is expected and
        should cost one sample; anything else -- a malformed payload, an unknown model id, a
        credentials failure -- is a bug in the run and must crash on the first call rather than be
        absorbed into hundreds of empty records that read as a model saying nothing.

        That list spans two packages because one transport failure arrives under two class names
        depending on when it happens, and getting this wrong is not hypothetical: it shipped.
        ``botocore.httpsession`` re-raises urllib3's errors as its own, but only inside the call that
        returns the response object, so a socket that goes quiet there becomes botocore's
        ``ReadTimeoutError`` and a broken connection its ``ConnectionClosedError``. The reply body is
        then read by iterating the event stream (:func:`accumulate_converse_stream`), outside that
        wrapping, where the same two failures reach this seam as urllib3's own ``ReadTimeoutError``
        and ``ProtocolError``. While the list named only the botocore spellings, every failure on the
        streaming path escaped into :meth:`generate_detailed` and discarded the replies its siblings
        had already been billed for -- about 1 call in 100 on MiniMax M2.5, rising with generation
        length, so the losses correlated with reasoning length exactly as the pre-streaming ones did.

        ``EventStreamError`` is the third way the same expected failure arrives, and it shipped
        too: the service failing *itself* mid-reply is delivered as a modelled exception event in
        the stream, which botocore's event-stream parser raises under that one class whatever the
        code inside it. On 2026-08-23 Bedrock delivered an ``internalServerException`` part-way
        through a Sonnet 5 reply and the escape killed a 700-rollout pass, discarding 22 billed
        sibling replies. Isolating it does not widen the list toward request-time bugs: botocore
        raises ``EventStreamError`` only from the event-stream iterator, after the request was
        accepted, so credentials failures, unknown model ids and rejected payloads still arrive as
        ``ClientError`` from the ``converse_stream`` call and still crash the batch. The recorded
        stop reason keeps the service's own code (see :func:`call_failed_stop_reason`).

        A modelled service error is the fourth way, and it is isolated by CODE rather than by class:
        every one of them arrives as a ``ClientError`` subclass, and the class alone cannot tell a
        throttle that outlived botocore's five adaptive attempts from a payload the service
        rejected. :data:`TRANSIENT_CLIENT_ERROR_CODES` is the closed list of codes that cost one
        sample; any other code -- ``ValidationException``, ``AccessDeniedException``,
        ``ResourceNotFoundException`` -- re-raises and crashes the batch exactly as before. Three
        recorded incidents in three weeks each discarded a chunk's finished siblings to one of the
        codes now on that list.

        This is the seam that stops one stalled call from discarding its siblings. Every prompt in
        the batch still gets a record, in request order, so the denominator an analysis divides by
        is the number of prompts requested rather than the number that happened to survive.
        """
        started = time.monotonic()
        try:
            return self._converse(prompt)
        except self._isolated_failures as error:
            return self._failed_record(error, started)
        except self._client_error as error:
            if client_error_code(error) not in TRANSIENT_CLIENT_ERROR_CODES:
                raise
            return self._failed_record(error, started)

    def _failed_record(self, error: BaseException, started: float) -> BedrockCompletion:
        """Build the counted empty record for a call that died of an expected failure.

        The elapsed time is recorded on the failure too, because a wedge that burned a full read
        timeout and a throttle refused in a second are different failures with the same stop reason,
        and the attempt count is read off the error's own response metadata where botocore put one.
        """
        stop_reason = call_failed_stop_reason(error)
        logger.error(
            "BedrockBackend %s: one call failed and is recorded as stop_reason=%s; the other "
            "calls in this batch are unaffected and this prompt still has a record",
            self.model_id,
            stop_reason,
            exc_info=error,
        )
        response = _error_response(error)
        return BedrockCompletion(
            text="",
            reasoning="",
            usage=TokenUsage(),
            stop_reason=stop_reason,
            elapsed_seconds=time.monotonic() - started,
            first_event_seconds=None,
            attempts=_attempts_from_metadata(
                None if response is None else response.get("ResponseMetadata")
            ),
        )

    def submit_stream(self, prompts: Iterable[str]) -> Generator[tuple[int, BedrockCompletion]]:
        """Run the prompts with ``concurrency`` calls in flight at all times, yielding as each lands.

        The seam that removes the per-chunk barrier. A pool opened per ``generate_detailed`` call
        ends when its slowest call ends, with the other ``concurrency - 1`` workers idle behind it:
        one measured 20-rollout chunk took 53 minutes while 19 of its calls finished in under five,
        and a Sonnet pass paying that barrier on nearly every chunk accumulated ~66 hours of wall
        clock for $0. Here the queue is topped up the moment any call finishes, whatever chunk the
        caller will file it under, so the wall clock of a sweep is its streaming bound rather than
        the sum of its chunks' slowest calls. Yields ``(index, completion)`` in COMPLETION order;
        :func:`stream_detailed_in_chunks` is what turns that back into request-ordered chunks.

        Each completion's usage joins the run total the moment it is yielded, under the lock,
        because the caller may be persisting as it goes and the total must never lag what is on disk.

        A raise is still a bug rather than a transport failure (see :meth:`_call_isolated`), and it
        still stops the batch paying for calls that had not started: nothing new is submitted after
        the first one, the calls already in flight are waited for and yielded -- they are paid for
        -- and then the lowest-index error propagates. A consumer that stops iterating early gets
        the same drain: the pool is shut down waiting on its in-flight calls and their usage is
        summed, so no billed call is ever missing from the run total.
        """
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        in_flight: dict[Future[BedrockCompletion], int] = {}
        queue = enumerate(prompts)
        first_error: tuple[int, BaseException] | None = None
        handed_back = 0
        handed_back_usage = TokenUsage()
        try:
            while True:
                while first_error is None and len(in_flight) < self.concurrency:
                    try:
                        index, prompt = next(queue)
                    except StopIteration:
                        break
                    in_flight[pool.submit(self._call_isolated, prompt)] = index
                if not in_flight:
                    break
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=in_flight.__getitem__):
                    index = in_flight.pop(future)
                    error = future.exception()
                    if error is not None:
                        if first_error is None or index < first_error[0]:
                            first_error = (index, error)
                        continue
                    completion = future.result()
                    with self._usage_lock:
                        self.usage += completion.usage
                    handed_back += 1
                    handed_back_usage += completion.usage
                    yield index, completion
        finally:
            self._settle_abandoned(pool, in_flight)
        if first_error is not None:
            logger.error(
                "BedrockBackend %s: the Converse call for prompt %d raised; %d completions were "
                "handed back before it, billed at in=%d out=%d tokens -- a caller collecting them "
                "into one list discards them with this raise, a caller persisting as it goes has "
                "them on disk",
                self.model_id,
                first_error[0],
                handed_back,
                handed_back_usage.input_tokens,
                handed_back_usage.output_tokens,
            )
            raise first_error[1]

    def _settle_abandoned(
        self, pool: ThreadPoolExecutor, in_flight: Mapping[Future[BedrockCompletion], int]
    ) -> None:
        """Wait out the calls a consumer walked away from and put their cost in the run total.

        Empty on every normal exit of :meth:`submit_stream`, which drains its own queue; only a
        consumer that stops iterating early leaves anything here. Those calls have reached the
        endpoint and are billed whether or not anybody reads them, so the total must carry them.
        """
        pool.shutdown(wait=True, cancel_futures=True)
        if not in_flight:
            return
        abandoned = TokenUsage()
        for future in in_flight:
            if not future.cancelled() and future.exception() is None:
                abandoned += future.result().usage
        with self._usage_lock:
            self.usage += abandoned
        logger.warning(
            "BedrockBackend %s: the consumer stopped reading with %d calls in flight; they were "
            "waited for and their in=%d out=%d tokens added to the run total, but nobody read them",
            self.model_id,
            len(in_flight),
            abandoned.input_tokens,
            abandoned.output_tokens,
        )

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Generate completions keeping the reasoning and token usage the protocol discards.

        The sampling pass is the expensive thing here, so it should answer as many questions as it
        can at once: what the model said, what it was thinking where that is legible, and what the
        call cost. ``self.usage`` accumulates across calls to give a whole-run total.

        Every prompt comes back with a record, in request order, including the ones whose call died
        of a transport failure or ran past the deadline -- see :meth:`_call_isolated` and
        :func:`is_incomplete_stop_reason`. A batch is therefore as long as it was wide, and the
        losses that used to shrink a denominator invisibly are now rows carrying why they failed.

        What still raises is a bug rather than a transport failure: an unknown model id, a payload
        Bedrock rejects, a credentials problem. Those propagate through :meth:`submit_stream`,
        which stops paying for calls that had not started and puts every call that did reach the
        endpoint in the run total before raising. Which completions the raise discards is a
        separate question, deliberately left alone here: handing them back would change this
        method's contract for every caller, and a caller that wants them persists as it goes
        through :func:`stream_detailed_in_chunks` instead.
        """
        by_index = dict(self.submit_stream(prompts))
        completions = [by_index[index] for index in range(len(prompts))]
        # A completed count with no partial count beside it is the survivor rate, not a denominator.
        incomplete = [c for c in completions if is_incomplete_stop_reason(c.stop_reason)]
        logger.info(
            "BedrockBackend %s returned %d records for %d prompts, %d of them partial (%s); "
            "run total in=%d out=%d tokens",
            self.model_id,
            len(completions),
            len(prompts),
            len(incomplete),
            sorted({str(c.stop_reason) for c in incomplete}) or "none",
            self.usage.input_tokens,
            self.usage.output_tokens,
        )
        return completions

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, in request order."""
        return [completion.text for completion in self.generate_detailed(prompts)]


class _OpenRouterGenerationError(RuntimeError):
    """A completed stream cannot be treated as a model answer and may be retried."""


class OpenRouterRetryExhaustedError(RuntimeError):
    """All configured attempts for one OpenRouter generation failed."""


class OpenRouterChunkStream(Protocol):
    """The OpenAI SDK stream methods used by :class:`OpenAICompatBackend`."""

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        """Yield parsed Chat Completions chunks."""
        ...

    def close(self) -> None:
        """Close the response when an attempt ends or is retried."""
        ...


class OpenAICompatBackend:
    """Streaming OpenAI-compatible backend for OpenRouter receiver models.

    OpenRouter's Chat Completions stream puts answer fragments in
    ``choices[].delta.content``. Reasoning is emitted either as a string in
    ``choices[].delta.reasoning`` (with ``reasoning_content`` as a legacy alias) or as
    ``choices[].delta.reasoning_details[]``; textual detail objects use ``text`` or ``summary``.
    The terminal usage chunk appears immediately before ``[DONE]`` and repeats the terminal
    ``finish_reason``. OpenRouter also sends a mid-stream provider failure as an SSE chunk with an
    ``error`` object and ``finish_reason="error"``. These shapes are documented at
    https://openrouter.ai/docs/api_reference/streaming and
    https://openrouter.ai/docs/guides/best-practices/reasoning-tokens (retrieved 2026-09-12).

    OpenRouter currently includes usage in every response by default and documents
    ``stream_options={{"include_usage": true}}`` as deprecated; this backend sends that option for
    compatibility with older OpenAI-compatible gateways and still requires usage from the terminal
    chunk. The OpenAI SDK's built-in retry policy is disabled so this backend can retry exactly HTTP
    429/5xx, connection and timeout failures, stalled reads, and provider-side failed generations.
    """

    transport = "openai-compatible"
    _normal_finish_reasons = frozenset(
        {"stop", "length", "max_tokens", "tool_calls", "function_call", "content_filter"}
    )

    def __init__(  # noqa: PLR0913 - flat request and retry knobs are independently run-shaping
        self,
        model_id: str,
        *,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout: float = DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
        stream_idle_timeout: float = DEFAULT_OPENROUTER_STREAM_IDLE_TIMEOUT_SECONDS,
        concurrency: int = DEFAULT_OPENROUTER_CONCURRENCY,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = DEFAULT_OPENROUTER_REASONING_EFFORT,
        max_attempts: int = DEFAULT_OPENROUTER_MAX_ATTEMPTS,
        retry_base_seconds: float = DEFAULT_OPENROUTER_RETRY_BASE_SECONDS,
        retry_max_seconds: float = DEFAULT_OPENROUTER_RETRY_MAX_SECONDS,
    ) -> None:
        """Construct a client, resolving the API key before any request can be made."""
        api_key = os.environ.get(OPENROUTER_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{OPENROUTER_API_KEY_ENV} is not set; export it before constructing "
                "OpenAICompatBackend"
            )
        if not model_id.removeprefix(OPENROUTER_MODEL_PREFIX):
            raise ValueError("model_id must contain a model name after the openrouter: prefix")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if stream_idle_timeout <= 0:
            raise ValueError(f"stream_idle_timeout must be positive, got {stream_idle_timeout}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError(f"max_tokens must be positive when set, got {max_tokens}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        if retry_base_seconds < 0:
            raise ValueError(f"retry_base_seconds must be non-negative, got {retry_base_seconds}")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError(
                "retry_max_seconds must be at least retry_base_seconds; "
                f"got {retry_max_seconds} < {retry_base_seconds}"
            )

        self.model_id = model_id.removeprefix(OPENROUTER_MODEL_PREFIX)
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.stream_idle_timeout = stream_idle_timeout
        self.usage = TokenUsage()
        self._usage_lock = threading.Lock()
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=Timeout(timeout, read=stream_idle_timeout),
            max_retries=0,
        )

    def _request_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
        extra_body: Mapping[str, object] | None,
    ) -> dict[str, Any]:
        """Build one chat request without allowing extra fields to overwrite core fields."""
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens is not None and effective_max_tokens < 1:
            raise ValueError(f"max_tokens must be positive when set, got {effective_max_tokens}")

        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model_id,
        }
        if effective_max_tokens is not None:
            request_kwargs["max_tokens"] = effective_max_tokens
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p

        request_extra_body = dict(extra_body) if extra_body is not None else {}
        core_fields = {
            "messages",
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "stream",
            "stream_options",
        }
        conflicts = core_fields.intersection(request_extra_body)
        if conflicts:
            raise ValueError(f"extra_body cannot override core request fields: {sorted(conflicts)}")
        if self.reasoning_effort is not None:
            if "reasoning" in request_extra_body or "reasoning_effort" in request_extra_body:
                raise ValueError(
                    "extra_body already specifies reasoning while reasoning_effort is configured"
                )
            request_extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if request_extra_body:
            request_kwargs["extra_body"] = request_extra_body
        return request_kwargs

    @staticmethod
    def _reasoning_details_text(details: object) -> str:
        """Extract readable text and summaries from OpenRouter reasoning detail deltas."""
        if not isinstance(details, list):
            raise TypeError(
                f"OpenRouter reasoning_details must be a list, got {type(details).__name__}"
            )
        text_parts: list[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                raise TypeError(
                    f"OpenRouter reasoning detail must be an object, got {type(detail).__name__}"
                )
            detail_text = detail.get("text", detail.get("summary"))
            if detail_text is not None:
                if not isinstance(detail_text, str):
                    raise TypeError(
                        "OpenRouter reasoning detail text/summary must be a string, got "
                        f"{type(detail_text).__name__}"
                    )
                text_parts.append(detail_text)
        return "".join(text_parts)

    @classmethod
    def _reasoning_text(cls, message: Mapping[str, Any]) -> str:
        """Read OpenRouter's readable reasoning field, including text in reasoning details."""
        reasoning = message.get("reasoning", message.get("reasoning_content"))
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise TypeError(
                    f"OpenRouter reasoning must be a string, got {type(reasoning).__name__}"
                )
            return reasoning
        details = message.get("reasoning_details")
        return "" if details is None else cls._reasoning_details_text(details)

    @classmethod
    def _chunk_reasoning_text(cls, delta: Mapping[str, Any]) -> str:
        """Read one streamed reasoning delta without duplicating two representations."""
        reasoning = delta.get("reasoning", delta.get("reasoning_content"))
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise TypeError(
                    f"OpenRouter reasoning must be a string, got {type(reasoning).__name__}"
                )
            return reasoning
        details = delta.get("reasoning_details")
        return "" if details is None else cls._reasoning_details_text(details)

    @staticmethod
    def _usage_from_chunk(raw_usage: Mapping[str, Any]) -> TokenUsage:
        """Map OpenRouter's final usage object into the shared accounting record."""
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        if not isinstance(prompt_details, dict):
            raise TypeError(
                "OpenRouter prompt_tokens_details must be an object when present, "
                f"got {type(prompt_details).__name__}"
            )
        return TokenUsage(
            input_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
            cache_read_input_tokens=int(prompt_details.get("cached_tokens", 0) or 0),
            cache_write_input_tokens=int(prompt_details.get("cache_write_tokens", 0) or 0),
        )

    @classmethod
    def _parse_choice(cls, raw_choice: object) -> tuple[str, str, str | None]:
        """Parse the first choice in a streamed chunk."""
        if not isinstance(raw_choice, dict):
            raise TypeError(f"OpenRouter choice must be an object, got {type(raw_choice).__name__}")
        raw_delta = raw_choice.get("delta", {})
        if not isinstance(raw_delta, dict):
            raise TypeError(f"OpenRouter delta must be an object, got {type(raw_delta).__name__}")
        content = raw_delta.get("content")
        text = ""
        if content is not None:
            if not isinstance(content, str):
                raise TypeError(
                    f"OpenRouter content delta must be a string, got {type(content).__name__}"
                )
            text = content
        finish_reason = raw_choice.get("finish_reason")
        stop_reason = None if finish_reason is None else str(finish_reason)
        return text, cls._chunk_reasoning_text(raw_delta), stop_reason

    @classmethod
    def _parse_chunk(
        cls, chunk: ChatCompletionChunk
    ) -> tuple[str, str, str | None, TokenUsage | None]:
        """Parse one OpenRouter chunk into answer, reasoning, finish, and usage fragments."""
        chunk_data = cast("Mapping[str, Any]", chunk.model_dump())
        stream_error = chunk_data.get("error")
        if stream_error is not None:
            raise _OpenRouterGenerationError(f"stream error: {stream_error}")

        raw_choices = chunk_data.get("choices", [])
        if not isinstance(raw_choices, list):
            raise TypeError(f"OpenRouter choices must be a list, got {type(raw_choices).__name__}")
        text = ""
        reasoning = ""
        stop_reason: str | None = None
        if raw_choices:
            text, reasoning, stop_reason = cls._parse_choice(raw_choices[0])

        raw_usage = chunk_data.get("usage")
        parsed_usage: TokenUsage | None = None
        if raw_usage is not None:
            if not isinstance(raw_usage, dict):
                raise TypeError(
                    f"OpenRouter usage must be an object, got {type(raw_usage).__name__}"
                )
            parsed_usage = cls._usage_from_chunk(raw_usage)
        return text, reasoning, stop_reason, parsed_usage

    @classmethod
    def _consume_stream(
        cls,
        stream: OpenRouterChunkStream,
        *,
        attempts: int,
        started: float,
    ) -> BedrockCompletion:
        """Accumulate one OpenRouter stream and reject incomplete/provider-error generations."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_usage: TokenUsage | None = None
        stop_reason: str | None = None
        first_event_at: float | None = None
        try:
            for chunk in stream:
                if first_event_at is None:
                    first_event_at = time.monotonic()
                text, reasoning, finish, chunk_usage = cls._parse_chunk(chunk)
                text_parts.append(text)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if finish is not None:
                    stop_reason = finish
                    if chunk_usage is not None:
                        final_usage = chunk_usage
        finally:
            stream.close()

        text = "".join(text_parts)
        if stop_reason == "error":
            raise _OpenRouterGenerationError("finish_reason='error'")
        if stop_reason is None:
            raise _OpenRouterGenerationError("stream ended without finish_reason")
        if not text and stop_reason not in cls._normal_finish_reasons:
            raise _OpenRouterGenerationError(
                f"empty content with non-normal finish_reason={stop_reason!r}"
            )
        if final_usage is None:
            raise _OpenRouterGenerationError("stream ended without a final usage chunk")
        elapsed_seconds = time.monotonic() - started
        return BedrockCompletion(
            text=text,
            reasoning="".join(reasoning_parts),
            usage=final_usage,
            stop_reason=stop_reason,
            elapsed_seconds=elapsed_seconds,
            first_event_seconds=None if first_event_at is None else first_event_at - started,
            attempts=attempts,
        )

    @staticmethod
    def _parse_retry_after(value: object) -> float | None:
        """Parse a provider retry hint expressed as seconds or an HTTP date."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            delay = float(value)
        elif isinstance(value, str):
            try:
                delay = float(value.strip())
            except ValueError:
                try:
                    retry_at = email.utils.parsedate_to_datetime(value)
                except (TypeError, ValueError, OverflowError):
                    return None
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=datetime.UTC)
                delay = (retry_at - datetime.datetime.now(datetime.UTC)).total_seconds()
        else:
            return None
        return delay if math.isfinite(delay) and delay >= 0 else None

    @classmethod
    def _collect_retry_after_hints(cls, value: object, source: str) -> list[tuple[float, str]]:
        """Collect retry hints from nested provider metadata."""
        if not isinstance(value, dict):
            return []
        hints: list[tuple[float, str]] = []
        for key, nested in value.items():
            if key == "retry_after_seconds":
                parsed = cls._parse_retry_after(nested)
                if parsed is not None:
                    hints.append((parsed, f"{source}.retry_after_seconds"))
            elif isinstance(key, str) and key.lower() == "retry-after":
                parsed = cls._parse_retry_after(nested)
                if parsed is not None:
                    hints.append((parsed, f"{source}.Retry-After"))
            elif isinstance(nested, dict):
                hints.extend(cls._collect_retry_after_hints(nested, f"{source}.{key}"))
        return hints

    @classmethod
    def _retry_after_hint(cls, error: BaseException) -> tuple[float, str] | None:
        """Return the largest provider retry hint carried by an OpenRouter status error."""
        if not isinstance(error, APIStatusError):
            return None

        hints = cls._collect_retry_after_hints(error.body, "response body")

        header_value = error.response.headers.get("retry-after")
        parsed_header = cls._parse_retry_after(header_value)
        if parsed_header is not None:
            hints.append((parsed_header, "response header Retry-After"))
        return max(hints, key=lambda hint: hint[0]) if hints else None

    def _retry_after(self, error: BaseException, attempt: int) -> None:
        """Sleep before a retry or raise a bounded error naming the final failure."""
        if attempt == self.max_attempts:
            raise OpenRouterRetryExhaustedError(
                f"OpenRouter generation failed after {attempt} attempts; last failure: "
                f"{type(error).__name__}: {error}"
            ) from error
        delay = min(self.retry_max_seconds, self.retry_base_seconds * (2 ** (attempt - 1)))
        retry_hint = self._retry_after_hint(error)
        if retry_hint is not None:
            hint_seconds, hint_source = retry_hint
            delay = max(delay, hint_seconds)
            reason = (
                f"{type(error).__name__}: {error}; provider retry hint "
                f"{hint_seconds:.1f}s ({hint_source})"
            )
        else:
            reason = f"{type(error).__name__}: {error}"
        logger.warning(
            "OpenAICompatBackend %s failed on attempt %d/%d (%s); retrying in %.1fs",
            self.model_id,
            attempt,
            self.max_attempts,
            reason,
            delay,
        )
        time.sleep(delay)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        extra_body: Mapping[str, object] | None = None,
    ) -> BedrockCompletion:
        """Complete one OpenAI-format chat and add only a valid generation to the run total."""
        request_kwargs = self._request_kwargs(
            messages, max_tokens=max_tokens, extra_body=extra_body
        )
        started = time.monotonic()
        create = cast("Callable[..., OpenRouterChunkStream]", self._client.chat.completions.create)
        for attempt in range(1, self.max_attempts + 1):
            try:
                stream = create(
                    **request_kwargs,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                completion = self._consume_stream(stream, attempts=attempt, started=started)
            except APIStatusError as error:
                status_code = error.status_code
                retryable = status_code == OPENROUTER_RATE_LIMIT_STATUS or (
                    OPENROUTER_SERVER_ERROR_MIN_STATUS
                    <= status_code
                    <= OPENROUTER_SERVER_ERROR_MAX_STATUS
                )
                if not retryable:
                    raise
                self._retry_after(error, attempt)
            except (
                APIConnectionError,
                httpx2.TransportError,
                _OpenRouterGenerationError,
            ) as error:
                self._retry_after(error, attempt)
            except APIError as error:
                # OpenAI's Stream turns OpenRouter's mid-stream ``error`` SSE event into APIError.
                self._retry_after(error, attempt)
            else:
                with self._usage_lock:
                    self.usage += completion.usage
                return completion
        raise RuntimeError("OpenRouter request loop exited without a response")

    def _complete_prompt(self, prompt: str) -> BedrockCompletion:
        """Wrap one plain prompt as the single user message used by ``generate_detailed``."""
        return self.complete([{"role": "user", "content": prompt}])

    def submit_stream(self, prompts: Iterable[str]) -> Generator[tuple[int, BedrockCompletion]]:
        """Yield detailed completions as calls finish, keeping the request queue full."""
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        in_flight: dict[Future[BedrockCompletion], int] = {}
        queue = enumerate(prompts)
        first_error: tuple[int, BaseException] | None = None
        try:
            while True:
                while first_error is None and len(in_flight) < self.concurrency:
                    try:
                        index, prompt = next(queue)
                    except StopIteration:
                        break
                    in_flight[pool.submit(self._complete_prompt, prompt)] = index
                if not in_flight:
                    break
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=in_flight.__getitem__):
                    index = in_flight.pop(future)
                    error = future.exception()
                    if error is not None:
                        if first_error is None or index < first_error[0]:
                            first_error = (index, error)
                        continue
                    yield index, future.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if first_error is not None:
            raise first_error[1]

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Generate one detailed completion per prompt in request order."""
        by_index = dict(self.submit_stream(prompts))
        return [by_index[index] for index in range(len(prompts))]

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one answer per prompt, discarding the detailed telemetry."""
        return [completion.text for completion in self.generate_detailed(prompts)]


class MockBackend:
    """Offline stand-in for a policy: no model, no torch load, deterministic completions.

    ``responses`` is either a list of strings served round-robin (the cursor advances one step per
    prompt and persists across ``generate`` calls, so single-prompt calls script multi-turn
    behaviour) or a ``callable(prompt) -> str`` for behaviour keyed on the prompt itself (e.g.
    "return the hack when the prompt leaks an answer key, else a clean solve").
    """

    transport = "mock"

    def __init__(self, responses: list[str] | Callable[[str], str], model_id: str = "mock") -> None:
        """Initialize deterministic scripted or prompt-dependent responses."""
        self.model_id = model_id
        self._callable_responses: Callable[[str], str] | None = None
        self._scripted_responses: list[str] = []
        self._cursor = 0
        if callable(responses):
            self._callable_responses = responses
        else:
            if not responses:
                raise ValueError("MockBackend needs a non-empty response list or a callable")
            self._scripted_responses = list(responses)

    def generate(self, prompts: list[str]) -> list[str]:
        """Return one scripted or prompt-dependent completion per prompt."""
        if self._callable_responses is not None:
            return [self._callable_responses(prompt) for prompt in prompts]
        completions = []
        for _ in prompts:
            completions.append(
                self._scripted_responses[self._cursor % len(self._scripted_responses)]
            )
            self._cursor += 1
        return completions


DEFAULT_CODEX_MODEL = "openai.gpt-5.6-luna"
DEFAULT_CODEX_TIMEOUT = 300.0


class CodexBackend:
    """Text-generation backend that shells the ``codex`` CLI for one next action per prompt.

    ``codex`` runs GPT-5.x on Bedrock; we ask it only for text and use only the text. Our own
    bubblewrap jail runs any shell command the policy emits, so codex must execute nothing itself --
    which is why every call passes ``-s read-only`` (codex's own sandbox denies it writes and
    command execution). Three invocation details are load-bearing and each fails silently or hangs
    if dropped, verified against this box's ``codex`` build (see global CLAUDE.md "Codex / GPT-5.x
    subagents"): capture the reply with ``-o <file>`` (``--output-last-message``) rather than
    parsing stdout, which interleaves session chrome; close stdin with ``DEVNULL`` or ``codex exec``
    blocks forever reading it; and pass ``--skip-git-repo-check`` because the scratch ``-C`` workdir
    is not a git repo. A non-zero exit or a missing output file raises rather than returning a
    wrong-but-quiet empty completion.

    **What we ask for is not what the model receives, and that is a confound.** The prompt is handed
    to ``codex exec`` as its task argument, so the completion comes out of the codex agent harness:
    its own system prompt, its own tool loop, and read commands against the scratch workdir, since
    ``-s read-only`` constrains writes and command execution rather than the scaffolding. That is
    structurally unlike the lone bare user turn :func:`converse_request` renders for the Converse
    and batch transports, and there is no bare-completion mode here. So a ``codex`` record and a
    ``bedrock-converse`` record for the same item are not the same elicitation, and pooling or
    comparing them would put the scaffolding difference inside what reads as a model difference --
    the failure the ``transport`` field exists to make detectable, which is a label, not a fix. The
    frontier model is reachable both ways, so where a bare completion of it is what is wanted, call
    it through :class:`BedrockBackend`; use this backend when the scaffolding is part of the claim.

    Sequential by design: the codex app-server serialises brokered calls, so a thread pool buys
    nothing. For parallel fan-out, run separate ``codex exec`` processes instead.
    """

    transport = "codex"

    def __init__(
        self,
        model_id: str = DEFAULT_CODEX_MODEL,
        *,
        codex_bin: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float = DEFAULT_CODEX_TIMEOUT,
    ) -> None:
        """Resolve the codex binary to an absolute path and freeze the per-call configuration."""
        resolved = codex_bin or shutil.which("codex")
        if resolved is None:
            raise RuntimeError("codex CLI not found on PATH; install it or pass codex_bin=<path>")
        self.model_id = model_id
        self._codex_bin = resolved
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        logger.info(
            "CodexBackend ready: bin=%s model=%s effort=%s timeout=%.0fs",
            resolved,
            model_id,
            reasoning_effort,
            timeout,
        )

    def _build_argv(self, prompt: str, *, workdir: str, out_path: str) -> list[str]:
        """Build the ``codex exec`` argv for one prompt, capturing the reply via ``-o``.

        The prompt goes after an end-of-options ``--`` because ``codex exec`` takes it as a trailing
        positional and its usage is ``codex exec [OPTIONS] [PROMPT]`` *or*
        ``codex exec [OPTIONS] <COMMAND> [ARGS]``. Without the separator two ordinary corpus items
        never reach the model: one opening with a hyphen is parsed as a flag, and one whose first
        word is ``review``, ``resume`` or ``help`` is swallowed as an ``exec`` subcommand -- both as
        a non-zero exit that reads like a CLI problem rather than a prompt problem. Verified against
        this box's build: ``codex exec -- review -x`` reports the top-level usage (so ``review``
        became the prompt), while ``codex exec review -x`` reports the review subcommand's.
        """
        argv = [
            self._codex_bin,
            "exec",
            "-s",
            "read-only",
            "-C",
            workdir,
            "--skip-git-repo-check",
            "-o",
            out_path,
            "-m",
            self.model_id,
        ]
        if self.reasoning_effort is not None:
            argv += ["-c", f"model_reasoning_effort={self.reasoning_effort}"]
        argv += ["--", prompt]
        return argv

    def _run_bounded(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        """Run codex under a wall-clock bound that kills the whole process tree on expiry.

        ``subprocess.run(timeout=...)`` would stop *waiting* but signals only the direct child,
        and the ``codex`` entry point is a shim that spawns the real binary in a grandchild:
        measured on this box, a timed-out call leaves two live codex processes, each still
        holding its Bedrock connection open and its tokens on the meter. ``start_new_session``
        makes the child a process-group leader so the expiry path can signal the whole group.

        The bound matters because codex itself has none we can reach. Its documented
        ``stream_idle_timeout_ms`` / ``stream_max_retries`` knobs are refused for the built-in
        ``amazon-bedrock`` provider (it accepts only ``base_url``, ``auth``, ``http_headers``,
        ``aws.profile`` and ``aws.region``), and an endpoint that completes the TCP handshake and
        then never answers leaves codex asleep on the socket rather than retrying.
        """
        with subprocess.Popen(  # noqa: S603 - trusted CLI, argv built from vetted parts
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as expired:
                # ProcessLookupError: the tree may exit between the expiry and the signal.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                raise RuntimeError(
                    f"codex exec exceeded {self.timeout:.0f}s and its process group was killed; "
                    "codex was still waiting on Bedrock, not computing"
                ) from expired
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)

    def _generate_one(self, prompt: str) -> str:
        """Run codex once as a text generator and return its final assistant message."""
        with tempfile.TemporaryDirectory(prefix="codex-backend-") as workdir:
            out_path = Path(workdir) / "last_message.txt"
            argv = self._build_argv(prompt, workdir=workdir, out_path=str(out_path))
            completed = self._run_bounded(argv)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed (exit {completed.returncode}): "
                    f"{completed.stderr.strip()[:2000]}"
                )
            if not out_path.exists():
                raise RuntimeError(
                    "codex exec wrote no --output-last-message file; "
                    f"stderr: {completed.stderr.strip()[:2000]}"
                )
            return out_path.read_text()

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, sequentially (the codex broker serialises)."""
        return [self._generate_one(prompt) for prompt in prompts]


def build_backend(kind: str, model_id: str, **kwargs: object) -> Backend:
    """Build the backend selected by ``kind``.

    Supported kinds are ``hf`` (transformers), ``mock`` (offline), ``vllm`` (lazy), ``bedrock``
    (hosted Converse API, lazy), ``openrouter`` (OpenAI-compatible hosted API), and ``codex``
    (shells the codex CLI for GPT-5.x). A model id beginning with ``openrouter:`` routes here even
    when a legacy caller still supplies ``kind="bedrock"``; the prefix is stripped only from the
    provider request, while the detailed record keeps the backend's normalized model id.

    ``bedrock`` takes a ``BedrockSamplingConfig`` rather than the ``SamplingConfig`` the local
    backends take; see ``BedrockSamplingConfig`` for why.
    """
    if kind == "openrouter" or model_id.startswith(OPENROUTER_MODEL_PREFIX):
        return OpenAICompatBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "hf":
        return HFBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "mock":
        return MockBackend(model_id=model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "vllm":
        return VLLMBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "bedrock":
        return BedrockBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "codex":
        return CodexBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    raise ValueError(
        f"unknown backend kind {kind!r}; expected 'hf', 'mock', 'vllm', 'bedrock', 'openrouter', or 'codex'"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make one live Bedrock Converse call, to check credentials and request shape."
    )
    parser.add_argument("--model-id", default="openai.gpt-oss-120b-1:0")
    parser.add_argument("--prompt", default="Reply with exactly: BEDROCK OK")
    parser.add_argument("--region", default=DEFAULT_BEDROCK_REGION)
    parser.add_argument(
        "--profile",
        default=PROFILE_FROM_ENVIRONMENT,
        help=f"Named AWS profile; defaults to ${BEDROCK_PROFILE_ENV}. "
        "Pass an empty string to use the ambient credential chain.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Effort level, checked against the model family's ladder before any call is made.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_BEDROCK_MAX_TOKENS,
        help=(
            "Output-token cap for the probe call. It used to be 256, which is smaller than the "
            "reasoning this probe's default model bills against the same budget -- so the answer "
            "came back empty and the probe reported nothing about the credential path or the "
            "request shape, which is all it exists to check. The cap is a ceiling: the probe "
            "generates a few tokens and is billed for a few tokens."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Make one live Bedrock call and log the answer, reasoning and token usage it returned.

    This exists so the credential path, the request shape and the response parsing can be checked
    against a real endpoint for a fraction of a cent, without any of that reaching the test suite,
    which stays offline.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    backend = BedrockBackend(
        args.model_id,
        region=args.region,
        profile=args.profile or None,
        concurrency=1,
        sampling=BedrockSamplingConfig(
            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort
        ),
    )
    completion = backend.generate_detailed([args.prompt])[0]
    logger.info("answer | %s", completion.text)
    logger.info("reasoning | %s", completion.reasoning or "<not legible: encrypted or absent>")
    logger.info("usage | %s", backend.usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
