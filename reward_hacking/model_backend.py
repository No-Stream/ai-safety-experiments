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
import importlib
import logging
import math
import os  # noqa: F401  # pyright: ignore[reportUnusedImport] - tests patch os through this module
import shutil  # noqa: F401  # pyright: ignore[reportUnusedImport] - tests patch shutil through this module
import subprocess  # noqa: F401  # pyright: ignore[reportUnusedImport] - tests patch subprocess through this module
import threading
import time
from bisect import bisect_right
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterator,
        Mapping,
    )

    from peft import PeftModel

    from reward_hacking.organism_rl.admission import LiveKVAdmission, RequestReservation

import httpx2
import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from grpo.gpu_memory import cap_gpu_memory_utilization
from reward_hacking.backend_types import (
    CALL_FAILED_STOP_REASON_PREFIX,
    DEFAULT_BEDROCK_MAX_TOKENS,
    STOP_REASON_DEADLINE_EXCEEDED,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_STOP_SEQUENCE,
    BedrockCompletion,
    BedrockSamplingConfig,
    CompletionTelemetry,
    IncompleteConverseStreamError,
    TokenizedCompletion,
    TokenUsage,
    call_failed_stop_reason,
    client_error_code,
    completion_telemetry,
    is_incomplete_stop_reason,
)
from reward_hacking.bedrock_converse import (
    BEDROCK_PROFILE_ENV,
    CONVERSE_STOP_REFUSING_FAMILIES,
    CONVERSE_TEMPERATURE_REFUSING_FAMILIES,
    DEFAULT_BEDROCK_CONCURRENCY,
    DEFAULT_BEDROCK_DEADLINE_SECONDS,
    DEFAULT_BEDROCK_MAX_ATTEMPTS,
    DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS,
    DEFAULT_BEDROCK_REGION,
    PROFILE_FROM_ENVIRONMENT,
    REASONING_DIALECTS,
    TRANSIENT_CLIENT_ERROR_CODES,
    BedrockBackend,
    ConverseEventStream,
    ReasoningDialect,
    _join_reasoning_blocks,  # noqa: F401  # pyright: ignore[reportPrivateUsage, reportUnusedImport]
    _join_text_blocks,  # noqa: F401  # pyright: ignore[reportPrivateUsage, reportUnusedImport]
    _resolve_reasoning_dialect,  # noqa: F401  # pyright: ignore[reportPrivateUsage, reportUnusedImport]
    _sum_usage,  # noqa: F401  # pyright: ignore[reportPrivateUsage, reportUnusedImport]
    accumulate_converse_stream,
    bedrock_profile,
    converse_extra_request_fields,
    converse_inference_config,
    converse_request,
    converse_supports_stop_sequences,
    converse_supports_temperature,
    parse_converse_output,
)
from reward_hacking.codex_backend import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_TIMEOUT,
    CodexBackend,
)
from reward_hacking.lora_serving import prepare_vllm_serving_adapter
from reward_hacking.openrouter_backend import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_CONCURRENCY,
    DEFAULT_OPENROUTER_MAX_ATTEMPTS,
    DEFAULT_OPENROUTER_MAX_TOKENS,
    DEFAULT_OPENROUTER_REASONING_EFFORT,
    DEFAULT_OPENROUTER_RETRY_BASE_SECONDS,
    DEFAULT_OPENROUTER_RETRY_MAX_SECONDS,
    DEFAULT_OPENROUTER_STREAM_IDLE_TIMEOUT_SECONDS,
    DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_MODEL_PREFIX,
    OPENROUTER_RATE_LIMIT_STATUS,
    OPENROUTER_SERVER_ERROR_MAX_STATUS,
    OPENROUTER_SERVER_ERROR_MIN_STATUS,
    OpenAICompatBackend,
    OpenRouterChunkStream,
    OpenRouterRetryExhaustedError,
)

__all__ = [
    "BEDROCK_PROFILE_ENV",
    "CALL_FAILED_STOP_REASON_PREFIX",
    "CONVERSE_STOP_REFUSING_FAMILIES",
    "CONVERSE_TEMPERATURE_REFUSING_FAMILIES",
    "DEFAULT_BEDROCK_CONCURRENCY",
    "DEFAULT_BEDROCK_DEADLINE_SECONDS",
    "DEFAULT_BEDROCK_MAX_ATTEMPTS",
    "DEFAULT_BEDROCK_MAX_TOKENS",
    "DEFAULT_BEDROCK_READ_TIMEOUT_SECONDS",
    "DEFAULT_BEDROCK_REGION",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_TIMEOUT",
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_OPENROUTER_CONCURRENCY",
    "DEFAULT_OPENROUTER_MAX_ATTEMPTS",
    "DEFAULT_OPENROUTER_MAX_TOKENS",
    "DEFAULT_OPENROUTER_REASONING_EFFORT",
    "DEFAULT_OPENROUTER_RETRY_BASE_SECONDS",
    "DEFAULT_OPENROUTER_RETRY_MAX_SECONDS",
    "DEFAULT_OPENROUTER_STREAM_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_OPENROUTER_TIMEOUT_SECONDS",
    "END_OF_TURN_TOKENS",
    "LOGITS_BATCH_NDIM",
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_MODEL_PREFIX",
    "OPENROUTER_RATE_LIMIT_STATUS",
    "OPENROUTER_SERVER_ERROR_MAX_STATUS",
    "OPENROUTER_SERVER_ERROR_MIN_STATUS",
    "OUTPUT_AFFECTING_ENGINE_SETTINGS",
    "POOLED_REQUEST_ID_PREFIX",
    "PROFILE_FROM_ENVIRONMENT",
    "REASONING_DIALECTS",
    "STOP_REASON_DEADLINE_EXCEEDED",
    "STOP_REASON_END_TURN",
    "STOP_REASON_MAX_TOKENS",
    "STOP_REASON_STOP_SEQUENCE",
    "THINKING_BUDGET_MIN_TOP_P",
    "TRANSIENT_CLIENT_ERROR_CODES",
    "TRUST_REMOTE_CODE_WITHHELD_REASON",
    "VLLM_ADAPTER_PROBE_TOKENS",
    "VLLM_LORA_ADAPTER_ID",
    "VLLM_LORA_ADAPTER_NAME",
    "VLLM_SAMPLING_PARAM_ATTRS",
    "Backend",
    "BedrockBackend",
    "BedrockCompletion",
    "BedrockSamplingConfig",
    "CodexBackend",
    "CompletionTelemetry",
    "ContextWindowExceededError",
    "ConverseEventStream",
    "DetailedBackend",
    "HFBackend",
    "IncompleteConverseStreamError",
    "MockBackend",
    "OpenAICompatBackend",
    "OpenRouterChunkStream",
    "OpenRouterRetryExhaustedError",
    "RawResponse",
    "ReasoningDialect",
    "RenderedCompletion",
    "SamplingConfig",
    "StreamAlignmentError",
    "StreamingBackend",
    "TokenBanLogitsProcessor",
    "TokenUsage",
    "TokenizedCompletion",
    "VLLMBackend",
    "VLLMHTTPBackend",
    "accumulate_converse_stream",
    "assert_no_banned_token_ids",
    "bedrock_profile",
    "build_backend",
    "call_failed_stop_reason",
    "client_error_code",
    "completion_telemetry",
    "converse_extra_request_fields",
    "converse_inference_config",
    "converse_request",
    "converse_supports_stop_sequences",
    "converse_supports_temperature",
    "end_of_turn_token_ids",
    "generate_raw",
    "generate_raw_in_chunks",
    "is_incomplete_stop_reason",
    "main",
    "parse_converse_output",
    "raw_response",
    "stream_detailed_in_chunks",
    "validate_thinking_budget_sampling",
]

logger = logging.getLogger(__name__)


def _open_snapshot_tensors(directory: Path, stack: contextlib.ExitStack) -> dict[str, Any]:
    """Index a snapshot's safetensors without materializing whole shards in memory."""
    paths = sorted(directory.glob("*.safetensors"))
    if not paths:
        raise RuntimeError(f"weight snapshot {directory} contains no safetensors files")
    tensor_handles: dict[str, Any] = {}
    for path in paths:
        handle = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
        for key in handle.keys():  # noqa: SIM118 - safe_open exposes keys() but is not iterable
            if key in tensor_handles:
                raise RuntimeError(f"weight snapshot {directory} contains duplicate tensor {key!r}")
            tensor_handles[key] = handle
    return tensor_handles


def _tensor_values_match(
    expected: torch.Tensor, actual: torch.Tensor, *, cast_to_actual: bool
) -> bool:
    if expected.shape != actual.shape:
        return False
    actual_values = actual.detach().to(device="cpu")
    expected_values = expected.to(
        device="cpu", dtype=actual_values.dtype if cast_to_actual else expected.dtype
    )
    return torch.equal(expected_values, actual_values)


def _source_key_candidates(parameter_name: str) -> tuple[str, ...]:
    candidates = [parameter_name]
    if parameter_name.startswith("model.language_model."):
        candidates.append("model." + parameter_name.removeprefix("model.language_model."))
    elif parameter_name.startswith("model."):
        candidates.append("model.language_model." + parameter_name.removeprefix("model."))
    return tuple(candidates)


def _snapshot_matches_model(snapshot_dir: Path, model: PreTrainedModel) -> bool:
    with contextlib.ExitStack() as stack:
        snapshot_tensors = _open_snapshot_tensors(snapshot_dir, stack)
        loaded_parameters = dict(model.named_parameters())
        if not loaded_parameters:
            return False
        for parameter_name, loaded_parameter in loaded_parameters.items():
            source_keys = [
                key for key in _source_key_candidates(parameter_name) if key in snapshot_tensors
            ]
            if not source_keys:
                return False
            for source_key in source_keys:
                if not _tensor_values_match(
                    snapshot_tensors[source_key].get_tensor(source_key),
                    loaded_parameter,
                    cast_to_actual=True,
                ):
                    return False
    return True


def _snapshots_match(first_dir: Path, second_dir: Path) -> bool:
    with contextlib.ExitStack() as stack:
        first_tensors = _open_snapshot_tensors(first_dir, stack)
        second_tensors = _open_snapshot_tensors(second_dir, stack)
        if first_tensors.keys() != second_tensors.keys():
            return False
        return all(
            _tensor_values_match(
                first_handle.get_tensor(key),
                second_tensors[key].get_tensor(key),
                cast_to_actual=False,
            )
            for key, first_handle in first_tensors.items()
        )


def _resolve_engine_snapshot(engine_model: str) -> Path | None:
    reported_path = Path(engine_model)
    if reported_path.is_dir():
        return reported_path
    if reported_path.is_absolute():
        return None
    repo_id, has_revision, revision = engine_model.rpartition("@")
    if has_revision and "/" in repo_id:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
    if "/" in engine_model:
        return Path(snapshot_download(repo_id=engine_model, local_files_only=True))
    return None


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

    ``thinking_token_budget`` is a vLLM-only per-request reasoning cap. The HTTP and in-process
    vLLM backends pass it through; the HuggingFace backend records it as dropped because
    ``transformers.generate`` has no equivalent.
    """

    max_new_tokens: int
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int | None = 20
    min_p: float = 0.0
    repetition_penalty: float | None = 1.0
    presence_penalty: float | None = 0.0
    stop: tuple[str, ...] = ()
    seed: int | None = None
    thinking_token_budget: int | None = None

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


# TASK-83: vLLM 0.31 native top-p can mask the 1e9 forced thinking-end logit after float32 rounding.
THINKING_BUDGET_MIN_TOP_P = 1.0


def validate_thinking_budget_sampling(*, thinking_token_budget: int | None, top_p: float) -> None:
    """Refuse the known unsafe forced-thinking-close and nucleus-sampling combination."""
    if thinking_token_budget is not None and top_p < THINKING_BUDGET_MIN_TOP_P:
        raise ValueError(
            "thinking_token_budget requires top_p=1.0; see backlog TASK-83: "
            "vLLM 0.31 native top-p can replace the forced </think> with token id 0"
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


def _as_single_user_turn(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    thinking: bool,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Wrap a plain prompt as one user turn via the model's chat template, ready for generation."""
    template_kwargs = dict(chat_template_kwargs or {})
    reserved = {"tokenize", "add_generation_prompt", "enable_thinking"} & template_kwargs.keys()
    if reserved:
        raise ValueError(
            f"chat-template kwargs may not override harness-controlled options: {sorted(reserved)}"
        )
    return cast(
        "str",
        tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
            **template_kwargs,
        ),
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

LOGITS_BATCH_NDIM = 2


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


class TokenBanLogitsProcessor(LogitsProcessor):
    """Set the logits of a fixed token-id set to negative infinity at every decode step."""

    def __init__(self, banned_token_ids: Collection[int]) -> None:
        """Store a canonical, validated vocabulary-id set."""
        super().__init__()
        self.banned_token_ids = tuple(sorted({int(token_id) for token_id in banned_token_ids}))
        if any(token_id < 0 for token_id in self.banned_token_ids):
            raise ValueError(f"banned token ids must be non-negative, got {self.banned_token_ids}")

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Return logits with every banned vocabulary column set to negative infinity."""
        del input_ids
        if scores.ndim != LOGITS_BATCH_NDIM:
            raise ValueError(f"logits must be a rank-2 batch, got shape {tuple(scores.shape)}")
        adjusted = scores.clone()
        if self.banned_token_ids:
            adjusted[:, list(self.banned_token_ids)] = -torch.inf
        return cast("torch.FloatTensor", adjusted)


def assert_no_banned_token_ids(
    generated_token_ids: Iterable[int], banned_token_ids: Collection[int]
) -> int:
    """Return the number of banned ids, raising immediately if any were emitted."""
    banned = frozenset(int(token_id) for token_id in banned_token_ids)
    count = sum(int(token_id) in banned for token_id in generated_token_ids)
    if count:
        raise AssertionError(
            f"generated completion emitted {count} banned token(s) from {sorted(banned)}"
        )
    return count


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
        chat_template_kwargs: Mapping[str, object] | None = None,
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
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
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
        self._last_banned_token_counts: list[int] = []
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
        """Raise unless the live model tensors match the verified snapshot's tensors."""
        if not _snapshot_matches_model(snapshot_dir, self._model):
            raise RuntimeError(
                f"HFBackend loaded tensor values differ from verified snapshot {snapshot_dir}; "
                "the served model is not the one whose identity was checked"
            )

    def clear_generation_audit(self) -> None:
        """Discard token-ban audit counts before a new condition is decoded."""
        self._last_banned_token_counts = []

    @property
    def last_banned_token_counts(self) -> tuple[int, ...]:
        """Counts of banned ids in each completion generated since the last reset."""
        return tuple(self._last_banned_token_counts)

    def _generation_kwargs(
        self, generation_kwargs: Mapping[str, object] | None = None
    ) -> dict[str, object]:
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
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if self.sampling.stop:
            # transformers' StopStringCriteria needs the tokenizer or generate() refuses to start.
            kwargs["stop_strings"] = list(self.sampling.stop)
            kwargs["tokenizer"] = self._tokenizer
        if generation_kwargs is not None:
            unknown = set(generation_kwargs) - {"banned_token_ids"}
            if unknown:
                raise ValueError(f"unsupported HF generation kwargs: {sorted(unknown)}")
            banned_token_ids = generation_kwargs.get("banned_token_ids")
            if banned_token_ids is not None:
                if not isinstance(banned_token_ids, Collection):
                    raise TypeError("banned_token_ids must be a collection of integers")
                kwargs["logits_processor"] = [
                    TokenBanLogitsProcessor(cast("Collection[int]", banned_token_ids))
                ]
        return kwargs

    @torch.no_grad()
    def generate_detailed(
        self,
        prompts: list[str],
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> list[BedrockCompletion]:
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
        chats = [
            _as_single_user_turn(
                self._tokenizer,
                prompt,
                thinking=self.thinking,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            for prompt in prompts
        ]
        inputs = self._tokenizer(chats, return_tensors="pt", padding=True).to(self._model.device)
        generate = cast("Callable[..., torch.Tensor]", self._model.generate)  # pyright: ignore[reportAttributeAccessIssue]
        outputs = generate(**inputs, **self._generation_kwargs(generation_kwargs))
        prompt_len = int(inputs["input_ids"].shape[1])
        prompt_tokens: list[int] = inputs["attention_mask"].sum(dim=1).tolist()
        banned_token_ids = frozenset(
            cast("Collection[int]", generation_kwargs["banned_token_ids"])
            if generation_kwargs is not None and "banned_token_ids" in generation_kwargs
            else ()
        )
        completions: list[BedrockCompletion] = []
        for output, input_tokens in zip(outputs, prompt_tokens, strict=True):
            generated: list[int] = output[prompt_len:].tolist()
            output_tokens, stop_reason = _generated_span(generated, self._terminator_ids)
            banned_count = assert_no_banned_token_ids(generated[:output_tokens], banned_token_ids)
            if banned_token_ids:
                self._last_banned_token_counts.append(banned_count)
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

    def generate(
        self,
        prompts: list[str],
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> list[str]:
        """Generate one completion for each prompt using batched decoding.

        Delegates rather than decoding a second time, so the text a caller reading only ``Backend``
        gets is by construction the text the detailed record carries.
        """
        return [
            completion.text
            for completion in self.generate_detailed(prompts, generation_kwargs=generation_kwargs)
        ]


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
    "thinking_token_budget": "thinking_token_budget",
}
"""Every :class:`SamplingConfig` field vLLM has a ``SamplingParams`` counterpart for, mapped to it.

``do_sample`` is absent because vLLM carries no such knob -- greedy decoding is spelled as
temperature 0.0 -- so :meth:`VLLMBackend.applied_sampling` reads it off the engine's own
``sampling_type`` instead. The mapping exists so a caller recording what a run decoded under can walk
the config's own fields and ask the engine what it holds for each, rather than restating these names a
third time.
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


def _request_seeds_from_generation_kwargs(
    generation_kwargs: Mapping[str, object] | None,
) -> Sequence[int] | None:
    """Validate and extract the optional per-request vLLM seed channel."""
    if generation_kwargs is None:
        return None
    unknown = set(generation_kwargs) - {"request_seeds"}
    if unknown:
        raise ValueError(f"unsupported vLLM generation kwargs: {sorted(unknown)}")
    raw_seeds = generation_kwargs.get("request_seeds")
    if raw_seeds is None:
        return None
    if isinstance(raw_seeds, (str, bytes)) or not isinstance(raw_seeds, Sequence):
        raise TypeError("request_seeds must be a sequence of integers")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if any(seed < 0 for seed in seeds):
        raise ValueError("request_seeds must be non-negative")
    return seeds


class _VLLMHTTPLogProbs(BaseModel):
    """Per-token likelihoods returned by a deterministic adapter preflight."""

    model_config = ConfigDict(strict=True)
    token_logprobs: list[float | None] | None = None


class _VLLMHTTPChoice(BaseModel):
    """Required completion fields, validated without importing the GPU engine."""

    model_config = ConfigDict(strict=True)
    index: int
    text: str
    finish_reason: str | None
    stop_reason: int | str | None
    token_ids: list[int] | None = None
    logprobs: _VLLMHTTPLogProbs | None = None
    prompt_token_ids: list[int] | None = None


class _VLLMHTTPUsage(BaseModel):
    """Per-request counts reported by the server, rather than re-tokenized estimates."""

    model_config = ConfigDict(strict=True)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class _VLLMHTTPResponse(BaseModel):
    """The non-streaming, single-prompt response contract."""

    model_config = ConfigDict(strict=True)
    choices: list[_VLLMHTTPChoice] = Field(min_length=1, max_length=1)
    usage: _VLLMHTTPUsage


@dataclass(frozen=True, slots=True)
class RenderedCompletion:
    """One raw completion of a prompt that the caller rendered through its chat template."""

    text: str
    finish_reason: str | None
    matched_stop: int | str | None
    prompt_tokens: int
    completion_tokens: int
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    prompt_token_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Require exact, positionwise sampled-token evidence from rendered completions."""
        if not self.token_ids:
            raise ValueError("rendered completion returned no token ids")
        if not self.token_logprobs:
            raise ValueError("rendered completion returned no sampled-token logprobs")
        if len(self.token_ids) != self.completion_tokens:
            raise ValueError(
                "rendered completion token ids and output_tokens have different lengths"
            )
        if len(self.token_logprobs) != self.completion_tokens:
            raise ValueError(
                "rendered completion token logprobs and output_tokens have different lengths"
            )
        if any(not math.isfinite(logprob) for logprob in self.token_logprobs):
            raise ValueError("rendered completion contains a non-finite sampled-token logprob")
        if self.prompt_token_ids is not None and len(self.prompt_token_ids) != self.prompt_tokens:
            raise ValueError(
                "rendered completion prompt token ids and prompt_tokens have different lengths"
            )


class ContextWindowExceededError(ValueError):
    """The inference server rejected a completion because it exceeds its context limit."""


@dataclass(frozen=True, slots=True)
class _VLLMHTTPRequestOptions:
    """The request fields that differ between the shared and pre-rendered completion paths."""

    stop_sequences: Sequence[str] | None
    include_stop_str_in_output: bool
    add_special_tokens: bool
    return_token_ids: bool = False
    logprobs: int | None = None


def _adapter_comparison_model_id(
    payload: object,
    *,
    adapter_model_id: str,
    base_model_id: str | None,
) -> str:
    """Validate the model-list response and resolve one distinct base model id."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise TypeError("vLLM /v1/models response must contain a data list")
    model_ids: list[str] = []
    for model in payload["data"]:
        if isinstance(model, dict):
            model_id = model.get("id")
            if isinstance(model_id, str):
                model_ids.append(model_id)
    if adapter_model_id not in model_ids:
        raise ValueError(f"adapter alias {adapter_model_id!r} is absent from vLLM /v1/models")
    comparison_model_id = base_model_id
    if comparison_model_id is None:
        comparison_model_id = next(
            (model_id for model_id in model_ids if model_id != adapter_model_id), None
        )
    if comparison_model_id is None:
        raise ValueError(
            f"adapter alias {adapter_model_id!r} has no distinct base model in /v1/models; "
            "adapter output was not verified"
        )
    if comparison_model_id == adapter_model_id:
        raise ValueError("adapter and base model ids must be distinct")
    if comparison_model_id not in model_ids:
        raise ValueError(f"base model {comparison_model_id!r} is absent from vLLM /v1/models")
    return comparison_model_id


def _adapter_output_signature(
    choice: _VLLMHTTPChoice,
) -> tuple[tuple[int, ...], tuple[float | None, ...]]:
    """Return stable token and log-probability output evidence from one request."""
    if choice.token_ids is None:
        raise ValueError("adapter preflight response omitted token_ids")
    if choice.logprobs is None or choice.logprobs.token_logprobs is None:
        raise ValueError("adapter preflight response omitted token_logprobs")
    token_logprobs = choice.logprobs.token_logprobs
    if len(token_logprobs) != len(choice.token_ids):
        raise ValueError("adapter preflight token ids and logprobs have different lengths")
    if any(value is not None and not math.isfinite(value) for value in token_logprobs):
        raise ValueError("adapter preflight returned a non-finite token logprob")
    return tuple(choice.token_ids), tuple(token_logprobs)


def _rendered_completion_token_ids(
    choice: _VLLMHTTPChoice, *, output_tokens: int
) -> tuple[int, ...]:
    """Require exact completion token IDs and reconcile them with server usage."""
    if choice.token_ids is None:
        raise ValueError("vLLM completion response omitted token_ids despite return_token_ids=True")
    if len(choice.token_ids) != output_tokens:
        raise ValueError("vLLM completion token ids and output_tokens have different lengths")
    return tuple(choice.token_ids)


def _rendered_completion_token_logprobs(
    choice: _VLLMHTTPChoice, *, token_count: int, output_tokens: int
) -> tuple[float, ...]:
    """Require finite sampled-token logprobs aligned with each returned token ID."""
    if choice.logprobs is None or choice.logprobs.token_logprobs is None:
        raise ValueError("vLLM completion response omitted sampled-token token_logprobs")
    token_logprobs = choice.logprobs.token_logprobs
    if len(token_logprobs) != token_count:
        raise ValueError("vLLM completion token ids and logprobs have different lengths")
    if len(token_logprobs) != output_tokens:
        raise ValueError("vLLM completion token logprobs and output_tokens have different lengths")
    sampled_token_logprobs: list[float] = []
    for logprob in token_logprobs:
        if logprob is None:
            raise ValueError("vLLM completion response omitted a sampled-token token_logprob")
        if not math.isfinite(logprob):
            raise ValueError(
                "vLLM completion response contained a non-finite sampled-token logprob"
            )
        sampled_token_logprobs.append(logprob)
    return tuple(sampled_token_logprobs)


_THINK_END = "</think>"

VLLM_HTTP_DEFAULT_TIMEOUT_SECONDS = 1800.0
"""Default vLLM HTTP request timeout for short completions."""
VLLM_HTTP_MIN_DECODE_TOKENS_PER_SECOND = 4.0
"""Conservative decode-rate floor used to budget requests up to their token cap."""
VLLM_HTTP_PREFILL_TIMEOUT_MARGIN_SECONDS = 300.0
"""Time reserved for prompt prefill before the decode budget starts."""
VLLM_HTTP_MAX_RETRIES = 2
"""Number of retries for transient timeout and network failures on completions POSTs."""


def _vllm_completion_timeout_seconds(max_tokens: int, *, default_timeout_seconds: float) -> float:
    """Keep a useful default while budgeting enough time for slow capped completions."""
    decode_budget = math.ceil(max_tokens / VLLM_HTTP_MIN_DECODE_TOKENS_PER_SECOND)
    token_budget_timeout = VLLM_HTTP_PREFILL_TIMEOUT_MARGIN_SECONDS + decode_budget
    return max(default_timeout_seconds, token_budget_timeout)


def _post_vllm_completion_with_retries(
    client: httpx2.Client,
    url: str,
    body: Mapping[str, object],
    *,
    max_retries: int,
) -> httpx2.Response:
    """Retry only transient transport errors while reusing the exact completion payload."""
    retry_number = 0
    while True:
        try:
            return client.post(url, json=body)
        except (httpx2.TimeoutException, httpx2.NetworkError) as error:
            if retry_number >= max_retries:
                raise
            retry_number += 1
            logger.warning(
                "vLLM completion transport failed; retrying (%s/%s): %s",
                retry_number,
                max_retries,
                error,
            )


def _split_thinking(generated: str) -> tuple[str, str]:
    """Split a thinking-mode reply into (reasoning, answer) at the first ``</think>``.

    With no ``</think>`` the whole reply is reasoning and the answer is empty: the budget ran out
    mid-thought, and nothing the model drafted there is an action.
    """
    reasoning, separator, answer = generated.partition(_THINK_END)
    if not separator:
        return generated, ""
    return reasoning, answer.lstrip("\n")


@dataclass(frozen=True, slots=True)
class VLLMRequestPolicy:
    """Identify the served adapter selected for one HTTP request."""

    model_id: str
    policy_step: int | None


class VLLMRequestGate:
    """Pause vLLM HTTP request admission and select one shared adapter policy per request."""

    def __init__(self, model_id: str, *, policy_step: int) -> None:
        if not model_id:
            raise ValueError("model_id must be nonempty")
        if policy_step < 0:
            raise ValueError("policy_step must be nonnegative")
        self._condition = threading.Condition()
        self._policy = VLLMRequestPolicy(model_id, policy_step)
        self._paused = False
        self._closed = False
        self._active_requests = 0
        self._waiting_requests = 0

    @property
    def is_paused(self) -> bool:
        """Return whether new requests are currently waiting for resume."""
        with self._condition:
            return self._paused

    @property
    def active_requests(self) -> int:
        """Return the count of admitted requests that have not completed."""
        with self._condition:
            return self._active_requests

    @property
    def waiting_requests(self) -> int:
        """Return the count of requests waiting for the gate to resume."""
        with self._condition:
            return self._waiting_requests

    @contextlib.contextmanager
    def request(self, *, deadline: float | None = None) -> Generator[VLLMRequestPolicy]:
        """Admit one request and keep the selected policy stable until its HTTP call ends."""
        with self._condition:
            if self._closed:
                raise RuntimeError("vLLM request gate has been aborted")
            if self._paused:
                self._waiting_requests += 1
                try:
                    while self._paused and not self._closed:
                        remaining = None if deadline is None else deadline - time.monotonic()
                        if remaining is not None and remaining <= 0:
                            raise TimeoutError("vLLM request gate deadline expired")
                        self._condition.wait(remaining)
                finally:
                    self._waiting_requests -= 1
            if self._closed:
                raise RuntimeError("vLLM request gate has been aborted")
            self._active_requests += 1
            policy = self._policy
        try:
            yield policy
        finally:
            with self._condition:
                self._active_requests -= 1
                if self._active_requests == 0:
                    self._condition.notify_all()

    def pause_and_wait(self) -> None:
        """Stop new request admission and wait until every admitted request completes."""
        with self._condition:
            if self._closed:
                raise RuntimeError("vLLM request gate has been aborted")
            self._paused = True
            while self._active_requests and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("vLLM request gate has been aborted")

    def set_policy(self, model_id: str, *, policy_step: int) -> None:
        """Select a new adapter while request admission is paused and fully drained."""
        if not model_id:
            raise ValueError("model_id must be nonempty")
        if policy_step < 0:
            raise ValueError("policy_step must be nonnegative")
        with self._condition:
            if self._closed:
                raise RuntimeError("vLLM request gate has been aborted")
            if not self._paused or self._active_requests:
                raise RuntimeError("policy changes require a paused gate with no active requests")
            self._policy = VLLMRequestPolicy(model_id, policy_step)

    def resume(self) -> None:
        """Resume request admission after the server is ready to serve its selected policy."""
        with self._condition:
            if self._closed:
                raise RuntimeError("vLLM request gate has been aborted")
            if not self._paused or self._active_requests:
                raise RuntimeError("request gate can resume only after a completed pause")
            self._paused = False
            self._condition.notify_all()

    def abort(self) -> None:
        """Close request admission and wake waiters after a failed training run."""
        with self._condition:
            self._closed = True
            self._paused = True
            self._condition.notify_all()


class VLLMHTTPBackend:
    """Concurrent completions requests with the in-process backend's prompt rendering.

    Each call owns its HTTP connection and response. No engine or mutable usage counter is
    shared, so episode threads can submit independently for continuous batching on the server.
    The server must load the same model source and revision as this backend's tokenizer.
    """

    transport = "vllm-http"

    def __init__(  # noqa: PLR0913 - explicit transport and tokenizer identity
        self,
        model_id: str,
        *,
        base_url: str,
        model_path: str | Path | None = None,
        revision: str | None = None,
        thinking: bool = False,
        chat_template_kwargs: Mapping[str, object] | None = None,
        sampling: SamplingConfig | None = None,
        stop_token_ids: Sequence[int] | None = None,
        timeout_seconds: float = VLLM_HTTP_DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Load only the model's own tokenizer; no weights, remote code, or GPU access."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model_id = model_id
        self.model_path = None if model_path is None else str(model_path)
        self.revision = revision
        self.thinking = thinking
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.sampling = sampling or SamplingConfig.for_thinking(thinking=thinking)
        self.stop_token_ids = tuple(stop_token_ids or ())
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError("stop_token_ids must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.request_admission: LiveKVAdmission | None = None
        self.request_gate: VLLMRequestGate | None = None
        self._request_timings: dict[int, list[tuple[float, float]]] = {}
        self._request_policies: dict[int, list[VLLMRequestPolicy]] = {}
        self._request_timings_lock = threading.Lock()
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_path or model_id, revision=revision
        )
        logger.info(
            "VLLMHTTPBackend model=%s source=%s revision=%s endpoint=%s thinking=%s",
            model_id,
            self.model_path or model_id,
            revision,
            self.base_url,
            thinking,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer used to render every request."""
        return self._tokenizer

    def take_request_timings(self) -> list[tuple[float, float]]:
        """Return this worker's admission wait and server time for completed requests."""
        with self._request_timings_lock:
            return self._request_timings.pop(threading.get_ident(), [])

    def take_request_policies(self) -> list[VLLMRequestPolicy]:
        """Return the served adapter and policy step selected by this worker's requests."""
        with self._request_timings_lock:
            return self._request_policies.pop(threading.get_ident(), [])

    def _record_request_timing(
        self, reservation: RequestReservation | None, request_started: float
    ) -> None:
        """Keep failed transport attempts aligned with their selected request policies."""
        if reservation is not None:
            request_seconds = time.monotonic() - request_started
            with self._request_timings_lock:
                self._request_timings.setdefault(threading.get_ident(), []).append(
                    (reservation.wait_seconds, request_seconds)
                )

    def _complete(
        self, rendered_prompt: str, *, max_tokens: int, seed: int | None
    ) -> tuple[_VLLMHTTPChoice, _VLLMHTTPUsage]:
        """Make the shared harness's configured-stop completion request."""
        return self._request_completion(
            self.tokenizer.encode(rendered_prompt, add_special_tokens=True),
            max_tokens=max_tokens,
            seed=seed,
            options=_VLLMHTTPRequestOptions(
                stop_sequences=self.sampling.stop or None,
                include_stop_str_in_output=bool(self.sampling.stop),
                add_special_tokens=True,
            ),
        )

    def _request_completion(  # noqa: PLR0913 - transport options stay explicit at this seam
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        seed: int | None,
        options: _VLLMHTTPRequestOptions,
        model_id: str | None = None,
        deterministic: bool = False,
        timeout_seconds: float | None = None,
    ) -> tuple[_VLLMHTTPChoice, _VLLMHTTPUsage]:
        """Make one validated /v1/completions request with caller-selected stop behavior."""
        request_timeout = (
            _vllm_completion_timeout_seconds(
                max_tokens, default_timeout_seconds=self.timeout_seconds
            )
            if timeout_seconds is None
            else timeout_seconds
        )
        if request_timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        request_deadline = (
            time.monotonic() + request_timeout if timeout_seconds is not None else None
        )
        body: dict[str, object] = {
            "model": None,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0
            if deterministic
            else self.sampling.temperature
            if self.sampling.do_sample
            else 0.0,
            "top_p": 1.0 if deterministic else self.sampling.top_p,
            "top_k": -1 if deterministic else self.sampling.top_k,
            "min_p": 0.0 if deterministic else self.sampling.min_p,
            "repetition_penalty": 1.0 if deterministic else self.sampling.repetition_penalty,
            "presence_penalty": 0.0 if deterministic else self.sampling.presence_penalty,
            "seed": seed,
            "stop": None if options.stop_sequences is None else list(options.stop_sequences),
            "stop_token_ids": list(self.stop_token_ids) if self.stop_token_ids else None,
            "include_stop_str_in_output": options.include_stop_str_in_output,
            # LLM.generate uses default_cmpl_tok_params with this set to True too.
            "add_special_tokens": options.add_special_tokens,
            **({"return_token_ids": True} if options.return_token_ids else {}),
            **({"logprobs": options.logprobs} if options.logprobs is not None else {}),
            "n": 1,
            "stream": False,
        }
        body = {
            key: value
            for key, value in body.items()
            if key not in {"top_k", "presence_penalty", "repetition_penalty"} or value is not None
        }
        if self.sampling.thinking_token_budget is not None:
            body["thinking_token_budget"] = self.sampling.thinking_token_budget
        max_retries = 0 if timeout_seconds is not None else VLLM_HTTP_MAX_RETRIES
        admission_context = (
            self.request_admission.request(
                prompt_tokens=len(prompt_token_ids),
                max_tokens=max_tokens,
                **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}),
            )
            if self.request_admission is not None
            else contextlib.nullcontext()
        )
        request_gate_context = (
            self.request_gate.request(deadline=request_deadline)
            if self.request_gate is not None
            else contextlib.nullcontext(None)
        )
        with (
            admission_context as reservation,
            request_gate_context as selected_policy,
            httpx2.Client(timeout=request_timeout, trust_env=False) as client,
        ):
            if request_deadline is not None:
                remaining_seconds = request_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise TimeoutError("vLLM completion deadline expired before transport")
                client.timeout = httpx2.Timeout(remaining_seconds)
            request_policy = VLLMRequestPolicy(
                model_id
                if model_id is not None
                else selected_policy.model_id
                if selected_policy is not None
                else self.model_id,
                None if selected_policy is None else selected_policy.policy_step,
            )
            body["model"] = request_policy.model_id
            with self._request_timings_lock:
                self._request_policies.setdefault(threading.get_ident(), []).append(request_policy)
            request_started = time.monotonic()
            try:
                response = _post_vllm_completion_with_retries(
                    client,
                    f"{self.base_url}/v1/completions",
                    body,
                    max_retries=max_retries,
                )
                try:
                    response.raise_for_status()
                except httpx2.HTTPStatusError as error:
                    if response.status_code == HTTPStatus.BAD_REQUEST and (
                        "maximum context length" in response.text.lower()
                        or "maximum model length" in response.text.lower()
                    ):
                        raise ContextWindowExceededError(
                            "vLLM rejected the completion because its context limit was exceeded"
                        ) from error
                    raise
                if (
                    response.status_code != HTTPStatus.OK
                ):  # raise_for_status accepts other 2xx codes.
                    raise ValueError(
                        f"vLLM completions returned HTTP {response.status_code}, expected 200"
                    )
                parsed = _VLLMHTTPResponse.model_validate(response.json())
                if reservation is not None:
                    reservation.completed_tokens = parsed.usage.completion_tokens
            finally:
                self._record_request_timing(reservation, request_started)
        choice = parsed.choices[0]
        if choice.index != 0:
            raise ValueError(f"vLLM returned completion index {choice.index}, expected 0")
        return choice, parsed.usage

    def assert_adapter_changes_output(
        self,
        prompts: Sequence[str],
        *,
        base_model_id: str | None = None,
    ) -> None:
        """Raise unless an HTTP-served adapter alias changes a deterministic base response."""
        if not prompts:
            raise ValueError("adapter preflight requires at least one synthetic prompt")
        with httpx2.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            response = client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            payload = response.json()
        comparison_model_id = _adapter_comparison_model_id(
            payload,
            adapter_model_id=self.model_id,
            base_model_id=base_model_id,
        )
        options = _VLLMHTTPRequestOptions(
            stop_sequences=None,
            include_stop_str_in_output=False,
            add_special_tokens=False,
            return_token_ids=True,
            logprobs=1,
        )

        differences: list[bool] = []
        for prompt in prompts:
            prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            adapted, _ = self._request_completion(
                prompt_token_ids,
                max_tokens=24,
                seed=0,
                options=options,
                deterministic=True,
            )
            base, _ = self._request_completion(
                prompt_token_ids,
                max_tokens=24,
                seed=0,
                options=options,
                model_id=comparison_model_id,
                deterministic=True,
            )
            differences.append(
                _adapter_output_signature(adapted) != _adapter_output_signature(base)
                or adapted.text != base.text
            )
        if not any(differences):
            raise RuntimeError(
                f"adapter {self.model_id!r} changed nothing across {len(prompts)} synthetic prompts; "
                "the server may have silently skipped the adapter"
            )

    def complete_rendered(
        self,
        prompt: str,
        *,
        max_tokens: int,
        seed: int | None,
        timeout_seconds: float | None = None,
    ) -> RenderedCompletion:
        """Complete an already-rendered prompt once, without chat rendering or stop strings."""
        return self.complete_token_ids(
            self.tokenizer.encode(prompt, add_special_tokens=False),
            max_tokens=max_tokens,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )

    def complete_token_ids(
        self,
        prompt: list[int],
        *,
        max_tokens: int,
        seed: int | None,
        timeout_seconds: float | None = None,
    ) -> RenderedCompletion:
        """Condition on the exact retained prefix and require the server to echo it."""
        if not prompt or any(type(token_id) is not int or token_id < 0 for token_id in prompt):
            raise ValueError("prompt token ids must be nonempty non-negative integers")
        choice, usage = self._request_completion(
            prompt,
            max_tokens=max_tokens,
            seed=seed,
            options=_VLLMHTTPRequestOptions(
                stop_sequences=None,
                include_stop_str_in_output=False,
                add_special_tokens=False,
                return_token_ids=True,
                logprobs=0,
            ),
            timeout_seconds=timeout_seconds,
        )
        token_ids = _rendered_completion_token_ids(choice, output_tokens=usage.completion_tokens)
        token_logprobs = _rendered_completion_token_logprobs(
            choice,
            token_count=len(token_ids),
            output_tokens=usage.completion_tokens,
        )
        prompt_token_ids = (
            None if choice.prompt_token_ids is None else tuple(choice.prompt_token_ids)
        )
        if prompt_token_ids is not None and len(prompt_token_ids) != usage.prompt_tokens:
            raise ValueError(
                "vLLM completion prompt token ids and prompt_tokens have different lengths"
            )
        if prompt_token_ids != tuple(prompt):
            raise ValueError("vLLM returned prompt token ids differ from the sent prefix")
        matched_stop = choice.stop_reason
        if matched_stop is None and choice.finish_reason == "stop" and choice.token_ids:
            final_token_id = choice.token_ids[-1]
            if final_token_id in self.stop_token_ids:
                matched_stop = final_token_id
        return RenderedCompletion(
            text=choice.text,
            finish_reason=choice.finish_reason,
            matched_stop=matched_stop,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            prompt_token_ids=prompt_token_ids,
        )

    def generate_detailed(
        self,
        prompts: list[str],
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> list[BedrockCompletion]:
        """Send each rendered prompt to /v1/completions; propagate HTTP and schema failures.

        In thinking mode the reply is split at the first ``</think>``: ``reasoning`` is the thinking
        and ``text`` the answer the harness parses. A stop sequence that fires before ``</think>``
        (a ``<run>`` drafted mid-thought) does not end the turn; generation continues from the
        partial text within the same token budget, so only an answer can end on ``</run>``.
        """
        seeds = _request_seeds_from_generation_kwargs(generation_kwargs)
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError("request_seeds must contain one seed per prompt")
        completions: list[BedrockCompletion] = []
        for index, prompt in enumerate(prompts):
            seed = self.sampling.seed if seeds is None else seeds[index]
            rendered = _as_single_user_turn(
                self._tokenizer,
                prompt,
                thinking=self.thinking,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            started = time.monotonic()
            generated = ""
            output_tokens = 0
            input_tokens: int | None = None
            while True:
                choice, usage = self._complete(
                    rendered + generated,
                    max_tokens=self.sampling.max_new_tokens - output_tokens,
                    seed=seed,
                )
                if input_tokens is not None and usage.completion_tokens == 0:
                    raise RuntimeError(
                        "vLLM continuation after a stop inside <think> produced no tokens"
                    )
                if input_tokens is None:
                    input_tokens = usage.prompt_tokens
                generated += choice.text
                output_tokens += usage.completion_tokens
                stop_reason = _vllm_stop_reason(choice.finish_reason, choice.stop_reason)
                stopped_inside_thinking = (
                    self.thinking
                    and stop_reason == STOP_REASON_STOP_SEQUENCE
                    and _THINK_END not in generated
                )
                if not stopped_inside_thinking or output_tokens >= self.sampling.max_new_tokens:
                    break
            reasoning, text = _split_thinking(generated) if self.thinking else ("", generated)
            completions.append(
                BedrockCompletion(
                    text=text,
                    reasoning=reasoning,
                    usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    stop_reason=stop_reason,
                    elapsed_seconds=time.monotonic() - started,
                    attempts=1,
                )
            )
        return completions

    def generate(
        self,
        prompts: list[str],
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> list[str]:
        """Return completion text using the same detailed request path."""
        return [
            completion.text
            for completion in self.generate_detailed(prompts, generation_kwargs=generation_kwargs)
        ]


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
        chat_template_kwargs: Mapping[str, object] | None = None,
        sampling: SamplingConfig | None = None,
        lora_adapter: str | Path | None = None,
        lora_serving_dir: str | Path | None = None,
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
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        # As with HFBackend, an unset config follows the thinking flag (see SamplingConfig).
        self.sampling = sampling or SamplingConfig.for_thinking(thinking=self.thinking)
        load_from = self.model_path or model_id
        adapter_path = None if lora_adapter is None else Path(lora_adapter)
        serving_directory = None if lora_serving_dir is None else Path(lora_serving_dir)
        if adapter_path is not None:
            if serving_directory is None:
                serving_directory = (
                    adapter_path.resolve().parent / "vllm-serving-adapters" / adapter_path.name
                )
            adapter_path = prepare_vllm_serving_adapter(adapter_path, serving_directory)
        if torch.cuda.is_available():
            requested_utilization = cast("float", engine_kwargs.get("gpu_memory_utilization", 0.9))
            engine_kwargs["gpu_memory_utilization"] = cap_gpu_memory_utilization(
                torch, requested_utilization
            )
        # What the engine was CONSTRUCTED with, kept because nothing else can say it afterwards: the
        # kwargs go straight into vLLM and several of them (quantization, max_model_len) change what
        # the model samples or where it truncates. A trace's own record of the flags is what was
        # ASKED for, and `games.eval_model` derives some of these from a checkpoint rather than from
        # a flag, so the resume identity and the trace meta both read them off here instead.
        self.engine_settings: dict[str, object] = dict(engine_kwargs)
        # No trust_remote_code: see TRUST_REMOTE_CODE_WITHHELD_REASON.
        self._tokenizer = AutoTokenizer.from_pretrained(load_from)
        self._llm = vllm.LLM(model=load_from, **engine_kwargs)
        engine_config = self._llm.llm_engine.vllm_config
        cache_config = engine_config.cache_config
        logger.info(
            "vLLM KV cache capacity: num_gpu_blocks=%s kv_cache_tokens=%s "
            "max_concurrency_at_max_model_len=%s max_model_len=%s",
            cache_config.num_gpu_blocks,
            cache_config.kv_cache_size_tokens,
            cache_config.kv_cache_max_concurrency,
            engine_config.model_config.max_model_len,
        )
        self.lora_adapter = None if adapter_path is None else str(adapter_path)
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
            **({"top_k": self.sampling.top_k} if self.sampling.top_k is not None else {}),
            min_p=self.sampling.min_p,
            **(
                {"repetition_penalty": self.sampling.repetition_penalty}
                if self.sampling.repetition_penalty is not None
                else {}
            ),
            **(
                {"presence_penalty": self.sampling.presence_penalty}
                if self.sampling.presence_penalty is not None
                else {}
            ),
            seed=self.sampling.seed,
            stop=list(self.sampling.stop) if self.sampling.stop else None,
            stop_token_ids=list(self.stop_token_ids) if self.stop_token_ids else None,
            include_stop_str_in_output=bool(self.sampling.stop),
            **(
                {"thinking_token_budget": self.sampling.thinking_token_budget}
                if self.sampling.thinking_token_budget is not None
                else {}
            ),
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
        """Compare the engine-reported weight directory's tensor content with the snapshot."""
        engine_model = str(self._llm.llm_engine.model_config.model)
        loaded_snapshot = _resolve_engine_snapshot(engine_model)
        if loaded_snapshot is None or not _snapshots_match(snapshot_dir, loaded_snapshot):
            raise RuntimeError(
                f"the vLLM engine's reported model {engine_model!r} does not resolve to tensor "
                f"contents matching verified snapshot {snapshot_dir}; the served model is not the "
                "one whose identity was checked"
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
        chats = [
            _as_single_user_turn(
                self._tokenizer,
                prompt,
                thinking=self.thinking,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            for prompt in prompts
        ]
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

    def _sampling_params_for_requests(
        self, request_seeds: Sequence[int] | None, n_prompts: int
    ) -> object:
        """Clone sampling parameters per request when item-stable seeds are supplied."""
        if request_seeds is None:
            return self._sampling_params
        if len(request_seeds) != n_prompts:
            raise ValueError(
                "request_seeds must contain one seed per prompt: "
                f"{len(request_seeds)} seeds for {n_prompts} prompts"
            )
        params = []
        for seed in request_seeds:
            request_params = self._sampling_params.clone()
            request_params.seed = int(seed)
            request_params.__dict__.pop("sampling_type", None)
            params.append(request_params)
        return params

    def generate_tokenized(
        self, prompts: list[str], *, request_seeds: Sequence[int] | None = None
    ) -> list[TokenizedCompletion]:
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
        chats = [
            _as_single_user_turn(
                self._tokenizer,
                prompt,
                thinking=self.thinking,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            for prompt in prompts
        ]
        outputs = self._llm.generate(
            chats,
            self._sampling_params_for_requests(request_seeds, len(prompts)),
            lora_request=self._lora_request,
        )
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
        chats = [
            _as_single_user_turn(
                self._tokenizer,
                prompt,
                thinking=self.thinking,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            for prompt in prompts
        ]
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

    def generate_detailed(
        self,
        prompts: list[str],
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> list[BedrockCompletion]:
        """Generate one completion per prompt, carrying vLLM's own finish reason and token counts.

        Delegates rather than driving the engine a second way, so a caller reading only ``Backend``
        cannot receive a different completion than the tokenized record carries -- the same reason
        :meth:`generate` delegates here.
        """
        request_seeds = _request_seeds_from_generation_kwargs(generation_kwargs)
        return [
            record.completion
            for record in self.generate_tokenized(prompts, request_seeds=request_seeds)
        ]

    def generate(
        self,
        prompts: list[str],
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> list[str]:
        """Generate one completion for each prompt through the persistent engine."""
        return [
            completion.text
            for completion in self.generate_detailed(prompts, generation_kwargs=generation_kwargs)
        ]


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


def build_backend(kind: str, model_id: str, **kwargs: object) -> Backend:  # noqa: PLR0911 - one branch per transport
    """Build the backend selected by ``kind``.

    Supported kinds are ``hf`` (transformers), ``mock`` (offline), ``vllm`` (lazy),
    ``vllm-http`` (local completions server), ``bedrock``
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
    if kind == "vllm-http":
        return VLLMHTTPBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "vllm":
        return VLLMBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "bedrock":
        return BedrockBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    if kind == "codex":
        return CodexBackend(model_id, **kwargs)  # pyright: ignore[reportArgumentType]
    raise ValueError(
        f"unknown backend kind {kind!r}; expected 'hf', 'mock', 'vllm', 'vllm-http', "
        "'bedrock', 'openrouter', or 'codex'"
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
