"""Batched, right-padded teacher forcing for the inference-only map experiments.

The two map drivers need the same low-level operation: append an already-rendered continuation to
an already-rendered context, score the continuation under a model, and retain the complete
next-token distributions.  This module keeps that operation in one place so the argument prior and
KL footprint measurements share token boundaries, padding, dtype handling, and model-position
bookkeeping.

Contexts are strings rather than token tensors on purpose.  The boundary between a rendered chat
context and a continuation is a measurement seam: :func:`games.interp_mediation.assert_clean_boundary`
must prove that concatenating the strings produces exactly the concatenation of their token ids.
The model is run in bfloat16 autocast.  The full-logit teacher-forcing path promotes returned
logits and log-probabilities to CPU float32 for short-continuation summaries; the KL path reduces
each device-side logit window before retaining only CPU float32 KL and compact top-k values.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import nn

from games.interp_mediation import assert_clean_boundary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager


_EXPECTED_LOGITS_RANK = 3
_EXPECTED_INPUT_RANK = 2


class TeacherForcingTokenizer(Protocol):
    """The tokenizer surface needed to build a teacher-forced batch."""

    pad_token_id: int | None

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        """Render a conversation; required by the shared boundary protocol."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode text while making special-token handling explicit."""
        ...


@dataclass(frozen=True, slots=True)
class TeacherForcedBatch:
    """Right-padded token tensors and the lengths that define every scored position."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    continuation_mask: torch.Tensor
    continuation_ids: tuple[torch.Tensor, ...]
    context_lengths: tuple[int, ...]
    continuation_lengths: tuple[int, ...]

    @property
    def size(self) -> int:
        """Return the number of rows in this batch."""
        return len(self.context_lengths)


@dataclass(frozen=True, slots=True)
class TeacherForcedResult:
    """Full-vocabulary logits and continuation log-probabilities for one model."""

    logits: tuple[torch.Tensor, ...]
    continuation_logprobs: tuple[torch.Tensor, ...]
    continuation_ids: tuple[torch.Tensor, ...]
    context_lengths: tuple[int, ...]
    continuation_lengths: tuple[int, ...]

    @property
    def size(self) -> int:
        """Return the number of rows in this result."""
        return len(self.logits)


@dataclass(frozen=True, slots=True)
class TeacherForcedKLResult:
    """Reduced teacher-forcing outputs for a device-side, chunked KL calculation.

    Unlike :class:`TeacherForcedResult`, this result deliberately has no full-vocabulary logits.
    ``per_position_kl`` and the top-k tensors are CPU float32/int64 values whose size scales with
    continuation length rather than vocabulary size.
    """

    kl: tuple[torch.Tensor, ...]
    base_top_token_ids: tuple[torch.Tensor, ...]
    base_top_logits: tuple[torch.Tensor, ...]
    comparison_top_token_ids: tuple[torch.Tensor, ...]
    comparison_top_logits: tuple[torch.Tensor, ...]
    continuation_ids: tuple[torch.Tensor, ...]
    context_lengths: tuple[int, ...]
    continuation_lengths: tuple[int, ...]

    @property
    def size(self) -> int:
        """Return the number of rows in this result."""
        return len(self.kl)

    @property
    def per_position_kl(self) -> tuple[torch.Tensor, ...]:
        """Return the CPU float32 KL vector for each continuation row."""
        return self.kl

    @property
    def adapter_top_token_ids(self) -> tuple[torch.Tensor, ...]:
        """Return compact adapter top-k token ids for each continuation row."""
        return self.comparison_top_token_ids

    @property
    def adapter_top_logits(self) -> tuple[torch.Tensor, ...]:
        """Return compact adapter top-k logits for each continuation row."""
        return self.comparison_top_logits


def _as_token_ids(tokenizer: TeacherForcingTokenizer, text: str, *, special: bool) -> list[int]:
    """Encode one string and reject tokenizers that return a non-integral sequence."""
    encoded = tokenizer.encode(text, add_special_tokens=special)
    token_ids = [int(token_id) for token_id in encoded]
    if not token_ids:
        raise ValueError(
            "teacher-forced contexts and continuations must each contain at least one token"
        )
    return token_ids


def build_batch(  # noqa: C901 - validation and padding are one load-bearing boundary
    tokenizer: TeacherForcingTokenizer,
    contexts: Sequence[str],
    continuations: Sequence[str],
    *,
    context_token_ids: Sequence[Sequence[int]] | None = None,
    continuation_token_ids: Sequence[Sequence[int]] | None = None,
) -> TeacherForcedBatch:
    """Tokenize context/continuation pairs and right-pad one batch.

    ``contexts`` must already be rendered model inputs (for example, the output of
    :func:`games.interp_mediation.render_context`).  When ``continuation_token_ids`` is omitted,
    the continuation is appended verbatim and the retokenisation guard refuses a pair whose
    boundary changes the token sequence.  When exact continuation ids are supplied by a generation
    engine, those ids are authoritative: a decoded completion need not round-trip through the
    tokenizer at the context boundary, so checking its text would reject the sequence the engine
    actually generated.  Text-supplied continuations always retain the boundary guard.
    """
    if len(contexts) != len(continuations):
        raise ValueError(
            f"contexts and continuations must have equal lengths, got {len(contexts)} and "
            f"{len(continuations)}"
        )
    if not contexts:
        raise ValueError("cannot build an empty teacher-forced batch")
    if tokenizer.pad_token_id is None:
        raise ValueError("teacher forcing requires tokenizer.pad_token_id for right padding")
    if context_token_ids is not None and len(context_token_ids) != len(contexts):
        raise ValueError("context_token_ids must have one row per context")
    if continuation_token_ids is not None and len(continuation_token_ids) != len(continuations):
        raise ValueError("continuation_token_ids must have one row per continuation")

    context_token_rows: list[list[int]] = []
    continuation_token_rows: list[list[int]] = []
    for row_index, (context, continuation) in enumerate(zip(contexts, continuations, strict=True)):
        context_ids = _as_token_ids(tokenizer, context, special=True)
        if context_token_ids is not None:
            supplied_context_ids = [int(token_id) for token_id in context_token_ids[row_index]]
            if supplied_context_ids != context_ids:
                raise ValueError("saved context token ids do not match the current tokenizer")
        continuation_ids_row = (
            [int(token_id) for token_id in continuation_token_ids[row_index]]
            if continuation_token_ids is not None
            else _as_token_ids(tokenizer, continuation, special=False)
        )
        if not continuation_ids_row:
            raise ValueError("teacher-forced continuations must each contain at least one token")
        if continuation_token_ids is None:
            joint_ids = assert_clean_boundary(tokenizer, context, continuation)
            expected_joint = context_ids + continuation_ids_row
            if joint_ids != expected_joint:
                raise RuntimeError(
                    "the tokenizer boundary guard returned ids different from the explicit context "
                    "and continuation encodings; refusing to score an ambiguous tokenization"
                )
        context_token_rows.append(context_ids)
        continuation_token_rows.append(continuation_ids_row)

    context_lengths = tuple(len(row) for row in context_token_rows)
    continuation_lengths = tuple(len(row) for row in continuation_token_rows)
    full_lengths = tuple(
        context_length + continuation_length
        for context_length, continuation_length in zip(
            context_lengths, continuation_lengths, strict=True
        )
    )
    padded_width = max(full_lengths)
    pad_token_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(full_lengths), padded_width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    continuation_mask = torch.zeros(
        (len(full_lengths), max(continuation_lengths)), dtype=torch.bool
    )
    continuation_ids: list[torch.Tensor] = []
    for row_index, (context_ids, continuation_ids_row, full_length) in enumerate(
        zip(context_token_rows, continuation_token_rows, full_lengths, strict=True)
    ):
        full_ids = context_ids + continuation_ids_row
        input_ids[row_index, :full_length] = torch.tensor(full_ids, dtype=torch.long)
        attention_mask[row_index, :full_length] = 1
        continuation_mask[row_index, : len(continuation_ids_row)] = True
        continuation_ids.append(torch.tensor(continuation_ids_row, dtype=torch.long))

    return TeacherForcedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        continuation_mask=continuation_mask,
        continuation_ids=tuple(continuation_ids),
        context_lengths=context_lengths,
        continuation_lengths=continuation_lengths,
    )


def _model_device(model: nn.Module) -> torch.device:
    """Return the device of the first model parameter, refusing a parameterless model."""
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("teacher forcing requires a model with at least one parameter") from error


def _forward_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Run the model for exactly the positions that predict the continuations."""
    outputs = cast("Any", model)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=positions,
        use_cache=False,
    )
    logits = cast("torch.Tensor", outputs.logits)
    if (
        logits.ndim != _EXPECTED_LOGITS_RANK
        or logits.shape[0] != input_ids.shape[0]
        or logits.shape[1] != positions.numel()
    ):
        raise RuntimeError(
            "the model returned logits with shape "
            f"{tuple(logits.shape)} for {input_ids.shape[0]} rows and "
            f"{positions.numel()} requested positions"
        )
    return logits


def _validate_batch(batch: TeacherForcedBatch) -> None:
    """Reject malformed tensor/length metadata before any model work begins."""
    if batch.size == 0:
        raise ValueError("cannot teacher-force an empty batch")
    if (
        batch.input_ids.ndim != _EXPECTED_INPUT_RANK
        or batch.attention_mask.shape != batch.input_ids.shape
    ):
        raise ValueError(
            "input_ids and attention_mask must be two-dimensional tensors of equal shape"
        )
    if batch.continuation_mask.shape[0] != batch.size:
        raise ValueError("continuation_mask row count does not match the batch")
    if len(batch.continuation_ids) != batch.size:
        raise ValueError("continuation_ids row count does not match the batch")
    if len(batch.continuation_lengths) != batch.size:
        raise ValueError("continuation_lengths row count does not match the batch")


def _group_rows(batch: TeacherForcedBatch) -> dict[int, list[int]]:
    """Group rows by context length, validating lengths used to select logits."""
    grouped_rows: dict[int, list[int]] = defaultdict(list)
    for row_index, context_length in enumerate(batch.context_lengths):
        if context_length < 1:
            raise ValueError(f"context length must be positive, got {context_length}")
        if batch.continuation_lengths[row_index] < 1:
            raise ValueError("continuation lengths must be positive")
        grouped_rows[context_length].append(row_index)
    return grouped_rows


def _score_group(
    model: nn.Module,
    batch: TeacherForcedBatch,
    model_device: torch.device,
    context_length: int,
    row_indices: Sequence[int],
) -> tuple[tuple[int, torch.Tensor], ...]:
    """Forward one same-context-length group and trim each row to its real continuation."""
    max_continuation = max(batch.continuation_lengths[index] for index in row_indices)
    sequence_width = context_length + max_continuation
    rows = torch.tensor(row_indices, dtype=torch.long)
    input_ids = batch.input_ids.index_select(0, rows)[:, :sequence_width].to(model_device)
    attention_mask = batch.attention_mask.index_select(0, rows)[:, :sequence_width].to(model_device)
    positions = torch.arange(
        context_length - 1,
        context_length - 1 + max_continuation,
        dtype=torch.long,
        device=model_device,
    )
    with torch.autocast(device_type=model_device.type, dtype=torch.bfloat16):
        logits = _forward_logits(model, input_ids, attention_mask, positions)
    logits = logits.detach().to(device="cpu", dtype=torch.float32)
    return tuple(
        (
            row_index,
            logits[local_index, : batch.continuation_lengths[row_index]].contiguous(),
        )
        for local_index, row_index in enumerate(row_indices)
    )


def _validate_kl_result_inputs(  # noqa: PLR0913, PLR0917 - validation mirrors public options
    batch: TeacherForcedBatch,
    reference_model: nn.Module,
    comparison_model: nn.Module,
    position_chunk_size: int,
    top_count: int,
    row_group_size: int,
) -> tuple[torch.device, torch.device]:
    """Validate models and reductions before starting a potentially expensive forward pass."""
    _validate_batch(batch)
    if position_chunk_size < 1:
        raise ValueError("position_chunk_size must be positive")
    if top_count < 1:
        raise ValueError("top_count must be positive")
    if row_group_size < 1:
        raise ValueError("row_group_size must be positive")
    reference_device = _model_device(reference_model)
    comparison_device = _model_device(comparison_model)
    if reference_device != comparison_device:
        raise ValueError(
            "chunked KL requires reference and comparison models on the same device, got "
            f"{reference_device} and {comparison_device}"
        )
    return reference_device, comparison_device


def _chunk_kl_and_top_predictions(
    reference_logits: torch.Tensor,
    comparison_logits: torch.Tensor,
    *,
    top_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce one pair of bounded device-resident logit chunks before host transfer."""
    if reference_logits.shape != comparison_logits.shape:
        raise RuntimeError(
            "reference and comparison models returned different logit shapes: "
            f"{tuple(reference_logits.shape)} and {tuple(comparison_logits.shape)}"
        )
    if reference_logits.ndim != _EXPECTED_LOGITS_RANK:
        raise RuntimeError(
            "reference and comparison models must return rank-three logits, got "
            f"{reference_logits.ndim}"
        )
    top_count = min(top_count, reference_logits.shape[-1])
    reference_top_logits, reference_top_token_ids = torch.topk(
        reference_logits.float(), k=top_count, dim=-1
    )
    comparison_top_logits, comparison_top_token_ids = torch.topk(
        comparison_logits.float(), k=top_count, dim=-1
    )
    # Keep only three reduced workspaces after each LM-head output is available.  Mutating the
    # reference log-probabilities in place avoids an additional full-vocabulary difference tensor.
    reference_logprobs = torch.log_softmax(reference_logits.float(), dim=-1)
    comparison_logprobs = torch.log_softmax(comparison_logits.float(), dim=-1)
    reference_probs = reference_logprobs.exp()
    reference_logprobs.sub_(comparison_logprobs)
    reference_probs.mul_(reference_logprobs)
    kl = reference_probs.sum(dim=-1)
    return (
        kl.float(),
        reference_top_token_ids,
        reference_top_logits.float(),
        comparison_top_token_ids,
        comparison_top_logits.float(),
    )


@torch.inference_mode()
def teacher_force_kl(  # noqa: PLR0913 - context hooks keep one shared forward loop
    reference_model: nn.Module,
    comparison_model: nn.Module,
    batch: TeacherForcedBatch,
    *,
    position_chunk_size: int = 256,
    top_count: int = 3,
    row_group_size: int = 1,
    reference_context: Callable[[], AbstractContextManager[None]] | None = None,
    comparison_context: Callable[[], AbstractContextManager[None]] | None = None,
) -> TeacherForcedKLResult:
    """Compute per-position KL and top predictions without retaining full logits on the host.

    Each equal-context-length row group is forwarded in bounded position windows.  The two models'
    full-vocabulary logits coexist only for one such window on the model device; KL and top-k
    reductions happen before any value is copied to CPU.  Consequently host memory is independent
    of vocabulary size and GPU peak memory is bounded by the requested position window.
    """
    model_device, _ = _validate_kl_result_inputs(
        batch,
        reference_model,
        comparison_model,
        position_chunk_size,
        top_count,
        row_group_size,
    )
    reference_model.eval()
    comparison_model.eval()
    grouped_rows = _group_rows(batch)

    kl_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch.size)]
    reference_ids_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch.size)]
    reference_logits_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch.size)]
    comparison_ids_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch.size)]
    comparison_logits_by_row: list[list[torch.Tensor]] = [[] for _ in range(batch.size)]

    for context_length, grouped_row_indices in grouped_rows.items():
        for group_start in range(0, len(grouped_row_indices), row_group_size):
            row_indices = grouped_row_indices[group_start : group_start + row_group_size]
            max_continuation = max(batch.continuation_lengths[index] for index in row_indices)
            sequence_width = context_length + max_continuation
            rows = torch.tensor(row_indices, dtype=torch.long)
            input_ids = batch.input_ids.index_select(0, rows)[:, :sequence_width].to(model_device)
            attention_mask = batch.attention_mask.index_select(0, rows)[:, :sequence_width].to(
                model_device
            )
            for offset in range(0, max_continuation, position_chunk_size):
                chunk_length = min(position_chunk_size, max_continuation - offset)
                positions = torch.arange(
                    context_length - 1 + offset,
                    context_length - 1 + offset + chunk_length,
                    dtype=torch.long,
                    device=model_device,
                )
                with torch.autocast(device_type=model_device.type, dtype=torch.bfloat16):
                    with reference_context() if reference_context is not None else torch.no_grad():
                        reference_logits = _forward_logits(
                            reference_model, input_ids, attention_mask, positions
                        )
                    with (
                        comparison_context() if comparison_context is not None else torch.no_grad()
                    ):
                        comparison_logits = _forward_logits(
                            comparison_model, input_ids, attention_mask, positions
                        )
                reduced = _chunk_kl_and_top_predictions(
                    reference_logits, comparison_logits, top_count=top_count
                )
                for local_index, row_index in enumerate(row_indices):
                    valid_length = min(
                        batch.continuation_lengths[row_index] - offset,
                        chunk_length,
                    )
                    if valid_length <= 0:
                        continue
                    (
                        kl,
                        reference_ids,
                        reference_logits_top,
                        comparison_ids,
                        comparison_logits_top,
                    ) = reduced
                    kl_by_row[row_index].append(kl[local_index, :valid_length].float().cpu())
                    reference_ids_by_row[row_index].append(
                        reference_ids[local_index, :valid_length].long().cpu()
                    )
                    reference_logits_by_row[row_index].append(
                        reference_logits_top[local_index, :valid_length].float().cpu()
                    )
                    comparison_ids_by_row[row_index].append(
                        comparison_ids[local_index, :valid_length].long().cpu()
                    )
                    comparison_logits_by_row[row_index].append(
                        comparison_logits_top[local_index, :valid_length].float().cpu()
                    )
                # Do not let a completed window coexist with the next window's two LM-head outputs.
                # This matters at Qwen3.5's 248,320-token vocabulary even though only reduced values
                # are retained in the result.
                del reference_logits, comparison_logits

    def concatenate(chunks: list[list[torch.Tensor]], row_index: int) -> torch.Tensor:
        if not chunks[row_index]:
            raise RuntimeError(f"chunked KL did not produce row {row_index}")
        return torch.cat(chunks[row_index], dim=0).contiguous()

    return TeacherForcedKLResult(
        kl=tuple(concatenate(kl_by_row, row_index) for row_index in range(batch.size)),
        base_top_token_ids=tuple(
            concatenate(reference_ids_by_row, row_index) for row_index in range(batch.size)
        ),
        base_top_logits=tuple(
            concatenate(reference_logits_by_row, row_index) for row_index in range(batch.size)
        ),
        comparison_top_token_ids=tuple(
            concatenate(comparison_ids_by_row, row_index) for row_index in range(batch.size)
        ),
        comparison_top_logits=tuple(
            concatenate(comparison_logits_by_row, row_index) for row_index in range(batch.size)
        ),
        continuation_ids=batch.continuation_ids,
        context_lengths=batch.context_lengths,
        continuation_lengths=batch.continuation_lengths,
    )


def chunked_per_position_kl(  # noqa: PLR0913 - context hooks mirror teacher_force_kl
    reference_model: nn.Module,
    comparison_model: nn.Module,
    batch: TeacherForcedBatch,
    *,
    position_chunk_size: int = 256,
    row_group_size: int = 1,
    reference_context: Callable[[], AbstractContextManager[None]] | None = None,
    comparison_context: Callable[[], AbstractContextManager[None]] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return only chunked per-position KL vectors for callers that need no top-k details."""
    return teacher_force_kl(
        reference_model,
        comparison_model,
        batch,
        position_chunk_size=position_chunk_size,
        top_count=1,
        row_group_size=row_group_size,
        reference_context=reference_context,
        comparison_context=comparison_context,
    ).kl


@torch.inference_mode()
def teacher_force(model: nn.Module, batch: TeacherForcedBatch) -> TeacherForcedResult:
    """Score a right-padded batch, retaining full-vocabulary logits per continuation position.

    Rows with the same context length share one forward pass.  This keeps the LM head projection
    narrow through ``logits_to_keep`` even when continuation lengths differ, while retaining true
    batching within every context-length group.  The model is put in eval mode and executed under
    bfloat16 autocast; returned tensors are detached CPU float32 values.
    """
    _validate_batch(batch)
    model_device = _model_device(model)
    model.eval()
    grouped_rows = _group_rows(batch)

    logits_by_row: list[torch.Tensor | None] = [None] * batch.size
    for context_length, row_indices in grouped_rows.items():
        for row_index, row_logits in _score_group(
            model, batch, model_device, context_length, row_indices
        ):
            logits_by_row[row_index] = row_logits

    if any(row_logits is None for row_logits in logits_by_row):
        raise RuntimeError("teacher forcing did not produce logits for every input row")
    completed_logits = tuple(cast("torch.Tensor", row_logits) for row_logits in logits_by_row)
    continuation_logprobs = tuple(
        torch.log_softmax(row_logits, dim=-1)
        .gather(-1, batch.continuation_ids[row_index].unsqueeze(-1))
        .squeeze(-1)
        .contiguous()
        for row_index, row_logits in enumerate(completed_logits)
    )
    return TeacherForcedResult(
        logits=completed_logits,
        continuation_logprobs=continuation_logprobs,
        continuation_ids=batch.continuation_ids,
        context_lengths=batch.context_lengths,
        continuation_lengths=batch.continuation_lengths,
    )


def continuation_logprob_sums(result: TeacherForcedResult) -> tuple[float, ...]:
    """Return one summed continuation log-probability per row."""
    return tuple(float(logprobs.sum().item()) for logprobs in result.continuation_logprobs)


def continuation_logprob_means(result: TeacherForcedResult) -> tuple[float, ...]:
    """Return one per-token mean continuation log-probability per row."""
    return tuple(float(logprobs.mean().item()) for logprobs in result.continuation_logprobs)


def per_position_kl(
    reference: TeacherForcedResult,
    comparison: TeacherForcedResult,
) -> tuple[torch.Tensor, ...]:
    """Return ``KL(reference || comparison)`` at every scored continuation position.

    The operands must have identical context/continuation lengths.  This is deliberately checked
    rather than relying on broadcasting: comparing a base run to a differently tokenized adapter
    run would produce a plausible tensor with the wrong scientific meaning.
    """
    if reference.context_lengths != comparison.context_lengths:
        raise ValueError("KL operands have different context lengths")
    if reference.continuation_lengths != comparison.continuation_lengths:
        raise ValueError("KL operands have different continuation lengths")
    for reference_ids, comparison_ids in zip(
        reference.continuation_ids, comparison.continuation_ids, strict=True
    ):
        if not torch.equal(reference_ids, comparison_ids):
            raise ValueError("KL operands were scored on different continuation token ids")
    values: list[torch.Tensor] = []
    for reference_logits, comparison_logits in zip(
        reference.logits, comparison.logits, strict=True
    ):
        if reference_logits.shape != comparison_logits.shape:
            raise ValueError("KL operands have different full-vocabulary logit shapes")
        reference_logprobs = torch.log_softmax(reference_logits.float(), dim=-1)
        comparison_logprobs = torch.log_softmax(comparison_logits.float(), dim=-1)
        values.append(
            (reference_logprobs.exp() * (reference_logprobs - comparison_logprobs))
            .sum(dim=-1)
            .float()
        )
    return tuple(values)


__all__ = [
    "TeacherForcedBatch",
    "TeacherForcedKLResult",
    "TeacherForcedResult",
    "TeacherForcingTokenizer",
    "build_batch",
    "chunked_per_position_kl",
    "continuation_logprob_means",
    "continuation_logprob_sums",
    "per_position_kl",
    "teacher_force",
    "teacher_force_kl",
]
