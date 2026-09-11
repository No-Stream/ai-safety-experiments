"""One tensor's RL delta against its base, and how far it moved.

Every delta is ``rl - round(base, rl.dtype)``: the release stores the DeltaNet ``A_log`` and its
group norm at bfloat16 where the base keeps float32, and without the rounding those tensors would
read as fully changed by the storage cast alone. Elementwise arithmetic is float32; **every
reduction is float64**, accumulated over chunks so the vocab-sized matrices never exist as one
float64 copy. That is not pedantry: ``torch.norm`` of a float32 tensor with a billion entries came
out 4% low on a synthetic check (301.1 against a true 313.5), which would have put every cosine and
relative delta of the embedding and lm_head rows off by a similar margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

REDUCTION_CHUNK_ELEMENTS = 1 << 26
"""Elements per float64 chunk in a reduction: 512 MB of float64 at a time."""


def _flat_chunks(tensor: torch.Tensor) -> Iterator[torch.Tensor]:
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), REDUCTION_CHUNK_ELEMENTS):
        yield flat[start : start + REDUCTION_CHUNK_ELEMENTS]


def sum_of_squares(tensor: torch.Tensor) -> float:
    """``sum(x^2)`` accumulated in float64 over chunks."""
    return sum(float(chunk.double().square().sum()) for chunk in _flat_chunks(tensor))


def frobenius_norm(tensor: torch.Tensor) -> float:
    """``||x||_F`` in float64, exact to rounding whatever the tensor's size."""
    return math.sqrt(sum_of_squares(tensor))


def inner_product(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return ``<a, b>`` over all entries, accumulated in float64 over chunks."""
    if a.shape != b.shape:
        raise ValueError(f"inner product of shapes {tuple(a.shape)} and {tuple(b.shape)}")
    return sum(
        float(torch.dot(chunk_a.double(), chunk_b.double()))
        for chunk_a, chunk_b in zip(_flat_chunks(a), _flat_chunks(b), strict=True)
    )


def row_norms(matrix: torch.Tensor) -> torch.Tensor:
    """Per-row ``||row||_2`` of a matrix in float64, chunked over rows."""
    if matrix.ndim != 2:  # noqa: PLR2004 - a matrix has two dimensions
        raise ValueError(f"row norms of a {matrix.ndim}-D tensor")
    out = torch.empty(matrix.shape[0], dtype=torch.float64)
    rows_per_chunk = max(1, REDUCTION_CHUNK_ELEMENTS // max(matrix.shape[1], 1))
    for start in range(0, matrix.shape[0], rows_per_chunk):
        block = matrix[start : start + rows_per_chunk].double()
        out[start : start + rows_per_chunk] = block.square().sum(dim=1).sqrt()
    return out


@dataclass(frozen=True)
class AlignedDelta:
    """Base and delta in float32, the base first rounded to the release's storage dtype.

    ``base32`` is exactly representable in ``storage_dtype``, which is what makes the rounding
    perturbations of the floor well defined: ``round(base32 + noise) - base32`` is zero wherever the
    noise did not cross a rounding boundary.
    """

    base32: torch.Tensor
    delta32: torch.Tensor
    storage_dtype: torch.dtype
    base_storage_dtype: torch.dtype


def aligned_delta(base: torch.Tensor, rl: torch.Tensor) -> AlignedDelta:
    """``rl - round(base, rl.dtype)`` in float32, refusing a shape mismatch."""
    if base.shape != rl.shape:
        raise ValueError(f"base {tuple(base.shape)} and rl {tuple(rl.shape)} differ in shape")
    base32 = base.to(rl.dtype).to(torch.float32)
    delta32 = rl.to(torch.float32) - base32
    return AlignedDelta(
        base32=base32, delta32=delta32, storage_dtype=rl.dtype, base_storage_dtype=base.dtype
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def one_ulp_moves(base32: torch.Tensor, delta32: torch.Tensor) -> int:
    """Count entries whose bfloat16 value moved to an adjacent representable value.

    Read off the bit patterns: bfloat16 values of one sign are ordered like their int16 patterns,
    so two values are neighbours exactly when the patterns differ by one. A move across zero is
    not counted as one step (it never is at these magnitudes).
    """
    base_bits = base32.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    rl_bits = (base32 + delta32).to(torch.bfloat16).view(torch.int16).to(torch.int32)
    same_sign = (base_bits < 0) == (rl_bits < 0)
    return int((((base_bits - rl_bits).abs() == 1) & same_sign).sum())


@dataclass(frozen=True)
class TensorMovement:
    """How far one tensor moved, in norms and in counts of entries."""

    shape: tuple[int, ...]
    n_elements: int
    storage_dtype: str
    base_storage_dtype: str
    base_norm: float
    delta_norm: float
    relative_delta: float
    delta_rms: float
    max_abs_delta: float
    n_changed: int
    changed_fraction: float
    n_one_ulp: int | None
    one_ulp_share_of_changed: float | None
    cosine_delta_base: float | None


def tensor_movement(delta: AlignedDelta) -> TensorMovement:
    """Norms, changed-entry counts and the one-ulp share for one aligned delta."""
    base32, delta32 = delta.base32, delta.delta32
    n_elements = delta32.numel()
    base_norm = frobenius_norm(base32)
    delta_norm = frobenius_norm(delta32)
    n_changed = int((delta32 != 0).sum())
    n_one_ulp: int | None = None
    one_ulp_share: float | None = None
    if delta.storage_dtype == torch.bfloat16:
        n_one_ulp = one_ulp_moves(base32, delta32)
        one_ulp_share = n_one_ulp / n_changed if n_changed else None
    cosine: float | None = None
    if base_norm > 0 and delta_norm > 0:
        cosine = inner_product(base32, delta32) / (base_norm * delta_norm)
    return TensorMovement(
        shape=tuple(delta32.shape),
        n_elements=n_elements,
        storage_dtype=_dtype_name(delta.storage_dtype),
        base_storage_dtype=_dtype_name(delta.base_storage_dtype),
        base_norm=base_norm,
        delta_norm=delta_norm,
        relative_delta=delta_norm / base_norm if base_norm else 0.0,
        delta_rms=delta_norm / math.sqrt(n_elements) if n_elements else 0.0,
        max_abs_delta=float(delta32.abs().max()) if n_elements else 0.0,
        n_changed=n_changed,
        changed_fraction=n_changed / n_elements if n_elements else 0.0,
        n_one_ulp=n_one_ulp,
        one_ulp_share_of_changed=one_ulp_share,
        cosine_delta_base=cosine,
    )


def changed_fraction(delta32: torch.Tensor) -> float:
    """Share of entries that are not exactly zero."""
    return float((delta32 != 0).sum()) / delta32.numel() if delta32.numel() else 0.0
