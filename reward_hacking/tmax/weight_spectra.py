"""Singular spectra and the shape statistics read off them, for weight tensors and their deltas.

Definitions, all on the descending singular values ``s`` of one matrix:

* **stable rank** ``sum(s^2) / s_1^2``, the number of equal directions the Frobenius mass would
  fill at the top singular value's strength;
* **effective rank** ``exp(H)`` with ``H`` the Shannon entropy of ``s / sum(s)`` (Roy & Vetterli
  2007), and **normalised entropy** ``H / log(n)`` so it reads on ``[0, 1]``;
* **top-k energy** ``sum(s_{1..k}^2) / sum(s^2)``, the share of squared mass in the top ``k``
  directions.

A zero matrix leaves every one of these undefined and they are reported as ``None`` rather than a
number that happens to fall out of a division.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

GRAM_MIN_DIM = 512
"""Smaller side at or above which singular values come from a float64 Gram matrix's eigenvalues.

Below it the matrix is small enough for exact ``svdvals`` in float64. The Gram route squares the
condition number, so singular values under about ``1e-8 s_1`` lose accuracy; none of the shape
statistics here depend on that tail.
"""

DEFAULT_TOP_K_SPECTRUM = 64
"""How many leading singular values each row keeps, enough to plot the head of any spectrum."""

MATRIX_NDIM = 2


def as_matrix(tensor: torch.Tensor) -> torch.Tensor | None:
    """Read a tensor as the matrix its spectrum is taken of; ``None`` for vectors and scalars.

    A 3-D convolution weight ``[channels, 1, kernel]`` flattens to ``[channels, kernel]``.
    """
    if tensor.ndim < MATRIX_NDIM:
        return None
    if tensor.ndim == MATRIX_NDIM:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def singular_values(matrix: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Descending singular values in float64 and the method that produced them.

    Above :data:`GRAM_MIN_DIM` the Gram matrix over the smaller side is accumulated in float64 over
    row chunks, so the vocab-sized embedding never exists as one float64 copy, and its eigenvalues
    are square-rooted; below it, exact ``svdvals``.
    """
    if matrix.ndim != MATRIX_NDIM:
        raise ValueError(f"expected a matrix, got shape {tuple(matrix.shape)}")
    rows, cols = matrix.shape
    if min(rows, cols) < GRAM_MIN_DIM:
        return torch.linalg.svdvals(matrix.to(torch.float64)), "svdvals_float64"
    tall = matrix if rows >= cols else matrix.T
    side = tall.shape[1]
    gram = torch.zeros(side, side, dtype=torch.float64)
    chunk_rows = max(1, (1 << 27) // side)
    for start in range(0, tall.shape[0], chunk_rows):
        block = tall[start : start + chunk_rows].to(torch.float64)
        gram.addmm_(block.T, block)
    eigenvalues = torch.linalg.eigvalsh(gram)
    return eigenvalues.clamp_min(0).sqrt().flip(0), "gram_eigvalsh_float64"


@dataclass(frozen=True)
class SpectralShape:
    """Shape statistics of one matrix's spectrum; ``None`` where a zero matrix leaves them undefined."""

    method: str
    n_singular_values: int
    frobenius: float
    stable_rank: float | None
    effective_rank: float | None
    sv_entropy_normalized: float | None
    top1_energy: float | None
    top8_energy: float | None
    top64_energy: float | None
    singular_values_top: tuple[float, ...]


def spectral_shape(matrix: torch.Tensor, *, top_k: int = DEFAULT_TOP_K_SPECTRUM) -> SpectralShape:
    """Stable rank, effective rank, normalised entropy and top-k energy shares of one matrix."""
    values, method = singular_values(matrix)
    n_values = int(values.numel())
    energy = values**2
    total_energy = float(energy.sum())
    top = tuple(float(v) for v in values[:top_k])
    if total_energy <= 0.0:
        return SpectralShape(
            method=method,
            n_singular_values=n_values,
            frobenius=0.0,
            stable_rank=None,
            effective_rank=None,
            sv_entropy_normalized=None,
            top1_energy=None,
            top8_energy=None,
            top64_energy=None,
            singular_values_top=top,
        )
    probabilities = values / values.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())

    def energy_share(k: int) -> float:
        return float(energy[: min(k, n_values)].sum()) / total_energy

    return SpectralShape(
        method=method,
        n_singular_values=n_values,
        frobenius=math.sqrt(total_energy),
        stable_rank=total_energy / float(energy[0]),
        effective_rank=math.exp(entropy),
        sv_entropy_normalized=entropy / math.log(n_values) if n_values > 1 else None,
        top1_energy=energy_share(1),
        top8_energy=energy_share(8),
        top64_energy=energy_share(64),
        singular_values_top=top,
    )
