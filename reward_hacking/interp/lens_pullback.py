"""Pull a vocabulary contrast back through a Jacobian lens into residual space: the ``v_lens`` direction.

A fitted lens transports a layer-``L`` residual vector into final-layer coordinates, ``J_L h + b_L``.
Its transpose runs the other way: given a direction in final-layer coordinates that the unembedding
reads as "grader vocabulary over reference vocabulary", ``J_L^T u`` is the layer-``L`` residual
direction whose transport moves the readout most along ``u``. That pullback is the one steering
direction in the wave that is derived FROM the lens rather than fitted from labels, so a lens that
decodes ``d_twin`` to assertion vocabulary and a ``v_lens`` that steers the hack readout are two
reads of one object.

The lens is only exposed here through ``transport``; the matrix is recovered numerically by
transporting the standard basis, with ``transport(0)`` as the affine part, so the same code serves
a real ``jlens`` lens and the stub lenses in the tests. The unembedding is NOT linearised here: the
caller supplies the final-layer rows of the vocabulary contrast (the effective linear unembedding at
the operating point, including the final norm's ``(1 + w)`` scale on this family -- a hand-rolled
unembed decodes plausibly and wrongly), because that is a fact about the model the lens adapter owns.

The built-in check: transporting the pulled-back direction must align with the contrast it came from
better than transporting a matched-norm random direction does. A pullback that fails it is a lens
whose transport is not the linear map the pullback assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from reward_hacking.interp.directions import cosine, matched_norm_random_direction

if TYPE_CHECKING:
    from reward_hacking.interp.jacobian import LensLike

logger = logging.getLogger(__name__)


def affine_transport(
    lens: LensLike, layer: int, *, dim: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover ``(J, b)`` with ``transport(h) = J h + b`` by transporting the standard basis."""
    bias = lens.transport(torch.zeros(dim, dtype=dtype), layer)
    columns = [
        lens.transport(torch.eye(dim, dtype=dtype)[index], layer) - bias for index in range(dim)
    ]
    return torch.stack(columns, dim=1), bias


def pullback(jacobian: torch.Tensor, readout_direction: torch.Tensor) -> torch.Tensor:
    """``J^T u``: the residual direction whose transport moves the readout most along ``u``."""
    if jacobian.shape[0] != readout_direction.shape[0]:
        raise ValueError(
            f"jacobian maps into {jacobian.shape[0]} dims but the readout direction has "
            f"{readout_direction.shape[0]}"
        )
    return jacobian.T @ readout_direction


def vocabulary_contrast(target_rows: torch.Tensor, reference_rows: torch.Tensor) -> torch.Tensor:
    """Mean target unembedding row minus mean reference row: the contrast in final-layer coordinates."""
    if target_rows.ndim != 2 or reference_rows.ndim != 2:  # noqa: PLR2004 - [tokens, hidden] by contract
        raise ValueError("unembedding rows must be [tokens, hidden] matrices")
    return target_rows.mean(dim=0) - reference_rows.mean(dim=0)


@dataclass(frozen=True)
class PullbackRead:
    """How well the pulled-back direction transports onto its own contrast, against the placebo."""

    layer: int
    alignment: float
    placebo_alignment_max: float
    n_placebos: int
    pulled_norm: float

    @property
    def beats_placebo(self) -> bool:
        """The pullback's transport aligns with the contrast better than every random direction's."""
        return self.alignment > self.placebo_alignment_max


def lens_direction(  # noqa: PLR0913 - a pullback is the lens, the layer, the contrast, its size and the placebo knobs
    lens: LensLike,
    layer: int,
    contrast: torch.Tensor,
    *,
    dim: int,
    n_placebos: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, PullbackRead]:
    """``v_lens`` at ``layer``: the pullback of ``contrast``, with the transport-alignment check."""
    jacobian, _bias = affine_transport(lens, layer, dim=dim, dtype=contrast.dtype)
    pulled = pullback(jacobian, contrast)
    alignment = cosine(jacobian @ pulled, contrast)
    placebo_alignments = [
        cosine(jacobian @ matched_norm_random_direction(pulled, generator), contrast)
        for _ in range(n_placebos)
    ]
    read = PullbackRead(
        layer=layer,
        alignment=alignment,
        placebo_alignment_max=max(placebo_alignments),
        n_placebos=n_placebos,
        pulled_norm=float(pulled.norm()),
    )
    logger.info(
        f"v_lens at layer {layer}: alignment={alignment:.3f} placebo_max={read.placebo_alignment_max:.3f} "
        f"beats_placebo={read.beats_placebo}"
    )
    return pulled, read
