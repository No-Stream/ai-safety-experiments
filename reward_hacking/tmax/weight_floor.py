"""How two deltas of one tensor relate, and the double-rounding null they are read against.

The releases are bfloat16 and most entries move by at most one unit in the last place, so a delta
is mostly rounding decisions: an entry flips to the neighbouring representable value or stays. Two
deltas' cosine therefore needs a null that carries the same rounding. The **floor** here perturbs
the base with Gaussian noise, rounds to the storage dtype, subtracts the base, twice with
independent seeds, and reads the cosine between the two perturbations. Two noise scales are
matched: the real delta's RMS and, stricter where the update is sparse, the scale whose rounded
perturbation flips the same fraction of entries as the real delta did. ``clears_floor`` requires
the real cosine to exceed the largest floor drawn at either scale.

One fact about that floor is worth knowing before reading it. A bfloat16 base value sits at the
exact centre of its rounding interval, so noise added to it flips up or down symmetrically and two
independent perturbations are uncorrelated: for a base that is itself exactly representable, as
every public bf16 release is, the floor above reads near zero. The correlated-rounding effect the
literature describes needs the shared base to be a *rounding of a hidden higher-precision parent*,
whose sub-ulp offset biases both checkpoints' flips the same way. That case is drawn too, as the
``offset`` floor: a shared per-entry offset uniform on half an ulp either side of the base, plus
independent noise. It is reported beside the gating floors as the bound that would apply if the
base had been such a rounding, and is not part of ``clears_floor`` because the TMAX runs were
initialised from the bf16 base itself. Seeds derive from the tensor name, so a resumed run draws
the same floor.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

from reward_hacking.tmax.weight_delta import (
    AlignedDelta,
    changed_fraction,
    frobenius_norm,
    inner_product,
)

DEFAULT_FLOOR_PAIRS = 3
SPARSITY_BISECTION_STEPS = 18
SPARSITY_BISECTION_SPAN = 1e3
"""The sparsity-matched noise scale is bisected over ``[rms / span, rms * span]`` in log space."""


def perturbation_seed(name: str, side: str, pair_index: int, kind: str) -> int:
    """Derive a seed from tensor identity only, so a resumed or re-run floor is the same floor."""
    digest = hashlib.blake2b(f"{name}|{side}|{pair_index}|{kind}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


STORAGE_MANTISSA_BITS: dict[torch.dtype, int] = {
    torch.bfloat16: 7,
    torch.float16: 10,
    torch.float32: 23,
}


def storage_spacing(base32: torch.Tensor, storage_dtype: torch.dtype) -> torch.Tensor:
    """Spacing of ``storage_dtype`` at each entry of ``base32`` (one ulp), zero at exact zeros."""
    if storage_dtype not in STORAGE_MANTISSA_BITS:
        raise ValueError(f"no spacing rule for storage dtype {storage_dtype}")
    _, exponent = torch.frexp(base32)
    spacing = torch.ldexp(
        torch.ones_like(base32), exponent - 1 - STORAGE_MANTISSA_BITS[storage_dtype]
    )
    return torch.where(base32 == 0, torch.zeros_like(spacing), spacing)


def rounded_perturbation(
    base32: torch.Tensor,
    rms: float,
    storage_dtype: torch.dtype,
    seed: int,
    *,
    offset_seed: int | None = None,
) -> torch.Tensor:
    """``round(base + N(0, rms^2)) - base`` in float32: what an unrelated update of that size leaves.

    With ``offset_seed`` the base is first displaced by a per-entry offset uniform on half an ulp
    either side, the hidden sub-ulp position a rounded parent would carry; two draws sharing the
    offset seed model two checkpoints rounded from the same higher-precision parent.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(base32.shape, generator=generator, dtype=torch.float32)
    noise.mul_(rms).add_(base32)
    if offset_seed is not None:
        offset_generator = torch.Generator(device="cpu").manual_seed(offset_seed)
        offset = torch.rand(base32.shape, generator=offset_generator, dtype=torch.float32)
        offset.sub_(0.5).mul_(storage_spacing(base32, storage_dtype))
        noise.add_(offset)
        del offset
    return noise.to(storage_dtype).to(torch.float32).sub_(base32)


def sparsity_matched_rms(
    base32: torch.Tensor,
    *,
    target_changed_fraction: float,
    rms_hint: float,
    storage_dtype: torch.dtype,
    seed: int,
) -> float:
    """Find the noise scale whose rounded perturbation flips the target fraction of entries.

    The flipped fraction rises monotonically with the noise scale, so a bisection in log space
    over ``[rms_hint / SPARSITY_BISECTION_SPAN, rms_hint * SPARSITY_BISECTION_SPAN]`` finds it; an
    endpoint comes back when the target lies outside what noise can produce (a tensor that changed
    everywhere, or nowhere).
    """
    if rms_hint <= 0.0:
        raise ValueError("rms_hint must be positive")
    span = SPARSITY_BISECTION_SPAN
    low, high = math.log(rms_hint / span), math.log(rms_hint * span)
    for _ in range(SPARSITY_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        fraction = changed_fraction(
            rounded_perturbation(base32, math.exp(middle), storage_dtype, seed)
        )
        if fraction < target_changed_fraction:
            low = middle
        else:
            high = middle
    return math.exp(0.5 * (low + high))


def _inner_and_norms(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    return inner_product(a, b), frobenius_norm(a), frobenius_norm(b)


def cosine_from_inner(inner: float, norm_a: float, norm_b: float) -> float | None:
    """Cosine from an inner product and two norms; ``None`` when either side is zero."""
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return inner / (norm_a * norm_b)


@dataclass(frozen=True)
class FloorDraws:
    """One floor kind's independent perturbation pairs, kept as inner products so they aggregate.

    A group-level floor (a layer, a family) is the cosine of the concatenated perturbations, which
    is ``sum(inner) / sqrt(sum(norm_a^2) sum(norm_b^2))`` over the group's tensors per pair index;
    storing the three numbers per pair rather than the cosine is what makes that roll-up exact.
    """

    inner: tuple[float, ...]
    norm_a: tuple[float, ...]
    norm_b: tuple[float, ...]
    rms_a: float
    rms_b: float

    @property
    def cosines(self) -> tuple[float | None, ...]:
        """One cosine per independent pair, ``None`` where a side had no flips at all."""
        return tuple(
            cosine_from_inner(inner, norm_a, norm_b)
            for inner, norm_a, norm_b in zip(self.inner, self.norm_a, self.norm_b, strict=True)
        )

    @property
    def max_cosine(self) -> float | None:
        """The strictest of the pairs: what a real cosine has to exceed."""
        defined = [c for c in self.cosines if c is not None]
        return max(defined) if defined else None


@dataclass(frozen=True)
class DeltaPair:
    """Two aligned deltas of the same tensor, the earlier checkpoint first."""

    name: str
    a: AlignedDelta
    b: AlignedDelta


FLOOR_KIND_RMS = "rms"
FLOOR_KIND_SPARSITY = "sparsity"
FLOOR_KIND_OFFSET = "offset"
FLOOR_KINDS: tuple[str, ...] = (FLOOR_KIND_RMS, FLOOR_KIND_SPARSITY, FLOOR_KIND_OFFSET)
GATING_FLOOR_KINDS: tuple[str, ...] = (FLOOR_KIND_RMS, FLOOR_KIND_SPARSITY)
"""The floors ``clears_floor`` is read against; the offset floor is reported, not gated on."""


def draw_floor(
    pair: DeltaPair, *, kind: str, scales: tuple[float, float], n_pairs: int
) -> FloorDraws:
    """Draw ``n_pairs`` independent rounded-perturbation pairs at ``scales`` (one per side).

    The ``offset`` kind gives both sides of each pair the same hidden sub-ulp offset (see
    :func:`rounded_perturbation`); the other kinds perturb the representable base directly.
    """
    if kind not in FLOOR_KINDS:
        raise ValueError(f"unknown floor kind {kind!r}; known: {FLOOR_KINDS}")
    a, b, name = pair.a, pair.b, pair.name
    rms_a, rms_b = scales
    inner: list[float] = []
    norms_a: list[float] = []
    norms_b: list[float] = []
    for pair_index in range(n_pairs):
        offset_seed = (
            perturbation_seed(name, "shared", pair_index, kind)
            if kind == FLOOR_KIND_OFFSET
            else None
        )
        perturbation_a = rounded_perturbation(
            a.base32,
            rms_a,
            a.storage_dtype,
            perturbation_seed(name, "a", pair_index, kind),
            offset_seed=offset_seed,
        )
        perturbation_b = rounded_perturbation(
            b.base32,
            rms_b,
            b.storage_dtype,
            perturbation_seed(name, "b", pair_index, kind),
            offset_seed=offset_seed,
        )
        product, norm_a, norm_b = _inner_and_norms(perturbation_a, perturbation_b)
        del perturbation_a, perturbation_b
        inner.append(product)
        norms_a.append(norm_a)
        norms_b.append(norm_b)
    return FloorDraws(
        inner=tuple(inner), norm_a=tuple(norms_a), norm_b=tuple(norms_b), rms_a=rms_a, rms_b=rms_b
    )


@dataclass(frozen=True)
class DeltaRelationship:
    """How the later delta ``b`` relates to the earlier delta ``a`` on one tensor."""

    inner: float
    norm_a: float
    norm_b: float
    cosine: float | None
    alpha_b_on_a: float | None
    residual_share: float | None
    norm_ratio_b_over_a: float | None
    changed_fraction_a: float
    changed_fraction_b: float
    floor_rms_matched: FloorDraws
    floor_sparsity_matched: FloorDraws
    floor_offset: FloorDraws

    @property
    def floors(self) -> dict[str, FloorDraws]:
        """Every floor kind drawn, keyed by kind."""
        return {
            FLOOR_KIND_RMS: self.floor_rms_matched,
            FLOOR_KIND_SPARSITY: self.floor_sparsity_matched,
            FLOOR_KIND_OFFSET: self.floor_offset,
        }

    @property
    def floor_max(self) -> float | None:
        """The strictest gating floor (rms- and sparsity-matched); the offset floor is not gated on."""
        candidates = [
            c
            for c in (self.floors[kind].max_cosine for kind in GATING_FLOOR_KINDS)
            if c is not None
        ]
        return max(candidates) if candidates else None

    @property
    def clears_floor(self) -> bool | None:
        """Whether the real cosine exceeds every floor drawn; ``None`` when either is undefined."""
        if self.cosine is None or self.floor_max is None:
            return None
        return self.cosine > self.floor_max

    def record(self) -> dict[str, object]:
        """Flatten for a JSONL row: the floors as lists, the derived readings alongside."""
        margin = (
            None if self.cosine is None or self.floor_max is None else self.cosine - self.floor_max
        )
        record: dict[str, object] = {
            "inner": self.inner,
            "norm_a": self.norm_a,
            "norm_b": self.norm_b,
            "cosine": self.cosine,
            "alpha_b_on_a": self.alpha_b_on_a,
            "residual_share": self.residual_share,
            "norm_ratio_b_over_a": self.norm_ratio_b_over_a,
            "changed_fraction_a": self.changed_fraction_a,
            "changed_fraction_b": self.changed_fraction_b,
            "floor_max": self.floor_max,
            "clears_floor": self.clears_floor,
            "margin_over_floor": margin,
        }
        for kind, draws in self.floors.items():
            record[f"floor_{kind}_inner"] = list(draws.inner)
            record[f"floor_{kind}_norm_a"] = list(draws.norm_a)
            record[f"floor_{kind}_norm_b"] = list(draws.norm_b)
            record[f"floor_{kind}_scale_a"] = draws.rms_a
            record[f"floor_{kind}_scale_b"] = draws.rms_b
            record[f"floor_{kind}_max"] = draws.max_cosine
        return record


def delta_relationship(
    *,
    name: str,
    a: AlignedDelta,
    b: AlignedDelta,
    n_floor_pairs: int = DEFAULT_FLOOR_PAIRS,
    floor_rms: tuple[float, float] | None = None,
) -> DeltaRelationship:
    """Cosine, scale fit and both floors between two deltas of the same tensor.

    ``floor_rms`` overrides the rms-matched floor's noise scales (default: each delta's own RMS);
    it exists so the floor can be scanned against the noise scale, and so a test can build deltas
    out of the floor's own perturbations and check they read as not clearing it.
    """
    inner, norm_a, norm_b = _inner_and_norms(a.delta32, b.delta32)
    cosine = cosine_from_inner(inner, norm_a, norm_b)
    alpha = inner / norm_a**2 if norm_a > 0 else None
    residual_share = None if cosine is None else max(0.0, 1.0 - cosine**2)
    n_elements = a.delta32.numel()
    rms_a = norm_a / math.sqrt(n_elements) if n_elements else 0.0
    rms_b = norm_b / math.sqrt(n_elements) if n_elements else 0.0
    fraction_a = changed_fraction(a.delta32)
    fraction_b = changed_fraction(b.delta32)
    scale_a, scale_b = floor_rms if floor_rms is not None else (rms_a, rms_b)
    empty = FloorDraws(inner=(), norm_a=(), norm_b=(), rms_a=scale_a, rms_b=scale_b)
    pair = DeltaPair(name=name, a=a, b=b)
    if scale_a <= 0.0 or scale_b <= 0.0:
        rms_floor = empty
        sparsity_floor = empty
        offset_floor = empty
    else:
        rms_floor = draw_floor(
            pair, kind=FLOOR_KIND_RMS, scales=(scale_a, scale_b), n_pairs=n_floor_pairs
        )
        offset_floor = draw_floor(
            pair, kind=FLOOR_KIND_OFFSET, scales=(scale_a, scale_b), n_pairs=n_floor_pairs
        )
        matched_a = sparsity_matched_rms(
            a.base32,
            target_changed_fraction=fraction_a,
            rms_hint=scale_a,
            storage_dtype=a.storage_dtype,
            seed=perturbation_seed(name, "a", 0, "sparsity-search"),
        )
        matched_b = sparsity_matched_rms(
            b.base32,
            target_changed_fraction=fraction_b,
            rms_hint=scale_b,
            storage_dtype=b.storage_dtype,
            seed=perturbation_seed(name, "b", 0, "sparsity-search"),
        )
        sparsity_floor = draw_floor(
            pair, kind=FLOOR_KIND_SPARSITY, scales=(matched_a, matched_b), n_pairs=n_floor_pairs
        )
    return DeltaRelationship(
        inner=inner,
        norm_a=norm_a,
        norm_b=norm_b,
        cosine=cosine,
        alpha_b_on_a=alpha,
        residual_share=residual_share,
        norm_ratio_b_over_a=norm_b / norm_a if norm_a > 0 else None,
        changed_fraction_a=fraction_a,
        changed_fraction_b=fraction_b,
        floor_rms_matched=rms_floor,
        floor_sparsity_matched=sparsity_floor,
        floor_offset=offset_floor,
    )
