"""Geometry a lens read is judged by: token-list overlap, standardisation, cosine bands, and the gate.

Three small pieces every decode in ``tmax_lens_reads`` leans on, kept separate because each is a
measurement with its own failure mode:

* **Rank overlap** between two decoded token lists (top-k Jaccard and rank-biased overlap). A decode
  is a ranking, not a probability, so "the base lens and the TMAX lens agree" is an overlap number,
  and that number means nothing until it sits beside the overlap two matched-norm random directions
  produce under the same lens -- the floor this module makes cheap to compute.
* **Standardisation** of a direction by the per-dimension scale of the residual stream. A handful of
  massive-activation dimensions once inflated a cross-model cosine from ~0.11-0.52 to 0.99
  (``RESULTS.md``), and a decode of a raw direction is dominated by the same dimensions. The
  standardized variant divides by the per-dimension std at that layer and restores the norm, so raw
  and standardized decodes are perturbations of the same size in different coordinates.
* **The stored-decode gate.** A lens is a linear map per source layer, and applying layer 12's map to
  a layer-20 residual returns a confident token list about nothing. The gate stores a few anchor
  residuals with the model's OWN next token at each anchor; a decode at layer ``L`` is admitted only
  if the lens recovers those targets at ``L`` and does worse when told the anchors sit at a far
  wrong layer. A decode that fails the gate returns no tokens, never a plausible list.

**The anchors sidecar.** The gate's anchors are written once, at lens-fit time on the GPU box, as two
files beside the lens artifact: :data:`GATE_TENSORS_FILENAME` (the residuals and targets) and
:data:`GATE_META_FILENAME` (a :class:`GateProvenance`: which lens file, which served weights, which
anchor corpus and which positions). The reader side (``tmax_lens_reads``) loads them through
:meth:`StoredDecodeGate.load` and holds them to the lens and the activations it is about to decode
through :func:`assert_gate_matches_lens_file` and :func:`assert_gate_matches_weights`, so anchors
recorded on one checkpoint can never quietly gate a lens fitted on another. This module is the one
place the format is defined; the fit driver writes it and nothing else does.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors.torch import load_file, save_file

from reward_hacking.interp.directions import STD_EPS, cosine, matched_norm_random_direction

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from reward_hacking.interp.jacobian import LensLike, UnembedModel

logger = logging.getLogger(__name__)

DEFAULT_RBO_PERSISTENCE = 0.9
GATE_SCHEMA = 1
GATE_KIND = "stored-decode-gate"
GATE_TENSORS_FILENAME = "stored-decode-gate.safetensors"
GATE_META_FILENAME = "stored-decode-gate.json"
LAYER_CONVENTION_POST_BLOCK = "post_block"
"""Row ``layer`` of an anchor is the residual AFTER block ``layer``: what ``jlens`` hooks and fits on."""
DEFAULT_GATE_TOP_K = 10
DEFAULT_MIN_HIT_RATE = 0.5
DEFAULT_MIN_MARGIN = 0.25
DEFAULT_WRONG_LAYER_OFFSET = 8


class GateProvenanceError(ValueError):
    """A gate paired with a lens, weights or corpus it was not written for."""


# --------------------------------------------------------------------------------------
# Token-list overlap
# --------------------------------------------------------------------------------------


def topk_overlap(first: Sequence[str], second: Sequence[str], k: int) -> float:
    """Jaccard overlap of the two lists' top ``k`` tokens; 1.0 is the same set, 0.0 disjoint."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    head_first, head_second = set(first[:k]), set(second[:k])
    if not head_first and not head_second:
        raise ValueError("both token lists are empty; an overlap of nothing is not a number")
    return len(head_first & head_second) / len(head_first | head_second)


def rank_biased_overlap(
    first: Sequence[str], second: Sequence[str], *, persistence: float = DEFAULT_RBO_PERSISTENCE
) -> float:
    """Rank-biased overlap (Webber et al. 2010) over the shared depth, weighting the top ranks most.

    ``persistence`` is the probability of looking one rank deeper; 0.9 puts ~86% of the weight on the
    first ten ranks. Truncated at the shorter list's length and normalised by the finite-depth total,
    so two identical lists read exactly 1.0 at any depth.
    """
    if not 0.0 < persistence < 1.0:
        raise ValueError(f"persistence must lie strictly inside (0, 1), got {persistence}")
    depth = min(len(first), len(second))
    if depth == 0:
        raise ValueError("a token list is empty; rank-biased overlap needs at least one rank")
    seen_first: set[str] = set()
    seen_second: set[str] = set()
    weighted = 0.0
    total_weight = 0.0
    for rank in range(1, depth + 1):
        seen_first.add(first[rank - 1])
        seen_second.add(second[rank - 1])
        agreement = len(seen_first & seen_second) / rank
        weight = persistence ** (rank - 1)
        weighted += weight * agreement
        total_weight += weight
    return weighted / total_weight


# --------------------------------------------------------------------------------------
# Standardisation and cosine bands
# --------------------------------------------------------------------------------------


def per_dimension_scale(features: torch.Tensor) -> torch.Tensor:
    """Per-dimension std of ``[n, d]`` activations plus the shared epsilon: the standardising scale."""
    if features.ndim != 2:  # noqa: PLR2004 - activations are a [rows, hidden] matrix by contract
        raise ValueError(f"expected [n, d] activations, got shape {tuple(features.shape)}")
    return features.std(dim=0, unbiased=True) + STD_EPS


def standardize_direction(direction: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Divide a direction by the per-dimension scale and restore its original norm.

    Restoring the norm is what keeps the raw and standardized reads comparable: both are perturbations
    of the same size, differing only in how the norm is spread across dimensions.
    """
    if direction.shape != scale.shape:
        raise ValueError(
            f"direction {tuple(direction.shape)} and scale {tuple(scale.shape)} differ"
        )
    rescaled = direction / scale
    return rescaled / rescaled.norm() * direction.norm()


@dataclass(frozen=True)
class CosineWithBand:
    """A cosine beside the |cosine| band matched-norm random directions produce at this dimension."""

    cosine: float
    placebo_abs_cosine_mean: float
    placebo_abs_cosine_max: float
    n_placebos: int

    @property
    def clears_band(self) -> bool:
        """|cosine| exceeds the largest placebo |cosine|."""
        return abs(self.cosine) > self.placebo_abs_cosine_max


def cosine_with_band(
    first: torch.Tensor, second: torch.Tensor, *, n_placebos: int, generator: torch.Generator
) -> CosineWithBand:
    """Cosine of two directions with the band of ``first`` against ``n_placebos`` random directions."""
    draws = [
        abs(cosine(first, matched_norm_random_direction(first, generator)))
        for _ in range(n_placebos)
    ]
    return CosineWithBand(
        cosine=cosine(first, second),
        placebo_abs_cosine_mean=sum(draws) / len(draws),
        placebo_abs_cosine_max=max(draws),
        n_placebos=n_placebos,
    )


def content_floor_cosines(
    target: torch.Tensor,
    controls: Mapping[str, torch.Tensor],
    *,
    n_placebos: int,
    generator: torch.Generator,
) -> dict[str, CosineWithBand]:
    """``target`` against each content control (comment line, statement literal, ...), banded.

    The content floor is what a cosine to ``target`` looks like for a direction that shares the twin
    corpus's *text* differences but not the rigged-assertion conflict; a cosine to ``d_twin`` that
    does not exceed these is a cosine to "one line differs", not to "a check is rigged".
    """
    return {
        name: cosine_with_band(target, control, n_placebos=n_placebos, generator=generator)
        for name, control in controls.items()
    }


# --------------------------------------------------------------------------------------
# The stored-decode gate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Whether a decode at ``layer`` is admitted, and the two hit rates that decided it."""

    lens_name: str
    layer: int
    wrong_layer: int
    hit_rate: float
    wrong_layer_hit_rate: float
    min_hit_rate: float
    min_margin: float
    n_anchors: int
    top_k: int

    @property
    def admitted(self) -> bool:
        """The lens recovers the anchors' own tokens at this layer and loses at the wrong one."""
        return (
            self.hit_rate >= self.min_hit_rate
            and self.hit_rate - self.wrong_layer_hit_rate >= self.min_margin
        )

    @property
    def reason(self) -> str:
        """One line saying why the gate came out as it did."""
        if self.admitted:
            return "admitted"
        if self.hit_rate < self.min_hit_rate:
            return (
                f"hit rate {self.hit_rate:.2f} at layer {self.layer} is under {self.min_hit_rate}"
            )
        return (
            f"hit rate {self.hit_rate:.2f} at layer {self.layer} is within {self.min_margin} of the "
            f"far-wrong-layer rate {self.wrong_layer_hit_rate:.2f} at layer {self.wrong_layer}"
        )


def wrong_layer_for(layer: int, *, n_layers: int, offset: int) -> int:
    """Return a layer ``offset`` away from ``layer`` that still exists: below it if possible, else above."""
    if offset <= 0:
        raise ValueError(f"the wrong-layer offset must be positive, got {offset}")
    if layer - offset >= 0:
        return layer - offset
    if layer + offset < n_layers:
        return layer + offset
    raise ValueError(f"no layer {offset} away from {layer} exists in a {n_layers}-layer capture")


@dataclass(frozen=True)
class GateProvenance:
    """Which lens file, which served weights and which anchor corpus a gate's anchors were recorded on.

    ``lens_sha256`` is the digest of the lens artifact the anchors were written beside, so a gate can
    only ever be read with that exact lens; ``weights_fingerprint`` is ``FullWeightsFacts.fingerprint``
    of the checkpoint whose forward produced the residuals, the same digest the capture ladder writes
    on every cell, so a reader can hold the gate to the activations it is about to decode;
    ``anchors_sha256`` and ``anchor_digest`` name the corpus file and the drawn anchor texts;
    ``anchor_rows`` says which ``(item_id, position)`` each anchor row is. ``skip_first`` is the
    fit's leading-sink exclusion the positions respect, ``layer_convention`` is post-block, and
    ``load_path`` names the loader whose module tree produced the residuals.
    """

    lens_name: str
    model_label: str
    weights_identity: str
    weights_fingerprint: str
    lens_sha256: str
    anchors_path: str
    anchors_sha256: str
    anchor_digest: str
    anchor_rows: tuple[tuple[str, int], ...]
    skip_first: int
    jlens_commit: str
    load_path: str
    layer_convention: str = LAYER_CONVENTION_POST_BLOCK

    def as_payload(self) -> dict[str, object]:
        """Return the JSON block, rows as two-element lists."""
        payload = asdict(self)
        payload["anchor_rows"] = [list(row) for row in self.anchor_rows]
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, source: Path) -> GateProvenance:
        """Rebuild from the JSON block, refusing one that lacks a field."""
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        missing = sorted(fields - set(payload))
        if missing:
            raise GateProvenanceError(
                f"{source} carries no {missing}; a gate without full provenance cannot be held to a "
                f"lens or to weights, so it is not read"
            )
        values = {name: payload[name] for name in fields}
        values["anchor_rows"] = tuple(
            (str(item_id), int(position)) for item_id, position in payload["anchor_rows"]
        )
        return cls(**values)


def sha256_of_file(path: Path) -> str:
    """Hash a file's bytes, streaming, so a gigabyte lens never sits in memory twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def anchor_rows_from_activations(
    residuals_by_layer: Sequence[torch.Tensor],
    positions: Sequence[int],
    logits_at_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut anchor rows out of one prompt's forward: ``[n_positions, n_layers, hidden]`` and their targets.

    ``residuals_by_layer[l]`` is block ``l``'s output for the prompt, ``[seq, hidden]``, and
    ``logits_at_positions`` is the model's own ``[n_positions, vocab]`` logits at ``positions`` (the
    final residual there through the model's unembedding; only there, since the whole sequence's
    logits at this vocabulary would be gigabytes). The target at position ``p`` is the argmax of the
    model's logits there: the token the model itself would say next, which is what a lens at layer
    ``l`` has to recover from ``residuals_by_layer[l][p]``. Rows are float32 on the CPU, targets int64.
    """
    if not residuals_by_layer:
        raise ValueError("no layers to cut anchor rows from")
    seq_len = int(residuals_by_layer[0].shape[0])
    for layer, tensor in enumerate(residuals_by_layer):
        if tensor.ndim != 2 or int(tensor.shape[0]) != seq_len:  # noqa: PLR2004 - [seq, hidden]
            raise ValueError(
                f"layer {layer} residuals are {tuple(tensor.shape)}, expected [{seq_len}, hidden]"
            )
    if not positions:
        raise ValueError("no anchor positions")
    out_of_range = [position for position in positions if not 0 <= position < seq_len]
    if out_of_range:
        raise ValueError(f"anchor positions {out_of_range} lie outside a {seq_len}-token prompt")
    if logits_at_positions.ndim != 2 or int(logits_at_positions.shape[0]) != len(positions):  # noqa: PLR2004 - [n_positions, vocab]
        raise ValueError(
            f"logits are {tuple(logits_at_positions.shape)}, expected [{len(positions)}, vocab]"
        )
    index = torch.as_tensor(list(positions), dtype=torch.long)
    rows = torch.stack(
        [tensor.detach().float().cpu()[index] for tensor in residuals_by_layer], dim=1
    )
    targets = logits_at_positions.detach().float().cpu().argmax(dim=-1).to(torch.long)
    return rows, targets


@dataclass(frozen=True)
class StoredDecodeGate:
    """Anchor residuals at every layer with the model's own next token at each anchor.

    ``residuals`` is ``[n_anchors, n_layers, hidden]`` (post-block convention, one row per anchor
    position) and ``target_token_ids`` is ``[n_anchors]``: the argmax of the model's actual final
    logits at that position. The GPU stage that fits a lens records both; every later decode is
    admitted through :meth:`check`. ``provenance`` is what :meth:`save` writes beside the tensors
    and what the reader holds the gate to; a gate built in memory may carry none, a gate on disk
    always does.
    """

    residuals: torch.Tensor
    target_token_ids: torch.Tensor
    lens_name: str
    provenance: GateProvenance | None = None

    def __post_init__(self) -> None:
        """Reject shapes that would silently index the wrong anchor."""
        if self.residuals.ndim != 3:  # noqa: PLR2004 - [anchors, layers, hidden] by contract
            raise ValueError(
                f"residuals must be [anchors, layers, hidden], got {tuple(self.residuals.shape)}"
            )
        if self.target_token_ids.shape != (self.residuals.shape[0],):
            raise ValueError(
                f"{self.residuals.shape[0]} anchors but {tuple(self.target_token_ids.shape)} targets"
            )
        if self.residuals.shape[0] == 0:
            raise ValueError("a gate with no anchors admits nothing and refuses nothing")
        if self.provenance is not None:
            if self.provenance.lens_name != self.lens_name:
                raise GateProvenanceError(
                    f"gate is named {self.lens_name!r} but its provenance says "
                    f"{self.provenance.lens_name!r}"
                )
            if len(self.provenance.anchor_rows) != self.residuals.shape[0]:
                raise GateProvenanceError(
                    f"{self.residuals.shape[0]} anchor rows but the provenance names "
                    f"{len(self.provenance.anchor_rows)} (item, position) pairs"
                )

    @property
    def n_anchors(self) -> int:
        """How many anchor positions the gate holds."""
        return int(self.residuals.shape[0])

    @property
    def n_layers(self) -> int:
        """How many layers the anchors were captured at."""
        return int(self.residuals.shape[1])

    def hit_rate(
        self, lens: LensLike, model: UnembedModel, *, layer: int, claimed_layer: int, k: int
    ) -> float:
        """Return the fraction of anchors whose own token is in the top ``k``.

        Layer-``layer`` residuals are transported as if they sat at ``claimed_layer``; the two differ
        only in the far-wrong-layer arm of the gate.
        """
        hits = 0
        for anchor in range(self.n_anchors):
            readout = model.unembed(lens.transport(self.residuals[anchor, layer], claimed_layer))
            top = torch.topk(readout, min(k, int(readout.shape[0]))).indices
            hits += int(bool((top == self.target_token_ids[anchor]).any()))
        return hits / self.n_anchors

    def check(  # noqa: PLR0913 - a gate check is the lens pair, the layer and the three thresholds
        self,
        lens: LensLike,
        model: UnembedModel,
        layer: int,
        *,
        k: int = DEFAULT_GATE_TOP_K,
        min_hit_rate: float = DEFAULT_MIN_HIT_RATE,
        wrong_layer_offset: int = DEFAULT_WRONG_LAYER_OFFSET,
        min_margin: float = DEFAULT_MIN_MARGIN,
    ) -> GateResult:
        """Admit a decode at ``layer`` only if the anchors decode there and not at a far wrong layer."""
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is outside this {self.n_layers}-layer gate")
        wrong = wrong_layer_for(layer, n_layers=self.n_layers, offset=wrong_layer_offset)
        result = GateResult(
            lens_name=self.lens_name,
            layer=layer,
            wrong_layer=wrong,
            hit_rate=self.hit_rate(lens, model, layer=layer, claimed_layer=layer, k=k),
            wrong_layer_hit_rate=self.hit_rate(lens, model, layer=layer, claimed_layer=wrong, k=k),
            min_hit_rate=min_hit_rate,
            min_margin=min_margin,
            n_anchors=self.n_anchors,
            top_k=k,
        )
        logger.info(f"stored-decode gate {result.lens_name} layer={layer}: {result.reason}")
        return result

    def save(self, out_dir: Path) -> None:
        """Write the anchors (safetensors) and the provenance (json) beside each other.

        Refuses a gate with no provenance: on disk a gate outlives the process that knew which lens
        and weights it belonged to, and the reader's refusals rest on that record.
        """
        if self.provenance is None:
            raise GateProvenanceError(
                f"gate {self.lens_name!r} has no provenance; a gate written to disk has to name the "
                f"lens file, the served weights and the anchor corpus it was recorded on"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "residuals": self.residuals.contiguous(),
                "target_token_ids": self.target_token_ids.contiguous(),
            },
            str(out_dir / GATE_TENSORS_FILENAME),
        )
        (out_dir / GATE_META_FILENAME).write_text(
            json.dumps(
                {
                    "schema": GATE_SCHEMA,
                    "kind": GATE_KIND,
                    "n_anchors": self.n_anchors,
                    "n_layers": self.n_layers,
                    "hidden": int(self.residuals.shape[2]),
                    "residual_dtype": str(self.residuals.dtype).removeprefix("torch."),
                    "provenance": self.provenance.as_payload(),
                },
                indent=1,
            )
            + "\n"
        )
        logger.info(
            f"stored-decode gate written, {out_dir=} anchors={self.n_anchors} layers={self.n_layers} "
            f"weights_fingerprint={self.provenance.weights_fingerprint[:12]}"
        )

    @classmethod
    def load(cls, out_dir: Path) -> StoredDecodeGate:
        """Read a gate back, holding the tensors to what the sidecar says they are."""
        meta_path = out_dir / GATE_META_FILENAME
        meta = cast("dict[str, Any]", json.loads(meta_path.read_text()))
        if meta.get("kind") != GATE_KIND or meta.get("schema") != GATE_SCHEMA:
            raise GateProvenanceError(
                f"{meta_path} is not a schema-{GATE_SCHEMA} {GATE_KIND} sidecar "
                f"(kind={meta.get('kind')!r} schema={meta.get('schema')!r})"
            )
        provenance = GateProvenance.from_payload(
            cast("Mapping[str, Any]", meta["provenance"]), source=meta_path
        )
        tensors = load_file(str(out_dir / GATE_TENSORS_FILENAME))
        gate = cls(
            residuals=tensors["residuals"],
            target_token_ids=tensors["target_token_ids"],
            lens_name=provenance.lens_name,
            provenance=provenance,
        )
        declared = (int(meta["n_anchors"]), int(meta["n_layers"]), int(meta["hidden"]))
        if tuple(gate.residuals.shape) != declared:
            raise GateProvenanceError(
                f"{out_dir}: the anchors on disk are {tuple(gate.residuals.shape)} but the sidecar "
                f"declares {declared}; the two files were not written together"
            )
        return gate


def assert_gate_matches_lens_file(gate: StoredDecodeGate, lens_path: Path) -> str:
    """Refuse a gate whose anchors were recorded beside a different lens artifact; return the digest."""
    if gate.provenance is None:
        raise GateProvenanceError(
            f"gate {gate.lens_name!r} carries no provenance to hold to a lens"
        )
    actual = sha256_of_file(lens_path)
    if actual != gate.provenance.lens_sha256:
        raise GateProvenanceError(
            f"gate {gate.lens_name!r} was recorded beside lens {gate.provenance.lens_sha256[:12]} "
            f"but {lens_path} digests to {actual[:12]}; anchors from one fit cannot gate another "
            f"fit's decodes"
        )
    return actual


def assert_gate_matches_weights(
    gate: StoredDecodeGate, weights_fingerprint: str, *, what: str
) -> None:
    """Refuse a gate recorded on weights other than ``what``'s (a lens, a captured cell), by name."""
    if gate.provenance is None:
        raise GateProvenanceError(
            f"gate {gate.lens_name!r} carries no provenance to hold to weights"
        )
    if gate.provenance.weights_fingerprint != weights_fingerprint:
        raise GateProvenanceError(
            f"gate {gate.lens_name!r} was recorded on weights "
            f"{gate.provenance.weights_fingerprint[:12]} ({gate.provenance.model_label}) but {what} "
            f"serves weights {weights_fingerprint[:12]}; anchors from one checkpoint cannot gate "
            f"residuals from another"
        )


def gate_payload(result: GateResult) -> dict[str, object]:
    """Return the gate result as a JSON-safe block, with the derived verdict spelled out."""
    return {**asdict(result), "admitted": result.admitted, "reason": result.reason}
