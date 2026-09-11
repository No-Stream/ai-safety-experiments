"""What a TMAX lens fit leaves on disk beside ``lens.pt``, and how a CPU-side read opens all of it.

The fit driver (``reward_hacking.interp.tmax_lens_fit``) writes one directory per lens. Besides the
``jlens`` lens file it holds the stored-decode gate's anchors (``lens_geometry.StoredDecodeGate``),
the model's unembedding, and a provenance sidecar written LAST, whose presence is what marks the
directory complete. Everything the reads stage needs is in that directory, so a read runs on a CPU
box with neither the model nor ``jlens`` importable:

* :class:`SavedLens` reads the ``jlens`` lens file and transports the way
  ``jlens.lens.JacobianLens.transport`` does at the pinned commit, ``residual @ J_l.T`` in float32.
* :class:`StoredUnembed` is the final RMSNorm weight and the ``lm_head`` as served, applied the way
  ``jlens.hf.HFLensModel.unembed`` applies them: the residual cast to the served dtype, normalised in
  float32 with this family's ``(1 + w)`` scale, cast back, then the head. A hand-rolled unembed that
  forgets the ``1 +`` decodes plausibly and wrongly (``RESULTS.md``), so the fit driver checks the
  stored copy against the live model's logits before writing the provenance.
* :class:`LensProvenance` names the lens file by digest, the served weights by fingerprint, the
  corpus by its digests and the fit by its knobs.
* :func:`load_lens_bundle` opens a directory and holds the gate to the lens file it sits beside and
  to the weights the lens was fitted on, refusing a mismatch by name.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors.torch import load_file, save_file

from reward_hacking.interp.lens_geometry import (
    GateProvenanceError,
    StoredDecodeGate,
    assert_gate_matches_lens_file,
    assert_gate_matches_weights,
    sha256_of_file,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

LENS_FILENAME = "lens.pt"
FIT_CHECKPOINT_FILENAME = "fit-checkpoint.pt"
UNEMBED_TENSORS_FILENAME = "unembed.safetensors"
UNEMBED_META_FILENAME = "unembed.json"
PROVENANCE_FILENAME = "lens-provenance.json"

LENS_SCHEMA = 1
LENS_KIND = "tmax-lens"
UNEMBED_SCHEMA = 1
UNEMBED_KIND = "stored-unembed"
NORM_CONVENTION_ONE_PLUS_WEIGHT = "rmsnorm(x) * (1 + w)"
"""Qwen3.5's final norm: the weight is a delta from one, initialised at zero, not a gain."""

LENS_FILE_KEYS: tuple[str, ...] = ("J", "n_prompts", "source_layers", "d_model")
"""What ``jlens.lens.JacobianLens.save`` writes at the pinned commit."""


class LensArtifactError(ValueError):
    """A lens directory that is incomplete, torn, or not the one asked for."""


# --------------------------------------------------------------------------------------
# The lens file, read without jlens
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedLens:
    """A ``jlens`` lens file opened on the CPU: per-layer ``J_l`` in float32 and the transport."""

    jacobians: dict[int, torch.Tensor]
    n_prompts: int
    d_model: int

    @property
    def source_layers(self) -> list[int]:
        """The fitted source layers, ascending."""
        return sorted(self.jacobians)

    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        """``J_l @ h`` for a ``[..., d_model]`` residual, as ``jlens.lens.JacobianLens.transport``."""
        if layer not in self.jacobians:
            raise KeyError(
                f"layer {layer} is not a fitted source layer of this lens "
                f"({self.source_layers[0]}..{self.source_layers[-1]})"
            )
        return direction.float() @ self.jacobians[layer].T

    @classmethod
    def load(cls, path: Path) -> SavedLens:
        """Open a lens file, refusing anything but the pinned ``jlens`` format."""
        checkpoint = cast("dict[str, Any]", torch.load(path, map_location="cpu", weights_only=True))
        missing = [key for key in LENS_FILE_KEYS if key not in checkpoint]
        if missing:
            raise LensArtifactError(
                f"{path} lacks {missing}, so it is not a jlens lens file (found {sorted(checkpoint)})"
            )
        jacobians = {
            int(layer): tensor.float()
            for layer, tensor in cast("dict[int, torch.Tensor]", checkpoint["J"]).items()
        }
        if sorted(jacobians) != [int(layer) for layer in checkpoint["source_layers"]]:
            raise LensArtifactError(
                f"{path}: the J block covers layers {sorted(jacobians)} but source_layers says "
                f"{checkpoint['source_layers']}"
            )
        d_model = int(checkpoint["d_model"])
        for layer, tensor in jacobians.items():
            if tuple(tensor.shape) != (d_model, d_model):
                raise LensArtifactError(
                    f"{path}: J at layer {layer} is {tuple(tensor.shape)}, not [{d_model}, {d_model}]"
                )
        return cls(jacobians=jacobians, n_prompts=int(checkpoint["n_prompts"]), d_model=d_model)


# --------------------------------------------------------------------------------------
# The unembedding, stored and applied on the CPU
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredUnembed:
    """The served model's final norm and head, applied as ``HFLensModel.unembed`` applies them."""

    norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    rms_eps: float

    def __post_init__(self) -> None:
        """Hold the two tensors to each other."""
        if self.norm_weight.ndim != 1 or self.lm_head_weight.ndim != 2:  # noqa: PLR2004 - [hidden], [vocab, hidden]
            raise LensArtifactError(
                f"norm weight {tuple(self.norm_weight.shape)} and head "
                f"{tuple(self.lm_head_weight.shape)} are not [hidden] and [vocab, hidden]"
            )
        if self.norm_weight.shape[0] != self.lm_head_weight.shape[1]:
            raise LensArtifactError(
                f"norm weight is {int(self.norm_weight.shape[0])} wide but the head takes "
                f"{int(self.lm_head_weight.shape[1])}"
            )
        if self.norm_weight.dtype != self.lm_head_weight.dtype:
            raise LensArtifactError(
                f"norm weight is {self.norm_weight.dtype} but the head is {self.lm_head_weight.dtype}"
            )

    @property
    def served_dtype(self) -> torch.dtype:
        """The dtype the model ran in, which the residual is cast to before the norm."""
        return self.lm_head_weight.dtype

    @property
    def vocab_size(self) -> int:
        """How many logits a decode returns."""
        return int(self.lm_head_weight.shape[0])

    def unembed(self, transported: torch.Tensor) -> torch.Tensor:
        """``lm_head(norm(residual))`` with the served model's dtype path, on whatever device holds the weights."""
        x = transported.to(self.lm_head_weight.device).to(self.served_dtype)
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.rms_eps)
        normed = (normed * (1.0 + self.norm_weight.float())).to(self.served_dtype)
        return normed @ self.lm_head_weight.T

    def save(self, out_dir: Path, *, weights_fingerprint: str) -> None:
        """Write the two tensors and the facts a reader needs to apply them the same way."""
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "norm_weight": self.norm_weight.detach().cpu().contiguous(),
                "lm_head_weight": self.lm_head_weight.detach().cpu().contiguous(),
            },
            str(out_dir / UNEMBED_TENSORS_FILENAME),
        )
        (out_dir / UNEMBED_META_FILENAME).write_text(
            json.dumps(
                {
                    "schema": UNEMBED_SCHEMA,
                    "kind": UNEMBED_KIND,
                    "rms_eps": self.rms_eps,
                    "norm_convention": NORM_CONVENTION_ONE_PLUS_WEIGHT,
                    "served_dtype": str(self.served_dtype).removeprefix("torch."),
                    "vocab_size": self.vocab_size,
                    "hidden": int(self.norm_weight.shape[0]),
                    "weights_fingerprint": weights_fingerprint,
                },
                indent=1,
            )
            + "\n"
        )

    @classmethod
    def load(cls, out_dir: Path) -> tuple[StoredUnembed, str]:
        """Read the unembedding back with the fingerprint of the weights it came from."""
        meta_path = out_dir / UNEMBED_META_FILENAME
        meta = cast("dict[str, Any]", json.loads(meta_path.read_text()))
        if meta.get("kind") != UNEMBED_KIND or meta.get("schema") != UNEMBED_SCHEMA:
            raise LensArtifactError(f"{meta_path} is not a schema-{UNEMBED_SCHEMA} {UNEMBED_KIND}")
        if meta.get("norm_convention") != NORM_CONVENTION_ONE_PLUS_WEIGHT:
            raise LensArtifactError(
                f"{meta_path} records norm convention {meta.get('norm_convention')!r}; this reader "
                f"applies {NORM_CONVENTION_ONE_PLUS_WEIGHT!r} and would decode another convention "
                f"plausibly and wrongly"
            )
        tensors = load_file(str(out_dir / UNEMBED_TENSORS_FILENAME))
        unembed = cls(
            norm_weight=tensors["norm_weight"],
            lm_head_weight=tensors["lm_head_weight"],
            rms_eps=float(meta["rms_eps"]),
        )
        declared = (int(meta["vocab_size"]), int(meta["hidden"]))
        if tuple(unembed.lm_head_weight.shape) != declared:
            raise LensArtifactError(
                f"{out_dir}: the head on disk is {tuple(unembed.lm_head_weight.shape)} but the "
                f"sidecar declares {declared}"
            )
        return unembed, str(meta["weights_fingerprint"])


def unembed_agreement(
    stored: StoredUnembed, model_logits: torch.Tensor, residuals: torch.Tensor
) -> dict[str, float | int | bool]:
    """Compare the stored unembedding with the live model's logits on the same final residuals.

    ``residuals`` is ``[n, hidden]`` at the final layer and ``model_logits`` the model's own
    ``[n, vocab]`` there. Both argmax agreement and the largest logit difference are reported; the fit
    driver refuses to finish on an argmax disagreement, since that is the token the gate keys on.
    """
    stored_logits = stored.unembed(residuals).float().cpu()
    model_logits = model_logits.float().cpu()
    if stored_logits.shape != model_logits.shape:
        raise LensArtifactError(
            f"stored unembed gives {tuple(stored_logits.shape)} logits, the model "
            f"{tuple(model_logits.shape)}"
        )
    agree = int((stored_logits.argmax(-1) == model_logits.argmax(-1)).sum())
    return {
        "n_rows": int(model_logits.shape[0]),
        "argmax_agreement": agree,
        "argmax_all_agree": agree == int(model_logits.shape[0]),
        "max_abs_logit_diff": float((stored_logits - model_logits).abs().max()),
    }


# --------------------------------------------------------------------------------------
# The provenance sidecar, written last
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LensProvenance:
    """Everything that decided one lens, written beside it after every other file is complete.

    ``identity`` is the part a relaunch compares to decide whether the directory already holds the
    lens it is about to fit: the served weights, the corpus digests and role, the fit knobs and the
    jlens commit. ``report`` is the rest: timings, the reconstruction read, the gate's self-check,
    the unembed agreement, the kernel binding, the invocation.
    """

    lens_name: str
    model_label: str
    weights_identity: str
    weights_fingerprint: str
    load_path: str
    tokenizer: dict[str, str | None]
    corpus_dir: str
    corpus_digests: dict[str, str]
    fit_role: str
    n_fit_prompts: int
    n_prompts_fitted: int
    dim_batch: int
    max_seq_len: int
    skip_first: int
    checkpoint_every: int | None
    jlens_commit: str
    lens_sha256: str
    report: dict[str, Any]

    @property
    def identity(self) -> dict[str, object]:
        """The fields a relaunch holds an existing directory to."""
        return {
            "lens_name": self.lens_name,
            "model_label": self.model_label,
            "weights_identity": self.weights_identity,
            "weights_fingerprint": self.weights_fingerprint,
            "load_path": self.load_path,
            "corpus_digests": dict(self.corpus_digests),
            "fit_role": self.fit_role,
            "n_fit_prompts": self.n_fit_prompts,
            "dim_batch": self.dim_batch,
            "max_seq_len": self.max_seq_len,
            "skip_first": self.skip_first,
            "jlens_commit": self.jlens_commit,
        }

    def as_payload(self) -> dict[str, object]:
        """Return the JSON sidecar."""
        return {"schema": LENS_SCHEMA, "kind": LENS_KIND, **asdict(self)}

    def save(self, out_dir: Path) -> Path:
        """Write the sidecar; the directory is complete once this file exists."""
        path = out_dir / PROVENANCE_FILENAME
        staging = path.with_name(path.name + ".tmp")
        staging.write_text(json.dumps(self.as_payload(), indent=1, default=str) + "\n")
        staging.replace(path)
        return path

    @classmethod
    def load(cls, out_dir: Path) -> LensProvenance:
        """Read the sidecar back, refusing another kind or schema."""
        path = out_dir / PROVENANCE_FILENAME
        payload = cast("dict[str, Any]", json.loads(path.read_text()))
        if payload.get("kind") != LENS_KIND or payload.get("schema") != LENS_SCHEMA:
            raise LensArtifactError(
                f"{path} is not a schema-{LENS_SCHEMA} {LENS_KIND} sidecar "
                f"(kind={payload.get('kind')!r} schema={payload.get('schema')!r})"
            )
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        missing = sorted(fields - set(payload))
        if missing:
            raise LensArtifactError(f"{path} lacks {missing}")
        return cls(**{name: payload[name] for name in fields})


def assert_same_fit(
    existing: LensProvenance, wanted: Mapping[str, object], *, out_dir: Path
) -> None:
    """Refuse to treat a directory as the fit asked for when any identity field differs, by name."""
    differing = {
        key: (existing.identity[key], value)
        for key, value in wanted.items()
        if existing.identity.get(key) != value
    }
    if differing:
        raise LensArtifactError(
            f"{out_dir} already holds a complete lens fitted differently: "
            + "; ".join(
                f"{key}: on disk {have!r}, asked {want!r}"
                for key, (have, want) in differing.items()
            )
            + ". A lens directory is the identity of one fit; write a new directory."
        )


# --------------------------------------------------------------------------------------
# Opening a complete directory for a read
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LensBundle:
    """A complete lens directory opened for reading: lens, gate, unembedding and provenance."""

    lens: SavedLens
    gate: StoredDecodeGate
    unembed: StoredUnembed
    provenance: LensProvenance
    lens_dir: Path

    @property
    def lens_name(self) -> str:
        """What the decode tables call this lens."""
        return self.provenance.lens_name

    @property
    def weights_fingerprint(self) -> str:
        """The served weights every file in the bundle was recorded on."""
        return self.provenance.weights_fingerprint


def load_lens_bundle(lens_dir: Path) -> LensBundle:
    """Open a lens directory and hold every file in it to the others.

    The provenance is read first, since without it the directory is a torn fit; the gate is held to
    the lens file's digest and to the provenance's weights fingerprint; the unembedding is held to
    the same fingerprint. A mismatch anywhere is a refusal naming both sides, never a decode.
    """
    if not (lens_dir / PROVENANCE_FILENAME).is_file():
        raise LensArtifactError(
            f"{lens_dir} has no {PROVENANCE_FILENAME}, so the fit that wrote it did not finish; a "
            f"torn lens directory is not read"
        )
    provenance = LensProvenance.load(lens_dir)
    lens_path = lens_dir / LENS_FILENAME
    lens_sha256 = sha256_of_file(lens_path)
    if lens_sha256 != provenance.lens_sha256:
        raise LensArtifactError(
            f"{lens_path} digests to {lens_sha256[:12]} but the provenance names "
            f"{provenance.lens_sha256[:12]}; the lens file is not the one the sidecar describes"
        )
    lens = SavedLens.load(lens_path)
    gate = StoredDecodeGate.load(lens_dir)
    assert_gate_matches_lens_file(gate, lens_path)
    assert_gate_matches_weights(
        gate, provenance.weights_fingerprint, what=f"lens {provenance.lens_name}"
    )
    unembed, unembed_fingerprint = StoredUnembed.load(lens_dir)
    if unembed_fingerprint != provenance.weights_fingerprint:
        raise GateProvenanceError(
            f"{lens_dir}: the stored unembedding came from weights {unembed_fingerprint[:12]} but the "
            f"lens was fitted on {provenance.weights_fingerprint[:12]}"
        )
    if (
        int(unembed.norm_weight.shape[0]) != lens.d_model
        or int(gate.residuals.shape[2]) != lens.d_model
    ):
        raise LensArtifactError(
            f"{lens_dir}: widths disagree (lens {lens.d_model}, unembed "
            f"{int(unembed.norm_weight.shape[0])}, anchors {int(gate.residuals.shape[2])})"
        )
    logger.info(
        f"lens bundle opened, {lens_dir=} lens={provenance.lens_name} "
        f"weights={provenance.weights_fingerprint[:12]} layers={len(lens.source_layers)} "
        f"anchors={gate.n_anchors}"
    )
    return LensBundle(
        lens=lens, gate=gate, unembed=unembed, provenance=provenance, lens_dir=lens_dir
    )
