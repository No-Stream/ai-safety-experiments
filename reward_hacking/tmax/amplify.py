r"""Task-arithmetic amplification of a released RL'd checkpoint: ``W = W_base + alpha (W_rl - W_base)``.

The behavioural screen asks whether TMAX's RL taught a hacking disposition that its released
checkpoints express only weakly. Scaling the training delta is the readout for a disposition below
the behavioural threshold: alpha 1 is the released checkpoint itself, alpha 0 the base, and if the
hack rate rises monotonically with alpha past 1 there is a direction in weight space that RL moved
along and that the screen can see. A unit built here is served exactly like a hub checkpoint --
through :func:`games.eval_model.resolve_served_model`'s full-weights rung -- so nothing downstream
knows it was assembled.

**Which tensors, and in whose dtype.** The RL'd release carries the language model only (426
tensors under ``model.language_model.`` at 4B and 9B; the vision tower and the MTP head were
stripped before publication), so the output carries exactly the RL'd checkpoint's tensor set and
every one of them must exist in the base at the same shape -- a missing or reshaped tensor is
refused, never skipped, because an amplification that silently left a block at base would read as
a weaker disposition. Arithmetic runs in float32 and each result is cast to the RL'd checkpoint's
own dtype for that tensor: the release stores the Gated DeltaNet ``A_log`` and per-layer norm
weights at bfloat16 where the base keeps them at float32, and the unit should load through the same
path the release does, so it copies the release's storage rather than the base's.

**What the directory carries.** The tensors as one ``model.safetensors``; the RL'd release's own
``config.json``, ``generation_config.json`` and tokenizer files, so the unit renders under the same
template and terminators as the checkpoint it amplifies; and a
:data:`~games.eval_model.WEIGHTS_PROVENANCE_FILENAME` sidecar naming both inputs (label, directory,
resolved commit where known), the alpha, the tensor count, the per-tensor delta statistics, and the
digest of the tensors just written -- which :func:`games.eval_model.resolve_full_weights` re-hashes
and holds the directory to before any engine loads it.

**The control that can fail: a permuted copy of the real delta.** A rising hack rate along the
alpha ladder is only evidence about the *direction* RL moved along if a perturbation of the same
size in some other direction does not do the same thing. The obvious control, a Gaussian tensor at
the same per-tensor Frobenius norm, is inert by construction: its operator norm is about
``2 / sqrt(hidden)`` of that norm, so at hidden 4,096 it moves the function tens of times less than
a structured delta of equal norm, and "random stays at noise" would be guaranteed rather than
observed. :data:`DELTA_TRANSFORM_PERMUTE_AXES` builds the control from the real delta instead:
every delta tensor is permuted independently along each of its axes with a permutation seeded by
``(seed, tensor name, axis)``. Row and column permutations are orthogonal, so every singular value
and the Frobenius norm of each 2-D delta survive exactly (a 1-D or 3-D delta keeps its multiset of
entries and so its norm), while the alignment between the delta's coordinates and the base's is
destroyed. Served through the same path at the same alpha, it is a perturbation matched on
magnitude and spectrum whose only difference from the real amplification is the alignment -- so a
control unit that hacks like the real one says the readout is measuring damage, not disposition.
The sidecar records the transform, the seed and, per tensor, the cosine between the real delta and
the transformed one (1.0 under the identity), so a reader can see how far the alignment moved.

**Storage dtype, and why bf16 cannot carry a small alpha step.** The release stores bf16, whose 8
mantissa bits make one unit in the last place about 0.4% of a weight, while RL moved the weights by
~0.15% on average: measured on the real ``allenai/tmax-9b@step_500`` delta (2026-09-04), 40-50% of
the entries of a weight matrix did not move at all and 58-60% of the ones that did moved by exactly
one bf16 ulp. ``base + alpha x delta`` rounded back to bf16 therefore snaps a one-ulp step to one
ulp for any alpha in (0.5, 1.5): the realized delta norm was 1.015x, 1.06x and 1.17x the release's
at nominal alpha 1.1, 1.2 and 1.3, and the screen's alpha-1.5 unit realized 1.55x. A bf16 ladder
between 1 and 1.3 is close to a no-op. ``--storage-dtype float16`` stores the result with three
more mantissa bits (every bf16 weight of these checkpoints is exactly representable in fp16, which
the same measurement confirmed at zero rounding noise), realizing 1.13x, 1.23x and 1.28x at the same
nominal alphas; ``float32`` realizes exactly and doubles the footprint. The unit's ``config.json``
dtype keys are rewritten to the storage dtype so the engine serves it as stored, and every unit's
sidecar records ``storage_dtype`` and the per-tensor ``realized_alpha_ratio`` (the stored delta norm
over the intended one), so a ladder that silently realized nothing is visible in its own provenance
rather than in a flat readout. fp16's narrower exponent range is the risk the storage buys: the
weights themselves are far inside it, and activations are the engine's, which is what the kit's
fidelity smoke on the 0.8B tier checks before a box is rented.

**Tensors copied from the release rather than amplified.** Forty-eight tensors of the 9B (the Gated
DeltaNet ``A_log`` and ``linear_attn.norm.weight`` of every layer) are float32 in the base and bf16
in the release, so ``W_rl - W_base`` for them is the release's cast-to-bf16 rounding error, not
anything RL did, and scaling it by alpha would scale noise. Any tensor whose base dtype is float32
and whose release dtype is bfloat16 is therefore copied from the release unchanged (dtype included),
under every alpha, transform and storage dtype; the sidecar lists them as ``copied_from_release``
and their delta statistics stay recorded so the size of the cast error is on the record.

**Memory.** Tensors are read lazily through ``safe_open`` and the finished set is held once at the
storage dtype (about 8.4 GB at 4B, 18 GB at 9B in bf16 or fp16) before one ``save_file``; peak host
memory is that plus one tensor in float32. A g7e host has room; the local dev box has far more.

    uv run python -m reward_hacking.tmax.amplify --base Qwen/Qwen3.5-4B --base-revision main \\
        --rl allenai/tmax-4b --rl-revision step_380 --alpha 1.5 \\
        --out /var/tmp/amplified/tmax-4b-step380-alpha1.5
    uv run python -m reward_hacking.tmax.amplify --base Qwen/Qwen3.5-9B --base-revision main \\
        --rl allenai/tmax-9b --rl-revision step_500 --alpha 1.2 \\
        --delta-transform permute-axes --permutation-seed 20260903 \\
        --out /var/tmp/amplified/tmax-9b-step500-permuted-alpha1.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from games.eval_model import (
    SAFETENSORS_SUFFIX,
    WEIGHTS_PROVENANCE_FILENAME,
    FullWeightsFacts,
    FullWeightsSource,
    resolve_full_weights,
    sha256_of_file,
)
from reward_hacking.tmax.artifacts import DROPPED_TOWER_KEY_PREFIXES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

OUTPUT_WEIGHTS_FILENAME = "model.safetensors"
SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"

NON_LANGUAGE_MODEL_PREFIXES: tuple[str, ...] = DROPPED_TOWER_KEY_PREFIXES
"""The tensor families the upstream Qwen3.5 checkpoints carry and the RL'd releases do not.

The vision tower and the multi-token-prediction head, as the registry spells them. Everything else
in the base -- the language model under ``model.language_model.`` and, on untied checkpoints,
``lm_head.weight`` -- must appear in the release name for name, or the release is not a full
language model of this base and the amplification would be defined on a subset nobody chose.
"""

COPY_FROM_RELEASE_DTYPES: tuple[torch.dtype, torch.dtype] = (torch.float32, torch.bfloat16)
"""(base dtype, release dtype) of a tensor whose whole delta is the release's cast error: copied, not
amplified (module docstring)."""

COPIED_SIDE_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
"""Everything a served directory needs besides tensors, copied from the RL'd release when present.

``config.json`` is required (a directory without it is not a checkpoint); the rest travel when the
release ships them, so the unit renders under the release's own tokenizer and chat template.
"""

PROVENANCE_SCHEMA = 3
"""Bumped from 1 when the sidecar gained ``delta_transform``, ``permutation_seed`` and the cosines;
from 2 when it gained ``storage_dtype``, ``config_dtype_rewritten`` and the realized-alpha ratios."""

STORAGE_DTYPE_RELEASE = "release"
STORAGE_DTYPES: tuple[str, ...] = (STORAGE_DTYPE_RELEASE, "float16", "float32")
"""How the amplified tensors are stored: in each tensor's release dtype, or all in one named dtype."""
_TORCH_DTYPE_BY_NAME: dict[str, torch.dtype] = {"float16": torch.float16, "float32": torch.float32}
CONFIG_DTYPE_KEYS: tuple[str, ...] = ("dtype", "torch_dtype")
"""The config.json keys the engine reads its serving dtype from (transformers spells both)."""

DELTA_TRANSFORM_IDENTITY = "identity"
DELTA_TRANSFORM_PERMUTE_AXES = "permute-axes"
DELTA_TRANSFORMS: tuple[str, ...] = (DELTA_TRANSFORM_IDENTITY, DELTA_TRANSFORM_PERMUTE_AXES)
"""The real delta, or the control built from it by permuting every delta tensor along every axis."""


def _axis_permutation_seed(seed: int, tensor_name: str, axis: int) -> int:
    """Derive one axis's permutation seed from the run seed and the tensor's identity.

    From identity rather than from position in the tensor loop, so the same unit rebuilt with the
    same seed is the same unit tensor for tensor whatever order the shards are read in, and two
    tensors of one build never share a permutation by accident.
    """
    digest = hashlib.blake2b(f"{seed}\0{tensor_name}\0{axis}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def permute_delta_axes(delta: torch.Tensor, *, tensor_name: str, seed: int) -> torch.Tensor:
    """Permute ``delta`` independently along each of its axes, seeded by ``(seed, name, axis)``.

    On a 2-D tensor this is ``P delta Q`` for permutation matrices ``P`` and ``Q``, which are
    orthogonal, so every singular value and the Frobenius norm are preserved exactly while the
    alignment with the base's row and column coordinates is destroyed. A 1-D tensor is permuted
    along its one axis and an N-D tensor along each, which keeps the multiset of entries and so the
    norm. An axis of length one has nothing to permute and is left alone.
    """
    permuted = delta
    for axis, size in enumerate(delta.shape):
        if size == 1:
            continue
        generator = torch.Generator().manual_seed(_axis_permutation_seed(seed, tensor_name, axis))
        permuted = permuted.index_select(axis, torch.randperm(size, generator=generator))
    return permuted


@dataclass(frozen=True)
class DeltaTransform:
    """What is done to the training delta before alpha scales it: nothing, or the permuted control."""

    name: str = DELTA_TRANSFORM_IDENTITY
    seed: int | None = None

    def __post_init__(self) -> None:
        """Refuse an unknown transform, a permutation without a seed, or a seed with nothing to seed."""
        if self.name not in DELTA_TRANSFORMS:
            raise ValueError(f"unknown delta transform {self.name!r}; known: {DELTA_TRANSFORMS}")
        if (self.name == DELTA_TRANSFORM_PERMUTE_AXES) != (self.seed is not None):
            raise ValueError(
                f"delta transform {self.name!r} with permutation seed {self.seed!r}: the permuted "
                f"control needs a seed on the record, and the identity has nothing to seed"
            )

    @property
    def is_identity(self) -> bool:
        """Whether this unit scales the real delta rather than the permuted control."""
        return self.name == DELTA_TRANSFORM_IDENTITY

    def apply(self, delta: torch.Tensor, *, tensor_name: str) -> torch.Tensor:
        """Return the delta this unit actually scales."""
        if self.is_identity:
            return delta
        return permute_delta_axes(delta, tensor_name=tensor_name, seed=cast("int", self.seed))


IDENTITY_DELTA_TRANSFORM = DeltaTransform()
"""The real delta, unchanged: what every amplification scaled before the control existed."""


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine between two tensors read as vectors; 1.0 when both are zero (nothing moved)."""
    norms = float(a.norm()) * float(b.norm())
    if norms == 0.0:
        return 1.0
    return float(torch.dot(a.flatten(), b.flatten())) / norms


@dataclass(frozen=True)
class TensorDelta:
    """How far one tensor moved from base to RL'd, in the norms a reader can compare across layers.

    ``cosine_to_real_delta`` is between the RL'd delta and the delta this unit actually scaled: 1.0
    under the identity transform, and near zero for a permuted control whose alignment is gone.
    ``realized_alpha_ratio`` is the norm of the delta the STORED tensor actually carries over the norm
    of the intended ``alpha x T(delta)``: 1.0 when storage rounding lost nothing, and the number that
    exposes a bf16 ladder snapping every small step back to the release (module docstring).
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    base_norm: float
    delta_norm: float
    cosine_to_real_delta: float = 1.0
    realized_alpha_ratio: float = 1.0
    copied_from_release: bool = False

    @property
    def relative_delta(self) -> float:
        """``||W_rl - W_base|| / ||W_base||``, zero for a tensor whose base norm is zero."""
        return self.delta_norm / self.base_norm if self.base_norm else 0.0


@dataclass(frozen=True)
class AmplifyReport:
    """What one amplification did, as written into the sidecar beside the tensors."""

    label: str
    alpha: float
    n_tensors: int
    weights_sha256: dict[str, str]
    base: dict[str, object]
    rl: dict[str, object]
    max_relative_delta: float
    mean_relative_delta: float
    dtypes: dict[str, int]
    delta_transform: str = DELTA_TRANSFORM_IDENTITY
    permutation_seed: int | None = None
    mean_abs_cosine_to_real_delta: float = 1.0
    storage_dtype: str = STORAGE_DTYPE_RELEASE
    config_dtype_rewritten: bool = False
    mean_realized_alpha_ratio: float = 1.0
    n_copied_from_release: int = 0

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the sidecar body, schema-versioned so a later reader knows which fields to expect."""
        return {
            "schema": PROVENANCE_SCHEMA,
            "kind": "task-arithmetic-amplification",
            **asdict(self),
        }


def _tensor_locations(checkpoint_dir: Path) -> dict[str, Path]:
    """Map every tensor name in a checkpoint directory to the shard file that holds it.

    A sharded checkpoint carries an index naming the shard per tensor; a single-file one does not,
    so the header of every ``*.safetensors`` present is read instead. Either way the result is
    exhaustive, which is what lets a missing tensor be refused rather than overlooked.
    """
    index_path = checkpoint_dir / SAFETENSORS_INDEX_FILENAME
    if index_path.is_file():
        index = cast("dict[str, object]", json.loads(index_path.read_text(encoding="utf-8")))
        weight_map = cast("Mapping[str, str]", index["weight_map"])
        return {name: checkpoint_dir / shard for name, shard in weight_map.items()}
    locations: dict[str, Path] = {}
    shards = sorted(path for path in checkpoint_dir.iterdir() if path.suffix == SAFETENSORS_SUFFIX)
    if not shards:
        raise FileNotFoundError(f"{checkpoint_dir} holds no {SAFETENSORS_SUFFIX} file")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open handles are not mappings
                if name in locations:
                    raise ValueError(f"tensor {name!r} appears in two shards of {checkpoint_dir}")
                locations[name] = shard
    return locations


def _open_shards(paths: Iterable[Path]) -> dict[Path, Any]:
    """One lazy safetensors handle per shard; ``Any`` because safe_open's handle type is private."""
    return {path: safe_open(str(path), framework="pt") for path in sorted(set(paths))}


@dataclass(frozen=True)
class AlignedTensorSets:
    """The RL'd checkpoint's tensor set, located in both checkpoints and proven to be the base's.

    Built by :func:`align_tensor_sets`, which is where the two refusals live: a release tensor the
    base lacks (this is not that release's base) and a base language-model tensor the release lacks
    (the release is not a complete language model of this base). ``names`` is the release's full
    set, sorted; :meth:`open` returns lazy per-shard handles so a caller streams tensor by tensor
    rather than holding either checkpoint whole.
    """

    base_dir: Path
    rl_dir: Path
    names: tuple[str, ...]
    base_locations: Mapping[str, Path]
    rl_locations: Mapping[str, Path]

    def open(self) -> AlignedTensorReader:
        """Open one lazy safetensors handle per shard on each side."""
        return AlignedTensorReader(
            sets=self,
            base_shards=_open_shards(self.base_locations[name] for name in self.names),
            rl_shards=_open_shards(self.rl_locations.values()),
        )


@dataclass(frozen=True)
class AlignedTensorReader:
    """Lazy reads of the same-named tensor from both checkpoints, shape-checked on every read.

    The shape check sits on the read rather than in :func:`align_tensor_sets` because it needs the
    header of every shard, and reading them all up front would cost as much as one full pass; a
    shape mismatch is instead refused the moment the tensor is actually asked for.
    """

    sets: AlignedTensorSets
    base_shards: Mapping[Path, Any]
    rl_shards: Mapping[Path, Any]

    def shape_of(self, name: str) -> tuple[int, ...]:
        """Return the release's shape for ``name``, read from the shard header without loading data."""
        handle = self.rl_shards[self.sets.rl_locations[name]]
        return tuple(handle.get_slice(name).get_shape())

    def pair(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(base, rl)`` for one tensor in their storage dtypes, refusing a shape mismatch."""
        rl_tensor = cast(
            "torch.Tensor", self.rl_shards[self.sets.rl_locations[name]].get_tensor(name)
        )
        base_tensor = cast(
            "torch.Tensor", self.base_shards[self.sets.base_locations[name]].get_tensor(name)
        )
        if rl_tensor.shape != base_tensor.shape:
            raise ValueError(
                f"tensor {name!r} is {tuple(rl_tensor.shape)} in {self.sets.rl_dir} but "
                f"{tuple(base_tensor.shape)} in {self.sets.base_dir}; these checkpoints do not "
                f"share an architecture"
            )
        return base_tensor, rl_tensor


def align_tensor_sets(*, base_dir: Path, rl_dir: Path) -> AlignedTensorSets:
    """Locate the release's tensors in both checkpoints and refuse any set-level mismatch.

    The RL'd checkpoint's tensor set is the contract: every name it carries must exist in the
    base, and every language-model tensor of the base must exist in the release (only the vision
    tower and the MTP head may be absent). A delta computed on anything less would be defined on a
    subset nobody chose, so both directions are refused rather than skipped.
    """
    rl_locations = _tensor_locations(rl_dir)
    base_locations = _tensor_locations(base_dir)
    missing = sorted(set(rl_locations) - set(base_locations))
    if missing:
        raise ValueError(
            f"{len(missing)} tensor(s) of {rl_dir} have no counterpart in {base_dir} (first: "
            f"{missing[0]!r}); the base is not this release's base, and amplifying against it "
            f"would move weights along a direction nobody trained"
        )
    base_language_model = {
        name for name in base_locations if not name.startswith(NON_LANGUAGE_MODEL_PREFIXES)
    }
    unreleased = sorted(base_language_model - set(rl_locations))
    if unreleased:
        raise ValueError(
            f"{len(unreleased)} language-model tensor(s) of {base_dir} are absent from {rl_dir} "
            f"(first: {unreleased[0]!r}); the release is not a complete language model of this base "
            f"(only {NON_LANGUAGE_MODEL_PREFIXES} may differ), so the delta would be defined on a "
            f"subset nobody chose"
        )
    return AlignedTensorSets(
        base_dir=base_dir,
        rl_dir=rl_dir,
        names=tuple(sorted(rl_locations)),
        base_locations=base_locations,
        rl_locations=rl_locations,
    )


def _storage_torch_dtype(storage_dtype: str, release_dtype: torch.dtype) -> torch.dtype:
    """Return the dtype one amplified tensor is stored in: the release's own, or the one named."""
    if storage_dtype not in STORAGE_DTYPES:
        raise ValueError(f"unknown storage dtype {storage_dtype!r}; known: {STORAGE_DTYPES}")
    if storage_dtype == STORAGE_DTYPE_RELEASE:
        return release_dtype
    return _TORCH_DTYPE_BY_NAME[storage_dtype]


def _realized_alpha_ratio(
    stored: torch.Tensor, base32: torch.Tensor, intended32: torch.Tensor
) -> float:
    """Return ``||stored - base|| / ||intended delta||``, or 1.0 when nothing was intended (alpha 0)."""
    intended_norm = float(intended32.norm())
    if intended_norm == 0.0:
        return 1.0
    return float((stored.to(torch.float32) - base32).norm()) / intended_norm


def amplify_tensors(
    *,
    base_dir: Path,
    rl_dir: Path,
    alpha: float,
    delta_transform: DeltaTransform = IDENTITY_DELTA_TRANSFORM,
    storage_dtype: str = STORAGE_DTYPE_RELEASE,
) -> tuple[dict[str, torch.Tensor], list[TensorDelta]]:
    """Compute ``W_base + alpha T(W_rl - W_base)`` for every tensor of the RL'd checkpoint.

    ``T`` is ``delta_transform``: the identity for a real amplification, or the seeded axis
    permutation that builds the matched-magnitude control (see the module docstring). The
    alignment (and its refusals) is :func:`align_tensor_sets`; nothing the base carries beyond
    the release's set (vision tower, MTP head) is emitted, because the release itself does not
    carry them and the unit is served through the same text-only path. Arithmetic is float32;
    storage follows ``storage_dtype`` (the RL'd tensor's own dtype by default), and every tensor
    records how much of the intended step survived that storage.
    """
    reader = align_tensor_sets(base_dir=base_dir, rl_dir=rl_dir).open()
    amplified: dict[str, torch.Tensor] = {}
    deltas: list[TensorDelta] = []
    for name in reader.sets.names:
        base_tensor, rl_tensor = reader.pair(name)
        base32 = base_tensor.to(torch.float32)
        delta32 = rl_tensor.to(torch.float32) - base32
        if (base_tensor.dtype, rl_tensor.dtype) == COPY_FROM_RELEASE_DTYPES:
            # The whole delta is the release's cast error: copy the release, scale nothing.
            amplified[name] = rl_tensor.contiguous()
            deltas.append(
                TensorDelta(
                    name=name,
                    dtype=str(rl_tensor.dtype).removeprefix("torch."),
                    shape=tuple(rl_tensor.shape),
                    base_norm=float(base32.norm()),
                    delta_norm=float(delta32.norm()),
                    copied_from_release=True,
                )
            )
            continue
        scaled32 = delta_transform.apply(delta32, tensor_name=name)
        intended32 = alpha * scaled32
        target = _storage_torch_dtype(storage_dtype, rl_tensor.dtype)
        stored = (base32 + intended32).to(target).contiguous()
        amplified[name] = stored
        deltas.append(
            TensorDelta(
                name=name,
                dtype=str(target).removeprefix("torch."),
                shape=tuple(rl_tensor.shape),
                base_norm=float(base32.norm()),
                delta_norm=float(delta32.norm()),
                cosine_to_real_delta=1.0
                if delta_transform.is_identity
                else _cosine(delta32, scaled32),
                realized_alpha_ratio=_realized_alpha_ratio(stored, base32, intended32),
            )
        )
    return amplified, deltas


def _copy_side_files(rl_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in COPIED_SIDE_FILES:
        source = rl_dir / name
        if source.is_file():
            shutil.copyfile(source, out_dir / name)
            copied.append(name)
    if "config.json" not in copied:
        raise FileNotFoundError(f"{rl_dir} has no config.json; the unit would not be a checkpoint")
    return copied


def _rewrite_config_dtype(config_path: Path, storage_dtype: str) -> bool:
    """Point the unit's config at its storage dtype so an engine on ``dtype=auto`` serves it as stored.

    transformers spells the key ``dtype`` (current) or ``torch_dtype`` (older), at the top level and,
    on the composite Qwen3.5 configs, inside ``text_config`` as well; every one present is rewritten,
    and ``mamba_ssm_dtype`` (the DeltaNet state's own precision) is deliberately left alone. Returns
    whether any key was rewritten, which the sidecar records.
    """
    config = cast("dict[str, Any]", json.loads(config_path.read_text(encoding="utf-8")))
    rewritten = False
    blocks: list[dict[str, Any]] = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        blocks.append(cast("dict[str, Any]", text_config))
    for block in blocks:
        for key in CONFIG_DTYPE_KEYS:
            if key in block and block[key] != storage_dtype:
                block[key] = storage_dtype
                rewritten = True
    if not any(key in config for key in CONFIG_DTYPE_KEYS):
        # A config that names no dtype leaves the engine to guess; the unit says what it is.
        config[CONFIG_DTYPE_KEYS[0]] = storage_dtype
        rewritten = True
    if rewritten:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return rewritten


def _facts_record(label: str, facts: FullWeightsFacts) -> dict[str, object]:
    return {
        "label": label,
        "snapshot_dir": str(facts.snapshot_dir),
        "commit_sha": facts.commit_sha,
        "weights_fingerprint": facts.fingerprint,
    }


def build_amplified_checkpoint(  # noqa: PLR0913 - one keyword per recorded build decision
    *,
    base: FullWeightsFacts,
    rl: FullWeightsFacts,
    alpha: float,
    out_dir: Path,
    label: str,
    delta_transform: DeltaTransform = IDENTITY_DELTA_TRANSFORM,
    storage_dtype: str = STORAGE_DTYPE_RELEASE,
) -> AmplifyReport:
    """Assemble one amplified unit at ``out_dir``: tensors, side files, and the provenance sidecar.

    ``out_dir`` must not already exist: a unit is immutable once built, because its label is what
    a measurement files records under, and overwriting one in place is how two different sets of
    tensors end up under one label. Both inputs arrive as resolved facts so the sidecar can name
    the exact commits (and fingerprints) that went in. ``delta_transform`` is recorded in the
    sidecar beside the alpha: a permuted control and a real amplification at the same alpha are
    different units, and the sidecar is where a reader tells them apart.
    """
    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists; a unit is built once under its label, never overwritten"
        )
    amplified, deltas = amplify_tensors(
        base_dir=base.snapshot_dir,
        rl_dir=rl.snapshot_dir,
        alpha=alpha,
        delta_transform=delta_transform,
        storage_dtype=storage_dtype,
    )
    out_dir.mkdir(parents=True)
    weights_path = out_dir / OUTPUT_WEIGHTS_FILENAME
    save_file(amplified, str(weights_path), metadata={"format": "pt"})
    del amplified
    copied = _copy_side_files(rl.snapshot_dir, out_dir)
    config_dtype_rewritten = storage_dtype != STORAGE_DTYPE_RELEASE and _rewrite_config_dtype(
        out_dir / "config.json", storage_dtype
    )
    relative = [delta.relative_delta for delta in deltas]
    dtypes: dict[str, int] = {}
    for delta in deltas:
        dtypes[delta.dtype] = dtypes.get(delta.dtype, 0) + 1
    report = AmplifyReport(
        label=label,
        alpha=alpha,
        n_tensors=len(deltas),
        weights_sha256={OUTPUT_WEIGHTS_FILENAME: sha256_of_file(weights_path)},
        base=_facts_record(base.label, base),
        rl=_facts_record(rl.label, rl),
        max_relative_delta=max(relative),
        mean_relative_delta=sum(relative) / len(relative),
        dtypes=dtypes,
        delta_transform=delta_transform.name,
        permutation_seed=delta_transform.seed,
        mean_abs_cosine_to_real_delta=sum(abs(d.cosine_to_real_delta) for d in deltas)
        / len(deltas),
        storage_dtype=storage_dtype,
        config_dtype_rewritten=config_dtype_rewritten,
        mean_realized_alpha_ratio=sum(
            d.realized_alpha_ratio for d in deltas if not d.copied_from_release
        )
        / max(1, sum(1 for d in deltas if not d.copied_from_release)),
        n_copied_from_release=sum(1 for d in deltas if d.copied_from_release),
    )
    sidecar = report.to_json_dict()
    sidecar["copied_from_release"] = [d.name for d in deltas if d.copied_from_release]
    sidecar["copied_side_files"] = copied
    sidecar["tensor_deltas"] = [
        {**asdict(delta), "relative_delta": delta.relative_delta} for delta in deltas
    ]
    (out_dir / WEIGHTS_PROVENANCE_FILENAME).write_text(
        json.dumps(sidecar, indent=1) + "\n", encoding="utf-8"
    )
    logger.info(
        f"amplified unit built, {label=} {alpha=} n_tensors={report.n_tensors} "
        f"storage_dtype={storage_dtype} config_dtype_rewritten={config_dtype_rewritten} "
        f"mean_realized_alpha_ratio={report.mean_realized_alpha_ratio:.3f} "
        f"n_copied_from_release={report.n_copied_from_release} "
        f"max_relative_delta={report.max_relative_delta:.3e} "
        f"mean_relative_delta={report.mean_relative_delta:.3e} out={out_dir}"
    )
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base", required=True, help="base checkpoint: hub repo id or local directory"
    )
    parser.add_argument(
        "--base-revision", default=None, help="hub revision of --base (hub source only)"
    )
    parser.add_argument(
        "--rl", required=True, help="RL'd checkpoint: hub repo id or local directory"
    )
    parser.add_argument(
        "--rl-revision", default=None, help="hub revision of --rl (hub source only)"
    )
    parser.add_argument(
        "--alpha", type=float, required=True, help="delta multiplier; 1.0 reproduces --rl"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory; must not exist")
    parser.add_argument(
        "--label",
        default=None,
        help="unit label written into the sidecar; defaults to the output directory's name",
    )
    parser.add_argument(
        "--delta-transform",
        choices=DELTA_TRANSFORMS,
        default=DELTA_TRANSFORM_IDENTITY,
        help=(
            "identity scales the real delta; permute-axes scales a copy of it permuted along every "
            "axis (singular values and Frobenius norm preserved, alignment with the base destroyed): "
            "the matched-magnitude control for the alpha ladder. Needs --permutation-seed."
        ),
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="seed of the axis permutations under --delta-transform permute-axes; refused otherwise",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=STORAGE_DTYPES,
        default=STORAGE_DTYPE_RELEASE,
        help=(
            "dtype the amplified tensors are stored in: release (each tensor's own, bf16 here), "
            "float16 (three more mantissa bits, so a 1.1-1.3 alpha step survives rounding; the "
            "config's dtype keys are rewritten so the engine serves it as stored) or float32"
        ),
    )
    args = parser.parse_args(argv)
    args.delta_transform = DeltaTransform(name=args.delta_transform, seed=args.permutation_seed)
    return args


def main(argv: list[str] | None = None) -> int:
    """Resolve both inputs (fetching and verifying a hub source), then build the unit."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    base = resolve_full_weights(FullWeightsSource.parse(args.base, args.base_revision))
    rl = resolve_full_weights(FullWeightsSource.parse(args.rl, args.rl_revision))
    report = build_amplified_checkpoint(
        base=base,
        rl=rl,
        alpha=args.alpha,
        out_dir=args.out,
        label=args.label or args.out.name,
        delta_transform=args.delta_transform,
        storage_dtype=args.storage_dtype,
    )
    # The directory must pass the same gate the probe applies before serving it.
    served = resolve_full_weights(FullWeightsSource.parse(str(args.out), None))
    logger.info(
        f"unit verified as servable, label={served.label} fingerprint={served.fingerprint} "
        f"delta_transform={report.delta_transform} permutation_seed={report.permutation_seed} "
        f"mean_abs_cosine_to_real_delta={report.mean_abs_cosine_to_real_delta:.3f} "
        f"storage_dtype={report.storage_dtype} "
        f"mean_realized_alpha_ratio={report.mean_realized_alpha_ratio:.3f} "
        f"weights_sha256={report.weights_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
