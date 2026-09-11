"""What a FULL-WEIGHTS interp cell is, how its checkpoint is loaded and gated, and when two may be compared.

`games.interp_cells` defines a cell as one (arm, step) of cached activations and guards the measured
space (base model, layer convention, batch size, dtypes, render). Its cells come from LoRA adapters on
one resident base. The TMAX checkpoints are full weights at hub revisions, so a cell of that kind
needs three things the games contract does not carry, all defined here and all recorded in the cell's
provenance block so the games manifest format is unchanged:

* **A loading gate.** The checkpoint loads through TRL's `create_model_from_path`, the builder
  `games.lora.load_adapter_base` wraps (it instantiates the class the config names,
  `Qwen3_5ForConditionalGeneration`, whose text tower sits at `model.language_model.*`). The wrapper's
  signature has no room for `output_loading_info`, so the builder is called here directly and its
  report is held to one rule: zero missing and zero unexpected `model.language_model.*` keys. The
  vision tower may be missing (every TMAX release ships the language model only, and the tower never
  runs) and the multi-token-prediction head may be unexpected (the base mirrors carry one the class
  has no slot for). Anything else -- a renamed key, another size's shard, a mismatched shape -- is
  refused before a forward runs, because a randomly initialised layer under a checkpoint's name is a
  plausible null rather than an error.
* **A weights fingerprint and a replica declaration.** `weights_fingerprint` is
  `FullWeightsFacts.fingerprint`, the digest of the per-file digests. Two cells sharing one are
  refused unless one declares `replica_of` the other; the wave's declared pair is the base and its
  `hamishivi` mirror, whose displacement must read exactly 0.0. That zero is the cheapest proof that
  the capture is deterministic and that "checkpoint minus base" is about the weights. The converse
  guard is the full-weights form of `assert_adapter_moved_activations`: cells with different
  fingerprints whose activations are bit-identical mean the weights never reached the forward.
* **The coherence triple** (gradable, truncation and well-formed shares), carried from whatever
  record source the checkpoint was probed on, so a direction fitted on a cell can be read beside how
  coherent that checkpoint's generations were. NaN with a reason for a cell nobody probed.

`load_full_weights_ladder` is `games.interp_cells.load_ladder` plus these guards plus the games
DeltaNet-kernel mixing guard over each cell's recorded binding.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from trl.trainer.utils import create_model_from_path

from games.deltanet_kernels import DELTANET_KERNEL_FIELD, assert_one_deltanet_kernel
from games.eval_model import WEIGHTS_PROVENANCE_FILENAME, FullWeightsFacts, FullWeightsSource
from games.interp_cells import (
    STEP_DIR_PREFIX,
    CapturedCell,
    CellFormatError,
    Ladder,
    load_ladder,
    short_digest,
)
from reward_hacking.tmax.amplify import DELTA_TRANSFORMS
from reward_hacking.tmax.artifacts import DROPPED_TOWER_KEY_PREFIXES, LANGUAGE_MODEL_KEY_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch
    from torch import nn
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# A 9B prompt (8,192) plus TMAX's own 32,768 completion budget; anything longer is a different study.
MAX_PROMPT_TOKENS = 40_960

# Provenance fields this cell kind adds to the games manifest: written by `cell_provenance` only.
CELL_KIND_FIELD = "cell_kind"
CELL_KIND_FULL_WEIGHTS = "full-weights"
WEIGHTS_FINGERPRINT_FIELD = "weights_fingerprint"
REPLICA_OF_FIELD = "replica_of"
COHERENCE_FIELD = "coherence"
SPANS_SIDECAR_FIELD = "spans_sidecar"
SPAN_NAMES_FIELD = "span_names"
SPANS_ABSENT_FIELD = "spans_absent"

SPAN_SET_SEPARATOR = "@"
"""`<stimulus set>@<span name>`; no games or twin-corpus set name carries `@`, and tensor keys split on `/`."""


def derived_set_name(stimulus_set: str, span_name: str) -> str:
    """Name the stimulus set a span pooling is stored under: `<set>@<span>`.

    The capture writes one such set per requested span, holding `mean` and `last` pooled over the
    span alone; a reader wanting the `assertion32` pools of the twin set asks the cell for this name.
    """
    return f"{stimulus_set}{SPAN_SET_SEPARATOR}{span_name}"


LOADING_REPORT_KEYS: tuple[str, ...] = (
    "missing_keys",
    "unexpected_keys",
    "mismatched_keys",
    "error_msgs",
)
"""What transformers' `LoadStateDictInfo.to_dict` emits (5.15.0); a report lacking one is refused."""

COHERENCE_SHARE_KEYS: tuple[str, ...] = ("gradable_share", "truncation_share", "well_formed_share")

ERROR_EXAMPLE_COUNT = 5


class FullWeightsCellError(ValueError):
    """A full-weights cell cannot be built, or its checkpoint is not the one its label names."""


# --------------------------------------------------------------------------------------
# Cell specifications: which weights, under which label, replica of what
# --------------------------------------------------------------------------------------


def cell_label(arm: str, step: int) -> str:
    """Spell a cell the way the CLI does, `arm:step`."""
    return f"{arm}:{step}"


def games_label(label: str) -> str:
    """Translate the CLI spelling `arm:step` into the games manifest spelling `arm/step-N`."""
    arm, separator, step = label.rpartition(":")
    if not separator or not arm or not step.isdigit():
        raise FullWeightsCellError(f"{label!r} is not a cell label of the form arm:step")
    return f"{arm}/{STEP_DIR_PREFIX}{int(step)}"


@dataclass(frozen=True)
class FullWeightsCellSpec:
    """One unit of work: a labelled checkpoint to load, capture and free."""

    arm: str
    step: int
    source: FullWeightsSource
    replica_of: str | None = None

    @property
    def label(self) -> str:
        """How this cell is named on the CLI and in the replica table."""
        return cell_label(self.arm, self.step)


def parse_cell_spec(spec: str) -> FullWeightsCellSpec:
    """Read `arm:step=repo/name@revision` or `arm:step=/local/dir` into a cell spec.

    A hub source must spell its revision: the TMAX repos put a different training step on every
    branch and the default branch aliases one of them. A local directory takes no revision.
    """
    label, separator, source_spec = spec.partition("=")
    arm, step_separator, step_text = label.partition(":")
    if not separator or not step_separator or not arm or not step_text.isdigit() or not source_spec:
        raise FullWeightsCellError(
            f"--cell expects 'arm:step=repo/name@revision' or 'arm:step=/local/dir', got {spec!r}"
        )
    if Path(source_spec).is_dir():
        source = FullWeightsSource(repo_id=None, revision=None, local_dir=Path(source_spec))
    else:
        repo_id, at, revision = source_spec.rpartition("@")
        if not at or not repo_id or not revision:
            raise FullWeightsCellError(
                f"--cell {spec!r}: {source_spec!r} is not an existing directory, so it must be a hub "
                f"source spelled repo/name@revision (the revision is required: every branch of a "
                f"TMAX repo is a different step)"
            )
        source = FullWeightsSource.parse(repo_id, revision)
    return FullWeightsCellSpec(arm=arm, step=int(step_text), source=source)


def parse_replica_spec(spec: str) -> tuple[str, str]:
    """Read `arm:step=arm:step` as (replica label, anchor label)."""
    replica, separator, anchor = spec.partition("=")
    if not separator or not replica or not anchor or replica == anchor:
        raise FullWeightsCellError(
            f"--replica expects 'arm:step=arm:step' naming a cell and the cell it replicates, got "
            f"{spec!r}"
        )
    return replica, anchor


def apply_replica_declarations(
    specs: Sequence[FullWeightsCellSpec], declarations: Mapping[str, str]
) -> list[FullWeightsCellSpec]:
    """Attach each `--replica` declaration to its cell, refusing one naming a cell not on the agenda."""
    by_label = {spec.label: spec for spec in specs}
    if len(by_label) != len(specs):
        raise FullWeightsCellError(f"cells repeat a label: {[spec.label for spec in specs]}")
    named = set(declarations) | set(declarations.values())
    unknown = sorted(label for label in named if label not in by_label)
    if unknown:
        raise FullWeightsCellError(
            f"--replica names cells that are not on the agenda: {unknown}; agenda {sorted(by_label)}"
        )
    return [
        FullWeightsCellSpec(
            arm=spec.arm,
            step=spec.step,
            source=spec.source,
            replica_of=declarations.get(spec.label),
        )
        for spec in specs
    ]


def assert_fingerprints_declared(
    resolved: Mapping[str, tuple[FullWeightsCellSpec, FullWeightsFacts]],
) -> None:
    """Refuse two cells that serve the same bytes unless one declares itself a replica of the other.

    Runs before any GPU work, on the fingerprints `resolve_full_weights` computed. Silence here is the
    failure: two labels over identical weights read downstream as a checkpoint that moved nothing,
    which is indistinguishable from a real null. A declaration pointing at a cell with DIFFERENT
    bytes is refused too, since it claims a zero displacement the tensors will not deliver.
    """
    by_fingerprint: dict[str, list[str]] = {}
    for label, (_spec, facts) in resolved.items():
        by_fingerprint.setdefault(facts.fingerprint, []).append(label)
    for label, (spec, facts) in resolved.items():
        if spec.replica_of is None:
            continue
        anchor_spec, anchor_facts = resolved[spec.replica_of]
        if anchor_facts.fingerprint != facts.fingerprint:
            raise FullWeightsCellError(
                f"{label} declares itself a replica of {anchor_spec.label}, but their weights differ "
                f"({short_digest(facts.fingerprint)} vs {short_digest(anchor_facts.fingerprint)}); a "
                f"replica is a second label over the same bytes, and this pair would be read as a "
                f"zero displacement the tensors cannot deliver"
            )
    for fingerprint, labels in by_fingerprint.items():
        undeclared = [label for label in labels if resolved[label][0].replica_of is None]
        if len(labels) > 1 and len(undeclared) != 1:
            raise FullWeightsCellError(
                f"cells {labels} all serve weights with fingerprint {short_digest(fingerprint)}; "
                f"exactly one of them may stand undeclared and every other has to say --replica "
                f"<label>=<anchor> ({len(undeclared)} undeclared: {undeclared}). Otherwise a label "
                f"over duplicate weights reads as a checkpoint that changed nothing."
            )
    logger.info(
        f"fingerprints checked, cells={len(resolved)} distinct={len(by_fingerprint)} "
        f"replicas={[label for label, (spec, _f) in resolved.items() if spec.replica_of]}"
    )


# --------------------------------------------------------------------------------------
# Coherence triple
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CoherenceTriple:
    """How coherent a checkpoint's generations were on the records it was probed with.

    `gradable_share` is graded over examined, `truncation_share` reasoning-truncated over examined,
    `well_formed_share` well-formed submissions over examined. NaN across the board, with `reason`
    set, for a cell whose checkpoint was never probed (the permuted control before Phase 1 reaches
    it, a base mirror, a prompt-only corpus).
    """

    gradable_share: float
    truncation_share: float
    well_formed_share: float
    source: str | None
    reason: str | None

    @classmethod
    def unavailable(cls, reason: str) -> CoherenceTriple:
        """Build the honest empty triple: NaNs plus the reason nothing was measured."""
        return cls(math.nan, math.nan, math.nan, source=None, reason=reason)

    @property
    def available(self) -> bool:
        """Whether the three shares were measured."""
        return not math.isnan(self.gradable_share)

    def to_payload(self) -> dict[str, Any]:
        """JSON form: shares as numbers, or null with the reason (NaN is not JSON)."""
        shares = (self.gradable_share, self.truncation_share, self.well_formed_share)
        return {
            **{
                key: (share if self.available else None)
                for key, share in zip(COHERENCE_SHARE_KEYS, shares, strict=True)
            },
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, source: str) -> CoherenceTriple:
        """Read :meth:`to_payload` back: nulls become NaN, and a payload missing a share is refused."""
        missing = [key for key in (*COHERENCE_SHARE_KEYS, "source", "reason") if key not in payload]
        if missing:
            raise CellFormatError(f"{source}: the coherence block lacks {missing}")
        raw = [payload[key] for key in COHERENCE_SHARE_KEYS]
        if any(value is None for value in raw) != all(value is None for value in raw):
            raise CellFormatError(f"{source}: the coherence shares are partly null: {raw}")
        shares = [math.nan if value is None else float(value) for value in raw]
        return cls(
            gradable_share=shares[0],
            truncation_share=shares[1],
            well_formed_share=shares[2],
            source=cast("str | None", payload["source"]),
            reason=cast("str | None", payload["reason"]),
        )


def load_coherence_table(path: Path) -> dict[str, CoherenceTriple]:
    """Read a `{"arm:step": {gradable_share, truncation_share, well_formed_share, source}}` map."""
    raw = cast("dict[str, dict[str, Any]]", json.loads(path.read_text(encoding="utf-8")))
    table: dict[str, CoherenceTriple] = {}
    for label, entry in raw.items():
        missing = [key for key in COHERENCE_SHARE_KEYS if key not in entry]
        if missing:
            raise FullWeightsCellError(f"{path}: coherence entry {label!r} lacks {missing}")
        shares = {key: float(entry[key]) for key in COHERENCE_SHARE_KEYS}
        outside = {key: share for key, share in shares.items() if not 0.0 <= share <= 1.0}
        if outside:
            raise FullWeightsCellError(
                f"{path}: coherence entry {label!r} has shares outside [0, 1]: {outside}"
            )
        table[label] = CoherenceTriple(
            gradable_share=shares["gradable_share"],
            truncation_share=shares["truncation_share"],
            well_formed_share=shares["well_formed_share"],
            source=cast("str | None", entry.get("source")),
            reason=None,
        )
    return table


# --------------------------------------------------------------------------------------
# Loading: the games loaders' builder, plus the report they cannot return
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadingReport:
    """What the loading gate saw, as recorded in the cell manifest."""

    n_language_model_loaded: int
    n_missing_tower: int
    n_unexpected_tower: int

    def to_payload(self) -> dict[str, int]:
        """JSON form."""
        return {
            "n_language_model_loaded": self.n_language_model_loaded,
            "n_missing_tower": self.n_missing_tower,
            "n_unexpected_tower": self.n_unexpected_tower,
        }


def assert_language_model_complete(
    loading_info: Mapping[str, Sequence[Any]], *, label: str, n_language_model_keys: int
) -> LoadingReport:
    """Hold a loading report to: zero missing and zero unexpected `model.language_model.*` keys.

    Missing keys may only be vision-tower keys (the TMAX releases ship none, and the tower never
    runs here); unexpected keys may only be tower or multi-token-prediction keys (the base mirrors
    carry an MTP head the composite class has no slot for). A mismatched shape or a load error
    anywhere refuses. `n_language_model_keys` is how many language-model tensors the model expects,
    so the report states how many loaded rather than only that none were missing.
    """
    absent = [key for key in LOADING_REPORT_KEYS if key not in loading_info]
    if absent:
        raise FullWeightsCellError(
            f"{label}: the loading report lacks {absent}; transformers changed what "
            f"output_loading_info returns, so the gate cannot read it (has {sorted(loading_info)})"
        )
    missing = [str(key) for key in loading_info["missing_keys"]]
    unexpected = [str(key) for key in loading_info["unexpected_keys"]]
    mismatched = list(loading_info["mismatched_keys"])
    errors = [str(message) for message in loading_info["error_msgs"]]
    if errors or mismatched:
        raise FullWeightsCellError(
            f"{label}: loading reported {len(errors)} error(s) and {len(mismatched)} mismatched "
            f"tensor(s): {errors[:ERROR_EXAMPLE_COUNT]} {mismatched[:ERROR_EXAMPLE_COUNT]}"
        )
    missing_text = [key for key in missing if key.startswith(LANGUAGE_MODEL_KEY_PREFIX)]
    unexpected_text = [key for key in unexpected if key.startswith(LANGUAGE_MODEL_KEY_PREFIX)]
    if missing_text or unexpected_text:
        raise FullWeightsCellError(
            f"{label}: the language model did not load whole: {len(missing_text)} "
            f"{LANGUAGE_MODEL_KEY_PREFIX}* key(s) missing ({missing_text[:ERROR_EXAMPLE_COUNT]}) "
            f"and {len(unexpected_text)} unexpected ({unexpected_text[:ERROR_EXAMPLE_COUNT]}). A "
            f"missing key is a randomly initialised layer under this checkpoint's name; an "
            f"unexpected one is a tensor that never reached the model. Neither is the checkpoint "
            f"the label names."
        )
    stray_missing = [key for key in missing if not key.startswith(DROPPED_TOWER_KEY_PREFIXES)]
    stray_unexpected = [key for key in unexpected if not key.startswith(DROPPED_TOWER_KEY_PREFIXES)]
    if stray_missing or stray_unexpected:
        raise FullWeightsCellError(
            f"{label}: keys outside both the language model and the dropped towers "
            f"{DROPPED_TOWER_KEY_PREFIXES}: missing {stray_missing[:ERROR_EXAMPLE_COUNT]} "
            f"({len(stray_missing)}), unexpected {stray_unexpected[:ERROR_EXAMPLE_COUNT]} "
            f"({len(stray_unexpected)}); this checkpoint is not laid out like a Qwen3.5 release"
        )
    report = LoadingReport(
        n_language_model_loaded=n_language_model_keys,
        n_missing_tower=len(missing),
        n_unexpected_tower=len(unexpected),
    )
    logger.info(f"loading gate passed, {label=} {report.to_payload()}")
    return report


def language_model_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder blocks of a composite load, at `model.model.language_model.layers`.

    The shared capture (`reward_hacking.interp.directions`) resolves the trunk through its own chain;
    this names the one shape a full-weights cell can have, so span hooks land on the same modules and
    a differently laid-out checkpoint is refused rather than hooked somewhere plausible.
    """
    outer = getattr(model, "model", None)
    trunk = getattr(outer, "language_model", None)
    layers = getattr(trunk, "layers", None)
    if layers is None:
        raise FullWeightsCellError(
            f"{type(model).__name__} has no model.language_model.layers; a full-weights cell is the "
            f"composite class the checkpoint config names (Qwen3_5ForConditionalGeneration), which "
            f"is what create_model_from_path builds"
        )
    return cast("nn.ModuleList", layers)


def load_full_weights_model(
    facts: FullWeightsFacts, *, dtype: torch.dtype, device: torch.device
) -> tuple[PreTrainedModel, LoadingReport]:
    """Load one resolved checkpoint through the games loaders' builder and gate its loading report.

    `create_model_from_path` is what `games.lora.load_adapter_base` calls; it is called here directly
    because that wrapper's signature has nowhere to put `output_loading_info`, and the report is the
    whole point. `device_map=None` for the same reason the wrapper passes it (TRL would otherwise
    shard across every visible card); sdpa attention because a 40,960-token row would otherwise
    materialise an O(seq^2) score matrix on each of the eight full-attention layers.
    """
    loaded = cast(
        "tuple[PreTrainedModel, dict[str, list[Any]]]",
        create_model_from_path(
            str(facts.snapshot_dir),
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            attn_implementation="sdpa",
            output_loading_info=True,
        ),
    )
    model, loading_info = loaded
    n_text = sum(1 for name in model.state_dict() if name.startswith(LANGUAGE_MODEL_KEY_PREFIX))
    report = assert_language_model_complete(
        loading_info, label=facts.label, n_language_model_keys=n_text
    )
    text_config = getattr(model.config, "text_config", model.config)
    # Where transformers records the resolved attention backend; there is no public accessor.
    attention = text_config._attn_implementation  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if attention != "sdpa":
        raise FullWeightsCellError(
            f"{facts.label} loaded with attention {attention!r} rather than sdpa; the full-attention "
            f"layers would materialise an O(seq^2) score matrix at {MAX_PROMPT_TOKENS} tokens"
        )
    # transformers 5.x wraps `.to`, which pyright reads as an unbound __call__ wanting `self`.
    placed = model.to(device)  # pyright: ignore[reportArgumentType]
    return placed.eval(), report


def read_amplified_sidecar(facts: FullWeightsFacts) -> dict[str, Any] | None:
    """Recognise a `reward_hacking.tmax.amplify` unit by its sidecar and return the summary to record.

    `resolve_full_weights` has already held the tensors to the sidecar's digests; this reads what the
    unit IS -- alpha, transform, seed, storage dtype, the base and RL checkpoints it was built from --
    so a permuted control and a real amplification at the same alpha are told apart in the manifest.
    The per-tensor delta table stays behind: it is the unit's record, not the cell's. A directory
    with no sidecar is a plain checkpoint and returns None; a sidecar that does not describe an
    amplification is refused rather than recorded as one.
    """
    path = facts.snapshot_dir / WEIGHTS_PROVENANCE_FILENAME
    if not path.is_file():
        return None
    sidecar = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    required = (
        "schema",
        "kind",
        "label",
        "alpha",
        "delta_transform",
        "storage_dtype",
        "base",
        "rl",
    )
    missing = [key for key in required if key not in sidecar]
    if missing:
        raise FullWeightsCellError(
            f"{path} lacks {missing}, so it does not describe an amplified unit; a directory with a "
            f"sidecar this driver cannot read is not served under a label it cannot explain"
        )
    if sidecar["delta_transform"] not in DELTA_TRANSFORMS:
        raise FullWeightsCellError(
            f"{path}: delta_transform {sidecar['delta_transform']!r} is not one of {DELTA_TRANSFORMS}"
        )
    for side in ("base", "rl"):
        record = cast("dict[str, Any]", sidecar[side])
        if "weights_fingerprint" not in record or "label" not in record:
            raise FullWeightsCellError(f"{path}: the {side!r} record names no label or fingerprint")
    recorded = (
        "schema",
        "kind",
        "label",
        "alpha",
        "delta_transform",
        "permutation_seed",
        "storage_dtype",
        "mean_realized_alpha_ratio",
        "mean_abs_cosine_to_real_delta",
        "base",
        "rl",
    )
    summary = {key: sidecar.get(key) for key in recorded}
    logger.info(
        f"amplified unit recognised, label={summary['label']} alpha={summary['alpha']} "
        f"transform={summary['delta_transform']} seed={summary['permutation_seed']}"
    )
    return summary


# --------------------------------------------------------------------------------------
# The cell's provenance block: one writer, and the readers that spell each field the same way
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureRunFacts:
    """What one capture run records into every cell it writes, beyond the cell's own weights."""

    git_sha: str
    torch_version: str
    device: str
    stimuli_file: str
    spans_sidecar: str | None
    span_names: tuple[str, ...]
    tokenizer_id: str
    tokenizer_revision: str | None

    @property
    def identity_base(self) -> str:
        """Return the `base_model` identity field: the tokenizer of record, with its revision when pinned."""
        if self.tokenizer_revision is None:
            return self.tokenizer_id
        return f"{self.tokenizer_id}@{self.tokenizer_revision}"

    def to_payload(self) -> dict[str, Any]:
        """Return the run-level fields as the ladder manifest carries them."""
        return {
            "git_sha": self.git_sha,
            "torch_version": self.torch_version,
            "device": self.device,
            "identity_base": self.identity_base,
            "stimuli_file": self.stimuli_file,
            SPANS_SIDECAR_FIELD: self.spans_sidecar,
            SPAN_NAMES_FIELD: list(self.span_names),
            "tokenizer": {"id": self.tokenizer_id, "revision": self.tokenizer_revision},
        }


def cell_provenance(  # noqa: PLR0913 - the provenance block is exactly these facts, each named
    spec: FullWeightsCellSpec,
    facts: FullWeightsFacts,
    *,
    run: CaptureRunFacts,
    loading_report: LoadingReport,
    deltanet_kernel: Mapping[str, str],
    amplified: Mapping[str, Any] | None,
    coherence: CoherenceTriple,
    spans_absent: Mapping[str, Mapping[str, str]],
    seconds: float,
    peak_vram_bytes: int | None,
    captured_at: str,
) -> dict[str, Any]:
    """Build the provenance block of one full-weights cell: the one place its field names are written.

    `replica_of` holds the CLI label (`arm:step`) of the cell this one replicates, or null; readers go
    through :func:`declared_replica_anchor`, which translates it into the games label the cells carry.
    """
    return {
        "git_sha": run.git_sha,
        "torch_version": run.torch_version,
        "device": run.device,
        "stimuli_file": run.stimuli_file,
        DELTANET_KERNEL_FIELD: dict(deltanet_kernel),
        CELL_KIND_FIELD: CELL_KIND_FULL_WEIGHTS,
        "weights_label": facts.label,
        "weights_source": {
            "repo_id": spec.source.repo_id,
            "revision": spec.source.revision,
            "local_dir": None if spec.source.local_dir is None else str(spec.source.local_dir),
        },
        "weights_commit_sha": facts.commit_sha,
        WEIGHTS_FINGERPRINT_FIELD: facts.fingerprint,
        "weights_sha256": dict(facts.weights_sha256),
        "weights_chat_template_sha256": facts.chat_template_sha256,
        "declares_vision_config": facts.declares_vision_config,
        REPLICA_OF_FIELD: spec.replica_of,
        "amplified_unit": None if amplified is None else dict(amplified),
        "loading_report": loading_report.to_payload(),
        "attn_implementation": "sdpa",
        "tokenizer": {"id": run.tokenizer_id, "revision": run.tokenizer_revision},
        SPANS_SIDECAR_FIELD: run.spans_sidecar,
        SPAN_NAMES_FIELD: list(run.span_names),
        SPANS_ABSENT_FIELD: {name: dict(reasons) for name, reasons in spans_absent.items()},
        COHERENCE_FIELD: coherence.to_payload(),
        "seconds": seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "captured_at": captured_at,
    }


def _require_field(cell: CapturedCell, field: str) -> Any:  # noqa: ANN401 - JSON, typed by each reader
    if field not in cell.provenance:
        raise CellFormatError(
            f"{cell.label} records no {field!r} in its provenance; the full-weights capture writes it "
            f"for every cell, so this cell was written by something else or the field was renamed, "
            f"and a read that assumed a default here would be a read of nothing"
        )
    return cell.provenance[field]


def cell_fingerprint(cell: CapturedCell) -> str:
    """Return the `weights_fingerprint` a cell records, refusing a cell that is not a full-weights cell."""
    fingerprint = cell.provenance.get(WEIGHTS_FINGERPRINT_FIELD)
    if not isinstance(fingerprint, str):
        raise CellFormatError(
            f"{cell.label} records no {WEIGHTS_FINGERPRINT_FIELD}, so it is not a full-weights cell "
            f"and cannot be checked for duplicate weights"
        )
    return fingerprint


def declared_replica_anchor(cell: CapturedCell) -> str | None:
    """Return the games label (`arm/step-N`) of the cell this one declares itself a replica of, or None.

    The writer stores the CLI spelling (`arm:step`); every comparison against `CapturedCell.label`
    goes through here so the two spellings never meet. A cell without the field at all is refused by
    name: the writer records null for a cell that replicates nothing, so absence is not "no".
    """
    declared = _require_field(cell, REPLICA_OF_FIELD)
    if declared is None:
        return None
    if not isinstance(declared, str):
        raise CellFormatError(
            f"{cell.label} records {REPLICA_OF_FIELD}={declared!r}, which is not a cell label"
        )
    return games_label(declared)


def coherence_of(cell: CapturedCell) -> CoherenceTriple:
    """Return the coherence triple a cell carries for the checkpoint it captured."""
    return CoherenceTriple.from_payload(
        cast("Mapping[str, Any]", _require_field(cell, COHERENCE_FIELD)), source=cell.label
    )


def spans_absent_of(cell: CapturedCell) -> dict[str, dict[str, str]]:
    """Return, per derived span set, the stimuli the capture left out of it and the sidecar's reason for each."""
    recorded = cast("Mapping[str, Mapping[str, str]]", _require_field(cell, SPANS_ABSENT_FIELD))
    return {
        str(name): {str(stimulus_id): str(reason) for stimulus_id, reason in reasons.items()}
        for name, reasons in recorded.items()
    }


# --------------------------------------------------------------------------------------
# Reading a full-weights ladder back: the games guards plus the three this cell kind adds
# --------------------------------------------------------------------------------------


def _max_abs_displacement(a: CapturedCell, b: CapturedCell) -> float:
    return max(
        float((a.matrix(stimulus_set, pooling) - b.matrix(stimulus_set, pooling)).abs().max())
        for stimulus_set, pooling in sorted(a.activations)
    )


def _replica_anchors(cells: Sequence[CapturedCell]) -> dict[str, str]:
    """Each declared replica's games label -> its anchor's games label, refusing an absent anchor."""
    labels = {cell.label for cell in cells}
    anchors: dict[str, str] = {}
    for cell in cells:
        anchor = declared_replica_anchor(cell)
        if anchor is None:
            continue
        if anchor not in labels:
            raise CellFormatError(
                f"{cell.label} declares itself a replica of {anchor!r}, which is not in this "
                f"ladder ({sorted(labels)}); the pair cannot be checked, so it cannot be read"
            )
        anchors[cell.label] = anchor
    return anchors


def assert_replicas_and_displacement(cells: Sequence[CapturedCell]) -> dict[str, float]:
    """Refuse a replica pair that moved, or a non-replica pair that did not; return each pair's max |delta|.

    A declared replica (same bytes) whose activations differ by any amount says the forward is not
    deterministic, and every "checkpoint minus base" number in the ladder inherits that noise. Two
    cells with different weights that are bit-identical say the weights never reached the forward --
    the full-weights form of `games.interp_cells.assert_adapter_moved_activations`. Undeclared
    duplicate fingerprints are refused here too, for a ladder assembled from separate captures.
    """
    by_label = {cell.label: cell for cell in cells}
    fingerprints = {cell.label: cell_fingerprint(cell) for cell in cells}
    anchors = _replica_anchors(cells)
    displacements: dict[str, float] = {}
    ordered = sorted(by_label)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            delta = _max_abs_displacement(by_label[first], by_label[second])
            displacements[f"{first} vs {second}"] = delta
            same_bytes = fingerprints[first] == fingerprints[second]
            declared = anchors.get(first) == second or anchors.get(second) == first
            digests = f"{short_digest(fingerprints[first])} vs {short_digest(fingerprints[second])}"
            if same_bytes and not declared:
                raise CellFormatError(
                    f"{first} and {second} serve the same weights (fingerprint "
                    f"{short_digest(fingerprints[first])}) and neither declares {REPLICA_OF_FIELD}; "
                    f"a second label over the same bytes reads as a checkpoint that changed nothing"
                )
            if declared and not same_bytes:
                raise CellFormatError(
                    f"{first} and {second} are declared replicas but their fingerprints differ "
                    f"({digests})"
                )
            if same_bytes and delta != 0.0:
                raise CellFormatError(
                    f"replica pair {first} / {second} differ by max |delta| {delta:.3e} despite "
                    f"serving identical weights; the capture is not deterministic, so no "
                    f"displacement in this ladder is a fact about the weights alone"
                )
            if not same_bytes and delta == 0.0:
                raise CellFormatError(
                    f"{first} and {second} serve different weights ({digests}) yet their "
                    f"activations are bit-identical; the weights did not reach the forward"
                )
    logger.info(f"replica and displacement guard passed, {displacements}")
    return displacements


def assert_one_kernel(cells: Sequence[CapturedCell]) -> object:
    """Refuse a ladder whose cells ran under different DeltaNet kernel bindings (the games guard)."""
    return assert_one_deltanet_kernel(
        (cell.provenance.get(DELTANET_KERNEL_FIELD) for cell in cells),
        what=f"full-weights ladder {[cell.label for cell in cells]}",
    )


def load_full_weights_ladder(root: Path, *, stimuli_sha256: str | None = None) -> Ladder:
    """`games.interp_cells.load_ladder` plus the kernel, replica and displacement guards."""
    ladder = load_ladder(root, stimuli_sha256=stimuli_sha256)
    assert_one_kernel(ladder.cells)
    assert_replicas_and_displacement(ladder.cells)
    return ladder
