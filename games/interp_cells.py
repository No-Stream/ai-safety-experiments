"""The cached-activation cell format for the games before/after interp arc, and its identity guards.

A GPU pass over a checkpoint ladder happens once; every direction, probe, placebo, drift cosine and
projection read afterwards is arithmetic over what it cached. So the cache -- not any one analysis
-- is the durable product, and this module is its contract: what a cell is, what a cell must record
about itself, and which of those records have to agree before two cells may be compared.

A **cell** is one (arm, checkpoint step) pair: a `manifest.json` describing the model state and the
stimulus rows, beside an `activations.safetensors` holding one `[n_rows, n_layers, hidden]` tensor
per (stimulus set, pooling). A **ladder** is a set of cells that differ only in which checkpoint
produced them, which is the whole content of a before/after read and the reason the guards here are
not decoration:

* `assert_ladder_identity` refuses cells whose *measured space* differs -- a different base model,
  layer convention, pooling batch size, storage dtype or stimulus-render convention. Any of those
  moves the numbers by more than the trained delta does, and none of them shows up as an error.
* `assert_stimuli_match` refuses to analyse a ladder against a stimulus corpus it was not captured
  on. The digest covers the ids and texts actually captured, in order, so a smoke run over four
  stimuli gets a different digest from the full corpus rather than passing as a thin version of it.
* `assert_rows_align` refuses cells whose row order or membership differs, because every read here
  is a row-wise pairing and a silently reordered cell produces a plausible number.
* `assert_adapter_moved_activations` refuses an adapted cell that is bit-identical to the base. That
  is the cheapest available check that a checkpoint's adapter reached the forward pass at all, and
  it catches the one failure `assert_adapter_applies` cannot: an adapter that loaded onto a model
  the capture then did not drive.

**The layer convention is recorded, not assumed.** Here key `i` is the output of decoder block `i`
-- the residual *after* block `i`, `n_layers` of them -- matching `reward_hacking.interp.directions`
and everything downstream of it. A capture reading `output_hidden_states` instead gets `n_layers + 1`
tensors where index `i` is the residual *entering* block `i`, so the two conventions are off by one
and a comparison that mixes them silently compares different depths. `layer_convention` in the
manifest is what makes that refusable rather than discoverable.

**Batch size is an identity field for a measured reason.** On this hybrid Gated-DeltaNet
architecture batch-1 forwards are bit-reproducible while a batch of 4 differs by 1.6-3.1% relative
L2 at mid and late layers (measured on Qwen3.5-2B, 2026-08-20) -- the same order as the trained LoRA
delta itself. Harmless within a checkpoint, and fatal to a cross-checkpoint comparison unless both
sides batched identically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors.torch import load_file, save_file

from reward_hacking.interp.directions import POOLERS
from reward_hacking.interp.linear_probe import ConceptActivations

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

CELL_MANIFEST_FILENAME = "manifest.json"
ACTIVATIONS_FILENAME = "activations.safetensors"
PREFIX_ACTIVATIONS_FILENAME = "prefix-activations.safetensors"
NATURAL_PREFIX_ACTIVATIONS_FILENAME = "natural-prefix-activations.safetensors"
NATURAL_PREFIX_MANIFEST_FILENAME = "natural-prefix-manifest.json"
LADDER_MANIFEST_FILENAME = "ladder-manifest.json"

REQUIRED_STIMULUS_FIELDS: frozenset[str] = frozenset({"id", "set", "side", "pair_id", "text"})
# A partial chain of thought to teacher-force after the template's opening `<think>`. Optional because
# the decision-theory sets carry one and a bare-stem corpus need not.
# Capture metadata is separate from model-facing prompt fields. It records construct and
# scenario/template-group membership, plus natural-prefix selection details, without putting
# authored stimulus prose in tracked code.
STIMULUS_METADATA_FIELD = "metadata"
OPTIONAL_STIMULUS_FIELDS: frozenset[str] = frozenset({"assistant_prefix", STIMULUS_METADATA_FIELD})

# Key `i` is the output of decoder block `i`. See the module docstring for the off-by-one this names.
LAYER_CONVENTION_POST_BLOCK = "post_block"

# How a corpus becomes the string the model reads. Recorded per cell, and an identity field, because
# the two conventions produce different text from the same corpus: under `templated_here` the capture
# applies the chat template to `text` and appends `assistant_prefix`, so the teacher-forced reasoning
# lands inside the assistant turn's `<think>`; under `verbatim` the corpus already holds that string
# and the capture adds nothing. Getting it backwards nests one user turn inside another and pools a
# position that is identical for both sides of every pair -- measured on this corpus 2026-08-20, with
# nothing raising and every direction at noise.
STIMULUS_RENDER_VERBATIM = "verbatim"
STIMULUS_RENDER_TEMPLATED = "templated_here"
STIMULUS_RENDERS: frozenset[str] = frozenset({STIMULUS_RENDER_VERBATIM, STIMULUS_RENDER_TEMPLATED})

STEP_DIR_PREFIX = "step-"

# The un-adapted base model is one cell shared by every arm, so it is stored under its own name.
BASE_ARM = "base"
BASE_STEP = 0

STORE_DTYPES: dict[str, torch.dtype] = {"float16": torch.float16, "float32": torch.float32}
COMPUTE_DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float32": torch.float32}

# How many offending ids or fields an error quotes; the counts beside them carry the magnitude.
ERROR_EXAMPLE_COUNT = 5

# Characters of a sha256 shown in a log line or an error. Comparisons always use the full digest.
DIGEST_DISPLAY_CHARS = 12


def short_digest(digest: str) -> str:
    """Abbreviate a digest for human reading. Never used for comparison."""
    return digest[:DIGEST_DISPLAY_CHARS]  # HARNESS-SCAN-EXEMPT-subsampling


class StimulusFileError(ValueError):
    """The stimulus corpus cannot be read as a capture input."""


class CellFormatError(ValueError):
    """A cached cell is not readable as one, or two cells cannot be compared."""


@dataclass(frozen=True)
class Stimulus:
    """One prompt to run, carrying the identity fields every later read joins on."""

    stimulus_id: str
    stimulus_set: str
    side: str
    pair_id: str
    text: str
    assistant_prefix: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CellIdentity:
    """Everything about a capture that has to match before two cells may be compared.

    Deliberately not provenance: a git sha or a timestamp differing across cells is a fact about
    when the work ran, while any field here differing means the two cells measured different things.
    """

    base_model: str
    stimuli_sha256: str
    rendered_sha256: str
    layer_convention: str
    n_layers: int
    hidden_size: int
    batch_size: int
    compute_dtype: str
    store_dtype: str
    stimulus_render: str
    tokenizer_identity: str = "unspecified"
    kernel_identity: str = "unspecified"
    capture_prefix_states: bool = False
    prompt_end_rendered_sha256: str = "unspecified"
    teacher_forced_rendered_sha256: str = "unspecified"
    natural_prefix_layers: tuple[int, ...] = ()
    natural_selection_manifest_sha256: str = ""
    natural_prefix_stimuli_sha256: str = ""
    natural_prefix_rendered_sha256: str = ""

    def __post_init__(self) -> None:
        """Reject a render convention nothing knows how to reproduce."""
        if self.stimulus_render not in STIMULUS_RENDERS:
            raise CellFormatError(
                f"unknown stimulus_render {self.stimulus_render!r}; expected one of "
                f"{sorted(STIMULUS_RENDERS)}."
            )

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe form, as written into a cell manifest."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, source: Path) -> CellIdentity:
        """Read an identity back, naming the file when a field is missing rather than KeyError-ing."""
        required_fields = {
            name
            for name, definition in cls.__dataclass_fields__.items()
            if definition.default is MISSING and definition.default_factory is MISSING
        }
        missing = sorted(required_fields - set(payload))
        if missing:
            raise CellFormatError(
                f"{source} records no {missing} in its identity block, so it cannot be checked for "
                f"comparability against another cell. Re-capture it: a cell that cannot say what it "
                f"measured is not usable in a before/after read."
            )
        values = {name: payload[name] for name in cls.__dataclass_fields__ if name in payload}
        if "natural_prefix_layers" in values:
            values["natural_prefix_layers"] = tuple(
                int(layer) for layer in values["natural_prefix_layers"]
            )
        return cls(**values)


@dataclass(frozen=True)
class RowIndex:
    """The stimulus rows of one set, in the order their activations are stored."""

    stimulus_ids: tuple[str, ...]
    sides: tuple[str, ...]
    pair_ids: tuple[str, ...]
    token_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject a row index whose columns disagree on how many rows there are."""
        lengths = {
            "stimulus_ids": len(self.stimulus_ids),
            "sides": len(self.sides),
            "pair_ids": len(self.pair_ids),
            "token_counts": len(self.token_counts),
        }
        if len(set(lengths.values())) != 1:
            raise CellFormatError(f"row index columns have different lengths: {lengths}")

    @property
    def n_rows(self) -> int:
        """How many stimulus rows this set holds."""
        return len(self.stimulus_ids)


@dataclass(frozen=True)
class CapturedCell:
    """One (arm, step) unit of cached activations, plus what it records about itself."""

    arm: str
    step: int
    identity: CellIdentity
    rows: dict[str, RowIndex]
    activations: dict[tuple[str, str], torch.Tensor]
    applied_adapter_weights: int | None
    adapter_weights_sha256: str | None
    provenance: dict[str, Any]
    source: Path
    stimulus_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    prefix_activations: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)
    prefix_stimulus_ids: tuple[str, ...] = ()
    natural_prefix_activations: dict[str, torch.Tensor] = field(default_factory=dict)
    natural_prefix_selection_manifest: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        """How a cell names itself in a log line or an error."""
        return f"{self.arm}/{STEP_DIR_PREFIX}{self.step}"

    @property
    def stimulus_sets(self) -> tuple[str, ...]:
        """The stimulus sets this cell holds, in a stable order."""
        return tuple(sorted(self.rows))

    @property
    def poolings(self) -> tuple[str, ...]:
        """The poolings this cell holds, in a stable order."""
        return tuple(sorted({pooling for _, pooling in self.activations}))

    def matrix(self, stimulus_set: str, pooling: str) -> torch.Tensor:
        """Return `[n_rows, n_layers, hidden]` float32 activations for one (set, pooling)."""
        key = (stimulus_set, pooling)
        if key not in self.activations:
            raise CellFormatError(
                f"{self.label} holds no activations for set {stimulus_set!r} pooling {pooling!r}; "
                f"it holds {sorted(self.activations)}."
            )
        return self.activations[key].float()

    def layer(self, stimulus_set: str, pooling: str, layer: int) -> torch.Tensor:
        """Return `[n_rows, hidden]` float32 activations at one decoder-block output."""
        if not 0 <= layer < self.identity.n_layers:
            raise CellFormatError(
                f"layer {layer} is outside this {self.identity.n_layers}-layer capture "
                f"({self.identity.layer_convention} convention)."
            )
        return self.matrix(stimulus_set, pooling)[:, layer, :]

    @property
    def metadata(self) -> dict[str, dict[str, Any]]:
        """Metadata keyed by stimulus id, including natural-prefix selection records."""
        return self.stimulus_metadata


@dataclass(frozen=True)
class NaturalPrefixCapture:
    """A reduced natural-prefix artifact stored beside, rather than inside, a construct cell."""

    arm: str | None
    step: int | None
    identity: CellIdentity
    stimulus_ids: tuple[str, ...]
    activations: dict[str, torch.Tensor]
    selection_manifest: dict[str, Any]
    selection_records: tuple[dict[str, Any], ...]
    source: Path


@dataclass(frozen=True)
class PairLayout:
    """Which stored rows are the two sides of each matched pair, in a stable pair order.

    Pairs are ordered by first appearance in the capture's row order, so the pair index is stable
    across cells (the row order is itself pinned by `assert_rows_align`) and usable both as a
    cross-validation group and as the odd/even split behind a held-out read.
    """

    pair_ids: tuple[str, ...]
    positive_rows: torch.Tensor
    negative_rows: torch.Tensor
    positive_side: str
    negative_side: str

    @property
    def n_pairs(self) -> int:
        """How many matched pairs this layout covers."""
        return len(self.pair_ids)

    def half(self, parity: int) -> torch.Tensor:
        """Pair indices of one deterministic half: parity 0 is even pairs, 1 is odd.

        Odd/even rather than random so the split is reproducible without a seed and any authored
        ordering (a leading block of pairs from one source, say) lands on both sides.
        """
        indices = torch.arange(self.n_pairs)
        return indices[indices % 2 == parity]


def sha256_of_file(path: Path) -> str:
    """Hash a file, so a manifest can name the exact corpus file a cell was captured from."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_of_strings(values: Iterable[str]) -> str:
    """Length-delimited sha256 over an ordered sequence of strings.

    Length-delimited so neither a reordering nor a boundary shift between two adjacent values can
    collide: without it, `"ab" + "c"` and `"a" + "bc"` hash the same.
    """
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(str(len(encoded)).encode())
        digest.update(b"\x00")
        digest.update(encoded)
    return digest.hexdigest()


def stimuli_digest(stimuli: Sequence[Stimulus]) -> str:
    """Digest the stimuli a capture actually ran, in order: the identity every later read joins on.

    Covers the id, the text *and* the assistant prefix, because on these sets the prefix is where the
    contrast lives -- a digest over the stem alone would call the two sides of a pair the same
    material and let an edited continuation pass unnoticed. Digesting the captured rows rather than
    the source file is what makes a truncated smoke run visibly a different corpus instead of a thin
    version of the same one.
    """
    values: list[str] = []
    for stimulus in stimuli:
        values.extend((stimulus.stimulus_id, stimulus.text, stimulus.assistant_prefix or ""))
        if stimulus.metadata:
            values.append(
                json.dumps(
                    stimulus.metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
            )
    return digest_of_strings(values)


def load_stimuli(path: Path) -> list[Stimulus]:
    """Read the stimulus corpus, refusing anything a later read could not join on.

    Every refusal here is a failure that would otherwise surface as a silently short or mislabelled
    capture rather than an error: a duplicate id makes two rows indistinguishable afterwards, a
    missing field leaves a hole no join reports, an empty text captures the chat template alone, and
    an unrecognised field means the corpus meant something by it that the capture would drop.
    """
    if not path.is_file():
        raise StimulusFileError(f"{path} is not a file, so there are no stimuli to capture.")
    stimuli: list[Stimulus] = []
    seen: dict[str, int] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = cast("dict[str, Any]", json.loads(line))
        missing = sorted(REQUIRED_STIMULUS_FIELDS - set(row))
        if missing:
            raise StimulusFileError(
                f"{path}:{line_number} is missing {missing}; every row needs "
                f"{sorted(REQUIRED_STIMULUS_FIELDS)}."
            )
        unknown = sorted(set(row) - REQUIRED_STIMULUS_FIELDS - OPTIONAL_STIMULUS_FIELDS)
        if unknown:
            raise StimulusFileError(
                f"{path}:{line_number} carries unrecognised fields {unknown}; a capture would drop "
                f"them, so a corpus that means something by them would lose it silently. Known "
                f"fields are {sorted(REQUIRED_STIMULUS_FIELDS | OPTIONAL_STIMULUS_FIELDS)}."
            )
        stimulus_id = str(row["id"])
        if stimulus_id in seen:
            raise StimulusFileError(
                f"{path}:{line_number} repeats id {stimulus_id!r}, first seen at line "
                f"{seen[stimulus_id]}."
            )
        if not str(row["text"]).strip():
            raise StimulusFileError(
                f"{path}:{line_number} has empty text, which would capture the template alone."
            )
        seen[stimulus_id] = line_number
        prefix = row.get("assistant_prefix")
        metadata = row.get(STIMULUS_METADATA_FIELD, {})
        if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
            raise StimulusFileError(
                f"{path}:{line_number} has non-object {STIMULUS_METADATA_FIELD!r}; capture "
                "metadata must be a JSON object keyed by strings."
            )
        stimuli.append(
            Stimulus(
                stimulus_id=stimulus_id,
                stimulus_set=str(row["set"]),
                side=str(row["side"]),
                pair_id=str(row["pair_id"]),
                text=str(row["text"]),
                assistant_prefix=None if prefix is None else str(prefix),
                metadata=metadata,
            )
        )
    if not stimuli:
        raise StimulusFileError(f"{path} holds no stimulus rows.")
    logger.info(
        f"stimuli loaded, path={path} n={len(stimuli)} "
        f"digest={short_digest(stimuli_digest(stimuli))}"
    )
    return stimuli


def group_by_set(stimuli: Sequence[Stimulus]) -> dict[str, list[Stimulus]]:
    """Group stimuli by their `set`, preserving corpus order inside each set."""
    grouped: dict[str, list[Stimulus]] = {}
    for stimulus in stimuli:
        grouped.setdefault(stimulus.stimulus_set, []).append(stimulus)
    return grouped


# These names are metadata labels only. The scenario wording is supplied at runtime from the
# gitignored cooperation-generalization corpus; no authored stimulus appears in this module.
COSTLY_OTHER_REGARD_CONSTRUCT = "costly-other-regard"
DECISION_DEPENDENCE_CONSTRUCT = "decision-dependence"
CONSTRUCT_NAMES: tuple[str, str] = (
    COSTLY_OTHER_REGARD_CONSTRUCT,
    DECISION_DEPENDENCE_CONSTRUCT,
)
DECISION_DEPENDENCE_STOCHASTICITY_CONTROLS = frozenset(
    {
        "shared-randomness",
        "independent-randomness",
        "shared-deterministic-procedure",
        "independent-deterministic-procedure",
    }
)
DECISION_DEPENDENCE_PROCEDURE_REGIMES = frozenset({"stochastic", "deterministic"})
DEFAULT_CONSTRUCT_PAIRS = 12
PAIR_MEMBER_COUNT = 2
PAIR_SIDES = ("A", "B")
CONSTRUCT_SPLITS = frozenset({"fit", "heldout"})
TENSOR_RANK = 3
PRIVATE_CONSTRUCT_ROOT = Path("docs/scratch/cooperation-generalization")
PRIVATE_ARTIFACT_ROOT = Path("artifacts")


def _private_construct_path(path: Path) -> Path:
    """Return ``path`` after requiring a location that is private in this checkout.

    The default scratch and artifact roots are stable, but callers may put an exact manifest copy
    under a configurable run root. Such copies are accepted only when Git itself reports the path
    ignored, so this loader does not turn a convenient absolute path into a privacy bypass.
    """
    resolved = path.resolve()
    allowed_roots = (PRIVATE_CONSTRUCT_ROOT.resolve(), PRIVATE_ARTIFACT_ROOT.resolve())
    under_known_private_root = any(resolved.is_relative_to(root) for root in allowed_roots)
    git_ignored = False
    if not under_known_private_root:
        ignored = subprocess.run(  # noqa: S603 - fixed git command; path is one argv value
            ["git", "check-ignore", "--no-index", "--quiet", str(resolved)],  # noqa: S607 - trusted literal command
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        git_ignored = ignored.returncode == 0
    if not under_known_private_root and not git_ignored:
        raise StimulusFileError(
            f"construct stimulus text must live under {PRIVATE_CONSTRUCT_ROOT} or "
            f"{PRIVATE_ARTIFACT_ROOT}, or be gitignored; got {path}. The corpus is private "
            "authored material and must remain gitignored."
        )
    return resolved


def construct_group(stimulus: Stimulus) -> str:
    """Read a scenario/template group from one stimulus's metadata.

    ``scenario_group`` is the only accepted key. Using a missing or aliased group would allow a
    scenario's two sides to cross a split while appearing to have valid metadata.
    """
    value = stimulus.metadata.get("scenario_group")
    if value is not None and str(value).strip():
        return str(value)
    raise StimulusFileError(
        f"stimulus {stimulus.stimulus_id!r} has no scenario/template group metadata; "
        "construct fit/held-out splits require a whole-scenario group."
    )


def _validate_construct_pair(
    construct: str,
    pair_id: str,
    members: Sequence[Stimulus],
    reserved_groups: set[str],
    external_pair_ids: set[str],
) -> str:
    """Validate one matched construct pair and return its scenario group."""
    if pair_id in external_pair_ids:
        raise StimulusFileError(
            f"construct pair {pair_id!r} overlaps a reserved training/evaluation identity."
        )
    sides = [member.side for member in members]
    if sorted(sides) != list(PAIR_SIDES) or len(members) != PAIR_MEMBER_COUNT:
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} must contain exactly one A and "
            f"one B row, got sides {sides}."
        )
    declared_constructs = {
        str(member.metadata.get("construct"))
        for member in members
        if member.metadata.get("construct") is not None
    }
    if declared_constructs != {construct}:
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} declares metadata constructs "
            f"{sorted(declared_constructs)}; every row must name its construct explicitly."
        )
    declared_splits = [member.metadata.get("split") for member in members]
    if (
        len(declared_splits) != PAIR_MEMBER_COUNT
        or not all(isinstance(split, str) for split in declared_splits)
        or len(set(declared_splits)) != 1
        or declared_splits[0] not in CONSTRUCT_SPLITS
    ):
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} must declare one shared metadata "
            f"split of 'fit' or 'heldout', got {declared_splits}."
        )
    if any(member.metadata.get("measurement_boundary") != "pre_action" for member in members):
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} must declare the pre_action "
            "measurement boundary on both rows."
        )
    if any(member.metadata.get("action_commitment_present") is not False for member in members):
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} includes an action commitment; "
            "construct captures must end before action commitment."
        )
    groups = {construct_group(member) for member in members}
    if len(groups) != 1:
        raise StimulusFileError(
            f"pair {pair_id!r} straddles scenario/template groups {sorted(groups)}; "
            "both sides must stay in one fit or held-out split."
        )
    group = groups.pop()
    if group in reserved_groups:
        raise StimulusFileError(
            f"construct group {group!r} for pair {pair_id!r} overlaps a reserved "
            "training/evaluation group."
        )
    return group


def _validate_decision_control(construct: str, pair_id: str, members: Sequence[Stimulus]) -> None:
    """Validate pair-constant procedure regime and side-specific procedure controls."""
    controls = {member.side: member.metadata.get("dependence_mechanism") for member in members}
    if any(
        control not in DECISION_DEPENDENCE_STOCHASTICITY_CONTROLS for control in controls.values()
    ):
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} must declare a supported "
            f"dependence_mechanism on both rows, got {controls}."
        )
    regimes = {member.metadata.get("procedure_regime") for member in members}
    if len(regimes) != 1 or regimes - DECISION_DEPENDENCE_PROCEDURE_REGIMES:
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} must declare one supported "
            f"procedure_regime on both rows, got {sorted(regimes, key=str)}."
        )
    control_a = controls["A"]
    control_b = controls["B"]
    if not isinstance(control_a, str) or not isinstance(control_b, str):
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} must declare string "
            f"dependence mechanisms, got {controls}."
        )
    if control_a.startswith("independent-") or not control_b.startswith("independent-"):
        raise StimulusFileError(
            f"pair {pair_id!r} in construct {construct!r} must put the coupled "
            f"control on side A and the independent control on side B, got {controls}."
        )


def validate_construct_stimuli(  # noqa: PLR0913 - callers provide independent corpus isolation controls
    stimuli: Sequence[Stimulus],
    *,
    pairs_per_construct: int = DEFAULT_CONSTRUCT_PAIRS,
    required_constructs: Sequence[str] = CONSTRUCT_NAMES,
    reserved_groups: Sequence[str] = (),
    external_pair_ids: Sequence[str] = (),
    require_decision_control: bool = False,
) -> dict[str, dict[str, str]]:
    """Validate the private two-construct corpus and return pair-to-group mappings.

    This is deliberately pure and accepts in-memory stimuli so CPU tests do not depend on the
    gitignored runtime file existing. ``reserved_groups`` and ``external_pair_ids`` are the global
    isolation check: interp calibration/construct scenarios cannot overlap training or evaluation
    identities merely because each local split is internally disjoint.
    """
    if pairs_per_construct < PAIR_MEMBER_COUNT:
        raise ValueError(f"pairs_per_construct must be at least 2, got {pairs_per_construct}")
    required = tuple(required_constructs)
    if len(set(required)) != len(required) or not required:
        raise ValueError(f"required_constructs must be unique and non-empty, got {required}")
    by_construct: dict[str, dict[str, list[Stimulus]]] = {}
    for stimulus in stimuli:
        if stimulus.stimulus_set not in required:
            raise StimulusFileError(
                f"stimulus {stimulus.stimulus_id!r} has construct/set {stimulus.stimulus_set!r}; "
                f"expected exactly {list(required)}."
            )
        by_construct.setdefault(stimulus.stimulus_set, {}).setdefault(stimulus.pair_id, []).append(
            stimulus
        )
    missing_constructs = [name for name in required if name not in by_construct]
    if missing_constructs:
        raise StimulusFileError(f"construct corpus is missing {missing_constructs}.")
    reserved = set(reserved_groups)
    external_pairs = set(external_pair_ids)
    result: dict[str, dict[str, str]] = {}
    for construct in required:
        pairs = by_construct[construct]
        if len(pairs) != pairs_per_construct:
            raise StimulusFileError(
                f"construct {construct!r} has {len(pairs)} matched pairs; expected "
                f"{pairs_per_construct}."
            )
        mapping: dict[str, str] = {}
        for pair_id, members in pairs.items():
            mapping[pair_id] = _validate_construct_pair(
                construct, pair_id, members, reserved, external_pairs
            )
            if construct == DECISION_DEPENDENCE_CONSTRUCT and require_decision_control:
                _validate_decision_control(construct, pair_id, members)
        result[construct] = mapping
    return result


def load_private_construct_stimuli(
    path: Path | None = None,
    *,
    pairs_per_construct: int = DEFAULT_CONSTRUCT_PAIRS,
    reserved_groups: Sequence[str] = (),
    external_pair_ids: Sequence[str] = (),
) -> list[Stimulus]:
    """Load and validate the runtime private two-construct capture corpus.

    The path check is intentional: a tracked path would make future capture text public and
    invalidate the measurement. Tests should call :func:`validate_construct_stimuli` with synthetic
    rows instead of bypassing this runtime boundary.
    """
    corpus_path = _private_construct_path(
        PRIVATE_CONSTRUCT_ROOT / "construct-stimuli.jsonl" if path is None else path
    )
    stimuli = load_stimuli(corpus_path)
    validate_construct_stimuli(
        stimuli,
        pairs_per_construct=pairs_per_construct,
        reserved_groups=reserved_groups,
        external_pair_ids=external_pair_ids,
        require_decision_control=True,
    )
    return stimuli


def construct_split_payload(stimuli: Sequence[Stimulus]) -> dict[str, dict[str, Any]]:
    """Return the authored fit/held-out membership for a validated construct corpus."""
    pair_groups = validate_construct_stimuli(stimuli)
    result: dict[str, dict[str, Any]] = {}
    for construct, mapping in pair_groups.items():
        split_by_group: dict[str, str] = {}
        split_by_pair: dict[str, str] = {}
        for stimulus in stimuli:
            if stimulus.stimulus_set != construct:
                continue
            group = construct_group(stimulus)
            split = str(stimulus.metadata["split"])
            prior = split_by_group.setdefault(group, split)
            if prior != split:
                raise StimulusFileError(
                    f"scenario group {group!r} declares both {prior!r} and {split!r}"
                )
            split_by_pair[stimulus.pair_id] = split
        result[construct] = {
            "fit_pair_ids": sorted(
                pair_id for pair_id, split in split_by_pair.items() if split == "fit"
            ),
            "heldout_pair_ids": sorted(
                pair_id for pair_id, split in split_by_pair.items() if split == "heldout"
            ),
            "fit_groups": sorted(
                group for group, split in split_by_group.items() if split == "fit"
            ),
            "heldout_groups": sorted(
                group for group, split in split_by_group.items() if split == "heldout"
            ),
            "group_by_pair": dict(sorted(mapping.items())),
        }
    return result


def tensor_key(stimulus_set: str, pooling: str) -> str:
    """Return the safetensors key one (set, pooling) activation block is stored under."""
    return f"{stimulus_set}/{pooling}"


def parse_tensor_key(key: str) -> tuple[str, str] | None:
    """Split a stored key back into (set, pooling), or None for a key that is not an activation."""
    stimulus_set, separator, pooling = key.rpartition("/")
    if not separator or pooling not in POOLERS:
        return None
    return stimulus_set, pooling


def step_dir(root: Path, arm: str, step: int) -> Path:
    """Where one cell's files live, under a capture root."""
    return root / arm / f"{STEP_DIR_PREFIX}{step}"


def _validate_cell_metadata(
    rows: dict[str, RowIndex],
    stimulus_metadata: dict[str, dict[str, Any]] | None,
    prefix_activations: dict[tuple[str, str], torch.Tensor] | None,
    prefix_stimulus_ids: Sequence[str] | None,
) -> tuple[
    dict[str, dict[str, Any]], dict[tuple[str, str], torch.Tensor], tuple[str, ...], set[str]
]:
    """Validate metadata and the optional prefix row order before any file mutation."""
    metadata = dict(stimulus_metadata or {})
    row_ids = {stimulus_id for row in rows.values() for stimulus_id in row.stimulus_ids}
    unknown_metadata = sorted(set(metadata) - row_ids)
    if unknown_metadata:
        raise CellFormatError(
            f"stimulus metadata names rows absent from the cell {unknown_metadata[:ERROR_EXAMPLE_COUNT]}; "
            "metadata must describe only captured stimuli."
        )
    try:
        json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    except TypeError as error:
        raise CellFormatError(f"stimulus metadata is not JSON serialisable: {error}") from error
    prefix = dict(prefix_activations or {})
    prefix_ids = tuple(prefix_stimulus_ids or ())
    if prefix and set(prefix_ids) != row_ids:
        raise CellFormatError(
            f"prefix_stimulus_ids must cover exactly the cell rows, got {len(prefix_ids)} ids for "
            f"{len(row_ids)} rows"
        )
    if len(set(prefix_ids)) != len(prefix_ids):
        raise CellFormatError("prefix_stimulus_ids repeats a stimulus id")
    return metadata, prefix, prefix_ids, row_ids


def _store_main_activations(
    activations: dict[tuple[str, str], torch.Tensor], identity: CellIdentity
) -> dict[str, torch.Tensor]:
    """Cast ordinary activation blocks and refuse overflow before writing."""
    store_dtype = STORE_DTYPES[identity.store_dtype]
    stored: dict[str, torch.Tensor] = {}
    for (stimulus_set, pooling), tensor in sorted(activations.items()):
        cast_tensor = tensor.to(store_dtype)
        if not bool(torch.isfinite(cast_tensor).all()):
            raise CellFormatError(
                f"casting {stimulus_set}/{pooling} to {identity.store_dtype} produced non-finite "
                f"values (peak magnitude {float(tensor.abs().max()):.1f}). Residual-stream norms "
                f"grow with depth and float16 tops out at 65504, which maps to inf without a word "
                f"and poisons every direction computed from it. Store float32 instead."
            )
        stored[tensor_key(stimulus_set, pooling)] = cast_tensor
    return stored


def _store_supplied_prefix_activations(
    prefix: dict[tuple[str, str], torch.Tensor],
    row_count: int,
    identity: CellIdentity,
) -> dict[str, torch.Tensor]:
    """Validate and cast prompt-end and teacher-forced activation blocks."""
    stored: dict[str, torch.Tensor] = {}
    expected = (row_count, identity.n_layers, identity.hidden_size)
    for (state, pooling), tensor in sorted(prefix.items()):
        if tuple(tensor.shape) != expected:
            raise CellFormatError(
                f"prefix state {state!r}/{pooling!r} has shape {tuple(tensor.shape)}, expected "
                f"{expected} for this cell"
            )
        cast_tensor = tensor.to(STORE_DTYPES[identity.store_dtype])
        if not bool(torch.isfinite(cast_tensor).all()):
            raise CellFormatError(
                f"casting prefix state {state!r}/{pooling!r} to {identity.store_dtype} produced "
                "non-finite values"
            )
        stored[f"{state}/{pooling}"] = cast_tensor
    return stored


def _store_natural_prefix_activations(
    natural: dict[str, torch.Tensor], row_ids: set[str], identity: CellIdentity
) -> dict[str, torch.Tensor]:
    """Validate and cast reduced natural-prefix activation blocks."""
    if natural and not identity.natural_prefix_layers:
        raise CellFormatError(
            "natural-prefix activations require an explicit natural_prefix_layers identity"
        )
    stored: dict[str, torch.Tensor] = {}
    expected_tail = (len(identity.natural_prefix_layers), identity.hidden_size)
    for stimulus_id, tensor in sorted(natural.items()):
        if stimulus_id not in row_ids:
            raise CellFormatError(
                f"natural-prefix activation {stimulus_id!r} names a row absent from the cell"
            )
        if tensor.ndim != TENSOR_RANK or tuple(tensor.shape[1:]) != expected_tail:
            raise CellFormatError(
                f"natural-prefix activation {stimulus_id!r} has shape {tuple(tensor.shape)}; "
                f"expected [selected_positions, {len(identity.natural_prefix_layers)}, "
                f"{identity.hidden_size}]"
            )
        cast_tensor = tensor.to(STORE_DTYPES[identity.store_dtype])
        if not bool(torch.isfinite(cast_tensor).all()):
            raise CellFormatError(
                f"casting natural-prefix activation {stimulus_id!r} to {identity.store_dtype} "
                "produced non-finite values"
            )
        stored[stimulus_id] = cast_tensor
    return stored


def write_cell(  # noqa: PLR0913 - a cell is its identity, its rows, its tensors and its provenance
    cell_dir: Path,
    *,
    arm: str,
    step: int,
    identity: CellIdentity,
    rows: dict[str, RowIndex],
    activations: dict[tuple[str, str], torch.Tensor],
    applied_adapter_weights: int | None,
    adapter_weights_sha256: str | None,
    provenance: dict[str, Any],
    stimulus_metadata: dict[str, dict[str, Any]] | None = None,
    prefix_activations: dict[tuple[str, str], torch.Tensor] | None = None,
    prefix_stimulus_ids: Sequence[str] | None = None,
    natural_prefix_activations: dict[str, torch.Tensor] | None = None,
    natural_prefix_selection_manifest: dict[str, Any] | None = None,
) -> Path:
    """Write one cell's tensors and manifest, and return the manifest path.

    The manifest is written last, so a cell interrupted mid-write is not mistaken for a finished one
    by a resumed run: the marker of "done" is the file that describes the work, not the work.
    """
    metadata, prefix, prefix_ids, row_ids = _validate_cell_metadata(
        rows, stimulus_metadata, prefix_activations, prefix_stimulus_ids
    )
    natural = dict(natural_prefix_activations or {})
    stored = _store_main_activations(activations, identity)
    cell_dir.mkdir(parents=True, exist_ok=True)
    save_file(stored, str(cell_dir / ACTIVATIONS_FILENAME))
    prefix_stored = _store_supplied_prefix_activations(prefix, len(row_ids), identity)
    if prefix_stored:
        save_file(prefix_stored, str(cell_dir / PREFIX_ACTIVATIONS_FILENAME))
    natural_stored = _store_natural_prefix_activations(natural, row_ids, identity)
    if natural_stored:
        save_file(natural_stored, str(cell_dir / NATURAL_PREFIX_ACTIVATIONS_FILENAME))
    manifest: dict[str, Any] = {
        "arm": arm,
        "step": step,
        "identity": identity.to_payload(),
        "sets": {
            stimulus_set: {
                "stimulus_ids": list(row_index.stimulus_ids),
                "sides": list(row_index.sides),
                "pair_ids": list(row_index.pair_ids),
                "token_counts": list(row_index.token_counts),
                "shapes": {
                    pooling: list(activations[stimulus_set, pooling].shape)
                    for pooling in sorted(
                        pooling for held_set, pooling in activations if held_set == stimulus_set
                    )
                },
            }
            for stimulus_set, row_index in sorted(rows.items())
        },
        "applied_adapter_weights": applied_adapter_weights,
        "adapter_weights_sha256": adapter_weights_sha256,
        "provenance": provenance,
        "stimulus_metadata": metadata,
        "prefix_states": {
            state: {
                pooling: list(prefix[state, pooling].shape)
                for held_state, pooling in sorted(prefix)
                if held_state == state
            }
            for state in sorted({state for state, _ in prefix})
        },
        "prefix_stimulus_ids": list(prefix_ids),
        "natural_prefix_states": {
            stimulus_id: list(tensor.shape) for stimulus_id, tensor in sorted(natural.items())
        },
        "natural_prefix_selection_manifest": natural_prefix_selection_manifest,
    }
    manifest_path = cell_dir / CELL_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def write_natural_prefix_capture(  # noqa: C901, PLR0912, PLR0913 - linear durable artifact boundary
    capture_dir: Path,
    *,
    identity: CellIdentity,
    stimuli: Sequence[Stimulus],
    activations: dict[str, torch.Tensor],
    selection_records: Sequence[dict[str, Any]],
    selection_manifest: dict[str, Any],
    rendered_sha256: str,
    natural_state: str,
    arm: str | None = None,
    step: int | None = None,
    applied_adapter_weights: int | None = None,
    adapter_weights_sha256: str | None = None,
) -> Path:
    """Persist a reduced natural-prefix capture outside the matched construct cell.

    Natural rollout rows are a separate corpus: they do not have A/B construct pairs and therefore
    cannot be inserted into the construct cell's row index. Only selected positions and their
    metadata are retained; full rollout text remains in its own private source artifact.
    """
    if not identity.natural_prefix_layers:
        raise CellFormatError("natural-prefix capture identity has no explicit layer subset")
    if not rendered_sha256.strip():
        raise CellFormatError("natural-prefix capture requires a non-empty rendered corpus digest")
    if not identity.natural_selection_manifest_sha256:
        raise CellFormatError(
            "natural-prefix capture identity must record the predeclared selection manifest digest"
        )
    selection_digest = selection_manifest.get("sha256")
    if selection_digest != identity.natural_selection_manifest_sha256:
        raise CellFormatError(
            "natural-prefix selection manifest digest does not match the capture identity"
        )
    if tuple(selection_manifest.get("layers", ())) != identity.natural_prefix_layers:
        raise CellFormatError(
            "natural-prefix selection manifest layers do not match the capture identity"
        )
    if natural_state not in {"base", "final"}:
        raise CellFormatError(f"natural-prefix capture has unknown natural_state {natural_state!r}")
    expected_rollout_id = selection_manifest["rollout_ids_by_state"][natural_state]
    stimulus_ids = tuple(stimulus.stimulus_id for stimulus in stimuli)
    if set(activations) != set(stimulus_ids):
        raise CellFormatError("natural-prefix activations must cover exactly the supplied stimuli")
    if len(set(stimulus_ids)) != len(stimulus_ids):
        raise CellFormatError("natural-prefix stimuli repeat an id")
    records_by_id = {str(record["stimulus_id"]): record for record in selection_records}
    if set(records_by_id) != set(stimulus_ids):
        raise CellFormatError(
            "natural-prefix selection records must cover exactly the supplied stimuli"
        )
    for stimulus in stimuli:
        if stimulus.metadata.get("natural_state") != natural_state:
            raise CellFormatError(
                f"natural-prefix stimulus {stimulus.stimulus_id!r} has state "
                f"{stimulus.metadata.get('natural_state')!r}, expected {natural_state!r}"
            )
        if stimulus.metadata.get("rollout_id") != expected_rollout_id:
            raise CellFormatError(
                f"natural-prefix stimulus {stimulus.stimulus_id!r} has rollout "
                f"{stimulus.metadata.get('rollout_id')!r}, expected {expected_rollout_id!r}"
            )
    if any(record.get("natural_state") != natural_state for record in selection_records):
        raise CellFormatError("natural-prefix selection records do not share the capture state")
    try:
        json.dumps(selection_manifest, sort_keys=True, ensure_ascii=False)
        json.dumps(selection_records, sort_keys=True, ensure_ascii=False)
    except TypeError as error:
        raise CellFormatError(
            f"natural-prefix selection metadata is not JSON serialisable: {error}"
        ) from error
    stored: dict[str, torch.Tensor] = {}
    for stimulus_id in stimulus_ids:
        tensor = activations[stimulus_id]
        expected_tail = (len(identity.natural_prefix_layers), identity.hidden_size)
        if tensor.ndim != TENSOR_RANK or tuple(tensor.shape[1:]) != expected_tail:
            raise CellFormatError(
                f"natural-prefix activation {stimulus_id!r} has shape {tuple(tensor.shape)}, "
                f"expected [selected_positions, {len(identity.natural_prefix_layers)}, "
                f"{identity.hidden_size}]"
            )
        cast_tensor = tensor.to(STORE_DTYPES[identity.store_dtype])
        if not bool(torch.isfinite(cast_tensor).all()):
            raise CellFormatError(
                f"natural-prefix activation {stimulus_id!r} contains non-finite values"
            )
        stored[stimulus_id] = cast_tensor
    capture_dir.mkdir(parents=True, exist_ok=True)
    save_file(stored, str(capture_dir / NATURAL_PREFIX_ACTIVATIONS_FILENAME))
    manifest = {
        "arm": arm,
        "step": step,
        "applied_adapter_weights": applied_adapter_weights,
        "adapter_weights_sha256": adapter_weights_sha256,
        "identity": identity.to_payload(),
        "stimulus_ids": list(stimulus_ids),
        "stimuli_sha256": stimuli_digest(stimuli),
        "rendered_sha256": rendered_sha256,
        "natural_state": natural_state,
        "selection_manifest": selection_manifest,
        "selection_records": list(selection_records),
        "states": {
            stimulus_id: list(tensor.shape) for stimulus_id, tensor in sorted(stored.items())
        },
    }
    manifest_path = capture_dir / NATURAL_PREFIX_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _load_rows(manifest: dict[str, Any]) -> dict[str, RowIndex]:
    """Decode row identity metadata from a cell manifest."""
    return {
        stimulus_set: RowIndex(
            stimulus_ids=tuple(entry["stimulus_ids"]),
            sides=tuple(entry["sides"]),
            pair_ids=tuple(entry["pair_ids"]),
            token_counts=tuple(int(count) for count in entry["token_counts"]),
        )
        for stimulus_set, entry in cast("dict[str, Any]", manifest["sets"]).items()
    }


def _load_main_activations(
    cell_dir: Path, rows: dict[str, RowIndex], identity: CellIdentity
) -> dict[tuple[str, str], torch.Tensor]:
    """Load ordinary activations and verify every stored block shape."""
    activations: dict[tuple[str, str], torch.Tensor] = {}
    activation_path = cell_dir / ACTIVATIONS_FILENAME
    for key, tensor in load_file(str(activation_path)).items():
        parsed = parse_tensor_key(key)
        if parsed is None:
            raise CellFormatError(
                f"{activation_path} holds key {key!r}, which names no known pooling; expected "
                f"'<set>/<pooling>' with pooling in {sorted(POOLERS)}."
            )
        stimulus_set, pooling = parsed
        if stimulus_set not in rows:
            raise CellFormatError(
                f"{cell_dir} holds activations for set {stimulus_set!r} that its manifest does not "
                "describe, so their rows cannot be identified."
            )
        expected = (rows[stimulus_set].n_rows, identity.n_layers, identity.hidden_size)
        if tuple(tensor.shape) != expected:
            raise CellFormatError(
                f"{cell_dir} set {stimulus_set!r} pooling {pooling!r} is {tuple(tensor.shape)} on "
                f"disk but its manifest describes {expected}. The manifest and the tensors were "
                "written by different runs."
            )
        activations[parsed] = tensor
    return activations


def _load_prefix_activations(
    cell_dir: Path, rows: dict[str, RowIndex], identity: CellIdentity
) -> dict[tuple[str, str], torch.Tensor]:
    """Load optional supplied-prefix states and verify their shared row order and shape."""
    prefix_path = cell_dir / PREFIX_ACTIVATIONS_FILENAME
    if not prefix_path.is_file():
        return {}
    expected = (sum(row.n_rows for row in rows.values()), identity.n_layers, identity.hidden_size)
    prefix_activations: dict[tuple[str, str], torch.Tensor] = {}
    for key, tensor in load_file(str(prefix_path)).items():
        state, separator, pooling = key.partition("/")
        if not separator or not state or pooling not in POOLERS:
            raise CellFormatError(
                f"{prefix_path} holds key {key!r}, expected '<state>/<pooling>' with a known pooling"
            )
        if tuple(tensor.shape) != expected:
            raise CellFormatError(
                f"{prefix_path} state {state!r}/{pooling!r} has shape {tuple(tensor.shape)}, "
                f"expected {expected}"
            )
        prefix_activations[state, pooling] = tensor
    return prefix_activations


def _load_natural_prefix_activations(
    cell_dir: Path, rows: dict[str, RowIndex], identity: CellIdentity
) -> dict[str, torch.Tensor]:
    """Load reduced natural-prefix states and verify their selected-position axis."""
    natural_path = cell_dir / NATURAL_PREFIX_ACTIVATIONS_FILENAME
    if not natural_path.is_file():
        return {}
    if not identity.natural_prefix_layers:
        raise CellFormatError(
            f"{natural_path} contains natural activations but identity records no layer subset"
        )
    row_ids = {stimulus_id for row in rows.values() for stimulus_id in row.stimulus_ids}
    expected_tail = (len(identity.natural_prefix_layers), identity.hidden_size)
    natural_prefix_activations: dict[str, torch.Tensor] = {}
    for stimulus_id, tensor in load_file(str(natural_path)).items():
        if stimulus_id not in row_ids:
            raise CellFormatError(
                f"{natural_path} holds {stimulus_id!r}, which names no cell stimulus"
            )
        if tensor.ndim != TENSOR_RANK or tuple(tensor.shape[1:]) != expected_tail:
            raise CellFormatError(
                f"{natural_path} stimulus {stimulus_id!r} has shape {tuple(tensor.shape)}, "
                f"expected [selected_positions, {len(identity.natural_prefix_layers)}, "
                f"{identity.hidden_size}]"
            )
        natural_prefix_activations[stimulus_id] = tensor
    return natural_prefix_activations


def _validate_auxiliary_capture_files(
    cell_dir: Path,
    identity: CellIdentity,
    prefix_activations: dict[tuple[str, str], torch.Tensor],
    natural_prefix_activations: dict[str, torch.Tensor],
) -> None:
    """Require auxiliary files promised by the identity and validate their required states."""
    prefix_path = cell_dir / PREFIX_ACTIVATIONS_FILENAME
    if identity.capture_prefix_states:
        required_prefix_states = {"prompt_end", "teacher_forced"}
        found_prefix_states = {state for state, _ in prefix_activations}
        if not prefix_path.is_file():
            raise CellFormatError(
                f"{cell_dir} identity requires supplied-prefix states, but {prefix_path} is missing"
            )
        if not required_prefix_states <= found_prefix_states:
            raise CellFormatError(
                f"{cell_dir} supplied-prefix capture lacks states "
                f"{sorted(required_prefix_states - found_prefix_states)}"
            )
    natural_path = cell_dir / NATURAL_PREFIX_ACTIVATIONS_FILENAME
    if identity.natural_prefix_layers and not natural_prefix_activations:
        separate_manifest = cell_dir / "natural-prefix" / NATURAL_PREFIX_MANIFEST_FILENAME
        if not natural_path.is_file() and not separate_manifest.is_file():
            raise CellFormatError(
                f"{cell_dir} identity requires natural-prefix capture, but no natural prefix "
                f"artifact was found at {natural_path} or {separate_manifest}"
            )


def read_cell(cell_dir: Path) -> CapturedCell:
    """Read one cell back, checking the tensors on disk against what the manifest claims."""
    manifest_path = cell_dir / CELL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CellFormatError(f"{manifest_path} not found, so {cell_dir} is not a finished cell.")
    manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text()))
    identity = CellIdentity.from_payload(
        cast("dict[str, Any]", manifest.get("identity", {})), source=manifest_path
    )
    rows = _load_rows(manifest)
    activations = _load_main_activations(cell_dir, rows, identity)
    prefix_activations = _load_prefix_activations(cell_dir, rows, identity)
    prefix_stimulus_ids = tuple(cast("list[str]", manifest.get("prefix_stimulus_ids", [])))
    row_ids = tuple(stimulus_id for row in rows.values() for stimulus_id in row.stimulus_ids)
    if prefix_stimulus_ids and len(set(prefix_stimulus_ids)) != len(prefix_stimulus_ids):
        raise CellFormatError(f"{manifest_path} repeats ids in prefix_stimulus_ids")
    if prefix_stimulus_ids and set(prefix_stimulus_ids) != set(row_ids):
        raise CellFormatError(
            f"{manifest_path} prefix_stimulus_ids do not cover exactly the cell rows"
        )
    natural_prefix_activations = _load_natural_prefix_activations(cell_dir, rows, identity)
    _validate_auxiliary_capture_files(
        cell_dir, identity, prefix_activations, natural_prefix_activations
    )
    return CapturedCell(
        arm=str(manifest["arm"]),
        step=int(manifest["step"]),
        identity=identity,
        rows=rows,
        activations=activations,
        applied_adapter_weights=cast("int | None", manifest.get("applied_adapter_weights")),
        adapter_weights_sha256=cast("str | None", manifest.get("adapter_weights_sha256")),
        provenance=cast("dict[str, Any]", manifest.get("provenance", {})),
        source=cell_dir,
        stimulus_metadata=cast("dict[str, dict[str, Any]]", manifest.get("stimulus_metadata", {})),
        prefix_activations=prefix_activations,
        prefix_stimulus_ids=prefix_stimulus_ids,
        natural_prefix_activations=natural_prefix_activations,
        natural_prefix_selection_manifest=cast(
            "dict[str, Any] | None", manifest.get("natural_prefix_selection_manifest")
        ),
    )


def find_cells(root: Path) -> list[Path]:
    """Every finished cell directory under a capture root, in (arm, step) order."""
    if not root.is_dir():
        raise CellFormatError(f"capture root {root} is not a directory.")
    found: list[tuple[str, int, Path]] = []
    for manifest_path in root.glob(f"*/{STEP_DIR_PREFIX}*/{CELL_MANIFEST_FILENAME}"):
        cell_dir = manifest_path.parent
        step_name = cell_dir.name.removeprefix(STEP_DIR_PREFIX)
        if not step_name.isdigit():
            raise CellFormatError(
                f"{cell_dir} is not named '{STEP_DIR_PREFIX}<step>', so its checkpoint step cannot "
                f"be read and a ladder built from it would be in an arbitrary order."
            )
        found.append((cell_dir.parent.name, int(step_name), cell_dir))
    return [cell_dir for _, _, cell_dir in sorted(found)]


@dataclass(frozen=True)
class Ladder:
    """A comparable set of cells: the same stimuli, the same measured space, different checkpoints."""

    cells: tuple[CapturedCell, ...]

    @property
    def identity(self) -> CellIdentity:
        """The one identity every cell in this ladder shares."""
        return self.cells[0].identity

    @property
    def arms(self) -> tuple[str, ...]:
        """The arms present, in a stable order."""
        return tuple(sorted({cell.arm for cell in self.cells}))

    @property
    def stimulus_sets(self) -> tuple[str, ...]:
        """The stimulus sets present in every cell."""
        return self.cells[0].stimulus_sets

    @property
    def poolings(self) -> tuple[str, ...]:
        """The poolings present in every cell."""
        return self.cells[0].poolings

    def arm_cells(self, arm: str) -> tuple[CapturedCell, ...]:
        """One arm's cells in training-step order."""
        return tuple(sorted((cell for cell in self.cells if cell.arm == arm), key=lambda c: c.step))

    def cell(self, arm: str, step: int) -> CapturedCell:
        """One named cell."""
        for candidate in self.cells:
            if candidate.arm == arm and candidate.step == step:
                return candidate
        raise CellFormatError(
            f"no cell {arm}/{STEP_DIR_PREFIX}{step} in this ladder; it holds "
            f"{[cell.label for cell in self.cells]}."
        )


def assert_ladder_identity(cells: Sequence[CapturedCell]) -> None:
    """Raise unless every cell measured the same thing, reporting every field that differs.

    All discrepancies at once, not the first: a ladder assembled from two capture passes usually
    differs in several fields, and fixing them one error at a time means one re-read per field.
    """
    if not cells:
        raise CellFormatError("no cells to compare; a ladder of nothing has no identity.")
    reference = cells[0]
    differing: dict[str, dict[str, Any]] = {}
    for cell in cells[1:]:
        for identity_field in CellIdentity.__dataclass_fields__:
            mine = getattr(cell.identity, identity_field)
            theirs = getattr(reference.identity, identity_field)
            if mine != theirs:
                differing.setdefault(identity_field, {reference.label: theirs})[cell.label] = mine
    if differing:
        raise CellFormatError(
            f"these cells did not measure the same thing, so comparing them would read a change in "
            f"the apparatus as a change in the model: {json.dumps(differing, sort_keys=True)}. "
            f"Every field here moves the activations by at least as much as a trained delta does."
        )


def assert_capture_provenance(cells: Sequence[CapturedCell]) -> None:
    """Require concrete tokenizer and kernel identities for a cooperation experiment read."""
    unspecified = [
        f"{cell.label}:{field}"
        for cell in cells
        for field in ("tokenizer_identity", "kernel_identity")
        if getattr(cell.identity, field) == "unspecified"
    ]
    if unspecified:
        raise CellFormatError(
            f"capture cells lack concrete tokenizer/kernel provenance ({unspecified[:ERROR_EXAMPLE_COUNT]}). "
            "The cooperation capture path must record both identities before its activations are read."
        )


def assert_stimuli_match(cells: Sequence[CapturedCell], digest: str) -> None:
    """Raise unless every cell was captured on the corpus `digest` names.

    The guard between a stimulus file and the tensors an analysis is about to read as if they came
    from it. Editing, extending or truncating a corpus after a capture leaves the tensors keyed to
    the old rows while every id still resolves, so the numbers stay plausible and answer a question
    about text that is no longer there.
    """
    mismatched = {
        cell.label: cell.identity.stimuli_sha256
        for cell in cells
        if cell.identity.stimuli_sha256 != digest
    }
    if mismatched:
        raise CellFormatError(
            f"cells were captured on a different stimulus corpus than the one supplied "
            f"(digest {short_digest(digest)}): {json.dumps(mismatched, sort_keys=True)}. Re-capture, or "
            f"analyse against the corpus these cells actually ran; the row ids would resolve either "
            f"way and the mismatch would only show up as a result about text that changed."
        )


def assert_rows_align(cells: Sequence[CapturedCell]) -> None:
    """Raise unless every cell holds the same stimulus sets, with the same rows in the same order.

    Every read here is row-wise: a direction is a difference of row means, a projection is indexed
    by row, and a matched pair is two rows. A cell with an extra row, a missing row or the same rows
    in a different order produces a number rather than an error.
    """
    reference = cells[0]
    for cell in cells[1:]:
        if cell.stimulus_sets != reference.stimulus_sets:
            raise CellFormatError(
                f"{cell.label} holds sets {list(cell.stimulus_sets)} but {reference.label} holds "
                f"{list(reference.stimulus_sets)}."
            )
        if cell.poolings != reference.poolings:
            raise CellFormatError(
                f"{cell.label} holds poolings {list(cell.poolings)} but {reference.label} holds "
                f"{list(reference.poolings)}; a read present on one side of a comparison and absent "
                f"on the other is a narrower ladder than it looks."
            )
        for stimulus_set in reference.stimulus_sets:
            mine, theirs = cell.rows[stimulus_set], reference.rows[stimulus_set]
            if mine.stimulus_ids != theirs.stimulus_ids:
                only_mine = sorted(set(mine.stimulus_ids) - set(theirs.stimulus_ids))
                only_theirs = sorted(set(theirs.stimulus_ids) - set(mine.stimulus_ids))
                detail = (
                    f"only in {cell.label}: {only_mine[:ERROR_EXAMPLE_COUNT]} "
                    f"({len(only_mine)}); only in {reference.label}: "
                    f"{only_theirs[:ERROR_EXAMPLE_COUNT]} ({len(only_theirs)})"
                    if only_mine or only_theirs
                    else "same ids in a different order"
                )
                raise CellFormatError(
                    f"set {stimulus_set!r} does not hold the same rows in {cell.label} "
                    f"({mine.n_rows}) and {reference.label} ({theirs.n_rows}): {detail}."
                )
            if (mine.sides, mine.pair_ids) != (theirs.sides, theirs.pair_ids):
                raise CellFormatError(
                    f"set {stimulus_set!r} agrees on ids but not on their sides or pair ids between "
                    f"{cell.label} and {reference.label}, so the matched pairing differs."
                )


def assert_prefix_rows_align(cell: CapturedCell, stimulus_ids: Sequence[str]) -> None:
    """Refuse supplied-prefix tensors whose recorded row order differs from the source corpus."""
    expected = tuple(stimulus_ids)
    if cell.prefix_stimulus_ids != expected:
        raise CellFormatError(
            f"{cell.label} prefix activations record row order {cell.prefix_stimulus_ids}, "
            f"but the supplied corpus requires {expected}; shuffled prefix rows would pair the "
            "wrong activation with a construct stimulus."
        )


def assert_adapter_moved_activations(cells: Sequence[CapturedCell]) -> None:
    """Raise if an adapted cell's activations are bit-identical to the base cell's.

    The cheapest end-to-end check that a checkpoint's adapter reached the forward pass. It catches
    what `games.lora.assert_adapter_applies` cannot: an adapter that loaded onto a model object the
    capture then did not drive, which yields pristine base activations under a trained checkpoint's
    name and no complaint anywhere. Skipped, with a log line, when no base cell is present -- a
    ladder analysed without its anchor is a legitimate partial read, not a failure.
    """
    base = next((cell for cell in cells if cell.arm == BASE_ARM and cell.step == BASE_STEP), None)
    if base is None:
        logger.info(
            f"no {BASE_ARM}/{STEP_DIR_PREFIX}{BASE_STEP} cell present, so the adapter-moved-the-"
            f"activations check is not available for this ladder"
        )
        return
    inert: list[str] = []
    for cell in cells:
        if cell is base or cell.applied_adapter_weights is None:
            continue
        deltas = [
            float(
                (cell.matrix(stimulus_set, pooling) - base.matrix(stimulus_set, pooling))
                .abs()
                .max()
            )
            for stimulus_set, pooling in sorted(cell.activations)
        ]
        if deltas and max(deltas) == 0.0:
            inert.append(cell.label)
    if inert:
        raise CellFormatError(
            f"these cells are bit-identical to the base capture despite reporting an applied "
            f"adapter: {inert}. The adapter loaded and then did not reach the forward pass, so the "
            f"cells hold base activations under a trained checkpoint's name."
        )


def load_ladder(
    root: Path,
    *,
    arms: Sequence[str] | None = None,
    steps: Sequence[int] | None = None,
    stimuli_sha256: str | None = None,
    require_capture_provenance: bool = False,
) -> Ladder:
    """Read every cell under `root` that the filters keep, and refuse an incomparable set.

    `arms` and `steps` are filters over what is present; an arm or step asked for and absent is
    fatal, because a ladder quietly short one checkpoint reads later as a flat stretch of trend.
    `stimuli_sha256` additionally pins the corpus the caller is about to analyse against.
    """
    cells = [read_cell(cell_dir) for cell_dir in find_cells(root)]
    if not cells:
        raise CellFormatError(f"{root} holds no finished cells.")
    if arms is not None:
        wanted_arms = set(arms) | {BASE_ARM}
        absent = sorted(set(arms) - {cell.arm for cell in cells})
        if absent:
            raise CellFormatError(
                f"{root} holds no cells for arms {absent}; it holds "
                f"{sorted({cell.arm for cell in cells})}."
            )
        cells = [cell for cell in cells if cell.arm in wanted_arms]
    if steps is not None:
        wanted_steps = set(steps) | {BASE_STEP}
        for arm in sorted({cell.arm for cell in cells} - {BASE_ARM}):
            present = {cell.step for cell in cells if cell.arm == arm}
            absent_steps = sorted(set(steps) - present)
            if absent_steps:
                raise CellFormatError(
                    f"arm {arm!r} under {root} has no cells for steps {absent_steps}; it has "
                    f"{sorted(present)}."
                )
        cells = [cell for cell in cells if cell.step in wanted_steps]
    ordered = tuple(sorted(cells, key=lambda cell: (cell.arm, cell.step)))
    assert_ladder_identity(ordered)
    if require_capture_provenance:
        assert_capture_provenance(ordered)
    assert_rows_align(ordered)
    if stimuli_sha256 is not None:
        assert_stimuli_match(ordered, stimuli_sha256)
    assert_adapter_moved_activations(ordered)
    logger.info(
        f"ladder loaded, root={root} cells={[cell.label for cell in ordered]} "
        f"sets={list(ordered[0].stimulus_sets)} poolings={list(ordered[0].poolings)}"
    )
    return Ladder(cells=ordered)


def pair_layout(cell: CapturedCell, stimulus_set: str, *, positive_side: str) -> PairLayout:
    """Locate the two sides of every matched pair in one set's stored rows.

    A pair with only one side present, or with two rows on the same side, is refused rather than
    dropped: the sets are authored as pairs, so a half pair means the capture or the corpus lost a
    row, and silently narrowing the denominator is how that becomes invisible.
    """
    if stimulus_set not in cell.rows:
        raise CellFormatError(
            f"{cell.label} holds no set {stimulus_set!r}; it holds {list(cell.stimulus_sets)}."
        )
    row_index = cell.rows[stimulus_set]
    order: list[str] = []
    by_pair: dict[str, dict[str, int]] = {}
    for row, (pair_id, side) in enumerate(zip(row_index.pair_ids, row_index.sides, strict=True)):
        if pair_id not in by_pair:
            by_pair[pair_id] = {}
            order.append(pair_id)
        if side in by_pair[pair_id]:
            raise CellFormatError(
                f"{cell.label} set {stimulus_set!r} pair {pair_id!r} has two rows on side {side!r}, "
                f"so its two sides cannot be told apart."
            )
        by_pair[pair_id][side] = row
    sides = sorted({side for members in by_pair.values() for side in members})
    if positive_side not in sides:
        raise CellFormatError(
            f"positive side {positive_side!r} does not appear in {cell.label} set {stimulus_set!r}, "
            f"whose sides are {sides}."
        )
    if len(sides) != 2:  # noqa: PLR2004 - a matched pair has exactly two sides by construction
        raise CellFormatError(
            f"{cell.label} set {stimulus_set!r} carries sides {sides}; a matched-pair read needs "
            f"exactly two."
        )
    negative_side = next(side for side in sides if side != positive_side)
    incomplete = sorted(pair for pair, members in by_pair.items() if len(members) != 2)  # noqa: PLR2004
    if incomplete:
        raise CellFormatError(
            f"{cell.label} set {stimulus_set!r} has {len(incomplete)} pairs missing a side "
            f"({incomplete[:ERROR_EXAMPLE_COUNT]}). Every read here is a difference between the two "
            f"sides of a pair, so a half pair is a lost row rather than a smaller sample."
        )
    return PairLayout(
        pair_ids=tuple(order),
        positive_rows=torch.tensor([by_pair[pair][positive_side] for pair in order]),
        negative_rows=torch.tensor([by_pair[pair][negative_side] for pair in order]),
        positive_side=positive_side,
        negative_side=negative_side,
    )


def concept_activations(  # noqa: PLR0913 - one read is a cell, a set, a pooling, a layer and a pair subset
    cell: CapturedCell,
    stimulus_set: str,
    pooling: str,
    layer: int,
    layout: PairLayout,
    *,
    pairs: torch.Tensor | None = None,
) -> ConceptActivations:
    """Pivot one cell's cached rows into the pair-aligned type the probe machinery consumes.

    `pairs` restricts the read to a subset of pair indices -- the held-out half of a projection read,
    or the fit half a direction is extracted from.
    """
    matrix = cell.layer(stimulus_set, pooling, layer)
    positive_rows = layout.positive_rows if pairs is None else layout.positive_rows[pairs]
    negative_rows = layout.negative_rows if pairs is None else layout.negative_rows[pairs]
    return ConceptActivations(matrix[positive_rows], matrix[negative_rows])


def write_ladder_manifest(root: Path, payload: dict[str, Any]) -> Path:
    """Write the run-level record beside a capture's cells, and return its path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / LADDER_MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def row_index_for(stimuli: Iterable[Stimulus], token_counts: Sequence[int]) -> RowIndex:
    """Build the stored row index for one set, in the order the activations were captured."""
    members = list(stimuli)
    return RowIndex(
        stimulus_ids=tuple(member.stimulus_id for member in members),
        sides=tuple(member.side for member in members),
        pair_ids=tuple(member.pair_id for member in members),
        token_counts=tuple(int(count) for count in token_counts),
    )
