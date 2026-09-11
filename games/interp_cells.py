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
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors.torch import load_file, save_file

from reward_hacking.interp.directions import POOLERS
from reward_hacking.interp.linear_probe import ConceptActivations

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

CELL_MANIFEST_FILENAME = "manifest.json"
ACTIVATIONS_FILENAME = "activations.safetensors"
LADDER_MANIFEST_FILENAME = "ladder-manifest.json"

REQUIRED_STIMULUS_FIELDS: frozenset[str] = frozenset({"id", "set", "side", "pair_id", "text"})
# A partial chain of thought to teacher-force after the template's opening `<think>`. Optional because
# the decision-theory sets carry one and a bare-stem corpus need not.
OPTIONAL_STIMULUS_FIELDS: frozenset[str] = frozenset({"assistant_prefix"})

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
        missing = sorted(set(cls.__dataclass_fields__) - set(payload))
        if missing:
            raise CellFormatError(
                f"{source} records no {missing} in its identity block, so it cannot be checked for "
                f"comparability against another cell. Re-capture it: a cell that cannot say what it "
                f"measured is not usable in a before/after read."
            )
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


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
    return digest_of_strings(
        field
        for stimulus in stimuli
        for field in (stimulus.stimulus_id, stimulus.text, stimulus.assistant_prefix or "")
    )


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
        stimuli.append(
            Stimulus(
                stimulus_id=stimulus_id,
                stimulus_set=str(row["set"]),
                side=str(row["side"]),
                pair_id=str(row["pair_id"]),
                text=str(row["text"]),
                assistant_prefix=None if prefix is None else str(prefix),
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
) -> Path:
    """Write one cell's tensors and manifest, and return the manifest path.

    The manifest is written last, so a cell interrupted mid-write is not mistaken for a finished one
    by a resumed run: the marker of "done" is the file that describes the work, not the work.
    """
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
    cell_dir.mkdir(parents=True, exist_ok=True)
    save_file(stored, str(cell_dir / ACTIVATIONS_FILENAME))
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
    }
    manifest_path = cell_dir / CELL_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def read_cell(cell_dir: Path) -> CapturedCell:
    """Read one cell back, checking the tensors on disk against what the manifest claims.

    Shapes are re-read rather than trusted because the manifest and the tensors are two files: a
    cell whose activations were written by one run and whose manifest was written by another would
    otherwise read as coherent, and every number downstream would be keyed to the wrong rows.
    """
    manifest_path = cell_dir / CELL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise CellFormatError(f"{manifest_path} not found, so {cell_dir} is not a finished cell.")
    manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text()))
    identity = CellIdentity.from_payload(
        cast("dict[str, Any]", manifest.get("identity", {})), source=manifest_path
    )
    rows = {
        stimulus_set: RowIndex(
            stimulus_ids=tuple(entry["stimulus_ids"]),
            sides=tuple(entry["sides"]),
            pair_ids=tuple(entry["pair_ids"]),
            token_counts=tuple(int(count) for count in entry["token_counts"]),
        )
        for stimulus_set, entry in cast("dict[str, Any]", manifest["sets"]).items()
    }
    raw = load_file(str(cell_dir / ACTIVATIONS_FILENAME))
    activations: dict[tuple[str, str], torch.Tensor] = {}
    for key, tensor in raw.items():
        parsed = parse_tensor_key(key)
        if parsed is None:
            raise CellFormatError(
                f"{cell_dir / ACTIVATIONS_FILENAME} holds key {key!r}, which names no known "
                f"pooling; expected '<set>/<pooling>' with pooling in {sorted(POOLERS)}."
            )
        activations[parsed] = tensor
    for (stimulus_set, pooling), tensor in sorted(activations.items()):
        if stimulus_set not in rows:
            raise CellFormatError(
                f"{cell_dir} holds activations for set {stimulus_set!r} that its manifest does not "
                f"describe, so their rows cannot be identified."
            )
        expected = (rows[stimulus_set].n_rows, identity.n_layers, identity.hidden_size)
        if tuple(tensor.shape) != expected:
            raise CellFormatError(
                f"{cell_dir} set {stimulus_set!r} pooling {pooling!r} is {tuple(tensor.shape)} on "
                f"disk but its manifest describes {expected}. The manifest and the tensors were "
                f"written by different runs."
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
        for field in CellIdentity.__dataclass_fields__:
            mine = getattr(cell.identity, field)
            theirs = getattr(reference.identity, field)
            if mine != theirs:
                differing.setdefault(field, {reference.label: theirs})[cell.label] = mine
    if differing:
        raise CellFormatError(
            f"these cells did not measure the same thing, so comparing them would read a change in "
            f"the apparatus as a change in the model: {json.dumps(differing, sort_keys=True)}. "
            f"Every field here moves the activations by at least as much as a trained delta does."
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
